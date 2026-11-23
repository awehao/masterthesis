#!/usr/bin/env python3
"""保持期間整機振動診斷。

只讀既有趟次，不改任何參數、不重跑。輸入：
  wg2_out.json      節點每週期（request_body_presolve 整形前 / request_body 整形後 /
                    publish_sim_t 實際發布時刻 / err_p, err_r / shape）
  cmd_env.jsonl      四段命令封裝（solver / safety / adapter / endpoint），沿 source_seq 追蹤
  sim/wb_run.json    每物理步平台狀態（joint*_sp 設定點、joint*_act 實測角、底盤、TCP）

輸出：
  表1 鏈路各段 9 維命令的振幅、主頻、週期  -> 振盪最先出現在哪一段
  表2 平台側設定點/實測角/底盤/TCP 的振幅與主頻
  表3 設定點 -> 實測關節角在振盪頻率的增益與相位，對照已辨識一階模型
  表4 底盤實測相對命令未被解釋的部分
  一張同步時間圖

已知缺項（見 GAPS）。
"""
import argparse, json, collections, itertools
import numpy as np

CH = ['v_x^B', 'v_y^B', 'omega', 'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6']
STAGES = ['presolve', 'shaped', 'published', 'safety', 'endpoint']
TAU_SP = 0.100          # free4 辨識出的手臂設定點一階時間常數
GAPS = [
    'adapter 段的 stamp_sim_t 是 wall clock（該節點未設 use_sim_time），'
    '不能與其他三段相減；本分析改用錄製端的 recorder_sim_t 把它放上模擬時間軸，'
    '那是「錄製端收到」而非「adapter 發出」的時刻。',
    'wb_run.json 沒有底盤或關節的力/力矩欄位，因此「手臂反作用力帶動底盤」'
    '無法與輪端摩擦、接觸模型分開量化，只能給底盤運動的上限。',
    'publish_sim_t 與 sim_t 都讀自 100 Hz 模擬時鐘，時間解析度 10 ms；'
    '小於 10 ms 的發布時刻差異無法分辨。',
    'joint*_sp 以 1e-6 rad 紀錄，差分還原的套用速率量化階距為 1e-4 rad/s。',
]


# ---------------------------------------------------------------- 基本工具
def hold_window(pub, tol_p=0.005, tol_r=0.02):
    """最長一段同時滿足位置與姿態容差的連續週期，回傳其模擬時間起訖。"""
    ep = np.array([x['err_p'] for x in pub]); er = np.array([x['err_r'] for x in pub])
    t = np.array([x['sim_t'] for x in pub])
    I = np.where((ep <= tol_p) & (er <= tol_r))[0]
    if not len(I):
        raise SystemExit('此趟沒有同時進入位置與姿態容差的週期')
    seg = max(np.split(I, np.where(np.diff(I) > 1)[0] + 1), key=len)
    return float(t[seg[0]]), float(t[seg[-1]])


def spec(x, fs):
    """(RMS, 峰峰值, 主頻 Hz, 主頻 bin 佔變異比, lag-1 自相關)。"""
    x = np.asarray(x, float); n = len(x)
    if n < 4:
        return (np.nan,) * 5
    y = x - x.mean()
    P = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs); P[0] = 0.0
    tot = P.sum(); k = int(np.argmax(P))
    ss = (y ** 2).sum()
    return (float(np.sqrt((y ** 2).mean())), float(x.max() - x.min()), float(f[k]),
            float(P[k] / tot) if tot > 0 else np.nan,
            float(y[:-1] @ y[1:] / ss) if ss > 0 else np.nan)


def band(x, lo, hi, fs):
    """(頻帶佔變異比, 頻帶成分 RMS)。頻帶 RMS = 總 RMS * sqrt(佔比)。"""
    y = np.asarray(x, float); y = y - y.mean(); n = len(y)
    P = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs); P[0] = 0.0
    m = (f >= lo) & (f <= hi)
    fr = float(P[m].sum() / P.sum()) if P.sum() > 0 else np.nan
    return fr, float(np.std(y) * np.sqrt(fr))


def gain_phase(u, y, lo, hi, fs):
    """在 [lo,hi] 內取輸入能量最大的 bin，回傳該 bin 的 (增益, 相位 deg, 頻率)。"""
    n = len(u); w = np.hanning(n)
    Fu = np.fft.rfft((np.asarray(u, float) - np.mean(u)) * w)
    Fy = np.fft.rfft((np.asarray(y, float) - np.mean(y)) * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    m = np.where((f >= lo) & (f <= hi))[0]
    k = m[int(np.argmax(np.abs(Fu[m])))]
    r = Fy[k] / Fu[k]
    return float(abs(r)), float(np.degrees(np.angle(r))), float(f[k])


def acf(x, K=10):
    y = np.asarray(x, float); y = y - y.mean(); d = (y ** 2).sum()
    if d <= 0:
        return np.full(K + 1, np.nan)
    return np.array([float(y[:len(y) - k] @ y[k:] / d) for k in range(K + 1)])


def cycle_period(x, kmax=8):
    """以自相關在 lag>=2 的最大值定「幾個控制週期一輪」，另回傳該峰值強度。"""
    a = acf(x, kmax)
    k = int(np.argmax(a[2:kmax + 1])) + 2
    return k, float(a[k])


def load(run):
    w = json.load(open(run + '/wg2_out.json'))
    s = json.load(open(run + '/sim/wb_run.json'))
    rows = [json.loads(l) for l in open(run + '/cmd_env.jsonl')]
    return w, s, [r for r in rows if r.get('type') == 'env']


# ---------------------------------------------------------------- 主流程
def analyse(run, dt_c=0.05):
    w, s, env = load(run)
    pub = [x for x in w['log'] if x.get('published')]
    t0, t1 = hold_window(pub)
    cyc = [x for x in pub if t0 <= x['sim_t'] <= t1]
    fs_c = 1.0 / dt_c

    # --- 鏈路：節點 log 無 source_seq；solver 封裝的 stamp_sim_t 就是該週期的
    #     publish_sim_t，用 1 us 容差配對取回序號，再以序號接後續各段。
    by = collections.defaultdict(dict)
    for r in env:
        if r['derived']:
            by[r['stage_name']][r['source_seq']] = r
    sol = [r for r in env if r['stage_name'] == 'solver' and r['derived']]
    sst = np.array([r['stamp_sim_t'] for r in sol])
    chain, n_nosol = [], 0
    for x in cyc:
        k = int(np.argmin(np.abs(sst - x['publish_sim_t'])))
        if abs(sst[k] - x['publish_sim_t']) > 1e-6:
            n_nosol += 1; continue
        q = sol[k]['source_seq']
        chain.append(dict(
            sim_t=x['sim_t'], publish_sim_t=x['publish_sim_t'], seq=q,
            scale=x['shape']['scale'], reason=x['shape']['reason'],
            presolve=np.array(x['request_body_presolve'], float),
            shaped=np.array(x['request_body'], float),
            published=np.array(sol[k]['u'], float),
            safety=np.array(by['safety'][q]['u'], float) if q in by['safety'] else None,
            endpoint=np.array(by['endpoint'][q]['u'], float) if q in by['endpoint'] else None))
    miss = {k: sum(1 for c in chain if c.get(k) is None) for k in STAGES}
    miss['cycle_without_solver_env'] = n_nosol

    # --- 平台側
    ci = {c: i for i, c in enumerate(s['log_cols'])}
    L = np.array(s['log'], float); ts = L[:, ci['t']]
    dt_p = float(np.median(np.diff(ts))); fs_p = 1.0 / dt_p
    hw = (ts >= t0) & (ts <= t1); H = L[hw]; th = ts[hw]
    nonfinite = [c for c in s['log_cols'] if not np.all(np.isfinite(H[:, ci[c]]))]
    sp = np.stack([H[:, ci['joint%d_sp' % (i + 1)]] for i in range(6)], 1)
    act = np.stack([H[:, ci['joint%d_act' % (i + 1)]] for i in range(6)], 1)
    rate = np.stack([H[:, ci['joint%d_rate_meas' % (i + 1)]] for i in range(6)], 1)
    # 執行端每物理步 sp += qd*dt_p ⇒ 差分還原 100 Hz 實際套用的手臂速率
    spf = np.stack([L[:, ci['joint%d_sp' % (i + 1)]] for i in range(6)], 1)
    qd_app = (np.diff(spf, axis=0) / dt_p)[hw[1:]]
    th2 = ts[1:][hw[1:]]
    bcmd = H[:, [ci['vx_cmd'], ci['vy_cmd'], ci['wz_cmd']]]
    bmeas = np.stack([H[:, ci['phys_vx']], H[:, ci['phys_vy']], H[:, ci['base_ang_meas']]], 1)
    pose = np.stack([H[:, ci['base_x']], H[:, ci['base_y']], H[:, ci['base_yaw']]], 1)
    tcp = H[:, [ci['tcp_x'], ci['tcp_y'], ci['tcp_z']]]

    # --- 振盪週期（取振幅最大的手臂通道，用整形前序列）
    pre = np.stack([c['presolve'] for c in chain])
    jmax = 3 + int(np.argmax([np.std(pre[:, 3 + i]) for i in range(6)]))
    P, Pstr = cycle_period(pre[:, jmax])
    f_osc = fs_c / P
    lo, hi = max(f_osc - 1.0, 0.25), f_osc + 1.0

    # --- 發布時刻 vs 時槽
    plag = np.array([c['publish_sim_t'] - c['sim_t'] for c in chain])
    dwell = [len(list(g)) for _, g in itertools.groupby(H[:, ci['recv_seq']])]

    return dict(run=run, w=w, s=s, ci=ci, L=L, t0=t0, t1=t1, cyc=cyc, chain=chain,
                miss=miss, nonfinite=nonfinite, dt_p=dt_p, fs_p=fs_p, fs_c=fs_c,
                H=H, th=th, th2=th2, sp=sp, act=act, rate=rate, qd_app=qd_app,
                bcmd=bcmd, bmeas=bmeas, pose=pose, tcp=tcp,
                jmax=jmax, P=P, Pstr=Pstr, f_osc=f_osc, lo=lo, hi=hi,
                plag=plag, dwell=dwell)


def report(A):
    run = A['run'].rstrip('/').split('/')[-1]
    w, ci, H = A['w'], A['ci'], A['H']
    chain, lo, hi, fs_c, fs_p = A['chain'], A['lo'], A['hi'], A['fs_c'], A['fs_p']
    print(f"趟次 {run}   γ={w['stats'].get('near_target_gamma')}")
    print(f"保持窗 sim {A['t0']:.3f}–{A['t1']:.3f} s = {A['t1']-A['t0']:.3f} s；"
          f"{len(chain)} 控制週期 ({fs_c:.0f} Hz) / {len(H)} 物理步 ({fs_p:.0f} Hz)")
    print(f"窗內 lam 中位數 {np.median(H[:, ci['lam']]):.4f}，"
          f"E2 modified 比例 {H[:, ci['modified']].mean():.3f}，"
          f"整形 scale 中位數 {np.median([c['scale'] for c in chain]):.4f}")
    print(f"鏈路缺漏 {A['miss']}；保持窗內非有限值欄位 {A['nonfinite'] or '無'}")
    print(f"振盪週期（整形前 {CH[A['jmax']]}）= {A['P']} 個控制週期 "
          f"= {A['f_osc']:.2f} Hz，自相關峰 {A['Pstr']:+.2f}；分析頻帶 {lo:.2f}–{hi:.2f} Hz")

    print('\n=== 表1 鏈路各段 9 維命令（20 Hz，頻帶成分 RMS 在上述頻帶內）===')
    print(f"{'通道':>7} {'段':>10} {'RMS':>10} {'峰峰值':>10} {'頻帶RMS':>10} {'佔比':>6} "
          f"{'週期':>5} {'峰':>6}")
    for j, nm in enumerate(CH):
        for k in STAGES:
            v = [c[k][j] for c in chain if c.get(k) is not None]
            if len(v) < 4:
                continue
            m = spec(v, fs_c); fr, br = band(v, lo, hi, fs_c); p, ps = cycle_period(v)
            print(f'{nm:>7} {k:>10} {m[0]:10.5f} {m[1]:10.5f} {br:10.5f} {fr:6.2f} '
                  f'{p:5d} {ps:+6.2f}')
        print()

    print('=== 表2 平台側（100 Hz）===')
    print(f"{'量':>20} {'單位':>6} {'RMS':>10} {'峰峰值':>10} {'頻帶RMS':>10} {'佔比':>6} {'主頻Hz':>7}")

    def row(lbl, unit, x, fs=fs_p):
        m = spec(x, fs); fr, br = band(x, lo, hi, fs)
        print(f'{lbl:>20} {unit:>6} {m[0]:10.5f} {m[1]:10.5f} {br:10.5f} {fr:6.2f} {m[2]:7.2f}')
    for i in range(6):
        row(f'j{i+1} 設定點', 'rad', A['sp'][:, i])
        row(f'j{i+1} 實測角', 'rad', A['act'][:, i])
        row(f'j{i+1} 套用速率', 'rad/s', A['qd_app'][:, i])
        row(f'j{i+1} 實測速率', 'rad/s', A['rate'][:, i])
        print()
    for i, (lbl, u) in enumerate([('v_x', 'm/s'), ('v_y', 'm/s'), ('omega', 'rad/s')]):
        row(f'底盤 {lbl} 命令', u, A['bcmd'][:, i]); row(f'底盤 {lbl} 實測', u, A['bmeas'][:, i])
    print()
    for i, lbl in enumerate(['base_x', 'base_y', 'base_yaw']):
        row(f'底盤 {lbl}', 'm' if i < 2 else 'rad', A['pose'][:, i])
    print()
    for i, lbl in enumerate(['tcp_x', 'tcp_y', 'tcp_z']):
        row(f'TCP {lbl}', 'm', A['tcp'][:, i])
    row('TCP 位置誤差', 'm', [x['err_p'] for x in A['cyc']], fs_c)
    row('TCP 姿態誤差', 'rad', [x['err_r'] for x in A['cyc']], fs_c)

    print(f'\n=== 表3 設定點 -> 實測關節角 @ {A["f_osc"]:.2f} Hz，對照已辨識一階 '
          f'tau={TAU_SP:.3f} s ===')
    g0r = lambda f: 1.0 / np.sqrt(1 + (2 * np.pi * f * TAU_SP) ** 2)
    p0r = lambda f: -np.degrees(np.arctan(2 * np.pi * f * TAU_SP))
    print(f"{'關節':>5} {'f Hz':>6} {'設定點頻帶RMS':>12} {'實測增益':>8} {'一階預測':>8} "
          f"{'增益比':>7} {'實測相位':>8} {'一階預測':>8}")
    for i in range(6):
        g, ph, f0 = gain_phase(A['sp'][:, i], A['act'][:, i], lo, hi, fs_p)
        _, br = band(A['sp'][:, i], lo, hi, fs_p)
        note = '  <- 振幅接近紀錄捨入，不解讀' if br < 5e-5 else ''
        print(f'j{i+1:<4} {f0:6.2f} {br:12.6f} {g:8.3f} {g0r(f0):8.3f} {g/g0r(f0):7.2f} '
              f'{ph:8.1f} {p0r(f0):8.1f}{note}')

    print('\n=== 表4 底盤：實測相對命令未被解釋的部分 ===')
    print(f"{'':>8} {'命令p2p':>10} {'實測p2p':>10} {'差值p2p':>10} {'差RMS/實測RMS':>13} "
          f"{'增益@f':>7} {'相位@f':>8}")
    for i, (lbl, a_, b_) in enumerate([('v_x', 'vx_cmd', 'phys_vx'),
                                       ('v_y', 'vy_cmd', 'phys_vy'),
                                       ('omega', 'wz_cmd', 'base_ang_meas')]):
        c = H[:, ci[a_]]; y = H[:, ci[b_]]; e = y - c
        g, ph, _ = gain_phase(c, y, lo, hi, fs_p)
        print(f'{lbl:>8} {c.ptp():10.5f} {y.ptp():10.5f} {e.ptp():10.5f} '
              f'{np.std(e)/max(np.std(y),1e-12):13.2f} {g:7.3f} {ph:+8.1f}')
    pose, tcp = A['pose'], A['tcp']
    bp = 1e3 * np.sum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))
    tp = 1e3 * np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1))
    print(f'  底盤位姿 p2p: x {1e3*pose[:,0].ptp():.2f} mm, y {1e3*pose[:,1].ptp():.2f} mm, '
          f'yaw {1e3*pose[:,2].ptp():.2f} mrad；路徑 {bp:.2f} mm')
    print(f'  TCP p2p: x {1e3*tcp[:,0].ptp():.2f} y {1e3*tcp[:,1].ptp():.2f} '
          f'z {1e3*tcp[:,2].ptp():.2f} mm；路徑 {tp:.2f} mm（底盤的 {tp/max(bp,1e-9):.0f} 倍）')

    print('\n=== 表5 發布時刻 vs 時槽 ===')
    pl = 1e3 * A['plag']; dv = np.diff([c['publish_sim_t'] for c in chain])
    print(f'  時槽間隔: 全部 {1e3*np.median(np.diff([c["sim_t"] for c in chain])):.1f} ms')
    print(f'  發布落後時槽: p50 {np.median(pl):.1f} ms, min {pl.min():.1f}, max {pl.max():.1f}')
    print(f'  發布間隔分佈 (ms): {dict(sorted(collections.Counter(np.round(1e3*dv).astype(int)).items()))}')
    a = acf(pl, 4)
    print(f'  落後量自相關 lag1..4: ' + ' '.join(f'{v:+.2f}' for v in a[1:]))
    q = np.stack([c["presolve"] for c in chain])[:, A['jmax']]
    print(f'  落後量 vs 整形前 {CH[A["jmax"]]} 相關係數: {np.corrcoef(pl, q)[0,1]:+.3f}')
    print(f'  命令在模擬器停留步數分佈: '
          f'{dict(sorted(collections.Counter(A["dwell"]).items()))}（名目應為 5）')

    print('\n=== 已知缺項 ===')
    for g in GAPS:
        print('  - ' + g)


def make_plot(A, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    run = A['run'].rstrip('/').split('/')[-1]
    chain, j, th, th2 = A['chain'], A['jmax'], A['th'], A['th2']
    tc = np.array([c['sim_t'] for c in chain]); jj = j - 3
    fig, ax = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    a = ax[0]
    a.plot(tc, 1e3 * np.array([x['err_p'] for x in A['cyc']]), '.-', ms=4,
           color='#1b5e8a', label='位置誤差 mm')
    a.axhline(5.0, ls=':', lw=1, c='#1b5e8a')
    b = a.twinx()
    b.plot(tc, 1e3 * np.array([x['err_r'] for x in A['cyc']]), '.-', ms=4,
           color='#b5651d', label='姿態誤差 mrad')
    b.axhline(20.0, ls=':', lw=1, c='#b5651d')
    a.set_ylabel('位置誤差 mm'); b.set_ylabel('姿態誤差 mrad')
    a.set_title(f'{run}  保持段同步時間圖（γ={A["w"]["stats"].get("near_target_gamma")}，'
                f'sim {A["t0"]:.2f}–{A["t1"]:.2f} s，振盪週期 {A["P"]} 個控制週期 '
                f'= {A["f_osc"]:.2f} Hz）')
    a.legend(loc='upper left', fontsize=8); b.legend(loc='upper right', fontsize=8)

    a = ax[1]
    for k, st, c, lw in [('presolve', '-', '#9a9a9a', 1.6), ('shaped', '-', '#1b5e8a', 1.6),
                         ('published', '--', '#b5651d', 1.3), ('safety', '-.', '#7b4397', 1.1),
                         ('endpoint', ':', '#2e7d32', 1.6)]:
        v = [cc[k][j] for cc in chain if cc.get(k) is not None]
        tt = [cc['sim_t'] for cc in chain if cc.get(k) is not None]
        a.plot(tt, v, st, color=c, lw=lw, label=k)
    a.step(th2, A['qd_app'][:, jj], where='post', color='#c0392b', lw=0.8, alpha=0.8,
           label='100 Hz 實際套用')
    a.axhline(0, lw=0.6, c='#555')
    a.set_ylabel(f'j{jj+1} 速率 rad/s'); a.legend(fontsize=8, ncol=3)
    a.set_title(f'關節 {jj+1} 速率命令沿鏈路：求解器請求(presolve) → 整形後 → 實際發布 '
                f'→ 安全層 → E2 套用', fontsize=10)

    a = ax[2]
    a.plot(th, A['sp'][:, jj], color='#1b5e8a', lw=1.4, label=f'j{jj+1} 設定點')
    a.plot(th, A['act'][:, jj], color='#b5651d', lw=1.4, label=f'j{jj+1} 實測角')
    a.set_ylabel('rad'); a.legend(fontsize=8, loc='upper left')
    g = a.twinx()
    g.plot(th, 1e3 * (A['sp'][:, jj] - A['act'][:, jj]), color='#555', lw=0.9,
           label='sp − act mrad')
    g.set_ylabel('sp − act  mrad'); g.legend(fontsize=8, loc='lower right')
    a.set_title('手臂設定點與實測關節角（絕對值；灰線為兩者之差）', fontsize=10)

    a = ax[3]
    for i, (lbl, c) in enumerate([('v_x', '#1b5e8a'), ('v_y', '#b5651d'), ('ω', '#2e7d32')]):
        sc = 1e3
        a.plot(th, sc * A['bcmd'][:, i], color=c, lw=1.4, label=f'{lbl} 命令')
        a.plot(th, sc * A['bmeas'][:, i], color=c, lw=0.9, ls='--', alpha=0.75,
               label=f'{lbl} 實測')
    a.axhline(0, lw=0.6, c='#555')
    # 起步前兩個控制週期的暫態會把保持段壓扁，取穩定後的範圍定 y 軸
    k0 = int(2 * (fs := 1.0 / A['dt_p']) * 0.05)
    lim = 1.25 * max(np.abs(A['bcmd'][k0:]).max(), np.abs(A['bmeas'][k0:]).max()) * 1e3
    a.set_ylim(-lim, lim)
    a.set_ylabel('mm/s, mrad/s'); a.legend(fontsize=7, ncol=3)
    a.set_title(f'底盤速度命令與實測（y 軸依穩定後範圍，起步暫態超出框外）', fontsize=10)

    a = ax[4]
    a.plot(th, 1e3 * (A['pose'][:, 0] - A['pose'][0, 0]), color='#1b5e8a', lw=1.4, label='Δbase_x mm')
    a.plot(th, 1e3 * (A['pose'][:, 1] - A['pose'][0, 1]), color='#b5651d', lw=1.4, label='Δbase_y mm')
    a.plot(th, 1e3 * (A['pose'][:, 2] - A['pose'][0, 2]), color='#2e7d32', lw=1.1, label='Δbase_yaw mrad')
    a.set_ylabel('mm, mrad'); a.legend(fontsize=8, ncol=3)
    a.set_title('底盤位姿（相對保持起點）—— 與下方 TCP 同尺度比較', fontsize=10)

    a = ax[5]
    for i, (lbl, c) in enumerate([('x', '#1b5e8a'), ('y', '#b5651d'), ('z', '#2e7d32')]):
        a.plot(th, 1e3 * (A['tcp'][:, i] - A['tcp'][0, i]), color=c, lw=1.4, label=f'Δtcp_{lbl} mm')
    a.set_ylabel('mm'); a.set_xlabel('模擬時間 s'); a.legend(fontsize=8, ncol=3)
    a.set_title('TCP 位置（相對保持起點）', fontsize=10)

    yl = max(abs(np.array(ax[4].get_ylim())).max(), abs(np.array(ax[5].get_ylim())).max())
    ax[4].set_ylim(-yl, yl); ax[5].set_ylim(-yl, yl)
    for x in ax:
        x.grid(alpha=0.25)
        for cc in chain:
            x.axvline(cc['sim_t'], lw=0.3, c='#cccccc', zorder=0)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    A = analyse(a.run)
    report(A)
    if a.plot:
        make_plot(A, a.plot)
        print(f'\n同步時間圖 -> {a.plot}')


if __name__ == '__main__':
    main()
