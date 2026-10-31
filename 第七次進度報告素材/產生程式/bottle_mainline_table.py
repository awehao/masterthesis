#!/usr/bin/env python3
"""瓶子主線總表：只讀既有 run 紀錄（s1_report.json、s1_steps.npz、s1_cbf_replay.json、versions.sha256），不開 Isaac。
距離由逐步量測位姿獨立重算，不取判定程式的中間值；程式版本由 versions.sha256 回推 git 區間。
    python3 第七次進度報告素材/產生程式/bottle_mainline_table.py
輸出 第七次進度報告素材/資料/瓶子主線總表.json
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = os.path.join(ROOT, 'evaluation')
sys.path.insert(0, EV)
sys.path.insert(0, os.path.join(ROOT, 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

OUT = os.path.join(ROOT, '第七次進度報告素材', '資料', '瓶子主線總表.json')
RUNS = [('bottle_s1_5', 'G 凍結配置'), ('bottle_s1_g1', 'G 瓶 (+5,+5) mm'), ('bottle_s1_g2', 'G 瓶 (−5,−5) mm'),
        ('bottle_s1_g3', 'G 底盤 (+20,−20) mm、+1°'), ('bottle_s1_g4', 'G 底盤 (−20,+20) mm、−1°'),
        ('bottle_s1_p1', 'P 停車操作'), ('bottle_s1_m1', 'M MotM 式單步 QP'), ('bottle_s1_a1', 'G 拿掉 TCP 未來參考'),
        ('bottle_s2_1', 'S2 懸空方塊，6 點約束'), ('bottle_s2_2', 'S2 懸空方塊，包圍盒約束'), ('bottle_s3_1', 'S3 側牆＋懸空方塊'),
        ('bottle_n1_2', 'N1 真 ROS 掃描鏈'), ('bottle_n2_1', 'N2 真 ROS 掃描鏈＋Smac 路徑（只到抓取直線入口）')]
DEV = ['bottle_probe_1', 'bottle_smoke_1', 'bottle_s0a_1', 'bottle_s0a_2', 'bottle_s0b_1', 'bottle_s0b_2',
       'bottle_s1_1', 'bottle_s1_2', 'bottle_s1_3', 'bottle_s1_4', 'bottle_s1_6', 'bottle_s1_7', 'bottle_n1_1']
NEAR_STOP, T_DEC = 0.01, 0.10                           # 近停門檻 m/s；峰值減速度的差分窗 s


def box_of(s):
    v = [float(x) for x in s.split(',')]
    return np.array(v[:3]), np.array(v[3:])


def scene_dist(c, box):
    """量測底盤、關節與手指換算全部連桿點雲，量測瓶位姿換算瓶表面點，到方塊的逐步最小距離（真值評估）。"""
    BP, per, b = G.bottle_points(), {}, 9.9
    Q = np.column_stack([c[k] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
    qw, qx, qy, qz = c['bottle_qw'], c['bottle_qx'], c['bottle_qy'], c['bottle_qz']
    p = np.column_stack([c['bottle_base_x'], c['bottle_base_y'], c['bottle_base_z']])
    for i in range(len(Q)):
        for n, P in G.link_world(Q[i], float(c['finger1_q'][i])).items():
            per[n] = min(per.get(n, 9.9), float(G.box_dist(P, *box).min()))
        w, x, y, z = qw[i], qx[i], qy[i], qz[i]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        b = min(b, float(G.box_dist((R @ BP.T).T + p[i], *box).min()))
    k = min(per, key=per.get)
    return dict(link_mm=per[k] * 1e3, link=k, bottle_mm=b * 1e3)


def carry_motion(c, phases):
    """持瓶段讓路量：底盤偏航、底盤 y，與 TCP 相對底盤（底盤座標）對持瓶段起點的位移。"""
    k = [i for i, n in enumerate(phases) if str(n) == 'CARRY'][0]
    i = np.where(c['phase_index'].astype(int) == k)[0]
    Q = np.column_stack([c[f'q{j}'] for j in range(1, 7)])[i]
    P = np.array([G.C.K.fk(np.r_[0.0, 0.0, 0.0, q], 'link_tcp')[:3, 3] for q in Q[::5]])
    d = P - P[0]
    yaw = np.degrees(c['base_yaw'][i])
    return dict(yaw_peak_deg=float(yaw[np.argmax(np.abs(yaw))]), base_y_min=float(c['base_y'][i].min()), tcp_rel_y_mm=float(d[np.argmax(np.abs(d[:, 1])), 1]) * 1e3,
                tcp_rel_norm_mm=float(np.linalg.norm(d, axis=1).max()) * 1e3, dq_peak=[float(v) for v in np.abs(Q - Q[0]).max(axis=0)])


def disc_to_box_mm(x, y, box, radius=0.39):
    dx = np.maximum(np.abs(x - box[0][0]) - box[1][0] / 2, 0)
    dy = np.maximum(np.abs(y - box[0][1]) - box[1][1] / 2, 0)
    return float((np.hypot(dx, dy) - radius).min()) * 1e3


def metrics(run):
    D = os.path.join(EV, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t, x, y = c['t'], c['base_x'], c['base_y']
    sp = np.convolve(np.hypot(np.gradient(x, t), np.gradient(y, t)), np.ones(5) / 5, mode='same')
    t_done = next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None)
    t_end = t_done if t_done is not None else ((r['stop'] or {}).get('t', float(t[-1])))   # 受控停止後的減速不算進連續性
    tc, tl, tok = r['t_close'], r['t_lift_start'], r['t_lift_ok']
    mk = (t >= (tc if tc is not None else 1e9)) & (t <= (tl if tl is not None else -1.0))
    run_mk = t <= t_end
    w = int(round(T_DEC / 0.01))
    dec = -(sp[w:] - sp[:-w]) / T_DEC
    dec = dec[(t[:-w] >= 0.2) & (t[w:] <= t_end)]        # 前 0.2 s 平滑窗邊界不算
    ob = r['obstacle']
    face = float(np.hypot(np.maximum(np.abs(x - ob['c'][0]) - ob['sz'][0] / 2, 0), np.maximum(np.abs(y - ob['c'][1]) - ob['sz'][1] / 2, 0)).min())
    forb_k = [k for k in c if k.startswith('forb_')]
    fp = r['forbidden_pulses']
    ms = np.array(r['solve_ms_all'], float)
    out = dict(run=run, ok=bool(r.get('mission_done')) and bool(r['success']), success=bool(r['success']), mission_done=bool(r.get('mission_done')),
               stop=(r['stop'] or {}).get('why'), stop_detail=r['stop'], t_exit=r.get('t_exit_in'), t_done=t_done, slip_mm=r['slip_max_mm_deg'][0] if r['t_grasp'] is not None else None, slip_deg=r['slip_max_mm_deg'][1] if r['t_grasp'] is not None else None,
               t_grip=(tok - tc) if tok is not None and tc is not None else None,
               forb_peak_n=max(v['peak'] for v in fp.values()), forb_cols_peak_n=max(float(np.nanmax(c[k])) for k in forb_k), forb_nan_steps=int(sum(np.isnan(c[k]).sum() for k in forb_k)),   # 逐步欄沒有手指↔瓶，判定以 forbidden_pulses 為準
               forb_nodata=int(sum(v['n_nodata'] for v in fp.values())), forb_pulse={k: v for k, v in fp.items() if v['peak'] > 0},
               clear_bottle_mm=r['clear_bottle']['meas_min'] * 1e3 if r['clear_bottle']['meas_min'] is not None else None,
               clear_pred_mm=r['clear_bottle']['pred_min'] * 1e3 if r['clear_bottle']['pred_min'] is not None else None,
               clear_next_mm=r['clear_bottle']['next_min'] * 1e3 if r['clear_bottle']['next_min'] is not None else None,
               base_face_m=face, v_grip=(float(sp[mk].min()), float(sp[mk].max())) if mk.any() else None, near_grip_s=float((sp[mk] < NEAR_STOP).sum()) * 0.01,
               near_run_s=float((sp[run_mk] < NEAR_STOP).sum()) * 0.01, dec_peak=float(dec.max()), v_peak=float(sp[run_mk].max()),
               dec_t=float(t[:-w][(t[:-w] >= 0.2) & (t[w:] <= t_end)][int(np.argmax(dec))]),
               ms=dict(p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), max=float(ms.max()), n=int(ms.size), over50=int((ms > 50).sum())),
               world_ms=r['timing']['world_step'], scan_ms=r['timing'].get('scan'), shield_ms=r['timing'].get('shield'),
               qp=r['qp_counts'], fallback=r.get('n_fallback'), gi_fail=r.get('gi_fail'),
               ros=r['nav_chain'].get('ros'), plan=r.get('plan') and {k: v for k, v in r['plan'].items() if k not in ('raw', 'joined', 'log')},
               shield=r['nav_chain']['shield'], n_dyn=r['nav_chain']['n_dyn_max'], body_bind=r.get('body_bind'), body_rows=r.get('body_rows_by'))
    rp = os.path.join(D, 's1_cbf_replay.json')
    if os.path.exists(rp):
        p = json.load(open(rp))
        out['cbf'] = dict(changed=p['n_bound'], n=p['n'], phases=p['phase_bound'], du_max=p['du_max'], reproduce_max=p['reproduce_max'])
    rb = r['readback']
    if rb.get('s2_box'):
        out['box'] = scene_dist(c, box_of(rb['s2_box']))
        out['carry'] = carry_motion(c, Z['phases'])
    if rb.get('wall_box'):
        wb = box_of(rb['wall_box'])
        out['wall'] = dict(bottle_mm=scene_dist(c, wb)['bottle_mm'], base_disc_mm=disc_to_box_mm(x, y, wb))
    return out, Z['rows']


def provenance(run, log, memo):
    D = os.path.join(EV, 'runs', run)
    if not os.path.exists(os.path.join(D, 'versions.sha256')):
        return dict(run=run, git=None, note='無 versions.sha256')
    files = [ln.split() for ln in open(os.path.join(D, 'versions.sha256')) if ln.strip()]

    def blob(c, f):
        if (c, f) not in memo:
            p = subprocess.run(['git', '-C', ROOT, 'show', f'{c}:{f}'], capture_output=True)
            memo[c, f] = hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None
        return memo[c, f]
    hit = [c for c in log if all(blob(c, f) == h for h, f in files)]
    sha = {}
    for f in ('s1_report.json', 's1_steps.npz'):
        p = os.path.join(D, f)
        if os.path.exists(p):
            sha[f] = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    log_txt = open(os.path.join(D, 'run.log'), errors='replace').readline() if os.path.exists(os.path.join(D, 'run.log')) else ''
    args = log_txt.split('擾動參數［')[1].split('］')[0] if '擾動參數［' in log_txt else ''
    rep = os.path.join(D, 's1_report.json')
    gm = [h[:16] for h, f in files if f.endswith('ammr_wholebody_mpc/gmpc.py')]
    pick = {k: next((h[:8] for h, f in files if f.endswith(k)), None) for k in ('gmpc_wb.py', 'bottle_s1_ctrl.py', 'isaac_bottle_s1.py')}
    return dict(run=run, first=hit[-1][:9] if hit else None, last=hit[0][:9] if hit else None, n_files=len(files), n_hit=len(hit), sha=sha, args=args, gmpc=gm[0] if gm else None, pick=pick,
                time=time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(rep if os.path.exists(rep) else os.path.join(D, 'run.log')))))


def main():
    log = subprocess.run(['git', '-C', ROOT, 'log', '--format=%H', '-n', '200'], capture_output=True, text=True).stdout.split()
    memo, rows, raw = {}, [], {}
    for run, label in RUNS:
        m, R = metrics(run)
        m['label'] = label
        rows.append(m)
        raw[run] = R
    same = {f'bottle_s1_5=={o}': bool(np.array_equal(raw['bottle_s1_5'], np.load(os.path.join(EV, 'runs', o, 's1_steps.npz'))['rows'], equal_nan=True)) for o in ('bottle_s1_6', 'bottle_s1_7')}
    prov = [provenance(r, log, memo) for r, _ in RUNS] + [provenance(r, log, memo) for r in DEV]
    json.dump(dict(rows=rows, provenance=prov, identical=same, near_stop_mps=NEAR_STOP, dec_window_s=T_DEC), open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for e in rows:
        print(e['run'], '任務', e['ok'], '阻點', e['stop'], '出口', e['t_exit'], '完成', e['t_done'], f"滑移 {e['slip_mm']}/{e['slip_deg']} 抓取段 {e['t_grip']} 禁止峰 {e['forb_peak_n']:.3f} NaN {e['forb_nan_steps']} 無資料 {e['forb_nodata']} 瓶淨空 {e['clear_bottle_mm']:.2f} 底盤面 {e['base_face_m']:.3f}")
        print('   ', '夾持速', e['v_grip'], f"近停 抓取 {e['near_grip_s']:.2f} 全程 {e['near_run_s']:.2f} 峰減速 {e['dec_peak']:.2f}@{e['dec_t']:.2f} 峰速 {e['v_peak']:.3f}", {k: round(v, 1) for k, v in e['ms'].items()}, e['qp'], 'fb', e['fallback'], 'Sh', e['shield']['active'], 'CBF', e.get('cbf', {}).get('changed'), e.get('cbf', {}).get('n'))
        if 'box' in e or 'wall' in e:
            print('   ', e.get('box'), e.get('wall'), e['body_bind'], e['body_rows'], e.get('carry'))
    print(same)
    for p in prov:
        print(p['run'], p.get('first'), '..', p.get('last'), p.get('n_hit'), p.get('time'), p.get('pick'), p.get('sha', {}).get('s1_steps.npz'))


if __name__ == '__main__':
    sys.exit(main())
