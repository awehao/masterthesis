#!/usr/bin/env python3
"""MotM 第一包（5）時間帳：整趟控制拍統計＋同時負載下的控制拍耗時（本機桌機；每個程序限 1 核 CPUQuota=100%）。
控制器（各自整趟、假世界）：
  open   開門執行層 r8（closure；g2＝gravity＋ik_fc＋sweep_cache，與 r7 命令差 ≤8e-10，door_g2_compare_r1）名目計畫整趟到完成
  grasp  移動中抓持 MovingGraspCtrl（0.05 m/s 名目，test_door_motm_grasp_fake 同入口）到 READY 後 0.5 s
  exit   退出控制器（full_2 開門終態承接，scope_end＝X2）到 X2 完成或 15 s
背景負載（同時跑，各 1 核）：
  planner  開門規劃 door_wbopt r6 grad='analytic'，MotM 名目 READY 起點（含 v0）冷啟，重複求解
  percep×2 把手偵測 d1_handle_detect_v2.detect 在 Isaac 腕部相機實存深度影格（runs/d1_dev_f02P/wrist_v0）上，每路上限 30 Hz（雙相機）
量：每個控制拍耗時（首拍＝冷、p50／p95／max、>50 ms 拍數）單跑 vs 負載；規劃每次 wall 單跑 vs 負載；偵測每格 ms 與實際達成頻率。
這是計算時間量測，不是實時系統：控制器在假世界照模擬時間步進，不等牆鐘；>50 ms 的拍在實機／Isaac 實時下就是超時。
    python3 evaluation/door_motm_concurrency.py drive OUT.json"""
import json, math, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SP = os.environ.get('SP', '/tmp/claude-1000/-home-howardchen-masterthesis/2a47f71f-1a91-4ff0-9dbf-1892d8540042/scratchpad')
PY = ['env', '-u', 'VIRTUAL_ENV', 'PATH=/usr/local/bin:/usr/bin:/bin', f'SP={SP}', '/usr/bin/python3']
SCOPE = ['systemd-run', '--user', '--scope', '-q', '-p', 'CPUQuota=100%', '-p', 'MemoryMax=3G']
POOL = int(os.environ.get('POOL', '0'))                              # G3：控制器平行掃掠工作程序數；>0 時控制器範圍 CPU 額度＝(1＋POOL) 核
SCOPE_CTL = ['systemd-run', '--user', '--scope', '-q', '-p', f'CPUQuota={100 * (1 + POOL)}%', '-p', 'MemoryMax=6G']
PREP_FC = os.environ.get('PREP_FC', '0') == '1'
DEADLINE = None if os.environ.get('DEADLINE_MS') in (None, '', 'none', 'None') else float(os.environ['DEADLINE_MS'])
PY_CTL = ['env', '-u', 'VIRTUAL_ENV', 'PATH=/usr/local/bin:/usr/bin:/bin', f'SP={SP}', f'POOL={POOL}', f'PREP_FC={int(PREP_FC)}', f'DEADLINE_MS={DEADLINE}', '/usr/bin/python3']


def stats(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), first_cold=round(float(a[0]), 1), p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1), n_over_50=int((a > 50).sum()))


def fast_clear():
    import door_wbopt as WB
    import door_clear_door_fast as _F
    WB.clear_door = _F.clear_door_fast
    return WB


def ctrl_open(out):
    fast_clear()
    import test_door_open_exec as T
    X = T.X
    P = np.load(os.path.join(SP, 'ex4', 'plan_nominal.npz'))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
    c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q,
                       arm_mode='closure', g2=(dict(gravity2=True, ik_fc=True, pool=POOL, prep_fc=PREP_FC, deadline_ms=DEADLINE) if POOL else dict(gravity=True, ik_fc=True, sweep_cache=True)))
    wu = c.warmup(q0, T.Fake(c, q0).door()[0]) if POOL else None   # r10 交接前預熱（部署組）
    T.Fake(c, q0).run(t_end=40.0)
    c.close_pool()
    json.dump(dict(tick_ms=list(map(float, c.tick_ms)), done=c.done, stop=c.stop, n_deadline_miss=c.n_deadline_miss, warmup_ms=wu), open(out, 'w'), default=float)


def timed_class(cls, rec):
    f = cls.step

    def g(self, *a, **k):
        tick = self.n_step % 5 == 0
        t0 = time.perf_counter(); r = f(self, *a, **k)
        if tick:
            rec.append((time.perf_counter() - t0) * 1e3)
        return r
    cls.step = g


def ctrl_grasp(out):
    fast_clear()
    import runpy
    import door_motm_grasp_ctrl as G
    rec = []
    timed_class(G.MovingGraspCtrl, rec)
    os.environ['POOL'] = str(POOL)
    os.environ['FASTKIN'] = '1' if POOL else '0'                     # 池化組同時用抓持 r5 fastkin（命令差 ≤2e-13）
    sys.argv = ['test_door_motm_grasp_fake.py', 'nominal', '0.05', out.replace('.json', '_case.json')]
    runpy.run_path(os.path.join(HERE, 'test_door_motm_grasp_fake.py'), run_name='__main__')
    json.dump(dict(tick_ms=rec), open(out, 'w'))


def ctrl_exit(out):
    fast_clear()
    import door_exit_ctrl as XE
    rec = []
    timed_class(XE.DoorExitCtrl, rec)
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
    RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
    L = np.load(RUN + '/log.npz')['rows']; Pp = np.load(RUN + '/plan.npz')
    R7 = L[L[:, 1] == 7]; qx = np.r_[R7[-1, 5:8], R7[-1, 8:14]]; phx = float(R7[-1, 2]); om = float((R7[-1, 2] - R7[-11, 2]) / 0.1)
    sx = np.load(RUN + '/open_steps.npz')['ctrl'][-1, 32:38]
    xc = XE.DoorExitCtrl(ident, Pp['T_WH_bind'], float(Pp['phi_bind']), sx, [np.zeros(9)] * 8, np.zeros(9), np.zeros(9), al * (qx[3:] - sx), 0.0089, phx, om, np.zeros((1, 2)), qx[3:], scope_end='X2', pool=POOL)
    outs = []
    q = qx.copy()
    wu = xc.warmup(qx, phx) if POOL else None
    for k in range(1500):
        o = xc.step(k * 0.01, q.copy(), phx)
        outs.append(np.r_[o['base_twist'], o['arm_target']].tolist())
        if o.get('abort') or (xc.stop is not None and xc.stop_ok is not None) or (xc.done and xc.stop_ok is not None):
            break
        cs, sn = math.cos(q[2]), math.sin(q[2]); vx, vy, wz = o['base_twist']
        q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz]); q[3:] += al * (o['arm_target'] - q[3:] + (qx[3:] - sx))
    xc.close_pool()
    json.dump(dict(tick_ms=rec[1:] if POOL else rec, phase=xc.phase, stop=xc.stop, done=xc.done, sim_s=round(k * 0.01, 2), outs=outs, warmup_ms=wu), open(out, 'w'), default=float)


def bg_planner(out, stopf):
    WB = fast_clear()
    src = open(os.path.join(HERE, 'door_motm_reuse_check.py')).read()
    sys.argv = ['x', os.path.join(SP, 'mg3')]
    ns = dict(__file__=os.path.join(HERE, 'door_motm_reuse_check.py'), __name__='rc_head')
    exec(compile(src[:src.index("Zn = np.load(")], 'reuse_head', 'exec'), ns)
    Zn = np.load(os.path.join(SP, 'mg3', 'nominal_0.05_ready.npz'))
    T_e, x0, v0, T_HE = ns['start_of'](Zn)
    b = x0[:3]
    Tr = 1.875 * (ns['GOAL'] - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
    N = int(math.ceil((Tr + 2.0) / 0.2))
    walls = []
    while not os.path.exists(stopf) and len(walls) < 50:
        WB.set_model(phi0=float(Zn['phi']), finger_q=ns['FQ'])
        o = WB.DoorWBOpt(T_e, np.eye(4), x0, ns['GOAL'], N=N, dt=0.2, T_HE=T_HE, project_start=False, v0=v0, grad='analytic'); o.init_corotate(Tr)
        t0 = time.perf_counter(); info = o.solve(max_iter=45); walls.append(dict(wall_s=round(time.perf_counter() - t0, 2), why=info['why'], iters=info['iters']))
        json.dump(dict(walls=walls), open(out, 'w'))
        if os.environ.get('ONCE'):
            break


def bg_percep(out, stopf, seed):
    import d1_handle_detect_v2 as D1
    V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    Kc = np.array(m['intrinsics_readback']['K'])
    fr = []
    for f in m['frames']:
        dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
        fr.append((dep[:, :, 0] if dep.ndim == 3 else dep, D1.cam_T(f)))
    rng = np.random.default_rng(seed)
    ms, t_start, i = [], time.perf_counter(), 0
    t_end = t_start + float(os.environ.get('PERCEP_S', '1e9'))
    while not os.path.exists(stopf) and time.perf_counter() < t_end:
        t0 = time.perf_counter()
        dep, Tc = fr[i % len(fr)]; i += 1
        _, d = D1.detect(dep, Kc, Tc, rng)
        ms.append(float(d))
        lag = 1.0 / 30.0 - (time.perf_counter() - t0)
        if lag > 0:
            time.sleep(lag)
    el = time.perf_counter() - t_start
    json.dump(dict(detect_ms=ms, n=len(ms), elapsed_s=round(el, 2), rate_hz=round(len(ms) / max(el, 1e-9), 2)), open(out, 'w'))


def sub(args, bg=False):
    cmd = (SCOPE_CTL + PY_CTL if args[0].startswith('ctrl_') else SCOPE + PY) + [os.path.abspath(__file__)] + args
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) if bg else subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3000)


def drive(out):
    W = os.path.join(SP, 'conc'); os.makedirs(W, exist_ok=True)
    res = dict(alone={}, loaded={}, bg_alone={}, bg_loaded={})
    # 背景負載單跑
    p = subprocess.run(SCOPE + ['env', 'ONCE=1'] + PY + [os.path.abspath(__file__), 'bg_planner', f'{W}/pl_alone.json', f'{W}/nostop'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3000)
    res['bg_alone']['planner'] = json.load(open(f'{W}/pl_alone.json'))['walls']
    subprocess.run(SCOPE + ['env', 'PERCEP_S=10'] + PY + [os.path.abspath(__file__), 'bg_percep', f'{W}/pc_alone.json', f'{W}/nostop', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    a = json.load(open(f'{W}/pc_alone.json')); res['bg_alone']['percep'] = dict(rate_hz=a['rate_hz'], detect_ms=stats(a['detect_ms']))
    print('bg alone', res['bg_alone'], flush=True)
    for cond in ('alone', 'loaded'):
        stopf = f'{W}/stop_{cond}'
        if os.path.exists(stopf):
            os.remove(stopf)
        bgs = []
        if cond == 'loaded':
            bgs = [sub(['bg_planner', f'{W}/pl_loaded.json', stopf], bg=True), sub(['bg_percep', f'{W}/pc_loaded_1.json', stopf, '1'], bg=True), sub(['bg_percep', f'{W}/pc_loaded_2.json', stopf, '2'], bg=True)]
            time.sleep(5.0)
        for ctl in ('open', 'grasp', 'exit'):
            r = sub([f'ctrl_{ctl}', f'{W}/{ctl}_{cond}.json'])
            if r.returncode != 0:
                res[cond][ctl] = dict(error=r.stderr.decode()[-800:]); continue
            d = json.load(open(f'{W}/{ctl}_{cond}.json'))
            res[cond][ctl] = dict(tick_ms=stats(d['tick_ms']), **{k: d[k] for k in d if k not in ('tick_ms', 'outs')})
            print(cond, ctl, res[cond][ctl]['tick_ms'], flush=True)
        if bgs:
            open(stopf, 'w').close()
            for b in bgs:
                b.wait(timeout=600)
            res['bg_loaded']['planner'] = json.load(open(f'{W}/pl_loaded.json'))['walls']
            for i in (1, 2):
                a = json.load(open(f'{W}/pc_loaded_{i}.json')); res['bg_loaded'][f'percep_{i}'] = dict(rate_hz=a['rate_hz'], detect_ms=stats(a['detect_ms']))
            print('bg loaded', res['bg_loaded'], flush=True)
    res['note'] = '計算時間（牆鐘）；背景每程序限 1 核、控制器範圍 (1＋POOL) 核；假世界照模擬時間步進。偵測頻率為每路上限 30 Hz 下實際達成值。'
    res['pool'] = POOL
    res['prep_fc'] = PREP_FC
    res['deadline_ms'] = DEADLINE
    res['machine'] = dict(cpu=open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': '), nproc=os.cpu_count())
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'drive':
        drive(sys.argv[2])
    elif m == 'ctrl_open':
        ctrl_open(sys.argv[2])
    elif m == 'ctrl_grasp':
        ctrl_grasp(sys.argv[2])
    elif m == 'ctrl_exit':
        ctrl_exit(sys.argv[2])
    elif m == 'bg_planner':
        bg_planner(sys.argv[2], sys.argv[3])
    elif m == 'bg_percep':
        bg_percep(sys.argv[2], sys.argv[3], int(sys.argv[4]))
