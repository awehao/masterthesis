#!/usr/bin/env python3
"""G1 對照：door_wbopt r6 grad='fd'（＝r5）與 grad='analytic' 同題各解一次，比求解狀態、迭代、時間、非線性重核與解的差異。
淨空值兩者都用 door_clear_door_fast（與原式逐構型相同）。案（事先登記，跑前寫定）：
  full2_cold      full_2 綁定起點冷啟（N＝73，同執行器參數）
  run3_cold       第三趟（m8house_door_open_3）綁定起點冷啟
  full2_warm_*    full_2 解為初值，起點擾動 xy1cm／xy2cm／xy5cm／yaw1deg（同時間帳 B，手臂投影回閉鏈）
  motm_pre        MotM 名目 0.05 m/s READY 起點（含 v0）冷啟（同 door_motm_reuse_check 預先規劃）
  motm_warm_*     重用核對不過的三個實際案（combined／v095／yawdist）以預先規劃為初值熱重解
每案記：why、迭代（含拒絕）、接受步數、總時間與碰撞線性化時間、verify（feasible、failed、最小淨空）、終點門角、J，以及兩版解的節點最大差。
「求解狀態」（converged／stalled／qp_infeasible）與「verify 可行」分開列。
    python3 evaluation/door_wbopt_g1_compare.py OUT.json [案名前綴…]"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
PHI_GOAL = math.radians(85.0)
OUT, only = sys.argv[1], sys.argv[2:]
tm = {}


def timed(o):
    f = o.active_collisions

    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); tm['col'] += time.perf_counter() - t0; return r
    o.active_collisions = g
    return o


def run(tag, build):
    out = {}
    for grad in ('fd', 'analytic'):
        tm['col'] = 0.0
        o = timed(build(grad))
        t0 = time.perf_counter(); info = o.solve(max_iter=45); ts = time.perf_counter() - t0
        v = o.verify()
        m = o.merit(o.X)
        out[grad] = dict(why=info['why'], iters=info['iters'], accepted=sum(1 for r in o.log if 'step' in r), rejected=sum(1 for r in o.log if 'reject' in r or 'qp' in r),
                         wall_s=round(ts, 2), col_s=round(tm['col'], 2), feasible=v['feasible'], failed=v['failed'], min_clear=v['min_clear_m'],
                         phi_end_deg=round(math.degrees(float(o.X[-1, 9])), 3), merit=round(float(m[0]), 5), J=round(float(m[1]), 5), viol=round(float(m[2]), 6), X=o.X.copy())
    d = np.abs(out['fd']['X'] - out['analytic']['X']).max(axis=0) if out['fd']['X'].shape == out['analytic']['X'].shape else None
    res = dict(tag=tag, fd={k: v for k, v in out['fd'].items() if k != 'X'}, analytic={k: v for k, v in out['analytic'].items() if k != 'X'},
               X_maxdiff=None if d is None else dict(base_xy_mm=round(float(d[:2].max()) * 1e3, 3), yaw_deg=round(math.degrees(float(d[2])), 4), arm_rad=round(float(d[3:9].max()), 5), phi_deg=round(math.degrees(float(d[9])), 4)))
    print(tag, 'fd', res['fd'], '\n   an', res['analytic'], '\n   dX', res['X_maxdiff'], flush=True)
    res['_X'] = {g: out[g]['X'] for g in out}
    return res


cases = []


def want(tag):
    return not only or any(tag.startswith(p) for p in only)


# ---- full_2 / run3 冷啟
for tag, run_ in (('full2_cold', 'm8house_door_full_2'), ('run3_cold', 'm8house_door_open_3')):
    if not want(tag):
        continue
    P = np.load(os.path.join(HERE, 'runs', run_, 'plan.npz')); R = json.load(open(os.path.join(HERE, 'runs', run_, 'result.json')))
    fq = float(R['plan'].get('finger_q_model', 0.00174))
    T_WH_b, phi_b, T_HE, x0 = P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], P['X'][0].copy()

    def b_cold(grad, T_WH_b=T_WH_b, phi_b=phi_b, T_HE=T_HE, x0=x0, fq=fq):
        WB.set_model(phi0=phi_b, finger_q=fq)
        b = x0[:3]
        Tr = 1.875 * (PHI_GOAL - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
        o = WB.DoorWBOpt(T_WH_b, np.eye(4), x0, PHI_GOAL, N=int(math.ceil((Tr + 2.0) / 0.2)), dt=0.2, T_HE=T_HE, project_start=False, grad=grad)
        o.init_corotate(Tr)
        return o
    r = run(tag, b_cold); r['finger_q'] = fq
    r['fd_vs_logged_plan_max'] = float(np.abs(r['_X']['fd'] - P['X']).max()) if r['_X']['fd'].shape == P['X'].shape else None   # fd 版應重現實跑所用計畫
    cases.append(r)

# ---- full_2 熱啟擾動（同時間帳 B）
P = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
X_sol, T_WH_b, phi_b, T_HE = P['X'], P['T_WH_bind'], float(P['phi_bind']), P['T_HE']
for tag, db in (('full2_warm_xy1cm', np.r_[0.01, 0.0, 0.0]), ('full2_warm_xy2cm', np.r_[0.014, 0.014, 0.0]), ('full2_warm_xy5cm', np.r_[0.05, 0.0, 0.0]), ('full2_warm_yaw1deg', np.r_[0.0, 0.0, math.radians(1.0)])):
    if not want(tag):
        continue
    WB.set_model(phi0=phi_b, finger_q=0.00174)
    o2 = WB.DoorWBOpt(T_WH_b, np.eye(4), X_sol[0], PHI_GOAL, N=len(X_sol) - 1, dt=0.2, T_HE=T_HE, project_start=False)
    xs = o2.project(X_sol[0] + np.r_[db, np.zeros(7)], iters=30)

    def b_warm(grad, xs=xs, db=db):
        WB.set_model(phi0=phi_b, finger_q=0.00174)
        o = WB.DoorWBOpt(T_WH_b, np.eye(4), xs, PHI_GOAL, N=len(X_sol) - 1, dt=0.2, T_HE=T_HE, project_start=False, grad=grad)
        Xi = X_sol.copy(); Xi[:, :3] += db * np.linspace(1, 0, len(Xi))[:, None]
        Xi = np.array([Xi[0]] + [o.project(x, iters=12) for x in Xi[1:]]); Xi[0] = xs
        o.X = Xi; o.init = dict(kind='warm')
        return o
    cases.append(run(tag, b_warm))

# ---- MotM 預先規劃與熱重解
if any(want(t) for t in ('motm_pre', 'motm_warm_combined', 'motm_warm_v095', 'motm_warm_yawdist')):
    sys.argv = [sys.argv[0], os.environ.get('READY_DIR', '/tmp/claude-1000/-home-howardchen-masterthesis/2a47f71f-1a91-4ff0-9dbf-1892d8540042/scratchpad/mg3')]
    src = open(os.path.join(HERE, 'door_motm_reuse_check.py')).read()
    head = src[:src.index("Zn = np.load(")]                          # 只取 start_of／mk 定義（不跑預先規劃與核對迴圈）
    ns = dict(__file__=os.path.join(HERE, 'door_motm_reuse_check.py'), __name__='rc_head')
    exec(compile(head, 'door_motm_reuse_check.py(head)', 'exec'), ns)
    rd = sys.argv[1]
    Zn = np.load(os.path.join(rd, 'nominal_0.05_ready.npz'))
    st_n = ns['start_of'](Zn)
    T_e, x0n, v0n, T_HEn = st_n
    bn = x0n[:3]
    Trn = 1.875 * (PHI_GOAL - x0n[9]) * (float(np.hypot(bn[0], bn[1])) + WB.L_W) / (0.9 * WB.W_LIM)
    Nn = int(math.ceil((Trn + 2.0) / 0.2))
    phin = float(Zn['phi'])

    def b_pre(grad):
        WB.set_model(phi0=phin, finger_q=ns['FQ'])
        o = WB.DoorWBOpt(T_e, np.eye(4), x0n, PHI_GOAL, N=Nn, dt=0.2, T_HE=T_HEn, project_start=False, v0=v0n, grad=grad)
        o.init_corotate(Trn)
        return o
    r = run('motm_pre', b_pre)
    cases.append(r)
    WB.set_model(phi0=phin, finger_q=ns['FQ'])
    o_ref = b_pre('fd'); o_ref.solve(max_iter=45); X_pre = o_ref.X.copy()          # 熱啟初值固定用 fd 版預先規劃（同重用核對）
    for cs in ('combined', 'v095', 'yawdist'):
        tag = f'motm_warm_{cs}'
        if not want(tag):
            continue
        st = ns['start_of'](np.load(os.path.join(rd, f'{cs}_0.05_ready.npz')))

        def b_w(grad, st=st):
            T_e2, x0a, v0a, T_HEa = st
            WB.set_model(phi0=phin, finger_q=ns['FQ'])
            o = WB.DoorWBOpt(T_e2, np.eye(4), x0a, PHI_GOAL, N=Nn, dt=0.2, T_HE=T_HEa, project_start=False, v0=v0a, grad=grad)
            o.X = X_pre.copy(); o.X[0] = o.c0; o.init = dict(kind='warm')
            return o
        cases.append(run(tag, b_w))
for c in cases:
    c.pop('_X', None)
json.dump(dict(cases=cases, note='淨空值兩版相同（door_clear_door_fast）；只有淨空列的線性化不同。時間為本機桌機單執行緒 wall。'), open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
