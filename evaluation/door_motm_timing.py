#!/usr/bin/env python3
"""MotM 開門第一包（5）計算時間帳——量測部分（本機桌機，單執行緒）：
A 開門規劃 door_wbopt：full_2 綁定起點冷啟（同執行器參數）的總時間、迭代數、每迭代各項（碰撞線性化、σ 梯度、QP、投影、merit）；verify 與 dense_reference。
B 熱啟動：以 full_2 的解為初值，起點加擾動（底盤位置 1／2／5 cm、偏航 1°；手臂投影回閉鏈）後的迭代數與時間（只量時間與是否可行，重用判準在第一包（4））。
C 控制拍：開門執行層（closure）與退出控制器在假世界的每拍耗時分解（求解、閉鏈 IK、停止掃掠）。
    python3 evaluation/door_motm_timing.py OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
if os.environ.get('FAST_CLEAR', '0') == '1':
    import door_clear_door_fast as _F
    WB.clear_door = _F.clear_door_fast                             # 等值快速版（door_clear_door_fast_check 逐構型核過）
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P = np.load(os.path.join(RUN, 'plan.npz'))
X_sol, T_WH_b, phi_b, T_HE = P['X'], P['T_WH_bind'], float(P['phi_bind']), P['T_HE']
x0 = X_sol[0].copy()
PHI_GOAL = math.radians(85.0)
WB.set_model(phi0=phi_b, finger_q=0.00174)
out = {}

# 計時包裝
tm = dict(active_collisions=0.0, sigma_grad=0.0, qp=0.0, project=0.0, merit=0.0)
def wrap(obj_or_mod, name, key, is_method):
    f = getattr(obj_or_mod, name)
    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); tm[key] += time.perf_counter() - t0; return r
    setattr(obj_or_mod, name, g)
WB.sigma_grad_orig = WB.sigma_grad
def mk(x0_, X_init=None):
    b = x0_[:3]
    T_ramp = 1.875 * (PHI_GOAL - x0_[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
    N = int(math.ceil((T_ramp + 2.0) / 0.2))
    if X_init is not None:
        N = len(X_init) - 1                                            # 熱啟動：沿用既有解的節點數
    o = WB.DoorWBOpt(T_WH_b, np.eye(4), x0_, PHI_GOAL, N=N, dt=0.2, T_HE=T_HE, project_start=False)
    if X_init is None:
        o.init_corotate(T_ramp)
    else:
        o.X = X_init.copy(); o.X[0] = x0_; o.init = dict(kind='warm')
    for nm, key in (('active_collisions', 'active_collisions'), ('qp', 'qp'), ('project', 'project'), ('merit', 'merit')):
        f = getattr(o, nm)
        def g(*a, _f=f, _k=key, **k):
            t0 = time.perf_counter(); r = _f(*a, **k); tm[_k] += time.perf_counter() - t0; return r
        setattr(o, nm, g)
    return o
_sg = WB.sigma_grad
def sg(*a, **k):
    t0 = time.perf_counter(); r = _sg(*a, **k); tm['sigma_grad'] += time.perf_counter() - t0; return r
WB.sigma_grad = sg
# A 冷啟
for k_ in tm: tm[k_] = 0.0
o = mk(x0)
t0 = time.perf_counter(); info = o.solve(max_iter=45); t_solve = time.perf_counter() - t0
t0 = time.perf_counter(); v = o.verify(); t_ver = time.perf_counter() - t0
t0 = time.perf_counter(); o.dense_reference(4); t_dense = time.perf_counter() - t0
n_acc = sum(1 for r in o.log if 'step' in r)
out['A_cold'] = dict(N=o.N, iters_logged=info['iters'], accepted=n_acc, why=info['why'], solve_s=round(t_solve, 2), per_iter_s=round(t_solve / max(1, info['iters'] - 1), 3), breakdown_s={k: round(v_, 2) for k, v_ in tm.items()},
                     verify_s=round(t_ver, 2), dense_s=round(t_dense, 2), feasible=v['feasible'], same_as_full2=float(np.abs(o.X - X_sol).max()) if o.X.shape == X_sol.shape else None)
print('A', out['A_cold'], flush=True)
# B 熱啟動
out['B_warm'] = []
for tag, db in ((('none', np.zeros(3)), ('xy2cm', np.r_[0.014, 0.014, 0.0])) if os.environ.get('FAST_CLEAR') == '1' else (('none', np.zeros(3)), ('xy1cm', np.r_[0.01, 0.0, 0.0]), ('xy2cm', np.r_[0.014, 0.014, 0.0]), ('xy5cm', np.r_[0.05, 0.0, 0.0]), ('yaw1deg', np.r_[0.0, 0.0, math.radians(1.0)]))):
    for k_ in tm: tm[k_] = 0.0
    xs = x0.copy(); xs[:3] += db
    o2 = mk(x0)                                                        # 用來投影手臂
    xs = o2.project(xs, iters=30)
    try:
        e0 = float(np.abs(o2.closure(xs)[0]).max())
        o3 = mk(xs, X_init=X_sol) if e0 < 1e-6 else None
    except ValueError as e_:
        o3 = None
    if o3 is None:
        out['B_warm'].append(dict(tag=tag, start_projection_failed=True)); continue
    Xi = X_sol.copy(); Xi[:, :3] += db * np.linspace(1, 0, len(Xi))[:, None]   # 初值：起點擾動沿時間線性淡出
    Xi = np.array([Xi[0]] + [o3.project(x, iters=12) for x in Xi[1:]]); Xi[0] = xs
    o3.X = Xi
    t0 = time.perf_counter(); inf3 = o3.solve(max_iter=45); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); v3 = o3.verify(); tv = time.perf_counter() - t0
    out['B_warm'].append(dict(tag=tag, iters=inf3['iters'], why=inf3['why'], solve_s=round(ts, 2), verify_s=round(tv, 2), feasible=v3['feasible'], failed=v3['failed'], breakdown_s={k: round(v_, 2) for k, v_ in tm.items()}))
    print('B', out['B_warm'][-1], flush=True)
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
