#!/usr/bin/env python3
"""v3（跨桌高）轉回時長選定：對本批各案的實際收近起點重核三段所需時長（Codex 20261102_070114「時長需重新核對」）。
來源：已到收近的物理趟（A1–A4、B2、C1；C2 未離桌）——收近開始時刻（compact_start）的手臂構型、TCP 底盤座標位姿（＝側持 T_side）、
TCP→瓶相對位姿（逐步紀錄的瓶位姿）；z_back＝第二桌面＋Z_PAD＋H_CLEAR − 包絡最低點（同 bottle_w1_vhigh_relwin_xh_ctrl）。
三段：收近（T_side → 緊湊）、升高（緊湊 → z_back）、外伸（→ T_back），各 40 步 IK 延續（bottle_w1_compact_timing.path_q）。
B1 沒有物理趟（未進物理），另由 test_bottle_w1_xh 的假世界收近起點核對。
選定值＝各案所需最大值 ×1.05 後進位到 0.05 s（不低於原 T_TO／T_UP／T_OUT）。輸出 results/bottle/xh_timing.json。
    python3 evaluation/bottle_xh_timing.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compact_timing as TM                   # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2m              # noqa: E402
import bottle_w1_compactcarry3_ctrl as C3m              # noqa: E402

RUNS = ['bottle_motm_multi_a1_2', 'bottle_motm_multi_a2_1', 'bottle_motm_multi_a3_1', 'bottle_motm_multi_a4_2', 'bottle_motm_multi_b2_1', 'bottle_motm_multi_c1_1']
H_CLEAR, TOP2 = 0.029, 0.60
OUT = os.path.join(HERE, 'results', 'bottle', 'xh_timing.json')


def quat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t0 = next(e['t'] for e in r['events'] if e['what'] == 'compact_start')
    k = int(np.argmin(np.abs(c['t'] - t0)))
    qa = np.array([c[f'q{j}'][k] for j in range(1, 7)])
    q9 = np.r_[c['base_x'][k], c['base_y'][k], c['base_yaw'][k], qa]
    T_side = G.C.K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
    Tw = G.C.K.fk(q9, 'link_tcp')
    Tb = np.eye(4)
    Tb[:3, :3] = quat(c['bottle_qw'][k], c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k])
    Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
    T_rel = np.linalg.inv(Tw) @ Tb
    BP = G.bottle_points(n_ang=12, n_z=5)
    Pt = np.vstack([(T_rel[:3, :3] @ BP.T).T + T_rel[:3, 3], C2m.grip_grid_tcp(qa)])
    zmin = float((T_side[:3, :3] @ Pt.T)[2].min())
    z_back = TOP2 + C2m.Z_PAD + H_CLEAR - zmin
    T_back = T_side.copy()
    T_back[2, 3] = z_back
    T_c = T_side.copy()
    T_c[:3, 3] = C2m.TCP_C
    T_mid = T_c.copy()
    T_mid[2, 3] = z_back
    GC = CP.grip_corners(qa)
    out = dict(run=run, t_compact_start=t0, side_xyz=np.round(T_side[:3, 3], 4).tolist(), zmin_rel=round(zmin, 4), z_back=round(z_back, 4))
    try:
        Q1, s1 = TM.path_q(qa, T_side, T_c, T_rel, GC)
        Q2, s2 = TM.path_q(Q1[-1], T_c, T_mid, T_rel, GC)
        Q3, s3 = TM.path_q(Q2[-1], T_mid, T_back, T_rel, GC)
    except Exception as e:
        out['error'] = repr(e)
        return out
    for nm, sg in (('to_compact', s1), ('raise', s2), ('out', s3)):
        out[nm] = dict(T_need=round(sg['T'], 3), T_v=round(sg['T_v'], 3), T_a=round(sg['T_a'], 3), self_mm=round(sg['worst']['self'] * 1e3, 1),
                       bottle_link_mm=round(sg['worst']['bottle_link'] * 1e3, 1), deck=round(sg['worst']['deck'], 3), margin=round(sg['worst']['margin'], 3))
    return out


def main():
    res = [one(r) for r in RUNS]
    need = {nm: max(x[nm]['T_need'] for x in res if nm in x) for nm in ('to_compact', 'raise', 'out')}
    base = dict(to_compact=C2m.T_TO, raise_=C2m.T_UP, out=C3m.T_OUT)
    pick = {nm: max(math.ceil(need[nm] * 1.05 / 0.05 - 1e-9) * 0.05, {'to_compact': C2m.T_TO, 'raise': C2m.T_UP, 'out': C3m.T_OUT}[nm]) for nm in need}
    out = dict(h_clear=H_CLEAR, top2=TOP2, runs=res, need_max=need, baseline=base, pick={k: round(v, 2) for k, v in pick.items()},
               note='選定＝各案所需最大 ×1.05 進位到 0.05 s，不低於原值；B1 另由假世界收近起點核對')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for x in res:
        print(x['run'], x.get('side_xyz'), 'z_back', x.get('z_back'), {nm: x[nm]['T_need'] for nm in ('to_compact', 'raise', 'out') if nm in x}, x.get('error', ''))
    print('需要最大', need, '⇒ 選定', out['pick'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
