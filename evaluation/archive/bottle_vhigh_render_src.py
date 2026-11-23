#!/usr/bin/env python3
"""V_HIGH B 案序列的合成來源（給 isaac_bottle_render_seq.py 只渲染、不跑控制；Codex 20261102_021353 核可 0.60、0.75 各一次）。
**合成序列，不是實跑紀錄**：底盤＝名目趟 bottle_w1_segtarget_nominal1 的實測軌跡；手臂＝bottle_vhigh_seq_check 的 B 案（K2_READY→V_HIGH→保持→名目備妥→原漸變到 T_pre，
漸變段以 IK 延續求關節）；手指維持名目趟同時刻量測值（張開）；瓶靜置在第一桌（Z1）名目瓶位、直立。0.01 s 一步（由 0.05 s 序列線性插值）。
輸出 runs/vhigh_seq_src_zXXX/：s1_steps.npz（重播需要的欄位）、s1_report.json（readback、obstacle 取自名目趟；table1_z＝Z1；synthetic＝true 與序列參數）。
    python3 evaluation/bottle_vhigh_render_src.py 0.75
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BRF                                # noqa: E402
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_segtarget_plan as SP                      # noqa: E402
import bottle_vhigh_seq_check as VC                     # noqa: E402

DT_OUT = 0.01


def sequence_B(tz):
    """回傳 (T, Q9, info)：0.05 s 一步的 B 案序列（t 由 0 起；t_u0 之前手臂在 K2_READY）。"""
    L = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library.json')))
    q_vh = np.array(L['views']['V_HIGH']['q'])
    D = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))
    q_ready = np.array(D['q_ready'])
    t_all, base_all = VC.base_traj()
    bx = lambda t: np.array([float(np.interp(t, t_all, base_all[:, i])) for i in range(3)])   # noqa: E731
    t_blend = float(np.interp(VC.X_BLEND, base_all[:, 0], t_all))
    T_U = math.ceil(1.875 * float(np.abs(q_vh - VC.Q_TUCK).max()) / B.W_ARM['v_arm'] / VC.DT) * VC.DT
    T_VR = math.ceil(1.875 * float(np.abs(q_ready - q_vh).max()) / B.W_ARM['v_arm'] / VC.DT) * VC.DT
    t_dep = t_blend - T_VR
    t_v0 = t_dep - VC.T_OBS_NEED
    t_u0 = t_v0 - VC.T_SETTLE - T_U
    P = SP.poses(VC.BOTTLE[0], VC.BOTTLE[1], tz)
    T_rel = G.C.K.fk(np.r_[0.0, 0.0, 0.0, q_ready], 'link_tcp')
    T, Q = [], []
    q = q_ready.copy()
    for t in np.arange(0.0, t_blend + VC.T_BLEND + 1e-9, VC.DT):
        b = bx(t)
        if t < t_u0:
            qa = VC.Q_TUCK
        elif t < t_u0 + T_U:
            qa = VC.jline(VC.Q_TUCK, q_vh, (t - t_u0) / T_U)
        elif t <= t_dep:
            qa = q_vh
        elif t <= t_blend:
            qa = VC.jline(q_vh, q_ready, (t - t_dep) / T_VR)
        else:
            Tt = BRF.interp_pose(BRF.pose_world(b, T_rel), P['pre'], BRF.min_jerk((t - t_blend) / VC.T_BLEND))
            ok, _, qn, *_r = RS.ik(b, Tt, [q], iters=200)
            if not ok:
                raise RuntimeError(f'漸變 IK 失敗 t={t:.2f}')
            qa = qn
        if t > t_blend:
            q = qa
        T.append(t)
        Q.append(np.r_[b, qa])
    info = dict(t_u0=t_u0, T_U=T_U, t_v0=t_v0, t_dep=t_dep, T_VR=T_VR, t_blend=t_blend, t_end=t_blend + VC.T_BLEND, q_vhigh=q_vh.tolist(), q_ready=q_ready.tolist(),
                x_u0=float(bx(t_u0)[0]), x_dep=float(bx(t_dep)[0]))
    return np.array(T), np.array(Q), info


def main():
    tz = float(sys.argv[1])
    T, Q, info = sequence_B(tz)
    src = os.path.join(HERE, 'runs', VC.SRC)
    Z = np.load(os.path.join(src, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = np.round(np.arange(0.0, T[-1] + 1e-9, DT_OUT), 6)
    out = dict(t=t)
    for i, k in enumerate(('base_x', 'base_y', 'base_yaw')):
        out[k] = np.interp(t, T, Q[:, i])
    for j in range(6):
        out[f'q{j + 1}'] = np.interp(t, T, Q[:, 3 + j])
    for k in ('finger1_q', 'finger2_q'):
        out[k] = np.interp(t, c['t'], c[k])
    out['bottle_base_x'], out['bottle_base_y'], out['bottle_base_z'] = np.full_like(t, VC.BOTTLE[0]), np.full_like(t, VC.BOTTLE[1]), np.full_like(t, tz)
    out['bottle_qw'], out['bottle_qx'], out['bottle_qy'], out['bottle_qz'] = np.ones_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    cols = list(out.keys())
    d = os.path.join(HERE, 'runs', f'vhigh_seq_src_z{int(round(tz * 100)):03d}')
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, 's1_steps.npz'), rows=np.column_stack([out[k] for k in cols]), cols=np.array(cols))
    rep0 = json.load(open(os.path.join(src, 's1_report.json')))
    rep = dict(readback=rep0['readback'], obstacle=rep0['obstacle'], table1_z=tz, synthetic=True, base_from=VC.SRC, sequence='V_HIGH_B', seq_info=info,
               note='合成序列（非實跑）：底盤取名目趟實測軌跡，手臂為 B 案設計參考，瓶靜置；只供只渲染核對')
    json.dump(rep, open(os.path.join(d, 's1_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"{d}：{len(t)} 步；展臂 {info['t_u0']:.2f}（x {info['x_u0']:.3f}）→ V_HIGH 觀察 {info['t_v0']:.2f}–{info['t_dep']:.2f}（x {info['x_dep']:.3f}）→ 備妥 {info['t_blend']:.2f} → 預抓 {info['t_end']:.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
