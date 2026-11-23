#!/usr/bin/env python3
"""主線 B（Codex 20261117_021349）：機身輔助視角能否覆蓋完整接近候選的盲段（手腕相機最後有效影格 → READY／綁定）。
軌跡＝door_b_plan.BPlanner 在 Smac 實際門前線上算出的候選（觀察→預抓位過渡的盲段部分＋抓持窗口：插入 S_PRE→0、底盤沿 −a 以 V 走到 READY）。
每個候選相機（固定在 base_link；位置、偏航、俯仰、水平視野；640×480 針孔；RGB 近距 0.1 m、深度有效 0.3 m 起）逐時刻核：
  把手中心與兩端是否在影像內（邊距 20 px）、距離、是否被手臂連桿碰撞點雲遮擋（相機→把手點的線段 5 mm 內有連桿點；把手點 4 cm 內的點不算——夾爪本來就在把手上）、
  每 mm 像素數（fx／距離）。輸出覆蓋率（中心可見且未遮擋且深度 ≥0.3 m 的時間比例）與最長連續看不到的時間。
只是候選幾何與視野分析；**不改任何資產**（新增或調整相機須 Howard 確認）。
    python3 door_aux_cam_coverage.py RUN_DIR PROBE_JSON OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
import door_b_plan as BP
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
K, DOC = TS.K, TS.W.DOC
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[0]['plan']], float)
bp = BP.BPlanner(TS.T_H, TS.d, TS.n_, TS.yaw, TS.Kc)
pl = bp.plan(path, Q_NAV)
assert pl['why'] is None, pl['why']
sb, a_w, a0, T = pl['door_line_sb'], pl['a_w'], pl['a_t0'], pl['T_tr']
q_obs = np.array([f['q'] for f in pl['obs'] if abs(f['a'] - a0) < 1e-6][0])
q_pre = np.array(pl['q_pre'])
print('plan', {k: pl[k] for k in ('door_line_sb', 'a_w', 'a_t0', 'T_tr', 'last_vis_t', 'blind_s', 'goal_xy')}, flush=True)
# 盲段軌跡：過渡（自最後可見時刻）＋抓持窗口
traj = []
t_lv = pl['last_vis_t']
for t in np.arange(t_lv, T + 1e-9, 0.1):
    u = t / T; s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    traj.append((t - t_lv, bp.base(a0 + (a_w - a0) * u, sb), q_obs + s * (q_pre - q_obs)))
seed = q_pre.copy()
for k in range(1, 21):                                               # 抓持：0.1 s 一步，到 READY 2.05 s
    a_ = a_w - bp.V * 0.1 * k
    r_ = BP.S_PRE * max(0.0, 1.0 - k / 12)
    b = bp.base(a_, sb)
    q, _ = bp.solve(b, bp.target(r_), seed)
    seed = q
    traj.append((T - t_lv + 0.1 * k, b, q))
H = bp.H
PTS = [H, H + bp.T_H[:3, 0] * 0.06, H - bp.T_H[:3, 0] * 0.06]
ARM = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2']


def cam_pose(b, p, yaw_deg, pitch_deg):
    """base_link 座標（x 前、y 左、z 上）的相機：位置 p、偏航、俯仰（正＝抬頭）→ 世界的光學座標（z 前、x 右、y 下）。"""
    Tb = np.eye(4); c, s = math.cos(b[2]), math.sin(b[2]); Tb[:2, :2] = [[c, -s], [s, c]]; Tb[:2, 3] = b[:2]; Tb[2, 3] = 0.05
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    fwd = np.array([cy * cp, sy * cp, sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)
    Tc = np.eye(4); Tc[:3, :3] = np.c_[right, down, fwd]; Tc[:3, 3] = p
    return Tb @ Tc


def occluded(cpos, pt, q9):
    seg = pt - cpos; L = np.linalg.norm(seg); dirv = seg / L
    for n in ARM:
        Tn = K.fk(q9, n)
        C = (Tn[:3, :3] @ DOC.CL[n].T).T + Tn[:3, 3]
        if n.startswith('uflite_finger'):
            pass
        v = C - cpos; tt = np.clip(v @ dirv, 0.0, L)
        dd = np.linalg.norm(v - tt[:, None] * dirv, axis=1)
        near_pt = np.linalg.norm(C - pt, axis=1) < 0.04
        if np.any((dd < 0.005) & ~near_pt & (tt < L - 0.01)):
            return n
    return None


CANDS_V = {'C3_hfov69': ((0.10, 0.20, 0.90), -25, -10, 69.0), 'C3_h0.80_hfov69': ((0.10, 0.20, 0.80), -25, -10, 69.0), 'C3_h1.00_hfov69': ((0.10, 0.20, 1.00), -25, -15, 69.0),
           'C3_yaw-15_hfov69': ((0.10, 0.20, 0.90), -15, -10, 69.0), 'C3_yaw-35_hfov69': ((0.10, 0.20, 0.90), -35, -10, 69.0), 'C3_y0.15_hfov69': ((0.10, 0.15, 0.90), -25, -10, 69.0),
           'C6_hfov69': ((0.15, 0.22, 0.70), -35, 10, 69.0)}
CANDS = CANDS_V if os.environ.get('AUX_VARIANTS') == '1' else {'C0_existing_base_D435': ((0.25, 0.0, 0.30), 0, 0, 87.0),
         'C1_front_mast_0.60_up35': ((0.20, 0.0, 0.60), 0, 35, 87.0),
         'C2_mast_top_1.10_down30': ((0.0, 0.0, 1.10), 0, -30, 87.0),
         'C3_left_0.90_yaw-25_down10': ((0.10, 0.20, 0.90), -25, -10, 87.0),
         'C4_right_0.90_yaw25_down10': ((0.10, -0.20, 0.90), 25, -10, 87.0),
         'C5_right_0.70_yaw35_up10': ((0.15, -0.22, 0.70), 35, 10, 87.0),
         'C6_left_0.70_yaw-35_up10': ((0.15, 0.22, 0.70), -35, 10, 87.0)}
res = {}
for name, (p, yw, pt, hfov) in CANDS.items():
    fx = 320.0 / math.tan(math.radians(hfov) / 2)
    rows = []
    for (t, b, q) in traj:
        q9 = np.r_[b, q]
        Tc = cam_pose(b, np.array(p), yw, pt)
        Ti = np.linalg.inv(Tc)
        vis_c = False; inimg = 0; dist = None; occ_by = None
        for j, X in enumerate(PTS):
            pc = (Ti @ np.r_[X, 1.0])[:3]
            if pc[2] < 0.1:
                continue
            u, v = 320 + fx * pc[0] / pc[2], 240 + fx * pc[1] / pc[2]
            if 20 <= u <= 620 and 20 <= v <= 460:
                ob = occluded(Tc[:3, 3], X, q9)
                if ob is None:
                    inimg += 1
                    if j == 0:
                        vis_c = True; dist = float(pc[2])
                elif j == 0:
                    occ_by = ob
        rows.append(dict(t=round(t, 2), center=vis_c, n_pts=inimg, depth=dist, occ=occ_by))
    ok = [r for r in rows if r['center'] and r['depth'] is not None and r['depth'] >= 0.3]
    gaps, cur = [], 0.0
    for r in rows:
        if r['center'] and r['depth'] is not None and r['depth'] >= 0.3:
            gaps.append(cur); cur = 0.0
        else:
            cur += 0.1
    gaps.append(cur)
    dd = [r['depth'] for r in ok]
    res[name] = dict(mount_base_link=dict(xyz=p, yaw_deg=yw, pitch_deg=pt, hfov_deg=hfov), coverage=round(len(ok) / len(rows), 3), longest_gap_s=round(max(gaps), 2),
                     depth_m=[round(min(dd), 3), round(max(dd), 3)] if dd else None, px_per_mm=[round(fx / max(dd) / 1000, 2), round(fx / min(dd) / 1000, 2)] if dd else None,
                     occluders=sorted({r['occ'] for r in rows if r['occ']}), n_samples=len(rows))
    print(name, {k: res[name][k] for k in ('coverage', 'longest_gap_s', 'depth_m', 'px_per_mm', 'occluders')}, flush=True)
json.dump(dict(note='輔助視角候選的幾何覆蓋分析（不改資產）；軌跡＝door_b_plan 在 Smac 實際門前線的候選盲段（過渡後段＋抓持到 READY）', plan={k: pl[k] for k in ('door_line_sb', 'a_w', 'a_t0', 'T_tr', 'last_vis_t', 'blind_s', 'goal_xy')}, blind_traj_s=round(traj[-1][0], 2), cams=res), open(sys.argv[3], 'w'), ensure_ascii=False, indent=1)
