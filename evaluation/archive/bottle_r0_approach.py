#!/usr/bin/env python3
"""R0：從當下預測構型出發、左右側都可的移動抓取接近（離線；Codex 20261111_004556 第 2、3 項；Howard 11/11 00:18「不要手臂有預抓取姿勢，物品在右邊怎麼辦」）。
不做座標鏡射；不經舊 q_ready；不把固定關節姿態改名成預抓分支。輸入：
  量測物件（區域 xy、支撐高 z）、命令生效時刻的預測狀態（底盤位姿、沿車道速度、手臂關節）、桌面先驗（區域；第一桌方盒）。
接近候選（依量測與當下狀態生成）：
  車道 y＝y_meas − 0.4275（同 bottle_tlane）；行進方向＝當下車頭方向（不轉頭；底盤若需轉向，本版不處理，列為不可行候選而非繞成左側）。
  抓取姿態＝側向水平（工具 z 朝物件、＝區域 +y），手指開合軸取 ±x 兩種（flip），IK 種子依序＝當下手臂構型、當下手臂但 q1 轉向量測物件在機身座標的方位、固定亂數種子（不含 q_ready）。
  窗＝沿行進方向在量測 x 前後（行進 +x：[x−0.35, x+0.15]；行進 −x：[x+0.35, x−0.15] 依序）。
核對（每個候選）：
  1 窗內 IK 分支連續（預抓、抓取、抬升三位姿；每 2 cm 延拓；位置 < 1 mm、姿態 < 0.01 rad；關節離 LITE6_SAFE ≥ 0.05 rad）。
  2 轉移：當下手臂 → 窗起點的預抓構型，關節空間 min-jerk（起訖速度＝0：未承接實際手臂速度，也未證明與窗內追蹤速度連續——R1 必修）；時長取速度 v_arm／加速度 a_arm 的較嚴者（v_arm、a_arm＝控制器 W_ARM）；
    須 ≤ 底盤以當下沿車道速度到窗起點的時間（不改底盤速度）；報峰值關節速度／加速度對限制的比。
  3 碰撞：轉移與窗內每個取樣，手臂連桿點雲（手指、夾爪殼除外於窗內抓取段）對第一桌方盒 ≥ 0.02 m、自碰撞（非相鄰、非設計接觸）≥ 0.005 m；
    底盤同時沿車道移動（全身），底盤圓盤 0.39 對桌 ≥ 0.02 m。
  4 機身座標：物件在底盤座標的 y（左＋右−）、TCP 在底盤座標的 y 與 q1 符號（手臂從哪側伸出）；轉移路徑離 q_ready 的最小關節距離（不經過）。
    python3 evaluation/bottle_r0_approach.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import solve_ik  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from verify_self_collision import DESIGNED_CONTACT, link_clouds  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s1_geom as G  # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
XML = open(URDF).read()
K = WholeBodyKinematics.from_urdf_string(XML)
CL = link_clouds(XML, max_pts=300)
IDX = [K.dof_names.index(f'joint{i}') for i in range(1, 7)]
ARM_LINKS = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
GRIP = ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link')
V_ARM, A_ARM = 0.9999, 19.984
PRE_M, LIFT_M, DEPTH, BAND, Y_REACH = 0.06, 0.05, 0.004, 0.10, 0.4275
LIM_M, ENV_M, SELF_M, BASE_M, R_BASE = 0.05, 0.02, 0.005, 0.02, 0.39
Q_VHIGH = np.array([0.002, 0.014, 2.19, -2.387, -1.292, 2.473])
Q_TUCK = np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
Q_READY_NOM = np.array([0.3890373283602744, -0.19178500500278725, 1.054655868294558, -1.4408633140911007, -1.2030321024228567, 1.222226731547478])   # 名目趟 readback（只用來量「不經過」）
_adj, _rigid = set(), {}


def _find(x):
    while _rigid.get(x, x) != x:
        x = _rigid[x]
    return x


for _j in K.joints.values():                                   # 同 verify_self_collision：相鄰與固定接頭剛性群組不計自碰撞
    _adj.add(frozenset((_j.parent, _j.child)))
    if _j.jtype not in ('revolute', 'prismatic', 'continuous'):
        _a, _b = _find(_j.parent), _find(_j.child)
        if _a != _b:
            _rigid[_a] = _b
SELF_PAIRS = [(x, y) for i, x in enumerate(ARM_LINKS) for y in ARM_LINKS[i + 1:]
              if frozenset((x, y)) not in _adj and frozenset((x, y)) not in DESIGNED_CONTACT and _find(x) != _find(y)]
BASE_H = 0.45                                                   # 底盤（含光達座）頂高上限：只核對貼地、低於此高度的方盒（桌腳）；桌面板離地 0.565 m 不與底盤圓盤相交
_RNG = np.random.default_rng(7)                                 # 單一固定亂數產生器依序取 6 組（原本每組重建 rng(7) ⇒ 六組相同；Codex 20261111_011111）
RNG_SEEDS = [_RNG.uniform(LITE6_SAFE.lower * 0.8, LITE6_SAFE.upper * 0.8) for _ in range(6)]


def table_boxes(z1=0.58):
    return [(n, np.asarray(c, float), np.asarray(s, float)) for n, c, s in G.table_boxes(dict(G.S1_TABLE, top_z=z1))]


def box_dist(P, c, s):
    d = np.abs(P - c) - s / 2
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0)


def poses(xb, yb, zs, flip):
    T_g = BR.grasp_pose(xb, yb, zs + BAND, flip=flip)
    T_g[:3, 3] += DEPTH * T_g[:3, 2]
    T_p = T_g.copy()
    T_p[:3, 3] -= PRE_M * T_g[:3, 2]
    T_l = T_g.copy()
    T_l[2, 3] += LIFT_M
    return T_p, T_g, T_l


def ik(base, seed, T):
    qf = np.zeros(9)
    qf[:3] = base
    qf[IDX] = seed
    r = solve_ik(K, qf, T)
    if r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2:
        return r.q
    return None


def jmargin(qa):
    return float(min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k]) for k in range(6)))


def env(qf, boxes, skip_grip=False):
    W = {n: (lambda T: (T[:3, :3] @ CL[n].T).T + T[:3, 3])(K.fk(qf, n)) for n in ARM_LINKS}
    best = (np.inf, None)
    for n, P in W.items():
        if skip_grip and n in GRIP:
            continue
        for bn, c, s in boxes:
            d = float(box_dist(P, c, s).min())
            if d < best[0]:
                best = (d, f'{n}|{bn}')
    sc = (np.inf, None)
    for a, b in SELF_PAIRS:
        d = float(np.min(np.linalg.norm(W[a][:, None, :] - W[b][None, ::3, :], axis=2)))
        if d < sc[0]:
            sc = (d, f'{a}|{b}')
    low = [(bn, c, s) for bn, c, s in boxes if c[2] - s[2] / 2 < BASE_H]
    bc = min((float(box_dist(np.array([[qf[0], qf[1], c[2]]]), c, s)[0]) - R_BASE for bn, c, s in low), default=np.inf)
    return best, sc, bc


def branch(xs, y_lane, yaw, Ts, seeds):
    """沿窗（依行進順序 xs）延拓三位姿；回第一個全窗可行的種子結果。"""
    for si, sd in enumerate(seeds):
        qs, prev, ok = [], np.asarray(sd, float), True
        for x in xs:
            row = []
            for T in Ts:
                q = ik((x, y_lane, yaw), prev, T)
                if q is None or jmargin(q[IDX]) < LIM_M:
                    ok = False
                    break
                row.append(q)
                prev = q[IDX]
            if not ok:
                break
            qs.append(row)
            prev = row[0][IDX]
        if ok:
            return si, qs
    return None, None


def min_jerk_T(dq):
    dq = np.abs(dq)
    return float(max(np.max(1.875 * dq / V_ARM), np.max(np.sqrt(5.7735 * dq / A_ARM)), 1e-3))


def case(name, xb, yb, zs, heading, x_now, v_along, q_now, boxes):
    yaw = 0.0 if heading > 0 else math.pi
    y_lane = yb - Y_REACH
    if heading > 0:
        xs = np.arange(xb - 0.35, xb + 0.15 + 1e-9, 0.02)
    else:
        xs = np.arange(xb + 0.35, xb - 0.15 - 1e-9, -0.02)
    out = dict(name=name, heading='+x' if heading > 0 else '−x', yaw=yaw, y_lane=round(y_lane, 4), window=[round(float(xs[0]), 3), round(float(xs[-1]), 3)],
               q_now=np.round(q_now, 3).tolist(), x_now=x_now, v_along=v_along)
    d0 = abs(xs[0] - x_now)
    t_base = d0 / max(v_along, 1e-6)
    out['t_base_to_window_s'] = round(t_base, 2)
    cands = []
    for flip in (False, True):
        Ts = poses(xb, yb, zs, flip)
        bear = math.atan2(*((np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]) @ (np.array([xb, yb]) - np.array([xs[0], y_lane])))[::-1]))
        q_bear = np.array(q_now, float).copy()
        q_bear[0] = float(np.clip(bear, LITE6_SAFE.lower[0] + LIM_M, LITE6_SAFE.upper[0] - LIM_M))   # 當下手臂、只把 q1 轉向量測物件在機身座標的方位（種子，不是必經姿態）
        seeds = [q_now, q_bear] + RNG_SEEDS
        si, qs = branch(xs, y_lane, yaw, Ts, seeds)
        if qs is None:
            cands.append(dict(flip=flip, ok=False, why='no_continuous_branch'))
            continue
        q_pre0 = qs[0][0][IDX]
        dq = q_pre0 - q_now
        T_tr = min_jerk_T(dq)
        n = 21
        worst_env, worst_self, worst_base, dmin_ready = (np.inf, None), (np.inf, None), np.inf, np.inf
        for k in range(n):
            s = k / (n - 1)
            sm = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            qa = q_now + sm * dq
            x_b = x_now + heading * min(v_along * s * T_tr, d0)       # 底盤同時沿車道前進（全身）
            qf = np.r_[x_b, y_lane, yaw, qa]
            e, sc, bc = env(qf, boxes)
            worst_env = min(worst_env, e, key=lambda z: z[0])
            worst_self = min(worst_self, sc, key=lambda z: z[0])
            worst_base = min(worst_base, bc)
            dmin_ready = min(dmin_ready, float(np.linalg.norm(qa - Q_READY_NOM)))
        we_win, ws_win = (np.inf, None), (np.inf, None)
        for row in qs[::3]:
            for q in row:
                e, sc, bc = env(q, boxes, skip_grip=True)
                we_win = min(we_win, e, key=lambda z: z[0])
                ws_win = min(ws_win, sc, key=lambda z: z[0])
                worst_base = min(worst_base, bc)
        qmid = qs[len(qs) // 2][1]
        Tb = np.eye(4)
        Tb[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        Tb[:2, 3] = qmid[:2]
        obj_b = np.linalg.inv(Tb) @ np.r_[xb, yb, zs, 1.0]
        tcp_b = np.linalg.inv(Tb) @ np.r_[K.fk(qmid, 'link_tcp')[:3, 3], 1.0]
        vpk = float(np.max(1.875 * np.abs(dq) / T_tr) / V_ARM)
        apk = float(np.max(5.7735 * np.abs(dq) / T_tr ** 2) / A_ARM)
        ok = (T_tr <= t_base and worst_env[0] >= ENV_M and we_win[0] >= ENV_M and worst_self[0] >= SELF_M and ws_win[0] >= SELF_M and worst_base >= BASE_M)
        cands.append(dict(flip=flip, ok=bool(ok), seed_index=si, seed='當下手臂' if si == 0 else ('當下手臂＋q1 轉向物件方位' if si == 1 else f'亂數種子#{si - 1}'), T_transition_s=round(T_tr, 2),
                          fits_before_window=bool(T_tr <= t_base), v_peak_ratio=round(vpk, 3), a_peak_ratio=round(apk, 3),
                          env_transition=(round(worst_env[0], 4), worst_env[1]), self_transition=(round(worst_self[0], 4), worst_self[1]),
                          env_window=(round(we_win[0], 4), we_win[1]), self_window=(round(ws_win[0], 4), ws_win[1]), base_table_min=round(worst_base, 4),
                          window_joint_margin_min=round(min(jmargin(q[IDX]) for row in qs for q in row), 4), q_pre_start=np.round(q_pre0, 3).tolist(),
                          object_body_y=round(float(obj_b[1]), 3), tcp_body_y=round(float(tcp_b[1]), 3), q1_mid=round(float(qmid[IDX][0]), 3),
                          side='左' if obj_b[1] > 0 else '右', arm_side='左' if tcp_b[1] > 0 else '右', min_dist_to_q_ready=round(dmin_ready, 3)))
    feas = [c for c in cands if c['ok']]
    best = min(feas, key=lambda c: c['T_transition_s']) if feas else None
    out.update(candidates=cands, chosen=None if best is None else dict(flip=best['flip'], T_transition_s=best['T_transition_s']), ok=best is not None)
    return out


def main():
    out_path = sys.argv[1]
    xb, yb, zs = -0.0232, 0.4081, 0.58                        # 登錄瓶位（量測值同等；名目趟量測 (−0.0227, 0.4081)）
    boxes = table_boxes(zs)
    res = []
    for q_name, q0 in (('V_HIGH 觀察姿態', Q_VHIGH), ('K2_READY 收攏姿態', Q_TUCK)):
        res.append(case(f'左側（行進 +x）＋{q_name}', xb, yb, zs, +1, xb - 0.35 - 1.2, 0.35, q0, boxes))
        res.append(case(f'右側（行進 −x）＋{q_name}', xb, yb, zs, -1, xb + 0.35 + 1.2, 0.35, q0, boxes))
    S = dict(object=dict(x=xb, y=yb, z_support=zs), rules=dict(v_arm=V_ARM, a_arm=A_ARM, lim_m=LIM_M, env_m=ENV_M, self_m=SELF_M, base_m=BASE_M, pre_m=PRE_M, lift_m=LIFT_M),
             note='離線；不經 q_ready（只量距離）；種子＝當下手臂＋固定亂數；底盤沿當下車頭方向的車道等速（不轉向）；窗內手指／夾爪殼對桌不計（抓取段另由 GC 系列管）',
             cases=res)
    json.dump(S, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
    for c in res:
        ch = c['chosen']
        print(c['name'], '可行' if c['ok'] else '不可行', '' if ch is None else f"flip={ch['flip']} 轉移 {ch['T_transition_s']} s（到窗 {c['t_base_to_window_s']} s）")
        for k in c['candidates']:
            print('   ', {kk: k[kk] for kk in ('flip', 'ok', 'why') if kk in k} if not k.get('ok') and 'why' in k else
                  {kk: k.get(kk) for kk in ('flip', 'ok', 'seed', 'T_transition_s', 'v_peak_ratio', 'a_peak_ratio', 'env_transition', 'self_transition', 'env_window', 'base_table_min', 'side', 'arm_side', 'q1_mid', 'min_dist_to_q_ready')})


if __name__ == '__main__':
    main()
