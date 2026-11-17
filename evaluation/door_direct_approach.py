#!/usr/bin/env python3
"""開門「直接接近」離線候選（Howard 20261117：不要側轉觀察、不要預抓位停駐；Codex 20261117_103524）。
實測起點：m8house_b_capture_2 的實際底盤軌跡（log.npz，物理＝理想里程）與起點手臂構型（PREP 開始時 t0 的實測關節＝導航姿態）；
把手＝該趟真值（只作候選幾何；正式執行用提交的估計參考）。底盤在 t0 之後照實際紀錄，紀錄末端之後沿 −a 以 0.05 m/s 外推（抓持中底盤續走）。
候選（逐一列出，不只取最好）：抓法繞橫桿軸旋轉 ψ × 抓取時底盤沿門位置 a_g（＝TCP 到達抓取位姿的時刻 t_g）。
TCP 軌跡（世界座標，一條連續運動）：
  位置：t0→t_ins 三次 Hermite（起點速度＝起點實測 TCP 世界速度、終點速度＝沿接近軸插入速度）到「抓取位姿沿接近軸後退 S_PRE」的點；
        t_ins→t_g 沿接近軸等減速插入到 r＝0（S_PRE 只是軌跡上的計算位置，不停駐）；t_g 之後 TCP 在世界中保持抓取位姿（底盤續走，手臂補償）＝抓持段。
  姿態：t0→t_ins 以最小急動在 SO(3) 測地線由起點轉到抓取姿態（之後固定）。
逐 0.05 s：同分支 IK（以上一點為種子，跳變 ≤ V_ARM·dt）、σ≥下限、限位、淨空（規劃門檻；插入與抓持段手指—橫桿接觸由抓持控制器豁免，env 照核）、
  關節速率／加速度、相機幾何可見（把手中心＋兩端在影像內、相機前方 ≥0.12 m；不是偵測有效）。抓持段核 2.0 s（底盤沿 −a 0.05 m/s）＝可承接開門的抓持窗口代理。
輸出每候選：可行、首個失敗（時刻、原因）、TCP 路徑長、姿態轉角、最大關節速率、最後幾何可見時刻與距離、可見結束到抓取的時間。
運動學離線；不是停止證書（正式執行另以每拍完整核對）。
    python3 door_direct_approach.py RUN_DIR OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import door_motm_grasp_ctrl as G, door_wbopt as W, door_fast_kin as DFK, door_b_plan as BP, wrist_v0_capture as W0
RUN, OUT = sys.argv[1], sys.argv[2]
K = G.K
Rg0, T_HG0, A_H0 = G.PG.baseline_reference()
T_HG0, A_H0 = np.asarray(T_HG0, float), np.asarray(A_H0, float)
S_PRE = float(G.PG.S_PRE)
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
V_ARM, A_ARM, DT = 1.0, 10.0, 0.05
T_INS = float(os.environ.get('T_INS', '0.6'))
MODE = os.environ.get('MODE', 'joint')
T_MOVE = float(os.environ.get('T_MOVE', '0'))
D_LINE, T_LINE, T_BLEND = float(os.environ.get('D_LINE', '0.30')), float(os.environ.get('T_LINE', '2.5')), float(os.environ.get('T_BLEND', '1.0'))
SEEDS = [np.array(x, float) for x in ([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157], [-0.396, -0.118, 1.409, -1.471, -0.398, -0.102], [-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018], [0.0, -0.3, 1.2, 0.0, -0.9, 0.0], [0.4, -0.2, 1.3, 1.4, -0.4, 0.1])]
M = json.load(open(os.path.join(RUN, 'frames_meta.json'))); Kc = np.array(M['K'], float)
T_H = np.array(json.load(open(os.path.join(RUN, 'truth.json')))['T_handle_start'], float)
S = json.load(open(os.path.join(RUN, 'ros_bcap.json')))
t0 = float(os.environ.get('T0', [e['t'] for e in S['events'] if e['what'] == 'mode' and e.get('mode') == 'PREP'][0]))
L = np.load(os.path.join(RUN, 'log.npz'))['L']
i0 = int(np.searchsorted(L[:, 0], t0))
BT = L[i0:, 0:4].copy()                                            # t, x, y, yaw（實測）
v_ext = np.array([-0.05, 0.0])
tl = BT[-1, 0]
ext = [(tl + k * 0.01, BT[-1, 1] + v_ext[0] * k * 0.01, BT[-1, 2] + v_ext[1] * k * 0.01, BT[-1, 3]) for k in range(1, 801)]
BT = np.vstack([BT, np.array(ext)])
q_arm0 = L[i0, 4:10].copy()
n_ = (T_H[:3, :3] @ A_H0).copy(); n_[2] = 0; n_ /= np.linalg.norm(n_)
ax = T_H[:3, 0].copy(); ax[2] = 0; d_ = -ax / np.linalg.norm(ax)
yaw = math.atan2(-n_[1], -n_[0])
bp = BP.BPlanner(T_H, d_[:2], n_[:2], yaw, Kc)
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)


def Rx(psi):
    c, s = math.cos(psi), math.sin(psi)
    R = np.eye(4); R[1:3, 1:3] = [[c, -s], [s, c]]
    return R


def grasp_T(psi, r):
    a = Rx(psi)[:3, :3] @ A_H0
    Tt = np.eye(4); Tt[:3, 3] = r * a
    return T_H @ Tt @ Rx(psi) @ T_HG0


def base_at(t):
    k = int(np.clip(np.searchsorted(BT[:, 0], t), 1, len(BT) - 1))
    w = (t - BT[k - 1, 0]) / max(BT[k, 0] - BT[k - 1, 0], 1e-9)
    return BT[k - 1, 1:4] + min(max(w, 0.0), 1.0) * (BT[k, 1:4] - BT[k - 1, 1:4])


def a_of(b):
    rel = np.asarray(b[:2], float) - T_H[:2, 3]
    return float(-(rel @ d_[:2])), float(rel @ n_[:2])


def so3_log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def so3_exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def mj(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def hermite(p0, v0, p1, v1, T, t):
    s = min(max(t / T, 0.0), 1.0)
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1


def visible(q9):
    Tc = K.fk(q9, W0.OPT); Ti = np.linalg.inv(Tc)
    for j, X in enumerate([T_H[:3, 3], T_H[:3, 3] + T_H[:3, 0] * 0.1, T_H[:3, 3] - T_H[:3, 0] * 0.1]):
        pc = (Ti @ np.r_[X, 1.0])[:3]
        if pc[2] < 0.12:
            return None
        u, v = (Kc @ (pc / pc[2]))[:2]
        if not (5 < u < 635 and 5 < v < 475):
            return None
    return float(np.linalg.norm(T_H[:3, 3] - Tc[:3, 3]))


def candidate(psi_deg, a_g):
    psi = math.radians(psi_deg)
    ts = BT[:, 0]
    A = np.array([a_of(BT[k, 1:3])[0] for k in range(len(BT))])
    idx = np.where(A <= a_g)[0]
    if len(idx) == 0:
        return dict(why='base_never_reaches_a_g')
    t_g = float(ts[idx[0]]); t_ins = t_g - T_INS
    if t_ins - t0 < 1.0:
        return dict(why='too_little_time', t_g=round(t_g, 2))
    b0 = base_at(t0); q9_0 = np.r_[b0, q_arm0]
    T0 = DFK.FC.fk_jac(q9_0)[0]
    b0b = base_at(t0 + 0.05)
    v0 = (DFK.FC.fk_jac(np.r_[b0b, q_arm0])[0][:3, 3] - T0[:3, 3]) / 0.05   # 起點 TCP 世界速度（底盤帶動）
    Tp = grasp_T(psi, S_PRE); Tg = grasp_T(psi, 0.0)
    a_w = Rx(psi)[:3, :3] @ A_H0; a_w = T_H[:3, :3] @ a_w                  # 接近軸（世界；r 增加的方向＝離開把手）
    v_ins = -a_w * (2 * S_PRE / T_INS)                                    # 插入起始速度（等減速到 0）
    w_rot = so3_log(T0[:3, :3].T @ Tg[:3, :3])
    out = dict(psi=psi_deg, a_g=a_g, t_g=round(t_g, 2), T_total=round(t_g - t0, 2), rot_deg=round(math.degrees(float(np.linalg.norm(w_rot))), 1), why=None)
    seed = q_arm0.copy(); Q = [q_arm0.copy()]; tt = [t0]; last_vis = None; path = 0.0; p_prev = T0[:3, 3].copy(); vis_list = []
    t = t0
    q_pre_j = None
    if MODE == 'joint_line':                                           # 關節擺動（夾爪轉向把手）→ 平滑混接 → 沿接近軸直線進入；中途不停
        t_D = t_g - T_LINE                                             # 直線段起點時刻（離把手 D_LINE）
        t_s = max(t0, t_D - T_MOVE)
        T_D = grasp_T(psi, D_LINE); b_D = base_at(t_D); best = None
        for sd in SEEDS:
            qs, rs, ss = DFK.ik_closure_fc(T_D, b_D, sd)
            if rs < 1e-6 and np.isfinite(qs).all() and ss >= W.SIG_FLOOR and np.all(qs >= LO) and np.all(qs <= HI) and bp.ok_cfg(b_D, qs)[0]:
                dist = float(np.abs(qs - q_arm0).max())
                if best is None or dist < best[0]:
                    best = (dist, qs)
        if best is None:
            out['why'] = 'no_line_start_ik'; return out
        q_D = best[1]; out['q_line_start'] = [round(float(x), 3) for x in q_D]; out['joint_dist'] = round(best[0], 3)
        t_b0 = t_D - T_BLEND
        while t < t_g + 2.0 - 1e-9:
            t += DT
            b = base_at(t)
            if t <= t_g:
                qj = q_arm0 + mj((t - t_s) / max(t_D - t_s, 1e-6)) * (q_D - q_arm0)
                Tj = DFK.FC.fk_jac(np.r_[b, qj])[0]
                u = min(max((t - t_D + T_BLEND) / (T_LINE + T_BLEND), 0.0), 1.0)   # 直線段進度（含混接期），r 由 D_LINE 以最小急動減到 0
                r = D_LINE * (1 - mj(u)) if u > 0 else D_LINE
                Tl = grasp_T(psi, r)
                beta = mj((t - t_b0) / T_BLEND) if t > t_b0 else 0.0
                p = (1 - beta) * Tj[:3, 3] + beta * Tl[:3, 3]
                R = Tj[:3, :3] @ so3_exp(beta * so3_log(Tj[:3, :3].T @ Tl[:3, :3]))
                seg = 'swing' if beta <= 0 else ('blend' if beta < 1 else ('line' if r > 1e-4 else 'grasp'))
            else:
                p, R, seg = Tg[:3, 3], Tg[:3, :3], 'hold_grasp'
            Tt = np.eye(4); Tt[:3, :3] = R; Tt[:3, 3] = p
            q, res, sm = DFK.ik_closure_fc(Tt, b, seed if beta > 0 or t > t_g else qj) if t <= t_g else DFK.ik_closure_fc(Tt, b, seed)
            if not (res < 1e-6 and np.isfinite(q).all()):
                out['why'] = 'ik@%.2f(%s)' % (t - t0, seg); break
            if float(np.abs(q - seed).max()) > 1.5 * V_ARM * DT:
                out['why'] = 'joint_jump %.3f@%.2f(%s)' % (float(np.abs(q - seed).max()), t - t0, seg); break
            if sm < W.SIG_FLOOR:
                out['why'] = 'sigma %.4f@%.2f(%s)' % (sm, t - t0, seg); break
            if np.any(q < LO) or np.any(q > HI):
                out['why'] = 'limit@%.2f(%s)' % (t - t0, seg); break
            okc, _, cl = bp.ok_cfg(b, q)
            if not okc and not (seg in ('line', 'grasp', 'hold_grasp') and cl['env'] >= W.THR_PLAN['env'] and cl['self'] >= W.THR_PLAN['self'] and cl['base'] >= W.THR_PLAN['base']):
                out['why'] = 'clear@%.2f(%s) env %.4f self %.4f base %.4f' % (t - t0, seg, cl['env'], cl['self'], cl['base']); break
            vd = visible(np.r_[b, q])
            if vd is not None and seg != 'hold_grasp':
                last_vis = (round(t - t0, 2), round(vd, 3))
            path += float(np.linalg.norm(p - p_prev)); p_prev = p.copy()
            Q.append(q); tt.append(t); seed = q
        t_end_loop = True
    if MODE == 'joint':                                                # 關節空間一次到位：插入起點構型（依 t_ins 的預測底盤）由多種子解，取離目前構型最近者
        b_ins = base_at(t_ins); best = None
        for sd in SEEDS:
            qs, rs, ss = DFK.ik_closure_fc(Tp, b_ins, sd)
            if rs < 1e-6 and np.isfinite(qs).all() and ss >= W.SIG_FLOOR and np.all(qs >= LO) and np.all(qs <= HI) and bp.ok_cfg(b_ins, qs)[0]:
                dist = float(np.abs(qs - q_arm0).max())
                if best is None or dist < best[0]:
                    best = (dist, qs)
        if best is None:
            out['why'] = 'no_pre_insert_ik'; return out
        q_pre_j = best[1]; out['q_pre_insert'] = [round(float(x), 3) for x in q_pre_j]; out['joint_dist'] = round(best[0], 3)
    while MODE != 'joint_line' and t < t_g + 2.0 - 1e-9:
        t += DT
        b = base_at(t)
        if t <= t_ins and q_pre_j is not None:
            t_s = t0 if T_MOVE <= 0 else max(t0, t_ins - T_MOVE)       # 手臂保持導航姿態到 t_s，之後一次到位（關節最小急動）
            q = q_arm0 + mj((t - t_s) / (t_ins - t_s)) * (q_pre_j - q_arm0)
            p_ = DFK.FC.fk_jac(np.r_[b, q])[0]
            sm = float(np.linalg.svd(DFK.FC.fk_jac(np.r_[b, q])[1][:, 3:9], compute_uv=False).min())
            seg = 'approach_joint'
            if sm < W.SIG_FLOOR:
                out['why'] = 'sigma %.4f@%.2f(%s)' % (sm, t - t0, seg); break
            if np.any(q < LO) or np.any(q > HI):
                out['why'] = 'limit@%.2f(%s)' % (t - t0, seg); break
            okc, _, cl = bp.ok_cfg(b, q)
            if not okc:
                out['why'] = 'clear@%.2f(%s) env %.4f self %.4f base %.4f' % (t - t0, seg, cl['env'], cl['self'], cl['base']); break
            vd = visible(np.r_[b, q])
            if vd is not None:
                last_vis = (round(t - t0, 2), round(vd, 3))
            path += float(np.linalg.norm(p_[:3, 3] - p_prev)); p_prev = p_[:3, 3].copy()
            Q.append(q); tt.append(t); seed = q
            continue
        if t <= t_ins:
            p = hermite(T0[:3, 3], v0, Tp[:3, 3], v_ins, t_ins - t0, t - t0)
            R = T0[:3, :3] @ so3_exp(mj((t - t0) / (t_ins - t0)) * w_rot)
            seg = 'approach'
        elif t <= t_g:
            u = (t - t_ins) / T_INS
            r = S_PRE * (1 - u) ** 2                                       # 等減速插入
            p = Tg[:3, 3] + r * a_w; R = Tg[:3, :3]; seg = 'insert'
        else:
            p, R, seg = Tg[:3, 3], Tg[:3, :3], 'hold_grasp'
        Tt = np.eye(4); Tt[:3, :3] = R; Tt[:3, 3] = p
        q, res, sm = DFK.ik_closure_fc(Tt, b, seed)
        if not (res < 1e-6 and np.isfinite(q).all()):
            out['why'] = 'ik@%.2f(%s)' % (t - t0, seg); break
        if float(np.abs(q - seed).max()) > 1.5 * V_ARM * DT:
            out['why'] = 'joint_jump %.3f@%.2f(%s)' % (float(np.abs(q - seed).max()), t - t0, seg); break
        if sm < W.SIG_FLOOR:
            out['why'] = 'sigma %.4f@%.2f(%s)' % (sm, t - t0, seg); break
        if np.any(q < LO) or np.any(q > HI):
            out['why'] = 'limit@%.2f(%s)' % (t - t0, seg); break
        okc, _, cl = bp.ok_cfg(b, q)
        if not okc and not (seg != 'approach' and cl['env'] >= W.THR_PLAN['env'] and cl['self'] >= W.THR_PLAN['self'] and cl['base'] >= W.THR_PLAN['base']):
            out['why'] = 'clear@%.2f(%s) env %.4f self %.4f base %.4f' % (t - t0, seg, cl['env'], cl['self'], cl['base']); break
        vd = visible(np.r_[b, q])
        if vd is not None and seg != 'hold_grasp':
            last_vis = (round(t - t0, 2), round(vd, 3))
        path += float(np.linalg.norm(p - p_prev)); p_prev = p.copy()
        Q.append(q); tt.append(t); seed = q
    Q = np.array(Q)
    if len(Q) > 2:
        V = np.diff(Q, axis=0) / DT; Ac = np.diff(V, axis=0) / DT
        out.update(v_max=round(float(np.abs(V).max()), 3), acc_max=round(float(np.abs(Ac).max()), 2), joint_travel=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3))
        if out['why'] is None and (out['v_max'] > V_ARM or out['acc_max'] > A_ARM):
            out['why'] = 'rate %.2f acc %.2f' % (out['v_max'], out['acc_max'])
    out.update(path_m=round(path, 3), last_visible=last_vis, blind_to_grasp_s=None if last_vis is None else round(t_g - t0 - last_vis[0], 2),
               base_at_grasp=[round(x, 3) for x in a_of(base_at(t_g))], ok=out['why'] is None)
    return out


res = []
for psi in [int(x) for x in os.environ.get('PSI', '0,15,30,-15,-30').split(',')]:
    for a_g in [float(x) for x in os.environ.get('AG', '-0.08,-0.10,-0.12,-0.16,-0.20').split(',')]:
        r = candidate(psi, a_g); res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(dict(mode=MODE, T_MOVE=T_MOVE, D_LINE=D_LINE, T_LINE=T_LINE, T_BLEND=T_BLEND, note='直接接近離線候選；實測起點與底盤軌跡取自 m8house_b_capture_2；把手真值；運動學離線、非停止證書', t0=t0, q_arm0=q_arm0.tolist(), T_INS=T_INS, res=res), open(OUT, 'w'), ensure_ascii=False, indent=1)
