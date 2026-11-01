#!/usr/bin/env python3
"""依量測目標選抓法、觀察視點與整段動作核對（Codex 20261102_010259、010831；純運動學，給背景執行緒跑，工作量與逾時固定）。
plan_grasp(目標)：
  1 全窗（window_branch，原安全窗、車道 y 0）可行 ⇒ kind＝moving（原移動抓取）。
  2 否則停車搜尋：窗內 x（每 2 cm）× 所有種子分支（RS.SEEDS 與手腕翻轉）的相異預抓解，沿插入直線（預抓→抓取，1.2 cm 一步）與抬升直線（1 cm 一步）IK 延續
    （單步關節變化 ≤ MAX_STEP、餘裕 ≥ 0.05）；依名目餘裕排序逐一核「誤差包絡」：底盤 x ± EX、y ± EY、偏航 ± EYAW（27 組，角落先核、失敗即停）下同一條路徑都延續可行，
    找到 N_PASS 個通過即停，取包絡內最小餘裕最大者 ⇒ kind＝parked（停車策略，明示）。這是「本搜尋找到的子窗」內的點，不是最大可用窗口。
    （名目餘裕最大的點常靠近手腕奇異 j5≈0，包絡一偏就跨分支，所以不能只取前幾名。）
  3 都沒有 ⇒ unreachable；超過預算 ⇒ timeout（不放行）。
park_check(計畫, 目標, 實測底盤, 目前手臂)：停穩後在實測位姿重核（包絡內、預抓與插入／抬升 IK 延續、靠近直線與插入／抬升逐點淨空）。
view_search(實際底盤, 目前手臂)：觀察視點在實際停止後線上算（不查表）：相機光軸對準共同任務區（CD.LOOK_AT，兩瓶位中點），
  格點在底盤座標，IK 由目前手臂延續；約束＝餘裕、自碰、連桿對高度包絡（桌面 0.45–0.75 m）內每個桌高的桌面／桌腳與兩瓶位瓶身 ≥ 0.05 m，
  目前手臂→視點的關節直線（21 點）同樣滿足；分數＝包絡×瓶位案例中「相機高於桌面、支撐環 ≥ 50 %、瓶身 ≥ 80 % 在影像內」的數目，再取相機高。不讀桌高。
motion_check(計畫, 目標, …)：從觀察姿態起的整段動作逐點核（停車：回程→行駛→停車後關節直線靠近→插入＋抬升；移動：回程→漸變→窗兩端與中點的插入＋抬升）。
  停車抓取不在行駛中漸變：漸變路徑在 0.75 m 目標會擦過手腕奇異（j5≈0.12、j4／j6 反向轉約 1 rad），改成停穩後關節直線靠近（停在路徑上任一點都已核）。每點：關節餘裕、自碰 ≥ 8.39 mm、(i) 瓶心 0.10 m 內非手指連桿高於量測支撐面 ≥ 0.02 m、
  (ii) 全部連桿對真值第一桌（桌面＋桌腳）≥ 0.01 m（拒絕式安全網：只拒絕、不修正估計）、插入前全部連桿對量測瓶身 ≥ 0.01 m。
  移動抓取另在窗兩端與中點核插入／抬升。任何一項不過 ⇒ 不放行（why 指明哪一項）。
  關節直線段（回程、靠近、到視點）另要求 |j5| ≥ SING_MIN（不跨手腕奇異）；停車點搜尋先取與目前手臂同側的手腕分支。
  重選備妥時依序試 READY_OFFSETS（既有定義優先），取第一組全段通過者並記錄（0.75 m 目標的既有定義備妥自碰 7.26 mm，低於已驗證下限 8.39 mm）。
"""
import math
import time

import numpy as np

import bottle_branch as BB
import bottle_ref as BRF
import bottle_reach_scan as RS
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_geom as G
import bottle_seg3d_collect_design as CD
import bottle_seg3d_view_library as VL

PRE_M, LIFT_M, DX = 0.06, 0.05, 0.02
EX, EY, EYAW = 0.02, 0.01, 0.02
MAX_STEP, MIN_MARGIN = 0.40, 0.05
N_PASS, MAX_IK, MAX_S = 2, 6000, 60.0
ENV_COMBOS = sorted(((dx, dy, dw) for dx in (-EX, 0.0, EX) for dy in (-EY, 0.0, EY) for dw in (-EYAW, 0.0, EYAW)),
                    key=lambda d: -(abs(d[0]) / EX + abs(d[1]) / EY + abs(d[2]) / EYAW))   # 角落先核
LOCAL_R, CLEAR_SUP, CLEAR_TRUTH, CLEAR_BOTTLE, SELF_MIN = 0.10, 0.02, 0.01, 0.01, 0.00839
SING_MIN = 0.10      # 關節直線段（回程、靠近、到視點）上 |j5| 下限：跨過手腕奇異時 TCP 追蹤的 MPC 會滑到另一個手腕分支（離線 full 案例：j4／j6 差 1.9 rad、抬升撞限位）
READY_OFFSETS = ((0.14, 0.08), (0.10, 0.08), (0.14, 0.04), (0.10, 0.04))   # 備妥＝預抓後退、上抬（m）；第一組＝既有定義（ready_in_branch 預設）
VIEW_Z, VIEW_XY = (0.95, 0.90, 0.85), np.arange(0.05, 0.46, 0.10)


def poses(cx, cy, zs):
    Tg = C0.grasp_target(cx, cy, zs + 0.10)
    Tp = Tg.copy()
    Tp[:3, 3] -= PRE_M * Tg[:3, 2]
    return dict(pre=Tp, grasp=Tg, lift=BRF.lift_pose(Tg, LIFT_M))


class Budget:
    def __init__(self, max_ik=MAX_IK, max_s=MAX_S):
        self.n, self.t0, self.max_ik, self.max_s = 0, time.perf_counter(), max_ik, max_s

    def ik(self, base, T, seed, iters=200):
        self.n += 1
        if self.n > self.max_ik or time.perf_counter() - self.t0 > self.max_s:
            raise TimeoutError(f'budget n={self.n} t={time.perf_counter() - self.t0:.1f}s')
        return RS.ik(np.asarray(base, float), T, [np.asarray(seed, float)], iters=iters)


def _line(Ta, Tb, n):
    out = []
    for s in np.linspace(0.0, 1.0, n):
        T = Ta.copy()
        T[:3, 3] = Ta[:3, 3] + (Tb[:3, 3] - Ta[:3, 3]) * s
        out.append(T)
    return out


def grasp_path(P):
    """預抓→抓取（6 點）→抬升（6 點）；位姿同姿態，只有平移。"""
    return _line(P['pre'], P['grasp'], 6) + _line(P['grasp'], P['lift'], 6)[1:]


def _chain(bg, base, Ts, q0, iters=200):
    """沿一串位姿 IK 延續；回傳 (q 列表, 最小餘裕)；失敗回 (None, 失敗索引)。"""
    q, qs, mn = np.asarray(q0, float), [], math.inf
    for i, T in enumerate(Ts):
        ok, _, qn, pe, re, marg = bg.ik(base, T, q, iters)
        if not ok or marg < MIN_MARGIN or float(np.abs(qn - q).max()) > MAX_STEP:
            return None, i
        qs.append(qn)
        mn = min(mn, marg)
        q = qn
    return qs, mn


def _seeds():
    out = []
    for s in RS.SEEDS:
        out += [np.asarray(s, float), np.asarray(BB.wrist_flip(s), float)]
    return out


def plan_grasp(cx, cy, zs, win=(-0.35, 0.15), budget=None, q5_sign=None):
    P = poses(cx, cy, zs)
    t0 = time.perf_counter()
    try:
        wb = BB.window_branch(P['pre'], P['grasp'], P['lift'], win[0], win[1])
        return dict(kind='moving', margin=float(wb['margin']), q_mid=np.asarray(wb['q_mid']).tolist(), x_mid=float(wb['x_mid']), win=list(win), t_s=time.perf_counter() - t0)
    except ValueError as e:
        whole_err = str(e)
    bg = budget or Budget()
    path = grasp_path(P)
    cands, n_env = [], 0
    try:
        for x in np.round(np.arange(win[0], win[1] + 1e-9, DX), 4):
            base, seen = np.array([x, 0.0, 0.0]), []
            for s in _seeds():
                ok, _, qp, pe, re, marg = bg.ik(base, P['pre'], s)
                if not ok or marg < MIN_MARGIN or any(float(np.abs(qp - o).max()) < 1e-2 for o in seen):
                    continue
                seen.append(qp)
                qs, mn = _chain(bg, base, path, qp)
                if qs is not None:
                    cands.append((float(x), qp, mn))
        cands.sort(key=lambda c: (0 if q5_sign is None or np.sign(c[1][4]) == q5_sign else 1, -c[2]))   # 先取與目前手臂同側手腕分支（回程不跨奇異），再依名目餘裕
        best, n_pass = None, 0
        for x, qp, mn in cands:                                  # 依名目餘裕排序逐一核包絡（角落先核、失敗即停）；找到 N_PASS 個通過即停
            worst, okall = mn, True
            for d in ENV_COMBOS:
                n_env += 1
                qs, m2 = _chain(bg, [x + d[0], d[1], d[2]], path, qp)
                if qs is None:
                    okall = False
                    break
                worst = min(worst, m2)
            if okall:
                n_pass += 1
                same = q5_sign is None or np.sign(qp[4]) == q5_sign
                if best is None or (same, worst) > (best['same_side'], best['margin']):
                    best = dict(kind='parked', x_park=x, q_pre=qp.tolist(), margin=float(worst), err_env=dict(ex=EX, ey=EY, eyaw=EYAW), same_side=bool(same))
                if n_pass >= N_PASS:
                    break
    except TimeoutError as e:
        return dict(kind='timeout', why=str(e), whole_window=whole_err, n_ik=bg.n, t_s=time.perf_counter() - t0)
    info = dict(whole_window=whole_err, n_cands=len(cands), cand_x=sorted({c[0] for c in cands}), n_env_checked=n_env, n_ik=bg.n, t_s=time.perf_counter() - t0)
    if best is None:
        return dict(kind='unreachable', **info)
    best.update(info)
    return best


# ---- 觀察視點（線上）
def view_search(base, q_arm, T_mount, T_eef_tcp, budget=None):
    bg = budget or Budget(max_ik=400, max_s=40.0)
    base, q_arm = np.asarray(base, float), np.asarray(q_arm, float)
    cb, sb = math.cos(base[2]), math.sin(base[2])
    best, n_try = None, 0
    try:
        for zr in VIEW_Z:
            for xr in VIEW_XY:
                for yr in VIEW_XY:
                    cw = np.array([base[0] + cb * xr - sb * yr, base[1] + sb * xr + cb * yr, zr])
                    T_tcp = CD.look_T(cw, CD.LOOK_AT) @ np.linalg.inv(T_mount) @ T_eef_tcp
                    n_try += 1
                    ok, _, qa, pe, re, marg = bg.ik(base, T_tcp, q_arm, iters=150)
                    if not ok or marg < CD.JOINT_MARGIN or not VL.ok_clear(VL.clear_env(np.r_[base, qa])):
                        continue
                    Tc = CD.cam_T(np.r_[base, qa], T_mount)
                    vis = [CD.visibility(Tc, xy, tz) for tz in VL.ENVELOPE for xy in CD.BOTTLES]
                    sc = (sum(v['cam_above_table'] and v['support_ring_in_view'] >= 0.5 and v['body_in_view'] >= 0.8 for v in vis), round(float(Tc[2, 3]), 3))
                    if best is not None and sc <= best['score']:
                        continue
                    ok_p, worst = VL.path_ok(q_arm, qa, base)
                    if ok_p and min(abs(q_arm[4] + (qa[4] - q_arm[4]) * s_) for s_ in np.linspace(0.0, 1.0, 21)) >= SING_MIN:   # 到視點的直線不跨手腕奇異
                        best = dict(score=sc, q=qa.tolist(), cam=Tc[:3, 3].tolist(), pitch_deg=math.degrees(math.asin(max(-1.0, min(1.0, -Tc[2, 2])))), path_worst=worst,
                                    n_cases=len(vis))
            if best is not None and best['score'][0] == len(VL.ENVELOPE) * len(CD.BOTTLES):
                break                                            # 由高往低：這層已全案例可見就不再往下
    except TimeoutError as e:
        return dict(ok=False, why='timeout', error=str(e), n_try=n_try)
    if best is None or best['score'][0] == 0:
        return dict(ok=False, why='no_view', n_try=n_try, best=best)
    dq = float(np.abs(np.asarray(best['q']) - q_arm).max())
    return dict(ok=True, n_try=n_try, T_tr=max(1.0, math.ceil(1.875 * dq / B.W_ARM['v_arm'] / 0.05) * 0.05), dq_max=dq, **best)


# ---- 整段動作核對
def _cfg_check(q9, cx, cy, zs, truth_boxes, bottle=True, local=True):
    W = G.link_world(q9, G.Q_OPEN)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_all = np.vstack(list(W.values()))
    out = dict(truth=min(float(G.box_dist(P_all, c_, s_).min()) for c_, s_ in truth_boxes), self=float(G.C.self_clear(q9, G.Q_OPEN)[0]),
               margin=BB.margin(q9[3:]), local=math.inf, bottle=math.inf, sing=abs(float(q9[7])))
    if local:
        near = np.hypot(P_nf[:, 0] - cx, P_nf[:, 1] - cy) <= LOCAL_R
        if near.any():
            out['local'] = float((P_nf[near, 2] - zs).min())
    if bottle:
        out['bottle'] = float(G.cyl_dist(P_all, cx, cy, zs, zs + G.BOTTLE['h'], G.BOTTLE['r']).min())
    return out


def _bad(w, sing=False):
    for k, lim in (('truth', CLEAR_TRUTH), ('self', SELF_MIN), ('margin', MIN_MARGIN), ('local', CLEAR_SUP), ('bottle', CLEAR_BOTTLE)) + ((('sing', SING_MIN),) if sing else ()):
        if w[k] < lim:
            return k
    return None


def _acc(worst, w, nm=None):
    for k in worst:
        if k in w:
            worst[k] = min(worst[k], w[k])
    if nm is not None:
        worst.setdefault('sing_by_seg', {})
        worst['sing_by_seg'][nm] = min(worst['sing_by_seg'].get(nm, math.inf), w['sing'])


def _jline(base, qa, qb, n=21):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    return [np.r_[base, qa + (qb - qa) * s] for s in np.linspace(0.0, 1.0, n)]


def motion_check(plan, cx, cy, zs, truth_boxes, x_pre, blend_len, base_obs=None, q_from=None, ready=None, budget=None):
    """整段核對。ready 未給時重選備妥：依序試 READY_OFFSETS（第一組＝既有定義 後退 0.14、上抬 0.08），取第一組全段通過者；全部不過回第一組的失敗原因。"""
    bg = budget or Budget(max_ik=1500, max_s=60.0)
    offs = [(None, None)] if ready is not None else READY_OFFSETS
    tried = []
    for back, up in offs:
        r = _motion_one(plan, cx, cy, zs, truth_boxes, x_pre, blend_len, base_obs, q_from, ready, bg, back, up)
        tried.append(dict(offset=None if back is None else [back, up], ok=r['ok'], why=r['why'], seg=r.get('seg')))
        if r['ok'] or str(r['why']).startswith('motion_'):
            r['tried'] = tried
            return r
    first = _motion_one(plan, cx, cy, zs, truth_boxes, x_pre, blend_len, base_obs, q_from, ready, bg, *offs[0])
    first['tried'] = tried
    return first


def _motion_one(plan, cx, cy, zs, truth_boxes, x_pre, blend_len, base_obs, q_from, ready, bg, back, up):
    """回傳 dict(ok, why, seg, worst, q_ready, T_ready_rel, q_pre, x_grasp)。
    停車：回程（q_from→新備妥，底盤在 base_obs）→ 行駛（手臂固定備妥，底盤到 x_park）→ 靠近（停車後備妥→預抓關節直線）→ 插入＋抬升。
    移動：回程 → 漸變（底盤 max(x_pre−blend_len, 觀察位置)→x_pre，TCP 由備妥 min-jerk 到預抓，21 點 IK 延續）→ 窗兩端與中點的插入＋抬升。
    ready＝(q_ready, T_ready_rel) 時沿用給定備妥（接近途中採用、維持名目分支）；否則以 ready_in_branch 在計畫分支重選。"""
    P = poses(cx, cy, zs)
    worst = dict(truth=math.inf, self=math.inf, margin=math.inf, local=math.inf, bottle=math.inf, sing=math.inf)
    n = 0
    try:
        parked = plan['kind'] == 'parked'
        if parked:
            x_g, q_pre0, x_mid = plan['x_park'], np.asarray(plan['q_pre'], float), plan['x_park']
        else:
            x_g, q_pre0, x_mid = float(x_pre), np.asarray(plan['q_mid'], float), plan['x_mid']
        base_g = np.array([x_g, 0.0, 0.0])
        if ready is None:
            q_ready, T_ready_rel, q_pre = BB.ready_in_branch(base_g, P['pre'], q_pre0, x_mid, back=back, up=up)
        else:
            q_ready, T_ready_rel = np.asarray(ready[0], float), np.asarray(ready[1], float)
            r_ = BB.sweep(P['pre'], np.linspace(x_mid, x_g, max(int(abs(x_g - x_mid) / 0.02) + 1, 2)), q_pre0)[0]
            if r_ is None:
                return dict(ok=False, why='pre_ik', seg='pre', worst=worst)
            q_pre = r_[-1]
        segs = []
        if q_from is not None:                                   # 回程：q_from→備妥，底盤停在觀察位姿
            segs.append(('return', _jline(base_obs, q_from, q_ready)))
        if parked:
            if base_obs is not None:                             # 行駛：手臂固定備妥
                xs = np.arange(base_obs[0], x_g + 1e-9, 0.05).tolist() + [x_g]
                segs.append(('drive', [np.r_[x, base_obs[1] * (x_g - x) / max(x_g - base_obs[0], 1e-6), 0.0, q_ready] for x in xs]))
            segs.append(('approach', _jline(base_g, q_ready, q_pre)))   # 停車後手臂直線靠近：停在路徑上任一點都已核
        else:
            x_b0 = max(x_g - blend_len, base_obs[0]) if base_obs is not None else x_g - blend_len
            bl, q = [], q_ready
            for s in np.linspace(0.0, 1.0, 21):
                b = np.array([x_b0 + (x_g - x_b0) * s, 0.0, 0.0])
                T = BRF.interp_pose(BRF.pose_world(b, T_ready_rel), P['pre'], BRF.min_jerk(s))
                ok, _, qn, pe, re, marg = bg.ik(b, T, q)
                if not ok or marg < MIN_MARGIN or float(np.abs(qn - q).max()) > MAX_STEP:
                    return dict(ok=False, why='blend_ik', seg='blend', s=float(s), worst=worst)
                bl.append(np.r_[b, qn])
                q = qn
            segs.append(('blend', bl))
        xs_g = [x_g] if parked else [plan['win'][0], plan['x_mid'], plan['win'][1]]
        for xg in xs_g:                                          # 插入＋抬升（移動抓取：窗兩端與中點）
            b = np.array([xg, 0.0, 0.0])
            ok, _, q0, *_r = bg.ik(b, P['pre'], q_pre)
            qs, _m = _chain(bg, b, grasp_path(P), q0) if ok else (None, 0)
            if qs is None:
                return dict(ok=False, why='grasp_path_ik', seg='grasp', x=float(xg), worst=worst)
            segs.append((f'grasp@{xg:+.3f}', [np.r_[b, qa] for qa in qs]))
        for nm, Q in segs:
            for q9 in Q:
                pre_insert = not nm.startswith('grasp')
                w = _cfg_check(q9, cx, cy, zs, truth_boxes, bottle=pre_insert)
                if not pre_insert:
                    w['bottle'] = math.inf                       # 插入後手指本來就包住瓶身；非手指↔瓶由線上淨空核對
                n += 1
                _acc(worst, w, nm)
                k = _bad(w, sing=nm in ('return', 'approach'))      # 關節直線段不跨手腕奇異；漸變與插入／抬升是 IK 延續路徑，只記錄
                if k is not None:
                    return dict(ok=False, why=('truth_net' if k == 'truth' else k), seg=nm, worst=worst, n_cfg=n)
    except (TimeoutError, ValueError) as e:
        return dict(ok=False, why='motion_' + type(e).__name__, error=str(e), worst=worst, n_cfg=n)
    return dict(ok=True, why=None, worst=worst, n_cfg=n, q_ready=np.asarray(q_ready).tolist(), T_ready_rel=np.asarray(T_ready_rel).tolist(), q_pre=np.asarray(q_pre).tolist(),
                x_grasp=float(x_g), segs=[nm for nm, _ in segs], ready_offset=None if ready is not None else [back, up])


def park_check(plan, cx, cy, zs, truth_boxes, base, q_arm, budget=None):
    """停車後在實測底盤位姿重核：誤差在包絡內、預抓 IK（由計畫分支延續）、插入＋抬升 IK 延續、目前手臂→預抓關節直線與插入／抬升的逐點淨空。"""
    bg = budget or Budget(max_ik=200, max_s=20.0)
    P = poses(cx, cy, zs)
    base = np.asarray(base, float)
    err = dict(dx=float(base[0] - plan['x_park']), dy=float(base[1]), dyaw=float(base[2]))
    if abs(err['dx']) > EX or abs(err['dy']) > EY or abs(err['dyaw']) > EYAW:
        return dict(ok=False, why='park_out_of_envelope', err=err)
    worst = dict(truth=math.inf, self=math.inf, margin=math.inf, local=math.inf, bottle=math.inf, sing=math.inf)
    try:
        ok, _, q_pre, *_r = bg.ik(base, P['pre'], plan['q_pre'])
        qs, _m = _chain(bg, base, grasp_path(P), q_pre) if ok and float(np.abs(q_pre - np.asarray(plan['q_pre'])).max()) <= MAX_STEP else (None, 0)
        if qs is None:
            return dict(ok=False, why='park_ik', err=err)
        for nm, Q in (('approach', _jline(base, q_arm, q_pre)), ('grasp', [np.r_[base, qa] for qa in qs])):
            for q9 in Q:
                w = _cfg_check(q9, cx, cy, zs, truth_boxes, bottle=nm == 'approach')
                if nm != 'approach':
                    w['bottle'] = math.inf
                _acc(worst, w, nm)
                k = _bad(w, sing=nm == 'approach')
                if k is not None:
                    return dict(ok=False, why='park_' + ('truth_net' if k == 'truth' else k), seg=nm, err=err, worst=worst)
    except TimeoutError as e:
        return dict(ok=False, why='park_timeout', error=str(e), err=err)
    dq = float(np.abs(q_pre - np.asarray(q_arm, float)).max())
    return dict(ok=True, why=None, err=err, worst=worst, q_pre=q_pre.tolist(), dq_max=dq, T_tr=max(1.0, math.ceil(1.875 * dq / B.W_ARM['v_arm'] / 0.05) * 0.05))
