#!/usr/bin/env python3
"""主線 B 無接觸取像趟的監督節點 v2（Codex 20261117_044341／044741：操作端給「可接任務的狀態集合」，導航自己選點、規劃與局部收斂；操作依實際運動在線算手臂）。
與 v1（b_capture_node.py）的差別：
  - 導航：遠端目標照原鏈 /goal_pose → goal_to_plan_relay → Smac；不再合成門前線當導航路徑、也不再串進線點。
  - 操作端可行狀態集合（規劃程序，當次估計）：抓持窗口逐格核（門旁分支預抓位＋沿門窗口：σ、限位、淨空、Shield）——離門距離 SB_CAND × 接手點 AW_CAND，
    取可行的接手點與最寬的連續離門距離區間 ⇒ 走廊（門前線段 a_start→a_end、中心 sb_c、半寬 w）、朝向（面門）、沿門速度範圍。
    交給 gmpc_task_node 區域模式（導航選進入點、Smac 到進入點＋走廊參考、局部收斂）。
  - 觀察族與初始過渡沿走廊中心線算（BPlanner.plan，require_cover=False）——只作候選與觀察參數；真正的過渡在觀察段尾端線上重算：
    底盤預測＝gmpc_task_node 發布的 GMPC 1 s 預測，之後沿走廊依速度框外推（只用來搜尋候選，不當執行保證）；
    自「未來起點 t_s」的參考狀態（注視 IK 在預測底盤的解與其變化率；過渡中則為現行五次多項式在 t_s 的狀態）以五次多項式到接手預抓位，
    接手點可重選（AW_CAND）；篩選 σ、淨空、限位、關節速率／加速度、停止前推（篩選模型）。結果到時須 t_s 未過、版本相符才採用。
    過渡中每 0.5 s 重算；每拍仍走完整停止核對（實際設定點、在途管線、偏置）——commit／stop_now／abort。
  - 正常終點：過渡完成（預抓位）且底盤過接手點後速度框 0，整機停穩 0.3 s ⇒ /far/done。是否「在可接任務區內」照實記錄，不據以放行抓持（本趟不抓持）。
    python3 b_capture_node2.py OUT_PREFIX"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT, PHYS, NPH, D = 0.05, 0.01, 5, 8
V_APPROACH = float(os.environ.get('B_V_APPROACH', '0.20'))
V_ARM, A_ARM = 1.0, 10.0
LEAD_S = D * PHYS + DT + 0.10
EST = dict(n=3, spread_m=0.005, max_age_s=0.5, replan_shift_m=0.010)
ASSOC_TOL = dict(dxy_m=0.15, dz_m=0.10, dang_deg=10.0)
GOAL_A, SB_PRIOR = -0.49, 0.45
STILL = dict(base=0.003, yaw=0.003, arm=0.002, hold_s=0.3)
SB_CAND = [round(0.43 + 0.02 * k, 2) for k in range(5)]               # 0.43–0.51，0.02 一格
AW_CAND = (-0.08, -0.10, -0.12)
RETR_LEAD, RETR_PERIOD, RETR_TRIGGER = 0.8, 0.5, 0.10                 # 過渡起點在請求後 0.8 s（模擬）；過渡中每 0.5 s 重算；觀察段在預測 a ≤ 初始 a_t0＋0.10 起請求
REGION_A_END_MARGIN = 0.25
ENTER_MARGIN = 0.10
LOOK_REACH_M = 0.65                                                   # 注視 IK 前的可達預檢：TCP 目標離底盤原點水平距離上限（臂長＋掛載的寬鬆上界）


def _T3(p):
    import numpy as np
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


def quintic(q0, v0, a0, q1, T):
    """五次多項式係數（各關節）：q(0)=q0, q'(0)=v0, q''(0)=a0, q(T)=q1, q'(T)=0, q''(T)=0。"""
    import numpy as np
    q0, v0, a0, q1 = (np.asarray(x, float) for x in (q0, v0, a0, q1))
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    c3 = (20 * (q1 - q0) - (12 * v0) * T - (3 * a0) * T2) / (2 * T3)
    c4 = (30 * (q0 - q1) + (16 * v0) * T + (3 * a0) * T2) / (2 * T4)
    c5 = (12 * (q1 - q0) - (6 * v0) * T - a0 * T2) / (2 * T5)
    return [q0, v0, a0 / 2, c3, c4, c5]


def qeval(C, t, T):
    import numpy as np
    t = min(max(t, 0.0), T)
    q = sum(C[k] * t ** k for k in range(6))
    v = sum(k * C[k] * t ** (k - 1) for k in range(1, 6))
    a = sum(k * (k - 1) * C[k] * t ** (k - 2) for k in range(2, 6))
    return np.asarray(q, float), np.asarray(v, float), np.asarray(a, float)


def worker(conn):
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'motm_b'))
    import numpy as np
    import door_motm_grasp_ctrl as G, door_wbopt as W, door_fast_kin as DFK, door_b_plan as BP
    import b_stopcheck as SC
    import door_sweep_pool as SWP
    SC.SWEEP_S = float(os.environ.get('B_SWEEP_S', '0.040'))
    Rg, T_HG, a_H = G.PG.baseline_reference()
    LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    AL = np.asarray(ident['alpha'], float)
    Kc = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])
    SWP._die_with_parent()
    pool = SWP.SweepPool(int(os.environ.get('B_POOL_N_PLANNER') or os.environ.get('B_POOL_N', '2')))
    K = G.K
    grav = DFK.FastGravity3(K, G.BM.arm_bodies(K))
    KP_ARM = 600.0
    st = dict(T_est=None, off=None, plan=None, bp=None, hist=[], prep=None, qtr=None, q_des_prev=None, n_clip=0)

    def off_fn(q9):
        o = st['off']
        if o is None:
            return np.zeros(6)
        return o['off0'] + (np.asarray(grav(np.asarray(q9, float)), float) - o['G0']) / KP_ARM
    stop_check, to_map = SC.make(pool, lambda: st['T_est'], off_fn, AL, LO, HI)

    def door_frame(T):
        n = (T[:3, :3] @ np.asarray(a_H, float)).copy(); n[2] = 0; n /= np.linalg.norm(n)
        ax = T[:3, 0].copy(); ax[2] = 0; d = -ax / np.linalg.norm(ax)
        return d[:2], n[:2], math.atan2(-n[1], -n[0])

    def minjerk(u):
        u = min(max(u, 0.0), 1.0)
        return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5

    def make_bp(T_Hm, d_, n_, yaw_, pl=None):
        bp = BP.BPlanner(np.asarray(T_Hm, float), np.asarray(d_, float), np.asarray(n_, float), float(yaw_), Kc)
        bp._alpha = AL
        if pl is not None:
            q9 = np.r_[bp.base(float(pl['a_w']), float(pl['door_line_sb'])), np.asarray(pl['q_pre'], float)]
            bp.T_TC = np.linalg.inv(K.fk(q9, 'link_tcp')) @ K.fk(q9, BP.W0.OPT)
        return bp

    def look_ref(bp, fam, b, seed):
        A = np.array([f['a'] for f in fam]); Q = np.array([f['q'] for f in fam])
        a_, _ = bp.to_door(b[:2])
        a_c = float(np.clip(a_, A[0], A[-1]))
        i = int(np.clip(np.searchsorted(A, a_c), 1, len(A) - 1)); w = (a_c - A[i - 1]) / max(A[i] - A[i - 1], 1e-9)
        par = [fam[i - 1]['par'][k] + w * (fam[i]['par'][k] - fam[i - 1]['par'][k]) for k in range(3)]
        fseed = Q[i - 1] + w * (Q[i] - Q[i - 1])
        Tt = bp.look_at(np.asarray(b, float), *par)
        if float(np.linalg.norm(Tt[:2, 3] - np.asarray(b, float)[:2])) > LOOK_REACH_M:   # 目標超出手臂可達（離門太遠）：不解 IK（失敗迭代耗時），沿用觀察族構型
            return fseed, 'out_of_reach_family', par
        qi, sm = bp.solve(np.asarray(b, float), Tt, fseed if seed is None else seed)
        if qi is None or sm < W.SIG_FLOOR or float(np.abs(qi - fseed).max()) > 0.3:
            return fseed, 'fallback_family', par
        return qi, 'look_at', par

    conn.send(('ready', 0, dict(sweep_s=SC.SWEEP_S)))
    while True:
        m = conn.recv()
        if m is None:
            pool.close()
            break
        op, a = m
        ver = a.get('ver', 0)
        t_rx = time.perf_counter()
        try:
            if op == 'prior':
                import hashlib
                import door_asset_v3 as DA3, door_geom_v3 as DG
                sreg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
                ap_ = os.path.join(os.path.dirname(HERE), sreg['door']['asset'])
                dd = DA3.load(ap_)
                pl = dd['placement']['pose']
                T_prior = _T3([float(pl['x']), float(pl['y']), math.radians(float(pl['yaw_deg']))]) @ np.asarray(DG.handle_pose(dd, 0.0, dd['handle']['height']), float)
                st['T_prior'] = T_prior
                d_, n_, yaw_ = door_frame(T_prior)
                p0 = T_prior[:2, 3] + SB_PRIOR * n_
                conn.send(('prior', ver, dict(goal_xy=(p0 - GOAL_A * d_).tolist(), yaw=yaw_, d=d_.tolist(), n=n_.tolist(), handle_xy=T_prior[:2, 3].tolist(), T_prior=T_prior.tolist(),
                                              source=dict(asset=sreg['door']['asset'], sha256=hashlib.sha256(open(ap_, 'rb').read()).hexdigest(), note='名目位姿（門關閉）；不讀把手 prim 真值'))))
            elif op == 'obs':
                To = np.asarray(a['T_odom_handle'], float)
                h = [x for x in st['hist'] if float(a['t']) - x[1] <= EST['max_age_s']] + [(int(a['seq']), float(a['t_obs']), To)]
                st['hist'] = h[-EST['n']:]
                out = dict(n_hist=len(st['hist']))
                if len(st['hist']) >= EST['n']:
                    P = np.array([x[2][:3, 3] for x in st['hist']])
                    spread = float(max(np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(len(P))))
                    out['spread_mm'] = round(spread * 1e3, 2)
                    if spread <= EST['spread_m']:
                        Uq, _, Vq = np.linalg.svd(np.sum([x[2][:3, :3] for x in st['hist']], axis=0))
                        T_new = np.eye(4); T_new[:3, :3] = Uq @ np.diag([1.0, 1.0, np.linalg.det(Uq @ Vq)]) @ Vq; T_new[:3, 3] = np.median(P, axis=0)
                        Tm = _T3(np.asarray(a['T_mo'], float)) @ T_new
                        dR = st['T_prior'][:3, :3].T @ Tm[:3, :3]
                        assoc = dict(dxy_m=round(float(np.linalg.norm(Tm[:2, 3] - st['T_prior'][:2, 3])), 4), dz_m=round(float(abs(Tm[2, 3] - st['T_prior'][2, 3])), 4),
                                     dang_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), 3))
                        ok_as = assoc['dxy_m'] <= ASSOC_TOL['dxy_m'] and assoc['dz_m'] <= ASSOC_TOL['dz_m'] and assoc['dang_deg'] <= ASSOC_TOL['dang_deg']
                        out.update(assoc=assoc, assoc_ok=ok_as, seqs=[x[0] for x in st['hist']], t_obs=float(a['t_obs']))
                        if ok_as and (st.get('T_cand') is None or (a.get('allow_update') and float(np.linalg.norm(T_new[:3, 3] - st['T_cand'][:3, 3])) > EST['replan_shift_m'])):
                            st['T_cand'] = T_new
                            out['estimate_set'] = True
                            out['T_est'] = T_new.tolist()
                conn.send(('obs', ver, out))
            elif op == 'region_plan':                                  # 規劃程序：可接任務狀態集合（走廊）＋沿走廊的觀察族與初始候選過渡
                t0 = time.perf_counter()
                T_mo = np.asarray(a['T_mo'], float)
                T_Hm = _T3(T_mo) @ np.asarray(a['T_est'], float)
                d_, n_, yaw_ = door_frame(T_Hm)
                bp = make_bp(T_Hm, d_, n_, yaw_)
                feas = {}
                for aw in AW_CAND:
                    for sb in SB_CAND:
                        qp = bp.pregrasp(aw, sb)
                        ok_ = qp is not None and bp.window_ok(aw, sb, qp)
                        feas[(aw, sb)] = bool(ok_)
                best = None
                for aw in AW_CAND:                                     # 每個接手點取最寬的連續可行離門距離區間
                    run = []
                    for sb in SB_CAND + [None]:
                        if sb is not None and feas[(aw, sb)]:
                            run.append(sb); continue
                        if run and (best is None or (run[-1] - run[0]) > (best[2] - best[1]) + 1e-9):
                            best = (aw, run[0], run[-1])
                        run = []
                out = dict(feas={'%g,%g' % k: v for k, v in feas.items()}, wall_feas_s=round(time.perf_counter() - t0, 3), T_Hm=T_Hm.tolist(), d=d_.tolist(), n=n_.tolist(), yaw=yaw_, T_est=a['T_est'])
                if best is None:
                    out.update(ok=False, why='no_feasible_window_cell')
                    conn.send(('region_plan', ver, out)); continue
                aw_r, sb_lo, sb_hi = best
                sb_c = round(0.5 * (sb_lo + sb_hi), 4); w = round(max(0.5 * (sb_hi - sb_lo), 0.005), 4)
                line = [T_Hm[:2, 3] + sb_c * n_ - aa * d_ for aa in np.arange(0.70, -0.551, -0.02)]
                pl = bp.plan(np.array([np.asarray(a['robot_xy'], float)] + line), np.asarray(a['q_arm'], float), a_w_cands=(aw_r,), require_cover=False)
                pl['path_source'] = 'manip_region_centerline_sb%.3f' % sb_c
                keep = {k: v for k, v in pl.items() if k not in ('path_door',)}
                if pl.get('why') is not None:
                    out.update(ok=False, why='centerline_plan:' + str(pl['why']), plan=keep)
                    conn.send(('region_plan', ver, out)); continue
                a_start = max(f['a'] for f in pl['obs']) + 0.05
                a_end = float(aw_r) - REGION_A_END_MARGIN
                a_enter = float(pl['a_t0']) + ENTER_MARGIN               # 最晚進入點：初始候選過渡起點之前 ENTER_MARGIN（給導航選進入點；線上重算仍可另選）
                region = dict(sb_c=sb_c, half_width=w, sb_feasible=[sb_lo, sb_hi], aw_feasible=aw_r, a_start=round(a_start, 3), a_end=round(a_end, 3), a_enter_by=round(a_enter, 3), enter_by=round(a_start - a_enter, 3),
                              line=[(T_Hm[:2, 3] + sb_c * n_ - a_start * d_).tolist(), (T_Hm[:2, 3] + sb_c * n_ - a_end * d_).tolist()], heading=yaw_)
                st['bp_p'], st['pl_p'], st['region_p'] = make_bp(T_Hm, d_, n_, yaw_, pl), pl, region
                out.update(ok=True, plan=keep, region=region, wall_s=round(time.perf_counter() - t0, 3))
                conn.send(('region_plan', ver, out))
            elif op == 'retr':                                         # 規劃程序：自未來起點 t_s 的參考狀態線上重算過渡（接手點可重選）
                t0 = time.perf_counter()
                bp, pl, rg = st['bp_p'], st['pl_p'], st['region_p']
                fam = sorted(pl['obs'], key=lambda f: f['a'])
                BPm = np.asarray(a['base_pred'], float)                # 每列 t, x, y, yaw（地圖）；t 自 t_s 起算
                if a.get('qtr') is not None:                           # 過渡中：起點狀態＝現行五次多項式在 t_s 的狀態
                    qt = a['qtr']; C = [np.asarray(c, float) for c in qt['C']]
                    q_s, v_s, ac_s = qeval(C, float(a['t_s']) - float(qt['t_s']), float(qt['T']))
                    src = 'current_quintic'
                else:                                                  # 觀察段：注視 IK 在預測底盤的解（t_s 與 t_s＋0.05）
                    b0 = BPm[0, 1:4]; b1 = BPm[min(1, len(BPm) - 1), 1:4]
                    q_s, how0, _ = look_ref(bp, fam, b0, np.asarray(a['q_seed'], float))
                    q_s1, _, _ = look_ref(bp, fam, b1, q_s)
                    dtp = max(float(BPm[min(1, len(BPm) - 1), 0] - BPm[0, 0]), 1e-3)
                    v_s, ac_s = (q_s1 - q_s) / dtp, np.zeros(6)
                    src = 'look_at_' + how0
                AD = np.array([bp.to_door(r[1:3]) for r in BPm])        # (a, sb)
                cands, chosen = [], None
                for aw in AW_CAND:
                    rec = dict(aw=aw)
                    idx = np.where(AD[:, 0] <= aw)[0]
                    if len(idx) == 0:
                        rec['why'] = 'not_reached_in_prediction'; cands.append(rec); continue
                    k = int(idx[0]); T = float(BPm[k, 0]); sb_w = float(AD[k, 1])
                    rec.update(T=round(T, 3), sb_w=round(sb_w, 4))
                    if not (rg['sb_feasible'][0] - 1e-9 <= sb_w <= rg['sb_feasible'][1] + 1e-9):
                        rec['why'] = 'sb_outside_feasible_region'; cands.append(rec); continue
                    qp = bp.pregrasp(aw, sb_w)
                    if qp is None or not bp.window_ok(aw, sb_w, qp):
                        rec['why'] = 'window_infeasible_at_sb'; cands.append(rec); continue
                    if T < 1.0:
                        rec['why'] = 'too_short'; cands.append(rec); continue
                    C = quintic(q_s, v_s, ac_s, qp, T)
                    bad, t_geom, vmax, amax = None, None, 0.0, 0.0
                    for j, tt in enumerate(np.linspace(0.0, T, 41)):
                        qq, vv, aa_ = qeval(C, tt, T)
                        vmax, amax = max(vmax, float(np.abs(vv).max())), max(amax, float(np.abs(aa_).max()))
                        kb = int(np.clip(np.searchsorted(BPm[:, 0], tt), 0, len(BPm) - 1)); b = BPm[kb, 1:4]
                        if np.any(qq < LO) or np.any(qq > HI):
                            bad = 'limit@%.2f' % tt; break
                        okc, _, cl = bp.ok_cfg(b, qq)
                        if not okc:
                            bad = 'clear@%.2f' % tt; break
                        if bp.sigma6(np.r_[b, qq]) < W.SIG_FLOOR:
                            bad = 'sigma@%.2f' % tt; break
                        if bp.frame(np.r_[b, qq]) is not None:
                            t_geom = tt
                        if j % 4 == 0 and j > 0:                       # 停止前推（篩選模型：穩態管線、設定點＝關節）
                            kb1 = min(kb + 1, len(BPm) - 1)
                            vb_w = (BPm[kb1, 1:3] - BPm[kb, 1:3]) / max(BPm[kb1, 0] - BPm[kb, 0], 1e-3)
                            c_, s_ = math.cos(b[2]), math.sin(b[2])
                            ub = np.array([c_ * vb_w[0] + s_ * vb_w[1], -s_ * vb_w[0] + c_ * vb_w[1], 0.0])
                            okst, _, _ = bp.stop_sweep(b, qq, ub, vv, AL, every=4, max_steps=300)
                            if not okst:
                                bad = 'stop_sweep@%.2f' % tt; break
                    if bad is None and (vmax > V_ARM or amax > A_ARM):
                        bad = 'rate %.2f acc %.2f' % (vmax, amax)
                    rec.update(vmax=round(vmax, 3), amax=round(amax, 3), t_last_geom=None if t_geom is None else round(t_geom, 3))
                    if bad:
                        rec['why'] = bad; cands.append(rec); continue
                    rec['ok'] = True; cands.append(rec)
                    chosen = dict(aw=aw, sb_w=sb_w, T=T, C=[c.tolist() for c in C], q_pre=qp.tolist(), t_last_geom=t_geom)
                    break
                conn.send(('retr', ver, dict(ok=chosen is not None, chosen=chosen, cands=cands, src=src, t_s=float(a['t_s']), wall_s=round(time.perf_counter() - t0, 3))))
            elif op == 'adopt':                                        # 工作程序：採用區域規劃（觀察族、停止核對用估計）
                pl = a['plan']
                st['bp'], st['plan'], st['T_est'] = make_bp(a['T_Hm'], a['d'], a['n'], a['yaw'], pl), pl, np.asarray(a['T_est'], float)
                st['fam'] = sorted(pl['obs'], key=lambda f: f['a'])
                st['prep'], st['qtr'], st['qtr_next'] = None, None, None
                conn.send(('adopt', ver, dict(ok=True)))
            elif op == 'adopt_tr':                                     # 工作程序：採用（或替換）五次多項式過渡；只接受尚未開始的起點
                if float(a['t_now']) > float(a['t_s']) - 1e-6 and st['qtr'] is None:
                    conn.send(('adopt_tr', ver, dict(ok=False, why='t_s_passed'))); continue
                new = dict(C=[np.asarray(c, float) for c in a['C']], t_s=float(a['t_s']), T=float(a['T']), q_pre=np.asarray(a['q_pre'], float), aw=float(a['aw']))
                if st['qtr'] is None:
                    st['qtr'] = new
                else:
                    st['qtr_next'] = new                               # 替換段：到它的起點才切換（之前照現行段）
                conn.send(('adopt_tr', ver, dict(ok=True)))
            elif op == 'off_anchor':
                q_a, s_a = np.asarray(a['q'], float), np.asarray(a['s'], float)
                st['off'] = dict(off0=q_a[3:] - s_a, G0=np.asarray(grav(q_a), float))
                conn.send(('off_anchor', ver, dict(off0=[float(x) for x in st['off']['off0']])))
            elif op == 'tick':
                q, pipe, rel, s, s_end = np.asarray(a['q'], float), [np.asarray(u, float) for u in a['pipe']], np.asarray(a['rel'], float), np.asarray(a['s'], float), np.asarray(a['s_end'], float)
                T_mo = np.asarray(a['T_mo'], float)
                mode, t = a['mode'], float(a['t'])
                out = dict(mode=mode)
                ua = np.zeros(6)
                if st['T_est'] is not None and st['bp'] is not None and mode != 'NAV':
                    bp, pl = st['bp'], st['plan']
                    qm = to_map(q, T_mo)
                    am, sbm = bp.to_door(qm[:2])
                    vb = np.asarray(a.get('v_world', [0.0, 0.0]), float)
                    b_pred = np.r_[qm[:2] + vb * LEAD_S, qm[2] + float(a.get('wz', 0.0)) * LEAD_S]   # 預測底盤：實測位姿＋實測平移與偏航速度（含離門方向）
                    a_pred, _ = bp.to_door(b_pred[:2])
                    out.update(a=round(am, 4), sb=round(sbm, 4), a_pred=round(a_pred, 4))
                    fam = st['fam']
                    q_des, info = None, {}
                    if st.get('qtr_next') is not None and t + LEAD_S >= st['qtr_next']['t_s']:
                        st['qtr'], st['qtr_next'] = st['qtr_next'], None
                    qtr = st['qtr']
                    tau = None if qtr is None else (t + LEAD_S - qtr['t_s'])
                    if mode == 'PREP':
                        A = np.array([f['a'] for f in fam]); Q = np.array([f['q'] for f in fam])
                        pr = st['prep']
                        if pr is None:
                            a_o = float(np.clip(a_pred, A[0], A[-1]))
                            q_o = np.array([np.interp(a_o, A, Q[:, j]) for j in range(6)])
                            dq = q_o - (s + off_fn(q))
                            T_p = max(float(np.abs(dq).max()) * 1.875 / (0.8 * V_ARM), float(np.sqrt(float(np.abs(dq).max()) * 5.7735 / 10.0)), 1.0)
                            pr = st['prep'] = dict(t0=t, T=T_p, q0=(s + off_fn(q)).copy(), q1=q_o, a_o=a_o)
                        u = (t + LEAD_S - pr['t0']) / pr['T']
                        q_des = pr['q0'] + minjerk(u) * (pr['q1'] - pr['q0'])
                        info = dict(u=round(min(u, 1.0), 3), T=round(pr['T'], 3), a_o=round(pr['a_o'], 3))
                        out['prep_done'] = bool(u >= 1.0 and float(np.abs(q[3:] - pr['q1']).max()) < 0.02)
                    elif mode in ('OBSERVE', 'TRANSITION', 'HOLD') and (qtr is None or tau < 0.0):
                        q_des, how, par = look_ref(bp, fam, b_pred, st['q_des_prev'])
                        info = dict(ik=how, par=[round(x, 3) for x in par])
                    else:
                        qq, vv, _ = qeval(qtr['C'], tau, qtr['T'])
                        q_des = qq
                        info = dict(tau=round(tau, 3), T=round(qtr['T'], 3), aw=qtr['aw'])
                        out['in_transition'] = True
                        out['reach_pre'] = bool(tau >= qtr['T'])
                    if q_des is not None:
                        sp_des = q_des - off_fn(np.r_[q[:3], q_des])
                        ua_raw = (sp_des - s_end) / DT
                        ua = np.clip(ua_raw, -V_ARM, V_ARM)
                        if float(np.abs(ua - ua_raw).max()) > 1e-9:
                            st['n_clip'] += 1; info['clipped'] = True
                        st['q_des_prev'] = q_des
                        out['q_des'] = [round(float(x), 4) for x in q_des]
                    out['info'] = info
                    out['vis_geom'] = bool(bp.frame(np.r_[qm[:3], q[3:]]) is not None)
                if st['T_est'] is not None:
                    sc = stop_check(q, pipe, rel, s, ua, T_mo)
                    out['stop'] = dict(ok=sc['ok'], why=sc.get('why'), now=sc['now']['ok'] if isinstance(sc.get('now'), dict) else None, next=sc['next']['ok'] if isinstance(sc.get('next'), dict) else None,
                                       ms=sc.get('ms'), min_clear=sc.get('min_clear'), margin=sc.get('margin_after_interval_bound'))
                    out['disp'] = SC.disposition(sc)
                else:
                    out['disp'] = 'no_estimate'
                out['ua'] = [float(x) for x in ua]
                out['n_clip'] = st['n_clip']
                out['tm'] = (t_rx, time.perf_counter())
                conn.send(('tick', ver, out))
        except Exception:                                              # noqa: BLE001
            import traceback
            conn.send(('error', ver, traceback.format_exc()[-800:]))


def main():
    out = sys.argv[1]
    parent, child = mp.Pipe()
    proc = mp.get_context('fork').Process(target=worker, args=(child,), daemon=False)
    proc.start()
    pk, pk_child = mp.Pipe()
    os.environ['B_POOL_N_PLANNER'] = '1'
    proc_p = mp.get_context('fork').Process(target=worker, args=(pk_child,), daemon=False)
    proc_p.start()
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float64MultiArray, String
    import tf2_ros

    class Sup(Node):
        def __init__(self):
            super().__init__('b_capture_supervisor', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
            self.p_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
            self.p_nav = self.create_publisher(Twist, '/cmd_vel_nav', 10)
            self.p_arm = self.create_publisher(Float64MultiArray, '/far/arm_cmd', 10)
            self.p_task = self.create_publisher(String, '/task/conditions', 10)
            self.p_done = self.create_publisher(Bool, '/far/done', 10)
            self.p_abort = self.create_publisher(String, '/far/abort', 10)
            self.create_subscription(Twist, '/cmd_vel_nav_raw', self._nav_in, 20)
            self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
            self.create_subscription(Float64MultiArray, '/far/truth', self._truth, 20)
            self.create_subscription(Float64MultiArray, '/far/handle_obs', self._obs, 20)
            self.create_subscription(Path, '/plan', self._plan_in, 5)
            self.create_subscription(String, '/task/conditions_ack', self._ack, 20)
            self.create_subscription(String, '/task/region_status', self._rstat, 20)
            self.create_subscription(Float64MultiArray, '/task/gmpc_pred', self._pred, 20)
            self.create_subscription(String, '/far/exec_state', lambda m: setattr(self, 'xstate', m.data), 10)
            self.tfb = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tfb, self)
            self.truth = None; self.rel = np.zeros(3); self.path_t = None; self.xstate = None; self.pred = None; self.rs = None
            self.mode, self.ver, self.task_ver = 'INIT', 0, 0
            self.stopped, self.done_sent, self.aborted = None, False, None
            self.events, self.ticks, self.obs_log, self.acks, self.plans, self.retrs, self.rstat_log = [], [], [], [], [], [], []
            self.prior = None; self.plan = None; self.region = None; self.plan_ver = 0; self.T_est = None
            self.req = None; self.t_rp_try = -9.0; self.plan_pending = False
            self.goal_xy = None; self.t_goal_pub = -9.0; self.t_goal_set = 0.0
            self.qtr = None; self.t_retr = -9.0; self.n_retr_fail = 0
            self.still_t = None; self.n_late = 0; self.n_fwd = 0; self.b_prev = None
            self.create_timer(DT, self.tick)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def event(self, what, **kw):
            rec = dict(t=round(self.now(), 3), what=what, **kw)
            self.events.append(rec)
            print('[bcap2] ' + json.dumps(rec, ensure_ascii=False, default=str)[:420], flush=True)

        def ask(self, op, a, timeout):
            self.ver += 1
            a = dict(a, ver=self.ver)
            parent.send((op, a))
            w0 = time.perf_counter()
            while time.perf_counter() - w0 < timeout:
                if parent.poll(0.002):
                    tag, v, pay = parent.recv()
                    if v == self.ver:
                        return tag, pay
            return 'late', None

        def _nav_in(self, m):
            if self.stopped is not None:
                m = Twist()
            self.p_nav.publish(m); self.n_fwd += 1

        def _rel(self, m):
            self.rel = np.array([m.linear.x, m.linear.y, m.angular.z], float)

        def _truth(self, m):
            a_ = np.array(m.data, float)
            self.truth = dict(t=a_[0], q=a_[1:10], s=a_[10:16], s_end=a_[16:22], pipe=a_[22:94].reshape(8, 9))

        def _plan_in(self, m):
            self.path_t = self.now()

        def _ack(self, m):
            try:
                self.acks.append(json.loads(m.data))
            except Exception:                                          # noqa: BLE001
                pass

        def _rstat(self, m):
            try:
                self.rs = json.loads(m.data); self.rstat_log.append(self.rs)
            except Exception:                                          # noqa: BLE001
                pass

        def _pred(self, m):
            d = np.array(m.data, float)
            n = int(d[2])
            self.pred = dict(t=float(d[0]), dt=float(d[1]), P=d[3:3 + 3 * (n + 1)].reshape(n + 1, 3))

        def _obs(self, m):
            d = np.array(m.data, float)
            rec = dict(seq=int(d[0]), t_obs=float(d[1]), valid=bool(d[2] > 0.5), t_rx=round(self.now(), 3), t_proc=float(d[22]) if len(d) > 22 else None)
            self.obs_log.append(rec)
            if rec['valid'] and self.prior is not None:
                T_mo = self.tf2d('map', 'odom')
                if T_mo is None:
                    return
                T_oh = _T3(d[3:6]) @ d[6:22].reshape(4, 4)
                allow = self.mode in ('NAV', 'PREP')
                tag, pay = self.ask('obs', dict(seq=rec['seq'], t_obs=rec['t_obs'], t=self.now(), T_odom_handle=T_oh.tolist(), T_mo=T_mo.tolist(), allow_update=allow), 0.5)
                rec['est'] = None if pay is None else {k: pay.get(k) for k in ('n_hist', 'spread_mm', 'assoc_ok', 'estimate_set')}
                if tag == 'obs' and pay.get('estimate_set'):
                    self.T_est = pay['T_est']; self.plan_pending = True
                    self.event('estimate_set', **{k: pay.get(k) for k in ('seqs', 'assoc', 'spread_mm', 't_obs')})

        def tf2d(self, parent_, child):
            try:
                tr = self.tfb.lookup_transform(parent_, child, rclpy.time.Time())
            except Exception:                                          # noqa: BLE001
                return None
            o = tr.transform.rotation
            return np.array([tr.transform.translation.x, tr.transform.translation.y, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))])

        def pub_goal(self, xy, yaw):
            m = PoseStamped(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1]); m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.p_goal.publish(m)

        def pub_task(self, heading, box, region=None):
            self.task_ver += 1
            d_ = dict(task='b_capture', ver=self.task_ver, frame='map', stamp=self.now(), t_eff_s=0.5, valid_s=0.5, heading=float(heading), box=[[float(x) for x in r] for r in box])
            if region is not None:
                d_['region'] = region
            self.p_task.publish(String(data=json.dumps(d_)))

        def box_for(self, v_along, v_norm, yaw_now, d_world):
            c, s_ = math.cos(-yaw_now), math.sin(-yaw_now)
            db = np.array([c * d_world[0] - s_ * d_world[1], s_ * d_world[0] + c * d_world[1]])
            i = int(np.argmax(np.abs(db)))
            if abs(db[i]) >= math.cos(math.radians(15.0)):
                va = max(float(v_along), 0.0)
                box = [None, None, [-0.6, 0.6]]
                box[i] = [-0.02, va] if db[i] > 0 else [-va, 0.02]
                box[1 - i] = [-float(v_norm), float(v_norm)]
                return box
            v = max(min(float(v_along), float(v_norm)), 0.0)
            return [[-v, v], [-v, v], [-0.6, 0.6]]

        def stop_all(self, why, **kw):
            if self.stopped is None:
                self.stopped = dict(t=round(self.now(), 3), why=why, **kw)
                self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
                self.p_nav.publish(Twist())
                self.event('controlled_stop', why=why, **kw)

        def abort(self, why, **kw):
            if self.aborted is None:
                self.aborted = dict(t=round(self.now(), 3), why=why, **kw)
                self.p_abort.publish(String(data=why))
                self.event('engineering_abort', why=why, **kw)

        def base_prediction(self, t_s, T_max=7.0):
            """底盤預測（地圖；自 t_s 起每 0.05 s）：GMPC 1 s 預測（同座標、有效期內）→ 之後沿走廊依速度框外推（側向誤差 0.5 s 指數收斂、朝向取任務朝向）。只供搜尋候選。"""
            rg, pl = self.region, self.plan
            p0 = np.array(rg['line'][0]); p1 = np.array(rg['line'][1]); u = (p1 - p0) / np.linalg.norm(p1 - p0)
            nrm = np.array([-u[1], u[0]])
            pts = []
            if self.pred is not None and self.now() - self.pred['t'] <= 0.3:
                for k, P in enumerate(self.pred['P']):
                    pts.append((self.pred['t'] + k * self.pred['dt'], P[0], P[1], P[2]))
            else:
                b = self.cur_map
                pts.append((self.now(), b[0], b[1], b[2]))
            t_e, x, y, th = pts[-1]
            dt = 0.05
            while t_e < t_s + T_max:
                rel = np.array([x, y]) - p0
                s_, lat = float(rel @ u), float(rel @ nrm)
                a_door = float(-((np.array([x, y]) - np.array(self.h_xy)) @ np.array(pl['d'])))
                v = self.v_profile(a_door)
                s_ += v * dt; lat *= math.exp(-dt / 0.5)
                xy = p0 + s_ * u + lat * nrm
                th = th + max(-0.3 * dt, min(0.3 * dt, math.atan2(math.sin(rg['heading'] - th), math.cos(rg['heading'] - th))))
                t_e += dt; x, y = float(xy[0]), float(xy[1])
                pts.append((t_e, x, y, th))
            P = np.array(pts)
            P = P[P[:, 0] >= t_s - 1e-9]
            if len(P) == 0:
                return None
            P[:, 0] -= t_s
            return P

        def v_profile(self, a_door):
            pl = self.plan
            aw = self.qtr['aw'] if self.qtr is not None else float(pl['a_w'])
            if a_door <= aw:
                return 0.05
            return min(V_APPROACH, float(pl['v_tr']), math.sqrt(0.05 ** 2 + 2 * float(pl['a_dec']) * max(a_door - aw, 0.0)))

        def tick(self):
            t = self.now()
            if self.truth is None or self.aborted is not None:
                return
            T_mo = self.tf2d('map', 'odom'); b_od = self.tf2d('odom', 'base_footprint')
            if T_mo is None or b_od is None:
                return
            q = np.r_[b_od, self.truth['q'][3:9]]
            c, s_ = math.cos(T_mo[2]), math.sin(T_mo[2])
            self.cur_map = np.r_[c * b_od[0] - s_ * b_od[1] + T_mo[0], s_ * b_od[0] + c * b_od[1] + T_mo[1], b_od[2] + T_mo[2]]
            if self.mode == 'INIT':
                tag, pay = self.ask('prior', {}, 5.0)
                if tag != 'prior':
                    self.abort('prior_failed', detail=str(pay)[:200]); return
                self.prior = pay; self.event('prior', **pay)
                self.goal_xy = pay['goal_xy']; self.t_goal_set = t; self.pub_goal(self.goal_xy, pay['yaw']); self.t_goal_pub = t
                tag, pay2 = self.ask('off_anchor', dict(q=q.tolist(), s=self.truth['s'].tolist()), 2.0)
                self.event('off_anchor', ok=tag == 'off_anchor')
                self.mode = 'NAV'; self.event('mode', mode='NAV')
                return
            if self.goal_xy is not None and (self.path_t is None or self.path_t < self.t_goal_set) and t - self.t_goal_pub >= 1.0:
                self.pub_goal(self.goal_xy, self.prior['yaw']); self.t_goal_pub = t       # 首次發布可能早於探索完成：收到路徑前每 1 s 重送
            if self.b_prev is not None and t > self.b_prev[0]:
                v_w = (self.cur_map[:2] - self.b_prev[1][:2]) / (t - self.b_prev[0]); wz = (self.cur_map[2] - self.b_prev[1][2]) / (t - self.b_prev[0])
            else:
                v_w, wz = np.zeros(2), 0.0
            self.b_prev = (t, self.cur_map.copy())
            # ---- 規劃程序：可接任務狀態集合（走廊）
            src_ = 'estimate' if self.T_est is not None else 'prior'
            if self.req is None and self.mode in ('NAV', 'PREP') and (self.plan_pending or (self.plan is None and t - self.t_rp_try >= 1.0)) and (self.plan is None or src_ == 'estimate'):
                T_e = self.T_est
                if T_e is None:                                       # 估計未到：以地圖先驗（場景登錄的門資產名目位姿）給暫定走廊，讓導航提早收斂；估計到了再更新
                    T_e = (np.linalg.inv(_T3(T_mo)) @ np.asarray(self.prior['T_prior'], float)).tolist()
                self.plan_pending = False; self.t_rp_try = t; self.ver += 1
                self.req = dict(kind='region_plan', ver=self.ver, t=t, w=time.perf_counter(), src=src_)
                pk.send(('region_plan', dict(ver=self.ver, q_arm=q[3:].tolist(), T_mo=T_mo.tolist(), T_est=T_e, robot_xy=self.cur_map[:2].tolist())))
            if self.req is not None and pk.poll(0):
                tag, v, pay = pk.recv()
                rq, self.req = self.req, None
                if rq['kind'] == 'region_plan':
                    rec = dict(t_req=round(rq['t'], 3), t_done=round(t, 3), wall_s=round(time.perf_counter() - rq['w'], 3), tag=tag, ok=None if tag != 'region_plan' else pay.get('ok'), why=None if tag != 'region_plan' else pay.get('why'),
                               region=None if tag != 'region_plan' else pay.get('region'), feas=None if tag != 'region_plan' else pay.get('feas'), err=None if tag == 'region_plan' else str(pay)[:400],
                               plan=None if tag != 'region_plan' or not pay.get('plan') else {k: pay['plan'].get(k) for k in ('door_line_sb', 'a_w', 'a_t0', 'T_tr', 'last_valid_t', 'blind_s', 'path_source')},
                               obs=None if tag != 'region_plan' or not pay.get('plan') else [(f['a'], f['range'], f['oblique'], f['par']) for f in pay['plan'].get('obs', [])])
                    self.plans.append(rec); self.event('region_plan_result', **{k: rec[k] for k in ('t_req', 't_done', 'wall_s', 'ok', 'why', 'region', 'plan', 'err')})
                    rec['src'] = rq.get('src')
                    if tag == 'region_plan' and pay.get('ok') and self.mode in ('NAV', 'PREP') and self.qtr is None:
                        tag2, p2 = self.ask('adopt', dict(plan=pay['plan'], T_Hm=pay['T_Hm'], d=pay['d'], n=pay['n'], yaw=pay['yaw'], T_est=pay['T_est']), 5.0)
                        if tag2 == 'adopt':
                            self.plan = dict(pay['plan'], yaw=pay['yaw'], d=pay['d'], n=pay['n']); self.region = pay['region']; self.plan_ver += 1
                            self.h_xy = np.asarray(pay['T_Hm'], float)[:2, 3]; self.plan_src = rq.get('src')
                            if rq.get('src') == 'estimate' and self.mode == 'NAV':
                                self.mode = 'PREP'; self.event('mode', mode='PREP', plan_ver=self.plan_ver)
                            else:
                                self.event('region_adopted', plan_ver=self.plan_ver, src=rq.get('src'))
                        else:
                            self.event('adopt_failed', detail=str(p2)[:300])
                else:
                    self.on_retr(rq, tag, pay, t)
            # ---- 任務條件：朝向＋速度框＋區域
            inside = bool(self.rs is not None and self.rs.get('inside'))
            if self.plan is not None and self.stopped is None:
                pl, rg = self.plan, self.region
                a_m = float(-((self.cur_map[:2] - self.h_xy) @ np.array(pl['d'])))
                sb_m = float((self.cur_map[:2] - self.h_xy) @ np.array(pl['n']))
                in_sb = abs(sb_m - rg['sb_c']) <= rg['half_width']
                aw = self.qtr['aw'] if self.qtr is not None else float(pl['a_w'])
                if self.mode == 'HOLD' or (self.qtr is not None and a_m <= aw):
                    v_al, v_n = 0.0, 0.0
                elif in_sb:
                    v_al, v_n = self.v_profile(a_m), 0.05
                else:
                    v_al, v_n = V_APPROACH, V_APPROACH
                reg = dict(id='b%d' % self.plan_ver, line=rg['line'], half_width=rg['half_width'], extend=0.40, enter_by=rg['enter_by'])
                self.pub_task(pl['yaw'], self.box_for(v_al, v_n, float(self.cur_map[2]), np.array(pl['d'])), region=reg)
            # ---- 線上過渡重算請求
            if self.plan is not None and self.req is None and self.stopped is None:
                a_pred_now = float(-((self.cur_map[:2] + v_w * LEAD_S - self.h_xy) @ np.array(self.plan['d'])))
                want = (self.mode == 'OBSERVE' and self.qtr is None and a_pred_now <= float(self.plan['a_t0']) + RETR_TRIGGER) or \
                       (self.mode == 'TRANSITION' and self.qtr is not None and t + RETR_LEAD < self.qtr['t_s'] + self.qtr['T'] - 0.5)
                if want and t - self.t_retr >= RETR_PERIOD:
                    t_s = t + RETR_LEAD
                    BPm = self.base_prediction(t_s)
                    if BPm is not None:
                        self.t_retr = t; self.ver += 1
                        self.req = dict(kind='retr', ver=self.ver, t=t, w=time.perf_counter(), t_s=t_s)
                        qtr_send = None if self.qtr is None else dict(C=[np.asarray(c).tolist() for c in self.qtr['C']], t_s=self.qtr['t_s'], T=self.qtr['T'])
                        q_seed = self.ticks[-1].get('q_des') if self.ticks and self.ticks[-1].get('q_des') else q[3:].tolist()
                        pk.send(('retr', dict(ver=self.ver, t_s=t_s, base_pred=BPm.tolist(), qtr=qtr_send, q_seed=q_seed)))
            # ---- 手臂拍
            mode_w = self.mode if self.mode in ('PREP', 'OBSERVE', 'TRANSITION', 'HOLD') else 'NAV'
            w0 = time.perf_counter()
            tag, out_ = self.ask('tick', dict(t=t, mode=mode_w, q=q.tolist(), pipe=self.truth['pipe'].tolist(), rel=self.rel.tolist(), s=self.truth['s'].tolist(), s_end=self.truth['s_end'].tolist(),
                                              T_mo=T_mo.tolist(), v_world=v_w.tolist(), wz=wz), 0.9 * DT / max(float(os.environ.get('B_RTF', '0.4')), 1e-3))
            rec = dict(t=round(t, 3), mode=self.mode, wall_ms=round((time.perf_counter() - w0) * 1e3, 1), tag=tag, inside=inside, rs=None if self.rs is None else {k: self.rs.get(k) for k in ('s', 'lat', 'phase')})
            if tag != 'tick':
                self.n_late += 1
                rec['why'] = 'late_or_error' if tag == 'late' else str(out_)[:500]
                self.ticks.append(rec)
                if self.mode in ('PREP', 'OBSERVE', 'TRANSITION', 'HOLD') and self.stopped is None:
                    self.abort('tick_' + tag, detail=rec['why'])
                else:
                    self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
                return
            rec.update({k: out_.get(k) for k in ('a', 'sb', 'a_pred', 'disp', 'stop', 'info', 'ua', 'n_clip', 'vis_geom', 'prep_done', 'in_transition', 'reach_pre', 'q_des')})
            self.ticks.append(rec)
            disp = out_['disp']
            if self.stopped is not None:
                self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
            elif disp in ('commit', 'no_estimate'):
                self.p_arm.publish(Float64MultiArray(data=[float(x) for x in out_['ua']] if disp == 'commit' else [0.0] * 6))
            elif disp == 'stop_now':
                self.stop_all('stop_check_next_failed', stop=out_.get('stop'))
            else:
                self.abort('stop_check_now_failed', stop=out_.get('stop'))
                return
            if self.stopped is not None:
                return                                                  # 受控停止後不再切段（停止期間的工作程序輸出只記錄）
            if self.mode == 'PREP' and out_.get('prep_done'):
                self.mode = 'OBSERVE'; self.event('mode', mode='OBSERVE', a=out_.get('a'), sb=out_.get('sb'))
            elif self.mode == 'OBSERVE' and out_.get('in_transition'):
                self.mode = 'TRANSITION'; self.event('mode', mode='TRANSITION', a=out_.get('a'), sb=out_.get('sb'), inside=inside)
            elif self.mode == 'TRANSITION' and out_.get('reach_pre'):
                self.mode = 'HOLD'; self.event('mode', mode='HOLD', a=out_.get('a'), sb=out_.get('sb'), inside=inside, aw=self.qtr['aw'])
            if self.mode == 'HOLD' and not self.done_sent:
                still = float(np.linalg.norm(v_w)) < STILL['base'] and abs(wz) < STILL['yaw'] and float(np.abs(self.truth['pipe']).max()) < 1e-9 and float(np.abs(np.asarray(out_['ua'])).max()) < 1e-3
                if still:
                    self.still_t = self.still_t if self.still_t is not None else t
                    if t - self.still_t >= STILL['hold_s']:
                        self.done_sent = True; self.p_done.publish(Bool(data=True))
                        self.event('done_normal_end', a=out_.get('a'), sb=out_.get('sb'), inside=inside, region_sb=[self.region['sb_c'] - self.region['half_width'], self.region['sb_c'] + self.region['half_width']])
                else:
                    self.still_t = None

        def on_retr(self, rq, tag, pay, t):
            rec = dict(t_req=round(rq['t'], 3), t_s=round(rq['t_s'], 3), t_done=round(t, 3), wall_s=round(time.perf_counter() - rq['w'], 3), tag=tag, mode=self.mode)
            if tag != 'retr':
                rec['err'] = str(pay)[:500]; self.retrs.append(rec); self.event('retr_error', **rec); return
            rec.update(ok=pay['ok'], src=pay['src'], cands=pay['cands'], chosen=None if not pay['ok'] else {k: pay['chosen'][k] for k in ('aw', 'sb_w', 'T', 't_last_geom')})
            if pay['ok'] and t < rq['t_s'] - 1e-6:
                ch = pay['chosen']
                tag2, p2 = self.ask('adopt_tr', dict(C=ch['C'], t_s=rq['t_s'], T=ch['T'], q_pre=ch['q_pre'], aw=ch['aw'], t_now=t), 2.0)
                rec['adopted'] = bool(tag2 == 'adopt_tr' and p2.get('ok'))
                if rec['adopted']:
                    self.qtr = dict(C=ch['C'], t_s=rq['t_s'], T=ch['T'], aw=ch['aw']); self.n_retr_fail = 0
            else:
                rec['adopted'] = False
                if pay['ok']:
                    rec['why_not'] = 't_s_passed'
                elif self.mode == 'TRANSITION':
                    self.n_retr_fail += 1
            self.retrs.append(rec)
            self.event('retr', **{k: rec.get(k) for k in ('t_req', 't_s', 't_done', 'wall_s', 'ok', 'adopted', 'chosen', 'src')}, why=[c.get('why') for c in pay['cands']])

        def save(self):
            json.dump(dict(events=self.events, plans=self.plans, retrs=self.retrs, ticks=self.ticks, obs=self.obs_log, acks=self.acks[-100:], region_status=self.rstat_log[::5],
                           n_late=self.n_late, n_fwd=self.n_fwd, stopped=self.stopped, aborted=self.aborted,
                           params=dict(V_APPROACH=V_APPROACH, LEAD_S=LEAD_S, EST=EST, ASSOC_TOL=ASSOC_TOL, GOAL_A=GOAL_A, STILL=STILL, SB_CAND=SB_CAND, AW_CAND=AW_CAND, RETR_LEAD=RETR_LEAD, RETR_PERIOD=RETR_PERIOD,
                                       RETR_TRIGGER=RETR_TRIGGER, RTF=os.environ.get('B_RTF', '0.4'), SWEEP_S=os.environ.get('B_SWEEP_S', '0.040'), POOL_N=os.environ.get('B_POOL_N', '2'))),
                      open(out + '.json', 'w'), ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    tag, _v, pay = parent.recv()
    print('[bcap2] worker', tag, pay, flush=True)
    tagp, _vp, payp = pk.recv()
    print('[bcap2] planner', tagp, payp, flush=True)
    rclpy.init()
    n = Sup()
    try:
        while rclpy.ok() and not (n.done_sent and n.now() - (n.events[-1]['t'] if n.events else 0) > 3.0):
            rclpy.spin_once(n, timeout_sec=0.01)
            if n.xstate is not None and n.xstate.startswith('latched'):
                n.event('executor_latched', state=n.xstate); break
    except KeyboardInterrupt:
        pass
    finally:
        n.save()
        parent.send(None)
        pk.send(None)
        print('[bcap2] saved', flush=True)


if __name__ == '__main__':
    main()
