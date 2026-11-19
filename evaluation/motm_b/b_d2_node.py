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
APP_HO = os.environ.get('B_APP_HO', '1') == '1'              # 進走廊即交權給 W-GMPC 抓持控制器（主線修正）；0＝原直接接近＋r＝S_PRE 交權
APP_DIST, APP_V, APP_T_MIN, APP_T_LINE, APP_D_LINE = float(os.environ.get('B_APP_DIST', '0.30')), 0.05, 4.0, 2.5, 0.15   # 交權離抓取點距離、抓持控制器底盤速度（door_ahead_case V）、最短時長、沿接近軸段時長、接近軸起點退讓
UNIFIED = os.environ.get('B_UNIFIED', '0') == '1'          # 單一控制器（Codex 20261117_161844）：起點即交給同一個 W-GMPC 控制器；導航節點只當前端；導航目標＝抓取底盤位姿（Smac 決定來向）
PASS_GOAL = [float(x) for x in os.environ.get('B_PASS_GOAL', '0.5,0.8,1.5708').split(',')]   # 穿門目標（map；門口 x∈[0,1]、y=0，客廳側）
PASS_WP = [float(x) for x in os.environ.get('B_PASS_WP', '').split(',')] if os.environ.get('B_PASS_WP', '') else None   # 中途點：門扇外側、正對門口（開門後機器人在門扇後，LiDAR 看不到門口，關門時標的舊障礙格清不掉 ⇒ 先到看得見門口的位置）
PASS_WP_R = float(os.environ.get('B_PASS_WP_R', '0.45'))
PICK = os.environ.get('D2_PICK', '0') == '1'                          # 穿門後接茶几圓柱體取物＋返回
PICK_PRIOR = [float(x) for x in os.environ.get('B_PICK_PRIOR', '0.55,2.30,0.0').split(',')]   # 茶几先驗粗目標（map；茶几西側，面向桌）
PICK_RETURN = [float(x) for x in os.environ.get('B_PICK_RETURN', '1.5,-2.8,1.5708').split(',')]   # 返回＝起點
PICK_APP_DIST, PICK_T_MIN = float(os.environ.get('B_PICK_APP_DIST', '0.6')), float(os.environ.get('B_PICK_T_MIN', '4.0'))
DRAWER = os.environ.get('D2_DRAWER', '0') == '1'                      # Howard 11/18：開門→開抽屜→取筆→返回（需同時 D2_PICK＝1：沿用取物管線）
CHAIN = DRAWER and os.environ.get('D2_CHAIN', '0') == '1'             # Howard 11/19：同趟 茶几圓柱 → 車尾收納盒 → 抽屜筆 → 收納盒 → 返回
BIN = os.environ.get('D2_BIN', '0') == '1'                            # 車尾收納盒：放盒完成（pick_binned）才換下一個導航目標；下一件封包等收臂完成（pick_stowed）
PICK_APP_DYAW = float(os.environ['B_PICK_APP_DYAW']) if os.environ.get('B_PICK_APP_DYAW') else None   # 圓柱接近開始的偏航閘門（同抽屜 DR_APP_DYAW；s60：繞茶几角時距 0.6 m 即接近、車頭偏 1.36 rad ⇒ 收斂切桌角、底盤淨空 0.0993 被停）；未設＝原行為
DR_MOTM_V = float(os.environ['DR_MOTM_V']) if os.environ.get('DR_MOTM_V') else None   # Howard 11/19「門把、抽屜把手都要改」：抽屜把手移動中抓取的經過速度；未設＝停車抓
DOOR_CLOSE = os.environ.get('D2_DOOR_CLOSE', '0') == '1'             # Howard 11/19 回程關門：抽屜完成後先回臥室側關門，再回起點
DRAWER_CLOSE = os.environ.get('D2_DRAWER_CLOSE', '0') == '1'         # Howard 11/19「抽屜也要關阿」：筆放盒收臂後夾回把手推回，再返回
HANDLE_VISION = os.environ.get('D2_HANDLE_VISION', '0') == '1'      # Howard 11/19「不要把東西寫死 要實際用到視覺」：抽屜位姿由手腕深度把手偵測融合（登錄位姿只當搜尋先驗）
PICK_MOTM_V = float(os.environ['PICK_MOTM_V']) if os.environ.get('PICK_MOTM_V') else None   # Howard 11/19「茶几那個要移動中夾取 我要做整體速度最佳」：經過速度（m/s）；未設＝停車夾取
RET_CLEAR = os.environ.get('B_RET_CLEAR', '0') == '1'                # 返回目標送出前清一次全域代價圖（同穿門 PASS_CLEAR 的 Nav2 標準復原）；規劃持續失敗 RET_CLEAR_RETRY_S 再清（最多 3 次）（s66：串接路線沿途行人標記殘留 ⇒ 抽屜→起點 no valid path 582 次）
RET_CLEAR_RETRY_S = float(os.environ.get('B_RET_CLEAR_RETRY_S', '8.0'))
DR_APP_DYAW = float(os.environ.get('B_DR_APP_DYAW', '0.30'))         # 抓把手接近開始的底盤偏航差上限（rad）
DR_PRIOR = [float(x) for x in os.environ.get('B_DR_PRIOR', '-3.5,10.3,3.1416').split(',')]   # 選位完成前的粗目標（map；抽屜櫃前）
PASS_CLEAR = os.environ.get('B_PASS_CLEAR', '1') == '1'           # Howard 11/18：不要中途點，讓全域規劃自己找出路
PASS_CLEAR_WAIT_S = float(os.environ.get('B_PASS_CLEAR_WAIT_S', '1.2'))
APP_TRIG = float(os.environ.get('B_APP_TRIG', '0.6'))          # 單一控制器：有效估計且離導航目標 ≤ 此距離 ⇒ 接近開始（加入 TCP 任務）
FUSE_HO = os.environ.get('B_FUSE_HO', '1') == '1'            # 交權參考＝全部有效影格閘門內融合（c13／c22：最早 3 格估計沿接近軸誤差 2 mm 吃掉外殼—橫桿餘裕）
CL_LINE = os.environ.get('B_CL_LINE', '1') == '1'             # 混接／直線段以實際底盤即時解 IK（c7 阻點修正）；0＝原開迴路關節參考
EST = dict(n=3, spread_m=0.005, max_age_s=0.5, replan_shift_m=0.010)
ASSOC_TOL = dict(dxy_m=0.15, dz_m=0.10, dang_deg=10.0)
GOAL_A, SB_PRIOR = -0.49, 0.45
STILL = dict(base=0.003, yaw=0.003, arm=0.002, hold_s=0.3)
SB_CAND = [float(x) for x in os.environ.get('B_SB_CAND', '0.43,0.44,0.45,0.46').split(',')]   # 預設＝抓持窗口∩開門承接種子（假世界整鏈 0.45／0.46 可開、0.47 未綁、0.48 首拍加速度不過）；B_SB_CAND 供真承接核對
AW_CAND = (-0.10, -0.12)
DP_AW = (-0.10, -0.12)
DP_CANDS = [tuple(float(y) for y in x.split(':')) for x in os.environ.get('B_DP_CANDS', '4:0.18:4,3:0.18:4,4:0.15:4,4:0.18:3').split(',')]   # 直線起點須避開把手正前方（門旁抓法奇異帶 a≈0）：底盤約 0.05 m/s 時直線 4 s ⇒ 起點約在把手前 0.1 m   # 直線起點須避開把手正前方（門旁抓法奇異帶 a≈0）   # sb 0.445 離線：直線起點 0.22 自碰撞，0.15／0.18 可行   # （擺動秒、直線起點 m、直線秒）本輪候選參數；正式由當下狀態重算
DP_BLEND = 1.0
T_NEED_MARGIN = 1.0
BRANCH_CONT = os.environ.get('B_BRANCH_CONT', '1') == '1'
REGION_A_START, REGION_A_ENTER = 0.60, 0.30
RETR_LEAD, RETR_PERIOD, RETR_TRIGGER = 1.5, 1.0, 0.15
CHECK_EVERY = 2                                                      # 候選模擬每 2 拍（0.1 s）呼叫一次完整停止核對
V_CANDS = (0.10, 0.07, 0.05)
OBS_RATE = 0.8                                                       # 觀察律參考每拍變化上限＝0.8×關節速率上限×拍長
DEADLINE_MS = float(os.environ.get('B_DEADLINE_MS', '45.0'))                                                   # 每拍牆鐘期限（原登錄；不隨 RTF 延長）                 # 過渡起點在請求後 0.8 s（模擬）；過渡中每 0.5 s 重算；觀察段在預測 a ≤ 初始 a_t0＋0.10 起請求
REGION_A_END_MARGIN = 0.25
ENTER_MARGIN = 0.10
LOOK_REACH_M = 0.65                                                   # 注視 IK 前的可達預檢：TCP 目標離底盤原點水平距離上限（臂長＋掛載的寬鬆上界）


def _T3(p):
    import numpy as np
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


def mj_(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def mjd_(u):
    u = min(max(u, 0.0), 1.0)
    return 30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4


def mjdd_(u):
    u = min(max(u, 0.0), 1.0)
    return 60 * u - 180 * u ** 2 + 120 * u ** 3


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
    IS_PLANNER = bool(os.environ.get('B_POOL_N_PLANNER'))
    if IS_PLANNER:                                                     # 規劃程序（非即時的候選驗證）：放慢核、掃掠截止放寬；每拍工作程序仍用 B_SWEEP_S
        try:
            os.sched_setaffinity(0, {int(x) for x in os.environ.get('B_PLANNER_CPUS', '4,5,6,7,8,9,14,15,16,17,18,19').split(',')})
        except Exception:                                              # noqa: BLE001
            pass
    SC.SWEEP_S = float(os.environ.get('B_SWEEP_S_PL', '0.25')) if IS_PLANNER else float(os.environ.get('B_SWEEP_S', '0.040'))
    Rg, T_HG, a_H = G.PG.baseline_reference()
    S_PRE_ = float(G.PG.S_PRE)
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
        if a_ < A[0] - 0.05 or a_ > A[-1] + 0.30:                 # 底盤在觀察族範圍外：不解 IK（有上一拍參考就保持，否則觀察族端點構型）
            j_ = 0 if a_ < A[0] else -1
            return (np.asarray(seed, float).copy() if seed is not None else Q[j_].copy()), 'outside_family_range', list(fam[j_]['par'])
        a_c = float(np.clip(a_, A[0], A[-1]))
        i = int(np.clip(np.searchsorted(A, a_c), 1, len(A) - 1)); w = (a_c - A[i - 1]) / max(A[i] - A[i - 1], 1e-9)
        par = [fam[i - 1]['par'][k] + w * (fam[i]['par'][k] - fam[i - 1]['par'][k]) for k in range(3)]
        fseed = Q[i - 1] + w * (Q[i] - Q[i - 1])
        Tt = bp.look_at(np.asarray(b, float), *par)
        if float(np.linalg.norm(Tt[:2, 3] - np.asarray(b, float)[:2])) > LOOK_REACH_M:   # 目標超出手臂可達（離門太遠）：不解 IK（失敗迭代耗時）；有上一拍參考就保持
            return (np.asarray(seed, float).copy() if seed is not None else fseed), 'out_of_reach_hold', par
        qi, sm = bp.solve(np.asarray(b, float), Tt, fseed if seed is None else seed)
        if qi is None or sm < W.SIG_FLOOR or float(np.abs(qi - fseed).max()) > 0.3:
            if seed is not None:                                      # IK 不成立：保持上一拍參考（不跳到觀察族構型）
                return np.asarray(seed, float).copy(), 'hold_prev', par
            return fseed, 'fallback_family', par
        return qi, 'look_at', par

    def obs_ref(bp_, fam_, b_pred, prev):
        """觀察律：注視 IK 參考，再以 OBS_RATE×V_ARM 限制每拍變化（工作程序與候選模擬同式；避免命令夾限與參考跳動）。"""
        q_look, how, par = look_ref(bp_, fam_, b_pred, prev)
        if prev is None:
            return q_look, how, par
        prev = np.asarray(prev, float)
        return prev + np.clip(q_look - prev, -OBS_RATE * V_ARM * DT, OBS_RATE * V_ARM * DT), how, par

    def twist_at(Pv, t):
        """預測底盤在時刻 t 的本體扭轉（有限差分）與規劃座標位姿。"""
        k = int(np.clip(np.searchsorted(Pv[:, 0], t), 1, len(Pv) - 1))
        dtp = max(Pv[k, 0] - Pv[k - 1, 0], 1e-6)
        w = (t - Pv[k - 1, 0]) / dtp
        b = Pv[k - 1, 1:4] + min(max(w, 0.0), 1.0) * (Pv[k, 1:4] - Pv[k - 1, 1:4])
        vw = (Pv[k, 1:3] - Pv[k - 1, 1:3]) / dtp; wz = (Pv[k, 3] - Pv[k - 1, 3]) / dtp
        c_, s_ = math.cos(b[2]), math.sin(b[2])
        return b, np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], wz]), vw, wz

    def grasp_T(T_H, r):
        T = np.asarray(G.E.OT.object_target(T_H, T_HG, a_H, 0.0)['T_WE'], float).copy()
        T[:3, 3] += r * (np.asarray(T_H, float)[:3, :3] @ np.asarray(a_H, float))
        return T

    def so3_log(R):
        c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
        if th < 1e-9:
            return np.zeros(3)
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th)) * th

    def so3_exp(w):
        th = float(np.linalg.norm(w))
        if th < 1e-12:
            return np.eye(3)
        k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx

    def dp_ref_at(dp, t):
        """直接接近參考（關節）在時刻 t：t<t_s 回 None（保持）；之後取取樣點線性內插；超過 t_h 取末點。"""
        if dp is None or t < dp['t_s'] - 1e-9:
            return None
        T_ = np.asarray(dp['tq'], float)
        k = int(np.clip(np.searchsorted(T_, t), 1, len(T_) - 1))
        w = min(max((t - T_[k - 1]) / max(T_[k] - T_[k - 1], 1e-9), 0.0), 1.0)
        Q_ = dp['Q']
        return np.asarray(Q_[k - 1], float) + w * (np.asarray(Q_[k], float) - np.asarray(Q_[k - 1], float))

    def base_lead(q9_od, pipe, lead):
        """底盤在 t＋lead 的里程座標位姿：在途管線逐步積分（每步 PHYS），之後保持管線末筆命令（工作程序與候選模擬同式）。"""
        b = np.asarray(q9_od[:3], float).copy()
        us = [np.asarray(u_[:3], float) for u_ in pipe]
        n_rest = max(int(round(lead / PHYS)) - len(us), 0)
        for u_ in us + [us[-1]] * n_rest:
            c_, s_ = math.cos(b[2]), math.sin(b[2])
            b = b + PHYS * np.array([c_ * u_[0] - s_ * u_[1], s_ * u_[0] + c_ * u_[1], u_[2]])
        return b

    def cl_target(cur, tt, b_m):
        """閉迴路混接／直線（CL_LINE）：擺動關節參考掛在實際（前推）底盤上，與世界固定的接近直線依 β 混接，以 t 時刻預測底盤解 IK。
        回 (q 或 None, 原因)。"""
        t_D, t_bl, t_ln, d_ln = cur['t_D'], cur['t_blend'], cur['t_line'], cur['d_line']
        T_H = np.asarray(cur['T_H'], float)
        C = quintic(cur['q_s'], cur['v_s'], np.zeros(6), cur['q_D'], t_D - cur['t_s'])
        qj = qeval(C, tt - cur['t_s'], t_D - cur['t_s'])[0]
        beta = mj_((tt - (t_D - t_bl)) / t_bl)
        r = d_ln * (1 - mj_((min(tt, cur['t_g']) - (t_D - t_bl)) / (t_ln + t_bl)))
        Tj = DFK.FC.fk_jac(np.r_[b_m, qj])[0]; Tl = grasp_T(T_H, r)
        Tt = np.eye(4); Tt[:3, 3] = (1 - beta) * Tj[:3, 3] + beta * Tl[:3, 3]
        Tt[:3, :3] = Tj[:3, :3] @ so3_exp(beta * so3_log(Tj[:3, :3].T @ Tl[:3, :3]))
        seed = dp_ref_at(cur, tt)
        q, res, _ = DFK.ik_closure_fc(Tt, b_m, seed)
        if not (res < 1e-6 and np.isfinite(q).all()) or float(np.abs(q - seed).max()) > 0.5:
            return None, 'cl_ik'
        return q, None

    def law_ua(dp, dp_next, t, q9_od, s_end, pipe=None, T_mo=None):
        """直接接近命令律（工作程序與候選模擬同式）：參考取 t＋LEAD_S 的關節目標 − 偏置 ⇒ 設定點；ua＝(設定點−在途末端設定點)/DT，夾 ±V_ARM。
        參考未開始 ⇒ ua＝0（保持）。替換段到其 t_s 才切換。
        CL_LINE（預設開）：混接起點之後的參考不再用規劃時的預測底盤解好的關節，而是以實際底盤（在途管線前推到 t＋LEAD_S，規劃座標）即時解 IK——
        底盤實際走得比預測快／慢時 TCP 仍落在世界固定的接近線上（c7：開迴路參考使交權時 TCP 偏 19 mm，抓持控制器首拍手臂命令超限）。
        閉迴路 IK 失敗 ⇒ 回報夾限（候選模擬即拒絕）。"""
        cur = dp
        if dp_next is not None and t >= dp_next['t_s'] - 1e-9:
            cur = dp_next
        tt = t + LEAD_S
        qd = dp_ref_at(cur, tt)
        if qd is None:
            return np.zeros(6), False, None
        bad = False
        if CL_LINE and cur.get('T_H') is not None and pipe is not None and tt > cur['t_D'] - cur['t_blend'] + 1e-9:
            b_m = to_map(np.r_[base_lead(q9_od, pipe, LEAD_S), np.zeros(6)], np.asarray(cur.get('T_mo_plan', T_mo), float))[:3]   # 固定用規劃當時的 map←odom ⇒ 等同在里程座標追把手
            q_cl, why_ = cl_target(cur, tt, b_m)
            if q_cl is None:
                bad = True
            else:
                qd = q_cl
        sp = qd - off_fn(np.r_[q9_od[:3], qd])
        ua_raw = (sp - s_end) / DT
        ua = np.clip(ua_raw, -V_ARM, V_ARM)
        return ua, bool(bad or float(np.abs(ua - ua_raw).max()) > 1e-9), qd

    def build_dp(bp, T_H, Pv, t_s, q_s, v_s, aw, T_move, d_line, t_line, t_blend, sb_lim, cont=False):
        """擺動（關節五次多項式：承接 (q_s, v_s) → 直線起點構型 q_D 靜止）＋混接（任務空間權重）＋沿接近軸直線（r：d_line→0 最小急動，t_line）。
        t_g＝預測底盤到 aw 的時刻；直線 r＝S_PRE 的時刻＝交權 t_h；參考取樣到 t_h（之後由抓持控制器承接）。回 dict 或失敗原因。"""
        AD = np.array([bp.to_door(r[1:3]) for r in Pv])
        idx = np.where(AD[:, 0] <= aw)[0]
        if len(idx) == 0:
            return None, 'not_reached_in_prediction'
        t_g = float(Pv[idx[0], 0]); sb_g = float(AD[idx[0], 1])
        if not (sb_lim[0] - 1e-9 <= sb_g <= sb_lim[1] + 1e-9):
            return None, 'sb_at_grasp_%.3f' % sb_g
        t_D = t_g - t_line
        if cont:                                                       # 承接進行中的參考：擺動自 t_s 起、時長＝到直線起點
            if t_D - t_s < 1.0:
                return None, 'too_late_for_swing(t_g %.2f t_D %.2f t_s %.2f)' % (t_g, t_D, t_s)
            t_s2 = t_s
        else:
            if t_D - t_s < T_move - 1e-6:
                return None, 'too_late_for_swing(t_g %.2f t_D %.2f t_s %.2f)' % (t_g, t_D, t_s)
            t_s2 = t_D - T_move
        T_D = grasp_T(T_H, d_line)
        b_D = twist_at(Pv, t_D)[0]
        best = None
        if BRANCH_CONT:                                                # 直線起點取抓取所用門旁分支：門旁種子在 b_D 解抓取位（r＝0）→ 沿接近軸連續退到 d_line（同分支）
            sd_, okb = np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157]), True
            for r_ in np.linspace(0.0, d_line, 12):
                qn_, rn_, sn_ = DFK.ik_closure_fc(grasp_T(T_H, float(r_)), b_D, sd_)
                if not (rn_ < 1e-6 and np.isfinite(qn_).all()) or float(np.abs(qn_ - sd_).max()) > 0.5:
                    okb = False; break
                sd_ = qn_
            if okb and bp.sigma6(np.r_[b_D, sd_]) >= W.SIG_FLOOR and np.all(sd_ >= LO) and np.all(sd_ <= HI) and bp.ok_cfg(b_D, sd_)[0]:
                best = (float(np.abs(sd_ - q_s).max()), sd_)
            else:
                return None, 'no_line_start_on_grasp_branch'
        for sd in (() if BRANCH_CONT else (q_s, np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157]), np.array([-0.396, -0.118, 1.409, -1.471, -0.398, -0.102]),
                   np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018]), np.array([0.0, -0.3, 1.2, 0.0, -0.9, 0.0]), np.array([0.4, -0.2, 1.3, 1.4, -0.4, 0.1]),
                   np.array([-1.249, -0.681, 1.044, -1.626, -1.229, 0.164]))):   # 同 door_direct_approach 的種子組（＋其直線起點解）
            qs_, rs_, ss_ = DFK.ik_closure_fc(T_D, b_D, sd)
            if rs_ < 1e-6 and np.isfinite(qs_).all() and ss_ >= W.SIG_FLOOR and np.all(qs_ >= LO) and np.all(qs_ <= HI) and bp.ok_cfg(b_D, qs_)[0]:
                d_ = float(np.abs(qs_ - q_s).max())
                if best is None or d_ < best[0]:
                    best = (d_, qs_)
        if best is None:
            return None, 'no_line_start_ik'
        q_D = best[1]
        C = quintic(q_s, v_s, np.zeros(6), q_D, t_D - t_s2)
        # 先走一次確認可解（任務空間混接需逐點 IK）
        tq, Q = [], []
        seed = q_s.copy(); t = t_s2
        r_of = lambda tt: d_line * (1 - mj_((tt - (t_D - t_blend)) / (t_line + t_blend)))   # noqa: E731
        t_h = None
        while t <= t_g + 1e-9:
            b = twist_at(Pv, t)[0]
            qj, _, _ = qeval(C, t - t_s2, t_D - t_s2)
            beta = mj_((t - (t_D - t_blend)) / t_blend) if t > t_D - t_blend else 0.0
            if beta <= 0.0:
                q = qj
            else:
                Tj = DFK.FC.fk_jac(np.r_[b, qj])[0]
                r = r_of(t); Tl = grasp_T(T_H, r)
                Tt = np.eye(4); Tt[:3, 3] = (1 - beta) * Tj[:3, 3] + beta * Tl[:3, 3]
                Tt[:3, :3] = Tj[:3, :3] @ so3_exp(beta * so3_log(Tj[:3, :3].T @ Tl[:3, :3]))
                q, res, sm = DFK.ik_closure_fc(Tt, b, seed)
                if not (res < 1e-6 and np.isfinite(q).all()) or float(np.abs(q - seed).max()) > 1.5 * V_ARM * DT:
                    return None, 'ik_blend_line@%.2f' % (t - t_s2)
                if beta >= 1.0 and r <= S_PRE_ + 1e-9 and t_h is None:
                    t_h = t
            tq.append(t); Q.append(q.copy()); seed = q
            if t_h is not None:
                break
            t += DT
        if t_h is None:
            return None, 'no_handover_point'
        u_h = (t_h - (t_D - t_blend)) / (t_line + t_blend)
        Tl_ = t_line + t_blend
        v_h = -d_line * mjd_(u_h) / Tl_; a_h = -d_line * mjdd_(u_h) / Tl_ ** 2
        return dict(t_s=t_s2, t_D=t_D, t_g=t_g, t_h=t_h, sb_g=sb_g, aw=aw, T_move=T_move, d_line=d_line, t_line=t_line, t_blend=t_blend, tq=tq, Q=[x.tolist() for x in Q],
                    q_s=np.asarray(q_s, float).tolist(), v_s=np.asarray(v_s, float).tolist(), T_H=np.asarray(T_H, float).tolist(),
                    ins=dict(s0=float(r_of(t_h)), v0=float(v_h), a0=float(a_h), T=float(t_g - t_h)), q_D=q_D.tolist()), None

    def simulate_dp(a, Pv, T_mo, T_ref, dp_cur, dp_next, cand, bp, T_H):
        """請求時實測狀態（關節、設定點、真管線）→ 以現行命令律前推到候選 t_s → 候選命令律，逐拍模擬受控對象（8 步管線＋一階關節＋偏置）到交權 t_h；
        每拍核限位、夾限、可見（幾何）；每 CHECK_EVERY 拍以同一份 b_stopcheck（實際模擬狀態）核 now／next。"""
        Tinv = np.linalg.inv(_T3(T_ref))
        q9 = np.asarray(a['q'], float).copy(); s_ = np.asarray(a['s'], float).copy(); pipe = [np.asarray(u, float).copy() for u in a['pipe']]
        out = dict(why=None, n_check=0, n_clip=0, last_vis=None, min_margin=None, ms_check=0.0)
        t = float(a['t_req']); k_tick = 0; mm = dict(env=9.9, base=9.9, self=9.9)
        while t < cand['t_h'] - 1e-9:
            b_map, ub, vw, wz = twist_at(Pv, t)
            q9_od = np.r_[(Tinv @ np.r_[b_map[:2], 0.0, 1.0])[:2], b_map[2] - float(T_ref[2]), q9[3:]]
            s_end = s_ + PHYS * sum(p_[3:] for p_ in pipe)
            if t < cand['t_s'] - 1e-9:
                ua, clipped, _ = law_ua(dp_cur, dp_next, t, q9_od, s_end, pipe, T_ref)
            else:
                ua, clipped, _ = law_ua(cand, None, t, q9_od, s_end, pipe, T_ref)
            if clipped:
                out['n_clip'] += 1
                if t >= cand['t_s'] - 1e-9:
                    out['why'] = 'clip@%.2f' % (t - cand['t_s']); return out
            vd = None
            Tc = K.fk(np.r_[b_map, q9[3:]], BP.W0.OPT); Ti = np.linalg.inv(Tc); okv = True
            for X_ in (T_H[:3, 3], T_H[:3, 3] + T_H[:3, 0] * 0.1, T_H[:3, 3] - T_H[:3, 0] * 0.1):
                pc = (Ti @ np.r_[X_, 1.0])[:3]
                if pc[2] < 0.12:
                    okv = False; break
                u_, v_ = (Kc @ (pc / pc[2]))[:2]
                if not (5 < u_ < 635 and 5 < v_ < 475):
                    okv = False; break
            if okv:
                out['last_vis'] = round(t, 3)
            if k_tick % CHECK_EVERY == 0:
                rel = twist_at(Pv, t + D * PHYS)[1]
                w0 = time.perf_counter()
                sc = stop_check(q9_od, pipe, rel, s_, ua, T_mo)
                out['ms_check'] += (time.perf_counter() - w0) * 1e3; out['n_check'] += 1
                for k_ in mm:
                    mv = (sc.get('margin_after_interval_bound') or {}).get(k_)
                    if mv is not None:
                        mm[k_] = min(mm[k_], float(mv))
                if SC.disposition(sc) != 'commit':
                    out['why'] = 'stopcheck@%.2f:%s' % (t - float(a['t_req']), sc.get('why') or ('now' if not sc['now']['ok'] else 'next'))
                    out['min_margin'] = {k_: round(v_, 4) for k_, v_ in mm.items()}
                    try:                                               # 阻點診斷：當刻構型（規劃座標）、各項淨空與最近配對、手臂速率命令、停止核對細項
                        cl_ = W.clear_door(np.r_[b_map, q9[3:], 0.0], T_H)
                        out['block'] = dict(t_abs=round(t, 3), seg=('line' if t >= cand['t_D'] else ('blend' if t >= cand['t_D'] - DP_BLEND else 'swing')) if t >= cand['t_s'] else 'pre',
                                            q9_plan=[round(float(x), 4) for x in np.r_[b_map, q9[3:]]], s=[round(float(x), 4) for x in s_], ua=[round(float(x), 4) for x in ua],
                                            clear={k_: (round(float(v_), 4) if isinstance(v_, (int, float, np.floating)) else str(v_)[:80]) for k_, v_ in cl_.items()},
                                            now={k_: sc['now'].get(k_) for k_ in ('ok', 'limits_ok', 'settled', 'coverage_ok', 'min_clear', 'margin_after_interval_bound', 'settle_s')} if isinstance(sc.get('now'), dict) else None,
                                            next={k_: sc['next'].get(k_) for k_ in ('ok', 'limits_ok', 'settled', 'coverage_ok', 'min_clear', 'margin_after_interval_bound', 'settle_s')} if isinstance(sc.get('next'), dict) else None)
                    except Exception as e_:                            # noqa: BLE001
                        out['block'] = dict(err=repr(e_)[:200])
                    return out
            u_base = twist_at(Pv, t + D * PHYS)[1]
            for _ in range(NPH):
                pipe.append(np.r_[u_base, ua]); ue = pipe.pop(0)
                s_ = np.clip(s_ + PHYS * ue[3:], LO, HI)
                q9[3:] = q9[3:] + AL * (s_ + off_fn(q9) - q9[3:])
            if np.any(q9[3:] < LO) or np.any(q9[3:] > HI):
                out['why'] = 'limit@%.2f' % (t - float(a['t_req'])); return out
            t += DT; k_tick += 1
        out['min_margin'] = {k_: round(v_, 4) for k_, v_ in mm.items()}; out['ms_check'] = round(out['ms_check'], 1)
        out['q_h_sim'] = q9.tolist(); out['s_h_sim'] = s_.tolist()
        return out

    import gc
    gc.collect(); gc.freeze(); gc.disable()                          # 拍內不跑自動 GC；只在拍間空檔收第 0 代（同 AHEAD_GC_FREEZE 的作法）
    conn.send(('ready', 0, dict(sweep_s=SC.SWEEP_S, gc='frozen+manual_gen0_between_ticks')))
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
                sreg = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
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
                a_start, a_end, a_enter = REGION_A_START, float(aw_r) - REGION_A_END_MARGIN, REGION_A_ENTER
                region = dict(sb_c=sb_c, half_width=w, sb_feasible=[sb_lo, sb_hi], aw_feasible=aw_r, a_start=a_start, a_end=round(a_end, 3), a_enter_by=a_enter, enter_by=round(a_start - a_enter, 3),
                              line=[(T_Hm[:2, 3] + sb_c * n_ - a_start * d_).tolist(), (T_Hm[:2, 3] + sb_c * n_ - a_end * d_).tolist()], heading=yaw_,
                              note='離門距離候選只取 0.43–0.46（抓持窗口∩假世界整鏈可開門的種子 0.45／0.46；不是已證連續區間）')
                st['region_p'] = region
                st['T_est'] = np.asarray(a['T_est'], float)
                out.update(ok=True, plan=dict(a_w=aw_r, door_line_sb=sb_c), region=region, wall_s=round(time.perf_counter() - t0, 3))
                conn.send(('region_plan', ver, out))
            elif op == 'set_ref':                                      # 工作程序：停止核對用的把手參考換成新提交的參考（里程座標）
                st['T_est'] = np.asarray(a['T_est'], float)
                conn.send(('set_ref', ver, dict(ok=True)))
            elif op == 'dapp':                                         # 規劃程序：直接接近（擺動＋混接＋直線）候選，自請求時實測狀態逐拍模擬＋完整停止核對
                t0 = time.perf_counter()
                T_ref_m = np.asarray(a['T_ref_m'], float)               # 把手參考（規劃座標）
                st['T_est'] = np.asarray(a['T_est'], float)            # 停止核對用（里程座標）
                d_, n_, yaw_ = door_frame(T_ref_m)
                bp = make_bp(T_ref_m, d_, n_, yaw_)
                T_mo = np.asarray(a['T_mo'], float); T_ref = np.asarray(a['T_ref'], float)
                dp_cur, dp_next = a.get('dp_cur'), a.get('dp_next')
                Tinv = np.linalg.inv(_T3(T_ref))
                cands, chosen = [], None
                for (v_, Pv) in a['base_preds']:
                    Pv = np.asarray(Pv, float)
                    # 候選起點狀態：以現行命令律前推到候選 t_s 的參考（新段承接；現行段尚未開始 ⇒ 導航姿態靜止）
                    for (T_move, d_line, t_line) in DP_CANDS:
                        for aw in DP_AW:
                            rec = dict(v=v_, T_move=T_move, d_line=d_line, t_line=t_line, aw=aw)
                            t_s_min = float(a['t_req']) + RETR_LEAD
                            act = dp_next if (dp_next is not None and t_s_min >= dp_next['t_s']) else dp_cur
                            q_s = dp_ref_at(act, t_s_min) if act is not None else None
                            if q_s is None:                             # 現行參考尚未開始（或沒有）⇒ 手臂靜止於現狀，新段可晚於 t_s_min 開始
                                q_s, v_s, cont = np.asarray(a['q'], float)[3:].copy() if act is None else np.asarray(act['Q'][0], float), np.zeros(6), False
                            else:                                       # 現行參考進行中 ⇒ 新段在 t_s_min 承接（位置、速度連續）
                                v_s = (dp_ref_at(act, t_s_min + DT) - q_s) / DT; cont = True
                            cand, why = build_dp(bp, T_ref_m, Pv, t_s_min, q_s, v_s, aw, T_move, d_line, t_line, DP_BLEND, a['sb_lim'], cont=cont)
                            if cand is None:
                                rec['why'] = why; cands.append(rec); continue
                            rec.update(t_s=round(cand['t_s'], 3), t_h=round(cand['t_h'], 3), t_g=round(cand['t_g'], 3), sb_g=round(cand['sb_g'], 4))
                            sim = simulate_dp(a, Pv, T_mo, T_ref, dp_cur, dp_next, cand, bp, T_ref_m)
                            rec.update({k: sim.get(k) for k in ('why', 'n_check', 'n_clip', 'last_vis', 'min_margin', 'ms_check', 'block')})
                            if sim['why'] is None:
                                rec['ok'] = True; cands.append(rec)
                                chosen = dict(cand, v=v_, cont=bool(cont), cont_from=(float(act['t_s']) if cont else None), T_mo_plan=[float(x) for x in T_ref], last_vis=sim['last_vis'], min_margin=sim['min_margin'], q_h_sim=sim['q_h_sim'], s_h_sim=sim['s_h_sim'])
                                break
                            cands.append(rec)
                        if chosen is not None:
                            break
                    if chosen is not None:
                        break
                conn.send(('dapp', ver, dict(ok=chosen is not None, chosen=chosen, cands=cands, wall_s=round(time.perf_counter() - t0, 3))))
            elif op == 'adopt':                                        # 工作程序：採用區域規劃（觀察族、停止核對用估計）
                pl = a['plan']
                st['bp'], st['plan'], st['T_est'] = make_bp(a['T_Hm'], a['d'], a['n'], a['yaw']), pl, np.asarray(a['T_est'], float)
                st['dp'], st['dp_next'] = None, None
                conn.send(('adopt', ver, dict(ok=True)))
            elif op == 'adopt_dp':                                     # 工作程序：採用（或替換）直接接近參考；替換段到其 t_s 才切換
                new = dict(a['dp']); new.update(t_s=float(a['dp']['t_s']), t_h=float(a['dp']['t_h']))   # 完整參考（閉迴路直線需 t_D、T_H、q_s… 與規劃座標 T_mo_plan）
                if a.get('slot', 'replace') == 'replace':               # 位置由主程序決定（c9：承接型被當成替換，現行段沒跑、新段起點對不上）
                    st['dp'], st['dp_next'] = new, None
                else:
                    st['dp_next'] = new
                if a.get('T_est') is not None:
                    st['T_est_pending'] = (float(new['t_s']), np.asarray(a['T_est'], float))   # 新參考隨新段生效
                conn.send(('adopt_dp', ver, dict(ok=True)))
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
                if st.get('T_est_pending') is not None and t >= st['T_est_pending'][0] - 1e-9:
                    st['T_est'] = st['T_est_pending'][1]; st['T_est_pending'] = None
                if st['T_est'] is not None and st['bp'] is not None and mode in ('APPROACH', 'NAV'):
                    bp = st['bp']
                    qm = to_map(q, np.asarray(a.get('T_ref', T_mo), float))
                    am, sbm = bp.to_door(qm[:2])
                    out.update(a=round(am, 4), sb=round(sbm, 4))
                    if st.get('dp_next') is not None and t >= st['dp_next']['t_s'] - 1e-9:
                        st['dp'], st['dp_next'] = st['dp_next'], None
                    info = {}
                    if mode == 'APPROACH' and st.get('dp') is not None:
                        ua, clipped, qd = law_ua(st['dp'], st.get('dp_next'), t, q, s_end, pipe, np.asarray(a.get('T_ref', T_mo), float))
                        if clipped:
                            st['n_clip'] += 1; info['clipped'] = True
                        info['in_dp'] = qd is not None
                        if qd is not None:
                            out['q_des'] = [round(float(x), 4) for x in qd]
                        out['t_h'] = st['dp']['t_h']
                    out['info'] = info
                    out['vis_geom'] = bool(bp.frame(np.r_[qm[:3], q[3:]]) is not None)
                out['ms_law'] = round((time.perf_counter() - t_rx) * 1e3, 1)
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
                gc.collect(0)
        except Exception:                                              # noqa: BLE001
            import traceback
            conn.send(('error', ver, traceback.format_exc()[-800:]))


def main():
    out = sys.argv[1]
    parent, child = mp.Pipe()
    proc = mp.get_context('fork').Process(target=worker, args=(child,), daemon=False)
    proc.start()
    pk, pk_child = mp.Pipe()
    os.environ['B_POOL_N_PLANNER'] = os.environ.get('B_POOL_N_PL', '2')
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
            self.create_subscription(Twist, '/cmd_vel_grasp', self._grasp_in, 20)
            self.p_ho = self.create_publisher(String, '/far/handover', 10)
            self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
            self.create_subscription(Float64MultiArray, '/far/truth', self._truth, 20)
            self.create_subscription(Float64MultiArray, '/far/handle_obs', self._obs, 20)
            self.create_subscription(Path, '/plan', self._plan_in, 5)
            self.create_subscription(String, '/task/conditions_ack', self._ack, 20)
            self.create_subscription(String, '/task/region_status', self._rstat, 20)
            self.create_subscription(Float64MultiArray, '/task/gmpc_pred', self._pred, 20)
            self.create_subscription(String, '/far/exec_state', self._xs_in, 10)
            self.create_subscription(Float64MultiArray, '/cyl/obs', self._cyl_in, 20)
            from sensor_msgs.msg import LaserScan as _LS
            self.last_scan = None
            self.create_subscription(_LS, '/scan', self._scan_in, 5)   # 選位時用：地圖沒有、雷射看得到的未知障礙（Isaac 第三十二趟）
            self.hdl_obs = []
            self.leaf_obs, self.door_end, self.dc = [], None, dict(stage=None)
            self.create_subscription(Float64MultiArray, '/door/leaf_obs', self._leaf_in, 20)
            self.create_subscription(Float64MultiArray, '/dr/handle_obs', self._hdl_in, 20)
            self._CR = None
            if PICK:                                                  # 主執行緒預先 import（背景執行緒裡 import 會與 rclpy 收訊息衝突：s29 take_message 崩潰）
                if HERE not in sys.path:
                    sys.path.insert(0, HERE)
                _pre = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces', 'rosidl_runtime_py', 'nav2_msgs')
                _snap = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in _pre}   # 專案模組會裝假 std_msgs（ros_stub）⇒ 匯入後還原 ROS 模組（同 world_d2_node）
                import cyl_reach_map as _CR_
                if DRAWER:
                    import drawer_reach as _DRR_
                    self._DRR = _DRR_
                    import drawer_motm as _DRM_
                    self._DRM = _DRM_
                for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in _pre and k_ not in _snap]:
                    sys.modules.pop(k_, None)
                sys.modules.update(_snap)
                self._CR = _CR_
            self.cyl_obs, self.pick_goal, self.pick_sel, self.pick_th, self.pick_box, self.pick_app_t, self.pick_lifted, self.pick_ret = [], None, None, None, None, None, False, False
            self.dr = dict(stage='handle_sel', box=None, th=None, q_open=None, t_obs=None, sel=None)
            self.pass_seen, self.pass_t, self.t_pass_pub, self.pass_cleared = False, None, -9.0, False
            from nav2_msgs.srv import ClearEntireCostmap
            self._clr_cli = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')
            self._clr_req, self._clr_fut, self._clr_t = ClearEntireCostmap.Request, None, None
            self.tfb = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tfb, self)
            self.truth = None; self.rel = np.zeros(3); self.path_t = None; self.xstate = None; self.pred = None; self.rs = None
            self.mode, self.ver, self.task_ver = 'INIT', 0, 0
            self.stopped, self.done_sent, self.aborted = None, False, None
            self.events, self.ticks, self.obs_log, self.acks, self.plans, self.retrs, self.rstat_log = [], [], [], [], [], [], []
            self.prior = None; self.plan = None; self.region = None; self.plan_ver = 0; self.T_est = None
            self.req = None; self.t_rp_try = -9.0; self.plan_pending = False
            self.goal_xy = None; self.t_goal_pub = -9.0; self.t_goal_set = 0.0
            self.qtr = None; self.qtr_next = None; self.t_retr = -9.0; self.n_retr_fail = 0; self.v_active = V_CANDS[0]; self.ua_hist = [[0.0] * 6, [0.0] * 6]; self.n_deadline = 0
            self.dp, self.dp_next, self.owner, self.ho_sent, self.n_grasp_fwd = None, None, 'NAV', None, 0
            self.ref = None; self.ref_ver = 0; self.ref_cand = None; self.ref_log = []
            self.obs_T = []
            self.still_t = None; self.n_late = 0; self.n_fwd = 0; self.b_prev = None; self.stale = []; self.odom_rates = (9.0, 9.0, 9.0); self.od_prev = None; self.last_cert = None
            self.create_timer(DT, self.tick)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def event(self, what, **kw):
            kw = {('t_detail' if k_ == 't' else k_): v_ for k_, v_ in kw.items()}   # 參數裡的 t 一律改名（c6、u7：撞名使監督端崩潰）
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
                    self.stale.append(dict(t=round(self.now(), 3), tag=tag, v=v, cur=self.ver, worker_ms=None if not isinstance(pay, dict) or not pay.get('tm') else round((pay['tm'][1] - pay['tm'][0]) * 1e3, 1),
                                           info=pay.get('info') if isinstance(pay, dict) else str(pay)[:200], stop_ms=(pay.get('stop') or {}).get('ms') if isinstance(pay, dict) else None))
            return 'late', None

        def _nav_in(self, m):
            if self.owner != 'NAV':
                return                                                  # 交權後導航 GMPC 的命令不再轉送（命令來源唯一）
            if self.stopped is not None:
                m = Twist()
            self.p_nav.publish(m); self.n_fwd += 1

        def _grasp_in(self, m):                                        # 抓持／開門控制器的底盤提案：第一筆到達即換擁有者（之後只轉送它）
            if self.ho_sent is None:
                return
            if self.owner != 'GRASP':
                self.owner = 'GRASP'; self.event('owner_grasp')
            self.p_nav.publish(m); self.n_grasp_fwd += 1

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
                self.obs_T.append((rec['t_obs'], T_oh.copy()))           # 交權時的參考融合用（全部有效影格）
                allow = self.mode in ('NAV', 'PREP', 'APPROACH') and (self.ho_sent is None or (UNIFIED and getattr(self, 'app_sent', None) is None))
                tag, pay = self.ask('obs', dict(seq=rec['seq'], t_obs=rec['t_obs'], t=self.now(), T_odom_handle=T_oh.tolist(), T_mo=T_mo.tolist(), allow_update=allow), 0.5)
                rec['est'] = None if pay is None else {k: pay.get(k) for k in ('n_hist', 'spread_mm', 'assoc_ok', 'estimate_set')}
                if tag == 'obs' and pay.get('estimate_set'):
                    self.T_est = pay['T_est']; self.plan_pending = True; self.T_est_tobs = pay.get('t_obs')
                    self.ref_cand = dict(T=pay['T_est'], t_obs=pay.get('t_obs'), seqs=pay.get('seqs'))   # 新的近距參考候選：下一次直接接近重算通過（含可達、連續、停止核對）才提交
                    self.event('estimate_set', **{k: pay.get(k) for k in ('seqs', 'assoc', 'spread_mm', 't_obs')})

        def tf2d(self, parent_, child):
            try:
                tr = self.tfb.lookup_transform(parent_, child, rclpy.time.Time())
            except Exception:                                          # noqa: BLE001
                return None
            o = tr.transform.rotation
            return np.array([tr.transform.translation.x, tr.transform.translation.y, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))])

        def _cyl_in(self, m):
            d = np.asarray(m.data, float)
            if len(d) < 10 or d[2] < 0.5 or not np.isfinite(d[6:9]).all():
                return
            p_od = (_T3(d[3:6]) @ np.r_[d[6:9], 1.0])[:3]             # 擷取時刻底盤（odom）×底盤座標圓柱中心
            if len(d) >= 11 and np.isfinite(d[10]):                   # 抽屜筆：第 11 欄＝筆軸偏航（底盤座標）⇒ 里程座標
                p_od = np.r_[p_od, (float(d[10]) + float(d[5]) + math.pi / 2) % math.pi - math.pi / 2]
            self.cyl_obs.append((float(d[9]), p_od))
            self.cyl_obs = self.cyl_obs[-30:]

        def _leaf_in(self, m):
            d = np.asarray(m.data, float)
            if len(d) < 7 or d[2] < 0.5 or not np.isfinite(d[6]):
                return
            self.leaf_obs.append((float(d[1]), float(d[6])))
            self.leaf_obs = self.leaf_obs[-30:]

        def _scan_in(self, m):
            self.last_scan = (np.asarray(m.ranges, float), float(m.angle_min), float(m.angle_increment))

        def _scan_pts_od(self, r_max=4.0):
            """最新一筆雷射轉里程座標點（雷射在底盤中心，geom.json lidar_xy＝0）。"""
            ls = getattr(self, 'last_scan', None)
            if ls is None or getattr(self, 'b_od_now', None) is None:
                return None
            r, a0, da = ls
            b = np.asarray(self.b_od_now, float)
            k = np.nonzero(np.isfinite(r) & (r < r_max))[0]
            th = b[2] + a0 + k * da
            return np.c_[b[0] + r[k] * np.cos(th), b[1] + r[k] * np.sin(th)]

        def _door_geom_od(self):
            """門幾何（里程座標）：場景登錄鉸鏈＋門扇方向（同去程接近封包的 app['door']）；回程關門綁定共轉中門扇點不進 CBF（s97d：門扇被追蹤成迎面動態障礙 ⇒ CBF 把底盤往反方向推、門又被拉開）。"""
            try:
                T_mo = getattr(self, 'T_mo_now', None)
                if T_mo is None:
                    return None
                sreg_ = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
                ox_ = sreg_['door']['opening_x']
                Ti_ = np.linalg.inv(_T3(np.asarray(T_mo, float)))
                hg_ = Ti_ @ np.array([float(ox_[0]), 0.0, 0.0, 1.0]); dr_ = Ti_[:3, :3] @ np.array([1.0, 0.0, 0.0])
                return dict(hinge=[float(hg_[0]), float(hg_[1])], dir=[float(dr_[0]), float(dr_[1])], w=0.95, src='場景登錄（同去程）')
            except Exception:                                             # noqa: BLE001
                return None

        def _leaf_fuse(self, t):
            rec = [p_ for t_, p_ in self.leaf_obs if t - t_ <= 1.5][-5:]
            if len(rec) < 5 or float(np.ptp(rec)) > math.radians(0.5):
                return None
            return float(np.mean(rec))

        def _corot(self, b, dphi):
            c_, s_ = math.cos(-dphi), math.sin(-dphi)
            return np.array([c_ * b[0] - s_ * b[1], s_ * b[0] + c_ * b[1], b[2] - dphi])

        def door_tick(self, t):
            """回程關門：到開門結束底盤位附近觀察門扇 ⇒ 融合門角 ⇒ 共轉抓持站位 ⇒ 近站位送關門封包 ⇒ 關好收臂後回起點。回 True＝仍在關門流程。"""
            dc = self.dc
            if dc['stage'] is None:
                if self.door_end is None:
                    self.event('door_close_no_end_state'); dc['stage'] = 'skip'; return False
                dc.update(stage='seek', t0=t)
                self.pick_goal, self.pass_t = self._dr_goal(self.door_end['b']), t
                self.dr['goal_od'] = [float(x) for x in self.door_end['b']]   # 目標每拍重新換算用（否則沿用抽屜目標）
                if PASS_CLEAR and self._clr_cli.service_is_ready():     # 回程穿門前清一次全域代價圖（同穿門、返回）
                    self._clr_fut, self._clr_t = self._clr_cli.call_async(self._clr_req()), t
                self.event('door_close_seek', goal=[round(float(x), 4) for x in self.door_end['b']])
                return True
            if dc['stage'] == 'seek':
                ph = self._leaf_fuse(t)
                if ph is not None:
                    b_c = self._corot(self.door_end['b'], ph - self.door_end['phi'])
                    dc.update(stage='nav', phi_c=ph, b_c=b_c)
                    self.pick_goal, self.pass_t = self._dr_goal(b_c), t
                    self.dr['goal_od'] = [float(x) for x in b_c]
                    self.event('door_leaf_vision', phi_deg=round(math.degrees(ph), 3), b_c=[round(float(x), 4) for x in b_c], n_obs=len(self.leaf_obs))
                elif t - dc['t0'] > 90.0:
                    self.event('door_leaf_not_seen'); self.abort('door_leaf_not_seen')
                return True
            if dc['stage'] == 'nav':
                b_c = dc['b_c']
                dist_ = float(np.hypot(*(self.b_od_now[:2] - b_c[:2])))
                dyaw_ = abs((float(self.b_od_now[2]) - float(b_c[2]) + math.pi) % (2 * math.pi) - math.pi)
                if dist_ <= PICK_APP_DIST and (dyaw_ <= 0.30 or dist_ <= 0.15):
                    ph = self._leaf_fuse(t) or dc['phi_c']                # 送封包時以最新融合門角
                    self.p_ho.publish(String(data=json.dumps(dict(what='pick_app', app=dict(kind='door_close', door=self._door_geom_od(), phi_c=float(ph), base_goal=[float(x) for x in b_c], T=max(PICK_T_MIN, dist_ / 0.12 + 2.0), T_b=2.5, t_l0=1.5)))))
                    dc['stage'] = 'closing'
                    self.event('door_close_app_sent', phi_deg=round(math.degrees(ph), 3), dist=round(dist_, 3))
                return True
            if dc['stage'] == 'closing':
                return not dc.get('done')
            return False

        def _hdl_in(self, m):
            d = np.asarray(m.data, float)
            if len(d) < 9 or d[2] < 0.5 or not np.isfinite(d[6:9]).all():
                return
            self.hdl_obs.append((float(d[1]), d[6:9].copy()))
            self.hdl_obs = self.hdl_obs[-30:]

        def _hdl_fuse(self, t):
            rec = [p_ for t_, p_ in self.hdl_obs if t - t_ <= 1.5][-5:]
            if len(rec) < 5:
                return None
            P_ = np.array(rec)
            yw = np.unwrap(P_[:, 2])
            if float(np.linalg.norm(P_[:, :2] - P_[:, :2].mean(0), axis=1).max()) > 0.010 or float(np.ptp(yw)) > math.radians(0.5):
                return None
            return np.r_[P_[:, :2].mean(0), float(yw.mean())]

        def _cyl_fuse(self, t):
            rec = [p_ for t_, p_ in self.cyl_obs if t - t_ <= 1.0][-5:]
            if len(rec) < 5:
                return None
            P_ = np.array(rec)
            if float(np.linalg.norm(P_ - P_.mean(0), axis=1).max()) > 0.010:
                return None
            return P_.mean(0)

        def _xs_in(self, m):
            if '"what": "pick_lifted"' in m.data:
                self.pick_lifted = True
            if '"what": "pick_binned"' in m.data:                      # 收納盒（每件一次）
                self.n_binned = getattr(self, 'n_binned', 0) + 1
                if DRAWER and self.dr.get('stage') == 'marker':
                    if DRAWER_CLOSE:
                        self.dr['stage'] = 'close_wait'                 # 關抽屜：等收臂完成再送關抽屜封包
                    else:
                        self.dr['carried'] = True                      # 筆已放盒 ⇒ 送返回目標（收臂中底盤照導航前端）
            if '"what": "pick_stowed"' in m.data:
                self.n_stowed = getattr(self, 'n_stowed', 0) + 1
                if DRAWER and self.dr.get('stage') == 'close_wait':
                    self.dr['stage'] = 'close_send'
                elif DRAWER and self.dr.get('stage') == 'closing' and self.dr.get('closed'):
                    self.dr['carried'] = True                          # 抽屜已關、收臂完成 ⇒ 返回
            if '"what": "pick_dr_closed"' in m.data:
                self.dr['closed'] = True
            if DRAWER and not BIN and '"what": "pick_dr_carried"' in m.data:   # 抽屜：收臂完成才送返回目標（s50：抬升後立即送 ⇒ 前端障礙物 3→6、CBF 推動錨定底盤 3.8 cm ⇒ 夾爪被帶向面板）
                self.dr['carried'] = True
            for k_ in ('pick_dr_released', 'pick_dr_observing'):      # 抽屜任務事件（帶開度估計）
                if ('"what": "%s"' % k_) in m.data and k_ not in self.dr:
                    try:
                        self.dr[k_] = json.loads(m.data)
                    except Exception:                                  # noqa: BLE001
                        self.dr[k_] = dict(what=k_)
            self.xstate = m.data
            if '"what": "pass_start"' in m.data and self.door_end is None:   # 開門結束構型（回程關門共轉起算）
                try:
                    d_ = json.loads(m.data)
                    if d_.get('base_end') is not None:
                        self.door_end = dict(b=np.asarray(d_['base_end'], float), phi=math.radians(float(d_['phi_end_deg'])))
                except Exception:                                      # noqa: BLE001
                    pass
            if '"what": "pick_door_closed"' in m.data:
                self.dc['closed'] = True
            if '"what": "pick_stowed"' in m.data and self.dc.get('closed'):
                self.dc['done'] = True
            if '"what": "pass_start"' in m.data and not self.pass_seen:   # 開門完成、同一控制器接穿門：原導航鏈送穿門目標
                self.pass_seen = True

        def pick_tick(self, t):
            if CHAIN and getattr(self, 'chain', 'cyl') == 'cyl' and (getattr(self, 'n_binned', 0) >= 1 or (PICK_MOTM_V is not None and self.pick_lifted)):   # 移動中夾取：抬起確認即換下一目標（底盤不折回路線起點）   # 串接：圓柱已放盒 ⇒ 換抽屜任務（導航目標換抽屜櫃粗目標；選位由 drawer_tick 接手）
                self.chain, self.pick_goal, self.pass_t = 'drawer', None, t
                self.event('chain_to_drawer')
                return
            if self.pick_lifted and not self.pick_ret and (not BIN or getattr(self, 'n_binned', 0) >= 1) and not CHAIN:   # 已抬起且證據成立（收納盒：已放盒）⇒ 返回起點（經開著的門）
                self.pick_ret, self.pick_goal, self.pass_t = True, list(PICK_RETURN), t
                self.event('pick_return_goal', goal=PICK_RETURN)
                return
            if self.pick_sel is None and self.pick_th is None:
                est = self._cyl_fuse(t)
                if est is not None:
                    Tm_ = _T3(self.T_mo_now) @ np.r_[est, 1.0]
                    c_map = Tm_[:3]; b_now = self.b_od_now.copy()      # 選位在里程（世界）座標：家具真實位置與控制器淨空核對都在這個座標（第十三趟在 map 選位 ⇒ AMCL 偏 3 cm、底盤淨空只剩 101 mm）
                    self.pick_box = dict(c_map=c_map, est_od=est, t=t)
                    import threading
                    CR = self._CR
                    n_unk_ = CR.set_scan_pts(self._scan_pts_od()) if PICK_MOTM_V is not None else 0
                    self.event('pick_scan_unknown_pts', n=int(n_unk_))
                    def _sel(box=self.pick_box, c=est, b=b_now):
                        w0 = time.perf_counter()
                        box['sel'] = CR.select(c[:2], b) if PICK_MOTM_V is None else CR.select_motm(c[:2], b, PICK_MOTM_V, goal_next=np.asarray(DR_PRIOR[:2], float) if CHAIN else None)
                        box['ms'] = (time.perf_counter() - w0) * 1e3; box['done'] = True
                    self.pick_th = threading.Thread(target=_sel, daemon=True); self.pick_th.start()
                    self.event('pick_estimate', c_map=[round(float(x), 4) for x in c_map], n_obs=len(self.cyl_obs))
            if self.pick_th is not None and self.pick_box.get('done') and self.pick_sel is None:
                sel = self.pick_box['sel']; self.pick_th = None
                if sel is None:
                    self.event('pick_unreachable', c_map=[round(float(x), 4) for x in self.pick_box['c_map']], ms=round(self.pick_box['ms'], 1))
                    self.abort('pick_unreachable'); return
                Tbm_ = _T3(self.T_mo_now) @ _T3(sel['base_start'] if PICK_MOTM_V is not None else sel['base'])   # 選位結果（里程）→ 地圖（只給 Smac）；移動中夾取＝路線起點
                self.pick_sel = sel; self.pick_goal = [float(Tbm_[0, 3]), float(Tbm_[1, 3]), math.atan2(Tbm_[1, 0], Tbm_[0, 0])]; self.pass_t = t
                self.event('pick_goal', base=[round(float(x), 4) for x in sel['base']], side=sel['side'], flip=sel['flip'], sigma_min=round(sel['sigma_min'], 4), ms=round(self.pick_box['ms'], 1))
            if self.pick_sel is not None and self.pick_app_t is None and PICK_MOTM_V is not None:   # 移動中夾取：近路線起點即送封包（底盤由控制器沿路線帶過抓取位，不停）
                bs_ = np.asarray(self.pick_sel['base_start'], float)
                dist_ = float(np.hypot(*(self.b_od_now[:2] - bs_[:2])))
                dyaw_c_ = abs((float(self.b_od_now[2]) - float(bs_[2]) + math.pi) % (2 * math.pi) - math.pi)
                mv_ = getattr(self, '_m_still', None)                  # 底盤靜止計時（導航以容差到點而偏航未齊時的備用：s70 卡住）
                if mv_ is None or float(np.hypot(*(self.b_od_now[:2] - mv_[0][:2]))) > 0.005 or abs(float(self.b_od_now[2] - mv_[0][2])) > 0.01:
                    self._m_still = (self.b_od_now.copy(), t); mv_ = self._m_still
                still_ = t - mv_[1] >= 1.5
                if dist_ <= PICK_APP_DIST and (PICK_APP_DYAW is None or dyaw_c_ <= PICK_APP_DYAW or (still_ and dist_ <= 0.5)):   # s75：距離備用在偏航差 3.3 rad 放行 ⇒ 掉頭擦桌角 ⇒ 移動中夾取只用偏航或靜止備用
                    Tg_od = np.asarray(self.pick_sel['T_grasp'], float)
                    Tg_od[2, 3] = float(self.pick_box['est_od'][2])
                    b0_ = [float(x) for x in self.pick_sel['base']]
                    app = dict(T_grasp=Tg_od.tolist(), base_goal=b0_, T=4.0, T_b=2.5, t_l0=1.5,
                               motm=dict(b0=b0_, e=[float(x) for x in self.pick_sel['e']], v_g=float(PICK_MOTM_V), s0=float(self._CR.MOTM_S0), t_arm=float(self._CR.MOTM_T_ARM), q_grasp=[float(x) for x in self.pick_sel['q']]))
                    self.p_ho.publish(String(data=json.dumps(dict(what='pick_app', app=app))))
                    self.pick_app_t = t
                    self.event('pick_app_sent', dist=round(dist_, 3), motm=app['motm'])
                return
            if self.pick_sel is not None and self.pick_app_t is None:
                b_ = np.asarray(self.pick_sel['base'], float)          # 里程座標
                dist_ = float(np.hypot(*(self.b_od_now[:2] - b_[:2])))
                dyaw_c_ = abs((float(self.b_od_now[2]) - float(b_[2]) + math.pi) % (2 * math.pi) - math.pi)
                if dist_ <= PICK_APP_DIST and (PICK_APP_DYAW is None or dyaw_c_ <= PICK_APP_DYAW or dist_ <= 0.35):   # s70：只留 ≤0.10 會在導航以容差到點、偏航未齊時卡住 ⇒ 恢復 s62–s67 版（0.35）；s68 邊緣案例改以選位餘裕 CYL_BASE_MARGIN 處理。舊註：s68：距離 ≤0.35 的備用條件在偏航差 1.22 rad 時放行 ⇒ 轉向中收斂斜向桌角、淨空 0.0992 停 ⇒ 只用偏航條件（導航到點時車頭自然對齊目標）
                    Tb_od = _T3(b_)                                    # 底盤收斂目標（里程，與選位同座標）
                    e_od = np.asarray(self.pick_box['est_od'], float)   # 抓取目標一律留在里程座標（不經 map 來回：AMCL 修正會讓 TCP 偏離圓柱；s32 偏約 3 cm 夾空）
                    a_ = e_od[:2] - Tb_od[:2, 3]; a_ = a_ / max(float(np.linalg.norm(a_)), 1e-9)
                    Tg_od = np.asarray(self._CR.grasp_T(e_od[:2], a_, bool(self.pick_sel['flip'])), float); Tg_od[2, 3] = float(e_od[2])
                    Tapp = max(PICK_T_MIN, dist_ / 0.12 + 2.0)
                    app = dict(T_grasp=Tg_od.tolist(), base_goal=[float(Tb_od[0, 3]), float(Tb_od[1, 3]), math.atan2(Tb_od[1, 0], Tb_od[0, 0])], T=Tapp, T_b=max(1.5, Tapp - 1.5), t_l0=max(0.5, Tapp - 2.5))
                    self.p_ho.publish(String(data=json.dumps(dict(what='pick_app', app=app))))
                    self.pick_app_t = t
                    self.event('pick_app_sent', dist=round(dist_, 3), T=round(Tapp, 3))

        def _dr_thread(self, fn, *args):
            import threading
            box = dict(done=False)

            def _run(box=box):
                w0 = time.perf_counter()
                try:
                    box['sel'] = fn(*args)
                except Exception as e:                                 # noqa: BLE001
                    box['sel'], box['err'] = None, repr(e)[:300]
                box['ms'] = (time.perf_counter() - w0) * 1e3; box['done'] = True
            th = threading.Thread(target=_run, daemon=True); th.start()
            return box

        def _dr_goal(self, b_od):
            Tbm_ = _T3(self.T_mo_now) @ _T3(np.asarray(b_od, float))   # 選位（里程）→ 地圖（只給 Smac）
            g_ = [float(Tbm_[0, 3]), float(Tbm_[1, 3]), math.atan2(Tbm_[1, 0], Tbm_[0, 0])]
            pg_ = self.pick_goal
            near_robot_ = getattr(self, 'cur_map', None) is not None and float(np.hypot(g_[0] - self.cur_map[0], g_[1] - self.cur_map[1])) < 0.08   # s45：AMCL 漂移使新舊目標差 5.8 cm 仍重送 ⇒ 起點≈終點 Smac no_plan
            if pg_ is not None and (near_robot_ or float(np.hypot(g_[0] - pg_[0], g_[1] - pg_[1])) < 0.10) and abs((g_[2] - pg_[2] + math.pi) % (2 * math.pi) - math.pi) < 0.1:
                self.event('dr_goal_kept', new=[round(x, 4) for x in g_], kept=[round(float(x), 4) for x in pg_])   # s43：新目標離現目標 1.5 cm 仍重送 ⇒ 前端重規劃 >0.3 s 無輸出 ⇒ 控制器判過期停車；近距由控制器收斂到封包底盤位
                return pg_
            return g_

        def _dr_send(self, kind, base, t, **kw):
            dist_ = float(np.hypot(*(self.b_od_now[:2] - np.asarray(base[:2], float))))
            Tapp = max(PICK_T_MIN, dist_ / 0.12 + 2.0)
            app = dict(kind=kind, base_goal=[float(x) for x in base[:3]], T=Tapp, T_b=max(1.5, Tapp - 1.5), t_l0=max(0.5, Tapp - 2.5), **kw)
            self.p_ho.publish(String(data=json.dumps(dict(what='pick_app', app=app))))
            self.event('dr_app_sent', kind=kind, dist=round(dist_, 3), T=round(Tapp, 3), base=[round(float(x), 4) for x in base[:3]])

        def _mk_fuse(self, t, t_min):
            rec = [p_ for t_, p_ in self.cyl_obs if t - t_ <= 1.0 and t_ >= t_min and len(p_) >= 4][-5:]
            if len(rec) < 5:
                return None
            P_ = np.array(rec)
            if float(np.linalg.norm(P_[:, :3] - P_[:, :3].mean(0), axis=1).max()) > 0.010:
                return None
            yw = P_[:, 3]; yw = np.arctan2(np.sin(2 * yw).mean(), np.cos(2 * yw).mean()) / 2.0   # 軸向（模 π）平均
            return np.r_[P_[:, :3].mean(0), yw]

        def drawer_tick(self, t):
            """抽屜任務：選把手位 → 近距送抽屜封包 → 拉開放手後選觀察位 → 俯視融合筆觀測 → 選夾筆位 → 抬起後返回起點。"""
            dr, R_ = self.dr, self._DRR
            if dr.get('carried') and not self.pick_ret and DOOR_CLOSE and self.door_tick(t):   # 回程關門流程中（目標由 door_tick 設）
                return
            if dr.get('carried') and not self.pick_ret:
                self.pick_ret, self.pick_goal, self.pass_t = True, list(PICK_RETURN), t
                self.event('pick_return_goal', goal=PICK_RETURN)
                return
            st = dr['stage']
            if HANDLE_VISION and not dr.get('pose_vis'):              # 先看把手：導航到櫃前先驗點，融合把手深度偵測 ⇒ 以視覺位姿重設抽屜模型（選位、淨空、內部範圍都依此）
                est = self._hdl_fuse(t)
                if est is None:
                    dr.setdefault('t_seek', t)
                    if t - dr['t_seek'] > float(os.environ.get('DR_SEEK_TO_S', '90')):   # s104d：會車繞行吃掉時間（60→90 s）
                        self.event('dr_handle_not_seen'); self.abort('drawer_handle_not_seen')
                    return
                reg_ = R_.D.REG['drawer']['pose']
                R_.D.set_pose(float(est[0]), float(est[1]), float(est[2]))
                dr['pose_vis'] = [float(x) for x in est]
                self.event('dr_handle_vision', pose=[round(float(est[0]), 4), round(float(est[1]), 4), round(math.degrees(float(est[2])), 3)],
                           reg=[float(reg_[0]), float(reg_[1]), float(reg_[2])], n_obs=len(self.hdl_obs))
                return
            if st == 'handle_sel' and dr['box'] is None:
                dr['box'] = self._dr_thread(R_.select_handle, self.b_od_now.copy()) if DR_MOTM_V is None else self._dr_thread(self._DRM.select_handle_motm, self.b_od_now.copy(), DR_MOTM_V)
                self.event('dr_handle_select_start')
            elif st == 'handle_sel' and dr['box']['done']:
                sel = dr['box']['sel']
                if sel is None:
                    self.event('dr_handle_unreachable', err=dr['box'].get('err')); self.abort('drawer_handle_unreachable'); return
                dr.update(stage='handle_nav', sel=sel, box=None)
                g_od_ = sel['base_start'] if DR_MOTM_V is not None else sel['base']   # 移動中：導航到經過路線起點
                self.pick_goal, self.pass_t = self._dr_goal(g_od_), t
                dr['goal_od'] = list(g_od_)
                self.event('dr_handle_goal', base=[round(float(x), 4) for x in sel['base']], start=[round(float(x), 4) for x in g_od_], flip=sel['flip'], sigma_min=round(sel['sigma_min'], 4),
                           env=round(sel.get('env', sel.get('env_min', 0.0)), 4), motm=DR_MOTM_V is not None, ms=round(dr['box']['ms'] if dr['box'] else 0, 1))
            elif st == 'handle_nav' and CHAIN and getattr(self, 'n_stowed', 0) < 1:
                pass                                                   # 串接：圓柱放盒後收臂未完成 ⇒ 先不送抽屜封包（執行端只在收臂完成後接受）
            elif st == 'handle_nav' and DR_MOTM_V is not None:         # 移動中抓把手：近路線起點、偏航齊（或已停 1.5 s）⇒ 送封包（控制器沿路線帶過把手）
                bs_ = np.asarray(dr['sel']['base_start'], float)
                dist_ = float(np.hypot(*(self.b_od_now[:2] - bs_[:2])))
                dyaw_ = abs((float(self.b_od_now[2]) - float(bs_[2]) + math.pi) % (2 * math.pi) - math.pi)
                mv_ = getattr(self, '_dr_still', None)
                if mv_ is None or float(np.hypot(*(self.b_od_now[:2] - mv_[0][:2]))) > 0.005 or abs(float(self.b_od_now[2] - mv_[0][2])) > 0.01:
                    self._dr_still = (self.b_od_now.copy(), t); mv_ = self._dr_still
                if dist_ <= PICK_APP_DIST and (dyaw_ <= DR_APP_DYAW or (t - mv_[1] >= 1.5 and dist_ <= 0.5)):
                    sel = dr['sel']
                    self._dr_send('drawer', bs_, t, T_grasp=sel['T_grasp'], q_grasp=sel.get('q_grasp'), pull=[float(R_.D.PULL[0]), float(R_.D.PULL[1])], open_m=float(R_.OPEN_M), drawer_pose=dr.get('pose_vis'),
                                  motm=dict(b0=[float(x) for x in sel['base']], e=[float(x) for x in sel['e']], v_g=float(DR_MOTM_V), s0=float(self._DRM.S0), t_arm=float(sel['t_arm']), tau=float(sel['tau']), q_grasp=sel.get('q_grasp')))
                    dr['stage'] = 'wait_release'
            elif st == 'handle_nav':
                dyaw_ = abs((float(self.b_od_now[2]) - float(dr['sel']['base'][2]) + math.pi) % (2 * math.pi) - math.pi)
                if float(np.hypot(*(self.b_od_now[:2] - np.asarray(dr['sel']['base'][:2], float)))) <= PICK_APP_DIST and (dyaw_ <= DR_APP_DYAW or float(np.hypot(*(self.b_od_now[:2] - np.asarray(dr['sel']['base'][:2], float)))) <= 0.35):   # s51：偏航差 0.36 rad 時開始接近 ⇒ 手腕 j5 −1.76、自身淨空 4.6 mm
                    self._dr_send('drawer', dr['sel']['base'], t, T_grasp=dr['sel']['T_grasp'], q_grasp=dr['sel'].get('q_grasp'), pull=[float(R_.D.PULL[0]), float(R_.D.PULL[1])], open_m=float(R_.OPEN_M), drawer_pose=dr.get('pose_vis'))
                    dr['stage'] = 'wait_release'
            elif st == 'close_send':                                   # 關抽屜封包：拉開終點底盤位、目前開度的把手位姿、推回方向
                qo_ = float(dr.get('q_open') or R_.OPEN_M)
                b1_ = np.asarray(dr['sel']['base'], float) + np.r_[qo_ * R_.D.PULL[:2], 0.0]
                R_.D.set_open(qo_)
                Tg_ = np.asarray(R_.D.handle_T(qo_, bool(dr['sel'].get('flip', False))), float)
                if dr['sel'].get('T_grasp') is not None:               # 選位時的抓取位姿（含把手越過量）沿拉出方向平移開度
                    Tg_ = np.asarray(dr['sel']['T_grasp'], float).copy(); Tg_[:3, 3] = Tg_[:3, 3] + qo_ * np.asarray(R_.D.PULL, float)
                self.pick_goal, self.pass_t = self._dr_goal(b1_), t
                dr['goal_od'] = list(b1_)
                self._dr_send('dr_close', b1_, t, T_grasp=Tg_.tolist(), pull=[float(R_.D.PULL[0]), float(R_.D.PULL[1])], q_open=qo_)
                dr['stage'] = 'closing'
                self.event('dr_close_sent', q_open=round(qo_, 4), base=[round(float(x), 4) for x in b1_])
            elif st == 'wait_release' and 'pick_dr_released' in dr:
                dr['q_open'] = float(dr['pick_dr_released'].get('q_open', R_.OPEN_M))
                dr['box'] = self._dr_thread(R_.select_observe, dr['q_open'], self.b_od_now.copy(), bool(dr['sel']['flip']))
                dr['stage'] = 'obs_sel'
                self.event('dr_observe_select_start', q_open=dr['q_open'])
            elif st == 'obs_sel' and dr['box']['done']:
                sel = dr['box']['sel']
                if sel is None:
                    self.event('dr_observe_unreachable', err=dr['box'].get('err')); self.abort('drawer_observe_unreachable'); return
                g_new_ = self._dr_goal(sel['base'])
                if g_new_ is not self.pick_goal:                       # 目標保留 ⇒ 不重送（s46：同一目標因 pass_t 更新被重送 ⇒ no_plan）
                    self.pick_goal, self.pass_t = g_new_, t
                    dr['goal_od'] = list(sel['base'])
                self.event('dr_observe_goal', base=[round(float(x), 4) for x in sel['base']], c=[round(float(x), 4) for x in sel['c']], ms=round(dr['box']['ms'], 1))
                self._dr_send('observe', sel['base'], t, T_obs=sel['T_obs'], q_obs=sel['q_obs'])
                dr['R_obs'] = np.asarray(sel['T_obs'], float)[:3, :3]
                dr['T_obs'] = np.asarray(sel['T_obs'], float)
                dr.update(stage='wait_obs', box=None)
            elif st == 'wait_obs' and 'pick_dr_observing' in dr:
                dr.update(stage='mk_fuse', t_obs=t)
            elif st == 'mk_fuse':
                est = self._mk_fuse(t, dr['t_obs'])
                if est is not None:
                    dr['est'] = est
                    dr['box'] = self._dr_thread(R_.select_marker, est[:3], float(est[3]), dr['q_open'], self.b_od_now.copy(), 1.0, 0.02, dr.get('R_obs'), dr.get('T_obs'))   # 接近路徑檢查（s88d）
                    dr['stage'] = 'mk_sel'
                    self.event('dr_marker_estimate', c_od=[round(float(x), 4) for x in est[:3]], yaw_deg=round(math.degrees(float(est[3])), 2), n_obs=len(self.cyl_obs))
                elif t - dr['t_obs'] > 10.0:
                    self.event('dr_marker_not_seen'); self.abort('drawer_marker_not_seen'); return
            elif st == 'mk_sel' and dr['box']['done']:
                sel = dr['box']['sel']
                if sel is None:
                    self.event('dr_marker_unreachable', err=dr['box'].get('err')); self.abort('drawer_marker_unreachable'); return
                g_new_ = self._dr_goal(sel['base'])
                if g_new_ is not self.pick_goal:                       # 目標保留 ⇒ 不重送（s46：同一目標因 pass_t 更新被重送 ⇒ no_plan）
                    self.pick_goal, self.pass_t = g_new_, t
                    dr['goal_od'] = list(sel['base'])
                self.event('dr_marker_goal', base=[round(float(x), 4) for x in sel['base']], flip=sel['flip'], sigma_min=round(sel['sigma_min'], 4), env=round(sel['env'], 4), ms=round(dr['box']['ms'], 1))
                self._dr_send('marker', sel['base'], t, T_grasp=sel['T_grasp'])
                dr.update(stage='marker', box=None)

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
            if float(v_along) <= 0.0 and float(v_norm) <= 0.0:
                return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]              # 停止：三軸全零（e14：角速度未限使底盤持續慢轉、停穩判定不成立）
            c, s_ = math.cos(-yaw_now), math.sin(-yaw_now)
            db = np.array([c * d_world[0] - s_ * d_world[1], s_ * d_world[0] + c * d_world[1]])
            i = int(np.argmax(np.abs(db)))
            if abs(db[i]) >= math.cos(math.radians(15.0)):
                va = max(float(v_along), 0.0)
                rv = 0.02 if va > 0.0 else 0.0                          # 停止（沿門上限 0）時不留倒退餘量
                box = [None, None, [-0.6, 0.6]]
                box[i] = [-rv, va] if db[i] > 0 else [-va, rv]
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

        def ref_to_map(self, xy):
            T_ = _T3(self.T_mo_now) @ np.linalg.inv(_T3(self.T_ref_now))
            return (T_ @ np.r_[xy[0], xy[1], 0.0, 1.0])[:2].tolist()

        def map_to_ref(self, x, y, th):
            T_ = _T3(self.T_ref_now) @ np.linalg.inv(_T3(self.T_mo_now))
            p_ = T_ @ np.r_[x, y, 0.0, 1.0]
            return float(p_[0]), float(p_[1]), float(th + self.T_ref_now[2] - self.T_mo_now[2])

        def base_prediction(self, v_tr, T_max=12.0):
            """底盤預測（地圖；自現在起每 0.05 s，絕對模擬時間）：GMPC 1 s 預測（同座標、0.3 s 內）→ 之後沿走廊依候選速度剖面外推（側向誤差 0.5 s 指數收斂、朝向以 0.3 rad/s 轉到任務朝向）。只供搜尋候選，執行另以逐拍核對。"""
            rg, pl = self.region, self.plan
            p0 = np.array(rg['line'][0]); p1 = np.array(rg['line'][1]); u = (p1 - p0) / np.linalg.norm(p1 - p0)
            nrm = np.array([-u[1], u[0]])
            pts = []
            if self.pred is not None and self.now() - self.pred['t'] <= 0.3:
                for k, P in enumerate(self.pred['P']):                    # GMPC 預測在 map（當下定位）→ 規劃座標
                    pts.append((self.pred['t'] + k * self.pred['dt'], *self.map_to_ref(P[0], P[1], P[2])))
            else:
                b = self.cur_map
                pts.append((self.now(), b[0], b[1], b[2]))
            t_e, x, y, th = pts[-1]
            dt = 0.05
            if len(pts) >= 2:                                          # 外推起始速度＝GMPC 預測末段沿走廊的速度（不直接跳到速度框上限）
                v_prev = max(float((np.array(pts[-1][1:3]) - np.array(pts[-2][1:3])) @ u) / max(pts[-1][0] - pts[-2][0], 1e-3), 0.0)
            else:
                v_prev = max(float(np.asarray(getattr(self, 'v_w_last', np.zeros(2))) @ u), 0.0)
            while t_e < self.now() + T_max:
                rel = np.array([x, y]) - p0
                s_, lat = float(rel @ u), float(rel @ nrm)
                a_door = float(-((np.array([x, y]) - np.array(self.h_xy)) @ np.array(pl['d'])))
                v = self.v_profile(a_door, v_tr, adopted=True)
                if v_prev is not None:
                    v = max(min(v, v_prev + 0.1 * dt), v_prev - 0.5 * dt)   # 自實測速度起：緩升 0.1 m/s²（GMPC 實際未必到速度框上限）、減速 0.5 m/s²
                v_prev = v
                s_ += v * dt; lat *= math.exp(-dt / 0.5)
                xy = p0 + s_ * u + lat * nrm
                th = th + max(-0.3 * dt, min(0.3 * dt, math.atan2(math.sin(rg['heading'] - th), math.cos(rg['heading'] - th))))
                t_e += dt; x, y = float(xy[0]), float(xy[1])
                pts.append((t_e, x, y, th))
            return np.array(pts)

        def v_profile(self, a_door, v_tr=None, adopted=False):
            pl = self.plan
            v_tr = self.v_active if v_tr is None else v_tr
            aw = self.dp['aw'] if getattr(self, 'dp', None) is not None else float(pl['a_w'])
            if a_door <= aw:
                return 0.05
            v = min(V_APPROACH, float(v_tr), math.sqrt(0.05 ** 2 + 2 * float(pl['a_dec']) * max(a_door - aw, 0.0)))
            if self.mode == 'APPROACH':     # 與速度框同式：尚無計畫時依手臂所需時間限速（候選預測假設採用後的剖面 ⇒ adopted=True）
                T_need = RETR_LEAD + DP_CANDS[0][0] + DP_CANDS[0][2] + T_NEED_MARGIN
                v = min(v, max(0.05, (a_door - float(pl['a_w'])) / T_need))
            return v

        def tick(self):
            t = self.now()
            if self.truth is None or self.aborted is not None:
                return
            T_mo = self.tf2d('map', 'odom'); b_od = self.tf2d('odom', 'base_footprint')
            if T_mo is None or b_od is None:
                return
            q = np.r_[b_od, self.truth['q'][3:9]]
            self.b_od_now = np.asarray(b_od, float).copy()
            self.T_mo_now = T_mo
            T_ref = self.T_ref if getattr(self, 'T_ref', None) is not None else T_mo
            c, s_ = math.cos(T_ref[2]), math.sin(T_ref[2])
            self.cur_map = np.r_[c * b_od[0] - s_ * b_od[1] + T_ref[0], s_ * b_od[0] + c * b_od[1] + T_ref[1], b_od[2] + T_ref[2]]
            self.T_ref_now = np.asarray(T_ref, float)
            if self.mode == 'INIT':
                tag, pay = self.ask('prior', {}, 5.0)
                if tag != 'prior':
                    self.abort('prior_failed', detail=str(pay)[:200]); return
                self.prior = pay; self.event('prior', **{k: v for k, v in pay.items() if k != 'T_prior'})
                self.goal_xy = pay['goal_xy']; self.t_goal_set = t; self.pub_goal(self.goal_xy, pay['yaw']); self.t_goal_pub = t
                tag, pay2 = self.ask('off_anchor', dict(q=q.tolist(), s=self.truth['s'].tolist()), 2.0)
                pk.send(('off_anchor', dict(ver=0, q=q.tolist(), s=self.truth['s'].tolist()))); tp_ = pk.recv()
                self.event('off_anchor', ok=tag == 'off_anchor', planner=tp_[0])
                self.mode = 'NAV'; self.event('mode', mode='NAV')
                if UNIFIED:                                               # 起點交權：自此命令只由單一控制器出（手臂保持、底盤追導航前端參考）
                    n_h = int(round(t / PHYS)) + NPH + 3
                    T_pr = np.linalg.inv(_T3(T_mo)) @ np.asarray(self.prior['T_prior'], float)
                    msg = dict(n_h=n_h, ins=dict(s0=0.15, v0=0.0, a0=0.0, T=1e6), unified=True, T_ref=T_pr.tolist(), t_obs=t, ref_ver=0, win=[-1e9, 1e9])
                    self.p_ho.publish(String(data=json.dumps(msg)))
                    self.ho_sent = dict(t_sent=round(t, 3), n_h=n_h, unified=True)
                    self.app_sent, self.goal_ver = None, None
                    self.event('handover_requested', **self.ho_sent)
                return
            if self.pass_seen:                                       # 穿門段（D2_PASS）：撤銷任務條件（朝向／速度框／區域），目標＝門後客廳點；Smac 照原鏈規劃（全域代價圖含 LiDAR，門扇自然繞開）
                if self.pass_t is None:
                    self.pass_t = t
                    self.p_task.publish(String(data=json.dumps(dict(clear=True))))
                    self.event('pass_goal', goal=PASS_GOAL)
                    if PASS_CLEAR:                                    # 機器人自己開的門：關門時標在門口的障礙格已確定過期 ⇒ 呼叫 Nav2 標準復原（清除全域代價圖）一次；看得到的門扇由 LiDAR 下一次更新重新標上
                        if self._clr_cli.service_is_ready():
                            self._clr_fut, self._clr_t = self._clr_cli.call_async(self._clr_req()), t
                            self.event('pass_costmap_clear_called')
                        else:
                            self.event('pass_costmap_clear_unavailable')
                if PASS_CLEAR and self._clr_t is not None and t - self._clr_t < PASS_CLEAR_WAIT_S:
                    return                                            # 等障礙層重新標記（全域代價圖 1 Hz）再送目標
                g_ = PASS_WP if (PASS_WP is not None and not getattr(self, 'pass_wp_done', False)) else PASS_GOAL
                if g_ is PASS_WP and float(np.hypot(*(self.cur_map[:2] - np.asarray(PASS_WP[:2])))) <= PASS_WP_R:   # 中途點到了（門口已在 LiDAR 視野、舊標記已清）⇒ 改送穿門目標
                    self.pass_wp_done, self.pass_t = True, t
                    self.event('pass_waypoint_reached', at=[round(float(x), 3) for x in self.cur_map])
                    g_ = PASS_GOAL
                if PICK and CHAIN and getattr(self, 'chain', 'cyl') == 'cyl':   # 串接第一件：茶几圓柱（同 pick_tick；放盒後換抽屜）
                    self.pick_tick(t)
                    g_ = self.pick_goal if self.pick_goal is not None else PICK_PRIOR
                elif PICK and DRAWER:                                 # 開抽屜→取筆→返回（Howard 11/18）
                    self.drawer_tick(t)
                    god_ = self.dr.get('goal_od')
                    if god_ is not None and not self.pick_ret and self.pick_goal is not None and float(np.hypot(*(self.b_od_now[:2] - np.asarray(god_[:2], float)))) > PICK_APP_DIST:   # 近目標（接近段）不重送：避免起點≈終點 no_plan   # 第二十二趟：選位時 AMCL 偏航暫偏約 3° ⇒ 11 m 外換算差 0.6 m、目標落進櫃子膨脹區 ⇒ 每拍以當下 map↔odom 重新換算，差 >5 cm 才更新
                        Tg_ = _T3(self.T_mo_now) @ _T3(np.asarray(god_, float))
                        gm_ = [float(Tg_[0, 3]), float(Tg_[1, 3]), math.atan2(Tg_[1, 0], Tg_[0, 0])]
                        if float(np.hypot(gm_[0] - self.pick_goal[0], gm_[1] - self.pick_goal[1])) > 0.05:
                            self.event('dr_goal_reconverted', old=[round(float(x), 3) for x in self.pick_goal], new=[round(x, 3) for x in gm_])
                            self.pick_goal = gm_
                    g_ = self.pick_goal if self.pick_goal is not None else DR_PRIOR
                elif PICK:                                            # 取物（穿門後不停車）：粗目標＝茶几先驗 → 看到圓柱後線上選抓取底盤位 → 近距送取物封包 → 抬起後回起點
                    self.pick_tick(t)
                    g_ = self.pick_goal if self.pick_goal is not None else PICK_PRIOR
                if RET_CLEAR and self.pick_ret:                       # 返回：清全域代價圖後等障礙層重標再送；送出後久無路徑 ⇒ 再清
                    n_rc_ = getattr(self, 'n_ret_clear', 0)
                    no_path_ = self.path_t is None or self.path_t < self.pass_t
                    if (n_rc_ == 0 or (n_rc_ < 4 and no_path_ and t - getattr(self, 't_ret_clear', t) >= RET_CLEAR_RETRY_S)) and self._clr_cli.service_is_ready():
                        self._clr_fut, self._clr_t = self._clr_cli.call_async(self._clr_req()), t
                        self.n_ret_clear, self.t_ret_clear, self.pass_t = n_rc_ + 1, t, t
                        self.event('return_costmap_clear_called', n=n_rc_ + 1)
                    if self._clr_t is not None and t - self._clr_t < PASS_CLEAR_WAIT_S:
                        return
                g_chg_ = getattr(self, '_g_last', None) is None or float(np.hypot(g_[0] - self._g_last[0], g_[1] - self._g_last[1])) > 0.01 or abs(g_[2] - self._g_last[2]) > 0.01
                if (g_chg_ or self.path_t is None or self.path_t < self.pass_t) and t - self.t_pass_pub >= 1.0:   # 目標改變 ⇒ 立即重送（s38：選位 0.2 s 完成，舊粗目標的路徑晚到蓋過 path_t ⇒ 新目標沒送出）
                    self.pub_goal(g_[:2], g_[2]); self.t_pass_pub = t; self._g_last = list(g_)
                if self.xstate and ('"what": "world_end"' in self.xstate or 'engineering_abort' in self.xstate):
                    self.done_sent = True
                return
            if self.goal_xy is not None and (self.path_t is None or self.path_t < self.t_goal_set) and t - self.t_goal_pub >= 1.0:
                self.pub_goal(self.goal_xy, self.prior['yaw']); self.t_goal_pub = t
            if self.b_prev is not None and t > self.b_prev[0]:
                v_w = (self.cur_map[:2] - self.b_prev[1][:2]) / (t - self.b_prev[0]); wz = (self.cur_map[2] - self.b_prev[1][2]) / (t - self.b_prev[0])
            else:
                v_w, wz = np.zeros(2), 0.0
            self.b_prev = (t, self.cur_map.copy()); self.v_w_last = v_w
            if self.ho_sent is not None and not UNIFIED:                 # 交權後：手臂與底盤命令都屬抓持／開門控制器；本節點只記錄
                if self.xstate and ('"what": "world_end"' in self.xstate or 'engineering_abort' in self.xstate):
                    self.done_sent = True
                return
            # ---- 走廊（可接任務狀態集合）：先驗暫定 → 估計
            src_ = 'estimate' if self.T_est is not None else 'prior'
            if self.req is None and self.mode in ('NAV',) and (self.plan_pending or (self.plan is None and t - self.t_rp_try >= 1.0)) and (self.plan is None or (src_ == 'estimate' and self.ref is None)):
                T_e = self.T_est if self.T_est is not None else (np.linalg.inv(_T3(T_mo)) @ np.asarray(self.prior['T_prior'], float)).tolist()
                self.plan_pending = False; self.t_rp_try = t; self.ver += 1
                self.req = dict(kind='region_plan', ver=self.ver, t=t, w=time.perf_counter(), src=src_, T_mo=T_mo.copy(), T_est=T_e)
                c0, s0 = math.cos(T_mo[2]), math.sin(T_mo[2])
                pk.send(('region_plan', dict(ver=self.ver, q_arm=q[3:].tolist(), T_mo=T_mo.tolist(), T_est=T_e, robot_xy=[c0 * b_od[0] - s0 * b_od[1] + T_mo[0], s0 * b_od[0] + c0 * b_od[1] + T_mo[1]])))
            if self.req is not None and pk.poll(0):
                tag, v, pay = pk.recv()
                rq, self.req = self.req, None
                if rq['kind'] == 'region_plan':
                    rec = dict(t_req=round(rq['t'], 3), t_done=round(t, 3), wall_s=round(time.perf_counter() - rq['w'], 3), tag=tag, src=rq.get('src'), ok=None if tag != 'region_plan' else pay.get('ok'), why=None if tag != 'region_plan' else pay.get('why'),
                               region=None if tag != 'region_plan' else pay.get('region'), feas=None if tag != 'region_plan' else pay.get('feas'), err=None if tag == 'region_plan' else str(pay)[:400])
                    self.plans.append(rec); self.event('region_plan_result', **{k: rec[k] for k in ('t_req', 't_done', 'wall_s', 'src', 'ok', 'why', 'region', 'err')})
                    if tag == 'region_plan' and pay.get('ok'):
                        tag2, p2 = self.ask('adopt', dict(plan=pay['plan'], T_Hm=pay['T_Hm'], d=pay['d'], n=pay['n'], yaw=pay['yaw'], T_est=pay['T_est']), 5.0)
                        if tag2 == 'adopt':
                            self.plan = dict(pay['plan'], yaw=pay['yaw'], d=pay['d'], n=pay['n'], a_dec=0.2, v_tr=0.10); self.region = pay['region']; self.plan_ver += 1
                            self.h_xy = np.asarray(pay['T_Hm'], float)[:2, 3]; self.T_ref = np.asarray(rq['T_mo'], float)
                            if rq.get('src') == 'estimate':
                                self.ref = dict(T_m=np.asarray(pay['T_Hm'], float), T_od=np.asarray(rq['T_est'], float), t_obs=getattr(self, 'T_est_tobs', None), ver=1, src='region_estimate'); self.ref_ver = 1
                                self.ref_cand = None
                                self.ref_log.append(dict(t=round(t, 3), ver=1, src='region_estimate'))
                                self.mode = 'APPROACH'; self.event('mode', mode='APPROACH', plan_ver=self.plan_ver)
                            else:
                                self.event('region_adopted', plan_ver=self.plan_ver, src=rq.get('src'))
                        else:
                            self.event('adopt_failed', detail=str(p2)[:300])
                elif rq['kind'] == 'dapp':
                    self.on_dapp(rq, tag, pay, t)
            if UNIFIED:                                                  # 單一控制器：導航目標＝抓取底盤位姿（依當次區域規劃；Smac 決定來向，無走廊入口、無沿門直線）；接近開始依觀測觸發
                if self.owner != 'GRASP' and self.ho_sent is not None and self.ho_sent.get('unified') and t - self.ho_sent.get('t_resend', self.ho_sent['t_sent']) >= 0.5:
                    n_h = int(round(t / PHYS)) + NPH + 3                # 起點交權重送（ROS 探索未完成時第一筆會遺失；s3），直到第一筆抓持提案到達；控制伺服器已收過則忽略
                    T_pr = np.linalg.inv(_T3(T_mo)) @ np.asarray(self.prior['T_prior'], float)
                    self.p_ho.publish(String(data=json.dumps(dict(n_h=n_h, ins=dict(s0=0.15, v0=0.0, a0=0.0, T=1e6), unified=True, T_ref=T_pr.tolist(), t_obs=t, ref_ver=0, win=[-1e9, 1e9]))))
                    self.ho_sent['t_resend'] = t; self.ho_sent['n_resend'] = self.ho_sent.get('n_resend', 0) + 1
                if self.plan is not None and self.stopped is None:       # 朝向條件照原機制（車頭朝門，相機看得到把手）；速度框＝導航自身上下限（不另限速）
                    self.pub_task(self.plan['yaw'], [[-0.2, 0.35], [-0.25, 0.25], [-0.8, 0.8]])
                if self.plan is not None and self.goal_ver != self.plan_ver and self.stopped is None:
                    pl, rg = self.plan, self.region
                    d_, n_ = np.array(pl['d'], float), np.array(pl['n'], float)
                    gxy = np.asarray(self.h_xy, float) - float(pl['a_w']) * d_ + float(rg['sb_c']) * n_
                    self.goal_xy, self.goal_ver = gxy.tolist(), self.plan_ver
                    self.prior['yaw'] = float(pl['yaw']); self.t_goal_set = t; self.pub_goal(self.goal_xy, self.prior['yaw']); self.t_goal_pub = t
                    self.event('goal_grasp_pose', goal=[round(float(x), 4) for x in gxy], yaw=round(float(pl['yaw']), 4), a_w=pl['a_w'], sb_c=rg['sb_c'], plan_ver=self.plan_ver)
                if self.mode == 'APPROACH' and self.app_sent is None and self.goal_xy is not None and self.ref is not None and self.stopped is None:
                    dist_ = float(np.linalg.norm(self.cur_map[:2] - np.asarray(self.goal_xy, float)))
                    if dist_ <= APP_TRIG:
                        rf = self.ref
                        T_send, fz = np.asarray(rf['T_od'], float), None
                        keep = []
                        for (to_, Tm_) in self.obs_T:
                            dR_ = T_send[:3, :3].T @ Tm_[:3, :3]
                            if np.linalg.norm(Tm_[:3, 3] - T_send[:3, 3]) <= 0.010 and math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR_) - 1) / 2)))) <= 3.0:
                                keep.append(Tm_)
                        if len(keep) >= 3:
                            Tf = np.eye(4); Tf[:3, 3] = np.median([Tm_[:3, 3] for Tm_ in keep], axis=0)
                            U_, _, Vt_ = np.linalg.svd(np.sum([Tm_[:3, :3] for Tm_ in keep], axis=0))
                            Tf[:3, :3] = U_ @ np.diag([1.0, 1.0, np.linalg.det(U_ @ Vt_)]) @ Vt_
                            fz = dict(n=len(keep), n_all=len(self.obs_T), dpos_mm=round(float(np.linalg.norm(Tf[:3, 3] - T_send[:3, 3])) * 1e3, 3), t_last=max(to_ for to_, _ in self.obs_T))
                            T_send = Tf
                        Tapp = max(APP_T_MIN, dist_ / 0.08)
                        app = dict(T=Tapp, T_b=max(1.5, Tapp - 1.5), t_l0=max(0.5, Tapp - APP_T_LINE), d_line=APP_D_LINE, t_settle=1.0, s_final=float(os.environ.get('B_S_FINAL', '0.0015')))
                        Tg_od = np.linalg.inv(_T3(T_mo)) @ _T3([self.goal_xy[0], self.goal_xy[1], self.prior['yaw']])
                        app['base_goal'] = [float(Tg_od[0, 3]), float(Tg_od[1, 3]), math.atan2(Tg_od[1, 0], Tg_od[0, 0])]   # 抓取底盤位姿（里程座標）：導航到點後由同一控制器收斂到此
                        sreg_ = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
                        ox_ = sreg_['door']['opening_x']
                        Ti_ = np.linalg.inv(_T3(T_mo))
                        hg_ = Ti_ @ np.array([float(ox_[0]), 0.0, 0.0, 1.0]); dr_ = Ti_[:3, :3] @ np.array([1.0, 0.0, 0.0])
                        app['door'] = dict(hinge=[float(hg_[0]), float(hg_[1])], dir=[float(dr_[0]), float(dr_[1])], w=0.95, src='場景登錄（鉸鏈、門扇方向；往 y<0 開＝繞鉸鏈轉 −φ）')   # 操作對象：接近開始後門扇附近的障礙點不進 CBF（門的碰撞由門專屬淨空核對負責）
                        msg = dict(what='app', app=app, T_ref=T_send.tolist(), t_obs=(rf['t_obs'] if fz is None else fz['t_last']), ref_ver=rf['ver'] + (0 if fz is None else 1), fused=fz, dist=round(dist_, 4))
                        self.p_ho.publish(String(data=json.dumps(msg, default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_))))
                        self.app_sent = dict(t_app=round(t, 3), dist=round(dist_, 4), app=app, fused=fz)
                        self.event('approach_started', **self.app_sent)
                if self.xstate and ('"what": "world_end"' in self.xstate or 'engineering_abort' in self.xstate):
                    self.done_sent = True
                return
            # ---- 任務條件：朝向＋速度框＋區域
            inside = bool(self.rs is not None and self.rs.get('inside'))
            if self.plan is not None and self.stopped is None:
                pl, rg = self.plan, self.region
                a_m = float(-((self.cur_map[:2] - self.h_xy) @ np.array(pl['d'])))
                sb_m = float((self.cur_map[:2] - self.h_xy) @ np.array(pl['n']))
                in_sb = abs(sb_m - rg['sb_c']) <= rg['half_width'] + 0.02
                if in_sb:
                    v_al, v_n = self.v_profile(a_m), 0.05
                else:
                    v_al, v_n = V_APPROACH, V_APPROACH
                if self.mode == 'APPROACH':                              # 接近段（交權前）：沿門速度上限＝離接手點的距離 ÷ 手臂所需時間（重算延遲＋最短擺動＋最短直線＋餘量）；預測（v_profile）同式
                    T_need = RETR_LEAD + DP_CANDS[0][0] + DP_CANDS[0][2] + T_NEED_MARGIN
                    v_al = min(v_al, max(0.05, (a_m - float(pl['a_w'])) / T_need))
                reg = dict(id='b%d_r%d' % (self.plan_ver, self.ref_ver), line=[self.ref_to_map(p_) for p_ in rg['line']], half_width=rg['half_width'], extend=0.40, enter_by=rg['enter_by'])
                self.pub_task(pl['yaw'], self.box_for(v_al, v_n, float(self.cur_map[2]), np.array(pl['d'])), region=reg)
            # ---- 直接接近重算請求（擺動段結束前；交權前）
            if self.dp_next is not None and t >= self.dp_next['t_s'] - 1e-9:
                self.dp, self.dp_next = self.dp_next, None
            if self.mode == 'APPROACH' and not APP_HO and self.req is None and self.stopped is None and t - self.t_retr >= RETR_PERIOD:
                frozen = self.dp is not None and t + RETR_LEAD >= self.dp['t_D'] - DP_BLEND
                if not frozen:
                    self.t_retr = t; self.ver += 1
                    order = [self.v_active]                               # 接近段速度由限速式決定（預測同式）⇒ 各速度候選預測相同，只算一組
                    preds = [(v_, self.base_prediction(v_).tolist()) for v_ in order]
                    rc = self.ref_cand
                    if rc is not None:                                    # 近距參考候選（里程座標）→ 規劃座標；與現行參考差須在閘門內（≤20 mm、3°）
                        T_od_new = np.asarray(rc['T'], float); T_m_new = _T3(self.T_ref_now) @ T_od_new
                        dpos = float(np.linalg.norm(T_od_new[:3, 3] - self.ref['T_od'][:3, 3]))   # 閘門在里程座標比（c8：舊地圖位姿含 AMCL 修正前的 map←odom）
                        dR_ = self.ref['T_od'][:3, :3].T @ T_od_new[:3, :3]; dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR_) - 1) / 2))))
                        use = dpos <= 0.020 and dang <= 3.0
                        if not use:
                            self.event('ref_candidate_gated', dpos_mm=round(dpos * 1e3, 2), dang_deg=round(dang, 3)); self.ref_cand = None; rc = None
                    T_use_m = _T3(self.T_ref_now) @ (self.ref['T_od'] if rc is None else np.asarray(rc['T'], float))   # 手臂任務錨定里程座標：每次請求以當下 map←odom 換到規劃座標（c8：沿用估計當時的地圖位姿 ⇒ AMCL 修正後 TCP 偏 22 mm）
                    T_use_od = self.ref['T_od'] if rc is None else np.asarray(rc['T'], float)
                    self.req = dict(kind='dapp', ver=self.ver, t=t, w=time.perf_counter(), ref_cand=rc, T_m=T_use_m, T_od=T_use_od)
                    pk.send(('dapp', dict(ver=self.ver, t_req=t, q=q.tolist(), s=self.truth['s'].tolist(), pipe=self.truth['pipe'].tolist(), T_mo=T_mo.tolist(), T_ref=self.T_ref_now.tolist(),
                                          T_ref_m=T_use_m.tolist(), T_est=T_use_od.tolist(), dp_cur=self.dp, dp_next=self.dp_next, base_preds=preds,
                                          sb_lim=[self.region['sb_feasible'][0], self.region['sb_feasible'][1]])))
            # ---- 進走廊即交權（APP_HO，主線修正）：離抓取點 ≤ APP_DIST 且在走廊內 ⇒ 交給 W-GMPC 抓持控制器（接近、插入、閉爪、確認同一求解）
            if APP_HO and self.mode == 'APPROACH' and self.ho_sent is None and self.stopped is None and self.plan is not None and in_sb and inside:
                aw_ = float(DP_AW[-1])
                dist_ = a_m - aw_
                if dist_ <= APP_DIST:
                    Tapp = max(APP_T_MIN, dist_ / APP_V)
                    app = dict(T=Tapp, T_b=max(1.5, Tapp - 1.5), t_l0=max(0.5, Tapp - APP_T_LINE), d_line=APP_D_LINE)
                    n_h = int(round(t / PHYS)) + NPH + 3
                    rf = self.ref
                    T_send, fz = np.asarray(rf['T_od'], float), None
                    keep = []
                    for (to_, T_) in self.obs_T:
                        dR_ = T_send[:3, :3].T @ T_[:3, :3]
                        if np.linalg.norm(T_[:3, 3] - T_send[:3, 3]) <= 0.010 and math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR_) - 1) / 2)))) <= 3.0:
                            keep.append(T_)
                    if len(keep) >= 3:
                        Tf = np.eye(4); Tf[:3, 3] = np.median([T_[:3, 3] for T_ in keep], axis=0)
                        U_, _, Vt_ = np.linalg.svd(np.sum([T_[:3, :3] for T_ in keep], axis=0))
                        Tf[:3, :3] = U_ @ np.diag([1.0, 1.0, np.linalg.det(U_ @ Vt_)]) @ Vt_
                        fz = dict(n=len(keep), n_all=len(self.obs_T), dpos_mm=round(float(np.linalg.norm(Tf[:3, 3] - T_send[:3, 3])) * 1e3, 3), t_last=max(to_ for to_, _ in self.obs_T))
                        T_send = Tf
                    ins_ = dict(s0=APP_D_LINE, v0=0.0, a0=0.0, T=Tapp)
                    msg = dict(n_h=n_h, ins=ins_, app=app, T_ref=T_send.tolist(), t_obs=(rf['t_obs'] if fz is None else fz['t_last']), ref_ver=rf['ver'] + (0 if fz is None else 1), win=[0.0, APP_V * Tapp + 0.22], fused=fz)
                    self.p_ho.publish(String(data=json.dumps(msg, default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_))))
                    self.ho_sent = dict(fused=fz, t_sent=round(t, 3), n_h=n_h, app=app, a=round(a_m, 4), sb=round(sb_m, 4))
                    self.event('handover_requested', **self.ho_sent)
                    return
            # ---- 交權請求：直線到 r＝S_PRE 的時刻 t_h；提前一拍＋3 步送出（世界節點在 n_h−NPH 以真狀態前推建封包）
            if self.mode == 'APPROACH' and self.dp is not None and self.dp_next is None and self.ho_sent is None and self.stopped is None and t >= self.dp['t_h'] - 2 * DT + 1e-9:
                n_h = int(round(t / PHYS)) + NPH + 3
                rf = self.ref
                T_send, fz = np.asarray(rf['T_od'], float), None
                if FUSE_HO:                                              # 全部有效影格在現行參考閘門內（10 mm、3°）的中位數位置＋弦平均姿態（靜止把手；相對擷取同刻底盤已轉里程座標）
                    keep = []
                    for (to_, T_) in self.obs_T:
                        dR_ = T_send[:3, :3].T @ T_[:3, :3]
                        if np.linalg.norm(T_[:3, 3] - T_send[:3, 3]) <= 0.010 and math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR_) - 1) / 2)))) <= 3.0:
                            keep.append(T_)
                    if len(keep) >= 3:
                        Tf = np.eye(4); Tf[:3, 3] = np.median([T_[:3, 3] for T_ in keep], axis=0)
                        U_, _, Vt_ = np.linalg.svd(np.sum([T_[:3, :3] for T_ in keep], axis=0))
                        Tf[:3, :3] = U_ @ np.diag([1.0, 1.0, np.linalg.det(U_ @ Vt_)]) @ Vt_
                        fz = dict(n=len(keep), n_all=len(self.obs_T), dpos_mm=round(float(np.linalg.norm(Tf[:3, 3] - T_send[:3, 3])) * 1e3, 3), t_last=max(to_ for to_, _ in self.obs_T))
                        T_send = Tf
                msg = dict(n_h=n_h, ins=self.dp['ins'], T_ref=T_send.tolist(), t_obs=(rf['t_obs'] if fz is None else fz['t_last']), ref_ver=rf['ver'] + (0 if fz is None else 1), win=[0.0, 0.22], fused=fz)   # 實際採用參考（里程座標）與其原始來源時間
                self.p_ho.publish(String(data=json.dumps(msg)))
                self.ho_sent = dict(fused=fz, t_sent=round(t, 3), n_h=n_h, dp_t_h=round(self.dp['t_h'], 3), ins=self.dp['ins'], ref_ver=rf['ver'])
                self.event('handover_requested', **self.ho_sent)
                return                                                   # 本拍起不再發手臂命令（世界節點以最後命令前推、保持到接手）
            # ---- 手臂拍
            mode_w = 'APPROACH' if self.mode == 'APPROACH' else 'NAV'
            w0 = time.perf_counter()
            tag, out_ = self.ask('tick', dict(t=t, mode=mode_w, q=q.tolist(), pipe=self.truth['pipe'].tolist(), rel=self.rel.tolist(), s=self.truth['s'].tolist(), s_end=self.truth['s_end'].tolist(),
                                              T_mo=T_mo.tolist(), T_ref=self.T_ref_now.tolist(), v_world=v_w.tolist(), wz=wz), DEADLINE_MS / 1e3)
            rec = dict(t=round(t, 3), mode=self.mode, wall_ms=round((time.perf_counter() - w0) * 1e3, 1), tag=tag, inside=inside, rs=None if self.rs is None else {k: self.rs.get(k) for k in ('s', 'lat', 'phase')})
            if tag != 'tick':
                self.n_late += 1
                rec['why'] = 'late_or_error' if tag == 'late' else str(out_)[:500]
                self.ticks.append(rec)
                if self.mode == 'APPROACH' and self.stopped is None:
                    self.abort('tick_' + tag, detail=rec['why'])
                else:
                    self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
                return
            rec.update({k: out_.get(k) for k in ('a', 'sb', 'disp', 'stop', 'info', 'ua', 'n_clip', 'vis_geom', 'q_des', 't_h', 'ms_law')})
            self.ticks.append(rec)
            disp = out_['disp']
            if self.stopped is not None:
                self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
            elif disp in ('commit', 'no_estimate'):
                ua_pub = [float(x) for x in out_['ua']] if disp == 'commit' else [0.0] * 6
                self.p_arm.publish(Float64MultiArray(data=ua_pub)); self.ua_hist = [ua_pub, self.ua_hist[0]]
            elif disp == 'stop_now':
                self.stop_all('stop_check_next_failed', stop=out_.get('stop'))
            elif (out_.get('stop') or {}).get('why'):
                self.abort('stop_check_unverified', detail=(out_.get('stop') or {}).get('why')); return
            else:
                self.abort('stop_check_now_failed', stop=out_.get('stop')); return

        def on_dapp(self, rq, tag, pay, t):
            rec = dict(t_req=round(rq['t'], 3), t_done=round(t, 3), wall_s=round(time.perf_counter() - rq['w'], 3), tag=tag, with_ref_cand=rq['ref_cand'] is not None)
            if tag != 'dapp':
                rec['err'] = str(pay)[:600]; self.retrs.append(rec); self.event('dapp_error', **rec); return
            rec.update(ok=pay['ok'], cands=pay['cands'], chosen=None if not pay['ok'] else {k: pay['chosen'][k] for k in ('v', 'aw', 'T_move', 'd_line', 't_line', 't_s', 't_D', 't_h', 't_g', 'sb_g', 'last_vis', 'min_margin', 'ins')})
            adopted = False
            if pay['ok']:
                ch = pay['chosen']
                slot = None
                if not (t < ch['t_s'] - 1e-6 or self.dp is None):
                    rec['why_not'] = 't_s_passed'
                elif not ch.get('cont'):                                  # 非承接（起點＝手臂現狀靜止）：現行段未開始才替換，否則排下一段
                    slot = 'replace' if (self.dp is None or t < self.dp['t_s'] - 1e-9) else 'next'
                else:                                                     # 承接型：起點＝其承接段（cont_from）在 t_s 的狀態 ⇒ 該段必須一路保留到 t_s
                    cf = ch.get('cont_from')
                    if self.dp is not None and cf is not None and abs(self.dp['t_s'] - cf) < 1e-6:
                        slot = 'next'                                     # 承接現行段 ⇒ 排下一段（取代未開始的舊下一段）
                    else:
                        rec['why_not'] = 'cont_base_not_current'          # 承接的是尚未生效的下一段（或已被換掉）⇒ 不採用，等下次重算
                if slot is not None:
                    st_ = dict(T_est=rq['T_od'].tolist() if rq['ref_cand'] is not None else None)
                    tag2, p2 = self.ask('adopt_dp', dict(dp=ch, t_now=t, slot=slot, **st_), 2.0)
                    adopted = bool(tag2 == 'adopt_dp' and p2.get('ok'))
                    if adopted:
                        newdp = dict(ch)   # 完整參考（候選模擬的現行段也要用同一條閉迴路律）
                        if slot == 'replace':
                            self.dp, self.dp_next = newdp, None
                        else:
                            self.dp_next = newdp
                        self.v_active = ch['v']
                        if rq['ref_cand'] is not None:                   # 新參考隨可行重算一起提交（版本、來源影格、時間）
                            self.ref_ver += 1
                            self.ref = dict(T_m=rq['T_m'], T_od=rq['T_od'], t_obs=rq['ref_cand'].get('t_obs'), ver=self.ref_ver, src='near_estimate', seqs=rq['ref_cand'].get('seqs'))
                            self.ref_log.append(dict(t=round(t, 3), ver=self.ref_ver, src='near_estimate', seqs=rq['ref_cand'].get('seqs'), t_obs=rq['ref_cand'].get('t_obs')))
                            if self.ref_cand is rq['ref_cand']:
                                self.ref_cand = None
            rec['adopted'] = adopted
            self.retrs.append(rec)
            self.event('dapp', **{k: rec.get(k) for k in ('t_req', 't_done', 'wall_s', 'ok', 'adopted', 'with_ref_cand', 'chosen')}, why=[(c.get('v'), c.get('T_move'), c.get('d_line'), c.get('t_line'), c.get('aw'), c.get('why')) for c in pay['cands']][:6])

        def save(self):
            json.dump(dict(events=self.events, plans=self.plans, retrs=self.retrs, stale=self.stale, ref_log=self.ref_log, ho_sent=self.ho_sent, owner=self.owner, n_grasp_fwd=self.n_grasp_fwd, ticks=self.ticks, obs=self.obs_log, acks=self.acks[-100:], region_status=self.rstat_log[::5],
                           n_late=self.n_late, n_deadline=self.n_deadline, n_fwd=self.n_fwd, stopped=self.stopped, aborted=self.aborted,
                           params=dict(V_APPROACH=V_APPROACH, LEAD_S=LEAD_S, EST=EST, ASSOC_TOL=ASSOC_TOL, GOAL_A=GOAL_A, STILL=STILL, SB_CAND=SB_CAND, AW_CAND=AW_CAND, RETR_LEAD=RETR_LEAD, RETR_PERIOD=RETR_PERIOD,
                                       RETR_TRIGGER=RETR_TRIGGER, CHECK_EVERY=CHECK_EVERY, V_CANDS=V_CANDS, DEADLINE_MS=DEADLINE_MS, RTF=os.environ.get('B_RTF', '0.4'), SWEEP_S=os.environ.get('B_SWEEP_S', '0.040'), POOL_N=os.environ.get('B_POOL_N', '2'))),
                      open(out + '.json', 'w'), ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    tag, _v, pay = parent.recv()
    print('[bcap2] worker', tag, pay, flush=True)
    tagp, _vp, payp = pk.recv()
    print('[bcap2] planner', tagp, payp, flush=True)
    rclpy.init()
    n = Sup()
    try:
        while rclpy.ok() and not n.done_sent:
            rclpy.spin_once(n, timeout_sec=0.01)
            if n.xstate is not None and ('world_end' in n.xstate or 'latched' in n.xstate):
                n.event('executor_state', state=n.xstate[:200]); break
    except KeyboardInterrupt:
        pass
    finally:
        n.save()
        parent.send(None)
        pk.send(None)
        print('[bcap2] saved', flush=True)


if __name__ == '__main__':
    main()
