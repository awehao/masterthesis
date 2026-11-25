#!/usr/bin/env python3
"""開門後：放手→退出→收臂→穿門（Codex 20261117_193621、後續裁定；Howard：導航到操作同一個控制器）。
同一個 gmpc_wb 實例（開門執行層的 self.mpc，即抓持段建立的共用實例）；不重設求解器、不清管線（設定點、在途管線、前一筆命令原樣承接）。
階段（全部以實測完成條件推進；名目剖面時間自該階段開始起算）：
  UNBIND   命令＝零（綁定中的靜止閉鏈：底盤零速、手臂設定點不動 ⇒ 閉鏈自然成立，不產生任何會破壞閉鏈的動作）；
           out['release'] 帶解除請求（版本＝本控制器、請求拍）⇒ 控制伺服器發 /d2/ctl unbind ⇒ 執行端停用 FixedJoint 並回報 /d2/ack unbound（實際生效步）
           ⇒ 快照 unbound ⇒ 才進下一階段。逾時 UNBIND_TO_S ⇒ 停止。
  FINGERS  手指張開命令；命令仍零；兩指實測開度 ≥ FINGER_OK 才進退出；逾時 ⇒ 停止。
  RETREAT  TCP 沿抓取接近軸反向退出 S_RET（幾何同 DoorExitCtrl 的 Δ(s)＝T_at(把手, s)−T_at(把手, 0)；方向在階段開始時以底盤座標固定）；
           W-GMPC 手臂 active、tcp_rel_base；底盤參考＝錨定位姿。完成＝名目時間到且 TCP 相對底盤實際位移 ≥ 0.9·S_RET。
  STOW     TCP 相對底盤位姿混接到收臂姿態（起始導航構型 Q_STOW 的正運動學）；q_post＝Q_STOW 只引導分支。完成＝名目時間到且 TCP 相對誤差 ≤ 10 mm、3°。
  DRIVE    手臂 hold；底盤參考與障礙物＝導航前端（Smac；門扇不排除，照一般障礙物進 CBF）。
           完成（場景專用判定）＝導航到點（halted）≥1 s、命令小、底盤里程 y＞DONE_Y_MIN（已過門口）⇒ done ⇒ 有界減速 ⇒ finished。
尾段（Codex 裁定 2）：每拍候選＝本拍命令一拍＋同一停法（stop_traj_exit：真管線、有界減速）到停穩，逐取樣點核全身淨空（clear_door_fast，同控制伺服器實測監看的門模型、
  門角＝實測）與手臂設定點限位；通過 ⇒ 該軌跡就是本段提交的已核尾段（candidate_extra1）；不過 ⇒ 不提交候選、改有界減速，減速軌跡同樣核過才給尾段；
  核不過 ⇒ 不給尾段 ⇒ 執行器沿用上一份已核尾段。"""
import math
import os
import sys
import time

import numpy as np

import door_motm_grasp_ctrl as G
import yield_side as YS
YS_DOOR_R = float(os.environ.get('YS_DOOR_R', '1.6'))                # 會車側移不作用的門區半徑（離鉸鏈；里程原點≈鉸鏈）
YS_XI = os.environ.get('YS_XI', '1') == '1'
if YS.ENABLE:
    YS._load_map()                                                    # 地圖距離場在匯入時建好（不在控制拍內）
import door_clear_door_fast as CF

XO, XE, K, B, C0, S, W = G.XO, G.XE, G.K, G.B, G.C0, G.S, G.W
NPH, PHYS, DT, D_STEPS = G.NPH, G.PHYS, G.DT, G.D_STEPS
FINGER_OPEN, FINGER_OK = 0.0089, float(os.environ.get('PASS_FINGER_OK', '0.0085'))
UNBIND_TO_S, FINGER_TO_S = 3.0, 3.0
T_RET, S_RET = float(os.environ.get('PASS_T_RET', '1.5')), float(os.environ.get('PASS_S_RET', '0.10'))
T_STOW = float(os.environ.get('PASS_T_STOW', '3.0'))
PH_EXTRA_TO_S = 3.0                                                  # 名目時間後仍未達完成條件的寬限；逾時 ⇒ 停止
Q_STOW = np.array([float(x) for x in os.environ.get('PASS_Q_STOW', '-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018').split(',')])
W_Q_STOW = float(os.environ.get('PASS_W_Q', '2.0'))
DONE_HOLD_S = 1.0
STOW_MOVE = os.environ.get('PASS_STOW_MOVE', '0') == '1'            # Howard 11/19「開完門為什麼停著把手臂伸成那樣，不是應該邊走邊伸」⇒ 收臂段底盤照導航前端（限速框），不錨定
STOW_V, STOW_WZ = float(os.environ.get('PASS_STOW_V', '0.15')), float(os.environ.get('PASS_STOW_WZ', '0.5'))   # 收臂中底盤速度框（任務限制；收完進 DRIVE 即解除）
DOOR_V = float(os.environ['PASS_DOOR_V']) if os.environ.get('PASS_DOOR_V') else None   # 門口區（底盤 y＜DONE_Y_MIN）行駛速度框（任務限制，非導航參數；s61：門扇尾端旁全速 0.27 m/s ⇒ 停止軌跡離門扇 0.0984＜0.10 被停）；未設＝原行為
NEAR_V = float(os.environ['PASS_NEAR_V']) if os.environ.get('PASS_NEAR_V') else None   # 近家具減速（任務限制，非導航參數）：底盤對已知靜態物淨空 <NEAR_D ⇒ 速度框 NEAR_V（s60／s68／s69／第二十八趟：繞家具 0.24 m/s，停止軌跡底盤淨空 0.098–0.0998 被停）；未設＝原行為
NEAR_D = float(os.environ.get('PASS_NEAR_D', '0.20'))
CAP_BLEND = float(os.environ.get('PASS_CAP_BLEND', '0.40'))           # Howard 11/25 不平滑：近家具／門口區速度框在邊界一步解除（Isaac 124 108.5 s 0.12→0.24）⇒ 邊界外 CAP_BLEND 內平順升到 CAP_TOP（比原本保守，原限制不放寬）
CAP_TOP = float(os.environ.get('PASS_CAP_TOP', '0.26'))


def _ss(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


REF_VCAP = os.environ.get('D2_REF_VCAP', '1') == '1'                # Howard 11/25「茶几整段底盤很不平滑」：速度框只限車身 x、y 各軸，參考照導航速度跑在前面 ⇒ 兩軸同時貼框、底盤沿車身 45° 斜走（Isaac 124：限速 0.176 時指令 (0.17,0.17)）⇒ 先把參考視窗時間縮放到限速內，框留作上限
REF_VCAP_K = float(os.environ.get('D2_REF_VCAP_K', '0.95'))          # 參考速度＝框 × 此值（留餘裕，不貼框）
REF_WCAP = float(os.environ.get('D2_REF_WCAP', '0.30'))              # 參考偏航速率上限（rad/s；第 123、124 趟抓完以框上限 0.5 rad/s 轉 90°）


def G_xi_from_X(X, dt):
    """參考體速度＝相鄰參考位姿相對運動 /dt（同 door_pick_ctrl.xi_from_X）。"""
    xi = []
    for k in range(len(X) - 1):
        T1 = np.linalg.inv(X[k]) @ X[k + 1]
        xi.append(np.array([T1[0, 2] / dt, T1[1, 2] / dt, math.atan2(T1[1, 0], T1[0, 0]) / dt]))
    xi.append(xi[-1].copy() if xi else np.zeros(3))
    return xi


def ref_speed_cap(X, v_cap, dt, w_cap=None, yaw0=None):
    """參考視窗時間縮放（不改幾何）：沿視窗依序走，每段實際用時＝max(原 dt, 段長／v_cap, |偏航變化|／w_cap)，再以原拍距 dt 重新取樣（分數索引內插位置與偏航）。
    yaw0（目前車頭）＋w_cap：參考偏航改為自 yaw0 出發、每拍最多轉 w_cap·dt 追向原偏航（第 126 趟抓完時視窗第一點即路徑方向，與車頭差 85° ⇒ 偏航框 0.5 打滿）。
    回 (新視窗, 是否有改)。v_cap 或 w_cap 為 None ⇒ 該項不限。"""
    n = len(X)
    if n < 2:
        return X, False
    X, sc0_ = _ref_speed_cap(X, v_cap, dt, w_cap)
    if yaw0 is None or w_cap is None or w_cap <= 1e-6:
        return X, sc0_
    th_ = [math.atan2(float(Xk[1, 0]), float(Xk[0, 0])) for Xk in X]
    out, ch_ = [], False
    for k, Xk in enumerate(X):
        e_ = (th_[k] - float(yaw0) + math.pi) % (2 * math.pi) - math.pi
        lim_ = float(w_cap) * k * float(dt)
        if abs(e_) > lim_ + 1e-9:
            ch_ = True
            out.append(XO.from_xytheta(float(Xk[0, 2]), float(Xk[1, 2]), float(yaw0) + math.copysign(lim_, e_)))
        else:
            out.append(Xk)
    return (out if ch_ else X), (sc0_ or ch_)


YREF_RESET = float(os.environ.get('D2_YREF_RESET', '0.6'))            # 參考偏航狀態離實際車頭超過此值（rad）或停用超過 0.3 s ⇒ 重設為實際車頭


def yaw_ref_state(ctrl, qp, X, w_cap, dt):
    """偏航參考的持續狀態（第 127 趟：每拍以實際車頭為起點 ⇒ 多一個回授、與管線延遲疊成約 1 Hz ±2.5° 擺動）：
    狀態自上一拍的參考偏航每拍最多轉 w_cap·dt 追向導航視窗第一點偏航；不黏回實際車頭（只在差太多或久未使用時重設）。回本拍視窗起點偏航。"""
    th0 = math.atan2(float(X[0][1, 0]), float(X[0][0, 0]))
    yr, tl = getattr(ctrl, '_yref', None), getattr(ctrl, '_yref_t', -9.0)
    if yr is not None and abs(float(ctrl.t) - tl) < 1e-9:
        return yr                                                    # 同一拍重複求解（候選尾段等）不重複推進
    q2 = float(qp[2])
    if yr is None or float(ctrl.t) - tl > 0.3 or abs((yr - q2 + math.pi) % (2 * math.pi) - math.pi) > YREF_RESET:
        yr = q2
    e = (th0 - yr + math.pi) % (2 * math.pi) - math.pi
    yr = yr + max(-w_cap * dt, min(w_cap * dt, e))
    ctrl._yref, ctrl._yref_t = yr, float(ctrl.t)
    return yr


def _ref_speed_cap(X, v_cap, dt, w_cap=None):
    n = len(X)
    if n < 2:
        return X, False
    P_ = np.array([[float(Xk[0, 2]), float(Xk[1, 2])] for Xk in X])
    th_ = np.unwrap(np.array([math.atan2(float(Xk[1, 0]), float(Xk[0, 0])) for Xk in X]))
    seg = np.linalg.norm(np.diff(P_, axis=0), axis=1); dth = np.abs(np.diff(th_))
    need = np.full(n - 1, float(dt))
    if v_cap is not None and v_cap > 1e-6:
        need = np.maximum(need, seg / float(v_cap))
    if w_cap is not None and w_cap > 1e-6:
        need = np.maximum(need, dth / float(w_cap))
    if float(np.max(need)) <= float(dt) * (1.0 + 1e-6):
        return X, False
    tau = np.r_[0.0, np.cumsum(need)]                                  # 各原視窗點在縮放後的時刻
    u = np.interp(np.arange(n) * float(dt), tau, np.arange(n, dtype=float))   # 縮放後每拍對應的分數索引（超過視窗末端 ⇒ 停在末點）
    xs = np.interp(u, np.arange(n), P_[:, 0]); ys = np.interp(u, np.arange(n), P_[:, 1]); ths = np.interp(u, np.arange(n), th_)
    return [XO.from_xytheta(float(xs[k]), float(ys[k]), float(ths[k])) for k in range(n)], True


def near_v(b):
    """底盤位姿 b 對已知靜態物（牆、家具、已安裝的抽屜櫃；同 clear_door_fast 底盤項的物體與式子）淨空 <NEAR_D ⇒ 回 NEAR_V，否則 None。"""
    if NEAR_V is None:
        return None
    sh, _ = W._static()
    d = min((float(W.DOC.box_dist(np.array([[float(b[0]), float(b[1]), c[2]]]), c, R, hs)[0]) - W.DOC.RB for _, c, R, hs in sh if W.base_level(c, R, hs)), default=9.0)
    if d < NEAR_D:
        return NEAR_V
    if CAP_BLEND > 0.0 and d < NEAR_D + CAP_BLEND:
        return NEAR_V + (max(CAP_TOP, NEAR_V) - NEAR_V) * _ss((d - NEAR_D) / CAP_BLEND)
    return None


PARK_STRICT = os.environ.get('PARK_STRICT', '0') == '1'           # Howard 11/25「停車操作比較像停車後再動手臂，現在有點移動中操作了」⇒ 嚴格停車操作基準：
                                                                      #   手臂動作段（接近、展開、俯視）開始前等底盤到站且停穩；底盤起步（返回、穿門行駛）前等手臂停穩。開門、拉／推抽屜、回程推門等接觸段維持全身（同移動中組）
PARK_TOL_M = float(os.environ.get('PARK_TOL_M', '0.03'))             # 到站：離本段底盤目標 ≤ 此距離（m）
PARK_TOL_YAW = float(os.environ.get('PARK_TOL_YAW', '0.05'))         # 到站：偏航差 ≤ 此值（rad）
PARK_V = float(os.environ.get('PARK_V', '0.01'))                     # 停穩：上一拍底盤指令與實測平移速率 ≤ 此值（m/s）
PARK_W = float(os.environ.get('PARK_W', '0.02'))                     # 停穩：偏航速率 ≤ 此值（rad/s）
PARK_QD = float(os.environ.get('PARK_QD', '0.03'))                   # 手臂停穩：各關節實測速率 ≤ 此值（rad/s）
PARK_SETTLE_S = float(os.environ.get('PARK_SETTLE_S', '0.3'))        # 停穩須持續此秒數
PARK_WAIT_MAX_S = float(os.environ.get('PARK_WAIT_MAX_S', '12.0'))   # 等底盤停穩上限（逾時放行並記事件，不中止）
PARK_ARM_MAX_S = float(os.environ.get('PARK_ARM_MAX_S', '2.5'))      # 等手臂停穩上限
PARK_T_MIN = float(os.environ.get('PARK_T_MIN', '4.0'))              # 停穩後手臂時程＝監督封包公式在距離 0 的值（T＝4.0、T_b＝T−1.5、t_l0＝T−2.5）
PARK_CAP_V = float(os.environ.get('PARK_CAP_V', '0.02'))             # 停車後只動手臂的段：底盤速度框（m/s；s220d 手臂快伸時全身 QP 拉底盤 0.107 m/s 幫忙）
PARK_CAP_W = float(os.environ.get('PARK_CAP_W', '0.05'))             # 同上，偏航速率（rad/s）
PARK_FREE = ('PULL', 'BIND', 'SWING', 'ALIGN')                       # 接觸段（拉／推抽屜、回程關門綁定與共轉）：底盤照全身控制，不設框


def park_cap_apply(mpc, cap):
    """嚴格停車速度框套到 mpc.cfg（呼叫端負責求解後還原 u_min／u_max）；cap＝None 不動。"""
    if cap is None:
        return
    umax_ = np.minimum(np.asarray(mpc.cfg.u_max, float), cap)
    mpc.cfg.u_max = umax_
    mpc.cfg.u_min = np.minimum(np.maximum(np.asarray(mpc.cfg.u_min, float), -cap), umax_)   # 導航速度下限（前進下限）不可超過框


STOW_JOINT = os.environ.get('PASS_STOW_JOINT', '0') == '1'          # 收臂走關節空間直線（第二十五趟 TCP 混接 ⇒ j1 先甩到 −1.89 再回 0）
STOW_QD = float(os.environ.get('PASS_STOW_QD', '0.6'))              # 關節空間收臂峰速上限（rad/s；最小急動峰速 1.875×）
PICK_FOLLOW = os.environ.get('D2_PICK', '0') == '1'                 # 穿門後接取物：穿門段不自行判完成（由監督送取物封包換控制器）
DONE_Y_MIN = float(os.environ.get('PASS_DONE_Y_MIN', '0.3'))         # 場景專用：門口在 y＝0、客廳側 y＞0
KEYS = ('env', 'base', 'self', 'shell_bar')
PASS_POOL = os.environ.get('PASS_POOL', '0') == '1'
POOL_TIMEOUT_S = 0.5


def _T3(b):
    c, s = math.cos(b[2]), math.sin(b[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = b[:2]
    return T


def _log_R(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _exp_R(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx




FB_TICKS = int(os.environ.get('WG_FB_TICKS', '3'))                  # W-GMPC「底盤後處理改動」備援：連續至多幾拍可執行備援命令（0＝原行為：一律停）


def wg_result(ctrl, r, src):
    """全身求解結果 → (命令, 來源) 或 (None, 原因)。Isaac 93：回程關門展開第一拍底盤後處理改動（seq_consistent 不成立）即停整趟；
    求解器的備援＝hold 底盤解＋已驗證限制的手臂減速序列（arm_decel_valid）⇒ 連續 ≤FB_TICKS 拍照備援命令執行、下一拍照常重解（候選命令仍經執行器停止尾段核對），超過才停。"""
    ck = dict(r.arm_check or {})
    if not (ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved'):
        ctrl._fb_run = 0
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src
    if (FB_TICKS > 0 and ck.get('fallback') == 'base_postprocess_changed' and not ck.get('failed') and ck.get('arm_decel_valid')
            and r.u_arm_seq is not None and np.isfinite(np.asarray(r.u_opt, float)).all() and np.isfinite(np.asarray(r.u_arm_seq[0], float)).all()):
        ctrl._fb_run = getattr(ctrl, '_fb_run', 0) + 1
        if ctrl._fb_run <= FB_TICKS:
            ctrl.n_fb_ticks = getattr(ctrl, 'n_fb_ticks', 0) + 1
            if ctrl._fb_run == 1 and hasattr(ctrl, '_ev'):
                ctrl._ev('wgmpc_fallback_tick', why=ck.get('fallback'), n_total=ctrl.n_fb_ticks)
            return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src + '_fb'
    return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))

def yield_apply(ctrl, qp, X, obs, xi=None, manip=False):
    """會車選邊（yield_side；D2_YIELD_SIDE=1 才作用）：只平移導航參考視窗，事件進控制器紀錄。
    xi（參考體速度，就地改）：參考被改過 ⇒ 依改後位姿重算（s168d：等待時位置釘住、速度前饋仍是導航 0.22 ⇒ 機器人照常前進穿過停止線、被行人帶走）。"""
    if not YS.ENABLE:
        return None
    if math.hypot(float(qp[0]), float(qp[1])) < YS_DOOR_R:              # s116d：門扇（開著、不在地圖）會被側移撞上 ⇒ 鉸鏈 1.6 m 內不做會車側移（門區有門專屬淨空，CBF 照常擋行人）
        ys_ = getattr(ctrl, '_ys', None)
        if ys_ is not None:
            ys_.lock, ys_.along, ys_.off = None, None, 0.0
            ys_.pchoice, ys_.p_off = None, 0.0
        return None
    ys = getattr(ctrl, '_ys', None)
    if ys is None:
        ys = ctrl._ys = YS.YieldSide()
    ys.ref_dt = DT
    c_, s_ = math.cos(float(qp[2])), math.sin(float(qp[2])); u_ = np.asarray(ctrl.u_prev[:2], float)
    n0 = len(ys.log)
    ys.manip = bool(manip)                                          # 抓取／操作中：預測選動作只可減速（s195d：加速 1.2 倍超過移動中抓取限速 ⇒ W-GMPC 底盤後處理改命令 ⇒ 停）
    info = ys.apply(float(ctrl.t), np.asarray(qp[:3], float), np.array([c_ * u_[0] - s_ * u_[1], s_ * u_[0] + c_ * u_[1]]), X, obs, NPH * PHYS)
    for e_ in ys.log[n0:]:
        e_ = dict(e_); w_ = e_.pop('what'); e_['t_ys'] = e_.pop('t', None)   # 控制器 _ev 自帶 t（s93d：重複關鍵字 ⇒ 工作程序例外）
        ctrl._ev(w_, **e_)
    if info is not None and xi is not None and YS_XI:
        X_ = list(X)
        for k in range(len(X_) - 1):
            T1 = np.linalg.inv(X_[k]) @ X_[k + 1]
            xi[k] = np.array([T1[0, 2] / DT, T1[1, 2] / DT, math.atan2(T1[1, 0], T1[0, 0]) / DT])
        if len(X_) > 1:
            xi[len(X_) - 1] = np.array(xi[len(X_) - 2], float).copy()
    return info


def _obs_near(obs, q):
    """（只記錄）最近的非靜態障礙物：表面距（扣半徑）、速度 vx, vy（世界）；s91d 診斷用。"""
    best = None
    for o in obs or []:
        if o.get('static'):
            continue
        d = math.hypot(float(o['x']) - float(q[0]), float(o['y']) - float(q[1])) - float(o.get('radius', 0.0))
        if best is None or d < best[0]:
            best = (round(d, 3), round(float(o.get('vx', 0.0)), 3), round(float(o.get('vy', 0.0)), 3))
    return best

class DoorPassCtrl:
    def __init__(self, src, t0, q_end, phi_end):
        self.mpc, self.cfg = src.mpc, src.cfg
        self.s = np.asarray(src.s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in src.pipe]
        self.u_last, self.u_prev = np.zeros(9), np.asarray(getattr(src, 'u_prev', src.u_last), float).copy()
        self.lim_lo, self.lim_hi = np.asarray(src.lim_lo, float), np.asarray(src.lim_hi, float)
        self.nav_in = getattr(src, 'nav_in', None)
        self.bias_obs = getattr(src, 'bias_obs', None)
        self.T_WH0 = np.asarray(src.T_WH_bind, float).copy()
        self._pool = getattr(src, '_pool', None)                     # 即時化：開門段的共用淨空計算池（穿門／取物段閒置）
        self.T_HE = None if getattr(src, 'T_HE', None) is None else np.asarray(src.T_HE, float).copy()   # 去程抓持變換（回程關門同一種抓法）
        self.door_end = dict(b=np.asarray(q_end[:3], float).copy(), q=np.asarray(q_end[3:9], float).copy(), phi=float(phi_end))   # 開門結束的抓持構型（回程關門共轉起算）
        self.t, self.n_step = float(t0), 0
        self.phase, self.t_ph0 = 'UNBIND', float(t0)
        self.stop, self.abort, self.done, self.finished = None, None, False, False
        self.release, self.unbound, self.finger_cmd, self.finger_meas = True, False, None, None
        self.phi_now = float(phi_end)
        self.tail_cert_mode = False
        self._rec_tail = {1: []}
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
        self.events, self.wg_log, self.tick_log, self.sweep_log = [], [], [], []
        self.clear_min = {k: 9.0 for k in KEYS}
        self.b_anchor = np.asarray(q_end[:3], float).copy()
        Rg, T_HG, a_H = G.PG.baseline_reference()
        T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
        Th = src.handle_T(float(phi_end))
        self.dvec_w = T_at(Th, S_RET)[:3, 3] - T_at(Th, 0.0)[:3, 3]  # 世界座標退出位移（長 S_RET，沿接近軸遠離把手）
        self.T_rel_r0, self.T_rel_s0 = None, None
        self.q_stow0, self.T_stow = None, T_STOW
        self.T_rel_stow = K.fk(np.r_[np.zeros(3), Q_STOW], 'link_tcp')
        self.t_halt, self.t_still = None, None
        self._ev('pass_start', phi_end_deg=round(math.degrees(float(phi_end)), 3), dvec_w=[round(float(x), 4) for x in self.dvec_w], q_stow=Q_STOW.tolist(),
                 base_end=[round(float(x), 5) for x in q_end[:3]], q_end=[round(float(x), 5) for x in q_end[3:9]])

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _stop(self, why, /, **kw):
        if 'why' in kw:
            kw['detail_why'] = kw.pop('why')                          # 違反 dict 自帶 why（如 clearance）⇒ 改名，避免與停止原因重複
        if self.stop is None:
            self.stop = dict(why=why, t=round(self.t, 3), **kw)
            self._ev('pass_stop', why=why, **kw)

    def _goto(self, ph, **kw):
        self._park_reset(ph)
        self.phase, self.t_ph0 = ph, self.t
        self._ev('pass_phase', phase=ph, **kw)

    # ---- 嚴格停車操作（PARK_STRICT；預設關＝原行為）
    def _park_reset(self, ph):
        """換段：進入手臂動作段（展開／接近，非由展開接續）重設「等底盤停穩」；進入底盤起步段重設「等手臂停穩」。"""
        if not PARK_STRICT:
            return
        self._park_hold_arm = False
        if ph == 'UNFOLD' or (ph == 'APPROACH' and getattr(self, 'phase', None) != 'UNFOLD'):
            self._park_ok, self._park_w0, self._park_ts = False, None, None
        if ph in ('APPROACH', 'UNFOLD'):
            self._park_retimed = False
        if ph in ('RETURN', 'DRIVE'):
            self._park_arm_ok, self._park_aw0, self._park_ats, self._park_b = False, None, None, None

    def _park_cap(self):
        """嚴格停車：底盤已停穩、只動手臂的段 ⇒ 底盤速度框 [vx, vy, wz]；等底盤中、接觸段、開車段 ⇒ None。"""
        if not PARK_STRICT or getattr(self, '_park_hold_arm', False) or self.phase in PARK_FREE:
            return None
        if self.phase in ('APPROACH', 'UNFOLD') and not getattr(self, '_park_ok', False):
            return None
        if self.phase in ('RETURN', 'DRIVE') and not getattr(self, '_park_hold_base', False):
            return None
        return np.array([PARK_CAP_V, PARK_CAP_V, PARK_CAP_W])

    def _pt(self, t_abs):
        """時間型手臂參考的取樣時刻：嚴格停車等待中整個視窗停在段起點（手臂參考不前進、不預伸）。"""
        return self.t_ph0 if getattr(self, '_park_hold_arm', False) else t_abs

    def _park_goal(self):
        ap_ = getattr(self, 'app_prof', None)
        if ap_ is not None and ap_.get('base_goal') is not None:
            return np.asarray(ap_['base_goal'], float)
        bg_ = getattr(self, '_bg', None)
        return None if bg_ is None else np.asarray(bg_, float)

    def _park_wait_base(self, q, tag):
        """手臂動作段開始前：底盤到本段目標且停穩（指令與實測皆近零、持續 PARK_SETTLE_S）。True＝仍在等（呼叫端凍結手臂時程）。"""
        if not PARK_STRICT or getattr(self, '_park_ok', False):
            return False
        q = np.asarray(q, float)
        if getattr(self, '_park_w0', None) is None:
            self._park_w0, self._park_ts = self.t, None
            self._ev('park_wait_base', tag=tag)
        g_ = self._park_goal()
        up_ = np.asarray(self.u_prev, float)
        pq_ = getattr(self, '_park_q', None)
        dt_ = None if pq_ is None else self.t - pq_[0]
        vm_ = 0.0 if not dt_ or dt_ <= 1e-6 else float(np.hypot(q[0] - pq_[1][0], q[1] - pq_[1][1])) / dt_
        wm_ = 0.0 if not dt_ or dt_ <= 1e-6 else abs((float(q[2]) - float(pq_[1][2]) + math.pi) % (2 * math.pi) - math.pi) / dt_
        self._park_q = (self.t, q[:3].copy())
        d_ = None if g_ is None else float(np.hypot(q[0] - g_[0], q[1] - g_[1]))
        dy_ = None if (g_ is None or len(g_) < 3) else abs((float(q[2]) - float(g_[2]) + math.pi) % (2 * math.pi) - math.pi)
        near_ = (d_ is None or d_ <= PARK_TOL_M) and (dy_ is None or dy_ <= PARK_TOL_YAW)
        still_ = float(np.hypot(up_[0], up_[1])) <= PARK_V and abs(float(up_[2])) <= PARK_W and vm_ <= 2.0 * PARK_V and wm_ <= 2.0 * PARK_W
        if near_ and still_:
            self._park_ts = self.t if self._park_ts is None else self._park_ts
        else:
            self._park_ts = None
        ok_ = self._park_ts is not None and self.t - self._park_ts >= PARK_SETTLE_S - 1e-9
        to_ = self.t - self._park_w0 > PARK_WAIT_MAX_S
        if ok_ or to_:
            self._park_ok = True
            self._ev('park_go' if ok_ else 'park_go_timeout', tag=tag, waited_s=round(self.t - self._park_w0, 2), d=None if d_ is None else round(d_, 4), dyaw=None if dy_ is None else round(dy_, 4))
            return False
        return True

    def _park_wait_arm(self, q, tag):
        """底盤起步前：手臂各關節實測速率 ≤ PARK_QD 持續 PARK_SETTLE_S。True＝仍在等（呼叫端錨定底盤於 _park_b）。"""
        if not PARK_STRICT or getattr(self, '_park_arm_ok', False):
            return False
        q = np.asarray(q, float)
        if getattr(self, '_park_aw0', None) is None:
            self._park_aw0, self._park_ats, self._park_aq = self.t, None, None
            self._park_b = np.asarray(q[:3], float).copy()
        pa_ = getattr(self, '_park_aq', None)
        dt_ = None if pa_ is None else self.t - pa_[0]
        qd_ = 9.0 if not dt_ or dt_ <= 1e-6 else float(np.abs(q[3:9] - pa_[1]).max()) / dt_
        self._park_aq = (self.t, q[3:9].copy())
        if qd_ <= PARK_QD:
            self._park_ats = self.t if self._park_ats is None else self._park_ats
        else:
            self._park_ats = None
        ok_ = self._park_ats is not None and self.t - self._park_ats >= PARK_SETTLE_S - 1e-9
        to_ = self.t - self._park_aw0 > PARK_ARM_MAX_S
        if ok_ or to_:
            self._park_arm_ok = True
            self._ev('park_arm_settled' if ok_ else 'park_arm_timeout', tag=tag, waited_s=round(self.t - self._park_aw0, 2))
            return False
        return True

    def _pred(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _pool_state(self):
        import door_clear_door_fast as _CF_
        st = dict(bin='single' in getattr(sys.modules.get('rear_bin'), '_ORIG', {}))
        ds = sys.modules.get('drawer_scene')
        if ds is not None and 'static' in getattr(ds, '_ORIG', {}):
            st['drawer'] = dict(q_open=float(ds.Q_OPEN[0]), pose=tuple(float(x) for x in ds.POSE))
        return st

    def _sweep_pooled(self, Qs):
        """即時化（11/20）：取樣點分給共用計算池（同一淨空函式，附幾何狀態快照）；依原順序找第一個違反、clear_min 只累計到該點（同逐點版）。失效回 None（呼叫端改逐點算）。"""
        X = np.array([np.r_[qs, self.phi_now] for qs in Qs])
        fq = W.FINGER_Q
        try:
            h = self._pool.submit(self.T_WH0, fq, W.PHI0, X, state=self._pool_state())
            res, why, _ = self._pool.collect(h, POOL_TIMEOUT_S)
        except Exception:                                             # noqa: BLE001
            return None
        if res is None:
            self.pool_fail = getattr(self, 'pool_fail', 0) + 1
            return None
        n = 0
        for c in res:
            n += 1
            for k in KEYS:
                self.clear_min[k] = min(self.clear_min[k], float(c[k]))
            bad = [k for k in KEYS if c[k] < W.THR[k]]
            if bad:
                return dict(why='clearance', bad=bad, at={k: round(float(c[k]), 4) for k in KEYS}, who={k: str((c.get('who') or {}).get(k)) for k in bad}), n
        return None, n

    def _sweep(self, q, u, extra, j0, u_from, rec):
        """同一停法前推到停穩（候選 extra＝1：本拍命令一拍後減速）；逐取樣點核全身淨空與設定點限位。回 (違反 dict 或 None, 取樣數)。"""
        Q, _, _ = XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, extra, rec=rec)
        if rec is not None and any(np.any(r_[0] < self.lim_lo - 1e-9) or np.any(r_[0] > self.lim_hi + 1e-9) for r_ in rec):
            return dict(why='setpoint_limit'), 0
        if PASS_POOL and getattr(self, '_pool', None) is not None:
            Qs, prev = [], None
            for qs in Q:
                if prev is not None and float(np.abs(qs - prev).max()) < 1e-7:
                    continue
                prev = qs; Qs.append(qs)
            r_ = self._sweep_pooled(Qs)
            if r_ is not None:
                return r_
        n, prev = 0, None
        for qs in Q:
            if prev is not None and float(np.abs(qs - prev).max()) < 1e-7:
                continue                                              # 停穩後構型不變（同一點不重算）
            prev = qs
            c = CF.clear_door_fast(np.r_[qs, self.phi_now], self.T_WH0)
            n += 1
            for k in KEYS:
                self.clear_min[k] = min(self.clear_min[k], float(c[k]))
            bad = [k for k in KEYS if c[k] < W.THR[k]]
            if bad:
                return dict(why='clearance', bad=bad, at={k: round(float(c[k]), 4) for k in KEYS}, who={k: str((c.get('who') or {}).get(k)) for k in bad}), n
        return None, n

    def _stop_traj(self, q, u, e, j0, u_from):
        """執行器在本段末另做停止尾段時呼叫（停止模式）：同一停法＋同一淨空核對；不過 ⇒ 回違反（不給尾段，執行器沿用上一份已核尾段）。"""
        rec = None if self._rec_tail is None else self._rec_tail.get(e)
        bad, n = self._sweep(q, u, e, j0, u_from, rec)
        self.sweep_log.append((round(self.t, 3), 'stop_tail', n, None if bad is None else bad.get('why')))
        return None, bad, None

    def _rel_target(self, t_abs):
        if self.phase == 'RETREAT':
            u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_RET, 0.0), 1.0))[0]
            T = self.T_rel_r0.copy()
            T[:3, 3] = T[:3, 3] + u * (_T3(self.b_anchor)[:3, :3].T @ self.dvec_w)
            return T
        if STOW_JOINT:
            return K.fk(np.r_[np.zeros(3), self._stow_q(t_abs)], 'link_tcp')
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_STOW, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]
        T[:3, :3] = A[:3, :3] @ _exp_R(u * _log_R(A[:3, :3].T @ Bm[:3, :3]))
        return T

    def _stow_q(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_stow, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_stow0 + u * Q_STOW

    def _solve(self, q):
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        if (self.phase != 'DRIVE' and not (STOW_MOVE and self.phase == 'STOW')) or (self.phase == 'DRIVE' and getattr(self, '_park_hold_base', False)):   # 操作段（嚴格停車：開車前等手臂停穩）：底盤參考＝錨定位姿，障礙物照導航前端
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if src == 'nav' and REF_VCAP and self.phase in ('STOW', 'DRIVE'):   # 參考先時間縮放到本拍速度框內（同下方框的式子）＋偏航速率 REF_WCAP（Isaac 124 繞門扇 0.15 框兩軸貼滿斜走、偏航 0.5 rad/s 打滿）
            dc_ = DOOR_V is not None and float(qp[1]) < DONE_Y_MIN + max(CAP_BLEND, 0.0)
            vd_ = (DOOR_V if float(qp[1]) < DONE_Y_MIN else DOOR_V + (max(CAP_TOP, DOOR_V) - DOOR_V) * _ss((float(qp[1]) - DONE_Y_MIN) / CAP_BLEND)) if dc_ else 9.0
            nvr_ = near_v(qp[:3])
            vr_ = min(STOW_V if (STOW_MOVE and self.phase == 'STOW') else 9.0, vd_, nvr_ if nvr_ is not None else 9.0)
            X, sc_ = ref_speed_cap(list(X), None if vr_ >= 9.0 else REF_VCAP_K * vr_, DT, min(REF_WCAP, STOW_WZ), yaw0=yaw_ref_state(self, qp, X, min(REF_WCAP, STOW_WZ), DT))
            if sc_:
                xi = G_xi_from_X(X, DT)
        if src == 'nav':                                              # 會車選邊（s92d）：行人正面迎來且側向不足 ⇒ 依地圖淨空選邊平移參考
            yield_apply(self, qp, X, obs, xi)
        arm = dict(mode='hold')
        if self.phase in ('RETREAT', 'STOW'):
            Ts = [self._rel_target(t_eff + k * DT) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', **B.W_ARM)
            if self.phase == 'STOW' and STOW_JOINT:
                arm.update(q_post=np.array([self._stow_q(t_eff + (k + 1) * DT) for k in range(XO.N)]), w_q=5.0)
            elif self.phase == 'STOW' and W_Q_STOW > 0.0:
                arm.update(q_post=np.tile(Q_STOW, (XO.N, 1)), w_q=W_Q_STOW)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        u_max_saved = np.asarray(mpc.cfg.u_max, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        door_cap_ = DOOR_V is not None and self.phase in ('STOW', 'DRIVE') and float(qp[1]) < DONE_Y_MIN + max(CAP_BLEND, 0.0)
        v_door_ = (DOOR_V if float(qp[1]) < DONE_Y_MIN else DOOR_V + (max(CAP_TOP, DOOR_V) - DOOR_V) * _ss((float(qp[1]) - DONE_Y_MIN) / CAP_BLEND)) if door_cap_ else 9.0
        nv_ = near_v(qp[:3]) if self.phase in ('STOW', 'DRIVE') else None
        if (STOW_MOVE and self.phase == 'STOW') or door_cap_ or nv_ is not None:   # 邊走邊收：收臂中底盤速度框；門口區速度框；近家具減速
            v_ = min(STOW_V if (STOW_MOVE and self.phase == 'STOW') else 9.0, v_door_, nv_ if nv_ is not None else 9.0)
            lim_ = np.array([v_, v_, STOW_WZ])
            mpc.cfg.u_max = np.minimum(u_max_saved, lim_)
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        w0 = time.perf_counter()
        try:
            park_cap_apply(mpc, self._park_cap())
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min, mpc.cfg.u_max = u_min_saved, u_max_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), _obs_near(obs, qp)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        if arm['mode'] == 'hold':
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)], src
        return wg_result(self, r, src)

    def _advance(self, q):
        """實測完成條件推進階段（q＝本拍預測構型；門角、手指＝實測）。"""
        el = self.t - self.t_ph0
        if self.phase == 'DRIVE':
            self._park_hold_base = self._park_wait_arm(q, 'pass_drive')   # 嚴格停車：收臂後手臂停穩才開車（預設關 ⇒ False）
        if self.phase == 'UNBIND':
            if self.unbound:
                self.finger_cmd = FINGER_OPEN
                self._goto('FINGERS')
            elif el > UNBIND_TO_S:
                self._stop('unbind_ack_timeout')
        elif self.phase == 'FINGERS':
            fm = self.finger_meas
            if fm is not None and min(fm) >= FINGER_OK:
                W.FINGER_Q = float(np.mean(fm))
                self.T_rel_r0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._goto('RETREAT', finger=[round(float(x), 5) for x in fm])
            elif el > FINGER_TO_S:
                self._stop('finger_open_timeout', finger=None if fm is None else [round(float(x), 5) for x in fm])
        elif self.phase == 'RETREAT':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            moved = float(np.dot(T_now[:3, 3] - self.T_rel_r0[:3, 3], _T3(self.b_anchor)[:3, :3].T @ self.dvec_w)) / S_RET
            if el >= T_RET and moved >= 0.9 * S_RET:
                self.T_rel_s0 = T_now
                self.q_stow0 = np.asarray(q[3:], float).copy()
                self.T_stow = max(T_STOW, 1.875 * float(np.abs(Q_STOW - self.q_stow0).max()) / STOW_QD + 0.5) if STOW_JOINT else T_STOW
                self._goto('STOW', retreat_m=round(moved, 4), T=round(self.T_stow, 2), move=STOW_MOVE, joint=STOW_JOINT)
            elif el > T_RET + PH_EXTRA_TO_S:
                self._stop('retreat_not_reached', retreat_m=round(moved, 4))
        elif self.phase == 'STOW':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self.T_rel_stow[:3, 3]))
            er = math.degrees(float(np.linalg.norm(_log_R(T_now[:3, :3].T @ self.T_rel_stow[:3, :3]))))
            if el >= self.T_stow and ep <= 0.010 and er <= 3.0:
                self._goto('DRIVE', stow_err_mm=round(ep * 1e3, 2), stow_err_deg=round(er, 2), base=[round(float(x), 4) for x in q[:3]])
            elif el > self.T_stow + PH_EXTRA_TO_S:
                self._stop('stow_not_reached', stow_err_mm=round(ep * 1e3, 2), stow_err_deg=round(er, 2))

    def step(self, t, q, phi_meas, finger_meas=None):
        self.t = float(t)
        q = np.asarray(q, float)
        self.phi_now = float(phi_meas)
        if finger_meas is not None:
            self.finger_meas = (float(finger_meas[0]), float(finger_meas[1]))
        if self.n_step % NPH == 0:
            w_t = time.perf_counter()
            u_from = np.asarray(self.pipe[-1][:3], float).copy()
            if self.stop is None and not self.done:
                self._advance(q)
            cand = None
            if self.stop is None and not self.done:
                if self.phase in ('UNBIND', 'FINGERS'):
                    cand = np.zeros(9)                               # 綁定中／放手中：靜止（閉鏈自然成立）
                else:
                    u, why = self._solve(q)
                    if u is None:
                        self._stop(why)
                    else:
                        cand = u
                if cand is not None and self.phase == 'DRIVE':
                    ni = self.nav_in
                    halted = ni is not None and not ni.get('stale') and bool(ni.get('halted'))
                    self.t_halt = (self.t_halt if self.t_halt is not None else self.t) if halted else None
                    if not PICK_FOLLOW and self.t_halt is not None and self.t - self.t_halt >= DONE_HOLD_S and float(np.abs(cand[:3]).max()) < 0.01 and float(q[1]) > DONE_Y_MIN:
                        self.done = True
                        self._ev('pass_done', base=[round(float(x), 4) for x in q[:3]])
                        cand = None
            if cand is not None:                                     # 候選＋完整尾段核對（通過 ⇒ 該軌跡即本段提交的已核尾段）
                rec = self._rec_tail.get(1) if self._rec_tail is not None else None
                if rec is not None:
                    rec.clear()
                bad, n = self._sweep(q, cand, 1, 0, u_from, rec)
                self.sweep_log.append((round(self.t, 3), 'candidate', n, None if bad is None else bad.get('why')))
                if bad is not None:
                    if rec is not None:
                        rec.clear()
                    self._stop('candidate_tail_rejected', **bad)
                else:
                    self.u_last = cand
            if self.stop is not None or self.done:
                da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
                self.u_last = self.u_last + np.clip(-self.u_last, -da, da)
                quiet = not np.any(self.u_last) and not any(np.any(u_) for u_ in self.pipe)
                self.t_still = (self.t_still if self.t_still is not None else self.t) if quiet else None
                if self.t_still is not None and self.t - self.t_still >= 0.3 and not self.finished:
                    self.finished = True
                    self._ev('pass_finished', done=self.done, stop=None if self.stop is None else self.stop.get('why'))
            self.u_prev = self.u_last.copy()
            self.u_from_base, self.ramp_j = u_from, 0
            self.tick_log.append((round(self.t, 3), round((time.perf_counter() - w_t) * 1e3, 3), self.phase))
        cmd = XO.ramp_cmd(self.u_from_base, self.u_last, self.ramp_j)
        self.ramp_j += 1
        self.pipe.append(cmd)
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), finger=self.finger_cmd, ready=False, ready_ver=None, stopped=self.stop is not None, abort=False, done=self.done)
