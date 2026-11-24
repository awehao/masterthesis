#!/usr/bin/env python3
"""穿門後：茶几圓柱體移動抓取→抬升→持物返回起點（Howard 11/17 完整任務；Claude 自行裁定 20261117_2315，Codex 用量用完期間）。
同一個 gmpc_wb 實例（承接穿門控制器的 mpc、設定點、在途管線、前一筆命令、導航前端輸入；不重設、不清管線、任務切換不強制停車）。
停止尾段、候選核對、步進骨架沿用 DoorPassCtrl（每拍候選＋同一停法到停穩的全身淨空核對；不過不給尾段）。
階段（實測完成條件推進）：
  APPROACH  監督送的抓取底盤位姿＝導航目標（Smac）；TCP 參考＝接近開始時 TCP 相對底盤位姿（掛在參考底盤上）→ 沿接近軸的抓取線（d_line→0）混接；
            導航到點後底盤收斂到抓取底盤位姿。完成＝名目時間到且 TCP 對抓取位姿 ≤ 5 mm／3°（實測構型）。
  CLOSE     底盤原地、手臂保持抓取位姿、手指命令斜降到 0。
  CONFIRM   夾住證據＝兩指接觸力各 ≥ CONTACT_N 且手指未全閉（≥ FINGER_EMPTY），連續 EVID_S；期限 EVID_TO_S ⇒ 停止（未夾到）。
  LIFT      TCP 垂直抬升 LIFT_M（最小急動 T_LIFT）。
  VERIFY    抬升後仍有夾住證據連續 EVID_S ⇒ pick_lifted；否則停止（掉落）。
  CARRY     TCP 相對底盤混接到收臂姿態（持物）；q_post 只引導分支。
  RETURN    手臂 hold；底盤參考＋障礙物＝導航前端（監督送起點目標，經開著的門）。完成（場景專用）＝導航到點 ≥1 s、命令小、底盤里程 y＜RET_Y_MAX ⇒ done ⇒ 有界減速 ⇒ finished。
不宣稱：持物期間淨空模型不含圓柱本體（只核機器人與場景）；抓持＝Isaac 真實摩擦接觸（不綁定、不把圓柱固定到 TCP）；成功判定只用接觸力與手指開度（真值只做驗收記錄）。"""
import math
import os
import time

import numpy as np

import door_pass_ctrl as P
import motm_merge as MM

G, XO, XE, K, B, C0, S, W = P.G, P.XO, P.XE, P.K, P.B, P.C0, P.S, P.W
NPH, PHYS, DT, D_STEPS = P.NPH, P.PHYS, P.DT, P.D_STEPS
_T3, _log_R, _exp_R = P._T3, P._log_R, P._exp_R
D_LINE = float(os.environ.get('PICK_D_LINE', '0.08'))
T_SETTLE = float(os.environ.get('PICK_T_SETTLE', '1.0'))
GRASP_DEPTH = float(os.environ.get('PICK_GRASP_DEPTH', '0.004'))     # 瓶子線 bottle_s0_ctrl.GRASP_DEPTH
T_CLOSE, EVID_S, EVID_TO_S = 0.6, 0.3, 3.0
CONTACT_N = float(os.environ.get('PICK_CONTACT_N', '0.5'))
FINGER_EMPTY = float(os.environ.get('PICK_FINGER_EMPTY', '0.0015'))   # 手指開度低於此 ⇒ 判定夾空（全閉）
LIFT_M, T_LIFT = float(os.environ.get('PICK_LIFT_M', '0.05')), 1.2
T_CARRY = 3.0
PICK_CARRY = os.environ.get('PICK_CARRY', '1') == '1'                 # Howard 11/18「夾到東西沒有收手臂，伸出來很危險」⇒ 預設收臂（理想連接後收臂不再滑落）
FACE_DR = [float(x) for x in os.environ.get('B_DR_PRIOR', '-3.5,10.3,3.1416').split(',')] if (os.environ.get('D2_CHAIN') == '1' and os.environ.get('D2_DRAWER') == '1') else None   # 抽屜櫃前粗目標（同監督 DR_PRIOR；地圖≈里程）
MERGE_V_HI = float(os.environ.get('PICK_MERGE_V_HI', '0.20'))       # 併線段速度上限（進來的速度更低就維持；近家具另有 near_v 限速）
MOTM_TR = float(os.environ.get('MOTM_TR', '1.0'))                   # 進線速度斜坡時間常數（s）
AIM_K, AIM_MAX, AIM_START_M = float(os.environ.get('DR_AIM_K', '0.1')), float(os.environ.get('DR_AIM_MAX', '0.01')), float(os.environ.get('DR_AIM_START_M', '0.01'))   # 把手瞄準補償：每控制拍增益、上限（m）
CARRY_V, CARRY_WZ = float(os.environ.get('PICK_CARRY_V', '0.2775')), float(os.environ.get('PICK_CARRY_WZ', '1.1327'))   # 持物底盤速度框（m/s、rad/s）；夾到後理想連接 ⇒ 預設＝硬體上限（不收緊）
RET_Y_MAX = float(os.environ.get('PICK_RET_Y_MAX', '-2.3'))           # 場景專用：起點 (1.5, −2.8)
PH_TO_S = 4.0
APP_GATE = os.environ.get('PICK_PATH') == '1' and os.environ.get('PICK_APP_GATE', '1') == '1'
APP_GATE_D0, APP_GATE_D1 = 0.70, 0.12                                 # 收斂接近：參考底盤離抓取站位 ≤0.70 m 才開始伸手、≤0.12 m 才走最後抓取線
APP_GATE_CAP_S = 30.0
FLOW_D0, FLOW_LINE_TOL = float(os.environ.get('PICK_FLOW_D0', '1.2')), float(os.environ.get('PICK_FLOW_LINE_TOL', '0.02'))
FLOW_D1, FLOW_MISS_R = float(os.environ.get('PICK_FLOW_D1', '0.30')), float(os.environ.get('PICK_FLOW_MISS_R', '0.50'))   # 照導航移動中夾取：到站半徑（路徑不一定正好穿過抓取點）、錯過判定                                                 # 狀態閘門下的總安全逾時（名目時長之後）
FLOW_HDL_DZ = float(os.environ.get('PICK_FLOW_HDL_DZ', '0.004'))     # 照導航抓把手：抓取線起始高度差上限（m；收斂趟 Isaac 82/80 線起點約 −3 mm）
MEAS_GATE = os.environ.get('PICK_MEAS_GATE', '0') == '1'          # 固定橫桿閉爪門檻用實測 TCP（Isaac 95：實體橫桿比模型低約 5 mm，舊的預測狀態判斷在實體上才對準 ⇒ 預設關）
LIM_PAD = float(os.environ.get('PICK_LIM_PAD', '0.02'))               # 手臂關節位置界限內縮（rad）：s162d 茶几照導航接近 j3 名目 2.87、界限 2.936，在途命令積分越界 ⇒ setpoint_limit 停


def arm_lim(qa):
    """QP 用的關節位置界限：機器安全界限內縮 LIM_PAD；目前值已在內縮帶內 ⇒ 該側放寬到目前值（不讓 QP 不可行）。"""
    qa = np.asarray(qa, float)
    return np.minimum(XO.LITE6_SAFE.lower + LIM_PAD, qa), np.maximum(XO.LITE6_SAFE.upper - LIM_PAD, qa)
DR_APP_GATE = APP_GATE and os.environ.get('DR_APP_GATE', '1') == '1'   # 抽屜把手（開、關）也用狀態閘門（Isaac 53：底盤 t=159 到位後照 17.6 s 時間表伸手、停 16 s）
T_BMIN = float(os.environ.get('PICK_T_BMIN', '2.5'))                  # 伸手進度 0→1 至少這麼久（s131d：已到站時目標一步跳到把手 ⇒ 全身 QP 走貼近自身的路徑、自身淨空 4.3 mm）
T_LINE_S = max(0.8, D_LINE / 0.06)                                    # 抓取線（預抓→抓取 D_LINE）時長：由狀態觸發後走完
APP_TOL_Z = float(os.environ['PICK_APP_TOL_Z']) if os.environ.get('PICK_APP_TOL_Z') else None   # 圓柱（直立、側夾）接近完成：水平 ≤5 mm、沿圓柱軸（世界 z）≤ 此值；未設＝原 5 mm 球（第二十七趟 Isaac：東側伸遠，TCP 穩定低 9 mm、水平 <1 mm ⇒ approach_not_reached）
APP_V = float(os.environ['PICK_APP_V']) if os.environ.get('PICK_APP_V') else None   # 接近段底盤速度框（任務限制；s60／s68／s69：沿茶几邊 0.19 m/s 接近，停止軌跡底盤淨空 0.098–0.099 被停）；未設＝原行為
BIN = os.environ.get('D2_BIN', '0') == '1'                            # Howard 11/19「車子後面放一個盒子 放夾取物」：抬起確認後放進車尾收納盒（rear_bin.py）→ 收臂（相機朝前）→ 返回／下一件
BIN_QD = float(os.environ.get('D2_BIN_QD', '0.8'))                    # 放盒關節空間移動峰速上限（rad/s；最小急動峰速 1.875×）
JBLEND = os.environ.get('PICK_JBLEND', '1') == '1'                   # Howard 11/24「motm 超卡、會震盪」：照導航夾圓柱的手臂展開改關節空間內插（收臂→選位抓取解），TCP 參考＝內插構型掛在預測底盤上的 FK，後段才併到世界固定抓取線；同時給關節姿態引導（原卡氏內插＋無姿態引導 ⇒ 替身與 Isaac 都 ±1 rad/s 來回打滿、約 1 Hz）
JB_G0 = float(os.environ.get('PICK_JB_G0', '0.55'))                    # β 超過此值後才開始由關節內插 FK 併到世界抓取線
JB_WQ = float(os.environ.get('PICK_JB_WQ', '2.0'))
JB_QD = float(os.environ.get('PICK_JB_QD', '0.65'))
JB_D1 = float(os.environ.get('PICK_JB_D1', '0.30'))                    # 關節內插展開：β 在離抓取站位此距離完成（>FLOW_D1 ⇒ 到站前手臂已在預抓點，到站即走抓取線；Howard 11/24「路過夾了就走」）
FLOW_CLOSE_K = float(os.environ.get('PICK_FLOW_CLOSE_K', '0.6'))       # 照導航夾取：閉爪／確認段底盤限速倍率
LINE_V_CYL = float(os.environ.get('PICK_LINE_V_CYL', '0.06'))          # 照導航夾圓柱：抓取線速度（m/s；抽屜把手維持 0.06）                    # 展開段關節峰速目標（rad/s）                     # 展開段關節姿態引導權重
FLOW_VRAMP = os.environ.get('PICK_FLOW_VRAMP', '1') == '1'             # 照導航夾取進場限速依距離平順下降（原：進 r_turn 一步從 0.27 跳到 v_cap）；抓完放盒起步加速度斜坡
FLOW_V_HI = float(os.environ.get('PICK_FLOW_V_HI', '0.26'))           # 進場限速上端（＝導航名目）
FLOW_PULL = os.environ.get('PICK_FLOW_PULL', '1') == '1'             # Howard 11/25 選 A（不要經過線）：導航路徑在抓取位附近側向平順拉近抓取底盤位 b0，離開後退回導航路徑（路徑形狀、方向、速度照導航；Isaac 124／126 導航離桌 0.75 m、手臂要 0.46 m ⇒ 底盤被 TCP 拉出 L 形）
FLOW_PULL_DIN = float(os.environ.get('PICK_FLOW_PULL_DIN', '0.25'))   # 沿路徑離 b0 此距離內 ⇒ 全量拉到 b0 的側向位置
FLOW_PULL_DOUT = float(os.environ.get('PICK_FLOW_PULL_DOUT', '2.5'))  # 此距離外 ⇒ 不拉（中間 smoothstep）；第 128 趟 1.3 m 時 0.6 m 側移擠在約 1 m 內 ⇒ 側向 0.24 m/s 橫衝 ⇒ 2.5 m（偏轉約 15°）
FLOW_PULL_MAX = float(os.environ.get('PICK_FLOW_PULL_MAX', '0.6'))    # 側向拉移上限（m）
FLOW_PULL_TIN = float(os.environ.get('PICK_FLOW_PULL_TIN', '4.0'))    # 接手後拉移量隨時間平順加上（第 129 趟接手時已在 1.2 m 內、拉移七成一步跳 ⇒ 側向 0.24 m/s）
FLOW_PULL_TOUT = float(os.environ.get('PICK_FLOW_PULL_TOUT', '3.0'))  # 抬起後：凍結當下拉移向量、此時間內平順退回 0（第 128／129 趟放盒中依路徑切向重算 ⇒ 方向一直翻、四角來回）


def flow_pull(X, b0, din=None, dout=None, lmax=None, wmul=1.0):
    """導航參考視窗側向拉近 b0：每點以本點路徑切向 t、法向 n，沿切向離 b0 的距離 d＝|(b0−p)·t|，側向差 l＝(b0−p)·n；
    p ← p＋w(d)·clip(l, ±lmax)·n，w＝1（d≤din）→0（d≥dout）smoothstep。偏航不動。回 (新視窗, 最大拉移量)。"""
    din = FLOW_PULL_DIN if din is None else din; dout = FLOW_PULL_DOUT if dout is None else dout; lmax = FLOW_PULL_MAX if lmax is None else lmax
    n_ = len(X)
    if n_ < 2:
        return X, 0.0
    P_ = np.array([[float(Xk[0, 2]), float(Xk[1, 2])] for Xk in X])
    tot_ = P_[-1] - P_[0]
    if float(np.linalg.norm(tot_)) < 0.01:
        return X, 0.0
    b0_ = np.asarray(b0, float)[:2]
    out, mx_, s0_ = [], 0.0, np.zeros(2)
    for k in range(n_):
        tk_ = (P_[min(k + 1, n_ - 1)] - P_[max(k - 1, 0)])
        tk_ = tk_ / float(np.linalg.norm(tk_)) if float(np.linalg.norm(tk_)) > 1e-4 else tot_ / float(np.linalg.norm(tot_))
        nk_ = np.array([-tk_[1], tk_[0]])
        dv_ = b0_ - P_[k]; d_ = abs(float(dv_ @ tk_)); l_ = float(dv_ @ nk_)
        x_ = min(max((dout - d_) / max(dout - din, 1e-3), 0.0), 1.0); w_ = x_ * x_ * (3.0 - 2.0 * x_)
        sh_ = float(wmul) * w_ * max(-lmax, min(lmax, l_))
        mx_ = max(mx_, abs(sh_))
        p_ = P_[k] + sh_ * nk_
        if k == 0:
            s0_ = sh_ * nk_
        out.append(XO.from_xytheta(float(p_[0]), float(p_[1]), math.atan2(float(X[k][1, 0]), float(X[k][0, 0]))))
    return out, mx_, s0_
EXIT_A, EXIT_AW = float(os.environ.get('PICK_EXIT_A', '0.25')), float(os.environ.get('PICK_EXIT_AW', '0.6'))   # 放盒起步：線速度／角速度上限隨時間增加率（m/s²、rad/s²）
BIN_MOVE = os.environ.get('D2_BIN_MOVE', '1') == '1'                  # 放盒／收臂中底盤照導航前端（限速框 BIN_V/BIN_WZ）；0＝底盤錨定
BIN_V, BIN_WZ = float(os.environ.get('D2_BIN_V', '0.15')), float(os.environ.get('D2_BIN_WZ', '0.5'))
DETACH_TO_S = 3.0
MOTM_CLOSE_LATE_S = float(os.environ.get('MOTM_CLOSE_LATE_S', '0.5'))   # 移動中夾取：名目到點後可延後閉爪的上限（底盤仍在走；超過 ⇒ motm_window_missed）
MOTM_W_Q = float(os.environ.get('MOTM_W_Q', '1.0'))                    # 移動中夾取接近段關節姿態引導權重（目前構型→選位抓取解 q_grasp；s74：只給 TCP 目標 ⇒ j2 滑到 −2.02、link4–link6 自身 4.5 mm）
MOTM_EVID_TO_S = float(os.environ.get('MOTM_EVID_TO_S', '0.8'))         # 移動中夾取：閉爪後夾住證據期限（窗口有限）
if BIN:
    import rear_bin as RBIN
    import door_clear_door_fast as _CF_



def xi_from_X(X, dt):
    """參考體速度＝相鄰參考位姿相對運動 /dt（最後一點沿用前一個）；移動參考不可配 xi＝0（s102d：速度被壓、追不上）。"""
    xi = []
    for k in range(len(X) - 1):
        T1 = np.linalg.inv(X[k]) @ X[k + 1]
        xi.append(np.array([T1[0, 2] / dt, T1[1, 2] / dt, math.atan2(T1[1, 0], T1[0, 0]) / dt]))
    xi.append(xi[-1].copy() if xi else np.zeros(3))
    return xi


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

_DRH = None


def _dr_handle_xy():
    """抽屜把手估計位置（場景登錄先驗；把手視覺更新後由抽屜控制器接手，不經此處）。"""
    global _DRH
    if _DRH is None:
        try:
            import drawer_scene as _DS
            _DRH = np.asarray(_DS.handle_T(0.0)[:2, 3], float)
        except Exception:                                             # noqa: BLE001
            _DRH = False
    return None if _DRH is False else _DRH


FURN_CBF_PAD = float(os.environ.get('FURN_CBF_PAD', '0.12')) if os.environ.get('PICK_PATH') == '1' else None


FURN_CBF_PAD_FAR = float(os.environ.get('FURN_CBF_PAD_FAR', '0.16'))  # 離抓取站位 >FURN_NEAR_GOAL 時（Isaac 44：底盤命令斜坡 0.4/0.6 使實際送出速度落後 CBF 一拍，繞角 0.12 m/s 穿邊界約 2 cm；0.42 與核對界 0.40 只差 2 cm）
FURN_NEAR_GOAL = 0.30


def _furn_obs(qp, r_near=0.8, pad=None):
    """與停止軌跡核對同一份場景家具模型（里程）：底盤附近家具的最近點當靜態 CBF 障礙（邊距＝底盤半徑 0.30＋FURN_CBF_PAD）。
    Isaac 40／41：W-GMPC 的障礙來自導航前端（地圖、雷射），與核對用的里程家具模型差幾 cm（AMCL）⇒ 繞茶几角時最佳化以為 0.13、核對算 0.0996 被停。
    牆、隔間、門不加（門口窄，另有門專屬淨空）。"""
    out = []
    sh, _ = P.W._static()
    for name, c, R, hs in sh:
        nm = str(name)
        if not nm.startswith('scene:') or any(k_ in nm for k_ in ('wall', 'partition', 'door')) or not P.W.base_level(c, R, hs):
            continue
        c = np.asarray(c, float); R = np.asarray(R, float); hs = np.asarray(hs, float)
        pl = R.T @ (np.array([float(qp[0]), float(qp[1]), c[2]]) - c)
        pw = c + R @ np.clip(pl, -hs, hs)
        if float(np.hypot(pw[0] - qp[0], pw[1] - qp[1])) < r_near:
            out.append(dict(x=float(pw[0]), y=float(pw[1]), radius=0.0, vx=0.0, vy=0.0, margin=0.30 + (FURN_CBF_PAD if pad is None else pad), static=True))
    return out


FURN_REF_D = float(os.environ.get('FURN_REF_D', '0.48'))             # 參考點離家具外框的中心距下限（遠段 CBF 0.46＋2 cm；Isaac 44 前為 0.44）
_FURN_BOX = None


def _furn_ref(X, bg=None):
    """Isaac 42：Smac 路徑在茶几西北角離桌面外框只有中心距 0.29（雷射只掃到桌腳，桌面角不在 costmap）⇒ 軟 CBF 被追蹤參考拉到 0.40 被停。
    不改導航：只把導航參考視窗中離家具外框 < FURN_REF_D 的點沿外法向推到 FURN_REF_D（同 yield_side 只平移參考；朝向不動）。回傳最大平移量 m。"""
    global _FURN_BOX
    if _FURN_BOX is None:
        _FURN_BOX = []
        sh, _ = P.W._static()
        for name, c, R, hs in sh:
            nm = str(name)
            if nm.startswith('scene:') and not any(k_ in nm for k_ in ('wall', 'partition', 'door')) and P.W.base_level(c, R, hs):
                _FURN_BOX.append((np.asarray(c, float), np.asarray(R, float), np.asarray(hs, float)))
    mx = 0.0
    for k in range(len(X)):
        p = np.array([float(X[k][0, 2]), float(X[k][1, 2])]); p0 = p.copy()
        if bg is not None and float(np.hypot(*(p - bg))) < FURN_NEAR_GOAL:
            continue                                                  # 抓取站位附近不推（站位本身由選位保證 ≥0.42）
        for _ in range(3):
            best = None
            for c, R, hs in _FURN_BOX:
                pl = R.T @ (np.array([p[0], p[1], c[2]]) - c)
                pw = (c + R @ np.clip(pl, -hs, hs))[:2]
                d = float(np.hypot(*(p - pw)))
                if 1e-6 < d < FURN_REF_D and (best is None or d < best[0]):
                    best = (d, pw)
            if best is None:
                break
            p = best[1] + (p - best[1]) / best[0] * FURN_REF_D
        if float(np.hypot(*(p - p0))) > 1e-6:
            Xk = np.array(X[k], float, copy=True); Xk[0, 2], Xk[1, 2] = float(p[0]), float(p[1]); X[k] = Xk
            mx = max(mx, float(np.hypot(*(p - p0))))
    return mx


class TablePickCtrl(P.DoorPassCtrl):
    def __init__(self, src, t0, app):
        """src＝穿門控制器（DRIVE 中）；app＝監督送的取物封包：T_grasp（odom 4×4）、base_goal（odom x,y,yaw）、T（接近時長）、T_b、t_l0。"""
        self.mpc, self.cfg = src.mpc, src.cfg
        self.s = np.asarray(src.s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in src.pipe]
        self.u_last, self.u_prev = np.asarray(src.u_last, float).copy(), np.asarray(src.u_prev, float).copy()
        self.lim_lo, self.lim_hi = src.lim_lo, src.lim_hi
        self.nav_in = getattr(src, 'nav_in', None)
        self.T_WH0, self.phi_now = src.T_WH0, src.phi_now
        self._pool = getattr(src, '_pool', None)
        self.t, self.n_step = float(t0), 0
        self.stop, self.abort, self.done, self.finished = None, None, False, False
        self.release, self.unbound, self.finger_cmd, self.finger_meas, self.finger_force = True, True, P.FINGER_OPEN, src.finger_meas, None
        self.tail_cert_mode = False
        self._rec_tail = {1: []}
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
        self.events, self.wg_log, self.tick_log, self.sweep_log = [], [], [], []
        self.clear_min = {k: 9.0 for k in P.KEYS}
        self.T_rel_stow = src.T_rel_stow
        self.T_HE, self.door_end = getattr(src, 'T_HE', None), getattr(src, 'door_end', None)   # 回程關門用（沿控制器鏈傳遞）
        self.t_halt, self.t_still = None, None
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()
        self.T_grasp[:3, 3] = self.T_grasp[:3, 3] + GRASP_DEPTH * self.T_grasp[:3, 2]   # 同瓶子線：TCP 沿工具軸越過圓柱軸 4 mm，軸心落在指片（TCP 後 1.3–21.3 mm）內（第十二趟 TCP 在軸心上 ⇒ 指尖夾弦、抬升滑落）
        self.app_prof = dict(base_goal=list(app['base_goal']), door=None)   # nav_window：導航到點後收斂到抓取底盤位姿
        self.T_app, self.T_b, self.t_l0 = float(app['T']), float(app['T_b']), float(app['t_l0'])
        self._bg, self._t_arr, self._t_line0, self._bp = np.asarray(app['base_goal'], float)[:2].copy(), None, None, 0.0
        self.T_rel0, self.b_anchor, self.n_ev, self.T_lift0, self.T_rel_s0 = None, None, 0, None, None
        self.ret_sent, self.attach = False, False
        self.obj_release, self.obj_detached, self.binned, self.q_b0, self.T_bm, self.bin_target = False, False, False, None, None, None
        if BIN:
            RBIN.install(_CF_)                                        # 盒體進自碰撞淨空（取物控制器建立時才裝；開門／穿門段不變）
        self.phase, self.t_ph0 = 'APPROACH', float(t0)
        self.motm = app.get('motm')                                   # Howard 11/19 移動中夾取：dict(b0, e, v_g, s0)（里程）；None＝原停車夾取
        self.flow = app.get('flow') if type(self) is TablePickCtrl else None   # Howard 11/21 晚：移動中夾取照導航（底盤參考＝導航前端；抓取點附近只改偏航與限速）
        self._jq0, self._jqg, self._ex_u0 = None, None, None                  # 關節內插展開（JBLEND）起點／終點；放盒起步斜坡起始速度
        self.m_s0, self.m_t0, self.T_rel_l0 = None, None, None
        self.m_G, self.m_ta0 = None, None
        self._ev('pick_start', T_grasp=[round(float(x), 4) for x in self.T_grasp[:3, 3]], base_goal=[round(float(x), 4) for x in app['base_goal']], T=round(self.T_app, 3))

    def _stop(self, why, /, **kw):
        if 'why' in kw:
            kw['detail_why'] = kw.pop('why')                          # 違反 dict 自帶 why（如 clearance）⇒ 改名，避免與停止原因重複
        if self.stop is None:
            self.stop = dict(why=why, t=round(self.t, 3), **kw)
            self._ev('pick_stop', why=why, phase=self.phase, **kw)

    def _goto(self, ph, **kw):
        self.phase, self.t_ph0 = ph, self.t
        self._ev('pick_phase', phase=ph, **kw)

    def _T_line(self, s_):
        T = self.T_grasp.copy()
        T[:3, 3] = T[:3, 3] - s_ * T[:3, 2]                            # 沿接近軸（TCP z）後退 s
        return T

    def _app_gated(self):
        return (APP_GATE and self.motm is None and getattr(self, '_bg', None) is not None
                and (type(self) is TablePickCtrl or (DR_APP_GATE and getattr(self, 'task', None) == 'handle')))

    def _d1(self):
        if self._jb_active():
            return max(FLOW_D1, JB_D1)
        return FLOW_D1 if getattr(self, 'flow', None) else APP_GATE_D1

    def _tbmin(self):
        if self._jb_active():                                           # 關節內插展開：最短展開時間依最大關節位移（最小急動峰速 1.875·Δq/T ≤ JB_QD；s215d 1.0 rad/s 打到上限有波紋）
            return max(T_BMIN, 1.875 * float(np.abs(self._jqg - self._jq0).max()) / JB_QD)
        return T_BMIN

    def _wp_app(self):
        if self.phase != 'APPROACH':
            return C0.W_NEAR
        ta_ = getattr(self, '_t_arr', None) if self.flow else None
        if ta_ is None:
            return C0.W_FAR
        if not (JBLEND and FLOW_VRAMP and getattr(self, '_jqg', None) is not None):   # 只改圓柱關節內插展開那條（抽屜把手照舊）
            return C0.W_NEAR
        u_ = XE.minjerk(min(max((self.t - ta_) / 1.0, 0.0), 1.0))[0]   # 到站後 1 s 平順加權（s215d：一步跳到 W_NEAR ⇒ 手臂該拍 0.85 rad/s 尖峰）
        return C0.W_FAR + (C0.W_NEAR - C0.W_FAR) * u_

    def _jb_active(self):
        return bool(JBLEND and getattr(self, 'flow', None) and self.phase == 'APPROACH' and self._app_gated()
                    and getattr(self, '_jqg', None) is not None and getattr(self, '_jq0', None) is not None)

    def _dl(self):
        return float(getattr(self, 'd_line', D_LINE))                   # 抓取線長＝預抓高度（取筆可為 0.12，見 drawer_reach.PRE_TIERS）

    def _tls(self):
        v_ = LINE_V_CYL if (getattr(self, 'flow', None) and getattr(self, '_jqg', None) is not None) else 0.06   # 圓柱照導航（關節內插那條）可用較快抓取線
        return max(0.8 if v_ <= 0.06 else 0.5, self._dl() / v_)        # 抓取線時長（同 T_LINE_S 公式，依本段線長）

    def _app_target(self, t_abs, Xk):
        """接近剖面：參考底盤上的起始相對位姿 → 抓取線（d_line→0，接近線提早 t_settle 到底）。"""
        t = t_abs - self.t_ph0
        Te = self.T_app - T_SETTLE
        u = min(max((t - self.t_l0) / max(Te - self.t_l0, 1e-6), 0.0), 1.0)
        s_ = self._dl() * (1.0 - XE.minjerk(u)[0]) if t < Te else 0.0
        Tl = self._T_line(max(s_, 0.0))
        beta = XE.minjerk(min(max(t / max(self.T_b, 1e-6), 0.0), 1.0))[0]
        if getattr(self, 'm_ta0', None) is not None:                    # 併線：偏航轉到 80% 後才伸手
            ta_ = self.m_ta0 - self.t_ph0
            beta = XE.minjerk(min(max((t - ta_) / max(self.T_b - ta_, 1e-6), 0.0), 1.0))[0]
        if self._app_gated():                                           # Isaac 43：時間表 β 在繞茶几角時已 0.68 ⇒ TCP 追蹤把全身拉向桌子另一側的圓柱、蓋過軟 CBF。Howard 11/20「何時伸手依剩餘距離」⇒ 伸手與抓取線都由參考底盤到抓取站位的距離決定（時間表只當上限）
            d_ = float(math.hypot(Xk[0] - self._bg[0], Xk[1] - self._bg[1]))
            d1_ = self._d1()
            d0_ = FLOW_D0 if getattr(self, 'flow', None) else APP_GATE_D0   # 照導航：提早伸手（經過時手臂已到位）
            p_ = min(max((d0_ - d_) / (d0_ - d1_), 0.0), 1.0)
            p_ = min(p_, getattr(self, '_bp', 0.0) + max(t_abs - self.t, 0.0) / self._tbmin())   # 進度限速（狀態觸發、不一步跳）
            beta = XE.minjerk(p_)[0]   # 只看狀態（Isaac 53 抽屜：時間表上限使手臂在底盤到位後再慢慢伸 16 s）
            tl0 = getattr(self, '_t_line0', None)
            u = 0.0 if tl0 is None else min(max((t_abs - tl0) / self._tls(), 0.0), 1.0)   # 抓取線：底盤到站且 TCP 到預抓點後才走
            s_ = self._dl() * (1.0 - XE.minjerk(u)[0])
            Tl = self._T_line(max(s_, 0.0))
        jq_ = self._jb_active()
        ao_ = getattr(self, 'aim_off', None)
        if ao_ is not None and (getattr(self, 'app_tol_fy', None) is not None or getattr(self, 'flow', None)) and float(np.abs(ao_).max()) > 0.0:
            Tl = Tl.copy(); Tl[:3, 3] = Tl[:3, 3] + ao_                 # 瞄準補償只加在指令目標
        if jq_:
            q_ref = self._jq0 + min(max(beta, 0.0), 1.0) * (self._jqg - self._jq0)
            if isinstance(getattr(self, '_jq_refs', None), list):
                self._jq_refs.append(q_ref)
            if beta >= 1.0 - 1e-12:
                return Tl
            Tj = K.fk(np.r_[float(Xk[0]), float(Xk[1]), float(Xk[2]), q_ref], 'link_tcp')
            gam = XE.minjerk(min(max((beta - JB_G0) / max(1.0 - JB_G0, 1e-6), 0.0), 1.0))[0]
            if gam <= 1e-12:
                return Tj
            T = np.eye(4)
            T[:3, 3] = (1 - gam) * Tj[:3, 3] + gam * Tl[:3, 3]
            T[:3, :3] = Tj[:3, :3] @ _exp_R(gam * _log_R(Tj[:3, :3].T @ Tl[:3, :3]))
            return T
        if beta >= 1.0 - 1e-12:
            return Tl
        c_, s2 = math.cos(Xk[2]), math.sin(Xk[2])
        Tb = np.eye(4); Tb[:2, :2] = [[c_, -s2], [s2, c_]]; Tb[:2, 3] = Xk[:2]
        Tb = Tb @ self.T_rel0
        T = np.eye(4)
        T[:3, 3] = (1 - beta) * Tb[:3, 3] + beta * Tl[:3, 3]
        T[:3, :3] = Tb[:3, :3] @ _exp_R(beta * _log_R(Tb[:3, :3].T @ Tl[:3, :3]))
        return T

    def _motm_t_arm(self, q):
        """手臂展開到抓取解所需時間：最大關節位移 ÷ 0.6 rad/s（最小急動峰速 1.875×）＋0.5 s，至少 t_arm（展開與底盤移動重疊、不停車）。"""
        t_a = float(self.motm.get('t_arm', 4.0))
        if self.motm.get('q_grasp') is not None:
            t_a = max(t_a, 1.875 * float(np.abs(np.asarray(self.motm['q_grasp'], float) - np.asarray(q[3:], float)).max()) / 0.6 + 0.5)
        return t_a

    def path_ext(self, app, t0):
        """依導航路徑的移動中抓取（Howard 11/20）：監督依實際行進方向線上算出經過方向與速度（cyl_reach_map.path_ext）後送來；
        自本拍起底盤沿該方向以斜坡速度經過抓取位（自目前投影起算、無對齊），其餘沿用移動中夾取的閉爪窗／確認／抬升。"""
        if self.phase == 'UNFOLD':                                    # 抽屜：手臂還在邊走邊展開 ⇒ 展開完進接近段時再套用
            self.pending_ext = (app, float(t0))
            self._ev('pick_path_ext_pending')
            return
        if self.phase != 'APPROACH' or self.motm is not None:
            self._ev('pick_path_ext_ignored', phase=self.phase)
            return
        self.motm = dict(app['motm'])
        self.m_s0, self.m_t0 = None, None
        self.app_qpost = None
        self.t = float(t0)
        self._goto('APPROACH', path_ext=dict(v_g=round(float(self.motm['v_g']), 3), s_grasp=self.motm.get('s_grasp'), e=[round(float(x), 3) for x in self.motm['e']]))
        self._reset_T_rel0 = True                                     # 下一拍 _advance 開頭由目前相對位姿重新混合到抓取線（不可設 None：同拍 _solve 要用，s119d 崩潰）

    def _merge_init(self, q, b0, e):
        v_ = float(self.motm['v_g'])
        cq_, sq_ = math.cos(float(q[2])), math.sin(float(q[2])); up_ = np.asarray(self.u_prev[:2], float)
        vw_ = np.array([cq_ * up_[0] - sq_ * up_[1], sq_ * up_[0] + cq_ * up_[1]])   # 目前世界速度（前一筆命令）
        t_arm_ = self._motm_t_arm(q)
        s_r = float(self.motm.get('s0', -0.05)) - 0.05 - v_ * t_arm_    # 偏航轉完、開始伸手的線上位置
        G_ = MM.merge_geom(q[:2], vw_ if float(np.linalg.norm(vw_)) > 0.03 else np.zeros(2), b0, e, s_cap=s_r)
        self.m_G, self.m_y0 = G_, float(q[2])
        self.m_v0 = float(np.clip(np.linalg.norm(vw_), 0.0, 0.2))
        self.m_sigrot = MM.sig_rot_for(G_, q[2], b0[2], s_r, min(max(self.m_v0, v_, 0.15), MERGE_V_HI))   # 偏航參考峰值角速度以併線段速度換算
        self.m_t0, self.m_s0 = self.t, float(G_['s_p'])
        self.m_law = MM.time_law(G_, self.m_v0, v_, s_r, v_hi_cap=MERGE_V_HI)   # 併線段維持進來的速度（≤MERGE_V_HI），伸手起點前減到經過速度
        sig_g = MM.sig_of_s(G_, 0.0)
        self.T_app = MM.t_at_sig(self.m_law, sig_g, v_)                 # 複合路徑上經過 s＝0 的時刻
        t_rot = MM.t_at_sig(self.m_law, self.m_sigrot, v_)
        self.m_ta0 = self.m_t0 + 0.8 * t_rot                            # 偏航轉到 80% 才開始伸手（收臂構型在底盤半徑內，轉向中不掃桌）
        ta_rel = self.m_ta0 - self.t_ph0
        self.T_b = max(ta_rel + 1.5, self.T_app - 1.5)
        self.t_l0 = max(ta_rel + 1.0, self.T_app - 2.5)
        if self.motm.get('q_grasp') is not None and MOTM_W_Q > 0.0:
            self.q_m0, self.q_mg = np.asarray(q[3:], float).copy(), np.asarray(self.motm['q_grasp'], float)
            self.app_qpost, self.app_w_q = self._motm_qpost, MOTM_W_Q
        self._ev('pick_merge_start', lat=round(G_['lat'], 3), s_p=round(G_['s_p'], 3), s_m=round(G_['s_m'], 3), Lc=round(G_['Lc'], 3), sig_rot=round(self.m_sigrot, 3),
                 dyaw=round(abs((float(b0[2]) - float(q[2]) + math.pi) % (2 * math.pi) - math.pi), 3), v0=round(self.m_v0, 3), T_app=round(self.T_app, 2), t_reach=round(ta_rel, 2), t_arm=round(t_arm_, 2))

    def _m_sig(self, t_abs):
        return MM.sig_at_t(self.m_law, max(0.0, t_abs - self.m_t0), float(self.motm['v_g']))

    def _line_ready(self, q, b0, e, s_st):
        """進線條件（Howard 11/20 茶几停頓）：偏航 ≤0.15 rad、離路線側向 ≤5 cm、沿線不超過起點 5 cm（不必停在起點上）。"""
        d_ = np.asarray(q[:2], float) - np.asarray(b0[:2], float)
        lat_ = abs(float(d_[0] * -e[1] + d_[1] * e[0]))
        dyaw_ = abs((float(q[2]) - float(b0[2]) + math.pi) % (2 * math.pi) - math.pi)
        return dyaw_ <= 0.15 and lat_ <= 0.05 and float(np.dot(d_, e)) <= s_st + 0.05

    def _motm_s_lin(self, t_abs):
        tau = max(0.0, t_abs - self.m_t0); v, v0 = float(self.motm['v_g']), float(getattr(self, 'm_v0', 0.0))
        return self.m_s0 + v * tau - (v - v0) * MOTM_TR * (1.0 - math.exp(-tau / MOTM_TR))   # 速度自進線時實際沿線速度 v0 以 TR 收斂到 v_g（不再階躍 ⇒ 不先追趕後等待）

    def _motm_v_lin(self, t_abs):
        tau = max(0.0, t_abs - self.m_t0); v, v0 = float(self.motm['v_g']), float(getattr(self, 'm_v0', 0.0))
        return v - (v - v0) * math.exp(-tau / MOTM_TR)

    def _motm_s(self, t_abs):
        """可行段終點 s_end（依導航路徑：正面接近時後面就是家具）：進入最後 v·TR 時指數減速、平滑停在 s_end。"""
        if getattr(self, 'm_G', None) is not None and self.motm.get('merge'):
            return MM.s_line(self.m_G, self._m_sig(t_abs))
        s_ = self._motm_s_lin(t_abs); se = self.motm.get('s_end')
        if se is None:
            return s_
        w = max(float(self.motm['v_g']) * MOTM_TR, 1e-3); s1 = float(se) - w
        return s_ if s_ <= s1 else float(se) - w * math.exp(-(s_ - s1) / w)

    def _motm_v(self, t_abs):
        v_ = self._motm_v_lin(t_abs); se = self.motm.get('s_end')
        if se is None:
            return v_
        w = max(float(self.motm['v_g']) * MOTM_TR, 1e-3); s1 = float(se) - w; s_ = self._motm_s_lin(t_abs)
        return v_ if s_ <= s1 else v_ * math.exp(-(s_ - s1) / w)

    def _motm_ref(self, t_eff):
        """經過路線參考：底盤沿 e 等速 v_g、朝向固定（全向底盤橫移）；回 (X 視窗, xi 視窗)。"""
        b0, e, v = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float), float(self.motm['v_g'])
        if getattr(self, 'm_G', None) is not None and self.motm.get('merge'):   # 併線複合路徑：位置沿弧長、偏航最小急動轉到線上；體速度由相鄰參考差分
            X = []
            for k in range(XO.N + 1):
                sg_ = self._m_sig(t_eff + k * DT); p_ = MM.at(self.m_G, sg_)
                X.append(XO.from_xytheta(float(p_[0]), float(p_[1]), MM.yaw_at(self.m_G, sg_, self.m_y0, b0[2], self.m_sigrot)))
            return X, xi_from_X(X, DT)
        th = float(b0[2]); c_, s_ = math.cos(th), math.sin(th)
        X = [XO.from_xytheta(b0[0] + self._motm_s(t_eff + k * DT) * e[0], b0[1] + self._motm_s(t_eff + k * DT) * e[1], th) for k in range(XO.N + 1)]
        xi = []
        for k in range(XO.N + 1):
            vk = self._motm_v(t_eff + k * DT)
            xi.append(np.array([c_ * vk * e[0] + s_ * vk * e[1], -s_ * vk * e[0] + c_ * vk * e[1], 0.0]))   # 世界速度轉本體（含進線速度斜坡）
        return X, xi

    def _motm_qpost(self, t_abs):
        if getattr(self, 'm_ta0', None) is not None:                    # 併線：偏航轉完才開始引導
            u = XE.minjerk(min(max((t_abs - self.m_ta0) / max(self.t_ph0 + self.T_app - T_SETTLE - self.m_ta0, 1e-6), 0.0), 1.0))[0]
            return (1.0 - u) * self.q_m0 + u * self.q_mg
        u = XE.minjerk(min(max((t_abs - self.m_t0) / max(self.T_app - T_SETTLE, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_m0 + u * self.q_mg

    def _lift_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_LIFT, 0.0), 1.0))[0]
        T = self.T_lift0.copy()
        T[2, 3] += u * LIFT_M
        return T

    def _carry_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_CARRY, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]                   # 只移位置：夾爪相對底盤姿態保持抬升時（圓柱維持直立；第十四趟混到收臂姿態 ⇒ 手腕繞接近軸轉、圓柱轉橫滑落）
        T[:3, :3] = A[:3, :3]
        return T

    def _held(self):
        f, fm = self.finger_force, self.finger_meas
        return f is not None and fm is not None and min(f) >= CONTACT_N and min(fm) >= FINGER_EMPTY

    def _advance(self, q):
        el = self.t - self.t_ph0
        if self.T_rel0 is None or getattr(self, '_reset_T_rel0', False):
            self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            self._reset_T_rel0 = False
        if JBLEND and self.flow and self.phase == 'APPROACH' and getattr(self, '_jq0', None) is None and self.flow.get('q_grasp') is not None:
            self._jq0, self._jqg = np.asarray(q[3:], float).copy(), np.asarray(self.flow['q_grasp'], float).copy()
            self._ev('pick_jblend_start', dq_max=round(float(np.abs(self._jqg - self._jq0).max()), 3))
        if self.motm is not None and self.m_s0 is None and self.phase == 'APPROACH' and not self.motm.get('no_align') and not self.motm.get('merge'):   # 進線前：底盤先到路線起點（位置＋偏航）再開始等速橫移
            b0, e = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float)   # （s76：偏航差 2.9 rad 邊轉邊走擦桌角；第三十一趟 Isaac：只轉不移、離線 0.31 m ⇒ 進線偏航漂、淨空 0.100）
            s_st = min(float(np.dot(np.asarray(q[:2], float) - b0[:2], e)), float(self.motm.get('s0', -0.05)) - 0.10, -float(self.motm['v_g']) * self._motm_t_arm(q))
            b_st = np.array([b0[0] + s_st * e[0], b0[1] + s_st * e[1], float(b0[2])])
            dyaw_ = abs((float(q[2]) - float(b0[2]) + math.pi) % (2 * math.pi) - math.pi)
            dpos_ = float(np.hypot(*(np.asarray(q[:2], float) - b_st[:2])))
            if not self._line_ready(q, b0, e, s_st):
                self.b_anchor = b_st
                self._goto('ALIGN', dyaw=round(dyaw_, 3), dpos=round(dpos_, 3), s_start=round(s_st, 3))
                return
        if self.phase == 'ALIGN':
            dyaw_ = abs((float(q[2]) - float(self.b_anchor[2]) + math.pi) % (2 * math.pi) - math.pi)
            dpos_ = float(np.hypot(*(np.asarray(q[:2], float) - self.b_anchor[:2])))
            b0_, e_ = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float)
            if self._line_ready(q, b0_, e_, float(np.dot(self.b_anchor[:2] - b0_[:2], e_))):
                self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._goto('APPROACH', aligned_s=round(self.t - self.t_ph0, 2))
            elif self.t - self.t_ph0 > 15.0:
                self._stop('motm_align_timeout', dyaw=round(dyaw_, 3), dpos=round(dpos_, 3))
            return
        if self.motm is not None and self.m_s0 is None:               # 移動中夾取時程：自目前底盤在路線上的投影起算，名目到點＝底盤經過 s＝0
            b0, e = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float)
            if self.motm.get('merge'):                                  # 邊走邊併線（Howard 11/21「要真的移動中抓」）：不原地對齊；自目前位置／速度以平滑曲線併入經過線，偏航在伸手前轉完
                self._merge_init(q, b0, e)
                return
            if self.motm.get('no_align'):                              # 依導航路徑（Howard 11/20 不要預設路線）：自目前投影起算、不後退、不對齊
                self.m_s0 = min(float(np.dot(np.asarray(q[:2], float) - b0[:2], e)), -0.02)
            else:
                self.m_s0 = min(float(np.dot(np.asarray(q[:2], float) - b0[:2], e)), float(self.motm.get('s0', -0.05)) - 0.10,
                                -float(self.motm['v_g']) * self._motm_t_arm(q))   # 手臂展開至少 t_arm（s73）
            self.m_t0 = self.t
            cq_, sq_ = math.cos(float(q[2])), math.sin(float(q[2])); up_ = np.asarray(self.u_prev[:2], float)
            self.m_v0 = float(np.clip(np.dot(np.array([cq_ * up_[0] - sq_ * up_[1], sq_ * up_[0] + cq_ * up_[1]]), e), 0.0, float(self.motm['v_g'])))   # 進線時已有的沿線速度
            self.T_app = (0.0 - self.m_s0 + (float(self.motm['v_g']) - self.m_v0) * MOTM_TR) / float(self.motm['v_g'])   # 斜坡下經過 s＝0 的時刻（e^{-T/TR} 項可忽略）
            self.T_b, self.t_l0 = max(1.0, self.T_app - 1.5), max(0.5, self.T_app - 2.5)
            if self.motm.get('q_grasp') is not None and MOTM_W_Q > 0.0:   # 底盤經過 s＝0 時恰在選位底盤位 ⇒ 選位抓取解即該時刻構型
                self.q_m0, self.q_mg = np.asarray(q[3:], float).copy(), np.asarray(self.motm['q_grasp'], float)
                self.app_qpost, self.app_w_q = self._motm_qpost, MOTM_W_Q
            self._ev('pick_motm_start', s0=round(self.m_s0, 4), T_app=round(self.T_app, 3), v_g=self.motm['v_g'])
        if self.phase == 'APPROACH':
            Tc = K.fk(q, 'link_tcp')
            qm_ = getattr(self, '_q_meas', None)
            Tcm = K.fk(qm_, 'link_tcp') if (MEAS_GATE and qm_ is not None and getattr(self, 'app_tol_fy', None) is not None) else Tc   # 固定橫桿：開合軸門檻與瞄準補償看實測 TCP（預測狀態不含手臂重力下垂）
            ep = float(np.linalg.norm(Tc[:3, 3] - self.T_grasp[:3, 3]))
            er = math.degrees(float(np.linalg.norm(_log_R(self.T_grasp[:3, :3].T @ Tc[:3, :3]))))
            tz_ = getattr(self, 'app_tol_z', APP_TOL_Z)
            if tz_ is not None:                                       # 分方向：沿圓柱軸的高度差不影響側夾
                d_ = Tc[:3, 3] - self.T_grasp[:3, 3]
                ep_ok = float(np.linalg.norm(d_[:2])) <= 0.005 and abs(float(d_[2])) <= tz_
            else:
                ep_ok = ep <= 0.005
            fy_ = getattr(self, 'app_tol_fy', None)
            gated_ = self._app_gated()
            if gated_:                                                  # 伸手進度：實際底盤距離決定、每拍至多前進 DT/T_BMIN
                d0g_ = FLOW_D0 if self.flow else APP_GATE_D0
                pd_ = min(max((d0g_ - float(math.hypot(q[0] - self._bg[0], q[1] - self._bg[1]))) / (d0g_ - self._d1()), 0.0), 1.0)
                self._bp = min(max(pd_, getattr(self, '_bp', 0.0)), getattr(self, '_bp', 0.0) + NPH * PHYS / self._tbmin())
            if gated_:                                                  # 狀態閘門：底盤到站、TCP 到預抓點 ⇒ 啟動抓取線
                if self.flow:                                          # 照導航：偏航權重只增不減（經過抓取點後不轉回）；錯過偵測（已到站但底盤離開 0.45 m 仍未閉爪 ⇒ 退回收斂）
                    df_ = float(math.hypot(q[0] - self._bg[0], q[1] - self._bg[1]))
                    rt_, ri_ = float(self.flow.get('r_turn', 1.0)), float(self.flow.get('r_in', 0.35))
                    self._flow_w = max(getattr(self, '_flow_w', 0.0), min(max((rt_ - df_) / max(rt_ - ri_, 1e-3), 0.0), 1.0))
                    if self._t_arr is not None and df_ > FLOW_MISS_R:
                        self._ev('pick_flow_missed', d=round(df_, 3), el=round(el, 2))
                        self.flow, self._t_arr, self._t_line0 = None, None, None
                if self._t_arr is None and float(math.hypot(q[0] - self._bg[0], q[1] - self._bg[1])) <= (FLOW_D1 if self.flow else APP_GATE_D1):
                    self._t_arr = self.t
                    self._ev('pick_base_arrived', el=round(el, 2))
                if self._t_arr is not None and getattr(self, '_t_line0', None) is None and float(np.linalg.norm(Tc[:3, 3] - self._T_line(self._dl())[:3, 3])) <= (FLOW_LINE_TOL if (self.flow and getattr(self, 'app_tol_fy', None) is None) else 0.01) and not (self.flow and getattr(self, 'app_tol_fy', None) is not None and abs(float(Tcm[2, 3] - self._T_line(self._dl())[2, 3])) > FLOW_HDL_DZ):   # 照導航抓把手另要求高度差 ≤ FLOW_HDL_DZ（底盤在動時手臂下垂）；把手（橫桿窄）維持 1 cm（Isaac 83：2 cm 放行時 z 下垂 −8 mm ⇒ 手指撞桿 42 N 鎖存）
                    self._t_line0 = self.t
                    self._ev('pick_line_start', el=round(el, 2))
            if (fy_ is not None and ((not gated_ and el >= self.T_app - T_SETTLE) or (gated_ and getattr(self, '_t_line0', None) is not None and ep < AIM_START_M) or (self.motm is not None and ep < AIM_START_M))) or (self.flow and getattr(self, '_t_line0', None) is not None and self.t - self._t_line0 >= self._tls() and ep < 0.03 and getattr(self, '_vtcp_last', 9.0) <= 0.02):   # 照導航夾取：底盤在動，手臂追蹤穩態偏差（s159d 13 mm）⇒ 同把手的積分瞄準補償   # 瞄準補償（積分；移動中抓取窗短 ⇒ 誤差 <1 cm 即開始累積）：Isaac 手臂穩態誤差約 3 mm（第三十五趟 approach_not_reached 3.31 mm）⇒ 指令目標加上累積誤差，判定仍用真目標   # s163d：抓取線還在走（暫態落後 20 mm）就積分 ⇒ 補償飽和 +15 mm、過衝夾空 ⇒ 抓取線走完且 TCP 幾乎不動（穩態偏差）才積分
                ao_ = getattr(self, 'aim_off', np.zeros(3))
                self.aim_off = np.clip(ao_ + AIM_K * (self.T_grasp[:3, 3] - Tcm[:3, 3]), -(0.02 if self.flow else AIM_MAX), (0.02 if self.flow else AIM_MAX))
            elif self.flow and fy_ is not None and gated_ and self._t_arr is not None and getattr(self, '_t_line0', None) is None and getattr(self, '_vtcp_last', 9.0) <= 0.02:   # 照導航抓把手：停在預抓點時的穩態下垂（Isaac 84 z −9.9 mm 持續 5 s）⇒ 對預抓點積分同一瞄準補償，抓取線沿用
                Tpre_ = self._T_line(self._dl())[:3, 3]
                if float(np.linalg.norm(Tcm[:3, 3] - Tpre_)) < 0.02 + float(np.abs(getattr(self, 'aim_off', np.zeros(3))).max()):
                    ao_ = getattr(self, 'aim_off', np.zeros(3))
                    self.aim_off = np.clip(ao_ + AIM_K * (Tpre_ - Tcm[:3, 3]), -0.02, 0.02)
            if fy_ is not None:                                       # 固定橫桿（抽屜把手）：沿兩指開合軸（TCP y）誤差另設門檻（Isaac 33b／34b：5 mm 球內閉爪、開合軸偏 ~2 mm ⇒ 單指接觸、證據逾時）
                ep_fy = abs(float((self.T_grasp[:3, :3].T @ (Tcm[:3, 3] - self.T_grasp[:3, 3]))[1]))
                ep_ok = ep_ok and ep_fy <= fy_
            v_tcp = float(np.linalg.norm(Tc[:3, 3] - self._tc_prev)) / (NPH * PHYS) if getattr(self, '_tc_prev', None) is not None else 9.0
            self._tc_prev = Tc[:3, 3].copy()
            self._vtcp_last = v_tcp
            path_ok = self.motm is not None and self.motm.get('path') and self.m_s0 is not None and v_tcp <= 0.02   # 依導航路徑：TCP 已到位且相對目標（世界固定）幾乎不動 ⇒ 即閉爪，不等名目時刻（Howard 11/20 時機由狀態決定）
            gate_ok = gated_ and self._t_arr is not None and getattr(self, '_t_line0', None) is not None and self.t - self._t_line0 >= self._tls() and v_tcp <= 0.02   # 狀態閘門：抓取線走完、TCP 到位幾乎不動 ⇒ 即閉爪
            if (el >= self.T_app or path_ok or gate_ok) and ep_ok and er <= 3.0:
                self.b_anchor = np.asarray(q[:3], float).copy()
                self.finger_cmd = P.FINGER_OPEN
                self._goto('CLOSE', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
            elif self.motm is not None and self.motm.get('path') and el > self.T_app + MOTM_CLOSE_LATE_S:   # 依導航路徑：經過窗錯過 ⇒ 退回收斂抓取（不中止；Howard 11/20 安全時限保留在收斂段）
                self._ev('pick_path_fallback', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2), s=round(self._motm_s(self.t), 4))
                self.motm, self.m_s0, self.m_t0, self.m_G, self.m_ta0 = None, None, None, None, None
                self.app_qpost = None                                    # 移動中姿態引導依 m_t0（s122d：退回後仍用 ⇒ TypeError）
                self.T_app, self.T_b, self.t_l0 = 4.0, 2.5, 1.5
                self._t_arr, self._t_line0, self._bp = None, None, 0.0
                self._reset_T_rel0 = True
                self._goto('APPROACH', fallback=True)
            elif self.motm is not None and el > self.T_app + MOTM_CLOSE_LATE_S:
                self._stop('motm_window_missed', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2), s=round(self._motm_s(self.t), 4))
            elif gated_:                                                # 狀態閘門：底盤到站後才計抓取逾時（安全總時限另保留）
                if (self._t_arr is not None and self.t - self._t_arr > PH_TO_S + T_SETTLE + 2.0 and not self.flow) or el > self.T_app + APP_GATE_CAP_S:   # 照導航：到站逾時不停（由錯過判定退回收斂）
                    self._stop('approach_not_reached', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2), gated=True)
            elif el > self.T_app + PH_TO_S:
                self._stop('approach_not_reached', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
        elif self.phase == 'CLOSE':
            self.finger_cmd = max(0.0, P.FINGER_OPEN * (1.0 - el / T_CLOSE))
            if el >= T_CLOSE:
                self.finger_cmd, self.n_ev = 0.0, 0
                self._goto('CONFIRM')
        elif self.phase == 'CONFIRM':
            self.n_ev = self.n_ev + 1 if self._held() else 0
            if self.n_ev * DT >= EVID_S - 1e-9:
                self.T_lift0 = K.fk(q, 'link_tcp')
                self.T_rel_l0 = np.linalg.inv(_T3(q[:3])) @ self.T_lift0     # 移動中夾取：連接後抬升改相對底盤（物體已理想連接，不再受世界固定可達窗限制）
                self.attach = True                                    # Howard 11/18「夾到就改成理想連接」：夾住證據成立 ⇒ 請執行端以 FixedJoint 固定圓柱（同門把理想綁定）
                self._goto('LIFT', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
            elif el > (MOTM_EVID_TO_S if self.motm is not None else EVID_TO_S):
                self._stop('grasp_evidence_timeout', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
        elif self.phase == 'LIFT':
            if el >= T_LIFT:
                self.n_ev = 0
                self._goto('VERIFY')
        elif self.phase == 'VERIFY':
            self.n_ev = self.n_ev + 1 if self._held() else 0
            if self.n_ev * DT >= EVID_S - 1e-9:
                self.T_rel_s0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._ev('pick_lifted', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
                if BIN and not getattr(self, 'bin_carry_first', False):
                    self._bin_start(q, 'BIN_MOVE')
                    return
                if BIN:                                               # 抽屜筆：先兩段式收臂抬過面板頂再轉去放盒（s63：抬升位直接轉 ⇒ link_eef 掃抽屜面板 5 mm 被停）
                    self._goto('CARRY')
                    return
                self._goto('CARRY' if PICK_CARRY else 'RETURN')               # 預設不收臂：手臂相對底盤保持抬升位姿（第十五趟收臂一開始手腕重配置 ⇒ 接觸瞬斷滑落；瓶子線持瓶也是手臂不動、底盤移動）
            elif el > EVID_TO_S:
                self._stop('lost_after_lift', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
        elif self.phase == 'CARRY':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self._carry_target(self.t_ph0 + T_CARRY)[:3, 3]))
            if el >= T_CARRY and ep <= 0.010:
                self._goto('RETURN', carry_err_mm=round(ep * 1e3, 2))
            elif el > T_CARRY + PH_TO_S:
                self._stop('carry_not_reached', carry_err_mm=round(ep * 1e3, 2))
            if not self._held():
                self._stop('lost_during_carry')
        elif self.phase in ('BIN_MOVE', 'BIN_REL', 'BIN_OPEN', 'BIN_DROP', 'BIN_STOW'):
            self._advance_bin(q, el)
        elif self.phase == 'RETURN':
            if not self.binned and not self._held():
                self._stop('lost_during_return')

    # ---- 車尾收納盒（D2_BIN）
    def _bin_start(self, q, ph):
        self.q_b0 = np.asarray(q[3:], float).copy()
        self.bin_target = RBIN.q_place()[0] if ph == 'BIN_MOVE' else P.Q_STOW
        self.T_bm = 1.875 * float(np.abs(self.bin_target - self.q_b0).max()) / BIN_QD + 0.5
        self._goto(ph, T=round(self.T_bm, 2), q_target=[round(float(x), 4) for x in self.bin_target])

    def _bin_q(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_bm, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_b0 + u * self.bin_target

    def _advance_bin(self, q, el):
        fm = self.finger_meas
        if self.phase in ('BIN_MOVE', 'BIN_STOW'):
            err = float(np.abs(np.asarray(q[3:], float) - self.bin_target).max())
            if self.phase == 'BIN_MOVE' and not self._held():
                self._stop('lost_during_bin_move', finger=None if fm is None else [round(float(x), 5) for x in fm])
            elif el >= self.T_bm and err <= 0.03:
                if self.phase == 'BIN_MOVE':
                    self.obj_release = True                           # 請執行端解除夾爪↔物體理想連接（FixedJoint）
                    self._goto('BIN_REL', err=round(err, 4))
                else:
                    self._ev('pick_stowed', err=round(err, 4), base=[round(float(x), 4) for x in q[:3]])   # 監督據此送下一件／返回目標
                    self._goto('RETURN', err=round(err, 4))
            elif el > self.T_bm + PH_TO_S:
                self._stop('bin_move_not_reached' if self.phase == 'BIN_MOVE' else 'bin_stow_not_reached', err=round(err, 4))
        elif self.phase == 'BIN_REL':
            if self.obj_detached:
                self.attach = False                                   # 連接已解除 ⇒ 同拍撤回連接請求（s66：撤回晚於伺服器重置 ⇒ 重送 attach_cyl，Isaac 會把物體焊回夾爪）
                self.finger_cmd = P.FINGER_OPEN
                self._goto('BIN_OPEN')
            elif el > DETACH_TO_S:
                self._stop('obj_detach_ack_timeout')
        elif self.phase == 'BIN_OPEN':
            if fm is not None and min(fm) >= P.FINGER_OK:
                self._goto('BIN_DROP', finger=[round(float(x), 5) for x in fm])
            elif el > P.FINGER_TO_S:
                self._stop('finger_open_timeout', finger=None if fm is None else [round(float(x), 5) for x in fm])
        elif self.phase == 'BIN_DROP':
            if el >= RBIN.DROP_WAIT_S:
                self.binned, self.attach = True, False
                self._ev('pick_binned')
                self._bin_start(q, 'BIN_STOW')

    def _flow_vc(self, qp):
        """照導航移動中夾取的抓取點附近限速（APPROACH 在 r_turn 外 ⇒ None）；與速度框同一式。"""
        b0f_ = np.asarray(self.flow['b0'], float)
        df_ = float(np.hypot(qp[0] - b0f_[0], qp[1] - b0f_[1]))
        if self.phase == 'APPROACH' and df_ >= float(self.flow.get('r_turn', 1.0)):
            return None
        vc_ = float(self.flow.get('v_cap', 0.08)) * (FLOW_CLOSE_K if self.phase in ('CLOSE', 'CONFIRM') else 1.0)
        if FLOW_VRAMP and self.phase == 'APPROACH' and getattr(self, '_t_arr', None) is None:   # 進場：r_turn→r_in 之間平順降到 v_cap（原一步跳）
            rt_, ri_ = float(self.flow.get('r_turn', 1.0)), float(self.flow.get('r_in', 0.35))
            x_ = min(max((df_ - ri_) / max(rt_ - ri_, 1e-3), 0.0), 1.0)
            vc_ = vc_ + (max(FLOW_V_HI, vc_) - vc_) * x_ * x_ * (3.0 - 2.0 * x_)
        return vc_

    def _ref_vcap(self, qp):
        """本拍會套到底盤的速度框中最小者（同 _solve 後段各框的條件）；無 ⇒ None。只用來時間縮放參考。"""
        c_ = []
        if self.flow and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):
            v_ = self._flow_vc(qp)
            if v_ is not None:
                c_.append(v_)
        if self.phase == 'APPROACH' and getattr(self, 'app_v', APP_V) is not None:
            c_.append(self.app_v if hasattr(self, 'app_v') else APP_V)
        if self.phase in ('APPROACH', 'RETURN', 'CARRY', 'ALIGN'):
            nv_ = P.near_v(qp[:3])
            if nv_ is not None:
                c_.append(nv_)
        if self.phase in ('RETURN', 'CARRY'):
            c_.append(CARRY_V)
        return min(c_) if c_ else None

    def _solve(self, q):
        if self.phase in ('BIN_MOVE', 'BIN_REL', 'BIN_OPEN', 'BIN_DROP', 'BIN_STOW'):
            return self._solve_bin(q)
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        if FURN_CBF_PAD is not None and type(self) is TablePickCtrl:
            bg_ = getattr(self, '_bg', None)
            far_ = self.motm is None and (bg_ is None or float(np.hypot(qp[0] - bg_[0], qp[1] - bg_[1])) > FURN_NEAR_GOAL)
            obs = list(obs) + _furn_obs(qp, pad=FURN_CBF_PAD_FAR if far_ else FURN_CBF_PAD)
            if src == 'nav':
                X = list(X)
                sh_ = _furn_ref(X, bg_)
                if sh_ > 0.01 and self.t - getattr(self, '_furn_ref_t', -9.0) > 2.0:
                    self._furn_ref_t = self.t
                    self._ev('furn_ref_shift', shift_m=round(sh_, 3))
        if self.phase == 'ALIGN':                                     # 原地對齊：底盤位置不動、轉到路線朝向；手臂保持
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'align'
        elif self.motm is not None and self.m_s0 is not None and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):   # 移動中夾取：底盤沿經過路線不停
            X, xi = self._motm_ref(t_eff)
            umin0, src = float('nan'), 'motm_line'
        elif self.phase in ('CLOSE', 'CONFIRM', 'LIFT', 'VERIFY', 'CARRY') and not (self.flow and self.phase != 'CARRY'):   # 接觸／持物操作段：底盤原地（照導航移動中夾取除外）
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if FLOW_PULL and self.flow and src == 'nav' and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):   # 抓取位附近導航路徑側向拉近 b0（不是經過線：形狀、方向、速度照導航）
            if getattr(self, '_pull_t0', None) is None:
                self._pull_t0 = float(self.t)
            X, mx_, self._pull_s0 = flow_pull(list(X), self.flow['b0'], wmul=P._ss((float(self.t) - self._pull_t0) / max(FLOW_PULL_TIN, 1e-3)))
            xi = xi_from_X(X, DT)
            if mx_ > 0.02 and self.t - getattr(self, '_pull_ev_t', -9.0) > 1.0:
                self._pull_ev_t = self.t; self._ev('flow_pull', phase=self.phase, shift_m=round(mx_, 3))
        if P.REF_VCAP and src in ('nav', 'base_goal'):                   # 參考先時間縮放到限速內（不讓兩軸速度框同時貼滿 ⇒ 斜走、左右翻）；框照舊當上限
            vr_ = self._ref_vcap(qp)
            X, sc_ = P.ref_speed_cap(list(X), None if vr_ is None else P.REF_VCAP_K * vr_, DT, P.REF_WCAP, yaw0=P.yaw_ref_state(self, qp, X, P.REF_WCAP, DT))
            if sc_:
                xi = xi_from_X(X, DT)
        if self.flow and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY') and src in ('nav', 'base_goal'):   # 照導航移動中夾取：位置照導航；離抓取點 r_turn 內偏航依距離轉到抓取朝向（要夾時才轉），夾到抬起後交回路徑朝向
            b0f_ = np.asarray(self.flow['b0'], float); rt_, ri_ = float(self.flow.get('r_turn', 1.0)), float(self.flow.get('r_in', 0.35))
            Xn_ = []
            for Xk in X:
                x_, y_ = float(Xk[0, 2]), float(Xk[1, 2]); yaw_ = math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))
                w_ = 1.0 if self.phase != 'APPROACH' else min(max((rt_ - math.hypot(x_ - b0f_[0], y_ - b0f_[1])) / max(rt_ - ri_, 1e-3), 0.0), 1.0)
                w_ = max(w_, getattr(self, '_flow_w', 0.0)) if self.phase == 'APPROACH' else w_
                Xn_.append(XO.from_xytheta(x_, y_, yaw_ + w_ * ((float(b0f_[2]) - yaw_ + math.pi) % (2 * math.pi) - math.pi)))
            X, xi = Xn_, xi_from_X(Xn_, DT)
            if src == 'nav':
                P.yield_apply(self, qp, X, obs, xi, manip=True)
        if self.phase in ('RETURN', 'BIN_MOVE', 'BIN_STOW'):         # 診斷（s137d 回程兩次 9 s 參考不動）：參考視窗幾乎不動 ⇒ 記來源與前端狀態（節流 1 s）
            try:
                mv_ = float(np.hypot(X[-1][0, 2] - X[0][0, 2], X[-1][1, 2] - X[0][1, 2]))
                if mv_ < 0.02 and self.t - getattr(self, '_refstill_t', -9.0) > 1.0:
                    self._refstill_t = self.t
                    ni_ = getattr(self, 'nav_in', None) or {}
                    self._ev('ref_still', src=src, phase=self.phase, mv=round(mv_, 4), halted=ni_.get('halted'), stale=ni_.get('stale'), n_obs=len(obs), q=[round(float(x), 3) for x in qp[:3]])
            except Exception:                                         # noqa: BLE001
                pass
        if self.phase == 'RETURN':
            self.app_prof = None                                      # 返回段：照導航前端（不再收斂到抓取位）
            if FACE_DR is not None and type(self) is TablePickCtrl and float(np.hypot(qp[0] - FACE_DR[0], qp[1] - FACE_DR[1])) < 0.6:
                h_ = _dr_handle_xy()                                     # Howard 11/20「朝向由物件與可見性決定」：位置留在導航到點處，偏航轉向櫃體把手估計位置（手腕相機收臂時朝前 ⇒ 把手落在視野中央）
                self.app_prof = dict(base_goal=[float(qp[0]), float(qp[1]), math.atan2(h_[1] - float(qp[1]), h_[0] - float(qp[0]))] if h_ is not None else list(FACE_DR), door=None)   # s104d：導航到抽屜櫃前粗目標不看偏航（會車後偏 0.8 rad、相機看不到把手）⇒ 到點後同一控制器慢轉到粗目標朝向（只在前端停止更新後作用）
            if src == 'nav':                                          # 會車選邊（s92d：行人 walker_r1 正面迎來、牆側不夠 ⇒ 被推著倒退後接觸）
                P.yield_apply(self, qp, X, obs, xi)
        arm = dict(mode='hold')
        Xp = [(float(Xk[0, 2]), float(Xk[1, 2]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X]
        if self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):
            if self.phase == 'APPROACH':
                self._jq_refs = []
                Ts = [self._app_target(t_eff + k * DT, Xp[k]) for k in range(XO.N + 1)]
            elif self.phase in ('LIFT', 'VERIFY') and (self.motm is not None or self.flow):   # 相對底盤抬升（移動中夾取：底盤在動）
                def _lr(tt):
                    T_ = self.T_rel_l0.copy(); T_[2, 3] += LIFT_M * XE.minjerk(min(max((tt - self.t_ph0) / T_LIFT, 0.0), 1.0))[0]; return T_
                Ts = [_lr(t_eff + k * DT) if self.phase == 'LIFT' else _lr(self.t_ph0 + T_LIFT + 1.0) for k in range(XO.N + 1)]
            elif self.phase in ('LIFT', 'VERIFY'):
                Ts = [self._lift_target(t_eff + k * DT) if self.phase == 'LIFT' else self._lift_target(self.t_ph0 + T_LIFT + 1.0) for k in range(XO.N + 1)]
            else:
                Tg_ = self.T_grasp
                if (getattr(self, 'app_tol_fy', None) is not None or self.flow) and getattr(self, 'aim_off', None) is not None:
                    Tg_ = self.T_grasp.copy(); Tg_[:3, 3] = Tg_[:3, 3] + self.aim_off   # 把手閉爪（及照導航夾取）：沿用瞄準補償（凍結）
                Ts = [Tg_] * (XO.N + 1)
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts),
                       w_p=self._wp_app(), lo=arm_lim(qp[3:])[0], hi=arm_lim(qp[3:])[1],
                       tcp_rel_base=bool((self.motm is not None or self.flow) and self.phase in ('LIFT', 'VERIFY')), qp_solver='gi', **B.W_ARM)
            qpf = getattr(self, 'app_qpost', None)                       # 選用：接近段關節姿態引導（抽屜把手；圓柱流程不設 ⇒ 不變）
            if qpf is not None and self.phase == 'APPROACH':
                arm.update(q_post=np.array([qpf(t_eff + (k + 1) * DT) for k in range(XO.N)]), w_q=float(getattr(self, 'app_w_q', 1.0)))
            elif self.phase == 'APPROACH' and self._jb_active() and len(getattr(self, '_jq_refs', [])) == XO.N + 1:
                arm.update(q_post=np.array(self._jq_refs[1:]), w_q=JB_WQ)   # 展開段關節姿態引導＝同一條關節內插（與 TCP 參考一致、不互相拉扯）
            self._jq_refs = None
        elif self.phase == 'CARRY':
            Ts = [self._carry_target(t_eff + k * DT) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=arm_lim(qp[3:])[0], hi=arm_lim(qp[3:])[1], tcp_rel_base=True, qp_solver='gi', **B.W_ARM)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        u_max_saved = np.asarray(mpc.cfg.u_max, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        if self.flow and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):   # 照導航移動中夾取：抓取點附近限速（閉爪確認更慢；物件只影響速度上限）
            vc_ = self._flow_vc(qp)
            if vc_ is not None:
                lim_ = np.array([vc_, vc_, 0.6])
                mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)
                mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        if self.phase == 'APPROACH' and getattr(self, 'app_v', APP_V) is not None:   # 接近段速度框（只圓柱；抽屜控制器設 None）
            lim_ = np.array([self.app_v if hasattr(self, 'app_v') else APP_V] * 2 + [0.6])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        nv_ = P.near_v(qp[:3]) if self.phase in ('APPROACH', 'RETURN', 'CARRY', 'ALIGN') else None
        if nv_ is not None:                                           # 近家具減速（同穿門段）
            lim_ = np.array([nv_, nv_, 9.0])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        if self.phase in ('RETURN', 'CARRY'):                          # 持物速度框（任務限制，非導航參數）：第十六趟伸臂持物時底盤原地轉 0.43 rad/s ⇒ 圓柱沿運動方向滑出
            lim_ = np.array([CARRY_V, CARRY_V, CARRY_WZ])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)   # 不覆蓋前面的近家具減速
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min, mpc.cfg.u_max = u_min_saved, u_max_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), _obs_near(obs, qp)))
        if self.phase == 'RETURN' and int(r.cbf_active) > 0 and self.t - getattr(self, '_cbfblk_t', -9.0) > 1.0:   # 診斷（Isaac 62：門扇尖端附近被推回 30 s、無事件）：參考在走、輸出幾乎不動 ⇒ 記最近障礙
            try:
                vr_ = float(np.linalg.norm(np.asarray(xi[0], float)[:2])); vu_ = float(np.linalg.norm(np.asarray(r.u_opt, float)[:2]))
                up_ = np.asarray(self.u_prev[:2], float); un_ = np.asarray(r.u_opt, float)[:2]
                rev_ = float(np.linalg.norm(un_)) > 0.15 and float(np.linalg.norm(up_)) > 0.1 and float(un_ @ up_) < 0.0   # Isaac 70：門扇尖端附近指令突然反向（最大速推回）
                if (vr_ > 0.1 and vu_ < 0.05) or rev_ or vu_ < 0.02:
                    self._cbfblk_t = self.t
                    nb_ = sorted(obs, key=lambda o: math.hypot(float(o['x']) - float(qp[0]), float(o['y']) - float(qp[1])))[:3]
                    self._ev('ret_cbf_block', rev=bool(rev_), q=[round(float(x), 3) for x in qp[:3]], v_ref=round(vr_, 3), u=[round(float(x), 3) for x in np.asarray(r.u_opt, float)[:3]], n_cbf=int(r.cbf_active),
                             near=[dict(x=round(float(o['x']), 3), y=round(float(o['y']), 3), r=round(float(o.get('radius', 0.0)), 3), m=o.get('margin'), st=bool(o.get('static')), vx=round(float(o.get('vx', 0.0)), 3), vy=round(float(o.get('vy', 0.0)), 3)) for o in nb_])
            except Exception:                                         # noqa: BLE001
                pass
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        if arm['mode'] == 'hold':
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)], src
        return P.wg_result(self, r, src)

    def _bin_vlim(self, qp):
        """放盒段底盤速度框 [vx, vy, wz]（任務限制＋近家具減速＋抓完起步斜坡）；同一式供速度框與參考時間縮放。"""
        nv_ = P.near_v(qp[:3])
        lim_ = np.array([min(BIN_V, nv_ if nv_ is not None else 9.0)] * 2 + [BIN_WZ])   # 放盒中底盤速度框（任務限制）＋近家具減速
        if FLOW_VRAMP and self.phase == 'BIN_MOVE':                    # 抓完起步斜坡（原：限速框一解除即 0.06→0.2 m/s、偏航 0.05→0.5 rad/s 同一拍）
            if getattr(self, '_ex_u0', None) is None:
                self._ex_u0 = (float(np.hypot(self.u_prev[0], self.u_prev[1])), abs(float(self.u_prev[2])), float(self.t))
            el_ = max(0.0, float(self.t) - self._ex_u0[2])
            lim_ = np.minimum(lim_, np.array([max(self._ex_u0[0], 0.04) + EXIT_A * el_] * 2 + [max(self._ex_u0[1], 0.08) + EXIT_AW * el_]))
        return lim_

    def _solve_bin(self, q):
        """放盒／收臂：手臂關節空間剖面（相對底盤，tcp_rel_base）；底盤照導航前端（限速框）或錨定。"""
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        self.app_prof = None
        if FLOW_PULL and getattr(self, 'flow', None) and BIN_MOVE and src == 'nav':   # 抬起後同一拉移隨離 b0 距離平順退回導航路徑（不在抬起那拍一步拉回）
            if getattr(self, '_pull_tb', None) is None:
                self._pull_tb = float(self.t)
            s_ = getattr(self, '_pull_s0', None)
            k_ = 0.0 if s_ is None else 1.0 - P._ss((float(self.t) - self._pull_tb) / max(FLOW_PULL_TOUT, 1e-3))
            if k_ > 1e-3:
                X = [XO.from_xytheta(float(Xk[0, 2]) + k_ * float(s_[0]), float(Xk[1, 2]) + k_ * float(s_[1]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X]
                xi = xi_from_X(X, DT)
        if BIN_MOVE and src == 'nav' and P.REF_VCAP:                  # 參考先時間縮放到本段速度框（含起步斜坡）內，偏航速率限 REF_WCAP（第 123／124 趟抓完以 0.5 rad/s 轉 90°、兩軸貼框斜走）
            vb_ = self._bin_vlim(qp)
            X, sc_ = P.ref_speed_cap(list(X), P.REF_VCAP_K * float(vb_[0]), DT, min(P.REF_WCAP, float(vb_[2])), yaw0=P.yaw_ref_state(self, qp, X, min(P.REF_WCAP, float(vb_[2])), DT))
            if sc_:
                xi = xi_from_X(X, DT)
        if BIN_MOVE and src == 'nav':                                   # s150d：放盒中底盤照導航（限速 0.15）沒接會車選邊 ⇒ 橫越行人從側後方追上撞到；同 RETURN 段接上
            X = list(X)
            P.yield_apply(self, qp, X, obs, xi)
        if not BIN_MOVE:
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if self.phase in ('BIN_MOVE', 'BIN_STOW'):
            qs = [self._bin_q(t_eff + k * DT) for k in range(XO.N + 1)]
        else:
            qs = [self.bin_target] * (XO.N + 1)                       # 解除／張爪／落下：保持放置構型
        Ts = [K.fk(np.r_[np.zeros(3), qk], 'link_tcp') for qk in qs]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts),
                   w_p=C0.W_FAR, lo=arm_lim(qp[3:])[0], hi=arm_lim(qp[3:])[1], tcp_rel_base=True, qp_solver='gi', q_post=np.array(qs[1:]), w_q=5.0, **B.W_ARM)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        u_max_saved = np.asarray(mpc.cfg.u_max, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        lim_ = self._bin_vlim(qp)
        mpc.cfg.u_max = np.minimum(u_max_saved, lim_)
        mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min, mpc.cfg.u_max = u_min_saved, u_max_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), _obs_near(obs, qp)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        return P.wg_result(self, r, src)

    def step(self, t, q, phi_meas, finger_meas=None, finger_force=None):
        if finger_force is not None:
            self.finger_force = (float(finger_force[0]), float(finger_force[1]))
        if self.n_step % NPH == 0 and self.phase == 'RETURN' and self.stop is None and not self.done:   # 返回完成判定（場景專用）
            ni = self.nav_in
            halted = ni is not None and not ni.get('stale') and bool(ni.get('halted'))
            self.t_halt = (self.t_halt if self.t_halt is not None else float(t)) if halted else None
            if self.t_halt is not None and float(t) - self.t_halt >= P.DONE_HOLD_S and float(q[1]) < RET_Y_MAX and float(np.abs(self.u_last[:3]).max()) < 0.01:
                self.done = True
                self._ev('pick_return_done', base=[round(float(x), 4) for x in q[:3]])
        return super().step(t, q, phi_meas, finger_meas=finger_meas)
