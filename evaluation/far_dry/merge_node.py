#!/usr/bin/env python3
"""遠距進線模式（MERGE）的監督＋命令選擇＋手臂追蹤節點（乾跑用；Codex 20261116 M1）。
命令路由（平滑器之前、單一有效來源）：/cmd_vel_nav_raw（導航 GMPC）或 /cmd_vel_merge（進線 GMPC）→ 本節點選一個轉發到 /cmd_vel_nav → 原平滑器 → Shield → GUARD → /cmd_vel。
進線 GMPC＝同一個 gmpc_node 程式、參數檔取自現場 dump 的導航 GMPC（CBF、限制、權重全同），只覆寫任務參考：plan_topic、cmd_vel_topic、v_nominal＝通過速度、heading_fixed＝通過線朝向、goal_tolerance_xy。
幾何計算（把手估計、通過線、觀察分支、手臂追蹤、交接核對）在 fork 出的工作程序做（專案模組會裝 rclpy 替身，不能與 rclpy 同程序）。
座標：把手估計、通過線、手臂任務都保存在連續的里程座標（odom）；影像估計以「擷取同刻」的里程位姿轉入；發布給進線 GMPC 的路徑才用當下 map←odom 轉成 map。真值底盤位姿只記錄供事後評估。
資料來源照實：底盤位姿取 TF odom→base_footprint；手臂關節、設定點、8 步在途命令取世界節點 /far/truth（驅動端讀回）；把手相對底盤的量測來自 online_1 的實際影像（時間戳設為本趟起點，之後沒有新影像）。
淨空：在地圖座標核（機器人構型與把手估計以當下 map←odom 轉入，與靜態環境幾何同座標）。限制：乾跑的里程是理想的（無滑移）。"""
import json, math, multiprocessing as mp, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V_PASS, DT, PHYS, NPH, D = 0.05, 0.05, 0.01, 5, 8
DEC = (6.25, 6.25, 25.51)                                             # 平滑器既有減速上限（現場參數快照 params_velocity_smoother.yaml；未改）
CAP_ALONG, CAP_LAT = (0.45, 1.40), 0.60                              # r2（Codex 20261116 F3）：沿線下界 0.62→0.45；只是允許嘗試進線的範圍，不是伸臂放行（手臂準備仍須滿足既有距離條件）
ENTER_EXTRA = 0.02                                                   # 斜向切入目標的額外沿線餘量（在途命令 0.13 s×通過速度＋切入收斂）
_OLD_CAP = (0.62, 1.40)                                    # 捕獲區：距抓取點的沿線距離、離線側向距離
PREP_MARGIN = 0.04
# 抓持窗口的真實座標（通過線沿線位置；door_far_branch_map_r1.json 門旁分支）：預抓位可行 −0.075～−0.325、抓取位可行 −0.10～−0.30。
GRASP_END_ABS = -0.285                                               # 抓持段絕對終點＝抓取位可行下界 −0.30 留停止行程 0.015（在途 8 步 4 mm＋減速約 1 cm）
GRASP_NEED = 0.115 + 0.025                                           # 接手後到 READY 的行程（排程 2.3 s×0.05）＋接受等待上限 0.5 s 的行程
HO_LATEST = GRASP_END_ABS + GRASP_NEED                               # ＝−0.145：最晚可交接的沿線位置（保證排程與接受等待都在絕對終點前）；更晚 ⇒ 不交接
TERM_END = HO_LATEST - 0.005                                         # 預抓位追蹤終點：過了最晚交接點仍未交接 ⇒ 協同停止
# 近距觀察段的連續視覺更新（N3；Codex 20261116）。只在觀察追蹤（TRACK）中、已核沿線範圍內啟用；只更新把手位置（姿態沿用固定參考）。
UPD = dict(max_age_s=0.3,          # 觀察年齡上限（同 FL.EST_MAX_AGE_S）
           gate_pos_m=0.020, gate_ang_deg=3.0,   # 對目前參考的粗差閘門（明顯不是同一目標或估計壞掉）
           n_consist=3, spread_m=0.005,          # 需最近 3 格有效觀察、彼此位置差 ≤5 mm 才形成更新目標（取各軸中位數）
           step_m=0.0001)                        # 參考每控制拍最多移動 0.1 mm（＝2 mm/s；目標與命令連續，不逐格覆寫、不清管線）。1 mm／拍在重播中使手臂命令偏離無更新版最多 0.49 rad/s，太大
# 終端接近 T1c（K3；運動學離線依據 door_far_branch_map_r1.json、door_far_terminal_t1c_check.json）：
#   遠距觀察分支（沿線 ≥0.025 才可行）與門旁抓持／開門分支（沿線 ≤−0.075 才可行）不相連，中間是腕部近奇異帶；從觀察分支直接抓，開門首段不可行（假世界整鏈核對）。
#   ⇒ 準備提前完成 → 手臂靜止觀察 → 鎖定參考 → 以關節直線（不是 TCP 直線，不經任務空間奇異帶）轉到門旁分支的預抓位解 → 追預抓位 → 交接給門旁已驗的移動抓持／開門
DS_SEED = (-0.354, -0.088, 1.435, -1.421, -0.358, -0.157)            # 門旁分支的手臂種子（m8house_door_full_2 綁定時構型；只作 IK 分支種子）
DS_ANCHOR_A = -0.115                                                 # 該種子對應的沿線位置（門旁綁定時底盤 x 0.724）
GRASP_ORIGIN_SHIFT = 0.20                                            # 交接判定用的抓取點原點＝通過線原點沿行進方向再 0.20 m（門旁分支預抓位可行到沿線 −0.20）；窗口條件本身不變
TERM = dict(a_obs=0.42,            # 手臂準備須在沿線 ≥ a_obs 前完成（之後手臂靜止在作業起點解，把手在畫面內：幾何投影 ≤0.40）
            a_lock=0.25,           # 在途前推後到這個沿線位置：鎖定參考並開始轉移。r2：0.15→0.25——m8house_far_door_1 實測手臂靜止觀察時把手只在沿線 0.37～0.21 m 有效（0.20 起出畫面），0.15 已無一致觀察
            a_pre=-0.075,          # 轉移結束、到門旁分支預抓位（後退 S_PRE）的沿線位置
            t_desc=6.5,            # 轉移時間（＝(a_lock−a_pre)/v_pass＝0.325/0.05；梯形速度，加減速各 20%；離線 door_far_terminal_t1c_check.json a_start 0.25：關節速率 0.50、淨空 env 0.043／self 0.0063）
            est_bound_m=0.00285,   # 影像估計＋擷取同步的位置誤差預算項＝同一個三格鎖定算法在既有 Isaac 影格（近距 ≤0.5 m，n=95，capture_1／3）上的已觀測誤差包絡最大值（far_lock_error_envelope.json）；工程預算，不是可靠上界
            est_bound_deg=0.885,   # 同上，姿態包絡最大值
            sync_s=0.011,          # 同步：一個物理步＋餘量 × 通過速度
            odom_frac=0.0)         # 里程漂移：理想里程分支（模擬里程＝物理位姿）列為零（Codex 20261116 允許；實機須另給）；追蹤與模型殘差由每拍實測 TCP 誤差項計入
ASSOC_TOL = dict(dxy_m=0.15, dz_m=0.10, dang_deg=10.0)               # 觀察對先驗名目把手位姿的關聯容許（粗核對：是不是目標門的把手）
STOP_ALONG = -0.05


def stop_disposition(tag, sc):
    """每拍停止核對結果 → 處置（Codex 20261116）：
      commit    ：now 與 next 都通過 ⇒ 可提交候選。
      stop_now  ：只有 next 不過、now 已驗證 ⇒ 不提交，走「現在停」（受控停止）。
      abort     ：now 也不過，或結果缺失／遲到／錯誤 ⇒ 模擬工程中止（禁止下一物理步；不稱受控停止）。"""
    if tag != 'tick' or not isinstance(sc, dict) or not isinstance(sc.get('now'), dict) or not isinstance(sc.get('next'), dict):
        return 'abort'
    if sc.get('why'):
        return 'abort'
    if sc['now'].get('ok') is True and sc['next'].get('ok') is True:
        return 'commit'
    if sc['now'].get('ok') is True:
        return 'stop_now'
    return 'abort'


def _trap(u, f=0.2):
    """梯形速度輪廓（加減速各佔 f）：峰值速度＝平均的 1/(1−f) 倍。"""
    u = min(max(float(u), 0.0), 1.0)
    vp = 1.0 / (1.0 - f)
    if u < f:
        return 0.5 * vp * u * u / f
    if u > 1 - f:
        return 1.0 - 0.5 * vp * (1 - u) ** 2 / f
    return vp * (u - 0.5 * f)


def _T3(p):
    import numpy as np
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


def worker(conn):
    sys.path.insert(0, HERE)
    import numpy as np
    import door_far_link as FL, door_far_route as FR, door_motm_grasp_ctrl as G, door_wbopt as W, door_clear_door_fast as _F, door_fast_kin as DFK
    W.clear_door = _F.clear_door_fast
    K = G.K
    RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1')
    meta = json.load(open(os.path.join(RUN, 'frames_meta.json')))
    fr = [f for f in meta['frames'] if f['stem'] == 'far_live'][0]
    Rg, T_HG, a_H = G.PG.baseline_reference()
    FL.set_approach_axis(a_H)
    T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
    S_PRE = float(G.PG.S_PRE)
    LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    tau = PHYS / np.asarray(ident['alpha'], float)
    obs = FL.HandleObs()
    T_est, oinfo = obs.intake(dict(dep=np.load(os.path.join(RUN, 'frames', 'far_live_depth_m.npy')), K=np.asarray(meta['K'], float), T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float),
                                   t_state=float(fr['t_obs']), pose_source='fk_at_render_time', frame_id='map'), float(fr['t_obs']) + 0.01)
    W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
    import door_sweep_pool as SWP
    import door_cpu_layout as CPU
    SWP._die_with_parent()
    cpus = [int(x) for x in os.environ.get('FAR_CPUS', '').split(',') if x.strip()]   # N2：排程配置（只改排程；45 ms 期限不改）。順序：幾何工作程序、監督主程序、其餘給掃掠池
    if len(cpus) >= 6:
        CPU.pin(0, cpus[0:2])                                           # 幾何工作程序與監督主程序共用前兩核（不一對一釘死：桌面程序佔到其中一核時排程器可避開；實測 Chrome 以 63% 佔住 CPU 5）
    blas_rb = CPU.blas_runtime(set_to=1)
    pool = SWP.SweepPool(int(os.environ.get('FAR_POOL_N', '4')))                                           # 建池在啟動時（不在控制拍內）
    T_est_W = T_est.copy()                                             # 歷史影像的估計（該影像自己的世界座標；含實際影像的估計誤差）
    clear_fn = lambda x: W.clear_door(np.r_[x, 0.0], T_est)            # noqa: E731
    AL = np.asarray(ident['alpha'], float)
    st = dict(pg=None, br=None, trk=None, off=None, q_pre=None, lock=None)
    grav = DFK.FastGravity3(K, G.BM.arm_bodies(K))
    KP_ARM = 600.0                                                    # 執行器手臂位置驅動增益（API 讀回 600）

    def off_fn(q9):
        """K2：手臂靜態偏置 q−s 的模型＝無接觸保持狀態的錨 off0＋(G(q)−G0)/kp（重力隨構型）。未錨定時回 0（原行為）。
        既有紀錄核對（far_arm_offset_model_check.json）：此模型加一階落後後，殘差最大 0.006 rad。"""
        o = st['off']
        if o is None:
            return np.zeros(6)
        return o['off0'] + (np.asarray(grav(np.asarray(q9, float)), float) - o['G0']) / KP_ARM

    def to_map(q, T_mo):
        """里程座標的機器人構型 → 地圖座標（靜態環境幾何所在的座標）。"""
        c, s_ = math.cos(T_mo[2]), math.sin(T_mo[2])
        return np.r_[c * q[0] - s_ * q[1] + T_mo[0], s_ * q[0] + c * q[1] + T_mo[1], q[2] + T_mo[2], q[3:]]

    import door_clear_fast as FAST
    from scipy.spatial import cKDTree
    PAIRS = FAST._pairs()
    CHAIN = dict(base_link=0, link_base=0, link1=1, link2=2, link3=3, link4=4, link5=5, link6=6, link_eef=6, uflite_gripper_link=6, uflite_finger1=6, uflite_finger2=6)
    SNAMES = sorted(set(FAST.DOC.ARM_LINKS) | {n_ for p_ in PAIRS for n_ in p_} | {f'link{k_}' for k_ in range(1, 7)})
    PAIR_CACHE = {}
    # 逐對自碰撞區間界用的槓桿長上界（對所有構型成立；URDF 固定平移長度的三角不等式）：
    #   r(k, y) ＝ Σ_{m=k}^{iy−1} |o_{m+1} − o_m| ＋ max_p |p_y − o_iy|
    #   o_m＝關節 m 原點（＝link m 座標原點）；相鄰關節原點距離與構型無關；y 的點雲相對 o_iy 的距離：y 與 link iy 之間只有固定變換（手指另加張開量），與構型無關。
    _T0 = _F.fk_all(np.zeros(9), SNAMES)
    _O = {m_: _T0[f'link{m_}'][:3, 3] for m_ in range(1, 7)}
    _SEG = {m_: float(np.linalg.norm(_O[m_ + 1] - _O[m_])) for m_ in range(1, 6)}
    _RY = {}
    for n_ in SNAMES:
        if n_ in FAST.DOC.CL and CHAIN.get(n_, 0) >= 1:
            Pn = (_T0[n_][:3, :3] @ FAST.DOC.CL[n_].T).T + _T0[n_][:3, 3]
            _RY[n_] = float(np.linalg.norm(Pn - _O[CHAIN[n_]], axis=1).max()) + (0.012 if n_ in FAST.DOC.FINGERS else 0.0)   # 手指沿滑動軸最多張開約 9 mm，加 12 mm
    LEVER = {(k_, n_): sum(_SEG[m_] for m_ in range(k_, CHAIN[n_])) + _RY[n_] for n_ in _RY for k_ in range(1, CHAIN[n_] + 1)}
    R_ENV = 1.10                                                      # 機器人任一點到底盤原點距離的保守上界（m）。Codex 20261116 以碰撞點雲＋URDF 固定平移長度的三角不等式核得約 0.9936 m
    R_JNT = 0.75                                                      # 任一手臂關節到其遠端任一點距離的保守上界（m）。同上核得最大約 0.6401 m

    def self_pairs(qf):
        """自碰撞逐對資料：回 (T, 各對下界 lb＝中心距−兩半徑, 中心, 需要時算精確距離的函式)。精確距離＝與 door_clear_fast.clear_full 同一算法（點雲對 KD 樹）。"""
        T = _F.fk_all(np.asarray(qf, float), SNAMES)
        fq = W.FINGER_Q
        for n_ in FAST.DOC.FINGERS:
            fqn = (fq[0] if n_.endswith('1') else fq[1]) if isinstance(fq, (tuple, list)) else fq
            if n_ in T and fqn:
                T[n_] = T[n_].copy()
                T[n_][:3, 3] += (1.0 if n_.endswith('1') else -1.0) * fqn * T[n_][:3, 1]
        Cw = {n_: T[n_][:3, :3] @ FAST._C[n_] + T[n_][:3, 3] for n_ in SNAMES if n_ in FAST._C}
        lb = {(x, y): float(np.linalg.norm(Cw[x] - Cw[y])) - FAST._R[x] - FAST._R[y] for x, y in PAIRS}

        def exact(x, y):
            if x not in FAST._TREE:
                FAST._TREE[x] = cKDTree(FAST.DOC.CL[x])
            Py = ((T[y][:3, :3] @ FAST.DOC.CL[y].T).T + T[y][:3, 3] - T[x][:3, 3]) @ T[x][:3, :3]
            return float(FAST._TREE[x].query(Py, k=1)[0].min())
        return T, lb, Cw, exact

    def stop_check(q, pipe, rel, s, ua, T_mo):
        """提交前停止核對，兩種實際會採用的停法都核：
          now ：本拍起就送零（候選不提交）——底盤＝在途 8 步 → 平滑器以既有減速上限逐拍降到零；手臂＝在途命令走完後設定點不再動，關節以一階模型收斂。
          next：本拍提交候選 ua、下一拍起送零。
        通過條件（全部成立）：
          (1) 限位：設定點與關節全程在限位內；
          (2) 停穩：在途管線排清後，底盤命令為零且關節每步變化 <1e-5 rad 連續 0.2 s（在預測時域內達成）；
          (3) 淨空含區間界：相鄰兩個樣本之間，任一點的位移不超過「路徑長上界」L（逐物理步累加，不是端點差）；
              區間內任一構型離較近端點的位移 ≤ L/2，所以要求 兩端淨空的較小值 − L/2 ≥ 門檻。門檻不改。
              環境／底盤：L＝底盤平移路徑長＋|偏航變化|·R_ENV＋Σ|關節變化|·R_JNT（對所有連桿取同一保守值）。
              自碰撞逐對：L＝Σ(兩連桿之間的關節 k)|關節變化|·r(k, 遠端連桿)；r＝對所有構型成立的槓桿長上界（相鄰關節原點距離之和＋遠端連桿點雲到其關節原點的最大距離），不超過 R_JNT；兩連桿之間沒有運動關節 ⇒ L＝0。
              某區間不過且中間還有物理步 ⇒ 加密到逐物理步再核；逐步仍不過 ⇒ 不通過。
        淨空在地圖座標核（機器人構型與把手估計以當下 map←odom 轉入；靜態環境幾何本來就在地圖座標）。"""
        t0 = time.perf_counter()
        T_map = _T3(T_mo) @ T_est
        thr = W.THR
        trajs = {}
        for mode in ('now', 'next'):
            cmds = [np.asarray(u, float) for u in pipe] + ([np.r_[rel, ua]] * NPH if mode == 'next' else [])
            v = np.asarray(rel, float).copy()
            while float(np.abs(v).max()) > 1e-9:
                v = v - np.sign(v) * np.minimum(np.abs(v), np.asarray(DEC) * DT)
                cmds += [np.r_[v, np.zeros(6)]] * NPH
            n_cmd = len(cmds)
            cmds += [np.zeros(9)] * 100
            b, a, sp = np.asarray(q[:3], float).copy(), np.asarray(q[3:], float).copy(), np.asarray(s, float).copy()
            off_c = off_fn(q)                                          # 停止期間的靜態偏置取當下構型的值（停止行程短，重力項變化小；與追蹤器同一模型）
            Q, lim_ok, still, k_still = [np.r_[b, a]], True, 0, None
            for k, u in enumerate(cmds):
                c, s_ = math.cos(b[2]), math.sin(b[2])
                b = b + PHYS * np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1], u[2]])
                sp = sp + PHYS * u[3:]
                a_new = a + AL * (sp + off_c - a)
                still = still + 1 if (k >= n_cmd and float(np.abs(a_new - a).max()) < 1e-5) else 0
                a = a_new
                lim_ok = lim_ok and bool(np.all(sp >= LO) and np.all(sp <= HI) and np.all(a >= LO) and np.all(a <= HI))
                Q.append(np.r_[b, a])
                if still >= 20:
                    k_still = k + 1
                    break
            Q = np.array(Q)
            dQ = np.abs(np.diff(Q, axis=0))
            dxy = np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1)
            idx, last = [0], 0
            for k in range(1, len(Q)):
                if float(np.linalg.norm(Q[k, :2] - Q[last, :2])) > 0.005 or float(np.abs(Q[k, 3:] - Q[last, 3:]).max()) > 0.01 or k == len(Q) - 1:
                    idx.append(k); last = k
            trajs[mode] = dict(Q=Q, dQ=dQ, dxy=dxy, idx=idx, lim_ok=lim_ok, settled=k_still is not None, settle_s=None if k_still is None else round(k_still * PHYS, 2), pipe_drained_s=round(n_cmd * PHYS, 2))
        cache = {}

        def key(x):
            return np.ascontiguousarray(x, dtype=np.float64).tobytes()

        def evaluate(cfgs, self_only=False):
            """回 None 或失敗原因。環境／底盤淨空走平行掃掠池；自碰撞逐對資料在本程序算。
            self_only：加密點只需自碰撞資料（環境／底盤已在初始取樣的區間界下通過時），全體最小自碰撞距離用同一精確算法在本程序算。"""
            todo = [x for x in cfgs if key(x) not in cache]
            uniq = {}
            for x in todo:
                uniq[key(x)] = x
            todo = list(uniq.values())
            if not todo:
                return None
            if self_only:
                for x in todo:
                    spx = self_pairs(x)
                    cache[key(x)] = dict(env=None, base=None, self_min=-1.0, sp=spx, ex={})   # self_min=-1：不提供全體最小值這個下界（逐對只用中心下界或精確值）
                return None
            h = pool.submit(T_map, W.FINGER_Q, W.PHI0, np.array([np.r_[to_map(x, T_mo), 0.0] for x in todo]))
            sp_ = [self_pairs(x) for x in todo]                        # 與池同時進行
            outp, why, _ = pool.collect(h, 0.040 - (h['t_submit'] - t0))   # 掃掠截止固定為 t0＋40 ms，時間耗盡即判未驗證、不補時間（collect 的期限以派工時刻起算，所以扣的是派工時刻；far_door_5 前誤扣成「現在」，實際上限只有約 36 ms）；整拍 45 ms 期限不變
            if outp is None:
                return 'sweep_' + str(why)
            for x, o, spx in zip(todo, outp, sp_):
                cache[key(x)] = dict(env=float(o['env']), base=float(o['base']), self_min=float(o['self']), sp=spx, ex={})
            return None

        def pair_d(c, pr, need, x_cfg):
            """該對的距離：下界已足夠就用下界（保守），否則算精確值。
            下界＝max(中心距−兩半徑, 該構型全體自碰撞最小距離)——後者是同一構型所有配對的精確最小值，對任一對都是有效下界。
            精確值依「兩連桿之間的關節角」快取（相對位形只由這些關節決定；鍵為實際位元組，只有完全相同的角度才共用）。"""
            lb = max(c['sp'][1][pr], c['self_min'])
            if lb >= need:
                return lb
            ix, iy = sorted((CHAIN[pr[0]], CHAIN[pr[1]]))
            kk = (pr, np.ascontiguousarray(x_cfg[3 + ix:3 + iy], dtype=np.float64).tobytes())   # 鍵＝實際角度位元組（不四捨五入：不同構型不共用距離）
            v_ = PAIR_CACHE.get(kk)
            if v_ is None:
                v_ = c['sp'][3](*pr)
                if len(PAIR_CACHE) > 200000:
                    PAIR_CACHE.clear()
                PAIR_CACHE[kk] = v_
            return v_

        def certify(mode, parts=('env', 'self')):
            tr = trajs[mode]
            Q, idx = tr['Q'], tr['idx_self'] if parts == ('self',) else tr['idx']
            worst, bad = dict(env=9.9, base=9.9, self=9.9), []
            for i, j in zip(idx[:-1], idx[1:]):
                ci, cj = cache[key(Q[i])], cache[key(Q[j])]
                tvq = tr['dQ'][i:j, 3:].sum(axis=0)
                ok_ij = True
                if 'env' in parts:
                    L_env = float(tr['dxy'][i:j].sum()) + float(tr['dQ'][i:j, 2].sum()) * R_ENV + float(tvq.sum()) * R_JNT
                    for k_ in ('env', 'base'):
                        m_ = min(ci[k_], cj[k_]) - 0.5 * L_env - thr[k_]
                        worst[k_] = min(worst[k_], m_)
                        ok_ij = ok_ij and m_ >= 0.0
                if 'self' in parts:
                    for pr in PAIRS:
                        x, y = pr
                        ix, iy = CHAIN[x], CHAIN[y]
                        if ix > iy:
                            ix, iy, y = iy, ix, x
                        L_p = 0.0
                        for kj in range(ix + 1, iy + 1):              # 只計兩連桿之間的運動關節；槓桿長用對所有構型成立的上界 LEVER（URDF 固定長度三角不等式），不超過 R_JNT
                            L_p += float(tvq[kj - 1]) * min(LEVER[(kj, y)], R_JNT)
                        need = thr['self'] + 0.5 * L_p
                        m_ = min(pair_d(ci, pr, need, Q[i]), pair_d(cj, pr, need, Q[j])) - need
                        worst['self'] = min(worst['self'], m_)
                        ok_ij = ok_ij and m_ >= 0.0
                if not ok_ij:
                    bad.append((i, j))
            return worst, bad

        res, n_ref = {}, 0
        t1 = time.perf_counter()
        why = evaluate([trajs[m]['Q'][k] for m in trajs for k in trajs[m]['idx']])
        t2 = time.perf_counter()
        for mode in ('now', 'next'):
            tr = trajs[mode]
            tr['idx_self'] = list(tr['idx'])
            if why:
                worst, bad_e, bad_s = dict(env=-9.9, base=-9.9, self=-9.9), [], []
            else:
                w_e, bad_e = certify(mode, ('env',))                     # 環境／底盤：初始取樣＋區間界（不過 ⇒ 不通過；這兩項餘裕大，不加密）
                w_s, bad_s = certify(mode, ('self',))
                if bad_s and any(j - i > 1 for i, j in bad_s):          # 自碰撞：不過的區間加密到逐物理步（只算自碰撞，不走池）
                    add = sorted({k for i, j in bad_s for k in range(i + 1, j)})
                    evaluate([tr['Q'][k] for k in add], self_only=True)
                    tr['idx_self'] = sorted(set(tr['idx']) | set(add)); n_ref += 1
                    w_s, bad_s = certify(mode, ('self',))
                worst = dict(env=w_e['env'], base=w_e['base'], self=w_s['self'])
            cov = (not why) and not bad_e and not bad_s
            mc = {k_: (round(min(cache[key(tr['Q'][k])][kk] for k in tr['idx']), 4) if not why else None) for k_, kk in (('env', 'env'), ('base', 'base'), ('self', 'self_min'))}
            res[mode] = dict(ok=bool(tr['lim_ok'] and tr['settled'] and cov), limits_ok=tr['lim_ok'], settled=tr['settled'], settle_s=tr['settle_s'], pipe_drained_s=tr['pipe_drained_s'], coverage_ok=bool(cov), min_clear=mc,
                             margin_after_interval_bound={k_: round(v_, 4) for k_, v_ in worst.items()}, n_samples=len(tr['idx']), n_samples_self=len(tr['idx_self']), n_steps=len(tr['Q']) - 1, all_still_s=tr['settle_s'])
        mcc = {k_: (None if why else min(res['now']['min_clear'][k_], res['next']['min_clear'][k_])) for k_ in ('env', 'base', 'self')}
        mab = {k_: min(res['now']['margin_after_interval_bound'][k_], res['next']['margin_after_interval_bound'][k_]) for k_ in ('env', 'base', 'self')}
        return dict(ok=bool(res['now']['ok'] and res['next']['ok']), why=why, now=res['now'], next=res['next'], min_clear=mcc if not why else {}, margin_after_interval_bound=mab, n_eval=len(cache), n_refine=n_ref,
                    coverage='區間界（路徑長上界 L；兩端淨空較小值 − L/2 ≥ 門檻；自碰撞逐對）＋不過則加密到逐物理步', ms_traj=round((t1 - t0) * 1e3, 1), ms_eval0=round((t2 - t1) * 1e3, 1), ms_cert=round((time.perf_counter() - t2) * 1e3, 1), ms=round((time.perf_counter() - t0) * 1e3, 1))

    if len(cpus) >= 6:
        for p_ in pool.procs:                                         # 掃掠池程序共用 FAR_CPUS 第 3 個起的全部核（不一對一釘死：其中一核被外部程序佔用時排程器可避開）
            CPU.pin(p_.pid, cpus[2:])
    sched = dict(far_cpus=cpus, worker_affinity=sorted(os.sched_getaffinity(0)), pool_affinity=[sorted(os.sched_getaffinity(p_.pid)) for p_ in pool.procs], blas_runtime=blas_rb, blas_all_single=CPU.blas_all_single(blas_rb) if blas_rb else '沒有載入 OpenBLAS（系統 Python 參考 BLAS）')
    conn.send(('ready', 0, dict(sched=sched, est=oinfo, capture_base_pose=[float(x) for x in np.asarray(fr['q'], float)[:3]])))
    while True:
        m = conn.recv()
        if m is None:
            pool.close()
            break
        op, a = m
        ver = a.get('ver', 0)
        t_rx = time.perf_counter()
        try:
            if op == 'prior':                                          # 地圖先驗：場景登錄的門資產名目把手位姿（門關閉）；只供粗導航目標與目標關聯，不是影像估計
                import hashlib
                import door_asset_v3 as DA3, door_geom_v3 as DG
                sreg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
                ap_ = os.path.join(os.path.dirname(HERE), sreg['door']['asset'])
                dd = DA3.load(ap_)
                pl = dd['placement']['pose']
                T_prior = _T3([float(pl['x']), float(pl['y']), math.radians(float(pl['yaw_deg']))]) @ np.asarray(DG.handle_pose(dd, 0.0, dd['handle']['height']), float)
                st['T_prior'] = T_prior
                n = (T_prior[:3, :3] @ np.asarray(a_H, float)).copy(); n[2] = 0; n /= np.linalg.norm(n)
                ax = T_prior[:3, 0].copy(); ax[2] = 0; d = -ax / np.linalg.norm(ax)
                conn.send(('prior', ver, dict(yaw=math.atan2(-n[1], -n[0]), d=d[:2].tolist(), n=n[:2].tolist(), p0=(T_prior[:2, 3] + 0.45 * n[:2]).tolist(), handle_xy=T_prior[:2, 3].tolist(), frame='map',
                                              source=dict(asset=sreg['door']['asset'], sha256=hashlib.sha256(open(ap_, 'rb').read()).hexdigest(), scene_reg='evaluation/experiments/20261112_m8_house_scene.json', note='名目位姿（門關閉 φ=0）；不讀把手 prim 真值'))))
            elif op == 'anchor':                                         # 把手估計轉入連續的里程座標：T_odom_handle＝T_odom_base(擷取同刻)·T_base_handle
                # 乾跑的量測合成：歷史影像不是在本趟起點拍的，所以用起點的真值底盤位姿把估計換成「起點同刻、相對底盤」的量測（真值只用在這一步合成量測），
                # 再用同刻的里程位姿轉入里程座標。實機／Isaac 取像時這一步＝相機＋FK 直接給出的相對量測。
                if a.get('T_base_handle') is not None:                 # 取像模式：執行器給的實際量測（相對擷取同刻底盤），不用真值
                    T_base_handle = np.asarray(a['T_base_handle'], float)
                else:
                    T_base_handle = np.linalg.inv(np.asarray(a['T_truth_base'], float)) @ T_est_W
                T_est = np.asarray(a['T_odom_base'], float) @ T_base_handle
                _Tmo_a = np.asarray(a['T_mo'], float)                     # 搜尋階段的淨空：以錨定時的 map←odom 轉入地圖座標（每拍核對另用當下轉換）
                clear_fn = lambda x, _T=_T3(_Tmo_a) @ T_est, _m=_Tmo_a: W.clear_door(np.r_[to_map(np.asarray(x, float), _m), 0.0], _T)   # noqa: E731
                n = (T_est[:3, :3] @ np.asarray(a_H, float)).copy(); n[2] = 0; n /= np.linalg.norm(n)
                ax = T_est[:3, 0].copy(); ax[2] = 0; d = -ax / np.linalg.norm(ax)
                yaw = math.atan2(-n[1], -n[0])
                st['ref'] = dict(ver=1, t_obs=float(a.get('t_obs', 0.0)), t_obs_last=float(a.get('t_obs', 0.0)), last_seq=int(a.get('seq', -1)), hist=[], goal=None, n_step_rejected=0, n_goal_expired=0, T_fixed=T_est.copy())
                assoc = None
                if st.get('T_prior') is not None:                      # 目標關聯：估計（轉到地圖座標）對先驗名目位姿的差
                    Tm = _T3(_Tmo_a) @ T_est
                    dR = st['T_prior'][:3, :3].T @ Tm[:3, :3]
                    assoc = dict(dxy_m=round(float(np.linalg.norm(Tm[:2, 3] - st['T_prior'][:2, 3])), 4), dz_m=round(float(abs(Tm[2, 3] - st['T_prior'][2, 3])), 4), dang_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), 3))
                conn.send(('anchor', ver, dict(yaw=yaw, d=d[:2].tolist(), n=n[:2].tolist(), p0=(T_est[:2, 3] + 0.45 * n[:2]).tolist(), handle_xy=T_est[:2, 3].tolist(), assoc=assoc)))
            elif op == 'plan':
                t0 = time.perf_counter()
                pg, info = FR.select_pregrasp_first(T_est, np.asarray(a['q'], float), T_at, S_PRE, K, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS, budget_s=a.get('budget', 8.0))
                br, binfo = (None, None)
                if pg is not None:
                    prep_path = [np.r_[pg['p0'] - pg['d'] * (FR.L_IN - u_ * (FR.L_IN - FR.OBS_ALONG[0])), pg['yaw']] for u_ in np.linspace(0, 1, 9)]
                    br, binfo = FR.observe_branch(T_est, pg, np.asarray(a['q'], float), prep_path, T_at, a_H, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS)
                st['pg'], st['br'] = pg, br
                term_info = None
                if br is not None and a.get('full'):                    # 終端：門旁分支預抓位解（由門旁種子在錨點解抓取位→後退 S_PRE→沿線連續解到 a_pre）
                    a_n_ = T_est[:3, :3] @ np.asarray(a_H, float)
                    sd_, why_t = np.asarray(DS_SEED, float), None
                    steps_ = [(DS_ANCHOR_A, 0.0), (DS_ANCHOR_A, S_PRE)] + [(float(a_), S_PRE) for a_ in np.arange(DS_ANCHOR_A + 0.005, TERM['a_pre'] + 1e-9, 0.005)]
                    for a_c, s_r in steps_:
                        b_c = np.r_[pg['p0'] - pg['d'] * a_c, pg['yaw']]
                        Tt_ = np.asarray(T_at(T_est, 0.0), float).copy(); Tt_[:3, 3] += s_r * a_n_
                        qn_, rn_, sm_ = DFK.ik_closure_fc(Tt_, b_c, sd_)
                        if not (rn_ < 1e-6 and np.isfinite(qn_).all()) or float(np.abs(qn_ - sd_).max()) > 0.6 or np.any(qn_ < LO) or np.any(qn_ > HI) or sm_ < W.SIG_FLOOR:
                            why_t = 'terminal_ik_limit_or_manipulability@a=%.3f,s=%.3f' % (a_c, s_r); break
                        if s_r >= S_PRE - 1e-9:                         # 抓取位（s=0）的兩指—橫桿淨空由抓持控制器的豁免處理；這裡只核預抓位
                            cl_ = clear_fn(np.r_[b_c, qn_])
                            if any(cl_[k_] < W.THR[k_] for k_ in ('env', 'base', 'self')):
                                why_t = 'terminal_clearance@a=%.3f' % a_c; break
                        sd_ = qn_
                    if why_t is None:
                        st['q_pre'] = sd_
                        term_info = dict(q_pre=[float(x) for x in sd_], a_pre=TERM['a_pre'], a_lock=TERM['a_lock'], t_desc=TERM['t_desc'], need_along_prep=TERM['a_obs'] + br['prep_time_s'] * V_PASS + PREP_MARGIN)
                    else:
                        term_info = dict(fail=why_t); br = None; st['br'] = None
                if br is not None:
                    q_s_, _, _ = DFK.ik_closure_fc(br['T_target'], np.r_[pg['p0'] - pg['d'] * STOP_ALONG, pg['yaw']], br['q_window'][int(np.argmin(np.abs(np.asarray(br['alongs']) - STOP_ALONG)))])   # 種子＝已核分支在終點附近的解（同一連續分支）
                    st['q_end_seed'] = q_s_
                st['trk'] = FR.ArmPreTracker(tau, off_fn, DT, PHYS, NPH, LO, HI)
                if br is not None:
                    stop_check(np.r_[np.asarray(a['q'], float)[:3], br['q_start_op']], [np.zeros(9)] * D, np.zeros(3), br['q_start_op'], np.zeros(6), np.zeros(3))   # 預熱（不在控制拍內）
                conn.send(('plan', ver, dict(terminal=term_info, ok=bool(pg is not None and br is not None), wall_s=round(time.perf_counter() - t0, 2), select=info, observe=binfo,
                                        pg=None if pg is None else dict(p0=pg['p0'].tolist(), d=pg['d'].tolist(), n=pg['n'].tolist(), yaw=pg['yaw'], standoff=pg['standoff'], along=pg['along']),
                                        br=None if br is None else dict(q_start_op=br['q_start_op'].tolist(), alongs=[float(br['alongs'][0]), float(br['alongs'][1]), float(br['alongs'][-1])], sigma_min=br['sigma_min'], min_clear=br['min_clear'],
                                                                         prep_time_s=br['prep_time_s'], track_rate_max=br['track_rate_max']))))
            elif op == 'off_anchor':                                    # 以無接觸、手臂靜止的保持狀態作錨（監督節點在手臂準備開始前送）
                q_a, s_a = np.asarray(a['q'], float), np.asarray(a['s'], float)
                st['off'] = dict(off0=q_a[3:] - s_a, G0=np.asarray(grav(q_a), float))
                qs_ = np.asarray(st['br']['q_start_op'], float)
                conn.send(('off_anchor', ver, dict(off0=[float(x) for x in st['off']['off0']], G0=[float(x) for x in st['off']['G0']], off_at_start_op=[float(x) for x in off_fn(np.r_[q_a[:3], qs_])])))
            elif op == 'stop_check':                                   # 離線重放／測試用：直接算一次停止核對
                conn.send(('stop_check', ver, stop_check(np.asarray(a['q'], float), [np.asarray(u, float) for u in a['pipe']], np.asarray(a['rel'], float), np.asarray(a['s'], float), np.asarray(a['ua'], float), np.asarray(a['T_mo'], float))))
            elif op == 'tick':
                q, pipe, rel, s, s_end = np.asarray(a['q'], float), [np.asarray(u, float) for u in a['pipe']], np.asarray(a['rel'], float), np.asarray(a['s'], float), np.asarray(a['s_end'], float)
                pg, br, out = st['pg'], st['br'], {}
                bp = [u[:3] for u in pipe]
                if a['mode'] == 'PREP':
                    dcs, di = FR.track_start_decision(q, bp + [rel] * NPH, pg, q[3:], br, PHYS)
                    out['decision'] = (dcs, di)
                    out['stop'] = stop_check(q, pipe, rel, s, np.asarray(a['ua_cand'], float), np.asarray(a['T_mo'], float))
                if a['mode'] in ('OBSERVE', 'DESCEND', 'PRETRACK'):
                    T_mo_t = np.asarray(a['T_mo'], float)
                    ref = st['ref']
                    tinfo = dict(ver=ref['ver'])
                    if a['mode'] == 'OBSERVE':                          # 手臂靜止；只累積／形成更新目標（不提交）；到鎖定點才試算並提交
                        ob = a.get('obs')
                        if ref['hist'] and float(a['t']) - ref['hist'][-1][1] > UPD['max_age_s']:
                            ref['hist'], ref['goal'] = [], None
                        if ob is not None and ob['seq'] > ref['last_seq']:
                            ref['last_seq'] = ob['seq']
                            To = np.asarray(ob['T'], float); age = float(a['t']) - float(ob['t_obs'])
                            if np.isfinite(To).all() and ob['t_obs'] > ref['t_obs_last'] and -1e-6 <= age <= UPD['max_age_s']:
                                dR = T_est[:3, :3].T @ To[:3, :3]
                                dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
                                dpos = float(np.linalg.norm(To[:3, 3] - T_est[:3, 3]))
                                if dpos <= UPD['gate_pos_m'] and dang <= ASSOC_TOL['dang_deg']:   # r4：OBSERVE 的姿態粗差閘門改用目標關聯容許 10°（far_door_3：遠處固定參考姿態偏約 3°，近距準確影格被 3° 閘門清掉）；一致性由三格兩兩角差與預算把關
                                    ref['t_obs_last'] = float(ob['t_obs'])
                                    ref['hist'] = (ref['hist'] + [(int(ob['seq']), float(ob['t_obs']), To[:3, 3].copy(), dang, To[:3, :3].copy())])[-UPD['n_consist']:]
                                    P = np.array([h_[2] for h_ in ref['hist']])
                                    spread = float(max(np.linalg.norm(P[i_] - P[j_]) for i_ in range(len(P)) for j_ in range(len(P)))) if len(P) > 1 else 0.0
                                    if len(P) >= UPD['n_consist'] and spread <= UPD['spread_m']:
                                        Uq, _, Vq = np.linalg.svd(np.sum([h_[4] for h_ in ref['hist']], axis=0))      # r3：鎖定姿態＝近距三格旋轉的弦平均（far_door_2：近距姿態誤差中位 0.12°，沿用的遠處姿態 1.58°）
                                        Rm = Uq @ np.diag([1.0, 1.0, np.linalg.det(Uq @ Vq)]) @ Vq
                                        HH = [h_[4] for h_ in ref['hist']]                     # 一致性項＝三格兩兩最大角差（Codex 20261117）
                                        dspread = max(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(HH[i_].T @ HH[j_]) - 1) / 2)))) for i_ in range(len(HH)) for j_ in range(i_ + 1, len(HH)))
                                        ref['goal'] = dict(p=np.median(P, axis=0), R=Rm, seqs=[h_[0] for h_ in ref['hist']], t_obs=float(ob['t_obs']), spread=spread, dang=dspread, dang_vs_fixed=max(h_[3] for h_ in ref['hist']))
                                    tinfo['obs'] = dict(seq=int(ob['seq']), age=round(age, 3), dpos_mm=round(dpos * 1e3, 2), n_hist=len(P), spread_mm=round(spread * 1e3, 2))
                                else:
                                    ref['hist'], ref['goal'] = [], None
                                    tinfo['obs'] = dict(seq=int(ob['seq']), reject='gate:%.4f_m,%.2f_deg' % (dpos, dang))
                            else:
                                tinfo['obs'] = dict(seq=int(ob['seq']), reject='stale_or_invalid:%.3f' % age)
                        ua0 = np.zeros(6)
                        out['stop'] = stop_check(q, pipe, rel, s, ua0, T_mo_t)
                        if a.get('want_lock'):                          # ---- 鎖定：以實測狀態、設定點、真管線試算下降首拍與停止核對；全過才提交
                            g_ = ref.get('goal')
                            lk = dict(ok=False)
                            if g_ is None:
                                lk['why'] = 'no_consistent_observation'
                            elif float(a['t']) - g_['t_obs'] > UPD['max_age_s']:
                                lk['why'] = 'lock_observation_not_fresh:%.3f' % (float(a['t']) - g_['t_obs'])
                            else:
                                T_new = T_est.copy(); T_new[:3, 3] = g_['p']; T_new[:3, :3] = g_['R']   # r3：位置與姿態都取近距三格（姿態預算項改為三格彼此的角度差）
                                dp_ = g_['p'] - T_est[:3, 3]
                                p0_new = pg['p0'] + dp_[:2]
                                b_pre = np.r_[p0_new - pg['d'] * TERM['a_pre'], pg['yaw']]
                                Tp_ = np.asarray(T_at(T_new, S_PRE), float)
                                q_pre, r_p, s_p = DFK.ik_closure_fc(Tp_, b_pre, st['q_pre'])
                                okp = bool(r_p < 1e-6 and np.isfinite(q_pre).all() and float(np.abs(q_pre - st['q_pre']).max()) < 0.3 and np.all(q_pre >= LO) and np.all(q_pre <= HI) and s_p >= W.SIG_FLOOR)
                                if not okp:
                                    lk['why'] = 'pregrasp_not_reachable_for_locked_reference'
                                else:
                                    off_pre = off_fn(np.r_[b_pre, q_pre])
                                    sp_goal = q_pre - off_pre
                                    al_now = float(-((q[:2] - p0_new) @ pg['d']))
                                    T_map_n = _T3(T_mo_t) @ T_new
                                    worst, bad = dict(env=9.9, base=9.9, self=9.9), None
                                    for u_ in np.linspace(0.0, 1.0, 12):   # 下降關節直線的淨空與限位（底盤沿線同步前進）；逐拍另有停止核對
                                        qq = q[3:] + _trap(u_) * (q_pre - q[3:])
                                        bb = np.r_[p0_new - pg['d'] * (al_now - u_ * V_PASS * TERM['t_desc']), pg['yaw']]
                                        if np.any(qq < LO) or np.any(qq > HI):
                                            bad = 'limit@u=%.2f' % u_; break
                                        cl_ = W.clear_door(np.r_[to_map(np.r_[bb, qq], T_mo_t), 0.0], T_map_n)
                                        for k_ in worst:
                                            worst[k_] = min(worst[k_], float(cl_[k_]) - W.THR[k_])
                                        if any(cl_[k_] < W.THR[k_] for k_ in ('env', 'base', 'self')):
                                            bad = 'clearance@u=%.2f' % u_; break
                                    if bad is not None:
                                        lk['why'] = 'descent_path:' + bad
                                    else:
                                        s_end_ = np.asarray(a['s_end'], float)
                                        ua1 = np.clip(((s_end_ + _trap(DT / TERM['t_desc']) * (sp_goal - s_end_)) - s_end_) / DT, -FR.V_ARM, FR.V_ARM)
                                        T_keep = T_est
                                        T_est = T_new                    # 試算用新參考（停止核對內的把手幾何）
                                        sc1 = stop_check(q, pipe, rel, s, ua1, T_mo_t)
                                        if stop_disposition('tick', sc1) != 'commit':
                                            T_est = T_keep
                                            lk['why'] = 'first_descent_tick_stop_check'; lk['check'] = dict(now=sc1['now']['ok'], next=sc1['next']['ok'], why=sc1.get('why'))
                                        else:                              # 提交鎖定：位姿、時間、序號、版本一起
                                            pg['p0'] = p0_new
                                            ref['ver'] += 1; ref['t_obs'] = g_['t_obs']
                                            st['q_pre'] = q_pre
                                            st['lock'] = dict(t=float(a['t']), t_obs=g_['t_obs'], age_at_lock_s=float(a['t']) - g_['t_obs'], seqs=g_['seqs'], spread=g_['spread'], dang=g_['dang'], T=T_new.copy(), base_at_lock=q[:3].copy(), ver=ref['ver'],
                                                              s0=s_end_.copy(), sp_goal=sp_goal, t0=float(a['t']), T_pre=Tp_)
                                            st['trk'].q_prev, st['trk'].u_prev = None, np.zeros(6)
                                            lk = dict(ok=True, T=T_new.tolist(), dang=g_['dang'], dang_vs_fixed_deg=round(g_['dang_vs_fixed'], 3), spread=g_['spread'], ver=ref['ver'], t_obs=g_['t_obs'], age_at_lock_s=round(float(a['t']) - g_['t_obs'], 3), seqs=g_['seqs'], spread_mm=round(g_['spread'] * 1e3, 2), shift_mm=[round(float(x) * 1e3, 2) for x in dp_], along_at_lock=round(al_now, 4),
                                                      q_pre=[float(x) for x in q_pre], sp_goal=[float(x) for x in sp_goal], s0=[float(x) for x in s_end_], pg_p0=[float(x) for x in p0_new], descent_margin={k_: round(v_, 4) for k_, v_ in worst.items()}, first_tick_stop=dict(ms=sc1['ms']))
                                            out['stop'] = sc1
                            out['lock'] = lk
                    else:
                        lk_ = st['lock']
                        if a['mode'] == 'DESCEND':                      # 設定點沿關節直線（梯形速度）到 預抓位解−偏置；命令＝(下一拍設定點目標−在途末端設定點)/dt
                            u_n = min((float(a['t']) + DT - lk_['t0']) / TERM['t_desc'], 1.0)
                            s_des = lk_['s0'] + _trap(u_n) * (lk_['sp_goal'] - lk_['s0'])
                            ua_d = np.clip((s_des - np.asarray(a['s_end'], float)) / DT, -FR.V_ARM, FR.V_ARM)
                            out['desc'] = dict(u=round(u_n, 4), ua=[float(x) for x in ua_d], done=bool(u_n >= 1.0), q_err_to_pre=round(float(np.abs(q[3:] - st['q_pre']).max()), 4))
                            out['stop'] = stop_check(q, pipe, rel, s, ua_d, T_mo_t)
                        else:                                           # PRETRACK：追「鎖定參考後退 S_PRE」的世界目標（抓持控制器接手後第一拍的目標）
                            trk = st['trk']
                            o = trk.step(q, bp, rel, s, s_end, lk_['T_pre'], trk.q_prev if trk.q_prev is not None else st['q_pre'])
                            out['track'] = dict(ok=o['ok'], why=o.get('why'), u=[float(x) for x in o['u']], err=o.get('track_err'), err_pred=o.get('track_err_pred'), clipped=o.get('clipped'), sigma=o.get('sigma'))
                            if o['ok']:
                                out['stop'] = stop_check(q, pipe, rel, s, np.asarray(o['u'], float), T_mo_t)
                        # ---- 鎖定任務參考的誤差預算（每拍）
                        travel = float(np.linalg.norm(q[:2] - lk_['base_at_lock'][:2]))
                        Tc_ = DFK.FC.fk_jac(q)[0]
                        tcp_err = float(np.linalg.norm(Tc_[:3, 3] - lk_['T_pre'][:3, 3])) if a['mode'] == 'PRETRACK' else float('nan')
                        dRt = lk_['T_pre'][:3, :3].T @ Tc_[:3, :3]
                        tcp_ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dRt) - 1) / 2)))) if a['mode'] == 'PRETRACK' else float('nan')
                        terms = dict(est_m=max(TERM['est_bound_m'], lk_['spread']), sync_m=V_PASS * TERM['sync_s'], odom_m=TERM['odom_frac'] * travel, arm_tcp_m=tcp_err)
                        pos_b = terms['est_m'] + terms['sync_m'] + terms['odom_m'] + (tcp_err if np.isfinite(tcp_err) else 0.0)
                        ang_b = TERM['est_bound_deg'] + lk_['dang'] + (tcp_ang if np.isfinite(tcp_ang) else 0.0)
                        bud = dict(pos_budget_m=round(pos_b, 5), ang_budget_deg=round(ang_b, 3), terms={k_: (None if not np.isfinite(v_) else round(v_, 5)) for k_, v_ in terms.items()}, travel_since_lock_m=round(travel, 4), elapsed_since_lock_obs_s=round(float(a['t']) - lk_['t_obs'], 3),
                                   age_at_lock_s=lk_['age_at_lock_s'], within=bool(pos_b <= FL.LOCK_BUDGET['pos_m'] and ang_b <= FL.LOCK_BUDGET['ang_deg']), arm_term_included=bool(np.isfinite(tcp_err)))
                        out['budget'] = bud
                        if a['mode'] == 'PRETRACK' and a.get('judge'):
                            sh = type('A', (), dict(owner='NAV'))()
                            pkt, why = FL.handover_request(a['t'], sh, dict(pg, p0=pg['p0'] + GRASP_ORIGIN_SHIFT * pg['d']), q, a['v_meas'], s, pipe, a['u_last'], a['u_prev'], dict(T_WH=lk_['T'], t_obs=lk_['t_obs']), V_PASS, D, off_static=off_fn(q), tick_phase=0,
                                                           stamps=dict(image=lk_['t_obs'], state=a['t'], cmd=a['t']), arm_track_err=o.get('track_err') if o['ok'] else None, locked=bud)
                            out['judge'] = dict(locked_reference=why)
                            out['handover_ok'] = bool(pkt is not None)
                    out['term'] = tinfo
                if a['mode'] == 'TRACK':
                    trk = st['trk']
                    T_mo_t = np.asarray(a['T_mo'], float)
                    ref = st['ref']
                    upd = dict(ver=ref['ver'], status='off' if not a.get('allow_update') else 'no_new_obs')
                    ob = a.get('obs')
                    if ref.get('goal') is not None and float(a['t']) - ref['goal']['t_obs'] > UPD['max_age_s']:   # 每拍重查更新目標的來源年齡：過期 ⇒ 停止位置更新，保留最後已提交參考及其原始來源時間；歷史清空（中斷後重新累積）
                        ref['goal'], ref['hist'] = None, []
                        ref['n_goal_expired'] += 1
                        upd['goal_expired'] = True
                    if ref['hist'] and float(a['t']) - ref['hist'][-1][1] > UPD['max_age_s']:
                        ref['hist'] = []
                    if a.get('allow_update') and ob is not None and ob['seq'] > ref['last_seq']:      # ---- 更新入口：一格新觀察（位姿、時間、版本一起進來）
                        ref['last_seq'] = ob['seq']
                        To = np.asarray(ob['T'], float)
                        age = float(a['t']) - float(ob['t_obs'])
                        dR = T_est[:3, :3].T @ To[:3, :3]
                        dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))) if np.isfinite(To).all() else float('nan')
                        dpos = float(np.linalg.norm(To[:3, 3] - T_est[:3, 3])) if np.isfinite(To).all() else float('nan')
                        if not np.isfinite(To).all():
                            upd['status'] = 'reject:non_finite'
                        elif ob['t_obs'] <= ref['t_obs_last']:
                            upd['status'] = 'reject:time_not_increasing'
                        elif age > UPD['max_age_s'] or age < -1e-6:
                            upd['status'] = 'reject:age:%.3f' % age
                        elif dpos > UPD['gate_pos_m'] or dang > UPD['gate_ang_deg']:
                            upd['status'] = 'reject:gate:%.4f_m,%.2f_deg' % (dpos, dang); ref['hist'] = []
                        else:
                            ref['t_obs_last'] = float(ob['t_obs'])
                            ref['hist'] = (ref['hist'] + [(int(ob['seq']), float(ob['t_obs']), To[:3, 3].copy())])[-UPD['n_consist']:]
                            P = np.array([h_[2] for h_ in ref['hist']])
                            spread = float(max(np.linalg.norm(P[i_] - P[j_]) for i_ in range(len(P)) for j_ in range(len(P)))) if len(P) > 1 else 0.0
                            if len(P) < UPD['n_consist']:
                                upd['status'] = 'accumulating:%d' % len(P)
                            elif spread > UPD['spread_m']:
                                upd['status'] = 'reject:spread:%.4f' % spread
                            else:
                                ref['goal'] = dict(p=np.median(P, axis=0), seqs=[h_[0] for h_ in ref['hist']], t_obs=float(ob['t_obs']), spread=spread)
                                upd['status'] = 'goal_set'
                        upd.update(obs_seq=int(ob['seq']), obs_age=round(age, 3), obs_dpos_mm=round(dpos * 1e3, 2) if np.isfinite(dpos) else None)
                    step_try = None
                    if a.get('allow_update') and ref.get('goal') is not None:                       # ---- 參考朝更新目標移動（每拍 ≤ step_m）；先試算，全部核過才提交
                        dv = ref['goal']['p'] - T_est[:3, 3]
                        nd = float(np.linalg.norm(dv))
                        if nd > 1e-5:
                            step_try = dv * min(1.0, UPD['step_m'] / nd)
                        else:                                          # 參考已與更新目標重合（≤0.01 mm）：該組觀察確認了現參考 ⇒ 來源時間更新為該組觀察的時間（位姿沒變，不升版）
                            ref['t_obs'] = ref['goal']['t_obs']
                            upd['confirmed_by_obs'] = True
                    o = None
                    if step_try is not None:
                        old = (T_est, br['T_target'], pg['p0'].copy(), None if trk.q_prev is None else trk.q_prev.copy(), trk.u_prev.copy(), trk.lag_max, trk.n_clip, trk.ready)
                        T_new = T_est.copy(); T_new[:3, 3] += step_try
                        T_est = T_new
                        br['T_target'] = FR.observe_target(T_est, T_at, a_H)
                        pg['p0'] = pg['p0'] + step_try[:2]
                        b_end = np.r_[pg['p0'] - pg['d'] * STOP_ALONG, pg['yaw']]                     # 可達性：更新後的目標在本段終點（通過 0.05 m）仍有解、在限位內、可操作度足夠
                        q_e, r_e, s_e = DFK.ik_closure_fc(br['T_target'], b_end, st['q_end_seed'])
                        reach_ok = bool(r_e < 1e-6 and np.isfinite(q_e).all() and np.all(q_e >= LO) and np.all(q_e <= HI) and s_e >= W.SIG_FLOOR and float(np.abs(q_e - st['q_end_seed']).max()) < 0.5)
                        o = trk.step(q, bp, rel, s, s_end, br['T_target'], trk.q_prev if trk.q_prev is not None else q[3:])
                        sc_n = stop_check(q, pipe, rel, s, np.asarray(o['u'], float), T_mo_t) if (o['ok'] and reach_ok) else None
                        if o['ok'] and reach_ok and o.get('sigma', 0.0) >= W.SIG_FLOOR and stop_disposition('tick', sc_n) == 'commit':
                            ref['ver'] += 1
                            ref['t_obs'] = ref['goal']['t_obs']
                            st['q_end_seed'] = q_e
                            upd.update(ver=ref['ver'], committed=True, step_mm=round(float(np.linalg.norm(step_try)) * 1e3, 3), from_seqs=ref['goal']['seqs'], t_obs=ref['t_obs'])
                            out['stop'] = sc_n
                        else:                                          # 任一不過：退回原參考（追蹤器狀態一併還原），本拍照原參考重算
                            why_ = 'tracker' if not o['ok'] else ('reach' if not reach_ok else ('manipulability' if o.get('sigma', 0.0) < W.SIG_FLOOR else 'stop_check'))
                            T_est, br['T_target'], pg['p0'] = old[0], old[1], old[2]
                            trk.q_prev, trk.u_prev, trk.lag_max, trk.n_clip, trk.ready = old[3], old[4], old[5], old[6], old[7]
                            ref['n_step_rejected'] += 1
                            upd.update(committed=False, step_rejected=why_)
                            o = None
                    if o is None:
                        o = trk.step(q, bp, rel, s, s_end, br['T_target'], trk.q_prev if trk.q_prev is not None else q[3:])
                        if o['ok']:
                            out['stop'] = stop_check(q, pipe, rel, s, np.asarray(o['u'], float), T_mo_t)
                    upd.update(ref_p=[float(x) for x in T_est[:3, 3]], ref_t_obs=ref['t_obs'], pg_p0=[float(x) for x in pg['p0']], goal_dist_mm=None if ref.get('goal') is None else round(float(np.linalg.norm(ref['goal']['p'] - T_est[:3, 3])) * 1e3, 3), n_step_rejected=ref['n_step_rejected'], n_goal_expired=ref['n_goal_expired'])
                    out['ref'] = upd
                    out['track'] = dict(ok=o['ok'], why=o.get('why'), u=[float(x) for x in o['u']], err=o.get('track_err'), err_pred=o.get('track_err_pred'), clipped=o.get('clipped'), sigma=o.get('sigma'))
                    if a.get('judge'):
                        j = {}
                        for nm, Tj, tobs in a['judge_refs']:
                            if nm == 'updated_reference':               # 用 worker 本拍實際提交的參考：位姿、來源時間、版本一起（不取監督節點傳入的上一拍時間）
                                Tj, tobs = T_est, st['ref']['t_obs']
                                out['judge_updated_ref'] = dict(ver=st['ref']['ver'], t_obs=tobs)              # 每個參考各自帶「位姿、時間」：固定參考用它自己的擷取時間；逐格觀察用該格自己的位姿與時間
                            sh = type('A', (), dict(owner='NAV'))()
                            _, why = FL.handover_request(a['t'], sh, pg, q, a['v_meas'], s, pipe, a['u_last'], a['u_prev'], dict(T_WH=T_est if Tj is None else np.asarray(Tj, float), t_obs=tobs), V_PASS, D, off_static=np.zeros(6), tick_phase=0,
                                                         stamps=dict(image=tobs, state=a['t'], cmd=a['t']), arm_track_err=o.get('track_err') if o['ok'] else None)
                            j[nm] = why
                        out['judge'] = j
                if a['mode'] in ('PREP', 'TRACK') or a.get('want_clear'):
                    T_mo_ = np.asarray(a['T_mo'], float)
                    cl = W.clear_door(np.r_[to_map(q, T_mo_), 0.0], _T3(T_mo_) @ T_est)
                    out['clear'] = {k_: float(cl[k_]) for k_ in ('env', 'base', 'self')}
                out['tm'] = (t_rx, time.perf_counter())             # 收到請求、算完的時刻（CLOCK_MONOTONIC，跨程序可比）
                conn.send(('tick', ver, out))
        except Exception as e:                                        # noqa: BLE001
            conn.send(('error', ver, repr(e)[:300]))


def main():
    out = sys.argv[1]
    UPDATE = len(sys.argv) > 2 and sys.argv[2] == 'obsupd'           # obsupd＝取像模式＋近距觀察段連續視覺更新（N3）
    HANDOVER = len(sys.argv) > 2 and sys.argv[2] == 'fullho'         # fullho＝full＋交接判定通過即交出命令權給執行器行程內的抓持／開門控制器
    FULL = len(sys.argv) > 2 and sys.argv[2] in ('full', 'fullho')               # full＝取像模式＋終端接近 T1b（準備提前→靜止觀察→鎖定→下降→追預抓位→交接判定）
    OBS_MODE = len(sys.argv) > 2 and sys.argv[2] in ('obs', 'obsupd', 'full', 'fullho')            # 取像模式：把手估計來自執行器 /far/handle_obs 的實際影像（起點第一格有效估計定通過線；之後各格只記錄與更新時間戳，不改線）
    ctx = mp.get_context('fork')
    pc, cc = ctx.Pipe()
    wp = ctx.Process(target=worker, args=(cc,), daemon=False)          # 非 daemon：它要再開平行掃掠池；以 PDEATHSIG 跟著父程序結束
    wp.start()                                                       # 先 fork（之後才匯入 rclpy）
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool, Float64MultiArray, String
    from tf2_ros import Buffer, TransformListener
    if not pc.poll(120.0):
        raise SystemExit('worker 未就緒')
    _, _, W0 = pc.recv()
    _fc = [int(x) for x in os.environ.get('FAR_CPUS', '').split(',') if x.strip()]
    if len(_fc) >= 6:
        os.sched_setaffinity(0, set(_fc[0:2]))
    W0['sched']['main_affinity'] = sorted(os.sched_getaffinity(0))
    co = {}
    rclpy.init()

    def w2(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    class M(Node):
        def __init__(self):
            super().__init__('far_merge', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.owner, self.arm, self.stop = 'NAV', 'HOLD', False
            self.ev, self.rows, self.sel_log = [], [], []
            self.truth, self.rel, self.rel_t = None, np.zeros(3), None
            self.last = dict(NAV=np.zeros(3), MERGE=np.zeros(3), GRASP=np.zeros(3))
            self.create_subscription(Twist, '/cmd_vel_grasp', lambda m: self._in('GRASP', m), 20)   # 抓持／開門控制器（執行器行程內）的底盤提案：同樣經選擇器→平滑器→Shield→GUARD
            self.p_ho = self.create_publisher(Float64MultiArray, '/far/handover', 10)
            self.create_subscription(String, '/far/handover_ack', self._hoack, 10)
            self.ho_pending, self.ho_ack, self.t_owner_grasp = None, None, None
            self.p_nav = self.create_publisher(Twist, '/cmd_vel_nav', 10)
            self.p_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
            self.p_plan = self.create_publisher(Path, '/merge/plan', 10)
            self.p_arm = self.create_publisher(Float64MultiArray, '/far/arm_cmd', 10)
            self.create_subscription(Twist, '/cmd_vel_nav_raw', lambda m: self._in('NAV', m), 20)
            self.create_subscription(Twist, '/cmd_vel_merge', lambda m: self._in('MERGE', m), 20)
            self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
            self.create_subscription(Float64MultiArray, '/far/truth', self._truth, 20)
            self.create_subscription(String, '/merge/gmpc/plan_ack', self._ack, 10)
            self.create_subscription(Float64MultiArray, '/far/handle_obs', self._obs, 20)
            self.create_subscription(String, '/far/exec_state', self._xstate, 10)
            self.p_done = self.create_publisher(Bool, '/far/done', 10)
            self.p_abort = self.create_publisher(String, '/far/abort', 10)
            self.aborted = None
            self.obs_first, self.obs_latest, self.obs_log, self.xstate, self.t_first_tick = None, None, [], None, None
            self.anchored, self.n_assoc_rej, self.t_near_goal = False, 0, None
            self.ref_t_obs, self.ref_log, self.ref_ver, self.path_shift = 0.0, [], 1, np.zeros(2)
            self.term_log, self.lock, self.t_handover_ok, self.budget_log = [], None, None, []
            self.p_act = self.create_publisher(Bool, '/merge/active', 10)
            self.tfb = Buffer(); self.tfl = TransformListener(self.tfb, self)
            self.ver, self.stale, self.T_mo_pub, self.path_odom, self.plan_stamp, self.ack, self.t_shadow, self.mrg_t = 0, 0, None, None, None, None, None, None
            self.anchor = None
            self.tm_log = []                                           # 每次請求：[t, 是否 tick, 派工(送出) ms, 排隊 ms, 計算 ms, 收回 ms, 總 ms, 遲到]
            self.tf_stamp, self.v_last = {}, np.zeros(2)
            self.plan, self.plan_req_t, self.plan_ready_t, self.pending = None, None, None, None
            self.gm, self.t_goal, self.t_prep0, self.s_prep0, self.u_arm, self.u_arm_prev = None, None, None, None, np.zeros(6), np.zeros(6)
            self.q_prev, self.t_stopcheck, self.n_fwd, self.no_reach = None, None, dict(NAV=0, MERGE=0, GRASP=0, dropped=0), None
            self.wait_tick = False
            self.n_goal = 1
            self.timer = self.create_timer(DT, self.tick)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def event(self, what, **kw):
            self.ev.append(dict(t=round(self.now(), 3), what=what, **kw))
            self.get_logger().info(f'[far] {what} {kw}')

        def _ack(self, m):
            self.ack = m.data

        def _repin_after_handover(self):
            """交權後本節點只剩命令選擇器在工作（幾何程序與掃掠池閒置）：把主程序移到導航鏈那組核，讓出抓持／開門計算用的核（只改排程）。"""
            cp = [int(x) for x in os.environ.get('FAR_AFTER_HO_CPUS', '').split(',') if x.strip()]
            if cp:
                try:
                    os.sched_setaffinity(0, set(cp))
                except Exception as e:                                # noqa: BLE001
                    self.repin = dict(error=repr(e)[:100]); return
            self.repin = dict(requested=cp, readback=sorted(os.sched_getaffinity(0)))

        def _hoack(self, m):
            self.ho_ack = m.data                                      # 'ack:<ver>' 或 'nack:<ver>:<原因>'（執行器以當下快照試跑首拍的結果）
            if self.ho_pending is not None and self.ho_ack == 'ack:%d' % self.ho_pending and getattr(self, 'grasp_stash', None) is not None and self.owner != 'GRASP':
                self.owner, self.t_owner_grasp = 'GRASP', self.now()
                self.p_nav.publish(self.grasp_stash); self.n_fwd['GRASP'] += 1
                self.grasp_stash = None
                self._repin_after_handover()

        def _obs(self, m):
            d = np.array(m.data, float)                               # seq, t_obs, valid, odom_base(3), T_base_handle(16), t_publish
            valid = bool(d[2] > 0.5)
            self.obs_log.append(dict(seq=int(d[0]), t_obs=float(d[1]), valid=valid, t_rx=round(self.now(), 3)))
            if valid:                                                 # 逐格觀察：位姿（里程座標，以該格擷取同刻的里程位姿轉入）、時間、版本一起保存
                o_ = dict(seq=int(d[0]), t_obs=float(d[1]), odom_base=d[3:6].copy(), T_base_handle=d[6:22].reshape(4, 4).copy())
                o_['T_odom_handle'] = _T3(o_['odom_base']) @ o_['T_base_handle']
                self.obs_latest = o_
                if self.obs_first is None and not self.anchored:
                    self.obs_first = o_

        def _xstate(self, m):
            self.xstate = m.data

        def tf2d(self, parent, child):
            try:
                tr = self.tfb.lookup_transform(parent, child, rclpy.time.Time())
            except Exception:                                         # noqa: BLE001
                return None
            o = tr.transform.rotation
            self.tf_stamp[child] = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
            return np.array([tr.transform.translation.x, tr.transform.translation.y, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))])

        def ask(self, op, a, timeout):
            """送一個帶版本的請求並等同版本回覆；舊版本回覆丟棄並計數。回 (tag, payload) 或 ('late', None)。"""
            self.ver += 1
            t_s0 = time.perf_counter()
            pc.send((op, dict(a, ver=self.ver)))
            t_s1 = time.perf_counter()
            t_end = t_s1 + timeout
            while True:
                rem = t_end - time.perf_counter()
                if rem <= 0 or not pc.poll(rem):
                    self.tm_log.append([self.now(), op == 'tick', (t_s1 - t_s0) * 1e3, np.nan, np.nan, np.nan, (time.perf_counter() - t_s0) * 1e3, 1.0])
                    return 'late', None
                tag, ver, res = pc.recv()
                if ver != self.ver:
                    self.stale += 1
                    continue
                t_r = time.perf_counter()
                tm = res.get('tm') if isinstance(res, dict) else None
                self.tm_log.append([self.now(), op == 'tick', (t_s1 - t_s0) * 1e3, np.nan if tm is None else (tm[0] - t_s1) * 1e3, np.nan if tm is None else (tm[1] - tm[0]) * 1e3, np.nan if tm is None else (t_r - tm[1]) * 1e3, (t_r - t_s0) * 1e3, 0.0])
                return tag, res

        def _in(self, src, m):
            v = np.array([m.linear.x, m.linear.y, m.angular.z])
            self.last[src] = v
            if src == 'MERGE':
                self.mrg_t = self.now()
            if self.stop:
                self.p_nav.publish(Twist()); self.n_fwd['dropped'] += 1
            elif src == 'GRASP' and self.owner != 'GRASP' and self.ho_pending is not None:
                # 交權邊界：須先有版本相符的 ack（＝執行器以當下快照試跑首拍通過）；第一筆 /cmd_vel_grasp 到達選擇器即為命令權切換點（自此 MERGE／NAV 丟棄）
                if self.ho_ack == 'ack:%d' % self.ho_pending:
                    self.owner, self.t_owner_grasp = 'GRASP', self.now()
                    self.p_nav.publish(m); self.n_fwd[src] += 1
                    self._repin_after_handover()
                elif self.ho_ack is None:
                    self.grasp_stash = m                               # 跨話題：首提案先於 ack 到 ⇒ 暫存最新一筆，ack 版本相符時才轉發
                else:
                    self.n_fwd['dropped'] += 1                         # nack 或版本不符：不交權
            elif src == self.owner:
                self.p_nav.publish(m); self.n_fwd[src] += 1
            else:
                self.n_fwd['dropped'] += 1

        def _rel(self, m):
            self.rel, self.rel_t = np.array([m.linear.x, m.linear.y, m.angular.z]), self.now()

        def _truth(self, m):
            a = np.array(m.data, float)
            self.truth = dict(t=a[0], q=a[1:10], s=a[10:16], s_end=a[16:22], pipe=a[22:94].reshape(8, 9))

        def start_merge_gmpc(self):
            side = out + '.side'
            os.makedirs(side, exist_ok=True)
            raw = subprocess.run(['ros2', 'param', 'dump', '/gmpc_controller'], capture_output=True, text=True, timeout=30).stdout
            if 'ros__parameters' not in raw:
                self.event('merge_gmpc_param_dump_failed'); return False
            open(os.path.join(side, 'params_gmpc_controller_live.yaml'), 'w').write(raw)
            lines = raw.split('\n')
            lines[0] = '/**:'
            pf = os.path.join(side, 'params_gmpc_merge_base.yaml')
            open(pf, 'w').write('\n'.join(lines))
            ov = dict(plan_topic='/merge/plan', cmd_vel_topic='/cmd_vel_merge', v_nominal=V_PASS, heading_fixed=float(w2(co['yaw'] + self.T_mo0[2])), goal_tolerance_xy=0.03, heading_enable=True)   # 朝向參考在 map 座標＝里程座標的通過線朝向＋起點的 map←odom 偏航
            json.dump(dict(base='現場 dump 的 /gmpc_controller 參數（逐項相同）', task_reference_overrides=ov, note='只覆寫任務參考；CBF、限制、權重、障礙來源未改'), open(os.path.join(side, 'merge_gmpc_overrides.json'), 'w'), ensure_ascii=False, indent=1)
            cmd = ['/usr/bin/python3', '-u', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gmpc_merge_wrap.py'), '--ros-args', '-r', '__node:=gmpc_merge', '--params-file', pf]   # 同一 GMPCNode 類別；外包只加「影子模式下以實際放行命令為上一命令」
            for k_, v_ in ov.items():
                cmd += ['-p', f'{k_}:={str(v_).lower() if isinstance(v_, bool) else v_}']
            for tname in ('diag', 'diag_v2', 'plan_ack', 'heading', 'solve_time_ms', 'min_h'):
                cmd += ['-r', f'/gmpc/{tname}:=/merge/gmpc/{tname}']
            self.gm = subprocess.Popen(cmd, stdout=open(out + '.gmpc_merge.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
            self.event('merge_gmpc_started', overrides=ov)
            return True

        def pub_goal(self, xy):
            m = PoseStamped(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1]); m.pose.orientation.w = 1.0
            self.p_goal.publish(m)

        def make_path_odom(self, q, pg):
            p0, dv = np.array(pg['p0']), np.array(pg['d'])
            al_now = float(-((q[:2] - p0) @ dv))
            br_ = self.plan['br']
            need = (self.plan['terminal']['need_along_prep'] if FULL else br_['alongs'][1] + br_['prep_time_s'] * V_PASS + PREP_MARGIN)     # 手臂準備開始時須有的沿線距離；full：準備須在沿線 a_obs 前完成
            al0 = max(al_now, need + ENTER_EXTRA)                      # F1：切入點＝max(目前沿線位置, 準備所需＋餘量)；沿線位置不足時斜向切到線上較遠的一點（全向底盤），仍由同一 GMPC／CBF 求解
            pts = [q[:2].copy()]
            entry = p0 - dv * al0
            n_in = max(int(np.linalg.norm(entry - q[:2]) / 0.005), 2)     # 路徑點距 5 mm：進線 GMPC 以最近路徑點為參考起點（ref_proj=vertex），點距粗時低速下會卡在兩點之間
            for u_ in np.linspace(1.0 / n_in, 1.0, n_in):
                pts.append(q[:2] + u_ * (entry - q[:2]))
            for al in np.arange(al0 - 0.005, (-0.30 if FULL else -0.10) - 1e-9, -0.005):     # 路徑延伸到已核觀察範圍終點（−0.10）；受控停止由監督節點在 STOP_ALONG 發起，不靠控制器到點
                pts.append(p0 - dv * al)
            self.path_odom, self.path_yaw = np.array(pts), float(pg['yaw'])
            return dict(n=len(pts), along_now=round(al_now, 3), along_entry=round(al0, 3), need_along=round(need, 3), diagonal=bool(al0 > al_now + 1e-9), end_odom=[round(float(v), 3) for v in pts[-1]])

        def pub_path_map(self, T_mo):
            """通過線保存在里程座標；發布給進線 GMPC（map 座標）時用當下的 map←odom 轉換。"""
            c, s_ = math.cos(T_mo[2]), math.sin(T_mo[2])
            P = self.path_odom @ np.array([[c, s_], [-s_, c]]) + T_mo[:2]
            yw = self.path_yaw + T_mo[2]
            m = Path(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            for x in P:
                ps = PoseStamped(); ps.header = m.header
                ps.pose.position.x, ps.pose.position.y = float(x[0]), float(x[1])
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
                m.poses.append(ps)
            self.p_plan.publish(m)
            self.T_mo_pub, self.plan_stamp, self.n_pub = T_mo.copy(), f'{m.header.stamp.sec}.{m.header.stamp.nanosec:09d}', getattr(self, 'n_pub', 0) + 1

        def engineering_abort(self, why, **kw):
            """模擬工程中止：通知執行器不再步進物理、保留現場資料。不是受控停止。"""
            if self.aborted is None:
                self.aborted = why
                for _ in range(3):
                    self.p_abort.publish(String(data=why))
                self.stop = True
                self.p_nav.publish(Twist())
                self.event('sim_engineering_abort', why=why, **kw)

        def global_stop(self, what, **kw):
            if not self.stop:
                self.stop = True
                self.p_nav.publish(Twist())
                self.event(what, **kw)

        def tick(self):
            tr = self.truth
            T_ob, T_mo = self.tf2d('odom', 'base_footprint'), self.tf2d('map', 'odom')
            if tr is None or T_ob is None or T_mo is None:
                return
            t = self.now()
            q = np.r_[T_ob, tr['q'][3:]]                                # 控制用位姿＝里程座標（TF odom→base_footprint）；手臂＝驅動端讀回。真值底盤位姿只記錄，不進任何判斷
            self.p_act.publish(Bool(data=self.owner == 'MERGE'))
            if self.owner == 'GRASP':                                 # 已交接：底盤命令權在抓持／開門控制器（提案仍經本節點選擇器→安全鏈）；本節點不再下任何手臂或底盤命令
                xs = self.xstate or ''
                if not getattr(self, 'ho_logged', False):
                    self.ho_logged = True
                    self.event('owner_GRASP_on_first_grasp_command', t_switch=round(self.t_owner_grasp, 3), ack=self.ho_ack, request_ver=self.ho_pending, repin=getattr(self, 'repin', None))
                if xs.startswith('done') or xs.startswith('latched'):
                    self.event('executor_finished_after_handover', state=xs, forwarded=dict(self.n_fwd))
                    raise SystemExit(0)
                if self.owner == 'GRASP':
                    return
            if self.xstate is not None and self.xstate.startswith('latched') and not self.stop:
                self.global_stop('executor_latched_global_stop', state=self.xstate)
            if self.t_first_tick is None:
                self.t_first_tick = t
            def send_coarse(note):
                self.pub_goal(self.coarse_goal); self.t_goal = t
                self.event('coarse_goal_sent', goal_map=[round(float(v), 3) for v in self.coarse_goal], map_odom=[round(float(v), 4) for v in T_mo], note=note)
                self.start_merge_gmpc()

            def request_plan():
                self.ver += 1
                pc.send(('plan', dict(q=q.tolist(), budget=8.0, ver=self.ver, full=FULL))); self.plan_req_t, self.plan_ver = t, self.ver

            def to_map_xy(p):
                c, s_ = math.cos(T_mo[2]), math.sin(T_mo[2])
                return np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1]]) + T_mo[:2]
            if self.t_goal is None and not OBS_MODE:                  # 乾跑：擷取同刻設為本趟起點、量測由真值合成；把手估計以同刻里程位姿轉入里程座標；粗目標立即送出；背景開始搜尋
                tag, res = self.ask('anchor', dict(T_odom_base=_T3(T_ob).tolist(), T_truth_base=_T3(tr['q'][:3]).tolist(), T_mo=T_mo.tolist()), 10.0)
                if tag != 'anchor':
                    self.global_stop('anchor_failed_global_stop', res=str(res)[:200]); raise SystemExit(1)
                co.update(res)
                self.anchor = dict(t=t, odom_base=T_ob.tolist(), truth_base=tr['q'][:3].tolist(), map_odom=T_mo.tolist(), capture_base_pose_of_image=W0['capture_base_pose'],
                                   note='乾跑設定：把「影像擷取同刻」定為本趟起點；把手相對底盤的量測取自 online_1 實際影像（該影像的實際擷取位姿與本趟起點不同，照實列出）')
                self.T_mo0, self.anchored = T_mo.copy(), True
                self.coarse_goal = to_map_xy(np.array(co['p0']) - np.array(co['d']) * 0.85)
                send_coarse('背景搜尋已送出')
                request_plan()
                return
            if self.t_goal is None and OBS_MODE:                      # 取像模式（S1）：粗導航目標來自地圖先驗（門資產名目位姿），不等影像；尚無有效觀察時 NAV 擁有底盤、手臂保持起始構型
                tag, res = self.ask('prior', {}, 10.0)
                if tag != 'prior':
                    self.global_stop('prior_failed_global_stop', res=str(res)[:200]); raise SystemExit(1)
                co.update(res)
                self.prior = res
                self.T_mo0 = np.zeros(3)                               # 先驗本來就在地圖座標
                self.coarse_goal = np.array(co['p0']) - np.array(co['d']) * float(os.environ.get('FAR_TEST_COARSE_ALONG', '0.85'))   # 0.85＝設計值；環境變數只供乾跑把導航到達位置往近端偏
                send_coarse('粗目標來自地圖先驗（' + res['source']['asset'] + '）；通過線待途中第一格有效影像估計')
                return
            if OBS_MODE and not self.anchored and self.obs_first is not None and self.owner == 'NAV' and not self.stop:
                o_ = self.obs_first                                   # 途中第一格有效觀察：以該格擷取同刻的里程位姿轉入里程座標，核對是否對應目標門，再定線並開始搜尋
                tag, res = self.ask('anchor', dict(T_odom_base=_T3(o_['odom_base']).tolist(), T_base_handle=o_['T_base_handle'].tolist(), T_mo=T_mo.tolist(), seq=o_['seq'], t_obs=o_['t_obs']), 0.5)
                if tag != 'anchor':
                    self.global_stop('worker_' + tag + '_global_stop', op='anchor', res=str(res)[:200])
                elif res['assoc'] is None or res['assoc']['dxy_m'] > ASSOC_TOL['dxy_m'] or res['assoc']['dz_m'] > ASSOC_TOL['dz_m'] or res['assoc']['dang_deg'] > ASSOC_TOL['dang_deg']:
                    self.n_assoc_rej += 1
                    self.event('observation_not_associated_with_target_door', seq=o_['seq'], assoc=res['assoc'], tol=ASSOC_TOL)
                    self.obs_first = None
                else:
                    co.update({k_: res[k_] for k_ in ('yaw', 'd', 'n', 'p0', 'handle_xy')})
                    self.anchored = True
                    self.ref_t_obs = o_['t_obs']
                    self.anchor = dict(t=t, seq=o_['seq'], t_obs=o_['t_obs'], age_s=round(t - o_['t_obs'], 3), odom_base=o_['odom_base'].tolist(), truth_base=tr['q'][:3].tolist(), map_odom=T_mo.tolist(), T_base_handle=o_['T_base_handle'].tolist(), T_odom_handle=o_['T_odom_handle'].tolist(), assoc=res['assoc'],
                                       version='fixed_reference_v1(seq %d)' % o_['seq'], note='固定參考＝途中第一格通過關聯核對的實際影像估計；之後各格另存，不改通過線')
                    request_plan()
                    self.event('visual_task_established', seq=o_['seq'], t_obs=round(o_['t_obs'], 3), age_s=round(t - o_['t_obs'], 3), assoc=res['assoc'], dist_to_coarse_goal=round(float(np.linalg.norm(to_map_xy(q[:2]) - self.coarse_goal)), 3))
            if self.owner == 'NAV' and self.n_goal < 3 and t - self.t_goal >= 1.0 * self.n_goal:   # 目標重送 3 次（剛建立的發布者可能尚未被訂閱端發現）
                self.pub_goal(self.coarse_goal); self.n_goal += 1
            if self.plan is None and self.plan_ready_t is None and pc.poll(0):
                tag, ver, res = pc.recv()
                if ver != self.plan_ver:
                    self.stale += 1
                else:
                    self.plan_ready_t = t
                    if tag == 'plan' and res['ok']:
                        self.plan = res
                        self.event('plan_ready', wall_s=res['wall_s'], sim_elapsed_s=round(t - self.plan_req_t, 2), n_cells=res['select'].get('n_cells'), standoff=res['pg']['standoff'], along_offset=res['pg']['along'], prep_time_s=round(res['br']['prep_time_s'], 2), sigma_min=round(res['br']['sigma_min'], 4))
                    else:
                        self.event('plan_failed', res=str(res)[:300])
            pg = None if self.plan is None else self.plan['pg']
            br = None if self.plan is None else self.plan['br']
            along = lat = None
            yaw_l = co['yaw']
            if pg is not None:
                r = q[:2] - np.array(pg['p0'])
                along, lat = float(-(r @ np.array(pg['d']))), float(r @ np.array(pg['n']))
            ts_b = self.tf_stamp['base_footprint']                      # 速度＝帶時間戳的里程位姿差分（用 TF 自己的時間戳，不用控制拍時間；時間戳沒前進就沿用上一個值）
            if self.q_prev is not None and ts_b - self.q_prev[0] > 1e-6:
                self.v_last = (q[:2] - self.q_prev[1][:2]) / (ts_b - self.q_prev[0])
            v_meas = self.v_last
            yaw_e = math.degrees(w2(q[2] - yaw_l))
            # ---- 進線路徑：影子模式（尚未取得命令權）每 0.25 s 由當下位姿重建；取得命令權後凍結，只在 map←odom 變動時以新轉換重發
            in_cap = pg is not None and CAP_ALONG[0] <= along <= CAP_ALONG[1] + 0.30 and abs(lat) <= CAP_LAT + 0.30
            if self.owner == 'NAV' and not self.stop and in_cap and (self.t_shadow is None or t - self.t_shadow >= 0.25):
                self.shadow_info = self.make_path_odom(q, pg); self.pub_path_map(T_mo); self.t_shadow = t
            elif self.owner == 'MERGE' and self.T_mo_pub is not None and (float(np.linalg.norm(T_mo[:2] - self.T_mo_pub[:2])) > 0.002 or abs(w2(T_mo[2] - self.T_mo_pub[2])) > math.radians(0.1)):
                self.pub_path_map(T_mo)
            # ---- NAV → MERGE（捕獲區、移動中）：進線 GMPC 的候選有效才切換
            if self.owner == 'NAV' and not self.stop and pg is not None and CAP_ALONG[0] <= along <= CAP_ALONG[1] and abs(lat) <= CAP_LAT and self.plan_stamp is not None:
                cand, lim = self.last['MERGE'], np.asarray(DEC) * DT
                b = q[:3].copy()
                for ub in list(tr['pipe'][:, :3]) + [cand] * NPH:        # 在途命令走完＋候選作用一拍後的位姿仍須在捕獲區內
                    c, s_ = math.cos(b[2]), math.sin(b[2]); b = b + PHYS * np.array([c * ub[0] - s_ * ub[1], s_ * ub[0] + c * ub[1], ub[2]])
                rb = b[:2] - np.array(pg['p0']); al_p, lat_p = float(-(rb @ np.array(pg['d']))), float(rb @ np.array(pg['n']))
                chk = dict(candidate_fresh=bool(self.mrg_t is not None and t - self.mrg_t <= 0.11), plan_acked=bool(self.ack == self.plan_stamp), candidate_nonzero=bool(float(np.abs(cand).max()) > 1e-4),
                           step_within_smoother_limit=bool(np.all(np.abs(cand - self.rel) <= lim + 1e-9)), moving=bool(float(np.linalg.norm(v_meas)) >= 0.05), predicted_in_capture=bool(CAP_ALONG[0] - 0.05 <= al_p <= CAP_ALONG[1] and abs(lat_p) <= CAP_LAT))
                if all(chk.values()):
                    self.owner = 'MERGE'
                    self.p_act.publish(Bool(data=True))
                    self.event('base_owner_NAV_to_MERGE', along=round(along, 3), lat=round(lat, 3), speed=round(float(np.linalg.norm(v_meas)), 3), merge_plan=self.shadow_info, released_cmd=[round(float(x), 4) for x in self.rel], merge_candidate=[round(float(x), 4) for x in cand],
                               nav_cmd=[round(float(x), 4) for x in self.last['NAV']], candidate_checks=chk, n_shadow_plans=getattr(self, 'n_pub', 0), note='影子模式下進線 GMPC 的上一命令＝實際放行命令（外包設定）；候選有效才切換')
                else:
                    self.sw_wait = getattr(self, 'sw_wait', 0) + 1
                    self.sw_last = chk
            # r2（F2）：刪除「離線 ≤0.6 m 且沿線不足即判離開捕獲區」（far_capture_2 誤拒：當時根本尚未進入）。候選未成立時保留 NAV，依下面的粗目標拒絕條件收尾。
            if self.owner == 'NAV' and pg is not None and CAP_ALONG[0] <= along <= CAP_ALONG[1] and abs(lat) <= CAP_LAT and not getattr(self, 'entered_cap', False):
                self.entered_cap = True
                self.event('entered_capture_zone', along=round(along, 3), lat=round(lat, 3))
            # ---- 到了粗目標仍未進線：協同停止，不停車等算完再進線（不冒充連續 MotM）
            if self.owner == 'NAV' and not self.stop:
                near = float(np.linalg.norm(to_map_xy(q[:2]) - self.coarse_goal)) < 0.35 and float(np.linalg.norm(v_meas)) < 0.02
                self.t_near_goal = (self.t_near_goal or t) if near else None
                if self.t_near_goal is not None and t - self.t_near_goal >= 1.0:
                    self.no_reach = ('no_valid_observation' if not self.anchored else ('plan_not_ready' if self.plan is None else 'no_valid_merge_candidate'))
                    self.global_stop('reached_coarse_goal_without_merge_global_stop', why=self.no_reach, n_obs=len(self.obs_log), n_valid=sum(1 for o in self.obs_log if o['valid']), n_assoc_rejected=self.n_assoc_rej, last_checks=getattr(self, 'sw_last', None))
            # ---- 手臂
            ua = np.zeros(6)
            if self.owner == 'MERGE' and not self.stop and self.arm == 'HOLD' and self.no_reach is None and abs(lat) < 0.03 and abs(yaw_e) < 3.0:
                need = (self.plan['terminal']['need_along_prep'] if FULL else br['alongs'][1] + br['prep_time_s'] * V_PASS + PREP_MARGIN)
                if along >= need:
                    tag_o, res_o = self.ask('off_anchor', dict(q=q.tolist(), s=tr['s'].tolist()), 0.045)
                    if tag_o != 'off_anchor':
                        self.engineering_abort('off_anchor_unavailable:' + str(tag_o), res=str(res_o)[:200])
                    else:
                        self.off_anchor = res_o
                        self.arm, self.t_prep0, self.s_prep0 = 'PREP', t, tr['s_end'].copy()
                        self.prep_goal = np.array(br['q_start_op']) - np.array(res_o['off_at_start_op'])   # 設定點目標＝作業起點關節角−該構型的靜態偏置（關節才會到作業起點）
                        self.event('arm_owner_HOLD_to_TRACK_prep', along=round(along, 3), need_along=round(need, 3), off0=[round(x, 4) for x in res_o['off0']], off_at_start_op=[round(x, 4) for x in res_o['off_at_start_op']])
                else:
                    self.no_reach = 'on_line_too_late_for_prep'
                    self.global_stop('prep_distance_insufficient_global_stop', why=self.no_reach, along=round(along, 3), need_along=round(need, 3))
            res, ua_cand = {}, np.zeros(6)
            if self.arm == 'PREP' and not self.stop:
                qs = self.prep_goal; dq = qs - self.s_prep0
                Tp = max(float(np.abs(dq).max()) / 0.5, DT); u_ = min((t + DT - self.t_prep0) / Tp, 1.0)
                s_des = self.s_prep0 + (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5) * dq
                ua_cand = np.clip((s_des - tr['s_end']) / DT, -1.0, 1.0)
            along_pred = None
            if pg is not None:                                        # 在途命令走完＋本拍放行命令一拍後的沿線位置
                b_ = q[:3].copy()
                for ub in list(tr['pipe'][:, :3]) + [self.rel] * NPH:
                    c_, s__ = math.cos(b_[2]), math.sin(b_[2]); b_ = b_ + PHYS * np.array([c_ * ub[0] - s__ * ub[1], s__ * ub[0] + c_ * ub[1], ub[2]])
                along_pred = float(-((b_[:2] - np.array(pg['p0'])) @ np.array(pg['d'])))
            if self.arm in ('PREP', 'TRACK', 'OBSERVE', 'DESCEND', 'PRETRACK') and not self.stop:
                judge = (self.arm == 'TRACK' and -0.05 <= along <= 0.25) or (self.arm == 'PRETRACK')
                tag, res = self.ask('tick', dict(want_lock=bool(self.arm == 'OBSERVE' and along_pred is not None and along_pred <= TERM['a_lock']), mode=self.arm, t=t, q=q.tolist(), pipe=tr['pipe'].tolist(), rel=self.rel.tolist(), s=tr['s'].tolist(), s_end=tr['s_end'].tolist(), judge=judge, obs=(None if not ((UPDATE or FULL) and self.obs_latest is not None) else dict(T=self.obs_latest['T_odom_handle'].tolist(), t_obs=self.obs_latest['t_obs'], seq=self.obs_latest['seq'])), allow_update=bool(UPDATE and self.arm == 'TRACK' and along is not None and STOP_ALONG <= along <= 0.25),
                                                  judge_refs=([('fixed_reference', self.anchor['T_odom_handle'], self.anchor['t_obs']), ('latest_observation', self.obs_latest['T_odom_handle'].tolist(), self.obs_latest['t_obs'])] + ([('updated_reference', None, self.ref_t_obs)] if UPDATE else []) if OBS_MODE else [('real_image_at_start', None, self.t_goal), ('synthetic_fresh_stamp', None, t - 0.05)]), v_meas=v_meas.tolist(), ua_cand=ua_cand.tolist(), T_mo=T_mo.tolist(),
                                                  u_last=np.r_[self.rel, self.u_arm].tolist(), u_prev=np.r_[self.rel, self.u_arm_prev].tolist()), 0.045)
                sc0 = res.get('stop') if isinstance(res, dict) else None
                if tag == 'tick' and self.arm in ('TRACK', 'PRETRACK') and (res.get('track') is None or not res['track']['ok']):
                    # 追蹤器無解：本拍沒有候選、也沒有本拍的停止核對。現在送零＝上一拍已核過的 next 停法（上一拍提交候選、這一拍起送零）
                    self.global_stop('tracker_fail_global_stop', why=None if res.get('track') is None else res['track']['why'], stop_certificate='上一拍的 next 停法核對')
                    disp = 'stopped'
                else:
                    disp = stop_disposition(tag, sc0)
                if disp == 'abort':                                   # now 不過／結果缺失／遲到／錯誤：沒有核過的停法 ⇒ 模擬工程中止
                    self.engineering_abort('stop_check_' + ('now_failed' if (tag == 'tick' and isinstance(sc0, dict) and not sc0.get('why')) else ('unavailable:' + str(tag) + ':' + str((sc0 or {}).get('why')))), arm=self.arm, check=None if not isinstance(sc0, dict) else {k_: sc0.get(k_) for k_ in ('now', 'next', 'why', 'ms')}, res=None if tag == 'tick' else str(res)[:200])
                    res = res if isinstance(res, dict) else {}
                elif disp == 'stop_now':
                    self.global_stop('stop_check_next_failed_stop_now_global_stop', arm=self.arm, next=sc0['next'], stop_certificate='本拍 now 停法核對通過')
                if not isinstance(res, dict):
                    res = {}
            rf = res.get('ref') if isinstance(res, dict) else None
            if rf is not None:                                        # 連續視覺更新的結果：只有 worker 已提交（全部核過）的參考才生效
                if rf.get('status') not in ('off', 'no_new_obs') or rf.get('committed') is not None or rf.get('goal_expired'):
                    self.ref_log.append(dict(t=round(t, 3), along=None if along is None else round(along, 4), **{k_: rf.get(k_) for k_ in ('ver', 'status', 'obs_seq', 'obs_age', 'obs_dpos_mm', 'committed', 'step_mm', 'step_rejected', 'from_seqs', 'goal_dist_mm', 'ref_t_obs', 'goal_expired', 'n_goal_expired')}))
                if rf.get('committed'):
                    dp = np.array(rf['pg_p0']) - np.array(self.plan['pg']['p0'])
                    self.plan['pg']['p0'] = rf['pg_p0']               # 通過線跟著參考平移（方向與朝向不變）
                    self.path_odom = self.path_odom + dp
                    self.path_shift = self.path_shift + dp
                    self.ref_ver, self.ref_t_obs = rf['ver'], rf['ref_t_obs']
                    if float(np.linalg.norm(self.path_shift)) > 0.001:   # 累積平移超過 1 mm 才重發路徑（位置、朝向、轉換版本一起生效）
                        self.pub_path_map(T_mo); self.path_shift = np.zeros(2)
            sc = res.get('stop')
            if self.arm == 'PREP' and not self.stop:
                dcs = res.get('decision')
                if dcs is not None and dcs[0] == 'refuse':
                    self.global_stop('refuse_enter_reach_zone_global_stop', stop_certificate='本拍 now 停法核對通過', **dcs[1])
                elif FULL:
                    ua = ua_cand
                    if u_ >= 1.0 and float(np.abs(q[3:] - np.array(br['q_start_op'])).max()) <= 0.02:    # 準備完成：手臂靜止在作業起點解，開始觀察
                        self.arm = 'OBSERVE'; self.event('arm_prep_done_observe', along=round(along, 4), arm_err=round(float(np.abs(q[3:] - np.array(br['q_start_op'])).max()), 4))
                    elif along_pred <= TERM['a_lock']:
                        self.global_stop('prep_not_done_before_lock_point_global_stop', along=round(along, 4), u=round(u_, 3), stop_certificate='本拍 now 停法核對通過')
                        ua = np.zeros(6)
                else:
                    ua = ua_cand                                       # 提交：now 與 next 停法核對都通過
                    if dcs is not None and dcs[0] == 'start':
                        self.arm = 'TRACK'; self.event('arm_track_start', **dcs[1])
            elif self.arm == 'OBSERVE' and not self.stop:
                ua = np.zeros(6)
                lk = res.get('lock')
                tinf = res.get('term') or {}
                if tinf.get('obs') is not None:
                    self.term_log.append(dict(t=round(t, 3), along=round(along, 4), **tinf['obs']))
                if lk is not None:
                    if lk['ok']:                                       # 鎖定已由 worker 試算並提交：通過線隨參考平移、重發路徑；下降開始
                        dp = np.array(lk['pg_p0']) - np.array(self.plan['pg']['p0'])
                        self.plan['pg']['p0'] = lk['pg_p0']
                        self.path_odom = self.path_odom + dp
                        self.pub_path_map(T_mo)
                        self.ref_ver, self.ref_t_obs, self.lock = lk['ver'], lk['t_obs'], lk
                        self.arm = 'DESCEND'
                        self.event('reference_locked_descend_start', **{k_: lk[k_] for k_ in ('ver', 't_obs', 'age_at_lock_s', 'seqs', 'spread_mm', 'shift_mm', 'along_at_lock', 'descent_margin')})
                    else:
                        self.no_reach = 'lock_failed:' + str(lk.get('why'))
                        self.global_stop('lock_failed_global_stop', why=lk.get('why'), detail=lk.get('check'), n_obs_in_observe=len(self.term_log), stop_certificate='本拍 now 停法核對通過')
            elif self.arm == 'DESCEND' and not self.stop:
                dsc = res.get('desc')
                ua = np.array(dsc['ua'])
                if dsc['done']:
                    self.arm = 'PRETRACK'; self.event('descent_done_pretrack', along=round(along, 4), q_err_to_pre=dsc['q_err_to_pre'])
            elif self.arm == 'PRETRACK' and not self.stop:
                tk = res.get('track')
                ua = np.array(tk['u'])
                bd = res.get('budget') or {}
                if res.get('handover_ok') and self.t_handover_ok is None and along >= HO_LATEST:
                    self.t_handover_ok = t
                    if HANDOVER:                                   # 交出命令權：鎖定參考（位姿、原始時間戳、鎖定時年齡、一致性）與通過線；設定點／真管線／上一命令在執行器手上原樣承接
                        lk_ = self.lock
                        msg = [t, *np.array(lk_['T'], float).reshape(-1), lk_['t_obs'], lk_['age_at_lock_s'], lk_['dang'], lk_['spread'], *self.plan['pg']['p0'], *self.plan['pg']['d'], self.plan['pg']['yaw'], float(lk_['ver'])]
                        self.p_ho.publish(Float64MultiArray(data=[float(x) for x in msg]))
                        self.ho_pending = int(lk_['ver'])
                        self.event('handover_requested', along=round(along, 4), budget=bd, ver=self.ho_pending, note='命令權仍在 MERGE、本節點續追預抓位；執行器以當下快照試跑首拍通過才接手（第一筆 /cmd_vel_grasp 到選擇器即切換）；nack ⇒ 保留前段所有權並停止')
                    else:
                        self.event('handover_request_accepted_first', along=round(along, 4), budget=bd, note='本程式只做判定與記錄；抓持控制器未接')
                if self.ho_pending is not None and self.ho_ack is not None and self.ho_ack != 'ack:%d' % self.ho_pending and not self.stop:   # nack 或版本不符
                    self.global_stop('handover_rejected_by_executor_global_stop', ack=self.ho_ack, stop_certificate='本拍 now 停法核對通過')
                    ua = np.zeros(6)
                elif along <= TERM_END:                               # 預抓位追蹤的已核範圍終點：協同停止
                    self.global_stop('pretrack_end_global_stop', along=round(along, 4), n_judgements=len(self.sel_log), handover_ok_first_t=self.t_handover_ok, stop_certificate='本拍 now 停法核對通過')
                    ua = np.zeros(6)
            elif self.arm == 'TRACK' and not self.stop:
                tk = res.get('track')
                ua = np.array(tk['u'])
                if along <= STOP_ALONG:                               # 通過把手正前方 0.05 m（設計值，同 door_far_route.STOP_ALONG；在已核範圍 −0.10 之內）：沒有被接受的交接 ⇒ 正常協同停止
                    self.global_stop('pass_end_no_handover_global_stop', along=round(along, 4), n_judgements=len(self.sel_log), stop_certificate='本拍 now 停法核對通過')
                    ua = np.zeros(6)
            if self.stop:
                ua = np.zeros(6); self.p_nav.publish(Twist())
            self.u_arm_prev, self.u_arm = self.u_arm, ua
            self.p_arm.publish(Float64MultiArray(data=[float(x) for x in ua]))
            if self.q_prev is None or ts_b - self.q_prev[0] > 1e-6:
                self.q_prev = (ts_b, q.copy())
            tk = res.get('track') or {}
            cl = res.get('clear') or {}
            scm = (sc or {}).get('min_clear') or {}
            self.rows.append([t, tr['t'], *q, *tr['s'], *self.rel, *ua, *self.last['NAV'], *self.last['MERGE'], dict(NAV=0, MERGE=1, GRASP=2)[self.owner], dict(HOLD=0, PREP=1, TRACK=2, OBSERVE=3, DESCEND=4, PRETRACK=5)[self.arm], float(self.stop),
                              np.nan if along is None else along, np.nan if lat is None else lat, yaw_e, tk.get('err') if tk.get('err') is not None else np.nan, tk.get('sigma') if tk.get('sigma') is not None else np.nan, float(bool(tk.get('clipped'))),
                              cl.get('env', np.nan), cl.get('base', np.nan), cl.get('self', np.nan), *v_meas, *tr['q'][:3], *T_mo, np.nan if sc is None else float(sc['ok']), scm.get('env', np.nan), scm.get('base', np.nan), scm.get('self', np.nan), np.nan if sc is None else sc['ms'], tk.get('err_pred') if tk.get('err_pred') is not None else np.nan, ts_b,
                              np.nan if sc is None else (sc['now']['all_still_s'] or np.nan), np.nan if sc is None else (sc['next']['all_still_s'] or np.nan), np.nan if sc is None else sc.get('n_eval', np.nan),
                              *[np.nan if sc is None else sc['margin_after_interval_bound'][k_] for k_ in ('env', 'base', 'self')], np.nan if sc is None else sc.get('n_refine', np.nan),
                              float(self.ref_ver), *([np.nan] * 3 if rf is None else rf['ref_p']), np.nan if rf is None or rf.get('goal_dist_mm') is None else rf['goal_dist_mm']])
            if isinstance(res, dict) and res.get('budget') is not None:
                self.budget_log.append(dict(t=round(t, 3), along=round(along, 4), arm=self.arm, **res['budget']))
            if res.get('judge') is not None:
                self.sel_log.append(dict(t=round(t, 2), along=round(along, 4), lat_mm=round(lat * 1e3, 2), judge=res['judge'], updated_ref=res.get('judge_updated_ref')))
            # ---- 結束條件：進線後底盤靜止 2.5 s
            moving = float(np.linalg.norm(v_meas)) > 1e-3 or float(np.abs(self.rel).max()) > 1e-4
            if (self.owner == 'MERGE' or self.stop) and not moving:
                self.t_stopcheck = self.t_stopcheck or t
                if t - self.t_stopcheck > 2.5:
                    for _ in range(5):
                        self.p_done.publish(Bool(data=True))
                    self.event('finished_base_stopped', along=None if along is None else round(along, 4), lat_mm=None if lat is None else round(lat * 1e3, 2))
                    raise SystemExit(0)
            else:
                self.t_stopcheck = None

        def save(self):
            cols = ['t', 't_world'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['rel_x', 'rel_y', 'rel_w'] + [f'ua{i}' for i in range(6)] + ['nav_x', 'nav_y', 'nav_w', 'mrg_x', 'mrg_y', 'mrg_w', 'base_owner(0NAV,1MERGE)', 'arm(0HOLD,1PREP,2TRACK)', 'stop',
                                                                                                           'along', 'lat', 'yaw_err_deg', 'arm_track_err', 'sigma', 'clipped', 'clear_env', 'clear_base', 'clear_self', 'vx_meas', 'vy_meas', 'truth_x', 'truth_y', 'truth_yaw', 'mo_x', 'mo_y', 'mo_yaw', 'stop_ok', 'stop_env', 'stop_base', 'stop_self', 'stop_ms', 'arm_track_err_pred', 'tf_stamp_base', 'stop_now_all_still_s', 'stop_next_all_still_s', 'stop_n_samples', 'stop_margin_env', 'stop_margin_base', 'stop_margin_self', 'stop_n_refine', 'ref_ver', 'ref_px', 'ref_py', 'ref_pz', 'ref_goal_dist_mm']
            np.savez_compressed(out + '_merge.npz', R=np.array(self.rows, float), cols=np.array(cols), TM=np.array(self.tm_log, float).reshape(-1, 8), tm_cols=np.array(['t', 'is_tick', 'dispatch_ms', 'queue_ms', 'compute_ms', 'return_ms', 'total_ms', 'late']))
            json.dump(dict(events=self.ev, plan=self.plan, judge=self.sel_log, forwarded=self.n_fwd, estimate=W0['est'], coarse=co, terminal=dict(params=TERM, end_along=TERM_END, lock=self.lock, observe_log=self.term_log, budget_log=self.budget_log, t_handover_ok_first=self.t_handover_ok) if FULL else None, anchor=self.anchor, off_anchor=getattr(self, 'off_anchor', None), sched=W0.get('sched'), obs_mode=OBS_MODE, update_mode=UPDATE, update_params=UPD, ref_log=self.ref_log, ref_ver_final=self.ref_ver, prior=getattr(self, 'prior', None), n_assoc_rejected=self.n_assoc_rej, obs_log=self.obs_log, stale_replies_dropped=self.stale, switch_wait_ticks=getattr(self, 'sw_wait', 0), n_path_publishes=getattr(self, 'n_pub', 0), pose_source='TF odom→base_footprint（控制用）；/far/truth 底盤位姿只記錄', no_reach=self.no_reach, engineering_abort=self.aborted, final_owner=dict(base=self.owner, arm=self.arm, stop=self.stop)), open(out + '_merge.json', 'w'), ensure_ascii=False, indent=1, default=str)

    node = M()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.save()
        if node.gm is not None:
            try:
                os.killpg(node.gm.pid, signal.SIGINT)
            except Exception:                                         # noqa: BLE001
                pass
        try:
            pc.send(None)
        except Exception:                                             # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
