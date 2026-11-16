#!/usr/bin/env python3
"""遠距整合的離線接線骨架（Codex 20261115：遠距第一包）：原導航 → 移動中影像定位把手 → 產生抓持通過線 → 交給 MovingGraspCtrl。
這是可執行的入口，不是成果：每一步資料不足就明確拒絕並給原因；不以真值替代感知、不沿用停車拍照流程、不讀 full_2 的給定通過線。

1. 導航基準（nav_baseline）：只認已核定的原全開導航鏈——規劃器 → gmpc_controller（heading 開）→ /cmd_vel_nav → velocity_smoother → /cmd_vel_pre_shield
   → scan_safety_shield → /cmd_vel_smoothed → wheel_limit_guard → /cmd_vel。參數一律取該基準趟的 param dump（附雜湊），本檔不內建另一套數值。
2. 命令權（Authority）：底盤命令與手臂設定點在任一時刻只有一個來源。NAV（導航鏈獨佔 /cmd_vel；手臂由預抓保持）→ GRASP（MovingGraspCtrl 獨佔兩者）→ OPEN。
   NAV→GRASP 的切換條件在 handover_request 內逐項核對；不成立就留在 NAV（或由導航自己的停止流程處理），不雙發布。
3. 把手觀測（HandleObs）：帶時間戳的深度影像＋內參＋「同一渲染時刻」的相機位姿（關節／底盤讀值 FK）→ 凍結偵測 d1_handle_detect_v2.detect → 分類 → 幾何估計。
   缺時間戳、缺同刻位姿、偵測無有效中心、估計失敗 ⇒ 拒收並記原因。不讀真值。
4. 通過線（pass_line）：由當下把手估計產生——門面法向取自估計的接近軸、通過方向取自把手桿軸、底盤到把手的側向距離與朝向取抓持幾何（預抓構型下 TCP 相對底盤的位置）。
5. 交接包（handover_request）：把設定點、在途命令管線、上一筆命令、估計與其年齡、座標系交給抓持控制器的建構參數；任何一項不合格 ⇒ 拒絕＋原因。"""
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_BASE_RUN = 'isaac_bigarena_gmpc_scan_heading__seed8__fullon_s8_1'
NAV_NODES = ('gmpc_controller', 'scan_safety_shield', 'wheel_limit_guard', 'scan_obstacle_tracker', 'velocity_smoother', 'planner_server')
NAV_CHAIN = ['planner_server', 'gmpc_controller → /cmd_vel_nav', 'velocity_smoother → /cmd_vel_pre_shield', 'scan_safety_shield → /cmd_vel_smoothed', 'wheel_limit_guard → /cmd_vel（唯一最終發布者）']
POSE_SOURCES = ('fk_at_render_time',)                               # 相機位姿的允許來源：該渲染時刻的關節／底盤讀值 FK（同一步的歷史紀錄）；之後接 ROS 時加「TF 於影像時間戳內插」
IMG_STATE_TOL_S = 0.011                                              # 影像時間戳與狀態時間戳的容許差（一個物理步）
EST_MAX_AGE_S = 0.3                                                  # 同抓持控制器 READY_Q['fresh_s']
SYNC_TOL_S = 0.02                                                    # 影像以外的狀態／命令時間戳對交接時刻的容許差（兩個物理步）
GRASP_WINDOW_M = (0.005, 0.25)                                       # 交接時距抓取點的沿線距離窗口（抓持控制器的前導＋插入需要的行程；上限＝目前排程可用的最長前導）
LOCK_BUDGET = dict(pos_m=0.005, ang_deg=3.0)                         # 鎖定任務參考在遮擋期間的總誤差預算上限（Codex 20261116）
ARM_TRACK_TOL = 0.01                                                 # rad：交接時手臂構型對「當下底盤位置下的目標 IK 解」的最大容許差
TICK_STEPS = 5                                                       # 每控制拍的物理步數（同控制器 NPH）
LINE_TOL = dict(lateral_m=0.010, yaw_rad=math.radians(1.0), speed_frac=0.2)   # 機器人須已在通過線上（側向、朝向）且沿線速度在 v_pass 的 ±20% 內


def _flat(d, pre=''):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + str(k) + '.'))
        else:
            out[pre + str(k)] = v
    return out


def nav_baseline():
    """讀已核定原導航基準趟的現場紀錄（方法設定、heading 讀回、各話題端點、/cmd_vel 唯一發布者）。回 dict(ok, why, …)。
    該趟側錄的各節點 param dump 是空檔（當時 dump 失敗），所以參數值的來源是起跑腳本 run_bigarena_isaac.sh＋lib/run_guards.sh 的 method_config()（附雜湊）；
    本檔不內建另一套數值。任何一項讀不到或不符 ⇒ ok=False。"""
    D = os.path.join(HERE, 'runs', NAV_BASE_RUN)
    out = dict(run=NAV_BASE_RUN, chain=NAV_CHAIN, ok=True, why=[])
    try:
        mm = json.load(open(os.path.join(D, 'method_manifest.json')))
        ep = json.load(open(os.path.join(D, 'endpoints.json')))
        rb = open(os.path.join(D, 'param_readback.log')).read()
    except Exception as e:                                            # noqa: BLE001
        return dict(out, ok=False, why=['baseline_record_missing:' + repr(e)[:80]])
    tp = ep.get('topics', {})
    pub = lambda t: tp.get(t, {}).get('publishers', [])               # noqa: E731
    chk = dict(method=mm.get('method'), heading=mm.get('heading'), heading_readback_true='heading_enable = True' in rb,
               cmd_vel_sole_publisher=pub('/cmd_vel'), cmd_vel_nav_publisher=pub('/cmd_vel_nav'), cmd_vel_smoothed_publisher=pub('/cmd_vel_smoothed'),
               shield_subscribes_scan='/scan_safety_shield' in tp.get('/scan', {}).get('subscribers', []), tracker_subscribes_scan='/scan_obstacle_tracker' in tp.get('/scan', {}).get('subscribers', []))
    out['record'] = chk
    if chk['method'] != 'gmpc_scan_heading' or chk['heading'] != 1 or not chk['heading_readback_true']:
        out['why'].append('heading_not_on')
    if chk['cmd_vel_sole_publisher'] != ['/wheel_limit_guard']:
        out['why'].append('guard_not_sole_cmd_vel_publisher')
    if chk['cmd_vel_smoothed_publisher'] != ['/scan_safety_shield'] or not chk['shield_subscribes_scan']:
        out['why'].append('shield_not_in_chain')
    if chk['cmd_vel_nav_publisher'] != ['/gmpc_controller']:
        out['why'].append('gmpc_not_nav_source')
    src = {}
    for rel in ('run_bigarena_isaac.sh', os.path.join('lib', 'run_guards.sh')):
        pth = os.path.join(HERE, rel)
        src[rel] = hashlib.sha256(open(pth, 'rb').read()).hexdigest()[:16] if os.path.exists(pth) else None
        if src[rel] is None:
            out['why'].append('config_source_missing:' + rel)
    out['config_source'] = src
    out['param_dump_note'] = '基準趟 side/params_*.yaml 為空檔（Node not found）；逐參數值未存檔——整合前須在基準配置下重新 dump（缺口）'
    out['ok'] = not out['why']
    return out


class Authority:
    """單一命令來源。owner ∈ NAV／GRASP／OPEN／STOP；switch() 只在條件成立時改 owner，並記錄；publish(src, …) 來源不是 owner ⇒ 拒絕（計數）。"""
    ORDER = ('NAV', 'GRASP', 'OPEN', 'STOP')

    def __init__(self):
        self.owner, self.log, self.n_rejected = 'NAV', [], 0

    def publish(self, src, t):
        if src != self.owner:
            self.n_rejected += 1
            self.log.append(dict(t=round(float(t), 3), what='publish_rejected', src=src, owner=self.owner))
            return False
        return True

    def switch(self, to, t, ok, why=None):
        if not ok or to not in self.ORDER:
            self.log.append(dict(t=round(float(t), 3), what='switch_refused', to=to, why=why))
            return False
        self.log.append(dict(t=round(float(t), 3), what='switch', frm=self.owner, to=to))
        self.owner = to
        return True


class HandleObs:
    """實際影像 → 把手位姿估計（世界座標）。frame＝dict(dep, K, T_cam, t_obs, q, fk_vs_render_m?)。"""

    def __init__(self, seed=20261115):
        import d1_handle_detect_v2 as D1
        import m8_door_detect as MD
        import door_motm_grasp_ctrl as G
        self.D1, self.MD, self.E, self.PG = D1, MD, G.E, G.PG
        self.Rg, self.T_HG, self.a_H = G.PG.baseline_reference()
        self.seed, self.n = seed, 0
        self.track = []                                              # (t_obs, T_WH)

    def intake(self, f, t_now):
        need = ('dep', 'K', 'T_cam', 't_obs')
        miss = [k for k in need if f.get(k) is None]
        if miss:
            return None, dict(ok=False, why='frame_incomplete:' + ','.join(miss))
        if f.get('q') is None:
            return None, dict(ok=False, why='no_synchronized_robot_state')   # 相機位姿必須來自同一渲染時刻的讀值
        # r3：影像—狀態—TF 的時間與來源核對（「有 q」不等於同步）
        if f.get('t_state') is None or f.get('pose_source') is None:
            return None, dict(ok=False, why='state_stamp_or_pose_source_missing')
        if f['pose_source'] not in POSE_SOURCES:
            return None, dict(ok=False, why='pose_source_not_allowed:' + str(f['pose_source']))
        if abs(float(f['t_state']) - float(f['t_obs'])) > IMG_STATE_TOL_S:
            return None, dict(ok=False, why='image_state_time_mismatch:%.4f' % abs(float(f['t_state']) - float(f['t_obs'])))
        if f.get('frame_id') not in (None, 'map'):
            return None, dict(ok=False, why='camera_pose_frame_not_map:' + str(f.get('frame_id')))
        if not np.isfinite(f['T_cam']).all() or not math.isfinite(float(f['t_obs'])):
            return None, dict(ok=False, why='nonfinite_pose_or_time')
        if f['t_obs'] > t_now + 1e-9:
            return None, dict(ok=False, why='timestamp_in_future')
        self.n += 1
        det, ms = self.D1.detect(f['dep'], np.asarray(f['K'], float), np.asarray(f['T_cam'], float), np.random.default_rng(self.seed + self.n))
        cls = self.MD.classify(det)
        info = dict(t_obs=round(float(f['t_obs']), 4), t_query=round(float(t_now), 4), age_s=round(float(t_now - f['t_obs']), 4), cls=cls, reject=det.get('reject'), detect_ms=round(float(ms), 1))
        if cls != 'center_valid':
            return None, dict(info, ok=False, why='detect:' + str(cls))
        est = self.E.estimate(dict(path='far_link', L1=det['L1'], L2=det['L2'], reject=None, reject_L2=None, depth=f['dep'], K=np.asarray(f['K'], float), T_cam=np.asarray(f['T_cam'], float), t_obs=f['t_obs']),
                              'N-obs', self.Rg, self.T_HG, self.a_H, query_t=t_now)
        if not est['ok']:
            return None, dict(info, ok=False, why='estimate:' + str(est.get('why')))
        T = np.asarray(est['T_WH'], float)
        self.track.append((float(f['t_obs']), T))
        return T, dict(info, ok=True, why=None)

    def est_fn(self):
        """MovingGraspCtrl 的 est_fn(t) 介面：回 (T_WH, t_obs, 版本) 或 None。"""
        tr = self.track

        def fn(t):
            for tc, T in reversed(tr):
                if tc <= t + 1e-9:
                    return T, tc, tc
            return None
        return fn


def pass_line(T_WH, T_HG, a_H, K, q_pregrasp_arm, T_at, s_pre):
    """由把手估計產生抓持通過線。回 dict(p0, d, yaw, standoff, n)。
    幾何：通過方向 d＝−把手桿軸（把手座標 x 軸）在水平面的投影；門面外法向 n＝接近軸在水平面的投影（指向機器人側）；
    朝向＝讓預抓構型的 TCP 對準抓取位姿所需的底盤偏航；側向位置＝抓取位姿 TCP 減去「預抓構型 TCP 相對底盤的偏移」。全部由估計與機器人運動學算出。"""
    T_WH = np.asarray(T_WH, float)
    ax = T_WH[:3, 0].copy(); ax[2] = 0.0
    if np.linalg.norm(ax) < 1e-6:
        return None, 'bar_axis_vertical'
    d = -ax / np.linalg.norm(ax)
    n = (T_WH[:3, :3] @ np.asarray(a_H, float)).copy(); n[2] = 0.0
    if np.linalg.norm(n) < 1e-6:
        return None, 'approach_axis_vertical'
    n = n / np.linalg.norm(n)
    Tg = np.asarray(T_at(T_WH, s_pre), float)                        # 通過線上手臂保持預抓構型：TCP 在抓取位姿沿接近軸後退 s_pre 處（插入由抓持控制器在移動中完成）
    T0 = K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(q_pregrasp_arm, float)], 'link_tcp')   # 底盤在原點、朝 +x 時預抓構型的 TCP
    # 底盤偏航：使 TCP 的水平朝向與抓取位姿一致
    yaw = math.atan2(Tg[1, 0], Tg[0, 0]) - math.atan2(T0[1, 0], T0[0, 0])
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    off = np.array([c * T0[0, 3] - s * T0[1, 3], s * T0[0, 3] + c * T0[1, 3]])
    p_grasp = Tg[:2, 3] - off                                        # 抓取瞬間底盤應在的位置
    return dict(p0=p_grasp, d=d[:2], n=n[:2], yaw=yaw, standoff=float((p_grasp - T_WH[:2, 3]) @ n[:2]), tcp_z_err=float(Tg[2, 3] - T0[2, 3])), None


def _fin(x):
    try:
        return x is not None and bool(np.isfinite(np.asarray(x, float)).all())
    except Exception:                                                 # noqa: BLE001
        return False


def handover_request(t, auth, line, q, v_meas_world, s_arm, pipe, u_last, u_prev, est, v_pass, d_steps, off_static=None, tick_phase=None, stamps=None, arm_track_err=None, locked=None):
    """NAV→GRASP 的交接核對（r2；r3 加 locked）。
    locked（Codex 20261116「鎖定任務參考」入口；不偽造影像新鮮度）＝dict(age_at_lock_s, pos_budget_m, ang_budget_deg, terms)：給了就不查影像年齡，改查
      鎖定當下觀察年齡 ≤ EST_MAX_AGE_S、遮擋期間位置總預算 ≤ LOCK_BUDGET['pos_m']、姿態 ≤ LOCK_BUDGET['ang_deg']；其餘條件完全相同。全過 ⇒ 回抓持控制器建構所需的交接包；否則回拒絕原因列表。不改任何狀態。
    est＝dict(T_WH, t_obs)：年齡以交接當下 t−t_obs 重算（不沿用先前算好的年齡）。
    v_meas_world＝實測底盤速度（世界座標 vx, vy）——移動條件用實測，不用命令。
    pipe＝實際送往受控對象的最近 d_steps 筆 9 維命令；任何一筆缺失（None／非有限／維度不符）⇒ 拒絕，不補零。
    stamps＝dict(image, state, cmd)：三者的時間戳（同一時鐘）；缺或彼此相差超過 SYNC_TOL_S ⇒ 拒絕。"""
    why = []
    if auth.owner != 'NAV':
        why.append('authority_not_nav:' + auth.owner)
    if not _fin(t):
        why.append('time_nonfinite')
    if est is None or est.get('T_WH') is None or not _fin(est.get('T_WH')) or not _fin(est.get('t_obs')):
        why.append('no_valid_handle_estimate')
        age = None
    else:
        age = float(t) - float(est['t_obs'])
        if age < -1e-9:
            why.append('estimate_from_future')
        elif locked is not None:
            if not all(_fin(locked.get(k)) for k in ('age_at_lock_s', 'pos_budget_m', 'ang_budget_deg')):
                why.append('locked_reference_fields_missing')
            else:
                if float(locked['age_at_lock_s']) > EST_MAX_AGE_S or float(locked['age_at_lock_s']) < -1e-9:
                    why.append('lock_observation_not_fresh:%.3f' % float(locked['age_at_lock_s']))
                if float(locked['pos_budget_m']) > LOCK_BUDGET['pos_m']:
                    why.append('locked_position_budget_exceeded:%.4f' % float(locked['pos_budget_m']))
                if float(locked['ang_budget_deg']) > LOCK_BUDGET['ang_deg']:
                    why.append('locked_orientation_budget_exceeded:%.2f' % float(locked['ang_budget_deg']))
        elif age > EST_MAX_AGE_S:
            why.append('estimate_stale:%.3f' % age)
    if stamps is None or any(not _fin(stamps.get(k)) for k in ('state', 'cmd')):
        why.append('timestamps_missing')
    elif max(abs(float(stamps['state']) - float(t)), abs(float(stamps['cmd']) - float(t))) > SYNC_TOL_S:
        why.append('state_or_cmd_not_synchronized')
    if not _fin(q) or len(q) != 9:
        why.append('robot_state_missing')
    if not _fin(v_meas_world) or len(v_meas_world) != 2:
        why.append('measured_velocity_missing')
    if line is None or not all(_fin(line.get(k)) for k in ('p0', 'd', 'n', 'yaw')):
        why.append('no_pass_line')
    elif _fin(q) and len(q) == 9:
        r = np.asarray(q[:2], float) - line['p0']
        lat = float(r @ line['n'])
        yaw_e = (float(q[2]) - line['yaw'] + math.pi) % (2 * math.pi) - math.pi
        if abs(lat) > LINE_TOL['lateral_m']:
            why.append('off_pass_line_lateral:%.4f' % lat)
        if abs(yaw_e) > LINE_TOL['yaw_rad']:
            why.append('yaw_error:%.4f' % yaw_e)
        along = float(-(r @ line['d']))                              # 距抓取點還有多遠（沿通過方向為正）
        if along <= 0.0:
            why.append('already_past_grasp_point')
        elif not (GRASP_WINDOW_M[0] <= along <= GRASP_WINDOW_M[1]):
            why.append('outside_grasp_window:%.4f' % along)
        if _fin(v_meas_world) and len(v_meas_world) == 2:
            vw = np.asarray(v_meas_world, float)
            v_al, v_lat = float(vw @ line['d']), float(vw @ line['n'])
            if abs(v_al - v_pass) > LINE_TOL['speed_frac'] * v_pass or abs(v_lat) > LINE_TOL['speed_frac'] * v_pass:
                why.append('not_moving_along_line:v_along=%.4f,v_lat=%.4f' % (v_al, v_lat))
    if pipe is None or len(pipe) != d_steps:
        why.append('pipeline_length')
    elif any((u is None) or (not _fin(u)) or len(u) != 9 for u in pipe):
        why.append('pipeline_has_missing_command')
    if not _fin(s_arm) or len(s_arm) != 6:
        why.append('setpoint_missing')
    if not _fin(u_last) or len(u_last) != 9:
        why.append('last_command_missing')
    if not _fin(u_prev) or len(u_prev) != 9:
        why.append('previous_command_missing')
    if not _fin(off_static) or len(off_static) != 6:
        why.append('static_offset_source_missing')
    if tick_phase is None or not (0 <= int(tick_phase) < TICK_STEPS):
        why.append('tick_phase_missing')
    # 手臂在交接當下必須已經在追「把手估計後退 s_pre」的世界目標（抓持控制器接手後第一拍就以此為目標；差太多會要求瞬移 ⇒ 第一拍命令超限）
    if arm_track_err is None or not _fin(arm_track_err):
        why.append('arm_tracking_state_missing')
    elif float(arm_track_err) > ARM_TRACK_TOL:
        why.append('arm_not_tracking_target:%.4f' % float(arm_track_err))
    if why:
        return None, why
    return dict(t=float(t), frames=dict(world='map（門鉸鏈座標；同門任務）', base='base_link'), d_world=np.asarray(line['d'], float).copy(), yaw=float(line['yaw']), p0=np.asarray(line['p0'], float).copy(), v_pass=float(v_pass),
                s=np.asarray(s_arm, float).copy(), pipe=[np.asarray(u, float).copy() for u in pipe], u_last=np.asarray(u_last, float).copy(), u_prev=np.asarray(u_prev, float).copy(), off_static=np.asarray(off_static, float).copy(),
                tick_phase=int(tick_phase), est=dict(T_WH=np.asarray(est['T_WH'], float).copy(), t_obs=float(est['t_obs']), age_s=age, locked=None if locked is None else dict(locked)),
                base_cmd_route='抓持控制器的底盤命令仍送入 /cmd_vel_pre_shield → Shield → GUARD → /cmd_vel（不旁路安全鏈）'), []


def pregrasp_from_estimate(T_WH, T_at, s_pre, K, clear_fn=None, thr=None, sig_floor=None, lo=None, hi=None, collect=None, standoffs=None, alongs=None):
    """由把手估計產生預抓構型與底盤通過線位置（不讀真值、不讀既有趟的預抓關節角）：
    候選：底盤在門面外法向上、朝向正對門（−n）、側向距離 standoff ∈ [0.36, 0.56] m（每 1 cm）；每個候選以通用種子格點多起點解 IK 到「抓取位姿後退 s_pre」，
    再由該解連續解到抓取位姿（s＝0）；兩個構型都須：殘差 <1e-6、在關節限位內、可操作度 ≥ sig_floor、淨空 ≥ thr（若給 clear_fn）。
    選擇：兩構型可操作度較小者最大的那一個（同分取 standoff 較小）。回 (dict 或 None, 原因／統計)。"""
    import itertools
    import door_fast_kin as DFK
    T_WH = np.asarray(T_WH, float)
    n = (T_WH[:3, :3] @ np.asarray(_A_H, float)).copy(); n[2] = 0.0
    ax = T_WH[:3, 0].copy(); ax[2] = 0.0
    if np.linalg.norm(n) < 1e-6 or np.linalg.norm(ax) < 1e-6:
        return None, dict(why='degenerate_handle_axes')
    n, d = n / np.linalg.norm(n), -ax / np.linalg.norm(ax)
    yaw = math.atan2(-n[1], -n[0])
    Tp, Tg = np.asarray(T_at(T_WH, s_pre), float), np.asarray(T_at(T_WH, 0.0), float)
    seeds = [np.array(s_) for s_ in itertools.product((-0.6, 0.0, 0.6), (-0.3, 0.2), (0.6, 1.4), (-1.5, 0.0, 1.5), (-0.7, 0.7), (0.0,))]
    best, stat = None, dict(n_candidates=0, n_ik_ok=0, n_feasible=0)
    for so in (np.arange(0.36, 0.5601, 0.01) if standoffs is None else standoffs):
        for lat_along in (np.arange(-0.12, 0.1201, 0.02) if alongs is None else alongs):               # 底盤沿通過方向相對把手的偏移（抓取瞬間）
            b = np.r_[T_WH[:2, 3] + so * n[:2] + lat_along * d[:2], yaw]
            sols = []
            for sd in seeds:
                stat['n_candidates'] += 1
                q1, r1, s1 = DFK.ik_closure_fc(Tp, b, sd)
                if not (r1 < 1e-6 and np.isfinite(q1).all()):
                    continue
                q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
                if any(np.abs(q1 - x[0]).max() < 1e-3 for x in sols):
                    continue
                q0, r0, s0 = DFK.ik_closure_fc(Tg, b, q1)
                if not (r0 < 1e-6 and np.isfinite(q0).all()):
                    continue
                stat['n_ik_ok'] += 1
                sols.append((q1, q0, s1, s0))
            for q1, q0, s1, s0 in sols:
                if lo is not None and (np.any(q1 < lo) or np.any(q1 > hi) or np.any(q0 < lo) or np.any(q0 > hi)):
                    stat['rej_joint_limit'] = stat.get('rej_joint_limit', 0) + 1
                    continue
                if sig_floor is not None and min(s1, s0) < sig_floor:
                    stat['rej_manipulability'] = stat.get('rej_manipulability', 0) + 1
                    continue
                if clear_fn is not None:
                    c1, c0 = clear_fn(np.r_[b, q1]), clear_fn(np.r_[b, q0])
                    bad = [k for k in ('env', 'base', 'self') if c1[k] < thr[k]] + [k + '@grasp' for k in thr if c0[k] < thr[k]]
                    if bad:
                        for k in bad:
                            stat['rej_clear_' + k] = stat.get('rej_clear_' + k, 0) + 1
                        continue
                stat['n_feasible'] += 1
                sc = min(s1, s0)
                if collect is not None:                               # 收集全部可行候選（供「先可連續到達再排序」的選擇）
                    collect.append(dict(along=float(lat_along), standoff=float(so), yaw=float(yaw), base=b.copy(), q_pregrasp=q1.copy(), q_grasp=q0.copy(), sigma_min=float(sc), d=d[:2].copy(), n=n[:2].copy(), p0=b[:2].copy()))
                if best is None or sc > best['sigma_min'] + 1e-9:
                    best = dict(along=float(lat_along), standoff=float(so), yaw=float(yaw), base=b.copy(), q_pregrasp=q1.copy(), q_grasp=q0.copy(), sigma_min=float(sc), d=d[:2].copy(), n=n[:2].copy())
    if best is None:
        return None, dict(stat, why='no_feasible_pregrasp')
    best['p0'] = best['base'][:2].copy()                          # 抓取瞬間底盤位置（通過線上的抓取點）
    return best, stat


_A_H = None


def set_approach_axis(a_H):
    global _A_H
    _A_H = np.asarray(a_H, float)
