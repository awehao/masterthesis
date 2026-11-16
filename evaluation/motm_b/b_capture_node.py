#!/usr/bin/env python3
"""主線 B 無接觸取像趟的監督節點（Codex 20261117_034316／035xxx：一條候選、一趟無接觸取像；不閉爪、不綁定、不開門）。
原導航鏈不改：/goal_pose → goal_to_plan_relay → Smac2D /plan → gmpc_task_node（封存 GMPC 的子類別）→ 本節點命令轉送 → 平滑器 → Shield → GUARD → /cmd_vel。
本節點只做：
  1 導航目標：起點先送先驗名目把手（場景登錄的門資產，門關閉）前方門前線上的遠端點；不寫死接手點。
  2 影像：/far/handle_obs 的實際估計（相對擷取同刻底盤）以擷取同刻里程轉入里程座標；三格位置差 ≤5 mm 才形成把手估計；對先驗做目標關聯。
  3 在線規劃（工作程序，計入真實時間）：取當下 Smac /plan 與把手估計呼叫 door_b_plan.BPlanner.plan(require_cover=False)——門前線、接手點、觀察族、過渡起點與時長；
    估計再變 >10 mm 且尚未進觀察段 ⇒ 重算（版本遞增）。規劃完成前底盤照導航走（速度框上限 V_APPROACH），不停車等。
  4 任務條件（/task/conditions）：朝向＝面門；速度框：沿門方向上限依底盤實測沿門座標 a 的剖面（v_tr，接手點前以 a_dec 減速到 0.05），過接手點 ⇒ 0（正常終點）。
  5 手臂（每 0.05 s 一拍，在工作程序）：PREP（最小急動到觀察族首構型）→ OBSERVE（以預測底盤位姿重解注視把手的 IK，參數依 a 內插觀察族）
    → TRANSITION（a 到過渡起點後，關節空間最小急動，進度依 a 的剖面反算）→ HOLD（預抓位）。
    每拍：候選命令先過完整停止核對（b_stopcheck：now／next 兩停法、實際設定點、在途管線、偏置模型、區間界）；commit 才發，stop_now ⇒ 手臂零、底盤由本節點轉送零（受控停止），
    abort（含逾時、例外）⇒ /far/abort（模擬工程中止）。
  6 正常終點：底盤過接手點後速度框 0、手臂保持預抓位，實測停穩 0.3 s ⇒ /far/done。
紀錄：每拍、每格觀察、每次規劃與任務條件回覆，寫 <out>.json。
    python3 b_capture_node.py OUT_PREFIX"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT, PHYS, NPH, D = 0.05, 0.01, 5, 8
V_APPROACH = float(os.environ.get('B_V_APPROACH', '0.20'))          # 規劃完成前的速度框上限（m/s；不停車）
V_ARM = 1.0
LEAD_S = D * PHYS + DT + 0.10                                        # 手臂參考前看：在途 8 步＋一拍＋關節一階落後（τ≈0.1 s）
EST = dict(n=3, spread_m=0.005, max_age_s=0.5, replan_shift_m=0.010)
ASSOC_TOL = dict(dxy_m=0.15, dz_m=0.10, dang_deg=10.0)               # 同 merge_node
GOAL_A = -0.49                                                       # 遠端導航目標的沿門座標（門前線、把手後方 0.49 m；與 smac_probe_p1 相同）；接手點由規劃決定，不是這個
SB_PRIOR = 0.45
SB_NAV_LINE = 0.4607                                                  # Smac 路徑沒有門前線段時的合成門前線離門距離＝先前 Smac 自然路徑實測（results/motm_b/smac_probe_p1.json）
STILL = dict(base=0.003, yaw=0.003, arm=0.002, hold_s=0.3)
ON_LINE = 0.05                                                       # 已上門前線的判定（|sb−門前線|，m）
ENTRY_EXTRA, ENTRY_MARGIN, ENTRY_SWITCH_R = 0.05, 0.15, 0.10         # 進線點沿門座標上限＝觀察族起點＋0.05、下限＝過渡起點＋0.15；離門前線 ≤0.10 m 就取消進線點（只剩遠端目標）
ENTRY_LOOK = 0.10                                                    # 進線點在目前沿門位置前方 0.10 m（路徑經過、不停）


def _T3(p):
    import numpy as np
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


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
    Kc = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])   # 腕部相機內參（同 far_capture／far_door 各趟 frames_meta 的 K）
    SWP._die_with_parent()
    pool = SWP.SweepPool(int(os.environ.get('B_POOL_N_PLANNER') or os.environ.get('B_POOL_N', '2')))
    grav = DFK.FastGravity3(K := G.K, G.BM.arm_bodies(G.K))
    KP_ARM = 600.0
    st = dict(T_est=None, off=None, plan=None, bp=None, hist=[], prep=None, tr=None, q_des_prev=None, n_clip=0)

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

    conn.send(('ready', 0, dict(pool_n=pool.n if hasattr(pool, 'n') else None, sweep_s=SC.SWEEP_S)))
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
                conn.send(('prior', ver, dict(goal_xy=(p0 - GOAL_A * d_).tolist(), yaw=yaw_, d=d_.tolist(), n=n_.tolist(), handle_xy=T_prior[:2, 3].tolist(),
                                              source=dict(asset=sreg['door']['asset'], sha256=hashlib.sha256(open(ap_, 'rb').read()).hexdigest(), note='名目位姿（門關閉）；不讀把手 prim 真值'))))
            elif op == 'obs':                                          # 一格有效觀察（里程座標）→ 三格一致才形成估計
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
                        shift = None if st['T_est'] is None else float(np.linalg.norm(T_new[:3, 3] - st['T_est'][:3, 3]))
                        Tm = _T3(np.asarray(a['T_mo'], float)) @ T_new
                        dR = st['T_prior'][:3, :3].T @ Tm[:3, :3]
                        assoc = dict(dxy_m=round(float(np.linalg.norm(Tm[:2, 3] - st['T_prior'][:2, 3])), 4), dz_m=round(float(abs(Tm[2, 3] - st['T_prior'][2, 3])), 4),
                                     dang_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), 3))
                        ok_as = assoc['dxy_m'] <= ASSOC_TOL['dxy_m'] and assoc['dz_m'] <= ASSOC_TOL['dz_m'] and assoc['dang_deg'] <= ASSOC_TOL['dang_deg']
                        out.update(assoc=assoc, assoc_ok=ok_as, shift_mm=None if shift is None else round(shift * 1e3, 2), seqs=[x[0] for x in st['hist']], t_obs=float(a['t_obs']))
                        if ok_as and (st.get('T_cand') is None or (a.get('allow_update') and float(np.linalg.norm(T_new[:3, 3] - st['T_cand'][:3, 3])) > EST['replan_shift_m'])):
                            st['T_cand'] = T_new                         # 規劃用估計（採用規劃時才成為停止核對用的 T_est）
                            out['estimate_set'] = True
                            out['T_est'] = T_new.tolist()
                conn.send(('obs', ver, out))
            elif op == 'plan':                                         # 線上規劃（地圖座標；把手估計以當下 map←odom 轉入）
                t0 = time.perf_counter()
                T_mo = np.asarray(a['T_mo'], float)
                T_Hm = _T3(T_mo) @ np.asarray(a['T_est'], float)
                d_, n_, yaw_ = door_frame(T_Hm)
                bp = BP.BPlanner(T_Hm, d_, n_, yaw_, Kc)
                bp._alpha = AL
                pl = bp.plan(np.asarray(a['path'], float), np.asarray(a['q_arm'], float), require_cover=False)
                pl['path_source'] = 'smac'
                if pl.get('why') == 'path_not_along_door_line_through_window':   # Smac 路徑沒有沿門前線段（斜切）⇒ 以門前線合成規劃路徑；導航以進線點沿途約束帶上線（仍由原導航規劃與避障）
                    P0 = np.asarray(a['path'], float)[0]
                    line = [T_Hm[:2, 3] + SB_NAV_LINE * np.asarray(n_) - aa * np.asarray(d_) for aa in np.arange(0.70, -0.551, -0.02)]
                    pl = bp.plan(np.array([P0] + line), np.asarray(a['q_arm'], float), require_cover=False)
                    pl['path_source'] = 'door_line_sb%.3f' % SB_NAV_LINE
                keep = {k: v for k, v in pl.items() if k not in ('path_door',)}
                conn.send(('plan', ver, dict(plan=keep, wall_s=round(time.perf_counter() - t0, 3), yaw=yaw_, d=d_.tolist(), n=n_.tolist(), T_Hm=T_Hm.tolist(), T_est=a['T_est'])))
            elif op == 'adopt':                                        # 採用規劃程序的結果（重建同一 BPlanner；把手估計換成該次規劃所用者）
                pl = a['plan']
                bp = BP.BPlanner(np.asarray(a['T_Hm'], float), np.asarray(a['d'], float), np.asarray(a['n'], float), float(a['yaw']), Kc)
                bp._alpha = AL
                q9 = np.r_[bp.base(float(pl['a_w']), float(pl['door_line_sb'])), np.asarray(pl['q_pre'], float)]
                bp.T_TC = np.linalg.inv(K.fk(q9, 'link_tcp')) @ K.fk(q9, BP.W0.OPT)
                st['bp'], st['plan'], st['T_est'] = bp, pl, np.asarray(a['T_est'], float)
                st['fam'] = sorted(pl['obs'], key=lambda f: f['a'])
                st['prep'], st['tr'] = None, None
                conn.send(('adopt', ver, dict(ok=True)))
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
                    qm = to_map(q, T_mo)                                # 地圖座標構型
                    am, sbm = bp.to_door(qm[:2])
                    v_al = float(a.get('v_along', 0.0))                 # 沿門速度實測（往 −a 為正）
                    a_pred = am - v_al * LEAD_S
                    b_pred = bp.base(a_pred, sbm); b_pred[2] = qm[2] + float(a.get('wz', 0.0)) * LEAD_S
                    out.update(a=round(am, 4), sb=round(sbm, 4), a_pred=round(a_pred, 4))
                    fam = st['fam']
                    A = np.array([f['a'] for f in fam]); Q = np.array([f['q'] for f in fam])
                    q_des, info = None, {}
                    if mode == 'PREP':
                        pr = st['prep']
                        if pr is None or a.get('prep_restart'):
                            a_o = float(np.clip(a_pred, A[0], A[-1]))
                            q_o = np.array([np.interp(a_o, A, Q[:, j]) for j in range(6)])
                            dq = q_o - (s + off_fn(q))
                            T_p = max(float(np.abs(dq).max()) * 1.875 / (0.8 * V_ARM), float(np.sqrt(float(np.abs(dq).max()) * 5.7735 / 10.0)), 1.0)
                            pr = st['prep'] = dict(t0=t, T=T_p, q0=(s + off_fn(q)).copy(), q1=q_o, a_o=a_o)
                        u = (t + LEAD_S - pr['t0']) / pr['T']
                        q_des = pr['q0'] + minjerk(u) * (pr['q1'] - pr['q0'])
                        info = dict(u=round(min(u, 1.0), 3), T=round(pr['T'], 3), a_o=round(pr['a_o'], 3))
                        out['prep_done'] = bool(u >= 1.0 and float(np.abs(q[3:] - pr['q1']).max()) < 0.02)
                    elif mode == 'OBSERVE':
                        a_c = float(np.clip(a_pred, A[0], A[-1]))
                        i = int(np.clip(np.searchsorted(A, a_c), 1, len(A) - 1)); w = (a_c - A[i - 1]) / max(A[i] - A[i - 1], 1e-9)
                        par = [fam[i - 1]['par'][k] + w * (fam[i]['par'][k] - fam[i - 1]['par'][k]) for k in range(3)]
                        seed = Q[i - 1] + w * (Q[i] - Q[i - 1])
                        qi, sm = bp.solve(b_pred, bp.look_at(b_pred, *par), seed if st['q_des_prev'] is None else st['q_des_prev'])
                        if qi is None or sm < W.SIG_FLOOR or float(np.abs(qi - seed).max()) > 0.3:
                            q_des = seed; info['ik'] = 'fallback_family'
                        else:
                            q_des = qi; info['ik'] = 'look_at'; info['sigma'] = round(sm, 4)
                        info['par'] = [round(x, 3) for x in par]
                        out['reach_t0'] = bool(a_pred <= float(pl['a_t0']))
                    elif mode in ('TRANSITION', 'HOLD'):
                        tr = st['tr']
                        if tr is None:
                            prof = bp.speed_profile(float(pl['a_t0']), float(pl['a_w']))
                            T_tr, aot = prof
                            tt = np.linspace(0, T_tr, 400); aa = np.array([aot(x) for x in tt])
                            tr = st['tr'] = dict(q0=(st['q_des_prev'] if st['q_des_prev'] is not None else s + off_fn(q)).copy(), q1=np.asarray(pl['q_pre'], float), tt=tt, aa=aa, T=T_tr)
                        u_t = float(np.interp(-a_pred, -tr['aa'], tr['tt'])) / tr['T']   # 進度依 a 反算（剖面 a(t) 單調遞減）
                        q_des = tr['q0'] + minjerk(u_t) * (tr['q1'] - tr['q0'])
                        info = dict(u=round(min(max(u_t, 0.0), 1.0), 3))
                        out['reach_pre'] = bool(u_t >= 1.0)
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
                    out['disp'] = 'no_estimate'                          # 還沒有把手估計：手臂不動（命令零）；底盤照導航
                out['ua'] = [float(x) for x in ua]
                out['n_clip'] = st['n_clip']
                out['tm'] = (t_rx, time.perf_counter())
                conn.send(('tick', ver, out))
        except Exception as e:                                         # noqa: BLE001
            import traceback
            conn.send(('error', ver, traceback.format_exc()[-600:]))


def main():
    out = sys.argv[1]
    parent, child = mp.Pipe()
    proc = mp.get_context('fork').Process(target=worker, args=(child,), daemon=False)
    proc.start()
    pk, pk_child = mp.Pipe()                                             # 獨立規劃程序（規劃不擋每拍核對與命令轉送；完成時刻＝實際收到的模擬時間）
    os.environ['B_POOL_N_PLANNER'] = '1'
    proc_p = mp.get_context('fork').Process(target=worker, args=(pk_child,), daemon=False)
    proc_p.start()
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float64MultiArray, String
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from nav2_msgs.action import ComputePathToPose
    import tf2_ros

    class Sup(Node):
        def __init__(self):
            super().__init__('b_capture_supervisor', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
            self.p_goal = self.create_publisher(PoseStamped, '/b_capture/goal', 10)   # 只記錄（不經 /goal_pose；路徑由本節點向 planner_server 要）
            self.p_plan_out = self.create_publisher(Path, '/plan', QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
            self.cp = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
            self.nav_goal, self.nav_via, self.nav_busy, self.t_nav_req, self.nav_dirty, self.nav_log = None, None, False, -9.0, False, []
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
            self.create_subscription(String, '/far/exec_state', lambda m: setattr(self, 'xstate', m.data), 10)
            self.tfb = tf2_ros.Buffer(); self.tfl = tf2_ros.TransformListener(self.tfb, self)
            self.truth = None; self.rel = np.zeros(3); self.path = None; self.path_t = None; self.xstate = None
            self.mode, self.ver, self.task_ver = 'INIT', 0, 0
            self.stopped, self.done_sent, self.aborted = None, False, None
            self.events, self.ticks, self.obs_log, self.acks, self.plans = [], [], [], [], []
            self.prior = None; self.est_set = False; self.plan_pending = False; self.plan = None; self.plan_ver = 0; self.T_est = None; self.plan_req = None
            self.goal_sent = None; self.still_t = None; self.n_late = 0; self.n_fwd = 0
            self.b_prev = None; self.on_line = False
            self.create_timer(DT, self.tick)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def event(self, what, **kw):
            rec = dict(t=round(self.now(), 3), what=what, **kw)
            self.events.append(rec)
            print('[bcap] ' + json.dumps(rec, ensure_ascii=False, default=str)[:400], flush=True)

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

        def _nav_in(self, m):                                          # 命令轉送（平滑器之前、唯一來源）：受控停止後改送零
            if self.stopped is not None:
                m = Twist()
            self.p_nav.publish(m); self.n_fwd += 1

        def _rel(self, m):
            self.rel = np.array([m.linear.x, m.linear.y, m.angular.z], float)

        def _truth(self, m):
            a_ = np.array(m.data, float)
            self.truth = dict(t=a_[0], q=a_[1:10], s=a_[10:16], s_end=a_[16:22], pipe=a_[22:94].reshape(8, 9))

        def _plan_in(self, m):
            self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses], float); self.path_t = self.now()

        def _ack(self, m):
            try:
                self.acks.append(json.loads(m.data))
            except Exception:                                          # noqa: BLE001
                pass

        def _obs(self, m):
            d = np.array(m.data, float)
            rec = dict(seq=int(d[0]), t_obs=float(d[1]), valid=bool(d[2] > 0.5), t_rx=round(self.now(), 3), t_proc=float(d[22]) if len(d) > 22 else None)
            self.obs_log.append(rec)
            if rec['valid'] and self.prior is not None:
                T_mo = self.tf2d('map', 'odom')
                if T_mo is None:
                    return
                T_oh = _T3(d[3:6]) @ d[6:22].reshape(4, 4)
                allow = self.mode in ('NAV', 'PLANNED', 'PREP')
                tag, pay = self.ask('obs', dict(seq=rec['seq'], t_obs=rec['t_obs'], t=self.now(), T_odom_handle=T_oh.tolist(), T_mo=T_mo.tolist(), allow_update=allow), 0.5)
                rec['est'] = pay
                if tag == 'obs' and pay.get('estimate_set'):
                    self.est_set = True; self.T_est = pay['T_est']; self.event('estimate_set', **{k: pay.get(k) for k in ('seqs', 'assoc', 'shift_mm', 'spread_mm', 't_obs')})
                    self.plan_pending = True

        def tf2d(self, parent_, child):
            try:
                tr = self.tfb.lookup_transform(parent_, child, rclpy.time.Time())
            except Exception:                                          # noqa: BLE001
                return None
            o = tr.transform.rotation
            return np.array([tr.transform.translation.x, tr.transform.translation.y, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))])

        def set_nav(self, goal_xy, yaw, via=None):
            """導航目標（地圖）與沿途進線點：本節點向原 planner_server 要分段 Smac 路徑（目前位置→進線點→目標），串成一條發 /plan 給原 GMPC。"""
            self.nav_goal = (list(goal_xy), float(yaw)); self.nav_via = None if via is None else list(via); self.nav_dirty = True
            self.pub_goal(goal_xy, yaw)

        def _pose(self, xy, yaw):
            m = PoseStamped(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1]); m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            return m

        def request_paths(self, start_xy, start_yaw):
            g, yaw = self.nav_goal
            pts = [list(start_xy)] + ([self.nav_via] if self.nav_via is not None else []) + [g]
            self.nav_busy, self.t_nav_req, self.nav_dirty = True, self.now(), False
            self.leg_req = dict(pts=pts, res=[], t=self.now(), w=time.perf_counter(), yaw=yaw, start_yaw=start_yaw)
            self._send_leg(0)

        def _send_leg(self, i):
            P = self.leg_req['pts']
            gl = ComputePathToPose.Goal()
            gl.start = self._pose(P[i], self.leg_req['start_yaw'] if i == 0 else self.leg_req['yaw'])
            gl.goal = self._pose(P[i + 1], self.leg_req['yaw']); gl.use_start = True; gl.planner_id = 'GridBased'
            fut = self.cp.send_goal_async(gl)
            fut.add_done_callback(lambda f, i=i: self._leg_acc(f, i))

        def _leg_acc(self, f, i):
            gh = f.result()
            if gh is None or not gh.accepted:
                self.nav_busy = False; self.nav_log.append(dict(t=round(self.now(), 3), ok=False, why='leg%d_rejected' % i)); return
            gh.get_result_async().add_done_callback(lambda r, i=i: self._leg_done(r, i))

        def _leg_done(self, r, i):
            try:
                path = r.result().result.path
                xy = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            except Exception as e:                                     # noqa: BLE001
                xy = []
            if len(xy) < 2:
                self.nav_busy = False; self.nav_log.append(dict(t=round(self.now(), 3), ok=False, why='leg%d_empty' % i)); return
            self.leg_req['res'].append(xy)
            if i + 2 < len(self.leg_req['pts']):
                self._send_leg(i + 1); return
            allp = list(self.leg_req['res'][0])
            for seg in self.leg_req['res'][1:]:
                allp += list(seg[1:])
            m = Path(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            for k, (x, y) in enumerate(allp):
                x2, y2 = allp[min(k + 1, len(allp) - 1)]
                th = math.atan2(y2 - y, x2 - x) if (x2, y2) != (x, y) else self.leg_req['yaw']
                ps = PoseStamped(); ps.header = m.header; ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
                m.poses.append(ps)
            self.p_plan_out.publish(m)
            self.nav_busy = False
            self.nav_log.append(dict(t=round(self.now(), 3), ok=True, n=len(allp), legs=len(self.leg_req['res']), via=self.nav_via, goal=self.nav_goal[0], wall_ms=round((time.perf_counter() - self.leg_req['w']) * 1e3, 1)))

        def pub_goal(self, xy, yaw):
            if self.goal_sent is None or list(xy) != list(self.goal_sent):
                self.t_goal = self.now()
            m = PoseStamped(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1]); m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.p_goal.publish(m)

        def pub_task(self, heading, box, valid_s=0.5, t_eff_s=0.5, clear=False):
            self.task_ver += 1
            d_ = dict(task='b_capture', ver=self.task_ver, frame='map', stamp=self.now(), t_eff_s=t_eff_s, valid_s=valid_s)
            if clear:
                d_['clear'] = True
            else:
                d_.update(heading=float(heading), box=[[float(x) for x in r] for r in box])
            self.p_task.publish(String(data=json.dumps(d_)))
            return d_

        def box_for(self, v_along, v_norm, yaw_now, d_world):
            """本體座標速度框：沿門方向（往 −a，世界 d_world）上限 v_along、離門法向 |v| ≤ v_norm。朝向與面門差 ≤15° 時把沿門／法向對到最接近的本體軸
            （沿門軸只允許行進方向＋0.02 的倒退餘量）；否則兩軸同限 min(v_along, v_norm)（保守）。"""
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

        def tick(self):
            t = self.now()
            if self.truth is None or self.aborted is not None:
                return
            T_mo = self.tf2d('map', 'odom'); b_od = self.tf2d('odom', 'base_footprint')
            if T_mo is None or b_od is None:
                return
            q = np.r_[b_od, self.truth['q'][3:9]]
            if self.mode == 'INIT':
                tag, pay = self.ask('prior', {}, 5.0)
                if tag != 'prior':
                    self.abort('prior_failed', detail=str(pay)[:200]); return
                self.prior = pay; self.event('prior', **pay)
                self.set_nav(pay['goal_xy'], pay['yaw']); self.goal_sent = pay['goal_xy']
                tag, pay2 = self.ask('off_anchor', dict(q=q.tolist(), s=self.truth['s'].tolist()), 2.0)
                self.event('off_anchor', ok=tag == 'off_anchor', off0=None if pay2 is None else pay2.get('off0'))
                self.mode = 'NAV'; self.event('mode', mode='NAV')
                return
            if self.nav_goal is not None and not self.nav_busy and (self.nav_dirty or t - self.t_nav_req >= 1.0) and self.cp.server_is_ready():
                bm = np.r_[math.cos(T_mo[2]) * b_od[0] - math.sin(T_mo[2]) * b_od[1] + T_mo[0], math.sin(T_mo[2]) * b_od[0] + math.cos(T_mo[2]) * b_od[1] + T_mo[1]]
                self.request_paths(bm.tolist(), float(b_od[2] + T_mo[2]))   # 每 1 s 由目前位置重算（同 goal_to_plan_relay 的重規劃週期）
            # 底盤速度（地圖座標、實測差分）
            qm_xy = None
            if self.b_prev is not None and t > self.b_prev[0]:
                v_w = (b_od[:2] - self.b_prev[1][:2]) / (t - self.b_prev[0]); wz = (b_od[2] - self.b_prev[1][2]) / (t - self.b_prev[0])
            else:
                v_w, wz = np.zeros(2), 0.0
            self.b_prev = (t, b_od.copy())
            # ---- 規劃（獨立程序、非同步；未進觀察段才可重算／採用）
            if self.plan is None and self.T_est is not None and self.plan_req is None and self.mode == 'NAV' and t - getattr(self, 't_plan_try', -9.0) >= 1.0:
                self.plan_pending = True                                  # 尚無可行規劃：每 1 s 以最新 Smac 路徑重試（底盤照導航繼續走，不停車）
            if self.plan_pending and self.plan_req is None and self.mode in ('NAV', 'PREP') and self.path is not None and len(self.path) > 3 and self.T_est is not None:
                self.t_plan_try = t
                self.plan_pending = False; self.ver += 1
                self.plan_req = dict(ver=self.ver, t=t, w=time.perf_counter())
                pk.send(('plan', dict(ver=self.ver, path=self.path.tolist(), q_arm=q[3:].tolist(), T_mo=T_mo.tolist(), T_est=self.T_est)))
                self.event('plan_request', ver=self.ver, n_path=len(self.path)); self.plan_paths = getattr(self, 'plan_paths', []) + [dict(ver=self.ver, t=round(t, 3), path=self.path.round(4).tolist())]
            if self.plan_req is not None and pk.poll(0):
                tag, v, pay = pk.recv()
                rq, self.plan_req = self.plan_req, None
                rec = dict(t_req=round(rq['t'], 3), t_done=round(t, 3), wall_s=round(time.perf_counter() - rq['w'], 3), tag=tag, why=(pay['plan'].get('why') if tag == 'plan' else str(pay)[:300]),
                           plan=None if tag != 'plan' else {k: pay['plan'].get(k) for k in ('door_line_sb', 'a_w', 'a_t0', 'T_tr', 'last_valid_t', 'blind_s', 'blind_def', 'goal_xy', 'v_tr', 'a_dec', 'path_source')},
                           obs=None if tag != 'plan' else [(f['a'], f['range'], f['oblique'], f['par']) for f in pay['plan'].get('obs', [])])
                self.plans.append(rec); self.event('plan_result', **{k: rec[k] for k in ('t_req', 't_done', 'wall_s', 'tag', 'why', 'plan')})
                if tag == 'plan' and pay['plan'].get('why') is None and self.mode in ('NAV', 'PREP'):
                    tag2, p2 = self.ask('adopt', dict(plan=pay['plan'], T_Hm=pay['T_Hm'], d=pay['d'], n=pay['n'], yaw=pay['yaw'], T_est=pay['T_est']), 5.0)
                    if tag2 == 'adopt':
                        self.plan = dict(pay['plan'], yaw=pay['yaw'], d=pay['d'], n=pay['n']); self.plan_ver += 1
                        self.a_obs_max = max(f['a'] for f in pay['plan']['obs'])
                        self.mode = 'PREP'; self.event('mode', mode='PREP', plan_ver=self.plan_ver)
                        Th_ = np.asarray(pay['T_Hm'], float); d_ = np.array(pay['d']); n_ = np.array(pay['n'])
                        self.entry_geo = (Th_[:2, 3].copy(), float(pay['plan']['door_line_sb']), n_, d_)
                        a_now = self.ticks[-1].get('a') if self.ticks and self.ticks[-1].get('a') is not None else float(self.a_obs_max)
                        a_E = float(np.clip(a_now - ENTRY_LOOK, float(pay['plan']['a_t0']) + ENTRY_MARGIN, float(self.a_obs_max) + ENTRY_EXTRA))
                        self.entry = (Th_[:2, 3] + float(pay['plan']['door_line_sb']) * n_ - a_E * d_); self.t_entry_pub = t
                        self.final_goal = pay['plan'].get('goal_xy')
                        self.goal_phase = 'entry'
                        self.set_nav(self.final_goal, pay['yaw'], via=self.entry.tolist()); self.goal_sent = self.final_goal   # 沿途約束：Smac 分段路徑經門前線上的進線點再到遠端目標；路徑規劃與避障仍由原導航
                        self.event('nav_goal_entry', entry=[round(float(x), 4) for x in self.entry], a_entry=round(a_E, 3), final=self.final_goal)
                    else:
                        self.event('adopt_failed', detail=str(p2)[:200])
                elif self.plan is None:
                    self.event('plan_infeasible_keep_nav', why=rec['why'])
            if getattr(self, 'goal_phase', None) == 'entry' and self.final_goal is not None:
                qm_ = np.r_[math.cos(T_mo[2]) * q[0] - math.sin(T_mo[2]) * q[1] + T_mo[0], math.sin(T_mo[2]) * q[0] + math.cos(T_mo[2]) * q[1] + T_mo[1]]
                h_, sbl, n_, d_ = self.entry_geo
                rel_ = qm_ - h_; a_now, sb_now = float(-(rel_ @ d_)), float(rel_ @ n_)
                if t - self.t_entry_pub >= 1.0:                         # 進線點＝門前線上、目前沿門位置往前 ENTRY_LOOK 的點（夾在 過渡起點＋邊距 與 觀察族起點 之間），每 1 s 更新
                    a_E = float(np.clip(a_now - ENTRY_LOOK, float(self.plan['a_t0']) + ENTRY_MARGIN, float(self.a_obs_max) + ENTRY_EXTRA))
                    e_new = h_ + sbl * n_ - a_E * d_
                    if float(np.linalg.norm(e_new - self.entry)) > 0.05:
                        self.entry = e_new; self.set_nav(self.final_goal, self.plan['yaw'], via=self.entry.tolist())
                        self.event('nav_goal_entry_update', entry=[round(float(x), 4) for x in self.entry], a_entry=round(a_E, 3))
                    self.t_entry_pub = t
                if abs(sb_now - sbl) <= ENTRY_SWITCH_R:
                    self.goal_phase = 'final'
                    self.set_nav(self.final_goal, self.plan['yaw']); self.goal_sent = self.final_goal
                    self.event('nav_goal_final', at=[round(float(x), 4) for x in qm_], a=round(a_now, 3), sb=round(sb_now, 3))
            # ---- 任務條件（朝向＋速度框）：沿門剖面只在已上門前線（|sb−門前線| ≤ ON_LINE）後生效
            on_line = False
            if self.plan is not None and self.stopped is None:
                pl = self.plan
                lt = self.ticks[-1] if self.ticks else {}
                a_m, sb_m = lt.get('a'), lt.get('sb')
                on_line = a_m is not None and sb_m is not None and abs(sb_m - float(pl['door_line_sb'])) <= ON_LINE
                if a_m is None or not on_line:
                    v_al, v_n = V_APPROACH, V_APPROACH
                    if a_m is not None and a_m <= float(pl['a_t0']) and not on_line and self.mode in ('PREP', 'OBSERVE'):
                        self.stop_all('missed_door_line_before_transition', a=a_m, sb=sb_m, sb_line=pl['door_line_sb'])
                elif a_m <= pl['a_w']:
                    v_al, v_n = 0.0, 0.0
                else:
                    v_dec = math.sqrt(0.05 ** 2 + 2 * float(pl['a_dec']) * max(a_m - float(pl['a_w']), 0.0))
                    v_al, v_n = min(V_APPROACH, float(pl['v_tr']), v_dec), 0.05
                self.pub_task(pl['yaw'], self.box_for(v_al, v_n, float(T_mo[2] + b_od[2]), np.array(pl['d'])))
            self.on_line = on_line
            # ---- 手臂拍
            v_al = float(-(v_w @ np.array(self.plan['d']))) if self.plan is not None else 0.0
            mode_w = self.mode if self.mode in ('PREP', 'OBSERVE', 'TRANSITION', 'HOLD') else 'NAV'
            w0 = time.perf_counter()
            tag, out_ = self.ask('tick', dict(t=t, mode=mode_w, q=q.tolist(), pipe=self.truth['pipe'].tolist(), rel=self.rel.tolist(), s=self.truth['s'].tolist(), s_end=self.truth['s_end'].tolist(),
                                              T_mo=T_mo.tolist(), v_along=v_al, wz=wz), 0.9 * DT / max(float(os.environ.get('B_RTF', '0.4')), 1e-3))
            rec = dict(t=round(t, 3), mode=self.mode, wall_ms=round((time.perf_counter() - w0) * 1e3, 1), tag=tag)
            if tag != 'tick':
                self.n_late += 1
                rec['why'] = 'late_or_error' if tag == 'late' else str(out_)[:300]
                self.ticks.append(rec)
                if self.mode in ('PREP', 'OBSERVE', 'TRANSITION', 'HOLD') and self.stopped is None:
                    self.abort('tick_' + tag, detail=rec['why'])          # 沒有核過的命令也沒有核過的停法 ⇒ 模擬工程中止（保守）
                else:
                    self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
                return
            rec.update({k: out_.get(k) for k in ('a', 'sb', 'a_pred', 'disp', 'stop', 'info', 'ua', 'n_clip', 'vis_geom', 'prep_done', 'reach_t0', 'reach_pre')})
            self.ticks.append(rec)
            disp = out_['disp']
            if self.stopped is not None:
                self.p_arm.publish(Float64MultiArray(data=[0.0] * 6))
            elif disp == 'commit' or disp == 'no_estimate':
                self.p_arm.publish(Float64MultiArray(data=[float(x) for x in out_['ua']] if disp == 'commit' else [0.0] * 6))
            elif disp == 'stop_now':
                self.stop_all('stop_check_next_failed', stop=out_.get('stop'))
            else:
                self.abort('stop_check_now_failed', stop=out_.get('stop'))
                return
            # ---- 段切換
            if self.mode == 'PREP' and out_.get('prep_done') and self.on_line:
                self.mode = 'OBSERVE'; self.event('mode', mode='OBSERVE', a=out_.get('a'), sb=out_.get('sb'))
            elif self.mode == 'OBSERVE' and out_.get('reach_t0') and self.on_line:
                self.mode = 'TRANSITION'; self.event('mode', mode='TRANSITION', a=out_.get('a'))
            elif self.mode == 'TRANSITION' and out_.get('reach_pre'):
                self.mode = 'HOLD'; self.event('mode', mode='HOLD', a=out_.get('a'))
            # ---- 正常終點：過接手點、速度框 0、整機停穩
            if self.mode == 'HOLD' and not self.done_sent:
                tr = self.truth
                still = float(np.linalg.norm(v_w)) < STILL['base'] and abs(wz) < STILL['yaw'] and float(np.abs(tr['pipe']).max()) < 1e-9 and float(np.abs(np.asarray(out_['ua'])).max()) < 1e-3
                if still:
                    self.still_t = self.still_t if self.still_t is not None else t
                    if t - self.still_t >= STILL['hold_s']:
                        self.done_sent = True; self.p_done.publish(Bool(data=True)); self.event('done_normal_end', a=out_.get('a'))
                else:
                    self.still_t = None

        def save(self):
            json.dump(dict(events=self.events, plans=self.plans, plan_paths=getattr(self, 'plan_paths', []), nav_log=self.nav_log, ticks=self.ticks, obs=self.obs_log, acks=self.acks[-200:], n_late=self.n_late, n_fwd=self.n_fwd, stopped=self.stopped, aborted=self.aborted,
                           params=dict(V_APPROACH=V_APPROACH, LEAD_S=LEAD_S, EST=EST, ASSOC_TOL=ASSOC_TOL, GOAL_A=GOAL_A, STILL=STILL, RTF=os.environ.get('B_RTF', '0.4'), SWEEP_S=os.environ.get('B_SWEEP_S', '0.040'), POOL_N=os.environ.get('B_POOL_N', '2'))),
                      open(out + '.json', 'w'), ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    tag, _v, pay = parent.recv()
    print('[bcap] worker', tag, pay, flush=True)
    tagp, _vp, payp = pk.recv()
    print('[bcap] planner', tagp, payp, flush=True)
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
        print('[bcap] saved', flush=True)


if __name__ == '__main__':
    main()
