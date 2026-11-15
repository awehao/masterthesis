#!/usr/bin/env python3
"""遠距進線模式（MERGE）的監督＋命令選擇＋手臂追蹤節點（乾跑用；Codex 20261116 M1）。
命令路由（平滑器之前、單一有效來源）：/cmd_vel_nav_raw（導航 GMPC）或 /cmd_vel_merge（進線 GMPC）→ 本節點選一個轉發到 /cmd_vel_nav → 原平滑器 → Shield → GUARD → /cmd_vel。
進線 GMPC＝同一個 gmpc_node 程式、參數檔取自現場 dump 的導航 GMPC（CBF、限制、權重全同），只覆寫任務參考：plan_topic、cmd_vel_topic、v_nominal＝通過速度、heading_fixed＝通過線朝向、goal_tolerance_xy。
幾何計算（把手估計、通過線、觀察分支、手臂追蹤、交接核對）在 fork 出的工作程序做（專案模組會裝 rclpy 替身，不能與 rclpy 同程序）。
資料來源照實：機器人狀態取世界節點 /far/truth（位姿、設定點、8 步在途命令＝驅動端讀回）；把手估計值來自 online_1 的實際影像（時間戳＝本趟起點，之後沒有新影像）。"""
import json, math, multiprocessing as mp, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V_PASS, DT, PHYS, NPH, D = 0.05, 0.05, 0.01, 5, 8
CAP_ALONG, CAP_LAT = (0.62, 1.40), 0.60                              # 捕獲區：距抓取點的沿線距離、離線側向距離
PREP_MARGIN = 0.04


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
    clear_fn = lambda x: W.clear_door(np.r_[x, 0.0], T_est)            # noqa: E731
    n = (T_est[:3, :3] @ np.asarray(a_H, float)).copy(); n[2] = 0; n /= np.linalg.norm(n)
    ax = T_est[:3, 0].copy(); ax[2] = 0; d = -ax / np.linalg.norm(ax)
    yaw = math.atan2(-n[1], -n[0])
    st = dict(pg=None, br=None, trk=None)
    conn.send(('ready', dict(est=oinfo, coarse=dict(yaw=yaw, d=d[:2].tolist(), n=n[:2].tolist(), p0=(T_est[:2, 3] + 0.45 * n[:2]).tolist()))))
    while True:
        m = conn.recv()
        if m is None:
            break
        op, a = m
        try:
            if op == 'plan':
                t0 = time.perf_counter()
                pg, info = FR.select_pregrasp_first(T_est, np.asarray(a['q'], float), T_at, S_PRE, K, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS, budget_s=a.get('budget', 8.0))
                br, binfo = (None, None)
                if pg is not None:
                    prep_path = [np.r_[pg['p0'] - pg['d'] * (FR.L_IN - u_ * (FR.L_IN - FR.OBS_ALONG[0])), pg['yaw']] for u_ in np.linspace(0, 1, 9)]
                    br, binfo = FR.observe_branch(T_est, pg, np.asarray(a['q'], float), prep_path, T_at, a_H, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS)
                st['pg'], st['br'] = pg, br
                st['trk'] = FR.ArmPreTracker(tau, lambda q9: np.zeros(6), DT, PHYS, NPH, LO, HI)
                conn.send(('plan', dict(ok=bool(pg is not None and br is not None), wall_s=round(time.perf_counter() - t0, 2), select=info, observe=binfo,
                                        pg=None if pg is None else dict(p0=pg['p0'].tolist(), d=pg['d'].tolist(), n=pg['n'].tolist(), yaw=pg['yaw'], standoff=pg['standoff'], along=pg['along']),
                                        br=None if br is None else dict(q_start_op=br['q_start_op'].tolist(), alongs=[float(br['alongs'][0]), float(br['alongs'][1]), float(br['alongs'][-1])], sigma_min=br['sigma_min'], min_clear=br['min_clear'],
                                                                         prep_time_s=br['prep_time_s'], track_rate_max=br['track_rate_max']))))
            elif op == 'tick':
                q, pipe, rel, s, s_end = np.asarray(a['q'], float), [np.asarray(u, float) for u in a['pipe']], np.asarray(a['rel'], float), np.asarray(a['s'], float), np.asarray(a['s_end'], float)
                pg, br, out = st['pg'], st['br'], {}
                bp = [u[:3] for u in pipe]
                if a['mode'] == 'PREP':
                    dcs, di = FR.track_start_decision(q, bp + [rel] * NPH, pg, q[3:], br, PHYS)
                    out['decision'] = (dcs, di)
                if a['mode'] == 'TRACK':
                    trk = st['trk']
                    o = trk.step(q, bp, rel, s, s_end, br['T_target'], trk.q_prev if trk.q_prev is not None else q[3:])
                    out['track'] = dict(ok=o['ok'], why=o.get('why'), u=[float(x) for x in o['u']], err=o.get('track_err'), clipped=o.get('clipped'), sigma=o.get('sigma'))
                    if a.get('judge'):
                        j = {}
                        for nm, tobs in (('real_image_at_start', a['t_img']), ('synthetic_fresh_stamp', a['t'] - 0.05)):
                            sh = type('A', (), dict(owner='NAV'))()
                            _, why = FL.handover_request(a['t'], sh, pg, q, a['v_meas'], s, pipe, a['u_last'], a['u_prev'], dict(T_WH=T_est, t_obs=tobs), V_PASS, D, off_static=np.zeros(6), tick_phase=0,
                                                         stamps=dict(image=tobs, state=a['t'], cmd=a['t']), arm_track_err=o.get('track_err') if o['ok'] else None)
                            j[nm] = why
                        out['judge'] = j
                if a['mode'] in ('PREP', 'TRACK') or a.get('want_clear'):
                    cl = clear_fn(q)
                    out['clear'] = {k_: float(cl[k_]) for k_ in ('env', 'base', 'self')}
                conn.send(('tick', out))
        except Exception as e:                                        # noqa: BLE001
            conn.send(('error', repr(e)[:300]))


def main():
    out = sys.argv[1]
    ctx = mp.get_context('fork')
    pc, cc = ctx.Pipe()
    wp = ctx.Process(target=worker, args=(cc,), daemon=True)
    wp.start()                                                       # 先 fork（之後才匯入 rclpy）
    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Float64MultiArray
    if not pc.poll(120.0):
        raise SystemExit('worker 未就緒')
    _, W0 = pc.recv()
    co = W0['coarse']
    yaw_l, d_c, n_c = co['yaw'], np.array(co['d']), np.array(co['n'])
    rclpy.init()

    class M(Node):
        def __init__(self):
            super().__init__('far_merge', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.owner, self.arm, self.stop = 'NAV', 'HOLD', False
            self.ev, self.rows, self.sel_log = [], [], []
            self.truth, self.rel, self.rel_t = None, np.zeros(3), None
            self.last = dict(NAV=np.zeros(3), MERGE=np.zeros(3))
            self.p_nav = self.create_publisher(Twist, '/cmd_vel_nav', 10)
            self.p_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
            self.p_plan = self.create_publisher(Path, '/merge/plan', 10)
            self.p_arm = self.create_publisher(Float64MultiArray, '/far/arm_cmd', 10)
            self.create_subscription(Twist, '/cmd_vel_nav_raw', lambda m: self._in('NAV', m), 20)
            self.create_subscription(Twist, '/cmd_vel_merge', lambda m: self._in('MERGE', m), 20)
            self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
            self.create_subscription(Float64MultiArray, '/far/truth', self._truth, 20)
            self.plan, self.plan_req_t, self.plan_ready_t, self.pending = None, None, None, None
            self.gm, self.t_goal, self.t_prep0, self.s_prep0, self.u_arm, self.u_arm_prev = None, None, None, None, np.zeros(6), np.zeros(6)
            self.q_prev, self.t_stopcheck, self.n_fwd, self.no_reach = None, None, dict(NAV=0, MERGE=0, dropped=0), None
            self.wait_tick = False
            self.n_goal = 1
            self.timer = self.create_timer(DT, self.tick)

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def event(self, what, **kw):
            self.ev.append(dict(t=round(self.now(), 3), what=what, **kw))
            self.get_logger().info(f'[far] {what} {kw}')

        def _in(self, src, m):
            v = np.array([m.linear.x, m.linear.y, m.angular.z])
            self.last[src] = v
            if self.stop:
                self.p_nav.publish(Twist()); self.n_fwd['dropped'] += 1
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
            ov = dict(plan_topic='/merge/plan', cmd_vel_topic='/cmd_vel_merge', v_nominal=V_PASS, heading_fixed=float(yaw_l), goal_tolerance_xy=0.03, heading_enable=True)
            json.dump(dict(base='現場 dump 的 /gmpc_controller 參數（逐項相同）', task_reference_overrides=ov, note='只覆寫任務參考；CBF、限制、權重、障礙來源未改'), open(os.path.join(side, 'merge_gmpc_overrides.json'), 'w'), ensure_ascii=False, indent=1)
            cmd = ['ros2', 'run', 'ammr_wholebody_mpc', 'gmpc_node', '--ros-args', '-r', '__node:=gmpc_merge', '--params-file', pf]
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

        def pub_merge_plan(self, q, pg):
            p0, dv = np.array(pg['p0']), np.array(pg['d'])
            al0 = float(-((q[:2] - p0) @ dv))
            pts = [q[:2].copy()]
            entry = p0 - dv * al0                                      # 垂直切入通過線（沿線位置不變），保留手臂準備所需的沿線距離
            n_in = max(int(np.linalg.norm(entry - q[:2]) / 0.005), 2)     # 路徑點距 5 mm：進線 GMPC 以最近路徑點為參考起點（ref_proj=vertex），點距粗（5 cm）時低速下會卡在兩點之間（第一、三次起跑實測）
            for u_ in np.linspace(1.0 / n_in, 1.0, n_in):
                pts.append(q[:2] + u_ * (entry - q[:2]))
            for al in np.arange(al0 - 0.005, -0.06 - 1e-9, -0.005):
                pts.append(p0 - dv * al)
            m = Path(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
            for x in pts:
                ps = PoseStamped(); ps.header = m.header
                ps.pose.position.x, ps.pose.position.y = float(x[0]), float(x[1])
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(pg['yaw'] / 2), math.cos(pg['yaw'] / 2)
                m.poses.append(ps)
            self.p_plan.publish(m)
            return dict(n=len(pts), along0=round(al0, 3), end=[round(float(v), 3) for v in pts[-1]])

        def tick(self):
            tr = self.truth
            if tr is None:
                return
            t, q = self.now(), tr['q']
            if self.t_goal is None:                                   # 起點：立即送粗目標（只用把手估計的幾何，不等搜尋）；背景開始搜尋
                self.coarse_goal = np.array(co['p0']) - d_c * 0.85
                self.pub_goal(self.coarse_goal); self.t_goal = t
                pc.send(('plan', dict(q=q.tolist(), budget=8.0))); self.plan_req_t = t
                self.event('coarse_goal_sent', goal=[round(float(v), 3) for v in self.coarse_goal], note='背景搜尋已送出')
                self.start_merge_gmpc()
                return
            if self.owner == 'NAV' and self.n_goal < 3 and t - self.t_goal >= 1.0 * self.n_goal:   # 目標重送 3 次（剛建立的發布者可能尚未被訂閱端發現；第二次起跑曾因此整趟沒動）
                self.pub_goal(self.coarse_goal); self.n_goal += 1
            if self.plan is None and self.plan_ready_t is None and pc.poll(0):
                tag, res = pc.recv()
                self.plan_ready_t = t
                if tag == 'plan' and res['ok']:
                    self.plan = res
                    self.event('plan_ready', wall_s=res['wall_s'], sim_elapsed_s=round(t - self.plan_req_t, 2), n_cells=res['select'].get('n_cells'), standoff=res['pg']['standoff'], along_offset=res['pg']['along'], prep_time_s=round(res['br']['prep_time_s'], 2), sigma_min=round(res['br']['sigma_min'], 4))
                else:
                    self.event('plan_failed', res=str(res)[:300])
            pg = None if self.plan is None else self.plan['pg']
            br = None if self.plan is None else self.plan['br']
            along = lat = None
            if pg is not None:
                r = q[:2] - np.array(pg['p0'])
                along, lat = float(-(r @ np.array(pg['d']))), float(r @ np.array(pg['n']))
            v_meas = np.zeros(2) if self.q_prev is None else (q[:2] - self.q_prev[1][:2]) / max(tr['t'] - self.q_prev[0], 1e-6)
            yaw_e = math.degrees((q[2] - yaw_l + math.pi) % (2 * math.pi) - math.pi)
            # ---- NAV → MERGE（捕獲區、移動中）
            if self.owner == 'NAV' and not self.stop and pg is not None and CAP_ALONG[0] <= along <= CAP_ALONG[1] and abs(lat) <= CAP_LAT:
                info = self.pub_merge_plan(q, pg)
                jump = float(np.abs(self.last['MERGE'] - self.last['NAV']).max())
                self.owner = 'MERGE'
                self.event('base_owner_NAV_to_MERGE', along=round(along, 3), lat=round(lat, 3), speed=round(float(np.linalg.norm(v_meas)), 3), merge_plan=info, cmd_jump_at_switch=round(jump, 4), note='切換時進線 GMPC 的上一命令為零；差額由原平滑器處理')
            if self.owner == 'NAV' and self.plan is None and self.no_reach is None and float(np.linalg.norm(q[:2] - self.coarse_goal)) < 0.35:
                self.no_reach = 'plan_not_ready_at_capture'
                self.event('no_reach', why=self.no_reach)
            # ---- 手臂
            ua = np.zeros(6)
            if self.owner == 'MERGE' and not self.stop and self.arm == 'HOLD' and self.no_reach is None and abs(lat) < 0.03 and abs(yaw_e) < 3.0:
                need = br['alongs'][1] + br['prep_time_s'] * V_PASS + PREP_MARGIN
                if along >= need:
                    self.arm, self.t_prep0, self.s_prep0 = 'PREP', t, tr['s_end'].copy()
                    self.event('arm_owner_HOLD_to_TRACK_prep', along=round(along, 3), need_along=round(need, 3))
                else:
                    self.no_reach = 'on_line_too_late_for_prep'
                    self.event('no_reach', why=self.no_reach, along=round(along, 3), need_along=round(need, 3))
            res = {}
            if self.arm in ('PREP', 'TRACK') and not self.stop:
                judge = self.arm == 'TRACK' and -0.05 <= along <= 0.25
                pc.send(('tick', dict(mode=self.arm, t=t, q=q.tolist(), pipe=tr['pipe'].tolist(), rel=self.rel.tolist(), s=tr['s'].tolist(), s_end=tr['s_end'].tolist(), judge=judge, t_img=self.t_goal, v_meas=v_meas.tolist(),
                                      u_last=np.r_[self.rel, self.u_arm].tolist(), u_prev=np.r_[self.rel, self.u_arm_prev].tolist())))
                if pc.poll(0.045):
                    tag, res = pc.recv()
                    if tag != 'tick':
                        self.event('worker_error', res=str(res)[:200]); res = {}
                else:
                    self.event('worker_late'); res = {}
                    self.wait_tick = True
            if self.arm == 'PREP' and not self.stop:
                qs = np.array(br['q_start_op']); dq = qs - self.s_prep0
                Tp = max(float(np.abs(dq).max()) / 0.5, DT); u_ = min((t + DT - self.t_prep0) / Tp, 1.0)
                s_des = self.s_prep0 + (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5) * dq
                ua = np.clip((s_des - tr['s_end']) / DT, -1.0, 1.0)
                dcs = res.get('decision')
                if dcs is not None and dcs[0] == 'start':
                    self.arm = 'TRACK'; self.event('arm_track_start', **dcs[1])
                elif dcs is not None and dcs[0] == 'refuse':
                    self.stop = True; self.event('refuse_enter_reach_zone_global_stop', **dcs[1])
            elif self.arm == 'TRACK' and not self.stop:
                tk = res.get('track')
                if tk is None:
                    ua = np.zeros(6)
                elif tk['ok']:
                    ua = np.array(tk['u'])
                else:
                    self.stop = True; self.event('tracker_fail_global_stop', why=tk['why'])
            if self.stop:
                ua = np.zeros(6); self.p_nav.publish(Twist())
            self.u_arm_prev, self.u_arm = self.u_arm, ua
            self.p_arm.publish(Float64MultiArray(data=[float(x) for x in ua]))
            self.q_prev = (tr['t'], q.copy())
            tk = res.get('track') or {}
            cl = res.get('clear') or {}
            self.rows.append([t, tr['t'], *q, *tr['s'], *self.rel, *ua, *self.last['NAV'], *self.last['MERGE'], dict(NAV=0, MERGE=1)[self.owner], dict(HOLD=0, PREP=1, TRACK=2)[self.arm], float(self.stop),
                              np.nan if along is None else along, np.nan if lat is None else lat, yaw_e, tk.get('err') if tk.get('err') is not None else np.nan, tk.get('sigma') if tk.get('sigma') is not None else np.nan, float(bool(tk.get('clipped'))),
                              cl.get('env', np.nan), cl.get('base', np.nan), cl.get('self', np.nan), *v_meas])
            if res.get('judge') is not None:
                self.sel_log.append(dict(t=round(t, 2), along=round(along, 4), lat_mm=round(lat * 1e3, 2), judge=res['judge']))
            # ---- 結束條件：進線後底盤靜止 2 s
            moving = float(np.linalg.norm(v_meas)) > 1e-3 or float(np.abs(self.rel).max()) > 1e-4
            if (self.owner == 'MERGE' or self.stop) and not moving:
                self.t_stopcheck = self.t_stopcheck or t
                if t - self.t_stopcheck > 2.5:
                    self.event('finished_base_stopped', along=None if along is None else round(along, 4), lat_mm=None if lat is None else round(lat * 1e3, 2))
                    raise SystemExit(0)
            else:
                self.t_stopcheck = None

        def save(self):
            cols = ['t', 't_world'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['rel_x', 'rel_y', 'rel_w'] + [f'ua{i}' for i in range(6)] + ['nav_x', 'nav_y', 'nav_w', 'mrg_x', 'mrg_y', 'mrg_w', 'base_owner(0NAV,1MERGE)', 'arm(0HOLD,1PREP,2TRACK)', 'stop',
                                                                                                           'along', 'lat', 'yaw_err_deg', 'arm_track_err', 'sigma', 'clipped', 'clear_env', 'clear_base', 'clear_self', 'vx_meas', 'vy_meas']
            np.savez_compressed(out + '_merge.npz', R=np.array(self.rows, float), cols=np.array(cols))
            json.dump(dict(events=self.ev, plan=self.plan, judge=self.sel_log, forwarded=self.n_fwd, estimate=W0['est'], coarse=co, no_reach=self.no_reach, final_owner=dict(base=self.owner, arm=self.arm, stop=self.stop)), open(out + '_merge.json', 'w'), ensure_ascii=False, indent=1, default=str)

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
