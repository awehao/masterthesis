src = open('evaluation/isaac_m8_house_door_motm.py').read()
s = src
def rep(a, b, c=1):
    global s
    assert s.count(a) == c, (s.count(a), a[:100]); s = s.replace(a, b)

# ---- H1 參數
rep("a = ap.parse_args()\n", """ap.add_argument('--start', default='1.5,-2.8,1.5708')
ap.add_argument('--arm', default='-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018')
ap.add_argument('--far-limit-s', type=float, default=150.0)
ap.add_argument('--cam-hz', type=float, default=5.0)
a = ap.parse_args()
a.arm_friction_model = 'no_passive_arm_friction'                    # 整合趟自載入起統一零被動手臂摩擦（Codex 20261116）；設定並後端讀回，不在交接時切模型
if a.smoke or a.probe or a.diag:
    raise SystemExit('遠距整合執行器不支援 --smoke／--probe／--diag')
""")
# ---- H2 ROS 匯入（專案模組之前）與還原
rep("import door_wbopt as WB                                          # noqa: E402\n", """import rclpy                                                     # noqa: E402  先匯入真正的 rclpy；專案模組（ros_stub）之後會把它換成替身，匯入完再還原
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy              # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String         # noqa: E402
from tf2_msgs.msg import TFMessage                               # noqa: E402
from builtin_interfaces.msg import Time as RosTime               # noqa: E402
_RCLPY_REAL = rclpy
_ROS_PREFIX = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces')
_ROS_SNAP = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in _ROS_PREFIX}
import door_wbopt as WB                                          # noqa: E402
""")
rep("from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402\n", """from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
import door_far_link as FARL                                     # noqa: E402
import far_chain_align as FCA                                    # noqa: E402
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in _ROS_PREFIX and k_ not in _ROS_SNAP]:
    sys.modules.pop(k_, None)
sys.modules.update(_ROS_SNAP)
if sys.modules.get('rclpy') is not _RCLPY_REAL or not hasattr(rclpy, 'init'):
    raise SystemExit('rclpy 還原失敗 ⇒ 拒跑')
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SCAN_N, SCAN_MIN, SCAN_MAX, RANGE_MIN, RANGE_MAX = 360, -3.14159, 3.14159, 0.12, 10.0
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
LIDAR_Z = float(json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))['lidar_z'])
GRASP_END_ABS = -0.285                                               # 抓持段絕對終點（通過線沿線座標；同監督節點）：抓取位可行下界 −0.30 留停止行程 0.015
EST_BOUND_M, EST_BOUND_DEG, SYNC_M = 0.00285, 0.885, 0.05 * 0.011    # 鎖定預算項（同監督節點 TERM；far_lock_error_envelope.json 的已觀測包絡，工程預算）
CMD_TIMEOUT_S, ARM_CMD_TIMEOUT_S = 0.5, 0.2


def T3(p):
    c_, s_ = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c_, -s_], [s_, c_]]; T[:2, 3] = p[:2]
    return T


class Bridge(Node):
    \"\"\"遠距前段與安全鏈的 ROS 介面（同 isaac_m8_house_far_capture.py）＋交接後底盤提案 /cmd_vel_grasp。\"\"\"

    def __init__(self):
        super().__init__('far_door_sim')
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.p_state = self.create_publisher(String, '/far/exec_state', 10)
        self.p_grasp = self.create_publisher(Twist, '/cmd_vel_grasp', 10)
        self.p_hoack = self.create_publisher(String, '/far/handover_ack', 10)
        self.rx_steps, self.ho_tried = [], False
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 20)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(Bool, '/far/done', self._done, 10)
        self.create_subscription(String, '/far/abort', self._abort, 10)
        self.create_subscription(Float64MultiArray, '/far/handover', self._ho, 10)
        self.t, self.n = 0.0, 0
        self.cmd, self.cmd_t, self.acmd, self.acmd_t, self.done, self.abort, self.ho = np.zeros(3), -1.0, np.zeros(6), -1.0, False, None, None
        self.ca = None                                               # 交接後：ChainAlign

    def _cmd(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z], float)
        if np.isfinite(v).all():
            self.cmd, self.cmd_t = v, self.t
            self.rx_steps.append(self.n)
            if self.ca is not None:
                self.ca.on_release(self.n, v)

    def _acmd(self, m):
        if len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t

    def _done(self, m):
        self.done = self.done or bool(m.data)

    def _abort(self, m):
        if self.abort is None:
            self.abort = str(m.data)

    def _ho(self, m):
        if self.ho is None and len(m.data) == 27 and np.isfinite(m.data).all():
            self.ho = np.array(m.data, float)

    def stamp(self):
        s_ = RosTime()
        s_.sec = int(self.t)
        s_.nanosec = int(round((self.t - int(self.t)) * 1e9))
        if s_.nanosec >= 1000000000:
            s_.sec, s_.nanosec = s_.sec + 1, s_.nanosec - 1000000000
        return s_
""")
# ---- M1／M2 起點
rep("    b_start = np.r_[q_pass[0] + 0.03 + L_LEAD, q_pass[1], q_pass[2]]\n", "    b_start = np.array([float(x) for x in a.start.split(',')], float)   # 遠距起點（同 far_capture 三趟）\n")
rep("    qa = ik(b_start, T_at(T_true0, PG.S_PRE), E4[8:14])\n", "    qa = np.array([float(x) for x in a.arm.split(',')], float)          # 遠距起始手臂構型（online_1 far_live 構型）\n")
# ---- M3 靜置檢查：基線是門旁起始構型專用
rep("        ok_r = bool(np.isfinite(dr).all() and float(np.abs(dr).max()) <= HOLD_DR_MAX)\n", "        hold_chk['baseline_note'] = '凍結基線 r0 是門旁起始構型的值；本趟起始構型不同 ⇒ r−r0 只記錄，判定改用絕對殘差 |r|≤HOLD_R_MAX（原 legacy 分支的工程判準）'\n        ok_r = bool(np.isfinite(r_hold).all() and float(np.abs(r_hold).max()) <= HOLD_R_MAX)\n")
# ---- M4 遠距前段
rep("    # ---- 理想感知影格\n", r'''    # ================= 遠距前段（ROS：原導航鏈＋監督節點；本行程只提供感測與受控對象）=================
    eef = [b for b in bodies if nm(b) == 'link_eef'][0]
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    cam = Camera(prim_path=f'{eef}/wrist_cam', resolution=CAM['res'], frequency=CAM['hz'])
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.set_horizontal_aperture(CAM['aperture'])
    cam.set_focal_length(CAM['aperture'] / (2.0 * math.tan(math.radians(CAM['hfov_deg']) / 2.0)))
    cam.set_clipping_range(*CAM['clip'])
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    for _ in range(5):
        apply(); base_cmd(np.zeros(3), 0.0); world.step(render=True)
    Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    def scan(q_):
        org, out_ = [float(q_[0]), float(q_[1]), LIDAR_Z], []
        for i_ in range(SCAN_N):
            th = float(q_[2]) + SCAN_MIN + i_ * SCAN_INC
            best = [float('inf')]

            def rep_(hit, best=best):
                coll = str(getattr(hit, 'collision', '') or '')
                body = str(getattr(hit, 'rigid_body', '') or getattr(hit, 'rigidBody', '') or '')
                if coll.startswith(ROBOT) or body.startswith(ROBOT):
                    return True
                d_ = float(getattr(hit, 'distance', 0.0))
                if RANGE_MIN <= d_ < best[0]:
                    best[0] = d_
                return True
            query.raycast_all(org, [math.cos(th), math.sin(th), 0.0], RANGE_MAX, rep_)
            out_.append(best[0])
        return out_
    rclpy.init()
    node = Bridge()
    obs = FARL.HandleObs()
    T0 = float(world.current_time)
    fdir = os.path.join(a.out, 'frames'); os.makedirs(fdir, exist_ok=True)
    hist, fmeta, far_rows, far_ev = {}, [], [], []
    last_rf, t_next_cam, seq = [None], 0.0, 0
    s_far = s0.copy()
    pipe_far = [np.zeros(9) for _ in range(G.D_STEPS)]
    scan_min = float('nan')
    grav_far = lambda q9_: BM.arm_gravity(K, q9_, bodies_arm)      # noqa: E731
    G_rest = grav_far(qs)

    def ros_step(q_, jp_, s_now, pipe_now, state):
        """每物理步：/clock；每 2 步：里程、關節、/far/truth、狀態；每 10 步：掃描。之後處理 ROS 回呼。"""
        ck = Clock(); ck.clock = node.stamp(); node.p_clock.publish(ck)
        if node.n % 2 == 0:
            m_ = Odometry(); m_.header.stamp = node.stamp(); m_.header.frame_id, m_.child_frame_id = 'odom', 'base_footprint'
            m_.pose.pose.position.x, m_.pose.pose.position.y = float(q_[0]), float(q_[1])
            m_.pose.pose.orientation.w, m_.pose.pose.orientation.z = math.cos(q_[2] / 2), math.sin(q_[2] / 2)
            node.p_odom.publish(m_)
            js = JointState(); js.header.stamp = node.stamp(); js.name = ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in q_[3:]] + [float(jp_[f1]), float(jp_[f2])]
            node.p_js.publish(js)
            s_end = s_now + DT * sum(p_[3:] for p_ in pipe_now)
            node.p_truth.publish(Float64MultiArray(data=[node.t, *[float(x) for x in q_], *[float(x) for x in s_now], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(pipe_now)]]))
            node.p_state.publish(String(data=state))
        if node.n % 10 == 0:
            r_ = scan(q_)
            m_ = LaserScan(); m_.header.stamp = node.stamp(); m_.header.frame_id = 'lidar_link'
            m_.angle_min, m_.angle_max, m_.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
            m_.range_min, m_.range_max = RANGE_MIN, RANGE_MAX
            m_.ranges = [float(x) if math.isfinite(x) else float('inf') for x in r_]
            node.p_scan.publish(m_)
        for _ in range(4):
            rclpy.spin_once(node, timeout_sec=0.0)

    def far_event(what, **kw):
        far_ev.append(dict(kw, t=round(node.t, 3), what=what))
        print(f'[motm] far t={node.t:.2f} {what} {json.dumps(clean(kw), ensure_ascii=False)[:300]}', flush=True)
    far_out = None
    HOX = {}

    def est_fn_lock(t_):
        return HOX['T_lock'], 1, HOX['lock']['t_obs_rel']                 # 鎖定任務參考（原始時間戳；不更新）

    def lock_budget(t_, q9_, ep_, er_):
        lk_ = HOX['lock']
        pos = max(EST_BOUND_M, lk_['spread']) + SYNC_M + float(ep_)      # 估計包絡（工程預算）＋同步＋手臂 TCP 實測；理想里程分支里程項為零
        ang = EST_BOUND_DEG + lk_['dang'] + float(er_)
        return dict(pos_m=round(pos, 5), ang_deg=round(ang, 3), ok=bool(pos <= FARL.LOCK_BUDGET['pos_m'] and ang <= FARL.LOCK_BUDGET['ang_deg']), terms=dict(est_m=max(EST_BOUND_M, lk_['spread']), sync_m=SYNC_M, odom_m=0.0, arm_tcp_m=round(float(ep_), 5)))
    w_far0 = time.perf_counter()
    while node.t < a.far_limit_s:
        t_sim = float(world.current_time)
        node.t = t_sim - T0
        q, jp = q9()
        pos_r, q_r = cam.get_world_pose(camera_axes='ros')
        Tr = np.eye(4); Tr[:3, :3] = W0.wxyz_to_R(np.asarray(q_r, float)); Tr[:3, 3] = np.asarray(pos_r, float)
        hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), T_render=Tr)
        if len(hist) > 300:
            for kk in sorted(hist)[:len(hist) - 300]:
                hist.pop(kk, None)
        oc_, ow_ = other_contacts()
        fc_ = finger_contacts()
        cmax = max(oc_, fc_[0][0], fc_[0][1], fc_[1][0], fc_[1][1])
        ddeg = abs(math.degrees(door_q() - door0))
        if not (np.isfinite(q).all() and math.isfinite(cmax)) or cmax > 0.05 or ddeg > 0.05:   # 遠距前段保護：任何接觸／門角變化 ⇒ 模擬工程中止（不再步進）
            far_out = dict(result='far_engineering_abort', why='contact_or_door_or_nonfinite', contact_n=cmax, who=ow_, door_deg=ddeg)
            far_event('latched_sim_engineering_abort', **far_out)
            node.p_state.publish(String(data='latched:' + far_out['why']))
            break
        ros_step(q, jp, s_far, pipe_far, 'run')
        if node.abort is not None:
            far_out = dict(result='far_engineering_abort', why='supervisor_abort', detail=node.abort[:200])
            far_event('latched_sim_engineering_abort', **far_out)
            break
        if node.ho is not None and not node.ho_tried:
            ph_ = FCA.best_phase(node.rx_steps)
            if ph_ is None or node.n % NPH == ph_:                     # 在安全鏈節拍的合適相位接手（名目放行延遲約 6 步，落在 8 步管線內）
                node.ho_tried = True
                HO = node.ho
                HOX.update(T_lock=HO[1:17].reshape(4, 4).copy(), t_far_ho=node.t,
                           lock=dict(t_obs=float(HO[17]), age_at_lock_s=float(HO[18]), dang=float(HO[19]), spread=float(HO[20]), p0=HO[21:23].copy(), d=HO[23:25].copy(), yaw=float(HO[25]), ver=int(HO[26]), t_obs_rel=float(HO[17]) - node.t))
                off_ho = off_static + (grav_far(q) - G_rest) / a.kp    # 靜態偏置：靜置實測錨＋重力隨構型（同 K2 模型）
                snap = dict(s=s_far.copy(), pipe=[u_.copy() for u_ in pipe_far], u_last=pipe_far[-1].copy(), u_prev=pipe_far[-1 - NPH].copy(), off_static=off_ho.copy())
                w_try = time.perf_counter()
                # 以當下快照建立獨立的抓持控制器實例並試算首拍：不發任何命令、不動真管線（全部是複本）。通過 ⇒ 這個實例與它的首拍輸出就是接手後的第一拍（同一狀態、同一步）；不過 ⇒ 丟棄、回 nack、前段所有權不變
                a_to = float(-((q[:2] - HOX['lock']['p0']) @ HOX['lock']['d']))   # 接手當下的沿線位置（通過線真實座標）
                HOX.update(a_to=a_to, win=(L_LEAD, a_to - GRASP_END_ABS))      # 控制器窗口以接手位置為原點 ⇒ 換算成相對量，絕對終點固定在 GRASP_END_ABS
                gc_try = G.MovingGraspCtrl(ident, est_fn_lock, HOX['lock']['d'].copy(), V, HOX['win'], snap['s'].copy(), [u_.copy() for u_ in snap['pipe']], snap['u_last'].copy(), snap['u_prev'].copy(), snap['off_static'].copy(),
                                           sched=dict(hold_s=0.3 + T_LEAD), pool=POOL, fastkin=True, deadline_ms=50.0, lock_budget_fn=lock_budget)
                o_try = gc_try.step(0.0, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))
                ok_try = (a_to - GRASP_END_ABS >= 0.115) and gc_try.stop is None and gc_try.abort is None and not o_try.get('abort') and np.isfinite(np.asarray(o_try['base_twist'], float)).all() and np.isfinite(np.asarray(o_try['arm_target'], float)).all()
                trial = dict(along_at_takeover=round(a_to, 4), window_rel=[float(x_) for x_ in HOX['win']], ok=bool(ok_try), stop=gc_try.stop, abort=gc_try.abort, wall_ms=round((time.perf_counter() - w_try) * 1e3, 1), phase=ph_, n=node.n, rx_mod=[x_ % NPH for x_ in node.rx_steps[-8:]], budget=gc_try.lock_budget,
                             first_cmd=[float(x_) for x_ in gc_try.u_last], prev_cmd=[float(x_) for x_ in snap['u_last']], first_cmd_step_max=float(np.abs(gc_try.u_last - snap['u_last']).max()))
                HOX['trial'] = trial
                if ok_try:
                    node.p_hoack.publish(String(data='ack:%d' % HOX['lock']['ver']))
                    HOX.update(gc=gc_try, o_first=o_try, snap=snap, off_ho=off_ho)
                    far_event('handover_trial_passed_takeover', **{k_: trial[k_] for k_ in ('wall_ms', 'phase', 'first_cmd_step_max', 'budget')})
                    break
                node.p_hoack.publish(String(data='nack:%d:%s' % (HOX['lock']['ver'], str((gc_try.stop or gc_try.abort or {}).get('why'))[:60])))
                far_event('handover_trial_failed_nack', **trial)
        if node.done:
            far_out = dict(result='far_ended_without_handover', why='supervisor_done')
            far_event('supervisor_done_without_handover')
            break
        u_in = node.cmd if (node.t - node.cmd_t <= CMD_TIMEOUT_S and node.cmd_t >= 0) else np.zeros(3)
        ua_in = node.acmd if (node.t - node.acmd_t <= ARM_CMD_TIMEOUT_S and node.acmd_t >= 0) else np.zeros(6)
        pipe_far.append(np.r_[u_in, ua_in])
        ue = pipe_far.pop(0)
        s_far = s_far + DT * ue[3:]
        for i, j in zip(idx, range(6)):
            tgt[i] = float(s_far[j])
        tgt[f1] = XO.FINGER_OPEN
        apply()
        base_cmd(ue[:3], float(q[2]))
        fr = cam.get_current_frame()                                  # 取像與偵測（同 far_capture）
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is not None and rf != last_rf[0] and rf != (0, 0) and col is not None and fr.get('distance_to_image_plane') is not None:
            last_rf[0] = rf
            t_r = fr.get('rendering_time')
            hp = None if t_r is None else hist.get(int(round(float(t_r) / DT)))
            if hp is not None and abs(hp['t_sim'] - float(t_r)) <= 0.25 * DT and float(t_r) - T0 >= t_next_cam - 1e-9:
                dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
                dep = dep[:, :, 0] if dep.ndim == 3 else dep
                t_obs = float(t_r) - T0
                t_next_cam = t_obs + 1.0 / a.cam_hz
                seq += 1
                T_fk = K.fk(hp['q'], W0.OPT)
                T_est_, info = obs.intake(dict(dep=dep, K=Kr, T_cam=T_fk, t_obs=t_obs, q=hp['q'], t_state=t_obs, pose_source='fk_at_render_time', frame_id='map', fk_vs_render_m=float(np.linalg.norm(T_fk[:3, 3] - hp['T_render'][:3, 3]))), node.t)
                T_bh = None if T_est_ is None else np.linalg.inv(T3(hp['q'][:3])) @ np.asarray(T_est_, float)
                np.savez_compressed(os.path.join(fdir, f'f{seq:04d}_depth_m.npz'), dep=dep)
                try:
                    import imageio.v2 as imageio
                    imageio.imwrite(os.path.join(fdir, f'f{seq:04d}_rgb.png'), np.asarray(col)[:, :, :3].astype(np.uint8))
                except Exception as e_:                               # noqa: BLE001
                    print(f'[motm] 彩色影像保存失敗 {e_!r}', flush=True)
                Th_ = truth_handle()
                fmeta.append(dict(seq=seq, stem=f'f{seq:04d}', t_obs=t_obs, t_processed=node.t, cam_T_fk=T_fk.tolist(), q=[float(v) for v in hp['q']], odom_base_at_capture=[float(v) for v in hp['q'][:3]], valid=T_est_ is not None, info=info,
                                  T_base_handle=None if T_bh is None else T_bh.tolist(), eval_only=dict(err_pos_mm=None if T_est_ is None else round(float(np.linalg.norm(np.asarray(T_est_)[:3, 3] - Th_[:3, 3])) * 1e3, 2), range_m=round(float(np.linalg.norm(T_fk[:3, 3] - Th_[:3, 3])), 3))))
                node.p_obs.publish(Float64MultiArray(data=[float(seq), t_obs, 1.0 if T_est_ is not None else 0.0, *[float(v) for v in hp['q'][:3]], *([float(v) for v in T_bh.reshape(-1)] if T_bh is not None else [float('nan')] * 16), float(node.t)]))
        world.step(render=True)
        node.n += 1
        far_rows.append([node.t, *q, *s_far, *ue, cmax, ddeg])
    np.savez_compressed(os.path.join(a.out, 'far_log.npz'), L=np.array(far_rows, float), cols=np.array(['t'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['ux', 'uy', 'uw'] + [f'ua{i}' for i in range(6)] + ['contact_n', 'door_deg']))
    json.dump(clean(dict(K=Kr, cam=CAM, T_mount=T_mount, frames=fmeta)), open(os.path.join(a.out, 'frames_meta.json'), 'w'), ensure_ascii=False, indent=1)
    res['far'] = dict(sim_s=round(node.t, 2), wall_s=round(time.perf_counter() - w_far0, 1), events=far_ev, n_frames=len(fmeta), n_valid=sum(1 for f_ in fmeta if f_['valid']), base_travel_m=round(float(np.abs(np.diff(np.array(far_rows)[:, 1:3], axis=0)).sum()), 3) if len(far_rows) > 1 else 0.0,
                      sources=dict(odom='物理位姿直接當里程（理想）', scan='PhysX 射線，略過機器人自身（忽略手臂自遮擋）', delay='8 步在途管線（軟體模型）', arm_friction='no_passive_arm_friction（自載入起）'))
    if HOX.get('gc') is None:
        if far_out is None and HOX.get('trial') is not None:
            far_out = dict(result='far_ended_handover_rejected', why='trial_failed', trial=HOX['trial'])
        if far_out is None:
            far_out = dict(result='far_ended_without_handover', why='far_limit')
        res.update(result=far_out['result'], far_end=far_out, aborted=far_out if 'abort' in far_out['result'] else None)
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print(f"[motm] 遠距前段結束（未交接）：{far_out}", flush=True)
        return 70 if 'abort' in far_out['result'] else 6
    # ---- 已接手：控制器實例＝試跑通過的那一個（設定點、真管線、上一／前一拍命令、偏置皆為接手當下的快照）
    T_lock, lock, t_far_ho = HOX['T_lock'], HOX['lock'], HOX['t_far_ho']
    node.ca = FCA.ChainAlign(G.D_STEPS, NPH, pre_release=node.cmd.copy())   # 接手前最後一筆放行值（前擁有者的命令在最初 8 步內仍在鏈內）
    chain = dict(stop=None)
    res['handover'] = dict(window_rel=HOX['win'], along_at_takeover=HOX['a_to'], grasp_end_abs=GRASP_END_ABS, t_far=round(t_far_ho, 3), base=q[:3].tolist(), lock={k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in lock.items()}, T_lock=T_lock.tolist(), s=HOX['snap']['s'].tolist(), off_static=HOX['off_ho'].tolist(),
                           u_last=HOX['snap']['u_last'].tolist(), u_prev=HOX['snap']['u_prev'].tolist(), pipe=[u_.tolist() for u_ in HOX['snap']['pipe']], trial=HOX['trial'], eval_only=dict(lock_err_mm=round(float(np.linalg.norm(T_lock[:3, 3] - truth_handle()[:3, 3])) * 1e3, 2)))
    # ---- 把手估計來源＝鎖定任務參考（原始時間戳；不更新）
''')
rep("""    def capture(t):
        if not frames or t >= frames[-1][0] + 1.0 / 30.0 - 1e-9:
            frames.append((round(t, 4), truth_handle()))
            if len(frames) > 200:
                del frames[:100]

    def est_fn(t):
        for tc, T in reversed(frames):
            if tc <= t + 1e-9:
                return T, tc, tc
        return None
""", """    def capture(t):
        return None

    def est_fn(t):
        return est_fn_lock(t)
""")
rep("gc = G.MovingGraspCtrl(ident, est_fn, np.array([-1.0, 0.0]), V, (L_LEAD, L_LEAD + 0.22), s0.copy(), [np.zeros(9) for _ in range(G.D_STEPS)], np.zeros(9), np.zeros(9), off_static,\n                           sched=dict(hold_s=0.3 + T_LEAD), pool=POOL, fastkin=True, deadline_ms=50.0)",
    "gc = HOX['gc']                                                    # 試跑通過並已接手的控制器實例（首拍已算，輸出在 HOX['o_first']）")
rep("    box = dict(mode='grasp', ex=None,", "    box = dict(o_first=HOX['o_first'], mode='grasp', ex=None,")
rep("        tick = gc.n_step % NPH == 0\n        # P2 首段", "        first_it = box.get('o_first') is not None\n        tick = first_it or gc.n_step % NPH == 0\n        # P2 首段")
# ---- M8 抓持段：底盤經安全鏈
rep("""        o = gc.step(t, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))   # r12：兩指各自實測開度供碰撞幾何
""", """        o = box.pop('o_first') if first_it else gc.step(t, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))   # 首拍＝接手試跑的輸出（同一狀態、同一步）
        node.t = float(world.current_time) - T0
        if tick:                                                     # A2(b)：底盤提案走 選擇器→平滑器→Shield→GUARD（GUARD 唯一出口）
            node.ca.propose(node.n, gc.u_last[:3])
            mt_ = Twist(); mt_.linear.x, mt_.linear.y, mt_.angular.z = (float(x_) for x_ in gc.u_last[:3]); node.p_grasp.publish(mt_)
        ros_step(q, jp, gc.s, gc.pipe, 'grasp')
        if node.abort is not None:                                   # 全域中止貫穿交接後：收完回呼、下一次物理步之前檢查；觸發即保存資料、不再步進
            box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（監督節點 /far/abort；不再步進，不是受控停止）', detail=node.abort[:200])
            log_event('supervisor_abort_engineering_abort', detail=node.abort[:200])
            break
        ckc = node.ca.check(node.n)                                  # 本步有提案生效：放行須與控制器核過的命令一致；缺失／不一致 ⇒ 在它生效前拒絕＝模擬工程中止（控制器的停止證書以它的管線為前提，管線已不成立）
        if not ckc['ok']:
            chain['stop'] = dict(t=round(t, 3), phase='grasp', **ckc)
            box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（安全鏈放行與已核命令不一致；不再步進，不是受控停止）', chain=ckc)
            log_event('safety_chain_mismatch_engineering_abort', **ckc)
            node.p_state.publish(String(data='latched:safety_chain_' + ckc['why']))
            break
""")
rep("""        diag_B()
        world.step(render=True)
        t += DT
        diag_C(t, 0, o['base_twist'], gc.pipe, gc.u_last)""", """        diag_B()
        world.step(render=True)
        node.n += 1
        t += DT
        diag_C(t, 0, o['base_twist'], gc.pipe, gc.u_last)""")
# ---- M9 開門段
rep("""            out = ex.step(tr, q, phm, fn_, w_ext=w_ext_)
""", """            out = ex.step(tr, q, phm, fn_, w_ext=w_ext_)
            node.t = float(world.current_time) - T0
            if tick:
                node.ca.propose(node.n, ex.u_last[:3])
                mt_ = Twist(); mt_.linear.x, mt_.linear.y, mt_.angular.z = (float(x_) for x_ in ex.u_last[:3]); node.p_grasp.publish(mt_)
            ros_step(q, jp, ex.s, ex.pipe, 'open')
            if node.abort is not None:                                   # 全域中止貫穿交接後：收完回呼、下一次物理步之前檢查；觸發即保存資料、不再步進
                box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（監督節點 /far/abort；不再步進，不是受控停止）', detail=node.abort[:200])
                log_event('supervisor_abort_engineering_abort', detail=node.abort[:200])
                break
            ckc = node.ca.check(node.n)
            if not ckc['ok']:
                chain['stop'] = dict(t=round(t, 3), phase='open', **ckc)
                box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（安全鏈放行與已核命令不一致；不再步進，不是受控停止）', chain=ckc)
                log_event('safety_chain_mismatch_engineering_abort', **ckc)
                node.p_state.publish(String(data='latched:safety_chain_' + ckc['why']))
                break
""")
rep("""            diag_B()
            world.step(render=True)
            t += DT
""", """            diag_B()
            world.step(render=True)
            node.n += 1
            t += DT
""")
rep("    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)\n    print(f\"[motm] 完成 result=", "    res['safety_chain'] = dict(node.ca.report(), stop=chain['stop'], note='抓持／開門段底盤提案經 選擇器→平滑器→Shield→GUARD；每筆提案在生效步核對放行與提案一致才執行；缺失／不一致 ⇒ 生效前模擬工程中止')\n    res['lock_budget_max'] = gc.lock_budget_max\n    try:\n        np.savez_compressed(os.path.join(a.out, 'chain_stream.npz'), prop_n=np.array([p_['n'] for p_ in node.ca.props]), prop_u=np.array([p_['u'] for p_ in node.ca.props]), prop_conf=np.array([-1 if p_['n_conf'] is None else p_['n_conf'] for p_ in node.ca.props]), rel_n=np.array([r_[0] for r_ in node.ca.releases]), rel_v=np.array([r_[1] for r_ in node.ca.releases]))\n    except Exception as e_:                                        # noqa: BLE001\n        print('[motm] chain_stream 存檔失敗', repr(e_), flush=True)\n    try:\n        node.p_state.publish(String(data='done')); rclpy.spin_once(node, timeout_sec=0.0)\n    except Exception:                                              # noqa: BLE001\n        pass\n    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)\n    print(f\"[motm] 完成 result=")
# ---- 離開碼寫檔（sim_app.close() 會以 0 結束行程）
rep("    sim_app.close()\n    sys.exit(rc)", "    try:\n        with open(os.path.join(a.out, 'exit_code.txt'), 'w') as fh:\n            fh.write(str(int(rc)))\n            fh.flush(); os.fsync(fh.fileno())\n    except Exception:                                              # noqa: BLE001\n        pass\n    print(f'[motm] exit_code {rc}', flush=True)\n    sim_app.close()\n    sys.exit(rc)")
rep("    def truth_handle():                                            # 理想感知的來源（也供評估）", "    def truth_handle():                                            # 把手 prim 真值：BindGate 實際抓持條件與事後評估用（本檔不作把手估計來源）")
rep("""               scope='理想感知（把手 prim 真值 30 Hz）＋理想抓持（FixedJoint）；非即時（控制計算期間物理不步進）；範圍：開到 85° 停穩為止（不解綁、不退出、不穿門）',
               init=dict(pass_line_from='full_2 綁定底盤位姿 y／偏航（場景初始條件）', b_start=b_start.tolist(), lead_m=L_LEAD, lead_s=T_LEAD, v_pass=V, q_arm_pregrasp=qa.tolist(),
                         note='起點瞬移到通過線上窗口前 0.01 m、手臂預抓構型（抓取同側分支），靜置後從靜止起步；起跑後不瞬移。前導受 x≈0.80 m 腕部奇異限制（冒煙 2：0.15 m 前導在 1.9 s 手臂速率 1.61 rad/s 超限停止）'),""",
"""               scope='遠距整合：把手來源＝腕部相機實際影像（凍結偵測＋幾何估計）；通過線＝途中第一格有效且通過關聯的估計；抓持與開門用鎖定任務參考（原始時間戳，不更新）；BindGate 的實際抓持條件以把手 prim 真值核（同門旁各趟）；理想抓持（FixedJoint）；理想里程；非即時；範圍：開到 85° 停穩為止（不解綁、不退出、不穿門）',
               init=dict(b_start=b_start.tolist(), q_arm_start=qa.tolist(), source='遠距起點（同 far_capture 三趟）；手臂＝online_1 far_live 構型；靜置後從靜止起步，起跑後不瞬移', v_pass=V,
                         note='通過線不是場景初始條件：由本趟途中影像估計在線決定（見 ros_merge.json anchor）'),""")
hdr = '"""遠距整合執行器（Codex 20261116 A1／A2(b)／A3）：isaac_m8_house_door_motm.py 的複本，前面接遠距前段。\n  遠距前段：本行程只提供感測與受控對象（ROS：/clock、/odom_raw、/scan_raw、/joint_states、/far/truth、/far/handle_obs；收 /cmd_vel、/far/arm_cmd），原導航鏈與監督節點（merge_node.py fullho）在外。\n  交接：監督節點交接判定通過發 /far/handover（鎖定參考與通過線）；本行程在下一個物理步邊界建立 MovingGraspCtrl r14（鎖定任務參考模式），設定點／真管線／上一命令／偏置原樣承接。\n  交接後：MovingGraspCtrl → BindGate → 理想綁定 → DoorOpenExec（同門旁）；底盤提案發 /cmd_vel_grasp 經 選擇器→平滑器→Shield→GUARD，放行值照控制器生效時刻套用（far_chain_align）。\n  自載入起 no_passive_arm_friction（後端讀回）。\n  以下為門旁原檔說明——其中「理想感知（把手 prim 真值 30 Hz）」「底盤瞬移到通過線」兩段不適用本檔：本檔把手來源是腕部相機實際影像＋鎖定任務參考，起點是遠距起點。\n'
assert s.startswith('#!/usr/bin/env python3\n"""')
s = s.replace('#!/usr/bin/env python3\n"""', '#!/usr/bin/env python3\n' + hdr, 1)
open('evaluation/isaac_m8_house_far_door_motm.py', 'w').write(s)
print('written', len(s.split('\n')))
