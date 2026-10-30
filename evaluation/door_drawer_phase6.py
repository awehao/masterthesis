#!/usr/bin/env python3
"""r6（2026-10-30；返程門操作；規格 results/specs/door_return_phase3_r1.md；Codex m6e）：r5 的版本化複本，只加（a.return_phase 未開 ⇒ 與 r5 同）：
  /return/req（String JSON：arm／close〔含 T〕／move／open／retreat／done／hb／abort）→ door_return_gates.ReturnGate（順向准入）；每物理步讀 v3 接觸紀錄（辨識 bar_in／bar_out）；
  手指命令改由准入決定（1 s 斜坡），閂鎖後凍結；中止 ⇒ 命令鏈過期、導航歸零，實際寫入連續 0.2 s 為零才停；發 /return/state（JSON）、/return/sync（t,step,q,qd,x,y,yaw,arm6,f1,f2,qf）。
r5（2026-10-30；mission_dd_6：門段最後一筆 /gripper/cmd（閉）殘留進抽屜段 ⇒ 抽屜導航全程手指閉合，且 r4 把這筆殘留當成「閉爪斜坡完成」（433.5，早於 ALIGN）⇒ 10 s 期限在真正閉爪時（447.7 後）到期、誤中止——我的錯誤）：r4 的版本化複本，只改：
  (1) 入口切換時一併清除殘留夾爪命令（grip_cmd → None ⇒ 手指回預設張開，等抽屜任務自己下命令；原值記入 start.state.cleared）；
  (2) 斜坡完成須在抽屜段先見到張開命令（≥ finger_open − 1e-6）後才首次到全閉才算。
r4（2026-10-30；mission_dd_5 split 手指＋10 mm 桿空夾至模擬上限；Codex m5g）：r3 的版本化複本，只加**有界抓持失敗出口**：
  全身持有時夾爪命令首次到全閉（≤ finger_closed）＝閉爪斜坡完成；其後 a.grasp_evidence_timeout_s（10 s）內接觸保持未達 GRASP_HOLD_S（2 s，＝抽屜任務原定夾持判準）
  ⇒ 記錄、發 /park/violation、以 grasp_no_contact 停止世界（不宣稱受控減速；全身持有、底盤由命令鏈管）。已建立過夾持證據後不再檢查（放手張開不誤判）。
r3（2026-10-30；mission_dd_3 櫃前交棒被拒 802 次「停車靜止閘門尚未通過」：drawer_mission_3 定版為 MOTM，抽屜執行端該分支 park_on＝False（switch_guard None、v_min_lin 0.01、無 /park/gate），我卻無條件套 PARK_HOLD 閘與 v_min_lin 0——配置錯誤）：r2 的版本化複本，只改：
  a.drawer_park_mode：none（預設；＝MOTM 定版：不設停位閘／保持監看、switch_guard None、v_min_lin＝a.v_min_lin、不發 /park/gate）｜hold（r2 行為）。
r2（2026-10-30；mission_dd_1／2 抽屜段導航命令全零的真因：ex.step 漏傳 fingers ⇒ R2 線上包絡監看「手指讀值缺資料」fail-closed、首步即判違反，
之後導航一律減速到零；且違反未處理（無聲）。我的實作錯誤）：r1 的版本化複本，只改：(1) ex.step 傳兩指量測（同門段 T5）；
(2) 包絡違反 ⇒ 記錄、取消交棒、發 /park/violation；**照常寫入 R2 回傳的減速命令**（既有變化率上限減到零），實際寫入連續 0.2 s 為零才結束（Codex m5d）。
r1：門→抽屜同趟串接：抽屜段主迴圈（設計 results/specs/door_drawer_chain_s1_r1.md；Codex m5a 核可 S1 修正版）。
由 isaac_door_k2_sim_ev9r8t.py 在門段主迴圈以 subtask_handoff 結束（控制者已交回導航）後呼叫；**同一 world／robot／命令鏈／執行層**，不重設任何狀態。
主迴圈逐段照 isaac_drawer_room_sim.py（抽屜管線驗證過的那份）：命令入口（/wb_vel_cmd 一般路徑）、交棒請求、導航／減速段逾時、
PARK_HOLD 靜止閘門與保持監看（錨點＝抽屜停位）、套用回報、設定點回報、/gripper/state、/joint_states、/odom、TF、/drawer/*、/handover/state、掃描。
差異（皆照門段現況，不改回抽屜舊配置）：
  - 手指驅動用門段的 apply_pos（主從夾爪：只驅動主指）與 k2_latch.grip（中止閂鎖凍結夾爪）；
  - 收臂姿態＝門段 STOW_F（停位閘的「已收臂」判準用它）；
  - 中止閂鎖：/door/k2_abort 或 /room/stop ⇒ 受控停止；
  - 紀錄另帶 door_q_rad（門扇角，供重播）。
"""
import json
import math
import os

import numpy as np

VERSION = 'door_drawer_phase/r6'
GRASP_HOLD_S = 2.0      # r4：抽屜任務原定夾持判準（兩指各 ≥ contact_min_n 連續 2 s）


def _np(x):
    return np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x, dtype=float)


def make_drawer_contact_fns(finger_v, drawer_pair_v, physics_dt):
    """照 isaac_drawer_room_sim 的 finger_contact_n／drawer_pairs（讀不到回 NaN／None，不當成 0）。"""
    warned = {'fc': False, 'dp': False}

    def finger_contact_n():
        if finger_v is None:
            return float('nan'), float('nan')
        try:
            Mx = np.asarray(finger_v.get_contact_force_matrix(dt=physics_dt))
            f = np.linalg.norm(Mx.reshape(Mx.shape[0], -1, 3).sum(axis=1), axis=1)
            return float(f[0]), float(f[1])
        except Exception as _e:
            if not warned['fc']:
                warned['fc'] = True
                print(f'[drawer] **手指↔抽屜接觸力讀不到**：{_e!r}（之後回 NaN）', flush=True)
            return float('nan'), float('nan')

    def drawer_pairs():
        if drawer_pair_v is None:
            return None, None
        try:
            Fm = _np(drawer_pair_v.get_contact_force_matrix(dt=physics_dt)).reshape(-1, 3)
            fn, pts, nrm, dist, cnt, start = (_np(v) for v in drawer_pair_v.get_contact_force_data(dt=physics_dt))
            ff, fpts, fcnt, fstart = (_np(v) for v in drawer_pair_v.get_friction_data(dt=physics_dt))
            cnt, start = cnt.reshape(-1), start.reshape(-1)
            fcnt, fstart = fcnt.reshape(-1), fstart.reshape(-1)
            out = []
            for j in range(len(cnt)):
                cc, s = int(cnt[j]), int(start[j])
                fc_, fs_ = int(fcnt[j]), int(fstart[j])
                if cc == 0 and fc_ == 0:
                    continue
                w = np.abs(fn[s:s + cc, 0])
                cen = ((w[:, None] * pts[s:s + cc]).sum(0) / w.sum() if cc and w.sum() > 0 else
                       (pts[s:s + cc].mean(0) if cc else np.full(3, np.nan)))
                out.append([j, cc, *[round(float(v), 4) for v in Fm[j]], *[round(float(v), 4) for v in ff[fs_:fs_ + fc_].sum(0)],
                            *[round(float(v), 5) for v in cen], round(float(dist[s:s + cc, 0].min()), 6) if cc else None])
            net = _np(drawer_pair_v.get_net_contact_forces(dt=physics_dt)).reshape(-1)[:3]
            return out, [round(float(v), 4) for v in net]
        except Exception as _e:
            if not warned['dp']:
                warned['dp'] = True
                print(f'[drawer] **抽屜接觸配對讀不到**：{_e!r}（之後記 None）', flush=True)
            return None, None
    return finger_contact_n, drawer_pairs


def run_drawer_phase(c):
    """c：dict（見 isaac_door_k2_sim_ev9r8t.py 建構處）。回傳 rec（已寫 room_run_drawer.json）。"""
    a, world, robot, node, ex, chain = c['a'], c['world'], c['robot'], c['node'], c['ex'], c['chain']
    idx, fidx, ARM, FJ = c['idx'], c['fidx'], c['ARM'], c['FJ']
    xfc, dprim, DY0 = c['xfc'], c['dprim'], c['DY0']
    finger_contact_n, drawer_pairs, pair_names = c['finger_contact_n'], c['drawer_pairs'], c['pair_names']
    AUTH_NAV, AUTH_WHOLEBODY, AUTH_GLIDE = c['AUTH_NAV'], c['AUTH_WHOLEBODY'], c['AUTH_GLIDE']
    ParkCfg, StaticGate, HoldMonitor, PhysicsContactHold = c['ParkCfg'], c['StaticGate'], c['HoldMonitor'], c['PhysicsContactHold']
    M = c['msgs']
    rclpy = c['rclpy']
    yaw_of, ray_range = c['yaw_of'], c['ray_range']
    stow = list(c['stow'])
    apply_pos, grip_latch = c['apply_pos'], c['grip_latch']
    door_q_fn = c['door_q_fn']
    step_id = int(c['step_id0'])
    out_dir = a.out

    # ---- PARK_HOLD：錨點＝抽屜停位（抽屜管線 MOTM 同一配置）----
    _pk = [float(v) for v in a.drawer_park_pose.split(',')]
    park_cfg = ParkCfg(park_x=_pk[0], park_y=_pk[1], park_yaw=_pk[2],
                       hold_s=max(0.5, float(a.max_cmd_age_s)), hold_mode=True)
    park_gate = StaticGate(park_cfg)
    park_on = (getattr(a, 'drawer_park_mode', 'none') == 'hold')                      # r3：MOTM 定版 ⇒ 無停位閘
    park_hold = None
    park_done = not park_on                                                           # none ⇒ 閘門更新與保持監看自然跳過
    park_state = {'violation': None, 'transfer_without_gate': False, 'external': None}
    park_prev_cmd = (float('nan'),) * 3
    park_last_sample = None

    def _park_guard(to, sid, st_):
        if park_state['violation'] is not None:
            return False, 'PARK_FIXED 已違規閂鎖'
        if not park_gate.passed:
            return False, '停車靜止閘門尚未通過'
        if not park_gate.ok_now:
            return False, f'停車條件當前不成立：{park_gate.why_now}'
        if park_gate.last is None or park_gate.last['step'] != int(sid):
            return False, (f'停車閘門沒有本步量測（最後一筆 '
                           f'{None if park_gate.last is None else park_gate.last["step"]}，本步 {sid}）')
        return True, None
    ex_prev = {'switch_guard': getattr(ex, 'switch_guard', None), 'v_min_lin': getattr(ex, 'v_min_lin', None),
               'owner': ex.owner, 'stow_setpoint': list(getattr(ex, 'stow_setpoint', []) or [])}
    ex.switch_guard = _park_guard if park_on else None
    ex.v_min_lin = 0.0 if park_on else float(a.v_min_lin)
    tenure_log = []
    if hasattr(ex, 'on_tenure_open'):                                                  # 門專用權威結束：任期回呼只記錄，不再喚醒門 W2
        ex_prev['r2'] = {'on_tenure_open': repr(ex.on_tenure_open), 'on_tenure_close': repr(ex.on_tenure_close),
                         'nav_ready_fn': repr(ex.nav_ready_fn), 'envelope_kept': ex.envelope is not None}
        ex.on_tenure_open = lambda k, t_: tenure_log.append({'open': k, 't': t_})
        ex.on_tenure_close = lambda k, t_: tenure_log.append({'close': k, 't': t_})

        def _nav_ready_d(t_):                                                          # 抽屜段：nav1 已結束 ⇒ 導航命令新鮮即備妥
            if node.nav_cmd_t is None or t_ - node.nav_cmd_t > 0.25:
                return False, f'導航命令不新鮮（{None if node.nav_cmd_t is None else round(t_ - node.nav_cmd_t, 3)} s）'
            return True, None
        ex.nav_ready_fn = _nav_ready_d
    node.park_ext_vio = None
    if not hasattr(node, 'park_gate_pub'):
        node.park_gate_pub = node.create_publisher(M['String'], '/park/gate', 10)
    if not hasattr(node, 'park_vio_pub'):
        node.park_vio_pub = node.create_publisher(M['String'], '/park/violation', 10)
    node.drawer_state_pub = node.create_publisher(M['Float64MultiArray'], '/drawer/state', 10)
    node.drawer_handle_pub = node.create_publisher(M['Float64MultiArray'], '/drawer/handle_pose', 10)
    node.drawer_sync_pub = node.create_publisher(M['Float64MultiArray'], '/drawer/sync_state', 10)
    cleared = {}                                                                        # Codex m5b：入口切換時清除舊段緩存／待處理命令（不動設定點與套用值）
    if getattr(node, 'grip_cmd', None) is not None:                                  # r5：殘留夾爪命令（門段最後一筆）
        cleared['grip_cmd'] = float(node.grip_cmd)
    node.grip_cmd = None
    for _k in ('wb_latest', 'wb_env_latest', 'ho_req', 'k3m_req', 'k3m_pause_req'):
        if getattr(node, _k, None) is not None:
            cleared[_k] = 'set'
        setattr(node, _k, None)
    for _k in ('w2_inbox', 'w2_req_q', 'w2_eff_q'):
        _v = getattr(node, _k, None)
        if _v:
            cleared[_k] = len(_v)
            _v.clear()
    _lv0, _av0 = robot.get_linear_velocity(), robot.get_angular_velocity()
    _q00 = robot.get_joint_positions()
    start_state = {'base_lin_vel_world': [float(v) for v in _lv0], 'base_ang_vel_world': [float(v) for v in _av0],
                   'q_arm': [float(_q00[idx[j]]) for j in ARM], 'cleared': cleared,
                   'chain_setpoint': None if getattr(chain, 'setpoint', None) is None else [float(x) for x in chain.setpoint],
                   'chain_u_prev': None if getattr(chain, 'u_prev', None) is None else [float(x) for x in chain.u_prev]}
    print(f'[drawer] **抽屜段開始** @ step {step_id}：控制者 {ex.owner}；PARK_HOLD 停位 {_pk}；收臂判準 {stow}；'
          f'停位模式 {"hold" if park_on else "none（MOTM 定版）"}；switch_guard {"抽屜停位閘" if park_on else "None"}、v_min_lin {ex_prev["v_min_lin"]}→{ex.v_min_lin}', flush=True)

    se = max(1, int(round((1.0 / a.scan_rate) / a.physics_dt)))
    ste = max(1, int(round((1.0 / a.state_rate) / a.physics_dt)))
    t0 = float(world.current_time)
    next_scan = next_state = next_grip = t0
    contact_hold = PhysicsContactHold(physics_dt=a.physics_dt, contact_min_n=a.contact_min_n)
    prev_pose = None
    rec = {'version': VERSION, 'phase': 'drawer', 'start': {'step': step_id, 'sim_t': t0, 'owner': ex.owner, 'ex_prev': ex_prev, 'state': start_state},
           'steps': [], 'events': [],
           'steps_cols': ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'kind',
                          'base_cmd_body', 'meas_vb_body', 'applied_arm',
                          'base_xyth', 'q_arm_meas', 'opening_m',
                          'q_finger', 'finger_contact_n', 'contact_held_s',
                          'drawer_pairs', 'drawer_net_n', 'park_mode', 'door_q_rad', 'return'],
           'drawer_pairs': {'source': 'every_physics_step',
                            'row': ['body_index', 'n_points', 'normal_force_xyz_N', 'friction_force_xyz_N',
                                    'force_weighted_point_xyz_m', 'min_separation_m'],
                            'frame': 'world; forces act on the drawer', 'bodies': pair_names},
           'owner_codes': {'0': AUTH_NAV, '1': AUTH_WHOLEBODY, '2': AUTH_GLIDE}}
    ARM_ = list(ARM)
    ret = None                                                                          # r6：返程門操作
    if getattr(a, 'return_phase', False):
        gate = c['ReturnGate'](finger_open=a.finger_open, finger_closed=a.finger_closed, contact_min_n=a.contact_min_n)
        ret = {'gate': gate, 'on': False, 'hb': None, 'T': None, 'cmd': float(a.finger_open), 'q': [], 'zero_since': None, 'fb': (0.0, 0.0), 'n_msg': 0, 'abort_done': False}

        def _ret_cb(m):
            try:
                ret['q'].append(json.loads(m.data))
            except Exception:
                ret['q'].append({'kind': 'abort', 'why': 'bad_json'})
        node.create_subscription(M['String'], '/return/req', _ret_cb, 50)
        node.ret_state_pub = node.create_publisher(M['String'], '/return/state', 10)
        node.ret_sync_pub = node.create_publisher(M['Float64MultiArray'], '/return/sync', 10)
    grasp = {'t_ramp_done': None, 'ok_t': None, 'max_held': 0.0, 'seen_open_t': None}     # r4／r5
    _gto = float(getattr(a, 'grasp_evidence_timeout_s', 10.0))
    while rclpy.ok():
        t = float(world.current_time)
        if t >= a.sim_limit:
            print(f'[drawer] 達模擬上限 {a.sim_limit} s', flush=True)
            rec['stop_reason'] = 'sim_limit'
            break
        if node.stop_req is not None:
            print(f'[drawer] **受控停止**：{node.stop_req}', flush=True)
            rec['stop_reason'] = node.stop_req
            break
        if getattr(node, 'k2_abort_msg', None) is not None and 'k2_abort' not in rec:
            rec['k2_abort'] = {'msg': node.k2_abort_msg, 'step': step_id, 'sim_t': t}
            ex.cancel_handover(f'k2_abort：{node.k2_abort_msg}')
            print(f'[drawer] **中止閂鎖**（/door/k2_abort）@ step {step_id}：{node.k2_abort_msg} ⇒ 受控停止', flush=True)
            rec['stop_reason'] = f'k2_abort：{node.k2_abort_msg}'
            break
        _n_this = 0
        for _ in range(64):
            _before = node.n_cb
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.n_cb == _before:
                break
            _n_this += 1
        node.n_drain_max = max(node.n_drain_max, _n_this)
        node.publish_clock(t)

        if ret is not None:                                                             # r6：請求（同輪中止優先由 gate 保證）
            while ret['q']:
                _m = ret['q'].pop(0)
                ret['n_msg'] += 1
                ret['on'] = True
                ret['hb'] = c['wall']()
                _k = _m.get('kind')
                if _k == 'arm':
                    _ok, _why = ret['gate'].arm(_m.get('op'), _m.get('bar'), t)
                    ret['T'] = None
                    if not _ok:
                        ret['gate']._abort(f'arm_refused:{_why}', t)
                elif _k == 'close':
                    ret['T'] = np.asarray(_m['T'], float).reshape(4, 4)
                    ret['gate'].request('close', t)
                elif _k in ('move', 'open', 'retreat', 'done'):
                    ret['gate'].request(_k, t)
                elif _k == 'abort':
                    ret['gate']._abort(f"task:{_m.get('why')}", t)
        _r_abort = ret is not None and ret['gate'].abort is not None
        if node.wb_latest is not None:
            if not _r_abort:
                chain.receive(list(node.wb_latest[0]), t)
            node.wb_latest = None

        if node.ho_req is not None:
            r = node.ho_req
            node.ho_req = None
            if c['latched']() and r.get('action') != 'cancel':
                rec['events'].append({'sim_t': t, 'step': step_id, 'k2': 'handover_refused_after_abort', 'request': r})
                print(f'[drawer] 中止已閂鎖 ⇒ 拒絕換手請求 {r}', flush=True)
            elif park_state['violation'] is not None and r.get('action') != 'cancel':
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'handover_refused_after_violation', 'request': r})
                print(f'[drawer] PARK 違規已閂鎖 ⇒ 拒絕換手請求 {r}', flush=True)
            elif r.get('action') == 'cancel':
                ex.cancel_handover(r.get('why', ''))
            elif r.get('to') == 'nav' and hasattr(ex, 'request_nav'):                 # 交回導航：同門段 R2 路徑（包絡准入、任期封存）
                _qm3 = robot.get_joint_positions()
                _vt3 = ex.request_nav(int(r.get('at_step', step_id + 5)), t, [float(_qm3[idx[j]]) for j in ARM_],
                                      [float(_qm3[fidx[j]]) for j in FJ] if len(FJ) == 2 else None,
                                      window_steps=int(r.get('window_steps', 0)))
                rec['events'].append({'sim_t': t, 'step': step_id, 'nav_request': r, 'ok': None if _vt3 is None else bool(_vt3.ok),
                                      'admission': (ex.nav_admission[-1] if getattr(ex, 'nav_admission', None) else None)})
            else:
                ex.request_handover(r.get('to', AUTH_WHOLEBODY), int(r.get('at_step', step_id + 5)), t,
                                    window_steps=int(r.get('window_steps', 0)))

        node.last_sim_t = t
        nav = node.nav_cmd
        if nav is not None and node.nav_cmd_t is not None and t - node.nav_cmd_t > a.cmd_timeout:
            nav = (0.0, 0.0, 0.0)
            if not getattr(node, '_to_logged', False):
                node._to_logged = True
                print(f'[drawer] **導航命令逾時** {t - node.nav_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ 歸零', flush=True)
        elif nav is not None:
            node._to_logged = False
        gld = node.glide_cmd
        if gld is not None and node.glide_cmd_t is not None and t - node.glide_cmd_t > a.cmd_timeout:
            gld = (0.0, 0.0, 0.0)
            if not getattr(node, '_gto_logged', False):
                node._gto_logged = True
                print(f'[drawer] **減速段命令逾時** {t - node.glide_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ 歸零', flush=True)
        elif gld is not None:
            node._gto_logged = False

        if _r_abort:
            nav = (0.0, 0.0, 0.0)
            gld = (0.0, 0.0, 0.0)
        q_meas = robot.get_joint_positions()
        q_arm = [float(q_meas[idx[j]]) for j in ARM_]
        _p0, _q0 = robot.get_world_pose()
        _yw0 = yaw_of(_q0)
        _mvb_now = None
        if prev_pose is not None:
            _ddx, _ddy = float(_p0[0]) - prev_pose[0], float(_p0[1]) - prev_pose[1]
            _cc, _ss = math.cos(_yw0), math.sin(_yw0)
            _mvb_now = [(_ddx * _cc + _ddy * _ss) / a.physics_dt, (-_ddx * _ss + _ddy * _cc) / a.physics_dt,
                        (_yw0 - prev_pose[2]) / a.physics_dt]
        if node.park_ext_vio is not None and park_state['violation'] is None:
            _v = {'step': step_id, 't': t, 'phase': 'external', 'why': f'external：{node.park_ext_vio.get("why")}',
                  'source': node.park_ext_vio}
            park_state['violation'] = _v
            park_state['external'] = node.park_ext_vio
            ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
            rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
            print(f'[drawer] **PARK 外部違規**（閂鎖中止）@ step {step_id}：{_v["why"]}', flush=True)
        if park_hold is None and not park_done and ex.owner != AUTH_WHOLEBODY:
            _stowed = max(abs(float(q_arm[k]) - float(stow[k])) for k in range(len(stow))) <= a.park_stow_tol
            park_gate.update(step_id, t, float(_p0[0]), float(_p0[1]), float(_yw0), park_prev_cmd, _stowed)
        _fq = [float(q_meas[fidx[j]]) for j in FJ] if len(FJ) == 2 else None          # r2：同門段 T5（R2 包絡監看需兩指）
        res = ex.step(step_id, t, a.physics_dt, nav_cmd=nav, glide_cmd=gld, q_arm_measured=q_arm, meas_vb=_mvb_now,
                      **({'fingers': _fq} if hasattr(ex, 'envelope') else {}))
        if getattr(ex, 'env_violated', False) and 'nav_envelope_violation' not in rec:  # r2：包絡違反 ⇒ 既有減速路徑到零再結束
            rec['nav_envelope_violation'] = {'step': step_id, 'sim_t': t, 'first': dict(getattr(ex.envelope, 'first', None) or {}),
                                             'base_cmd_at_detect': [float(x) for x in res.base_cmd]}
            ex.cancel_handover('導航包絡違反')
            node.park_vio_pub.publish(M['String'](data=json.dumps({'step': step_id, 't': t, 'why': 'nav_envelope_violation',
                                                                    'first': rec['nav_envelope_violation']['first']}, default=float)))
            print(f"[drawer] **導航包絡違反**（閂鎖）@ step {step_id}：{rec['nav_envelope_violation']['first']} ⇒ 既有減速路徑到零後停止", flush=True)
        if 'nav_envelope_violation' in rec:
            _z = max(abs(float(x)) for x in res.base_cmd) < 1e-9
            _zs = rec['nav_envelope_violation'].setdefault('zero_since', None)
            rec['nav_envelope_violation']['zero_since'] = (_zs if _zs is not None else t) if _z else None
            if _z and t - rec['nav_envelope_violation']['zero_since'] >= 0.2 - 1e-9:
                _lv1 = robot.get_linear_velocity()
                rec['nav_envelope_violation'].update(stopped_t=t, base_lin_vel_world=[float(v) for v in _lv1])
                print(f"[drawer] **包絡違反減速完成**（實際寫入連續 0.2 s 為零）@ sim {t:.2f} ⇒ 停止", flush=True)
                rec['stop_reason'] = 'nav_envelope_violation_stopped'
                break

        p_now, qn = robot.get_world_pose()
        yw = yaw_of(qn)
        vx, vy, wz = res.base_cmd
        wx = vx * math.cos(yw) - vy * math.sin(yw)
        wy = vx * math.sin(yw) + vy * math.cos(yw)
        robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))
        _fcmd = grip_latch(a.finger_open if node.grip_cmd is None
                           else min(max(float(node.grip_cmd), a.finger_closed), a.finger_open))
        if ret is not None and ret['on']:                                               # r6：手指由准入決定（1 s 斜坡；閂鎖後凍結）
            _tg = ret['gate'].finger_target()
            _dm = (a.finger_open - a.finger_closed) * a.physics_dt
            ret['cmd'] = float(np.clip(_tg, ret['cmd'] - _dm, ret['cmd'] + _dm))
            _fcmd = grip_latch(ret['cmd'])
        node.last_fcmd = float(_fcmd)
        if res.arm_setpoint is not None:
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM_):
                tgt[idx[j]] = res.arm_setpoint[k]
            for j in FJ:
                tgt[fidx[j]] = _fcmd
            apply_pos(tgt)

        _cmd3 = [float(x) for x in res.base_cmd]
        _x, _y = float(p_now[0]), float(p_now[1])
        if park_hold is None and not park_done and res.owner == AUTH_WHOLEBODY:
            if park_gate.anchor is None:
                park_state['transfer_without_gate'] = True
            park_hold = HoldMonitor(park_cfg, park_gate.anchor or (_x, _y, float(yw)), step_id, t, prev=park_last_sample)
            rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_start', 'anchor': park_hold.anchor,
                                  'gate_pass_step': park_gate.pass_step, 'prev_sample': park_last_sample})
        if park_hold is not None and park_hold.end is None:
            if res.owner == AUTH_WHOLEBODY:
                _v = park_hold.update(step_id, t, _x, _y, float(yw), _cmd3, phase=res.kind)
            elif res.owner == AUTH_NAV:
                _v = park_hold.update(step_id, t, _x, _y, float(yw), None, phase='handback')
                _stowed_hb = max(abs(float(q_arm[k]) - float(stow[k])) for k in range(len(stow))) <= a.park_stow_tol
                park_hold.close(step_id, t, stowed=_stowed_hb)
                park_done = True
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_end', 'verdict': park_hold.verdict(),
                                      'handback_stowed': _stowed_hb})
            else:
                _v = {'step': step_id, 't': t, 'phase': res.kind, 'why': f'保持期內控制者變成 {res.owner}（不是交還導航）'}
            if (_v is not None or park_state['transfer_without_gate']) and park_state['violation'] is None:
                _v = _v or {'step': step_id, 't': t, 'phase': res.kind, 'why': '未通過停車閘門即切換給全身'}
                park_state['violation'] = _v
                ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
                node.park_vio_pub.publish(M['String'](data=json.dumps(_v, default=float)))
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
                print(f'[drawer] **PARK 違規**（閂鎖中止）@ step {step_id}：{_v["why"]}', flush=True)
        if ret is not None and ret['on']:                                               # r6：本步監看
            _fq2 = [float(q_meas[fidx[j]]) for j in FJ]
            _rc = c['door_rec'](step_id, t)
            _qd, _ = c['door_qd']()
            _terr = None
            if ret['T'] is not None:
                _terr = float(np.linalg.norm(np.asarray(c['fk'](np.r_[_x, _y, float(yw), q_arm]))[:3, 3] - ret['T'][:3, 3]))
            _ab0 = ret['gate'].abort
            _ab = ret['gate'].step(t, _rc, float(_qd), _fq2, _terr, c['wrist'](), c['wall']() - ret['hb'], ret['cmd'] <= a.finger_closed + 1e-9, fcmd=ret['cmd'])
            _F = c['forces'](_rc)
            _b = c['bar'].get(ret['gate'].bar)
            ret['fb'] = (0.0, 0.0) if (_F is None or _b is None) else tuple(_F['fingers'][f_].get(_b, 0.0) for f_ in c['fingers'])
            if _ab is not None and _ab0 is None:
                rec['return_abort'] = dict(_ab, step=step_id)
                ex.cancel_handover(f"返程閘門中止：{_ab['why']}")
                c['expire_chain']()
                print(f"[return] **返程閘門中止** @ step {step_id} sim {t:.2f}：{_ab} ⇒ 既有減速到零後停止", flush=True)
            if _ab is not None:
                _zr = max(abs(float(v)) for v in res.base_cmd) < 1e-9
                ret['zero_since'] = (ret['zero_since'] if ret['zero_since'] is not None else t) if _zr else None
                if ret['zero_since'] is not None and t - ret['zero_since'] >= 0.2 - 1e-9:
                    print(f"[return] 減速完成 ⇒ 停止世界 @ sim {t:.2f}", flush=True)
                    rec['stop_reason'] = f"return_abort:{_ab['why']}"
                    break
        park_prev_cmd = tuple(_cmd3)
        park_last_sample = (step_id, t, _x, _y, float(yw))
        _pmode = (('hold' if (park_hold is not None and park_hold.end is None) else 'post' if park_done else 'gate') if park_on else None)
        if park_on and step_id % 2 == 0:
            _gs = park_gate.state()
            node.park_gate_pub.publish(M['String'](data=json.dumps({
                'step': step_id, 'sim_t': t, 'passed': _gs['passed'], 'ok_now': _gs['ok_now'], 'why_now': _gs['why_now'],
                'pass_step': _gs['pass_step'], 'anchor': _gs['anchor'], 'mode': _pmode,
                'violation': park_state['violation'] is not None}, default=float)))

        _mode = 0 if res.owner == AUTH_WHOLEBODY else 4
        am = M['Float64MultiArray']()
        am.data = ([float(step_id), float(t)] + [float(x) for x in res.u_applied]
                   + [float(_mode), float('nan'), float(getattr(chain, 'n_recv', 0)), float(getattr(chain, 'n_rejected', 0)),
                      1.0 if res.kind in ('nav_applied', 'glide_applied', 'wholebody_applied') else 0.0, -1.0, float('nan')])
        node.applied_pub.publish(am)
        if res.arm_setpoint is not None:
            sm = M['Float64MultiArray']()
            sm.data = [float(step_id), float(t), 1.0] + [float(x) for x in res.arm_setpoint] + [float(_mode), 1.0]
            node.sp_pub.publish(sm)

        _mvb = None
        if prev_pose is not None:
            _dx, _dy = float(p_now[0]) - prev_pose[0], float(p_now[1]) - prev_pose[1]
            _c, _s = math.cos(yw), math.sin(yw)
            _mvb = [(_dx * _c + _dy * _s) / a.physics_dt, (-_dx * _s + _dy * _c) / a.physics_dt, (yw - prev_pose[2]) / a.physics_dt]
        prev_pose = (float(p_now[0]), float(p_now[1]), yw)
        xfc.Clear()
        _dy_rec = float(xfc.GetLocalToWorldTransform(dprim).ExtractTranslation()[1])
        _contact = finger_contact_n() if FJ else (float('nan'), float('nan'))
        _contact_held = contact_hold.update(step_id, t, *_contact,
                                            closing=(node.grip_cmd is not None and node.grip_cmd < a.finger_open - 1e-12))
        if grasp['ok_t'] is None:                                                       # r4：有界抓持失敗出口
            grasp['max_held'] = max(grasp['max_held'], float(_contact_held))
            if float(_contact_held) >= GRASP_HOLD_S - 1e-9:
                grasp['ok_t'] = t
                rec['events'].append({'sim_t': t, 'step': step_id, 'grasp': 'evidence_ok', 'held_s': float(_contact_held), 'contact_n': list(_contact)})
            if grasp['seen_open_t'] is None and (node.grip_cmd is None or float(node.grip_cmd) >= a.finger_open - 1e-6):   # None＝預設張開
                grasp['seen_open_t'] = t                                                # r5：先見到抽屜任務的張開命令
            if grasp['ok_t'] is not None:
                pass
            elif (grasp['seen_open_t'] is not None and res.owner == AUTH_WHOLEBODY and node.grip_cmd is not None
                  and float(node.grip_cmd) <= a.finger_closed + 1e-9):
                if grasp['t_ramp_done'] is None:
                    grasp['t_ramp_done'] = t
                    rec['events'].append({'sim_t': t, 'step': step_id, 'grasp': 'close_ramp_done'})
                elif t - grasp['t_ramp_done'] >= _gto - 1e-9:
                    rec['grasp_no_contact'] = {'step': step_id, 'sim_t': t, 't_ramp_done': grasp['t_ramp_done'], 'timeout_s': _gto,
                                               'max_held_s': grasp['max_held'], 'contact_n': list(_contact),
                                               'q_finger': [float(q_meas[fidx[j]]) for j in FJ]}
                    node.park_vio_pub.publish(M['String'](data=json.dumps({'step': step_id, 't': t, 'why': 'grasp_no_contact'}, default=float)))
                    print(f"[drawer] **閉爪斜坡完成後 {_gto:.0f} s 無接觸證據**（最大保持 {grasp['max_held']:.2f} s、接觸 {list(_contact)}）⇒ 停止世界", flush=True)
                    rec['stop_reason'] = 'grasp_no_contact'
                    break
        rec['steps'].append([
            step_id, round(t, 4), {AUTH_NAV: 0, AUTH_WHOLEBODY: 1}.get(res.owner, 2), res.kind,
            [round(float(x), 6) for x in res.base_cmd], None if _mvb is None else [round(x, 6) for x in _mvb],
            [round(float(x), 6) for x in res.u_applied[3:]],
            [round(float(p_now[0]), 6), round(float(p_now[1]), 6), round(float(yw), 6)],
            [round(float(x), 6) for x in q_arm], round(float(DY0 - _dy_rec), 6),
            ([round(float(q_meas[fidx[j]]), 6) for j in FJ] if FJ else None),
            ([round(x, 4) for x in _contact] if (FJ and node.grip_cmd is not None) else None),
            float(_contact_held), *drawer_pairs(), _pmode, round(float(door_q_fn()), 8),
            None if ret is None or not ret['on'] else [ret['gate'].state, round(ret['fb'][0], 3), round(ret['fb'][1], 3), round(ret['cmd'], 6)]])

        if FJ and t >= next_grip - 1e-9:
            next_grip = max(next_grip + ste * a.physics_dt, t)
            gm = M['Float64MultiArray']()
            gm.data = [t, float(q_meas[fidx[FJ[0]]]), float(q_meas[fidx[FJ[-1]]]), *_contact,
                       float(step_id), float(_contact_held), a.contact_min_n]
            node.grip_pub.publish(gm)

        world.step(render=False)
        step_id += 1
        node.step_id = step_id
        t_after = float(world.current_time)

        if t_after >= next_state - 1e-9:
            next_state = max(next_state + ste * a.physics_dt, t_after)
            qm = robot.get_joint_positions()
            js = M['JointState']()
            js.header.stamp = node.stamp(t_after)
            js.name = list(ARM_)
            js.position = [float(qm[idx[j]]) for j in ARM_]
            node.js_pub.publish(js)
            p2, q2 = robot.get_world_pose()
            od = M['Odometry']()
            od.header.stamp = node.stamp(t_after)
            od.header.frame_id = 'odom'
            od.child_frame_id = 'base_footprint'
            od.pose.pose.position.x = float(p2[0])
            od.pose.pose.position.y = float(p2[1])
            od.pose.pose.orientation.w = float(q2[0])
            od.pose.pose.orientation.x = float(q2[1])
            od.pose.pose.orientation.y = float(q2[2])
            od.pose.pose.orientation.z = float(q2[3])
            node.odom_pub.publish(od)
            if node.tf_pub is not None:
                tfm = M['TransformStamped']()
                tfm.header.stamp = od.header.stamp
                tfm.header.frame_id = 'odom'
                tfm.child_frame_id = 'base_footprint'
                tfm.transform.translation.x = float(p2[0])
                tfm.transform.translation.y = float(p2[1])
                tfm.transform.rotation.w = float(q2[0])
                tfm.transform.rotation.x = float(q2[1])
                tfm.transform.rotation.y = float(q2[2])
                tfm.transform.rotation.z = float(q2[3])
                node.tf_pub.sendTransform(tfm)
            xfc.Clear()
            dy = float(xfc.GetLocalToWorldTransform(dprim).ExtractTranslation()[1])
            node.drawer_state_pub.publish(M['Float64MultiArray'](data=[float(t_after), float(DY0 - dy)]))
            _hm = xfc.GetLocalToWorldTransform(c['hprim'])
            _hM = np.array([[float(_hm[r][cc]) for r in range(4)] for cc in range(4)], float)
            node.drawer_handle_pub.publish(M['Float64MultiArray'](data=[float(t_after), float(DY0 - dy)] + [float(x) for x in _hM.reshape(-1)]))
            _yaw2 = math.atan2(2 * (float(q2[0]) * float(q2[3]) + float(q2[1]) * float(q2[2])),
                               1 - 2 * (float(q2[2]) ** 2 + float(q2[3]) ** 2))
            node.drawer_sync_pub.publish(M['Float64MultiArray'](data=(
                [float(t_after), float(step_id), float(DY0 - dy)] + [float(x) for x in _hM.reshape(-1)]
                + [float(p2[0]), float(p2[1]), _yaw2] + [float(qm[idx[j]]) for j in ARM_])))
            if ret is not None:
                _qs, _qds = c['door_qd']()
                _ys = math.atan2(2 * (float(q2[0]) * float(q2[3]) + float(q2[1]) * float(q2[2])), 1 - 2 * (float(q2[2]) ** 2 + float(q2[3]) ** 2))
                node.ret_sync_pub.publish(M['Float64MultiArray'](data=[float(t_after), float(step_id), float(_qs), float(_qds), float(p2[0]), float(p2[1]), _ys]
                                                                 + [float(qm[idx[j]]) for j in ARM_] + [float(ret['fb'][0]), float(ret['fb'][1]), float(qm[fidx[FJ[0]]])]))
                node.ret_state_pub.publish(M['String'](data=json.dumps({'sim_t': t_after, 'step': step_id, 'state': ret['gate'].state, 'op': ret['gate'].op, 'abort': ret['gate'].abort,
                                                                        'n_msg': ret['n_msg'], 'cmd': ret['cmd']}, default=float)))
            node.ho_pub.publish(M['String'](data=json.dumps({'owner': ex.owner, 'step': step_id, 'sim_t': t_after,
                                                              'chain_seeded': ex.wb_seeded, **ex.handover_pending_info(),
                                                              'nav_applied': ex.nav_applied_report()}, default=float)))

        if t_after >= next_scan - 1e-9:
            next_scan = max(next_scan + se * a.physics_dt, t_after)
            xfc.Clear()
            o5 = xfc.GetLocalToWorldTransform(c['lidar_prim']).ExtractTranslation()
            _, q3 = robot.get_world_pose()
            y5 = yaw_of(q3)
            S = c['scan']
            sc = M['LaserScan']()
            sc.header.stamp = node.stamp(t_after)
            sc.header.frame_id = 'lidar_link'
            sc.angle_min, sc.angle_max = S['min'], S['max']
            sc.angle_increment = S['inc']
            sc.range_min, sc.range_max = S['rmin'], S['rmax']
            org = [float(o5[0]), float(o5[1]), float(o5[2])]
            sc.ranges = [ray_range(c['query'], org, [math.cos(y5 + S['min'] + i * S['inc']), math.sin(y5 + S['min'] + i * S['inc']), 0.0],
                                   S['rmax'], c['excluded']) for i in range(S['n'])]
            node.scan_pub.publish(sc)

    rec['end'] = {'step': step_id, 'sim_t': float(world.current_time), 'owner': ex.owner}
    rec['summary'] = ex.summary()
    rec['chain'] = chain.summary()
    rec['tenure_log'] = tenure_log
    rec['return_gate'] = None if ret is None else ret['gate'].report()
    rec['grasp'] = dict(grasp, timeout_s=_gto, hold_s=GRASP_HOLD_S,
                        seen_open_semantics='seen_open_t＝本階段首次「無夾爪命令（預設張開）或命令 ≥ finger_open」的時刻；不代表已收到抽屜任務張開命令，也不代表實測已張開',
                        ramp_semantics='t_ramp_done 只設一次、不延後；期限自此起算')
    rec['park'] = {'park_pose': _pk, 'cfg': park_cfg.__dict__, 'gate': park_gate.state(), 'gate_last': park_gate.last,
                   'hold': None if park_hold is None else park_hold.report(), 'violation': park_state['violation'],
                   'transfer_without_gate': park_state['transfer_without_gate']}
    rec['config'] = {'drawer': c.get('dauth'), 'DY0': DY0, 'stow_for_gate': stow, 'drawer_pose': a.drawer_pose,
                     'drawer_park_pose': a.drawer_park_pose, 'finger_leader_only': bool(getattr(a, 'finger_leader_only', False))}
    out = os.path.join(out_dir, 'room_run_drawer.json')
    json.dump(rec, open(out, 'w'), ensure_ascii=False, indent=1, default=str)
    print(f'[drawer] 封存 {out}；控制者 {ex.owner}；步數 {step_id}', flush=True)
    return rec
