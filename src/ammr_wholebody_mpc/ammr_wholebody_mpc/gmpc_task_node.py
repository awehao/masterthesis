#!/usr/bin/env python3
"""原導航 GMPC 的任務擴充（主線 B；Codex 20261117：一個導航控制器、同一實例的兩種模式）。
封存的 gmpc_node.py／gmpc.py 不改：繼承 GMPCNode（同類別、同求解器、同參數檔、同節點名 gmpc_controller），路徑仍是 Smac2D 的 /plan。
沒有任務條件時與原節點逐行相同。任務條件（/task/conditions，std_msgs/String JSON）：
  {task, ver, frame:'map', stamp, heading（rad，map）, box:[[vx_lo,vx_hi],[vy_lo,vy_hi],[wz_lo,wz_hi]]（本體座標，含負向）, t_eff_s, valid_s}
  - 朝向：取代「路徑前視弦」的原始方向，**仍走原節點同一套角度環繞與變化率限制**（_yaw_ref／heading_rate_max）；進入、更新、退出任務都不重設參考。
  - 速度框：與原限制（vx/vy/wz_min/max）取交集，求解前寫進求解器的 u_min／u_max（求解後還原；與原節點 detour 下限同一存還方式）。
    新框比當下命令緊時，邊界以各軸加速度上限 a_max 由當下命令漸收到新框（提前減速）；漸收所需時間 > t_eff_s ⇒ 拒收該條件（不放寬、不硬切）。
  - 版本須遞增；任務進行中條件逾時（valid_s）或收到 {lost:true} ⇒ **不回一般導航**：速度框以 a_max 漸收到零（停下）並發 /task/status=lost；
    只有收到 {clear:true}（任務正常結束、交權後或監督節點撤銷）才回原行為。
  - /task/conditions_ack：每個條件的採用／拒收與原因；/task/status：idle／task／lost。
區域模式（主線 B；Codex 20261117_044741：操作端給「可接任務的狀態集合」，導航自己選點、規劃與局部收斂）：
  條件可帶 region＝{id, line:[[x0,y0],[x1,y1]]（走廊中心線，進入端→離開端，map）, half_width, extend}。導航：
  - 選進入點 E：走廊上「目前位置投影＋L_CONV」處，但不晚於操作端給的最晚進入點 enter_by−L_CONV（不早於走廊起點、不晚於離開端前 0.10 m）；已過 enter_by 仍未在區內 ⇒ 狀態標 late；
  - 向同一 planner_server（ComputePathToPose，GridBased）要 目前位置→E 的 Smac 路徑，每 1 s 重算，直到已收斂（側向誤差 ≤ 2×half_width 且投影已過 E）；
  - 參考路徑＝Smac(→E)＋走廊 E→離開端＋沿走廊延伸 extend（一般 0.30 m 到點容差不動；延伸只為了任務模式不在走廊內因到點容差提前停）；
    區域模式期間外部 /plan 只暫存不採用，撤銷區域後恢復最新外部路徑。
  - /task/region_status：側向誤差、沿走廊位置、是否在區域內（|側向| ≤ half_width）、階段（approach／inside）。
  區域只改參考路徑；CBF、Shield、GUARD、速度限制與求解器不變。"""
import json
import os
import math

import numpy as np
import rclpy
from std_msgs.msg import String
from ammr_wholebody_mpc.gmpc_node import GMPCNode


class GMPCTaskNode(GMPCNode):
    def __init__(self):
        super().__init__()
        self._t_mode, self._t_ver, self._t_cond, self._t_until = 'idle', -1, None, None
        self._t_box_eff = None                                       # 目前生效（漸收中）的速度框 3×2
        self._t_box_tgt = None
        self._t_last = None
        self._t_heading = None
        self._t_log = []
        self.create_subscription(String, '/task/conditions', self._t_cb, 10)
        self.p_t_ack = self.create_publisher(String, '/task/conditions_ack', 10)
        self.p_t_status = self.create_publisher(String, '/task/status', 10)
        self.create_timer(0.05, self._t_tick)
        # 區域模式
        from nav_msgs.msg import Path as _Path
        from geometry_msgs.msg import PoseStamped as _PS
        from rclpy.action import ActionClient
        from nav2_msgs.action import ComputePathToPose
        self._t_Path, self._t_PS, self._t_CP = _Path, _PS, ComputePathToPose
        self._t_cp = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self._t_region, self._t_reg_busy, self._t_reg_t, self._t_reg_leg, self._t_ext_plan, self._t_internal = None, False, -9.0, None, None, False
        self._t_reg_pub = self.create_publisher(String, '/task/region_status', 10)
        self.create_timer(0.1, self._t_region_tick)
        # 預測發布（主線 B 操作端用；只讀求解結果、不改任何控制決策）：ξ_k＝ξ_ref,k＋δξ_k（夾在速度上下限），自 X_now 以 SE(2) 指數積分 N 步。
        from std_msgs.msg import Float64MultiArray
        self._t_pred_pub = self.create_publisher(Float64MultiArray, '/task/gmpc_pred', 10)
        self._t_solve_in_pub = self.create_publisher(String, '/task/nav_solve_in', 10)
        self._t_F64 = Float64MultiArray
        # 共同核心（Howard 20261117：導航與操作都用同一個 gmpc_wb；導航內容不動）：求解器物件換成 gmpc_wb.GMPC，設定逐欄照抄原導航設定、
        #   節點與求解器共用同一個設定物件（節點暫改 u_min 的行為不變）；不傳 arm ⇒ 原底盤問題（離線 40 例含障礙與 CBF，u_opt 與封存 gmpc 逐位元相同）。
        #   GMPC_WB_CORE=0 退回封存 gmpc。
        if os.environ.get('GMPC_WB_CORE', '1') == '1':
            import dataclasses as _dc
            from . import gmpc_wb as _WB
            self._cfg = _WB.GMPCConfig(**{f.name: getattr(self._cfg, f.name) for f in _dc.fields(type(self._cfg))})
            self.mpc = _WB.GMPC(self._cfg)
            self.get_logger().info('gmpc_task_node: solver core = gmpc_wb (arm hold)')
        _orig_solve = self.mpc.solve

        self._t_frontend = os.environ.get('GMPC_FRONTEND', '0') == '1'   # 單一控制器（Codex 20261117_161844）：本節點只當參考＋障礙物前端，不求解、不發命令
        self._t_exec, self._t_n_in = np.zeros(3), 0
        if self._t_frontend:
            from geometry_msgs.msg import Twist as _Tw
            self.create_subscription(_Tw, '/cmd_vel', lambda m: setattr(self, '_t_exec', np.array([m.linear.x, m.linear.y, m.angular.z], float)), 20)

            class _NoPub:
                def publish(self, *_a, **_k):
                    pass
            self.cmd_pub = _NoPub()                                   # 名目底盤命令不發布（命令只由單一控制器出）
            self.get_logger().info('gmpc_task_node: FRONTEND mode (reference + obstacles only; no solve, no cmd)')

        # 11/27 N1（Howard 選「住宅任務版」）：純導航比較時把住宅任務的會車層（evaluation/yield_side.py，D2_YIELD_SIDE＋YS_PRED）
        #   接在這裡——每拍求解前只平移參考視窗（同 door_pass_ctrl.yield_apply 的做法：體速度依改後參考重算），CBF、Shield、求解器不變。
        #   開關 NAV_YIELD=1（另需 D2_YIELD_SIDE=1；地圖淨空用 M8_MAP）。住宅任務不設 NAV_YIELD ⇒ 導航節點行為不變（避免與 W-GMPC 的會車層重複套用）。
        self._t_ys = None
        if os.environ.get('NAV_YIELD', '0') == '1':
            import sys as _sys
            from pathlib import Path as _P
            _ev = os.environ.get('YS_EVAL_DIR') or str(_P(__file__).resolve().parents[6] / 'evaluation')
            if _ev not in _sys.path:
                _sys.path.insert(0, _ev)
            import yield_side as _YS
            self._t_YS = _YS
            self._t_ys = _YS.YieldSide() if _YS.ENABLE else None
            self._t_ys_n = 0
            self.get_logger().info(f'gmpc_task_node: NAV_YIELD 會車層 {"啟用" if self._t_ys else "未啟用（D2_YIELD_SIDE≠1）"}，eval={_ev}，M8_MAP={os.environ.get("M8_MAP")}')

        def _yield(X_now, X_ref_win, xi_ref_win, obstacles):
            ys = self._t_ys
            if ys is None or obstacles is None:
                return X_ref_win, xi_ref_win
            X_now = np.asarray(X_now, float)
            yaw = math.atan2(X_now[1, 0], X_now[0, 0]); c_, s_ = math.cos(yaw), math.sin(yaw)
            u_ = np.asarray(self.xi_prev[:2], float)
            ys.ref_dt = float(self._cfg.dt)
            X_ = [np.asarray(x, float).copy() for x in X_ref_win]
            n0 = len(ys.log)
            info = ys.apply(self.get_clock().now().nanoseconds * 1e-9, np.array([X_now[0, 2], X_now[1, 2], yaw]),
                            np.array([c_ * u_[0] - s_ * u_[1], s_ * u_[0] + c_ * u_[1]]), X_, obstacles, float(self.dt))
            for e_ in ys.log[n0:]:
                self._t_ys_n += 1
                if self._t_ys_n <= 400:
                    self.get_logger().info('YS ' + json.dumps({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in e_.items()}, ensure_ascii=False))
            if info is None:
                return X_ref_win, xi_ref_win
            xi_ = np.asarray(xi_ref_win, float).copy()
            for k in range(min(len(X_) - 1, len(xi_))):
                T1 = np.linalg.inv(X_[k]) @ X_[k + 1]
                xi_[k] = np.array([T1[0, 2] / self._cfg.dt, T1[1, 2] / self._cfg.dt, math.atan2(T1[1, 0], T1[0, 0]) / self._cfg.dt])
            if len(xi_) >= len(X_) and len(X_) > 1:
                xi_[len(X_) - 1] = xi_[len(X_) - 2].copy()
            return np.asarray(X_), xi_

        def _solve_pub(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=None, _o=_orig_solve):
            try:
                X_ref_win, xi_ref_win = _yield(X_now, X_ref_win, xi_ref_win, obstacles)
            except Exception as e_:                                   # noqa: BLE001
                self.get_logger().warn(f'NAV_YIELD 例外（本拍略過）：{e_!r}')
            if self._t_frontend:
                try:
                    self._t_pub_solve_in(np.asarray(X_now, float), np.asarray(X_ref_win, float), np.asarray(xi_ref_win, float), obstacles)
                except Exception as e_:                               # noqa: BLE001
                    self.get_logger().warn(f'solve_in publish failed: {e_!r}')
                from .gmpc_wb import GMPCResult as _GR
                N_ = len(X_ref_win) - 1
                return _GR(u_opt=np.asarray(self._t_exec, float).copy(), delta_xi_all=np.zeros((N_, 3)), e0=np.zeros(3), solve_time_s=0.0, status='frontend')   # 上一命令＝實際執行回饋
            r = _o(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=obstacles)
            try:
                self._t_pub_pred(np.asarray(X_now, float), np.asarray(xi_ref_win, float), np.asarray(r.delta_xi_all, float), np.asarray(r.u_opt, float))
            except Exception:                                         # noqa: BLE001
                pass
            try:
                self._t_pub_solve_in(np.asarray(X_now, float), np.asarray(X_ref_win, float), np.asarray(xi_ref_win, float), obstacles)
            except Exception:                                         # noqa: BLE001
                pass
            return r
        self.mpc.solve = _solve_pub

    def _halt(self, state, robot_xyth=None):
        super()._halt(state, robot_xyth=robot_xyth)
        if getattr(self, '_t_frontend', False) and robot_xyth is not None and self.cbf_enable:   # 前端：到點／無計畫時也照原規則持續提供障礙物（Codex 20261117_173459：障礙物不能因到點而消失）；參考＝原地（標記 halted）
            try:
                obstacles = list(self._obstacles)
                goal_xy = None
                if self.latest_path is not None and len(self.latest_path.poses):
                    gp = self.latest_path.poses[-1].pose.position; goal_xy = (float(gp.x), float(gp.y))
                rx, ry = robot_xyth[0], robot_xyth[1]
                dyn_near = (self.static_activate_range <= 0.0) or any(np.hypot(o['x'] - rx, o['y'] - ry) < self.static_activate_range for o in self._obstacles)
                if dyn_near:
                    for s_ in self._static_obstacles:
                        if goal_xy is None or np.hypot(s_['x'] - goal_xy[0], s_['y'] - goal_xy[1]) > self.static_goal_clear:
                            obstacles.append({**s_, 'static': True, 'margin': self.static_cbf_safe_margin})
                from .se2 import from_xytheta as _fx
                X_now = _fx(*robot_xyth)
                N_ = int(self.N)
                self._t_pub_solve_in(np.asarray(X_now, float), np.array([X_now] * (N_ + 1)), np.zeros((N_ + 1, 3)), obstacles, halted=state)
            except Exception as e_:                                   # noqa: BLE001
                self.get_logger().warn(f'halt solve_in publish failed: {e_!r}')

    def _t_pub_solve_in(self, X_now, X_ref, xi_ref, obstacles, halted=None):
        """（唯讀；單一控制器用）本拍導航原樣算好的求解輸入：參考視窗與障礙物（含動態邊距、靜態牆點、繞行偏移後的參考），
        以 X_now 為基準的相對座標（避開 map／odom 差異）；u_min[0]（繞行速度下限）一併附上。不改任何導航判定或命令。"""
        Xi = np.linalg.inv(X_now)
        cth, sth = X_now[0, 0], X_now[1, 0]
        rel = []
        for Xk in X_ref:
            R_ = Xi @ Xk
            rel += [float(R_[0, 2]), float(R_[1, 2]), math.atan2(float(R_[1, 0]), float(R_[0, 0]))]
        obs = []
        for o_ in (obstacles or []):
            dx, dy = float(o_['x']) - X_now[0, 2], float(o_['y']) - X_now[1, 2]
            vx, vy = float(o_.get('vx', 0.0)), float(o_.get('vy', 0.0))
            obs.append([cth * dx + sth * dy, -sth * dx + cth * dy, float(o_['radius']), cth * vx + sth * vy, -sth * vx + cth * vy,
                        float(o_.get('margin', float('nan'))), 1.0 if o_.get('static', False) else 0.0])
        self._t_n_in += 1
        msg = dict(t=self._t_now(), seq=self._t_n_in, frame='robot_at_t', N=len(X_ref) - 1, X_rel=rel, xi_ref=np.asarray(xi_ref, float).reshape(-1).tolist(), obs=obs, u_min0=float(self._cfg.u_min[0]), halted=halted)
        if self._t_n_in % 20 == 1:                                    # 導航求解設定（CBF、權重、限制）：單一控制器以同一組設定求解
            import dataclasses as _dc
            msg['cfg'] = {f.name: (np.asarray(getattr(self._cfg, f.name)).tolist() if isinstance(getattr(self._cfg, f.name), np.ndarray) else getattr(self._cfg, f.name)) for f in _dc.fields(self._cfg)
                          if isinstance(getattr(self._cfg, f.name), (int, float, bool, str, np.ndarray, type(None)))}
        self._t_solve_in_pub.publish(String(data=json.dumps(msg)))

    def _t_pub_pred(self, X, xi_ref, dxi, u0):
        cfg = self._cfg
        dt = float(cfg.dt)
        lo, hi = np.asarray(cfg.u_min, float), np.asarray(cfg.u_max, float)
        x, y, th = float(X[0, 2]), float(X[1, 2]), math.atan2(float(X[1, 0]), float(X[0, 0]))
        out = [self._t_now(), dt, float(len(dxi)), x, y, th]
        for k in range(len(dxi)):
            v = u0 if k == 0 else np.clip(xi_ref[k] + dxi[k], lo, hi)
            c, s_ = math.cos(th), math.sin(th)
            x += dt * (c * v[0] - s_ * v[1]); y += dt * (s_ * v[0] + c * v[1]); th += dt * v[2]
            out += [x, y, th]
        self._t_pred_pub.publish(self._t_F64(data=[float(z) for z in out]))

    # ---- 工具
    def _t_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _t_base_box(self):
        lo, hi = np.asarray(self._cfg.u_min, float), np.asarray(self._cfg.u_max, float)
        return np.c_[lo, hi]

    def _t_amax(self):
        return np.asarray(self._cfg.a_max, float)

    def _t_ack(self, ok, why, **kw):
        rec = dict(t=round(self._t_now(), 3), ver=self._t_ver if ok else kw.pop('ver', None), ok=bool(ok), why=why, mode=self._t_mode, **kw)
        self._t_log.append(rec)
        self.p_t_ack.publish(String(data=json.dumps(rec)))
        self.get_logger().info('[task] ' + json.dumps(rec))

    def _t_status(self):
        self.p_t_status.publish(String(data=json.dumps(dict(mode=self._t_mode, ver=self._t_ver, t=round(self._t_now(), 3)))))

    # ---- 條件
    def _t_cb(self, m):
        try:
            d = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            return
        if d.get('clear'):
            self._t_region_off('task_clear')
            self._t_mode, self._t_cond, self._t_until, self._t_box_tgt, self._t_heading = 'idle', None, None, None, None
            self._t_box_eff = None
            self._t_ack(True, 'clear')
            self._t_status()
            return
        if d.get('lost'):
            self._t_lose('supervisor_lost')
            return
        ver = int(d.get('ver', -1))
        if ver <= self._t_ver:
            self._t_ack(False, 'old_version', ver=ver); return
        if d.get('frame', 'map') != 'map':
            self._t_ack(False, 'frame', ver=ver); return
        h = d.get('heading')
        if h is None or not math.isfinite(float(h)):
            self._t_ack(False, 'heading_invalid', ver=ver); return
        base = self._t_base_box()
        box = np.array(d.get('box', base.tolist()), float).reshape(3, 2)
        tgt = np.c_[np.maximum(box[:, 0], base[:, 0]), np.minimum(box[:, 1], base[:, 1])]   # 與原限制取交集（只收緊）
        if np.any(tgt[:, 0] > tgt[:, 1]) or np.any(tgt[:, 0] > 0) or np.any(tgt[:, 1] < 0):
            self._t_ack(False, 'box_empty_or_excludes_zero', ver=ver); return
        u = np.asarray(self.xi_prev, float)
        over = np.maximum(u - tgt[:, 1], 0.0) + np.maximum(tgt[:, 0] - u, 0.0)
        need = float(np.max(over / np.maximum(self._t_amax(), 1e-6)))
        t_eff = float(d.get('t_eff_s', 0.0))
        if need > t_eff + 1e-9:
            self._t_ack(False, 'decel_infeasible', ver=ver, need_s=round(need, 3), t_eff_s=t_eff); return
        self._t_ver, self._t_cond = ver, d
        self._t_until = self._t_now() + float(d.get('valid_s', 1.0))
        self._t_heading = float(h)
        self._t_box_tgt = tgt
        if self._t_box_eff is None:
            self._t_box_eff = np.c_[np.maximum(np.minimum(u, tgt[:, 0]), base[:, 0]), np.minimum(np.maximum(u, tgt[:, 1]), base[:, 1])]
        self._t_mode = 'task'
        rg = d.get('region')
        if rg is not None:
            L = np.asarray(rg.get('line'), float).reshape(2, 2)
            if self._t_region is None or self._t_region['id'] != rg.get('id'):
                ph = 'approach' if self._t_region is None else self._t_region['phase']
                self._t_region = dict(id=rg.get('id'), p0=L[0], p1=L[1], w=float(rg.get('half_width', 0.02)), ext=float(rg.get('extend', 0.40)), enter_by=float(rg.get('enter_by', 1e9)),
                                      phase=ph, n_plan=0, t_set=self._t_now())
                self._t_reg_t = -9.0
            else:                                                     # 同一區域：線段以最新條件為準（操作端以當下 map←odom 換算；定位修正會使地圖座標的線移動）
                R_ = self._t_region
                mv = max(float(np.linalg.norm(L[0] - R_['p0'])), float(np.linalg.norm(L[1] - R_['p1'])))
                R_['p0'], R_['p1'] = L[0], L[1]
                if mv > 0.005:
                    self._t_reg_t = -9.0                              # 移動超過 5 mm ⇒ 下一個區域週期立即重設參考路徑
        elif self._t_region is not None and d.get('region_clear'):
            self._t_region_off('cleared')
        self._t_ack(True, 'conditions', heading=round(self._t_heading, 5), box=tgt.round(4).tolist(), need_s=round(need, 3), region=None if self._t_region is None else self._t_region['id'])
        self._t_status()

    def _t_lose(self, why):
        if self._t_mode == 'idle':
            return
        self._t_mode = 'lost'
        self._t_box_tgt = np.zeros((3, 2))                          # 漸收到零（停下）；朝向保持最後任務值（不回路徑前視）
        if self._t_box_eff is None:
            u = np.asarray(self.xi_prev, float)
            self._t_box_eff = np.c_[np.minimum(u, 0.0), np.maximum(u, 0.0)]
        self._t_ack(True, 'lost:' + why)
        self._t_status()

    def _t_tick(self):
        if self._t_mode == 'task' and self._t_until is not None and self._t_now() > self._t_until:
            self._t_lose('conditions_expired')

    def _t_box_step(self):
        """生效框每控制週期以 a_max·dt 向目標框收緊（放寬則立即到目標，但仍不超過原限制）。"""
        if self._t_box_tgt is None:
            return None
        if self._t_box_eff is None:
            self._t_box_eff = self._t_box_tgt.copy()
        da = self._t_amax() * float(self.dt)
        e, g = self._t_box_eff, self._t_box_tgt
        lo = np.where(g[:, 0] > e[:, 0], np.minimum(g[:, 0], e[:, 0] + da), g[:, 0])
        hi = np.where(g[:, 1] < e[:, 1], np.maximum(g[:, 1], e[:, 1] - da), g[:, 1])
        self._t_box_eff = np.c_[lo, hi]
        return self._t_box_eff

    # ---- 覆寫：朝向參考（同一套環繞＋變化率限制）
    def _desired_yaw(self, path_xyth, robot_xyth, now_s):
        if self._t_mode in ('task', 'lost') and self._t_heading is not None and self.heading_enable:
            raw = float(self._t_heading)
            self._yaw_raw = raw
            if self._yaw_ref is None or self._yaw_ref_t is None:
                self._yaw_ref, self._yaw_ref_t = raw, now_s
                return raw
            dt_ref = max(now_s - self._yaw_ref_t, 1e-3)
            step = self._wrap(raw - self._yaw_ref)
            lim = self.heading_rate * dt_ref
            self._yaw_ref = self._wrap(self._yaw_ref + max(-lim, min(lim, step)))
            self._yaw_ref_t = now_s
            return self._yaw_ref
        return super()._desired_yaw(path_xyth, robot_xyth, now_s)

    # ---- 區域模式
    L_CONV = 0.15

    def _plan_cb(self, msg):
        if self._t_region is not None and not self._t_internal:
            self._t_ext_plan = msg                                     # 區域模式：外部路徑只暫存
            return
        return super()._plan_cb(msg)

    def _t_region_off(self, why):
        if self._t_region is None:
            return
        self._t_region = None
        if self._t_ext_plan is not None:
            self._t_internal = True
            try:
                super()._plan_cb(self._t_ext_plan)
            finally:
                self._t_internal = False
        self.get_logger().info('[task] region off: ' + why)

    def _t_robot_map(self):
        try:
            tr = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:                                             # noqa: BLE001
            return None
        o = tr.transform.rotation
        return np.array([tr.transform.translation.x, tr.transform.translation.y, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))])

    def _t_region_geom(self, xy):
        R = self._t_region
        u = R['p1'] - R['p0']; Lr = float(np.linalg.norm(u)); u = u / Lr
        rel = np.asarray(xy, float) - R['p0']
        s_ = float(rel @ u); lat = float(u[0] * rel[1] - u[1] * rel[0])
        return s_, lat, Lr, u

    def _t_pose(self, xy, yaw):
        m = self._t_PS(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1]); m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m

    def _t_region_tick(self):
        R = self._t_region
        if R is None:
            return
        rb = self._t_robot_map()
        if rb is None:
            return
        s_, lat, Lr, u = self._t_region_geom(rb[:2])
        inside = abs(lat) <= R['w'] and 0.0 <= s_ <= Lr
        if R['phase'] == 'approach' and abs(lat) <= 2 * R['w'] and s_ >= R.get('s_E', 0.0) - 0.05:
            R['phase'] = 'inside'
            self._t_reg_t = -9.0
        self._t_reg_pub.publish(String(data=json.dumps(dict(id=R['id'], t=round(self._t_now(), 3), s=round(s_, 4), lat=round(lat, 4), w=R['w'], inside=bool(inside), phase=R['phase'], n_plan=R['n_plan'],
                                                            s_E=R.get('s_E'), enter_by=R['enter_by'], late=bool(s_ > R['enter_by'] and not inside)))))
        now = self._t_now()
        if now - self._t_reg_t < 1.0 or self._t_reg_busy:
            return
        self._t_reg_t = now
        yaw_line = math.atan2(u[1], u[0])
        if R['phase'] == 'inside':                                   # 已收斂：參考＝走廊（自目前投影起）＋延伸
            self._t_set_ref([], max(s_, 0.0), Lr, u, yaw_line)
            return
        s_E = float(np.clip(min(max(s_, 0.0) + self.L_CONV, R['enter_by'] - self.L_CONV), 0.0, max(Lr - 0.10, 0.0)))   # 不晚於操作端給的最晚進入點（減收斂距離）
        R['s_E'] = s_E
        E = R['p0'] + s_E * u
        gl = self._t_CP.Goal(); gl.start = self._t_pose(rb[:2], rb[2]); gl.goal = self._t_pose(E, yaw_line); gl.use_start = True; gl.planner_id = 'GridBased'
        if not self._t_cp.server_is_ready():
            return
        self._t_reg_busy = True
        fut = self._t_cp.send_goal_async(gl)

        def _acc(f, s_E=s_E, yaw_line=yaw_line):
            gh = f.result()
            if gh is None or not gh.accepted:
                self._t_reg_busy = False; return

            def _done(r):
                self._t_reg_busy = False
                try:
                    P = [(p.pose.position.x, p.pose.position.y) for p in r.result().result.path.poses]
                except Exception:                                     # noqa: BLE001
                    P = []
                if len(P) >= 2 and self._t_region is not None:
                    s2, _, L2, u2 = self._t_region_geom(self._t_region['p0'])
                    self._t_set_ref(P, s_E, L2, u2, yaw_line)
            gh.get_result_async().add_done_callback(_done)
        fut.add_done_callback(_acc)

    def _t_set_ref(self, P, s_from, Lr, u, yaw_line):
        R = self._t_region
        pts = list(P)
        for ss in np.arange(s_from, Lr + R['ext'] + 1e-9, 0.02):
            pts.append(tuple(R['p0'] + ss * u))
        m = self._t_Path(); m.header.frame_id = 'map'; m.header.stamp = self.get_clock().now().to_msg()
        for k, (x, y) in enumerate(pts):
            x2, y2 = pts[min(k + 1, len(pts) - 1)]
            th = math.atan2(y2 - y, x2 - x) if (x2, y2) != (x, y) else yaw_line
            ps = self._t_PS(); ps.header = m.header; ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            m.poses.append(ps)
        self._t_internal = True
        try:
            super()._plan_cb(m)
        finally:
            self._t_internal = False
        R['n_plan'] += 1

    # ---- 覆寫：控制步（求解器限制取交集，求解後還原）
    def _control_step(self):
        box = self._t_box_step()
        if box is None:
            return super()._control_step()
        lo0, hi0 = self._cfg.u_min, self._cfg.u_max
        try:
            self._cfg.u_min = np.maximum(np.asarray(lo0, float), box[:, 0])
            self._cfg.u_max = np.minimum(np.asarray(hi0, float), box[:, 1])
            return super()._control_step()
        finally:
            self._cfg.u_min, self._cfg.u_max = lo0, hi0


def main():
    rclpy.init()
    n = GMPCTaskNode()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
