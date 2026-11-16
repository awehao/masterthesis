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
  - /task/conditions_ack：每個條件的採用／拒收與原因；/task/status：idle／task／lost。"""
import json
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
        self._t_ack(True, 'conditions', heading=round(self._t_heading, 5), box=tgt.round(4).tolist(), need_s=round(need, 3))
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
