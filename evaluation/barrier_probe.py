"""Drive the arm at the box and record what the barrier does about it.

This is the INPUT side only. It publishes /wholebody_safety/cmd_in and never
touches the controller: the safety node filters, the adapter reshapes, the gate
publishes. Nothing here can move a joint that the barrier did not agree to.

The three phases exist because they separate effects that a single approach run
confounds:

  approach     TCP driven straight at the nearest face. The barrier should bite
               and keep biting.
  tangential   same speed, across the face. Distance barely changes, so a
               barrier that is really constraining the APPROACH RATE n^T J v
               and not merely the distance should mostly let this through.
  retreat      straight away from the face. The row is n^T J v <= alpha(...);
               moving away makes the left side negative, so it must never bind.
               A barrier that constrains retreat is the penetration sign error
               from section 10.11 coming back.

Per cycle it records the input and output commands, the barrier residual before
and after, which rows bound, the model's min(d - rho), the independent upper
bound, and the TCP speed measured from TF rather than from the command -- a
command that was reduced proves nothing about whether the arm actually slowed.

At the first cycle where the input violates a row, the output satisfies it, AND
the measured TCP speed is falling, a full snapshot is written: joint state,
every reported row, the binding set, both commands, both bounds. That is the
cycle the whole test exists to produce, and it is the one that needs to survive
for the offline exact-mesh check.

    python3 evaluation/barrier_probe.py --speed 0.05 --phase-s 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray
from tf2_ros import Buffer, TransformListener

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))
sys.path.insert(0, _HERE)

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
STATUS_OK = 0.0


def _expand(path):
    import subprocess
    return (subprocess.check_output(['xacro', path], text=True)
            if path.endswith('.xacro') else open(path).read())


def _quat_to_rot(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Probe(Node):

    def __init__(self, a) -> None:
        super().__init__('barrier_probe')
        self.a = a
        xml = _expand(a.urdf)
        self.K = WholeBodyKinematics.from_urdf_string(xml)
        self.n = len(self.K.dof_names)
        self.idx = [self.K.dof_names.index(j) for j in ARM]

        self.q = None
        self.q_t = 0.0
        self.rows = None
        self.binding = set()
        self.r_in = self.r_out = 0.0
        self.cmd_out = None
        self.min_d = float('nan')
        self.ub = float('nan')
        self.tcp_prev = None
        self.tcp_prev_t = None
        self.tcp_speed = 0.0

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 self._on_pts, 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/barrier',
                                 self._on_barrier, 10)
        self.create_subscription(Float64MultiArray, '/wholebody_safety/cmd_out',
                                 self._on_out, 10)
        self.create_subscription(Float32, '/barrier_viz/clearance_ub',
                                 self._on_ub, 10)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.log = []
        self.snapshot = None

    # ---------------------------------------------------------------- inputs
    def _on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if all(j in ix for j in ARM) and len(m.position) > max(ix[j] for j in ARM):
            self.q = np.array([m.position[ix[j]] for j in ARM])
            self.q_t = time.monotonic()

    def _on_pts(self, m):
        nf = m.point_step // 4
        self.rows = (np.frombuffer(m.data, dtype=np.float32).reshape(m.width, nf)
                     if m.width else None)
        if self.rows is not None:
            ok = self.rows[:, 7] == STATUS_OK
            rho = self.rows[:, 14] if nf >= 15 else np.zeros(len(self.rows))
            self.min_d = float((self.rows[ok, 6] - rho[ok]).min()) if ok.any() \
                else float('nan')

    def _on_barrier(self, m):
        b = np.asarray(m.data, dtype=float).reshape(-1, 3)
        self.binding = {int(i) for i, ri, _ in b if ri > 1e-9}
        self.r_in = float(b[:, 1].max()) if len(b) else 0.0
        self.r_out = float(b[:, 2].max()) if len(b) else 0.0

    def _on_out(self, m):
        self.cmd_out = np.asarray(m.data, dtype=float)

    def _on_ub(self, m):
        self.ub = float(m.data)

    # ------------------------------------------------------------------ util
    def _tcp(self):
        try:
            t = self.tf_buffer.lookup_transform(self.a.report_frame, self.a.tcp,
                                                rclpy.time.Time())
        except Exception:
            return None
        r = t.transform.translation
        return np.array([r.x, r.y, r.z])

    def _measure_tcp(self):
        p = self._tcp()
        now = time.monotonic()
        if p is None:
            return
        if self.tcp_prev is not None and now > self.tcp_prev_t + 1e-3:
            self.tcp_speed = float(np.linalg.norm(p - self.tcp_prev)
                                   / (now - self.tcp_prev_t))
        self.tcp_prev, self.tcp_prev_t = p, now

    def _q9(self):
        q = np.zeros(self.n)
        q[self.idx] = self.q
        return q

    def _joint_cmd(self, v_tcp):
        """Damped least squares on the arm DOFs only; base stays at zero.

        Damped, not pseudo-inverse: near a singularity the exact inverse asks
        for joint speeds the velocity box would then clip in a direction nobody
        chose, and the resulting command is neither the requested TCP motion nor
        a safe one.
        """
        q = self._q9()
        J = self.K.jacobian(q, self.a.tcp)[:3][:, self.idx]      # 3 x 6
        lam = self.a.damping
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(3), v_tcp)
        # The probe's own limit, well under the machine's, so that a mistake
        # here shows up as a slow wrong motion rather than a fast one.
        cap = min(self.a.joint_vmax, float(LITE6_SAFE.max_velocity.min()))
        s = cap / max(float(np.abs(dq).max()), 1e-9)
        if s < 1.0:
            dq = dq * s
        out = np.zeros(self.n)
        out[self.idx] = dq
        return out

    def _guard(self):
        if self.q is None or time.monotonic() - self.q_t > 0.3:
            return 'JointState 超過 0.3 s 未更新'
        lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
        if np.any(self.q < lo + 0.05) or np.any(self.q > hi - 0.05):
            return f'關節接近安全位置限位: {np.round(self.q, 3).tolist()}'
        if self.rows is None:
            return '尚未收到距離資料'
        if np.isfinite(self.min_d) and self.min_d < self.a.abort_d:
            return f'模型間距 {self.min_d*1000:.1f} mm 低於中止門檻'
        return None

    # ------------------------------------------------------------------- run
    def run(self):
        a = self.a
        # Fixed directions in the report frame, NOT read off the barrier's own
        # normals. +x is straight at the box face in this world. Taking the
        # direction from the reported rows would make the probe agree with the
        # safety model by construction, and the point of the run is to see
        # whether they agree.
        dirs = {'approach': np.array([1.0, 0.0, 0.0]),
                'tangential': np.array([0.0, 1.0, 0.0]),
                'retreat': np.array([-1.0, 0.0, 0.0])}
        dt = 1.0 / a.rate
        print(f'  相位各 {a.phase_s:.0f} s，TCP 目標速度 {a.speed*1000:.0f} mm/s，'
              f'中止門檻 {a.abort_d*1000:.0f} mm')
        for phase in a.phases:
            u = dirs[phase] * a.speed
            print(f'\n── {phase} ── 方向 {dirs[phase].tolist()}', flush=True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < a.phase_s:
                rclpy.spin_once(self, timeout_sec=dt * 0.5)
                why = self._guard()
                if why:
                    self._stop()
                    print(f'  中止：{why}', flush=True)
                    return False
                self._measure_tcp()
                # Ramp in, so the first cycle is not a step the acceleration box
                # has to absorb and then get blamed for.
                k = min(1.0, (time.monotonic() - t0) / max(a.ramp_s, 1e-3))
                v = self._joint_cmd(u * k)
                m = Float64MultiArray()
                m.data = [float(x) for x in v]
                self.pub.publish(m)
                self._record(phase, v)
                time.sleep(max(0.0, dt - 0.002))
            self._stop()
            time.sleep(0.5)
        return True

    def _stop(self):
        m = Float64MultiArray()
        m.data = [0.0] * self.n
        for _ in range(5):
            self.pub.publish(m)
            rclpy.spin_once(self, timeout_sec=0.01)

    def _record(self, phase, v_in):
        rec = {'t': time.monotonic(), 'phase': phase,
               'cmd_in': [float(x) for x in v_in],
               'cmd_out': None if self.cmd_out is None
                          else [float(x) for x in self.cmd_out],
               'resid_in': self.r_in, 'resid_out': self.r_out,
               'n_binding': len(self.binding),
               'min_d_model': self.min_d, 'clearance_ub': self.ub,
               'tcp_speed_measured': self.tcp_speed,
               'q': None if self.q is None else [float(x) for x in self.q]}
        self.log.append(rec)
        if self.snapshot is None and self._is_the_cycle():
            self.snapshot = dict(rec)
            self.snapshot['rows'] = self.rows.tolist()
            self.snapshot['binding'] = sorted(self.binding)
            self.snapshot['note'] = (
                '輸入違反屏障、輸出滿足、且實測 TCP 速度較前一週期下降的第一個週期')
            print(f'  ** 快照：resid_in {self.r_in*1000:+.2f} mm/s，'
                  f'resid_out {self.r_out*1000:+.2f} mm/s，'
                  f'{len(self.binding)} 列作用，'
                  f'TCP {self.tcp_speed*1000:.1f} mm/s', flush=True)

    def _is_the_cycle(self):
        if self.r_in <= 1e-9 or self.r_out > 1e-6 or self.rows is None:
            return False
        prev = [r for r in self.log[-6:-1] if r['tcp_speed_measured'] > 0]
        return bool(prev) and self.tcp_speed < 0.9 * max(
            r['tcp_speed_measured'] for r in prev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='src/my_omnibot_description/urdf/'
                                      'omni_bot_wholebody.urdf.xacro')
    ap.add_argument('--report-frame', default='world')
    ap.add_argument('--tcp', default='uflite_gripper_link')
    ap.add_argument('--speed', type=float, default=0.05, help='TCP m/s')
    ap.add_argument('--phase-s', type=float, default=6.0)
    ap.add_argument('--ramp-s', type=float, default=1.0)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--damping', type=float, default=0.05)
    ap.add_argument('--joint-vmax', type=float, default=0.4, help='rad/s')
    ap.add_argument('--abort-d', type=float, default=0.02,
                    help='model min(d-rho) below this aborts the run')
    ap.add_argument('--phases', nargs='+',
                    default=['approach', 'tangential', 'retreat'])
    ap.add_argument('--out', default='evaluation/results/barrier_probe.json')
    a = ap.parse_args()

    rclpy.init()
    p = Probe(a)
    print('  等待 /joint_states 與距離資料…', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15 and (p.q is None or p.rows is None):
        rclpy.spin_once(p, timeout_sec=0.1)
    if p.q is None or p.rows is None:
        print(f'  沒有等到資料（q={p.q is not None}, rows={p.rows is not None}）'
              '：堆疊沒起來，不下命令。', file=sys.stderr)
        p._stop()
        rclpy.try_shutdown()
        return 1
    ok = False
    try:
        ok = p.run()
    except KeyboardInterrupt:
        print('\n  中斷', flush=True)
    finally:
        p._stop()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({'args': vars(a), 'completed': ok, 'log': p.log,
                   'snapshot': p.snapshot},
                  open(a.out, 'w'), ensure_ascii=False)
        print(f'\n  {len(p.log)} 個週期寫入 {a.out}'
              + ('' if p.snapshot else '；沒有出現屏障作用的快照週期'), flush=True)
        rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
