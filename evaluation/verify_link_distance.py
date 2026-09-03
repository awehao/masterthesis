"""Section: acceptance for the link 3D distance interface (item 2).

Four claims, each checked against something the node cannot influence:

  1  distance and normal match the scene geometry
     Recompute d_i and n_i from the world SDF independently of the node and
     compare. A normal that points the wrong way is the failure that matters:
     it flips the sign of n^T J v and turns the safety constraint into a
     command to approach.

  2  frames and timestamps are right
     p_i must equal TF(report_frame -> detect frame). Getting this wrong is
     invisible until the base moves, because in a fixed-base test every frame
     coincides.

  3  arm-occluded directions are UNKNOWN, never FREE
     Put the arm in a pose that blocks the LiDAR and confirm affected points
     report UNKNOWN rather than a clear distance.

  4  staleness is detectable
     Stop the obstacle pose feed and confirm the status changes.

    python3 evaluation/verify_link_distance.py
"""
from __future__ import annotations

import math
import re
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_detection_points import (  # noqa: E402
    Obstacle, _closest_local, _inv, _iso, _quat_to_rot, _rpy_to_rot)

FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded']
STATUS = {0: 'OK', 1: 'UNKNOWN', 2: 'STALE', 3: 'NODATA'}
TOL_D = 2e-3
TOL_N = 2e-3


def obstacles_from_world(path: str) -> list[Obstacle]:
    sdf = open(path).read()
    out = []
    for m in re.finditer(r'<model name="(obs_\d+)">(.*?)</model>', sdf, re.S):
        name, body = m.group(1), m.group(2)
        pose = re.search(r'<pose>([-\d.eE\s]+)</pose>', body)
        if not pose:
            continue
        v = [float(x) for x in pose.group(1).split()]
        xyz, rpy = np.array(v[:3]), np.array(v[3:6] if len(v) >= 6 else [0, 0, 0])
        box = re.search(r'<box><size>([^<]+)</size>', body)
        cyl = re.search(r'<cylinder><radius>([\d.]+)</radius>\s*<length>([\d.]+)', body)
        o = Obstacle(name=name, model='', kind='')
        if box:
            o.kind = 'box'
            o.size = np.array([float(x) for x in box.group(1).split()])
        elif cyl:
            o.kind = 'cylinder'
            o.radius, o.height = float(cyl.group(1)), float(cyl.group(2))
        else:
            continue
        o.T_world_link = _iso(_rpy_to_rot(*rpy), xyz)
        o.T_link_collision = np.eye(4)
        out.append(o)
    return out


class V(Node):
    def __init__(self):
        super().__init__('verify_link_distance')
        self.pc = None
        self.diag = None
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 lambda m: setattr(self, 'pc', m), 10)
        self.create_subscription(Float32MultiArray, '/arm_link_distance/diag',
                                 lambda m: setattr(self, 'diag', m), 10)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)

    def spin(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.05)

    def tf(self, a, b):
        try:
            t = self.buf.lookup_transform(a, b, rclpy.time.Time())
        except Exception:
            return None
        q, tr = t.transform.rotation, t.transform.translation
        return _iso(_quat_to_rot(q.x, q.y, q.z, q.w),
                    np.array([tr.x, tr.y, tr.z]))


def unpack(msg):
    a = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.width, len(FIELDS))
    return a.astype(float)


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else \
        'src/ammr_bringup/worlds/random_room.sdf'
    frames = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
              'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
              'detect5', 'detect6']
    obs = obstacles_from_world(world)
    rclpy.init()
    n = V()
    n.spin(5.0)
    if n.pc is None:
        print('  no /arm_link_distance/points -- is the node running?',
              file=sys.stderr)
        return 2
    A = unpack(n.pc)
    print(f'  {A.shape[0]} 個偵測點   frame {n.pc.header.frame_id}   '
          f'場景 {len(obs)} 個障礙物')

    fails = []
    print(f'\n  [1][2] 距離／法向量 vs 場景幾何，位置 vs TF')
    print(f'    {"frame":10}{"d 節點":>9}{"d 獨立":>9}{"Δd":>10}'
          f'{"Δn":>10}{"Δp(TF)":>10}{"status":>9}')
    for i, fr in enumerate(frames):
        x, y, z, nx, ny, nz, d, st, age = A[i]
        T = n.tf(n.pc.header.frame_id, fr)
        dp = float(np.abs(np.array([x, y, z]) - T[:3, 3]).max()) if T is not None else float('nan')
        p = np.array([x, y, z])
        best_d, best_v = math.inf, None
        for o in obs:
            T_wc = o.T_world_link @ o.T_link_collision
            p_loc = (_inv(T_wc) @ np.append(p, 1.0))[:3]
            surf = (T_wc @ np.append(_closest_local(o, p_loc), 1.0))[:3]
            v = surf - p
            dd = float(np.linalg.norm(v))
            if dd < best_d:
                best_d, best_v = dd, v
        nn = best_v / max(best_d, 1e-9)
        dd_err = abs(d - best_d)
        dn_err = float(np.abs(np.array([nx, ny, nz]) - nn).max())
        ok = (st != 0) or (dd_err <= TOL_D and dn_err <= TOL_N and dp <= 5e-3)
        print(f'    {fr:10}{d:9.4f}{best_d:9.4f}{dd_err:10.2e}{dn_err:10.2e}'
              f'{dp:10.2e}{STATUS.get(int(st),"?"):>9}{"" if ok else "  ✗"}')
        if not ok:
            fails.append(fr)

    if n.diag:
        q = n.diag.data
        print(f'\n  診斷: n={int(q[0])} ok={int(q[1])} unknown={int(q[2])} '
              f'stale={int(q[3])} nodata={int(q[4])} worst_age={q[5]:.3f}s '
              f'min_d={q[6]:.3f}m')

    print('\n  ' + ('[1][2] 通過 ✓' if not fails else f'★ 失敗: {fails}'))
    n.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
