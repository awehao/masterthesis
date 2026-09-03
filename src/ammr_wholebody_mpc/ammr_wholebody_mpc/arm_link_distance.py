"""Link-level 3D distance interface for the whole-body safety constraint.

Publishes, for each of the twelve arm detection points, exactly what the
constraint in the Phase 2 plan section 8.2 consumes:

    n_i^T J_{p_i}(q) v  <=  alpha_i (d_i - d_stop_i)

    p_i   the point on the arm, in the reporting frame
    d_i   distance from p_i to the nearest obstacle SURFACE
    n_i   unit vector from p_i toward that surface
    t_i   when the obstacle pose it was computed from was measured

This is not the same interface as arm_detection_points.py. That node emits a
PolygonStamped whose point i is a free vector expressed in detection frame i --
a layout fixed by the policy that was trained against it, which cannot carry a
distance, a frame, a timestamp or a validity flag. A safety constraint needs
all four, so this is a separate output rather than a change to that one.

Three properties the safety layer depends on, none of them free:

  one frame, stated       p_i and n_i are published in `report_frame` (default
                          odom). The Jacobian used downstream is expressed in
                          the same frame, and odom is inertial -- base_link is
                          not, once the base moves, and the constraint would
                          need extra terms nobody would remember to add.

  occlusion is not free   space Where the arm blocks the LiDAR, removing its own
                          returns leaves NO measurement. A point whose nearest
                          surface lies in an occluded bearing is marked UNKNOWN,
                          never FREE. Reading the removal as clearance is
                          exactly the failure the self-filter would otherwise
                          introduce.

  staleness is visible    Every point carries the age of the data behind it, and
                          the message carries the worst age. A consumer that
                          cannot see staleness will happily plan against a
                          distance measured a second ago.

Output: sensor_msgs/PointCloud2 on ~/points, one point per detection frame,
fields  x y z nx ny nz d status age. PointCloud2 because it is one atomic
message with a header (frame and stamp), it never desynchronises the way two
parallel topics do, and Foxglove renders it without extra tooling.

  status  0 = OK, 1 = UNKNOWN (occluded), 2 = STALE, 3 = NO DATA

Distance source is gz ground truth / the known world model, as the plan
requires for this stage. That is a modelling input, not perception: it is here
to validate the interface and the constraint, and must be replaced by a
verified depth source before any claim about unknown 3D environments.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener

from .arm_detection_points import (Obstacle, _closest_local, _inv, _iso,
                                   _quat_to_rot, _rpy_to_rot)

STATUS_OK, STATUS_UNKNOWN, STATUS_STALE, STATUS_NODATA = 0.0, 1.0, 2.0, 3.0

BEST_EFFORT = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)

# `status` is about the DATA: is d_i usable at all. `occluded` is about
# OBSERVABILITY: is the direction toward that surface currently unseen, so that
# something not in the model could be nearer than d_i says.
#
# These were one field at first and it was wrong. With a known scene model the
# distance to a modelled box is valid whether or not the arm blocks the line of
# sight -- the model does not need to see it. Folding occlusion into `status`
# marked every point UNKNOWN the moment the arm crossed the beam, discarding
# distances that were perfectly good. Separated, item 3 can use d_i normally
# and additionally refuse to treat an occluded direction as clear.
FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded']


class ArmLinkDistance(Node):

    def __init__(self) -> None:
        super().__init__('arm_link_distance')
        p = self.declare_parameter
        p('detection_frames', [
            'detect0_1', 'detect0_2', 'detect1',
            'detect2_1', 'detect2_2', 'detect2_3',
            'detect3_1', 'detect3_2',
            'detect4_1', 'detect4_2',
            'detect5', 'detect6'])
        # Inertial by construction. See the module docstring: base_link would
        # make the constraint wrong the moment the base moves.
        p('report_frame', 'odom')
        p('lidar_frame', 'lidar_link')
        p('obstacles', [''])
        p('publish_rate', 30.0)
        p('pose_timeout', 0.5)      # s, obstacle pose older than this is stale
        p('occl_timeout', 0.5)      # s, occlusion info older than this is unusable
        # Absence of the occlusion feed is NOT evidence of no occlusion. With
        # the arm mounted this node cannot tell a clear bearing from one the
        # arm is standing in unless the self-filter is reporting, so it must
        # refuse to call anything OK rather than default to clear. Set false
        # only when running without the arm, where nothing can occlude.
        p('require_occlusion_feed', True)
        p('max_range', 3.0)         # m, beyond this a point reports NO DATA

        g = lambda k: self.get_parameter(k).value
        self.frames = list(g('detection_frames'))
        self.report_frame = str(g('report_frame'))
        self.lidar_frame = str(g('lidar_frame'))
        self.pose_timeout = float(g('pose_timeout'))
        self.occl_timeout = float(g('occl_timeout'))
        self.require_occl = bool(g('require_occlusion_feed'))
        self.max_range = float(g('max_range'))

        self.obstacles = self._parse([s for s in g('obstacles') if s.strip()])
        self._stamp: dict[str, float] = {}
        self._occl = None            # (angle_min, inc, n, set(indices))
        self._occl_t = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        for model in sorted({o.model for o in self.obstacles if o.model}):
            self.create_subscription(PoseStamped, f'/model/{model}/pose',
                                     self._make_cb(model), BEST_EFFORT)
        self.create_subscription(Float32MultiArray,
                                 '/scan_self_filter/occluded',
                                 self._on_occl, 10)
        self.pub = self.create_publisher(PointCloud2, '~/points', 10)
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)

        rate = max(1.0, float(g('publish_rate')))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'arm_link_distance: {len(self.frames)} points, '
            f'{len(self.obstacles)} obstacles, frame {self.report_frame}')

    # ------------------------------------------------------------- config
    # Two kinds of obstacle, distinguished by whether a model name is given:
    #
    #   name:model:kind:dims[:xyz:rpy]   pose arrives on /model/<model>/pose and
    #                                    carries an age, so it can go stale
    #   name::kind:dims:xyz:rpy          STATIC: the pose IS the world pose,
    #                                    taken from the scene model, never stale
    #
    # The static form is what the plan calls a known scene model. It is a
    # modelling input, not a measurement, and it is marked as such: age 0 is
    # honest here precisely because nothing is being measured.
    def _parse(self, specs: list[str]) -> list[Obstacle]:
        out = []
        for spec in specs:
            f = spec.split(':')
            if len(f) < 4:
                raise ValueError(f'obstacle spec needs >=4 fields: {spec!r}')
            o = Obstacle(name=f[0], model=f[1], kind=f[2])
            nums = [float(v) for v in f[3].replace(' ', '').split(',') if v]
            if o.kind == 'box':
                o.size = np.array(nums)
            elif o.kind == 'cylinder':
                o.radius, o.height = nums
            elif o.kind == 'sphere':
                o.radius = nums[0]
            else:
                raise ValueError(f'unsupported type {o.kind!r}')
            xyz = np.array([float(v) for v in f[4].split(',')]) if len(f) > 4 and f[4].strip() else np.zeros(3)
            rpy = np.array([float(v) for v in f[5].split(',')]) if len(f) > 5 and f[5].strip() else np.zeros(3)
            if o.model:
                o.T_link_collision = _iso(_rpy_to_rot(*rpy), xyz)
            else:
                # Static: xyz/rpy is the world pose of the collision body.
                o.T_world_link = _iso(_rpy_to_rot(*rpy), xyz)
                o.T_link_collision = np.eye(4)
            out.append(o)
        return out

    def _make_cb(self, model: str):
        def cb(msg: PoseStamped) -> None:
            q, t = msg.pose.orientation, msg.pose.position
            T = _iso(_quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))
            self._stamp[model] = self.get_clock().now().nanoseconds * 1e-9
            for o in self.obstacles:
                if o.model == model:
                    o.T_world_link = T
        return cb

    def _on_occl(self, msg: Float32MultiArray) -> None:
        d = list(msg.data)
        if len(d) < 3:
            return
        self._occl = (d[0], d[1], int(d[2]), set(int(v) for v in d[3:]))
        self._occl_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- geometry
    def _tf(self, parent: str, child: str):
        try:
            t = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        except Exception:
            return None
        q, tr = t.transform.rotation, t.transform.translation
        return _iso(_quat_to_rot(q.x, q.y, q.z, q.w),
                    np.array([tr.x, tr.y, tr.z]))

    def _occluded(self, p_lidar: np.ndarray) -> bool:
        """Is the bearing from the LiDAR toward this surface point one the arm
        was standing in? If the occlusion feed is stale we cannot tell, and
        'cannot tell' is not 'clear'."""
        if self._occl is None:
            # Never heard from the self-filter. Unknown, not clear.
            return self.require_occl
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._occl_t > self.occl_timeout:
            return True
        a0, inc, n, idx = self._occl
        if not idx or inc == 0.0:
            return False
        b = math.atan2(p_lidar[1], p_lidar[0])
        i = int(round((b - a0) / inc))
        return any((i + k) % n in idx for k in (-1, 0, 1))

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        T_rl = self._tf(self.report_frame, self.lidar_frame)
        rows = []
        worst_age = 0.0
        n_ok = n_unk = n_stale = n_nodata = 0

        for fr in self.frames:
            T = self._tf(self.report_frame, fr)
            if T is None:
                rows.append([0.0] * 6 + [0.0, STATUS_NODATA, -1.0, 0.0])
                n_nodata += 1
                continue
            p = T[:3, 3]

            best_d, best_v, best_age = math.inf, None, 0.0
            for o in self.obstacles:
                if o.T_world_link is None:
                    continue
                age = 0.0 if not o.model else now - self._stamp.get(o.model, 0.0)
                T_wc = o.T_world_link @ o.T_link_collision
                p_loc = (_inv(T_wc) @ np.append(p, 1.0))[:3]
                surf = (T_wc @ np.append(_closest_local(o, p_loc), 1.0))[:3]
                v = surf - p
                dist = float(np.linalg.norm(v))
                if dist < best_d:
                    best_d, best_v, best_age = dist, v, age

            if best_v is None or best_d > self.max_range:
                rows.append(list(p) + [0.0, 0.0, 0.0, self.max_range,
                                       STATUS_NODATA, -1.0, 0.0])
                n_nodata += 1
                continue

            n_hat = best_v / max(best_d, 1e-9)
            status = STATUS_OK
            if best_age > self.pose_timeout:
                status = STATUS_STALE
                n_stale += 1
            else:
                n_ok += 1
            occ = 0.0
            if T_rl is not None:
                p_l = (_inv(T_rl) @ np.append(p + best_v, 1.0))[:3]
                if self._occluded(p_l):
                    occ = 1.0
                    n_unk += 1
            worst_age = max(worst_age, best_age)
            rows.append(list(p) + list(n_hat) + [best_d, status, best_age, occ])

        self._publish(rows)
        d = Float32MultiArray()
        #  0 n_points 1 ok 2 occluded 3 stale 4 nodata 5 worst_age 6 min_d
        finite = [r[6] for r in rows if r[7] == STATUS_OK]
        d.data = [float(len(rows)), float(n_ok), float(n_unk), float(n_stale),
                  float(n_nodata), float(worst_age),
                  float(min(finite)) if finite else -1.0]
        self.diag.publish(d)

    def _publish(self, rows: list[list[float]]) -> None:
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.report_frame
        msg.height = 1
        msg.width = len(rows)
        msg.fields = [PointField(name=n, offset=4 * i,
                                 datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(FIELDS)]
        msg.is_bigendian = False
        msg.point_step = 4 * len(FIELDS)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = np.array(rows, dtype=np.float32).tobytes()
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ArmLinkDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
