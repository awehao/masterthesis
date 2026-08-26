"""Closest-obstacle vectors at sample points along the arm.

Port of collision_detection_points_node.cpp from the ROS 1 MastersThesis_ros
tree (my_mobile_manipulator), rewritten for ROS 2 Jazzy + gz Harmonic.

What it does, unchanged from the original: a set of "detection points" is
attached to the arm links (see arm_detection_points.xacro). Every cycle, for
each point, it finds the nearest point on the surface of any obstacle and
publishes the vector FROM the detection point TO that surface, expressed in
that detection point's own frame. Twelve points -> twelve 3-vectors, packed
into a PolygonStamped in a fixed order. The header frame_id is deliberately
not a real TF frame; each polygon point lives in its own detection frame.

That layout is not a design choice we are free to change -- it is the
observation vector an arm-avoidance policy was trained against, so the point
order and the per-point frame convention have to match the training rig
exactly. The offsets themselves live in the xacro.

Three things had to change in the port:

  poses     ROS 1 read every obstacle from /gazebo/link_states, one message
            carrying the whole world. gz Harmonic has no equivalent, so we
            subscribe per obstacle to /model/<name>/pose, which the dynamic
            launch already bridges as PoseStamped for the navigation metrics.
            Same information, one topic per obstacle instead of one global.

  base pose ROS 1 took the robot base pose from link_states too, i.e. ground
            truth, and composed base->detection from TF. Ground truth is fine
            for a training rig but wrong for anything we report, so the base
            pose comes from TF (map->base_link) by default. Set
            use_gz_base_pose:=true to reproduce the original behaviour.

  shutdown  the original called ros::shutdown() on a bad config. Here a bad
            config raises, so the failure is visible in the launch instead of
            leaving a live node publishing NaN.

A point with no obstacle in range publishes NaN rather than a large number, so
a consumer can tell "nothing near" from "something 10 m away" -- also
unchanged, because the policy was trained on that distinction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PolygonStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

NAN3 = (float('nan'),) * 3


def _rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _iso(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return _iso(R.T, -R.T @ T[:3, 3])


# --------------------------------------------------------------- obstacles
@dataclass
class Obstacle:
    """One collision primitive, posed relative to a gz model frame."""
    name: str
    model: str                       # gz model name -> /model/<model>/pose
    kind: str                        # 'box' | 'cylinder' | 'sphere'
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.0
    height: float = 0.0
    T_link_collision: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_world_link: np.ndarray | None = None   # filled by the pose subscription


def _closest_on_box(p: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = 0.5 * size
    if np.all(np.abs(p) <= half):
        # Inside: project to the nearest face rather than returning p itself,
        # so the vector keeps pointing out of the obstacle instead of
        # collapsing to zero the moment a point penetrates.
        d = half - np.abs(p)
        i = int(np.argmin(d))
        q = p.copy()
        q[i] = half[i] if p[i] >= 0.0 else -half[i]
        return q
    return np.clip(p, -half, half)


def _closest_on_sphere(p: np.ndarray, radius: float) -> np.ndarray:
    n = float(np.linalg.norm(p))
    if n < 1e-9:
        return np.array([radius, 0.0, 0.0])
    return radius * p / n


def _closest_on_cylinder(p: np.ndarray, radius: float, height: float) -> np.ndarray:
    half_h = 0.5 * height
    radial = math.hypot(p[0], p[1])
    d = np.array([1.0, 0.0]) if radial < 1e-9 else np.array([p[0], p[1]]) / radial
    in_r, in_h = radial <= radius, abs(p[2]) <= half_h
    cap = half_h if p[2] >= 0.0 else -half_h
    if not in_r and in_h:                       # side wall
        return np.array([d[0] * radius, d[1] * radius, p[2]])
    if in_r and not in_h:                       # cap
        return np.array([p[0], p[1], cap])
    if not in_r and not in_h:                   # rim
        return np.array([d[0] * radius, d[1] * radius, cap])
    # inside: nearer of side wall and cap
    if radius - radial <= half_h - abs(p[2]):
        return np.array([d[0] * radius, d[1] * radius, p[2]])
    return np.array([p[0], p[1], cap])


def _closest_local(obs: Obstacle, p: np.ndarray) -> np.ndarray:
    if obs.kind == 'box':
        return _closest_on_box(p, obs.size)
    if obs.kind == 'cylinder':
        return _closest_on_cylinder(p, obs.radius, obs.height)
    if obs.kind == 'sphere':
        return _closest_on_sphere(p, obs.radius)
    return p


# ------------------------------------------------------------------- node
class ArmDetectionPoints(Node):

    def __init__(self) -> None:
        super().__init__('arm_detection_points')

        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('detection_frames', [
            'detect0_1', 'detect0_2', 'detect1',
            'detect2_1', 'detect2_2', 'detect2_3',
            'detect3_1', 'detect3_2',
            'detect4_1', 'detect4_2',
            'detect5', 'detect6',
        ])
        # Flat encoding, because rclpy parameters cannot hold a list of dicts.
        # One string per obstacle:
        #   name:model:box:sx,sy,sz:cx,cy,cz:cr,cp,cy
        #   name:model:cylinder:radius,height:cx,cy,cz:cr,cp,cy
        #   name:model:sphere:radius:cx,cy,cz:cr,cp,cy
        # The two trailing fields (collision offset, collision rpy) are
        # optional and default to zero.
        self.declare_parameter('obstacles', [''])
        self.declare_parameter('use_gz_base_pose', False)
        self.declare_parameter('gz_base_model', 'omni_bot')
        self.declare_parameter('pose_timeout', 1.0)

        self.base_frame = self.get_parameter('robot_base_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.frames = list(self.get_parameter('detection_frames').value)
        self.use_gz_base = bool(self.get_parameter('use_gz_base_pose').value)
        self.pose_timeout = float(self.get_parameter('pose_timeout').value)

        self.obstacles = self._parse_obstacles(
            [s for s in self.get_parameter('obstacles').value if s.strip()])
        if not self.obstacles:
            self.get_logger().warn(
                'no obstacles configured -- every detection point will publish NaN')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # gz pose bridge is best-effort volatile, matching omni_bot_dynamic
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self._pose_stamp: dict[str, float] = {}
        self._subs = []
        for model in sorted({o.model for o in self.obstacles} |
                            ({self.gz_base_model} if self.use_gz_base else set())):
            self._subs.append(self.create_subscription(
                PoseStamped, f'/model/{model}/pose',
                self._make_pose_cb(model), qos))

        self._base_T: np.ndarray | None = None
        self.pub = self.create_publisher(
            PolygonStamped, 'collision_detection_points_state', 1)

        rate = float(self.get_parameter('publish_rate').value)
        if rate <= 0.0:
            self.get_logger().warn('publish_rate <= 0, using 30 Hz')
            rate = 30.0
        self.timer = self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().info(
            f'arm_detection_points: {len(self.frames)} points, '
            f'{len(self.obstacles)} obstacles, {rate:.0f} Hz, '
            f'base pose from {"gz ground truth" if self.use_gz_base else "TF"}')

    @property
    def gz_base_model(self) -> str:
        return str(self.get_parameter('gz_base_model').value)

    # ------------------------------------------------------------ config
    def _parse_obstacles(self, specs: list[str]) -> list[Obstacle]:
        out: list[Obstacle] = []
        for spec in specs:
            f = spec.split(':')
            if len(f) < 4:
                raise ValueError(f'obstacle spec needs >=4 fields: {spec!r}')
            name, model, kind = f[0], f[1], f[2]
            nums = [float(v) for v in f[3].replace(' ', '').split(',') if v]
            obs = Obstacle(name=name, model=model, kind=kind)
            if kind == 'box':
                if len(nums) != 3:
                    raise ValueError(f'box {name} needs sx,sy,sz: {spec!r}')
                obs.size = np.array(nums)
            elif kind == 'cylinder':
                if len(nums) != 2:
                    raise ValueError(f'cylinder {name} needs radius,height: {spec!r}')
                obs.radius, obs.height = nums
            elif kind == 'sphere':
                if len(nums) != 1:
                    raise ValueError(f'sphere {name} needs radius: {spec!r}')
                obs.radius = nums[0]
            else:
                raise ValueError(f'unsupported obstacle type {kind!r} in {spec!r}')
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if len(f) > 4 and f[4].strip():
                xyz = np.array([float(v) for v in f[4].split(',')])
            if len(f) > 5 and f[5].strip():
                rpy = np.array([float(v) for v in f[5].split(',')])
            obs.T_link_collision = _iso(_rpy_to_rot(*rpy), xyz)
            out.append(obs)
        return out

    # ------------------------------------------------------------- poses
    def _make_pose_cb(self, model: str):
        def cb(msg: PoseStamped) -> None:
            p, q = msg.pose.position, msg.pose.orientation
            T = _iso(_quat_to_rot(q.x, q.y, q.z, q.w),
                     np.array([p.x, p.y, p.z]))
            self._pose_stamp[model] = self.get_clock().now().nanoseconds * 1e-9
            if self.use_gz_base and model == self.gz_base_model:
                self._base_T = T
            for o in self.obstacles:
                if o.model == model:
                    o.T_world_link = T
        return cb

    def _world_base(self) -> np.ndarray | None:
        """T_world_base, from gz ground truth or from TF."""
        if self.use_gz_base:
            return self._base_T
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, rclpy.time.Time())
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().warn(
                f'{self.world_frame} -> {self.base_frame} unavailable: {exc}',
                throttle_duration_sec=2.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return _iso(_quat_to_rot(q.x, q.y, q.z, q.w),
                    np.array([t.x, t.y, t.z]))

    def _fresh(self, model: str, now: float) -> bool:
        ts = self._pose_stamp.get(model)
        return ts is not None and (now - ts) <= self.pose_timeout

    # -------------------------------------------------------------- loop
    def _on_timer(self) -> None:
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Not a TF frame: point i is expressed in detection_frames[i].
        msg.header.frame_id = 'each_vector_in_its_own_detection_frame'

        T_world_base = self._world_base()
        now = self.get_clock().now().nanoseconds * 1e-9
        live = [o for o in self.obstacles
                if o.T_world_link is not None and self._fresh(o.model, now)]

        for frame in self.frames:
            v = NAN3
            if T_world_base is not None and live:
                got = self._closest_vector(T_world_base, frame, live)
                if got is not None:
                    v = got
            msg.polygon.points.append(
                Point32(x=float(v[0]), y=float(v[1]), z=float(v[2])))
        self.pub.publish(msg)

    def _closest_vector(self, T_world_base: np.ndarray, frame: str,
                        live: list[Obstacle]) -> tuple[float, float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, frame, rclpy.time.Time())
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().warn(
                f'TF {self.base_frame} -> {frame} failed: {exc}',
                throttle_duration_sec=2.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        T_base_det = _iso(_quat_to_rot(q.x, q.y, q.z, q.w),
                          np.array([t.x, t.y, t.z]))
        T_world_det = T_world_base @ T_base_det
        p_world = T_world_det[:3, 3]

        best_d = math.inf
        best_v = None
        for o in live:
            T_wc = o.T_world_link @ o.T_link_collision
            p_local = (_inv(T_wc) @ np.append(p_world, 1.0))[:3]
            closest_world = (T_wc @ np.append(_closest_local(o, p_local), 1.0))[:3]
            vec = closest_world - p_world
            d = float(np.linalg.norm(vec))
            if d < best_d:
                best_d, best_v = d, vec
        if best_v is None:
            return None
        # Rotate into the detection frame; do NOT translate -- it is a free
        # vector from the point to the surface, not a position.
        local = T_world_det[:3, :3].T @ best_v
        return float(local[0]), float(local[1]), float(local[2])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmDetectionPoints()
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
