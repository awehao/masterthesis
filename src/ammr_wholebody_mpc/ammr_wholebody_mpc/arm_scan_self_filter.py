"""Remove the robot's own arm from its LiDAR scan.

The 2D scan plane sits at base_link z = 0.2152. The arm normally lives well
above it, but over the legal joint box 7.8% of configurations reach down to
z = 0.070 -- into the plane. There the LiDAR sees the arm, and every consumer
downstream treats it as an obstacle: the raw-scan shield brakes for the robot's
own elbow, and the tracker may cluster it as a mover.

scan_relay cannot do this. It masks FIXED bearings (the four chassis struts at
45/135/225/315 deg); the arm moves, so no static sector covers it.

Placement, by launch remap, so nothing downstream changes:

    /scan_raw -> scan_relay -> /scan_base -> THIS -> /scan

With use_arm:=false the node is not started and scan_relay publishes /scan
directly, exactly as before.

Why the arm's actual mesh cross-section and not a capsule
--------------------------------------------------------
Capsules were fitted first and rejected. A capsule around link_base needs
radius 0.123 m to enclose a link whose true horizontal extent is 0.094 m --
30% too big. Over-covering is the dangerous direction here: a mask wider than
the arm deletes real returns near it, and an obstacle 3 cm from the elbow is
exactly the one that matters. So the filter tests scan points against the
link meshes themselves, transformed by the live joint angles.

The rule is deliberately narrow: a point is dropped only if it lies within
`tolerance` of the arm's actual cross-section in the scan plane. Anything
further away survives, including obstacles the arm is nearly touching.

Diagnostics on /scan_self_filter/diag make over-masking visible rather than
silent: the count dropped, the nearest surviving return, and the nearest
dropped one. If the last two ever cross, the tolerance is too large.
"""
from __future__ import annotations

import math
import os
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from scipy.spatial import cKDTree
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float32MultiArray, String

from .wholebody_kinematics import WholeBodyKinematics, iso, rpy_to_rot

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
# Links that can enter the scan plane. The chassis is scan_relay's job.
ARM_LINKS = ['link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
             'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2']

BEST_EFFORT = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
LATCHED = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def load_stl(path: str) -> np.ndarray:
    d = open(path, 'rb').read()
    if d[:5] == b'solid' and b'facet' in d[:512]:
        return np.array([[float(x) for x in l.split()[1:4]]
                         for l in d.decode('ascii', 'ignore').splitlines()
                         if l.strip().startswith('vertex')])
    n = struct.unpack('<I', d[80:84])[0]
    return np.array([struct.unpack('<9f', d[84 + i * 50 + 12:84 + i * 50 + 48])
                     for i in range(n)]).reshape(-1, 3)


def link_collision_clouds(xml: str, links: list[str]) -> dict[str, np.ndarray]:
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        if name not in links:
            continue
        pts = []
        for col in link.findall('collision'):
            geo = col.find('geometry')
            if geo is None or len(geo) == 0 or geo[0].tag != 'mesh':
                continue
            p = geo[0].get('filename', '').replace('file://', '')
            if not os.path.exists(p):
                continue
            V = load_stl(p)
            sc = geo[0].get('scale')
            if sc:
                V = V * np.array([float(x) for x in sc.split()])
            o = col.find('origin')
            g = lambda k, d: np.array([float(v) for v in
                                       ((o.get(k) or d) if o is not None else d).split()])
            T = iso(rpy_to_rot(*g('rpy', '0 0 0')), g('xyz', '0 0 0'))
            pts.append((T[:3, :3] @ V.T).T + T[:3, 3])
        if pts:
            out[name] = np.vstack(pts)
    return out


class ArmScanSelfFilter(Node):

    def __init__(self) -> None:
        super().__init__('arm_scan_self_filter')
        p = self.declare_parameter
        p('scan_in', '/scan_base')
        p('scan_out', '/scan')
        p('lidar_frame', 'lidar_link')
        # Half-thickness of the slab of arm geometry considered to be "in" the
        # scan plane. The beam has finite width and the mesh is sampled, so a
        # zero-thickness slice would miss the arm between vertices.
        p('z_band', 0.030)
        # How close a return must be to the arm to be called the arm. Covers
        # LiDAR range noise and mesh sampling, nothing more. Raising this is
        # how real obstacles start disappearing.
        p('tolerance', 0.030)
        p('enable', True)
        p('joint_timeout', 0.5)

        g = lambda k: self.get_parameter(k).value
        self.enable = bool(g('enable'))
        self.lidar_frame = str(g('lidar_frame'))
        self.z_band = float(g('z_band'))
        self.tol = float(g('tolerance'))
        self.joint_timeout = float(g('joint_timeout'))

        self.K = None
        self.clouds: dict[str, np.ndarray] = {}
        self.z_scan = None          # scan plane height in the URDF root frame
        self.q = np.zeros(len(ARM_JOINTS))
        self.q_t = 0.0
        self._cycle = 0

        self.create_subscription(String, '/robot_description',
                                 self._on_urdf, LATCHED)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.pub = self.create_publisher(LaserScan, str(g('scan_out')), BEST_EFFORT)
        self.diag = self.create_publisher(Float32MultiArray,
                                          '/scan_self_filter/diag', 10)
        self.create_subscription(LaserScan, str(g('scan_in')), self._on_scan,
                                 BEST_EFFORT)
        self.get_logger().info(
            f'arm self-filter: {g("scan_in")} -> {g("scan_out")}  '
            f'tol {self.tol:.3f} m  z_band +-{self.z_band:.3f} m')

    # ------------------------------------------------------------ model
    def _on_urdf(self, msg: String) -> None:
        if self.K is not None:
            return
        try:
            self.K = WholeBodyKinematics.from_urdf_string(msg.data, ARM_JOINTS)
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().error(f'cannot build kinematics: {exc}')
            return
        self.clouds = link_collision_clouds(msg.data, ARM_LINKS)
        try:
            self.z_scan = float(self.K.fk(np.zeros(len(ARM_JOINTS)),
                                          self.lidar_frame)[2, 3])
        except Exception:
            self.z_scan = None
        n = sum(len(v) for v in self.clouds.values())
        self.get_logger().info(
            f'loaded {len(self.clouds)} arm links, {n} collision vertices; '
            f'scan plane z = {self.z_scan}')

    def _on_js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in ARM_JOINTS):
            return
        self.q = np.array([msg.position[idx[j]] for j in ARM_JOINTS])
        self.q_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- scan
    def _arm_xy_in_plane(self) -> np.ndarray:
        """Arm collision vertices near the scan plane, in lidar-frame XY."""
        if self.K is None or self.z_scan is None or not self.clouds:
            return np.empty((0, 2))
        T_lidar = self.K.fk(self.q, self.lidar_frame)
        T_inv = np.eye(4)
        T_inv[:3, :3] = T_lidar[:3, :3].T
        T_inv[:3, 3] = -T_lidar[:3, :3].T @ T_lidar[:3, 3]
        out = []
        for name, P in self.clouds.items():
            T = self.K.fk(self.q, name)
            W = (T[:3, :3] @ P.T).T + T[:3, 3]
            near = np.abs(W[:, 2] - self.z_scan) <= self.z_band
            if not near.any():
                continue
            L = (T_inv[:3, :3] @ W[near].T).T + T_inv[:3, 3]
            out.append(L[:, :2])
        return np.vstack(out) if out else np.empty((0, 2))

    def _on_scan(self, msg: LaserScan) -> None:
        self._cycle += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        stale = (now - self.q_t) > self.joint_timeout

        n_drop = 0
        nearest_kept = float('inf')
        nearest_dropped = float('inf')
        arm = np.empty((0, 2))

        if self.enable and not stale and self.K is not None:
            arm = self._arm_xy_in_plane()

        if len(arm) == 0:
            # Nothing of the arm is in the plane -- pass the scan through
            # untouched. Passing through is the correct default: an unfiltered
            # scan is over-cautious, a wrongly filtered one is blind.
            r = np.asarray(msg.ranges, dtype=float)
            fin = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
            nearest_kept = float(r[fin].min()) if fin.any() else float('inf')
            self.pub.publish(msg)
        else:
            r = np.asarray(msg.ranges, dtype=float)
            ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
            fin = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
            xy = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
            d = np.full(len(r), np.inf)
            if fin.any():
                d[fin] = cKDTree(arm).query(xy[fin], k=1)[0]
            drop = fin & (d <= self.tol)
            n_drop = int(drop.sum())
            if n_drop:
                nearest_dropped = float(r[drop].min())
            out = list(msg.ranges)
            for i in np.where(drop)[0]:
                out[i] = float('inf')
            keep = fin & ~drop
            nearest_kept = float(r[keep].min()) if keep.any() else float('inf')
            m2 = LaserScan()
            m2.header = msg.header
            m2.angle_min, m2.angle_max = msg.angle_min, msg.angle_max
            m2.angle_increment = msg.angle_increment
            m2.time_increment = msg.time_increment
            m2.scan_time = msg.scan_time
            m2.range_min, m2.range_max = msg.range_min, msg.range_max
            m2.ranges = out
            m2.intensities = msg.intensities
            self.pub.publish(m2)

        d = Float32MultiArray()
        #  0 cycle  1 n_arm_pts_in_plane  2 n_dropped  3 nearest_kept
        #  4 nearest_dropped  5 joints_stale  6 total_returns
        d.data = [float(self._cycle), float(len(arm)), float(n_drop),
                  nearest_kept if math.isfinite(nearest_kept) else -1.0,
                  nearest_dropped if math.isfinite(nearest_dropped) else -1.0,
                  1.0 if stale else 0.0, float(len(msg.ranges))]
        self.diag.publish(d)


def main() -> None:
    rclpy.init()
    node = ArmScanSelfFilter()
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
