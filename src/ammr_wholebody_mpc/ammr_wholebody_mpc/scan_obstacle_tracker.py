"""Detect & track UNKNOWN dynamic obstacles from /scan (no ground truth).

Pipeline (drop-in replacement for obstacle_aggregator, same wire format):

  /scan (LaserScan, lidar_link)
    -> transform hits to the global frame (map) via TF
    -> static background subtraction against the known /map
       (drop points that fall on / near a known-occupied cell = walls)
    -> cluster the remaining (dynamic) points by angular adjacency
       (the laser-scan equivalent of DBSCAN; sklearn-free)
    -> reject clusters that are too small / too large (walls, noise)
    -> associate cluster centroids to existing tracks (Hungarian, gated)
    -> one constant-velocity Kalman filter per track -> (x, y, vx, vy)
    -> publish /gmpc/obstacles : Float32MultiArray [x,y,r,vx,vy, ...]

This is the Sprint-B "real perception" path: it never reads obstacle ground
truth, so it covers obstacles that are NOT in any predefined list. The GMPC +
horizon-CBF controller consumes /gmpc/obstacles unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from .kalman_tracker import KalmanTracker2D

try:
    from scipy.ndimage import binary_dilation
    _HAVE_NDIMAGE = True
except Exception:                       # pragma: no cover
    _HAVE_NDIMAGE = False


# ---------------------------------------------------------------------------
# Pure-python core (unit-testable without ROS)
# ---------------------------------------------------------------------------
def cluster_adjacent(pts: np.ndarray, gap: float,
                     min_pts: int, max_radius: float
                     ) -> List[Tuple[float, float, float, int]]:
    """Cluster an ORDERED set of 2D points (by laser bearing) into compact
    blobs: start a new cluster whenever the Euclidean step between consecutive
    points exceeds `gap`. Returns [(cx, cy, radius, n_pts), ...] for clusters
    that pass the size gates.

    `pts` is (N,2), already ordered by scan angle and already background-
    subtracted (only dynamic candidate points).
    """
    out: List[Tuple[float, float, float, int]] = []
    if len(pts) == 0:
        return out

    start = 0
    for i in range(1, len(pts) + 1):
        split = (i == len(pts)) or (
            math.hypot(pts[i, 0] - pts[i - 1, 0],
                       pts[i, 1] - pts[i - 1, 1]) > gap)
        if split:
            seg = pts[start:i]
            start = i
            if len(seg) < min_pts:
                continue
            cx, cy = float(seg[:, 0].mean()), float(seg[:, 1].mean())
            radius = float(np.max(np.hypot(seg[:, 0] - cx, seg[:, 1] - cy)))
            if radius > max_radius:
                continue                # too big -> a wall segment, not an obstacle
            out.append((cx, cy, radius, len(seg)))
    return out


def associate(cluster_xy: np.ndarray, track_xy: np.ndarray, gate: float
              ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment between clusters and tracks under a distance gate.

    Returns (matches, unmatched_clusters, unmatched_tracks) where matches is a
    list of (cluster_idx, track_idx).
    """
    nc, nt = len(cluster_xy), len(track_xy)
    if nc == 0 or nt == 0:
        return [], list(range(nc)), list(range(nt))

    cost = np.hypot(
        cluster_xy[:, None, 0] - track_xy[None, :, 0],
        cluster_xy[:, None, 1] - track_xy[None, :, 1])
    rows, cols = linear_sum_assignment(cost)

    matches, um_c, um_t = [], set(range(nc)), set(range(nt))
    for r, c in zip(rows, cols):
        if cost[r, c] <= gate:
            matches.append((int(r), int(c)))
            um_c.discard(int(r))
            um_t.discard(int(c))
    return matches, sorted(um_c), sorted(um_t)


@dataclass
class Track:
    tid: int
    kf: KalmanTracker2D
    radius: float
    last_t_ns: int
    age: int = 1                 # number of successful updates
    misses: int = 0


# ---------------------------------------------------------------------------
class ScanObstacleTracker(Node):

    def __init__(self):
        super().__init__('scan_obstacle_tracker')

        p = self.declare_parameter
        p('scan_topic', '/scan')
        p('map_topic', '/map')
        p('output_topic', '/gmpc/obstacles')
        # Track in the SMOOTH odom frame, not map: AMCL (map) jumps make static
        # walls look like they move, defeating the velocity gate. Our gz ground-
        # truth odom is drift-free, so static geometry is truly stationary in
        # odom (v~=0) and only real dynamic obstacles move (v~=0.3). Obstacles
        # are then transformed to publish_frame (map) so the GMPC consumes them
        # in the same frame as the robot pose -- the AMCL offset cancels in the
        # relative distance the CBF actually uses.
        p('global_frame', 'map')          # frame to cluster + track in (AMCL)
        p('publish_frame', 'map')         # frame to publish obstacles in (GMPC's)
        p('use_map_subtraction', True)    # subtract static walls against /map
        p('cluster_gap', 0.30)            # m, split clusters on bigger jump
        p('min_cluster_pts', 2)
        p('max_cluster_radius', 0.60)     # m, reject bigger blobs (walls)
        p('default_radius', 0.25)         # m, inflate cluster radius up to this
        p('static_inflation', 0.50)       # m, drop hits within this of a wall
                                          # (>= worst-case AMCL error so walls
                                          # shifted into "free" cells are caught)
        p('assoc_gate', 0.80)             # m, max cluster<->track distance
        p('track_timeout', 0.60)          # s, drop track unseen this long
        p('min_track_age', 3)             # publish only after this many updates
        p('min_track_speed', 0.10)        # m/s, publish only MOVING tracks ->
                                          # rejects static map-subtraction leaks
                                          # (walls handled by costmap/planner).
        # KF tuning (meas noise inflated vs ground-truth: cluster centroids jitter)
        p('kf_sigma_pos', 0.01)
        p('kf_sigma_vel', 0.40)
        p('kf_sigma_meas', 0.05)
        p('kf_init_vel_var', 1.0)
        p('publish_markers', True)
        # --- static-CBF (Solution 1) ---------------------------------------
        # map_subtraction leaves the CBF blind to walls, so dodging a dynamic
        # obstacle can push the robot INTO static geometry. Fix: from the rays
        # that hit known walls, keep the NEAREST point per angular sector and
        # feed them to the GMPC as zero-velocity obstacles on a separate topic.
        # Sector selection (not global-nearest) keeps the constraints stable
        # (no chattering); v=0 makes them immune to localization jitter.
        p('static_cbf_enable', True)
        p('static_cbf_topic', '/gmpc/static_obstacles')
        p('static_cbf_range', 1.2)        # m, only walls within this matter
        p('static_cbf_sectors', 8)        # nearest wall point per 45-deg sector
        p('static_cbf_max', 4)            # cap total static points (QP size)
        # d_safe MUST include the robot's own half-extent: the CBF treats the
        # robot as a POINT (r_eff = obstacle_radius + cbf_safe_margin), so the
        # robot's 0.45 m chassis (half-width 0.225 m) is otherwise ignored ->
        # the body grazes the wall. Encode it here:
        #   robot center keep-out = static_cbf_radius + cbf_safe_margin(0.30)
        #   = 0.20 + 0.30 = 0.50 m  -> robot SIDE (0.225) clears wall by ~0.275 m
        # (0.20 = robot half-width ~0.225 folded with a small per-point gap; the
        #  0.30 margin then becomes real clearance instead of standing in for the
        #  body.)
        p('static_cbf_radius', 0.20)
        p('static_cbf_snap', 0.15)        # m, snap wall points to this grid ->
                                          # stable anchor, kills per-frame jitter

        g = lambda n: self.get_parameter(n).value
        self.global_frame   = str(g('global_frame'))
        self.publish_frame  = str(g('publish_frame'))
        self.use_map        = bool(g('use_map_subtraction'))
        self.cluster_gap    = float(g('cluster_gap'))
        self.min_pts        = int(g('min_cluster_pts'))
        self.max_radius     = float(g('max_cluster_radius'))
        self.default_radius = float(g('default_radius'))
        self.static_infl    = float(g('static_inflation'))
        self.assoc_gate     = float(g('assoc_gate'))
        self.track_timeout  = float(g('track_timeout'))
        self.min_age        = int(g('min_track_age'))
        self.min_speed      = float(g('min_track_speed'))
        self.static_cbf_en  = bool(g('static_cbf_enable'))
        self.static_range   = float(g('static_cbf_range'))
        self.static_sectors = int(g('static_cbf_sectors'))
        self.static_max     = int(g('static_cbf_max'))
        self.static_radius  = float(g('static_cbf_radius'))
        self.static_snap    = float(g('static_cbf_snap'))
        self._kf_kwargs = dict(
            sigma_pos=float(g('kf_sigma_pos')), sigma_vel=float(g('kf_sigma_vel')),
            sigma_meas=float(g('kf_sigma_meas')), init_vel_var=float(g('kf_init_vel_var')))

        self._tracks: List[Track] = []
        self._next_id = 0
        self._occ: np.ndarray | None = None   # inflated occupied mask (row,col)
        self._map_info = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, str(g('map_topic')),
                                 self._map_cb, map_qos)
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(LaserScan, str(g('scan_topic')),
                                 self._scan_cb, scan_qos)

        self.pub = self.create_publisher(Float32MultiArray, str(g('output_topic')), 10)
        # /scan_filtered: the input scan with DYNAMIC-obstacle rays removed, for
        # the global costmap. Keeps the planner seeing only static geometry
        # (walls + unknown static) so a passing dynamic obstacle never makes the
        # robot's start cell "occupied" -> no SmacPlanner start-occupied lockout.
        self.scan_mask_margin = 0.20
        self.filtered_pub = self.create_publisher(LaserScan, '/scan_filtered', scan_qos)
        self.static_pub = self.create_publisher(
            Float32MultiArray, str(g('static_cbf_topic')), 10)
        self.publish_markers = bool(g('publish_markers'))
        self.mpub = self.create_publisher(MarkerArray, '/gmpc/scan_obstacles_viz', 10) \
            if self.publish_markers else None

        self.get_logger().info(
            f'scan_obstacle_tracker up: /scan -> track in {self.global_frame} '
            f'(map_subtraction={self.use_map}) -> cluster(gap={self.cluster_gap}) '
            f'-> KF -> velocity gate(>={self.min_speed} m/s) -> '
            f'{g("output_topic")} in {self.publish_frame}')

    # ------------------------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid):
        info = msg.info
        w, h = info.width, info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        occ = grid >= 50                      # occupied cells
        if _HAVE_NDIMAGE and self.static_infl > 0.0:
            rad = max(1, int(round(self.static_infl / info.resolution)))
            # square structuring element of half-width `rad`
            occ = binary_dilation(occ, iterations=rad)
        self._occ = occ
        self._map_info = info
        self.get_logger().info(
            f'map received: {w}x{h} @ {info.resolution:.3f} m, '
            f'inflated occupied cells = {int(occ.sum())}')

    def _is_static(self, mx: float, my: float) -> bool:
        """True if the map-frame point falls on an (inflated) occupied cell."""
        if self._occ is None:
            return False                      # no map yet -> keep everything
        info = self._map_info
        col = int((mx - info.origin.position.x) / info.resolution)
        row = int((my - info.origin.position.y) / info.resolution)
        h, w = self._occ.shape
        if 0 <= row < h and 0 <= col < w:
            return bool(self._occ[row, col])
        return True                           # outside the known map -> a wall /
                                              # mis-registered hit, never a real
                                              # dynamic obstacle -> reject

    # ------------------------------------------------------------------
    def _scan_cb(self, scan: LaserScan):
        # TF lidar -> global at the scan stamp (fall back to latest).
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.global_frame, scan.header.frame_id, scan.header.stamp,
                timeout=Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.global_frame, scan.header.frame_id, Time())
            except Exception:
                return
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        cyaw, syaw = math.cos(yaw), math.sin(yaw)

        # Build points in the tracking (odom) frame, in scan order. Optional
        # map subtraction only makes sense if we track in the map frame.
        dyn = []
        static_hits = []          # (range, angle, mx, my) on known walls -> static-CBF
        a = scan.angle_min
        rmin, rmax = scan.range_min, scan.range_max
        for r in scan.ranges:
            if math.isfinite(r) and rmin < r < rmax:
                lx, ly = r * math.cos(a), r * math.sin(a)
                mx = tx + cyaw * lx - syaw * ly
                my = ty + syaw * lx + cyaw * ly
                if self.use_map and self._is_static(mx, my):
                    static_hits.append((r, a, mx, my))   # wall: feed CBF (v=0)
                else:
                    dyn.append((mx, my))
            a += scan.angle_increment
        dyn = np.array(dyn, dtype=float) if dyn else np.empty((0, 2))

        clusters = cluster_adjacent(dyn, self.cluster_gap, self.min_pts, self.max_radius)

        t_ns = (int(scan.header.stamp.sec) * 1_000_000_000
                + int(scan.header.stamp.nanosec)) or self.get_clock().now().nanoseconds
        self._update_tracks(clusters, t_ns)
        self._publish(t_ns, scan.header.stamp)
        self._publish_filtered_scan(scan, tx, ty, cyaw, syaw)
        self._publish_static_obstacles(static_hits, tx, ty)

    # ------------------------------------------------------------------
    def _publish_static_obstacles(self, static_hits, rx, ry):
        """Solution-1 static-CBF: from the rays that hit KNOWN walls, keep the
        nearest point per angular sector and publish them as zero-velocity
        obstacles. The GMPC merges these into its CBF set so it won't dodge a
        dynamic obstacle into a wall. Two anti-chatter measures: (1) sector by
        MAP-frame bearing (yaw-invariant -> a wall doesn't change sector when the
        omni base rotates); (2) snap the point to a coarse grid (stable anchor ->
        no per-frame scan-noise jitter). v=0 -> immune to localization jitter.
        Always publishes (empty array clears the controller's static set)."""
        arr = Float32MultiArray()
        if self.static_cbf_en and static_hits:
            best = {}                                  # sector -> (range, mx, my)
            two_pi = 2.0 * math.pi
            q = self.static_snap
            for (r, a, mx, my) in static_hits:
                d = math.hypot(mx - rx, my - ry)
                if d > self.static_range:
                    continue
                # snap to grid (stable anchor)
                sx = round(mx / q) * q if q > 0 else mx
                sy = round(my / q) * q if q > 0 else my
                # sector by MAP-frame bearing from robot (yaw-invariant)
                bearing = math.atan2(sy - ry, sx - rx)
                sec = int((bearing + math.pi) / two_pi * self.static_sectors) % self.static_sectors
                if sec not in best or d < best[sec][0]:
                    best[sec] = (d, sx, sy)
            data = []
            for (d, sx, sy) in sorted(best.values())[:self.static_max]:  # nearest first
                data.extend((sx, sy, self.static_radius, 0.0, 0.0))
            arr.data = data
        self.static_pub.publish(arr)

    # ------------------------------------------------------------------
    def _publish_filtered_scan(self, scan, tx, ty, cyaw, syaw):
        """Republish the scan with rays hitting confirmed MOVING obstacles set
        to +inf, so the global costmap (planner) sees static geometry only."""
        moving = [t for t in self._tracks
                  if t.age >= self.min_age
                  and math.hypot(*t.kf.velocity) >= self.min_speed]
        out = LaserScan()
        out.header = scan.header
        out.angle_min = scan.angle_min
        out.angle_max = scan.angle_max
        out.angle_increment = scan.angle_increment
        out.time_increment = scan.time_increment
        out.scan_time = scan.scan_time
        out.range_min = scan.range_min
        out.range_max = scan.range_max
        out.intensities = scan.intensities
        if not moving:
            out.ranges = scan.ranges
            self.filtered_pub.publish(out)
            return
        ranges = list(scan.ranges)
        a = scan.angle_min
        for i, r in enumerate(scan.ranges):
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                lx, ly = r * math.cos(a), r * math.sin(a)
                mx = tx + cyaw * lx - syaw * ly
                my = ty + syaw * lx + cyaw * ly
                for tr in moving:
                    px, py = tr.kf.position
                    if math.hypot(mx - px, my - py) <= tr.radius + self.scan_mask_margin:
                        ranges[i] = float('inf')
                        break
            a += scan.angle_increment
        out.ranges = ranges
        self.filtered_pub.publish(out)

    # ------------------------------------------------------------------
    def _update_tracks(self, clusters, t_ns: int):
        cl_xy = np.array([[c[0], c[1]] for c in clusters], dtype=float) \
            if clusters else np.empty((0, 2))
        tr_xy = np.array([[t.kf.position[0], t.kf.position[1]] for t in self._tracks],
                         dtype=float) if self._tracks else np.empty((0, 2))

        matches, um_c, um_t = associate(cl_xy, tr_xy, self.assoc_gate)

        for ci, ti in matches:
            tr = self._tracks[ti]
            tr.kf.step(t_ns=t_ns, y_xy=(clusters[ci][0], clusters[ci][1]))
            tr.radius = max(self.default_radius, clusters[ci][2])
            tr.last_t_ns = t_ns
            tr.age += 1
            tr.misses = 0

        for ci in um_c:                       # new track
            kf = KalmanTracker2D(init_xy=(clusters[ci][0], clusters[ci][1]),
                                 **self._kf_kwargs)
            self._tracks.append(Track(
                tid=self._next_id, kf=kf,
                radius=max(self.default_radius, clusters[ci][2]), last_t_ns=t_ns))
            self._next_id += 1

        for ti in um_t:
            self._tracks[ti].misses += 1

        # Age out stale tracks.
        keep = []
        for tr in self._tracks:
            if (t_ns - tr.last_t_ns) * 1e-9 <= self.track_timeout:
                keep.append(tr)
        self._tracks = keep

    # ------------------------------------------------------------------
    def _publish(self, t_ns: int, stamp):
        # Publish only sufficiently-aged AND moving tracks. In the smooth odom
        # frame, static geometry (walls) is genuinely stationary (v~=0) and only
        # real dynamic obstacles move (v~=0.3), so the velocity gate cleanly
        # rejects wall clusters without any map/AMCL dependence.
        confirmed = [t for t in self._tracks
                     if t.age >= self.min_age
                     and math.hypot(*t.kf.velocity) >= self.min_speed]

        # Transform tracking-frame (odom) tracks into publish_frame (map) so the
        # GMPC sees them in the same frame as the robot pose; the AMCL map<-odom
        # offset cancels in the relative distance the CBF uses.
        tfm = None
        if self.publish_frame != self.global_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.publish_frame, self.global_frame, Time())
                q = t.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                tfm = (t.transform.translation.x, t.transform.translation.y,
                       math.cos(yaw), math.sin(yaw))
            except Exception:
                tfm = None
        out_frame = self.publish_frame if tfm is not None else self.global_frame

        def to_pub(px, py, vx, vy):
            if tfm is None:
                return px, py, vx, vy
            tx, ty, c, s = tfm
            return (tx + c * px - s * py, ty + s * px + c * py,
                    c * vx - s * vy, s * vx + c * vy)

        flat: List[float] = []
        pub_xy: List[tuple] = []
        for tr in confirmed:
            px, py = tr.kf.position
            vx, vy = tr.kf.velocity
            PX, PY, VX, VY = to_pub(px, py, vx, vy)
            flat.extend([PX, PY, tr.radius, VX, VY])
            pub_xy.append((PX, PY))
        msg = Float32MultiArray(); msg.data = flat
        self.pub.publish(msg)

        if self.mpub is not None:
            ma = MarkerArray()
            for i, tr in enumerate(confirmed):
                PX, PY = pub_xy[i]
                m = Marker()
                m.header.frame_id = out_frame
                m.header.stamp = stamp
                m.ns = 'scan_obs'; m.id = i; m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = PX; m.pose.position.y = PY; m.pose.position.z = 0.5
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 2 * tr.radius; m.scale.z = 1.0
                m.color.r = 1.0; m.color.g = 0.3; m.color.b = 0.0; m.color.a = 0.5
                ma.markers.append(m)
            self.mpub.publish(ma)


def main():
    rclpy.init()
    node = ScanObstacleTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
