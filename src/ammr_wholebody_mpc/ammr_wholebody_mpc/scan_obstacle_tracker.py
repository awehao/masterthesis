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
from collections import deque
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
    from scipy.ndimage import binary_dilation, distance_transform_edt
    _HAVE_NDIMAGE = True
except Exception:                       # pragma: no cover
    _HAVE_NDIMAGE = False


# ---------------------------------------------------------------------------
# Pure-python core (unit-testable without ROS)
# ---------------------------------------------------------------------------
def cluster_adjacent(pts: np.ndarray, gap: float,
                     min_pts: int, max_radius: float,
                     sensor_xy=(0.0, 0.0), r_prior: float = 0.25
                     ) -> List[Tuple[float, float, float, int]]:
    """Cluster an ORDERED set of 2D points (by laser bearing) into compact
    blobs: start a new cluster whenever the Euclidean step between consecutive
    points exceeds `gap`. Returns [(cx, cy, radius, n_pts), ...] for clusters
    that pass the size gates.

    The reported (cx, cy) is the fitted OBJECT centre, not the arc mean -- see
    the surface-point representation for why no shape is fitted at all
    of the dynamic set.

    `pts` is (N,2), already ordered by scan angle and already background-
    subtracted (only dynamic candidate points), in the same frame as
    `sensor_xy`.
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
            # extent of the ARC decides whether this is an object or a wall;
            # the fitted circle decides WHERE it is
            am = seg.mean(axis=0)
            extent = float(np.max(np.hypot(seg[:, 0] - am[0], seg[:, 1] - am[1])))
            if extent > max_radius:
                continue                # too big -> a wall segment, not an obstacle
            # The centroid is kept ONLY for data association -- it is biased
            # towards the sensor and slides along the surface as the viewing
            # angle changes, which is fine for matching a track frame to frame
            # and useless as a geometric reference. The geometry the CBF sees is
            # the RETURNED POINTS themselves; nothing is fitted to them.
            out.append((float(am[0]), float(am[1]), extent, len(seg), seg.copy()))
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
    # (t_ns, x, y) history -> net-displacement test that separates true MOVERS
    # from static objects whose apparent KF velocity is inflated by occlusion /
    # centroid shift / cluster merging (measured up to 2 m/s on a stationary
    # cylinder). A static object jitters in place -> ~0 net displacement; a
    # mover translates. Instantaneous speed alone cannot tell them apart.
    hist: deque = field(default_factory=lambda: deque(maxlen=128))
    # Hysteresis state: whether this track was published as a mover last cycle,
    # and how many cycles of "no longer moving" it has been held for since.
    # A single threshold on a noisy speed estimate makes tracks flicker in and
    # out of /gmpc/obstacles, which changes the CBF's constraint SET from one
    # control step to the next. Measured on a real run: the obstacle count
    # changed 395 times in one episode, and acceleration saturated 26% of the
    # time within 0.2 s of such a change versus 10% elsewhere. Latching the
    # decision removes that churn without weakening the filter.
    is_mover: bool = False
    # The cluster's laser returns, in the map frame, refreshed every cycle.
    # These ARE the obstacle as far as the CBF is concerned: a lidar return is a
    # point on the surface, which is the only thing actually observed. Fitting a
    # shape to them adds an assumption (and, for anything longer than about 2:1,
    # an error larger than the bias it removes) without adding information.
    pts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    release: int = 0


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
        # Hysteresis on that gate. A track must reach min_track_speed to be
        # admitted but only drops out below release_track_speed, then is held
        # for track_release_frames more cycles. Set release == min and frames 0
        # to get the original single-threshold behaviour back.
        # Defaults reproduce the original single-threshold behaviour so the
        # validated configuration is unchanged; the A/B harness turns hysteresis
        # on explicitly (release 0.06 / 5 frames were the values verified to
        # remove the flicker).
        p('release_track_speed', 0.10)    # m/s, exit threshold (<= min_track_speed)
        p('track_release_frames', 0)      # cycles to keep publishing after exit
        # Net-displacement gate (separates true movers from static objects whose
        # apparent KF velocity is inflated by occlusion/centroid-shift). A track
        # is only published to the dynamic CBF if its AVERAGE speed over the last
        # `static_window_s` (= net displacement / elapsed) also exceeds
        # `min_net_speed`. Static objects jitter in place -> ~0 net displacement
        # -> dropped here, left to the local costmap/planner (their proper owner).
        # 0.8 s is too short: the observed centroid of an unknown cylinder
        # shifts as the viewing arc changes, which reads as net displacement over
        # a short window. Over 2 s that shift is bounded by the cylinder radius
        # while a real mover keeps accumulating.
        p('static_window_s', 2.0)         # s, window for the net-displacement test
        # Was 0.0, which SKIPPED the test entirely (the code guards on > 0.0).
        # With it off, static cylinders reached /gmpc/obstacles in 10-32% of
        # frames and -- once the planner was switched to the filtered scan --
        # were masked out of its costmap too, so nothing owned them.
        p('min_net_speed',   0.05)        # m/s, min AVERAGE speed over the window
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
        p('static_cbf_max', 4)            # cap total static points (QP size)
        p('static_cbf_radius', 0.05)
        # Surface-point CBF: how many of a cluster's own laser returns to hand
        # the controller, and how close they have to be to matter. A lidar
        # return IS a point on the obstacle's surface, so no shape is assumed --
        # a pillar, a wall corner, a person and an L-shaped cart are all just
        # point sets. The cap keeps the QP small: each point becomes one
        # constraint per horizon step.
        p('surf_pts_max', 4)              # per cluster
        p('surf_pts_range', 2.5)          # m, ignore points further than this
        p('surf_pts_min_sep', 0.20)       # m, thin them out so they spread      # m, point radius (small; margin does work)

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
        self.release_speed  = min(float(g('release_track_speed')), self.min_speed)
        self.release_frames = int(g('track_release_frames'))
        self.static_window_s = float(g('static_window_s'))
        self.min_net_speed   = float(g('min_net_speed'))
        self.static_cbf_en  = bool(g('static_cbf_enable'))
        self.static_range   = float(g('static_cbf_range'))
        self.static_max     = int(g('static_cbf_max'))
        self.static_radius  = float(g('static_cbf_radius'))
        self.surf_max       = int(g('surf_pts_max'))
        self.surf_range     = float(g('surf_pts_range'))
        self.surf_min_sep   = float(g('surf_pts_min_sep'))
        self._kf_kwargs = dict(
            sigma_pos=float(g('kf_sigma_pos')), sigma_vel=float(g('kf_sigma_vel')),
            sigma_meas=float(g('kf_sigma_meas')), init_vel_var=float(g('kf_init_vel_var')))

        self._tracks: List[Track] = []
        self._next_id = 0
        self._occ: np.ndarray | None = None   # inflated occupied mask (row,col)
        self._map_info = None
        # Stable nearest-wall-point field (from the KNOWN map, precomputed once):
        # for every free cell, the (row,col) of the closest occupied cell. The
        # static-CBF reads this instead of per-frame scan hits, so the wall
        # anchors move SMOOTHLY with the robot (no chattering near walls).
        self._wall_dist = None                 # (H,W) distance-to-wall in cells
        self._wall_nn_row = None
        self._wall_nn_col = None

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
        # Precompute the stable nearest-wall-point field from the RAW (un-inflated)
        # walls -> the static-CBF anchor is the true wall surface (margin adds the
        # safety distance) and moves smoothly as the robot drives, eliminating the
        # per-frame scan-point flicker that chattered the controller near walls.
        if _HAVE_NDIMAGE:
            raw_occ = grid >= 50
            if raw_occ.any():
                dist, idx = distance_transform_edt(~raw_occ, return_indices=True)
                self._wall_dist   = dist
                self._wall_nn_row = idx[0]
                self._wall_nn_col = idx[1]
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
        a = scan.angle_min
        rmin, rmax = scan.range_min, scan.range_max
        for r in scan.ranges:
            if math.isfinite(r) and rmin < r < rmax:
                lx, ly = r * math.cos(a), r * math.sin(a)
                mx = tx + cyaw * lx - syaw * ly
                my = ty + syaw * lx + cyaw * ly
                # Background subtraction: drop hits on known (inflated) walls; the
                # rest are dynamic-obstacle candidates. Walls are owned by the
                # costmap/planner and (for the CBF) by the map-based static-CBF.
                if not (self.use_map and self._is_static(mx, my)):
                    dyn.append((mx, my))
            a += scan.angle_increment
        dyn = np.array(dyn, dtype=float) if dyn else np.empty((0, 2))

        clusters = cluster_adjacent(dyn, self.cluster_gap, self.min_pts,
                                    self.max_radius, sensor_xy=(tx, ty),
                                    r_prior=self.default_radius)

        t_ns = (int(scan.header.stamp.sec) * 1_000_000_000
                + int(scan.header.stamp.nanosec)) or self.get_clock().now().nanoseconds
        self._update_tracks(clusters, t_ns)
        # ONE decision per cycle, shared by both consumers. _is_mover mutates the
        # track's hysteresis state, so it must not be evaluated twice; and the
        # scan mask previously used its OWN, laxer test (instantaneous KF speed
        # only), which meant the planner was blinded to objects the CBF did not
        # even consider dynamic -- unknown static cylinders leaked through it
        # 10-32% of frames and the robot then hit one at -0.036 m.
        movers = [tr for tr in self._tracks if self._is_mover(tr, t_ns)]
        mv = set(id(tr) for tr in movers)
        # Tracked clusters that are NOT movers are unknown STATIC objects: real
        # obstacles that are absent from the map, so the map-based static-CBF
        # cannot see them, and not dynamic, so the dynamic CBF ignores them.
        # Until now they were simply dropped and left to the planner's costmap --
        # which works only while the planner reads the raw scan. Decoupling the
        # planner removed that owner and the robot hit one (pillar at (9,8),
        # -0.036 m). They belong in the static-CBF set, at v = 0.
        statics = [tr for tr in self._tracks
                   if id(tr) not in mv and tr.age >= self.min_age]
        self._publish(t_ns, scan.header.stamp, movers, tx, ty)
        self._publish_filtered_scan(scan, tx, ty, cyaw, syaw, movers)
        self._publish_static_obstacles(tx, ty, statics)

    # ------------------------------------------------------------------
    def _publish_static_obstacles(self, rx, ry, statics=()):
        """Solution-1 static-CBF, MAP-BASED (stable). For the robot pose and a few
        small offsets (to catch walls on several sides in a corridor), look up the
        precomputed nearest-wall cell from the KNOWN map and publish those points
        as zero-velocity obstacles. The GMPC merges them into its CBF set so it
        won't dodge a dynamic obstacle into a wall.

        Why map-based: the old version re-picked nearest points from the per-frame
        /scan, so the anchors jumped (grid snap, sector reassignment, ray hits
        appearing/disappearing) -> the near-hard CBF chased a flickering constraint
        -> the controller chattered whenever it drove close to a wall. The map is
        stationary and known, so its nearest-wall point is a SMOOTH function of the
        robot position: no flicker, no chatter. Falls back to nothing if scipy is
        unavailable. Always publishes (empty array clears the controller's set)."""
        arr = Float32MultiArray()
        if self.static_cbf_en and self._wall_nn_row is not None:
            info = self._map_info
            res  = info.resolution
            ax, ay = info.origin.position.x, info.origin.position.y
            h, w = self._wall_dist.shape
            pts = {}
            # robot cell + 4 lateral probes (~robot radius) -> surrounding walls
            for dx, dy in ((0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)):
                col = int((rx + dx - ax) / res)
                row = int((ry + dy - ay) / res)
                if not (0 <= row < h and 0 <= col < w):
                    continue
                wx = ax + (int(self._wall_nn_col[row, col]) + 0.5) * res
                wy = ay + (int(self._wall_nn_row[row, col]) + 0.5) * res
                if math.hypot(wx - rx, wy - ry) > self.static_range:
                    continue
                pts[(round(wx / res), round(wy / res))] = (wx, wy)   # dedupe
            data = []
            for (wx, wy) in sorted(pts.values(),
                                   key=lambda p: math.hypot(p[0]-rx, p[1]-ry))[:self.static_max]:
                data.extend((wx, wy, self.static_radius, 0.0, 0.0))
            arr.data = data
        else:
            arr.data = []
        # Unknown static objects, appended at their FITTED radius rather than
        # the small wall-point radius: a pillar is an extended body, and the
        # circle fit already estimates how extended (0.283-0.313 m measured on
        # the 0.30 m cylinders). v = 0, so no prediction is involved.
        data = list(arr.data)
        # Unknown statics go in as SURFACE POINTS too, at v = 0 -- same
        # representation as the movers and as the map-based wall points, so the
        # controller sees one kind of thing throughout and no shape is assumed
        # anywhere in the pipeline.
        for tr in sorted(statics,
                         key=lambda z: math.hypot(z.kf.position[0]-rx,
                                                  z.kf.position[1]-ry))[:self.static_max]:
            if math.hypot(tr.kf.position[0] - rx,
                          tr.kf.position[1] - ry) > self.static_range + 1.0:
                continue
            for sx, sy in self._surface_points(tr, rx, ry):
                data.extend((float(sx), float(sy), 0.0, 0.0, 0.0))
        arr.data = data
        self.static_pub.publish(arr)

    # ------------------------------------------------------------------
    def _publish_filtered_scan(self, scan, tx, ty, cyaw, syaw, moving):
        """Republish the scan with rays hitting confirmed MOVING obstacles set
        to +inf, so the global costmap (planner) sees static geometry only.

        `moving` is the SAME set published to the dynamic CBF. Masking a ray the
        CBF is not protecting against would leave that object owned by nobody:
        invisible to the planner and only intermittently seen by the controller.
        """
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
            tr.pts = clusters[ci][4]
            tr.last_t_ns = t_ns
            tr.age += 1
            tr.misses = 0
            tr.hist.append((t_ns, tr.kf.position[0], tr.kf.position[1]))

        for ci in um_c:                       # new track
            kf = KalmanTracker2D(init_xy=(clusters[ci][0], clusters[ci][1]),
                                 **self._kf_kwargs)
            nt = Track(tid=self._next_id, kf=kf,
                       radius=max(self.default_radius, clusters[ci][2]),
                       last_t_ns=t_ns, pts=clusters[ci][4])
            nt.hist.append((t_ns, kf.position[0], kf.position[1]))
            self._tracks.append(nt)
            self._next_id += 1

        for ti in um_t:
            self._tracks[ti].misses += 1

        # Age out stale tracks.
        keep = []
        for tr in self._tracks:
            if (t_ns - tr.last_t_ns) * 1e-9 <= self.track_timeout:
                keep.append(tr)
        self._tracks = keep

    def _is_mover(self, tr, now_ns) -> bool:
        """True if a track is a genuine DYNAMIC obstacle for the CBF: aged,
        instantaneously moving, AND actually net-displaced over the window.
        The net-displacement test rejects static objects (unknown cylinders,
        map-subtraction leaks) whose apparent KF velocity is inflated by
        occlusion / centroid shift -> they jitter in place but don't translate.
        Such statics stay out of the CBF and are handled by the costmap/planner
        (which sees the raw /scan).

        The speed test is HYSTERETIC: a track has to reach `min_track_speed` to
        be admitted, but only drops out below `release_track_speed`, and even
        then is held for `track_release_frames` more cycles. With one threshold
        on a noisy estimate a track sitting near the gate toggles every few
        frames, and each toggle changes the CBF's constraint set discontinuously,
        which shows up directly as saturated acceleration. Note the estimate is
        the obstacle's speed in the MAP frame (absolute, robot motion already
        removed by the TF projection), so it also carries AMCL error and
        centroid drift -- all the more reason not to threshold it sharply.
        """
        if tr.age < self.min_age:
            return False
        speed = math.hypot(*tr.kf.velocity)
        gate = self.release_speed if tr.is_mover else self.min_speed
        if speed < gate:
            if tr.is_mover and tr.release < self.release_frames:
                tr.release += 1          # hold the old decision a little longer
                return True
            tr.is_mover = False
            tr.release = 0
            return False
        tr.release = 0
        if self.min_net_speed > 0.0 and tr.hist:
            cx, cy = tr.kf.position
            ot, ox, oy = tr.hist[0]
            for (t, x, y) in tr.hist:           # oldest entry within the window
                if (now_ns - t) * 1e-9 <= self.static_window_s:
                    ot, ox, oy = t, x, y
                    break
            elapsed = (now_ns - ot) * 1e-9
            if elapsed >= 0.3:                  # enough history to judge motion
                if math.hypot(cx - ox, cy - oy) / elapsed < self.min_net_speed:
                    tr.is_mover = False
                    return False                # jitters in place -> static
        tr.is_mover = True
        return True

    # ------------------------------------------------------------------
    def _surface_points(self, tr, rx, ry):
        """The cluster's own returns, nearest first, thinned and capped.

        Only the near ones can be violated, and two returns 2 cm apart carry the
        same constraint twice, so they are thinned by `surf_pts_min_sep` before
        the cap is applied -- otherwise the cap would spend itself on a single
        patch of surface and leave the rest of the object unconstrained.

        Falls back to the tracked centre if a track has no points this cycle
        (it was coasted through a miss), which is the old behaviour.
        """
        P = getattr(tr, 'pts', None)
        if P is None or len(P) == 0:
            cx, cy = tr.kf.position
            return [(float(cx), float(cy))]
        d = np.hypot(P[:, 0] - rx, P[:, 1] - ry)
        keep = d <= self.surf_range
        if not np.any(keep):
            keep = d <= np.min(d) + 1e-6
        Q = P[keep][np.argsort(d[keep])]
        out = []
        for q in Q:
            if len(out) >= self.surf_max:
                break
            if all(math.hypot(q[0] - o[0], q[1] - o[1]) >= self.surf_min_sep
                   for o in out):
                out.append((float(q[0]), float(q[1])))
        return out or [(float(Q[0][0]), float(Q[0][1]))]

    def _publish(self, t_ns: int, stamp, confirmed, rx=0.0, ry=0.0):
        # `confirmed` is decided once per cycle by the caller (see _scan_cb) and
        # shared with the scan mask, so the planner is blinded to exactly the
        # objects the CBF has taken responsibility for -- no more, no less.

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
            # One entry per covering disc. The CBF form is unchanged -- each is
            # still h = ||p - o||^2 - r_eff^2 -- so the QP stays convex and the
            # only cost is a few more rows. A compact obstacle yields exactly
            # one disc, so round obstacles pay nothing; an elongated one is
            # followed by its outline instead of by a circumscribed circle that
            # would be mostly empty space (a 1.2 x 0.3 m cart needs r = 0.65,
            # which with the margin is a 1.03 m keep-out -- wider than gaps it
            # physically fits through).
            for sx, sy in self._surface_points(tr, rx, ry):
                PX, PY, VX, VY = to_pub(sx, sy, vx, vy)
                # radius 0: the point is ON the surface already, so the whole
                # keep-out comes from the margin, exactly as it does for the
                # map-based wall points that have run all along.
                flat.extend([PX, PY, 0.0, VX, VY])
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
