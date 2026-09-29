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

  occlusion is never      Where the arm blocks the LiDAR, removing its own
  free space              returns leaves NO measurement, so a point whose
                          nearest surface lies in an occluded bearing is flagged
                          occluded, never clear. The same applies when the check
                          cannot be run at all: if the sensor transform is
                          missing, observability is unknown and is reported as
                          occluded rather than defaulting to clear.

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

import json
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
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener

from .arm_detection_points import (Obstacle, _closest_local, _inv, _iso,
                                   _quat_to_rot, _rpy_to_rot)

STATUS_OK, STATUS_UNKNOWN, STATUS_STALE, STATUS_NODATA = 0.0, 1.0, 2.0, 3.0
# Emitted when the band had to be truncated to stay inside a row limit.
# It rides in the cloud rather than in the diagnostic because the safety
# node reads the cloud: a fact the consumer must act on has to travel on
# the channel the consumer reads.
STATUS_OVERFLOW = 4.0

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
# link / ox / oy / oz / rho carry the whole-link representation: which link the
# row belongs to, where on that link the row sits in its own frame so the
# Jacobian can be taken there, and the certified covering radius to subtract.
#
# NOT backward compatible, despite the first ten fields keeping their meaning
# and offsets. A consumer that reads the message by FIELD NAME still works. One
# that assumes the width -- reshape(msg.width, 10), which is what the previous
# consumer in this package did -- gets a shape error, and a subscriber that
# indexes rows by a fixed stride reads garbage. Both ends of this wire were
# updated together; anything else reading it has to be updated too.
# `obs` 是產生該列距離的**障礙物索引**（−1 表示無）。加在最後，
# 既有欄位的名稱與位移不變；消費端一律由 point_step 取寬度，因此相容。
# 索引對應的名稱由 ~/obstacle_names 發布（JSON 陣列，latched）。
# `vox,voy,voz` 是該列**障礙物表面點**的速度（報告座標系，m/s），
# `vobs` 是其可用性（0 未知／1 宣告靜態／2 新鮮可用）。
# 距離變化是 d_dot = n^T (v_obs − v_link)，只看機器人側會把隨夾爪移動的
# 橫桿當成靜止。四個欄位加在最後，既有欄位的名稱與位移不變；
# 消費端一律由 point_step 取寬度。**兩端必須一起更新。**
FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded',
          'link', 'ox', 'oy', 'oz', 'rho', 'obs', 'vox', 'voy', 'voz', 'vobs']
VOBS_UNKNOWN, VOBS_STATIC, VOBS_OK = 0, 1, 2



def model_twist(T_prev, t_prev, T_now, t_now, min_dt=1.0e-4):
    """由兩個世界位姿估物件的 twist（v, omega）。回傳 (v, omega) 或 None。

    **旋轉也要算進去**：表面點的速度是 v + omega x (p − 原點)，
    物件轉動時即使原點不動，表面點仍有速度。
    """
    if T_prev is None or T_now is None:
        return None
    dt = float(t_now) - float(t_prev)
    if not np.isfinite(dt) or dt < min_dt:
        return None
    v = (T_now[:3, 3] - T_prev[:3, 3]) / dt
    R = T_now[:3, :3] @ T_prev[:3, :3].T
    # log(R) 的向量部分；小角度下取反對稱部分即可，大角度改用 arccos 標準式
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = float(np.arccos(c))
    nw = float(np.linalg.norm(w))
    omega = (w / dt / 2.0 if th < 1.0e-6 or nw < 1.0e-12
             else (w / nw) * (th / dt))
    if not (np.isfinite(v).all() and np.isfinite(omega).all()):
        return None
    return v, omega


def surface_point_velocity(twist, origin, pts):
    """表面點速度 v + omega x (p − 原點)。twist 為 None 時回傳 None。"""
    if twist is None:
        return None
    v, omega = twist
    return v[None, :] + np.cross(omega[None, :], pts - origin[None, :])


def parse_obstacles(specs):
    """障礙物設定字串 → Obstacle 串列。**離線核對與節點共用同一份解析。**"""
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


def expand_pair_rows(specs, exempt_specs, obstacle_names):
    """解析 `pair_rows` 與 `pair_rows_exempt`，回傳 ({連桿: [障礙物…]}, {連桿: {免列…}})。

    `<link>:*` 展開為**該連桿對所有非免列障礙物**。漏列問題不限於某一個物件：
    接觸例外對象成為最近者並被刪列後，**任何**其他物件都可能完全沒有列
    （面板 28 mm 合格、支柱 40 mm 違規卻無列可約束）。因此逐一補齊。

    免列對象是接觸例外物件本身；它仍有一般的最近列，只是在允許相位被濾掉。
    """
    known = set(obstacle_names)
    exempt: dict[str, set[str]] = {}
    for sp in [x for x in (exempt_specs or []) if str(x).strip()]:
        lk, ob = str(sp).split(':')
        if ob not in known:
            raise ValueError(f'pair_rows_exempt 指定了不存在的障礙物：{sp!r}')
        exempt.setdefault(lk, set()).add(ob)
    pairs: dict[str, list[str]] = {}
    for sp in [x for x in (specs or []) if str(x).strip()]:
        lk, ob = str(sp).split(':')
        if ob == '*':
            exp = [n for n in obstacle_names if n not in exempt.get(lk, set())]
            if not exp:
                raise ValueError(f'pair_rows {sp!r} 展開後為空')
            pairs.setdefault(lk, []).extend(exp)
        elif ob not in known:
            raise ValueError(f'pair_rows 指定了不存在的障礙物：{sp!r}')
        else:
            pairs.setdefault(lk, []).append(ob)
    for lk in pairs:                          # 去重、保持設定順序
        pairs[lk] = list(dict.fromkeys(pairs[lk]))
    return pairs, exempt


def forced_pair_rows(W, pts_local, obstacle, obs_idx, li, rho, status, age,
                     occ_of, max_range, max_rows=None, vobs_of=None):
    """對**指定的 (連桿, 障礙物) 配對**單獨算距離並產生列。

    為什麼需要：一般列只保留每個取樣點的**最近**障礙物。若最近的是接觸例外
    對象（例如橫桿），濾波器依規則刪掉該列之後，其他物件（例如面板）就**完全
    沒有列** —— 例外會連帶遮掉別的避碰。因此對有配對規則的組合另外補列。
    """
    from .arm_link_geometry import obstacle_distances as _od
    d, v, _w = _od(W, [obstacle])
    dmin = float(d.min())
    if dmin > max_range:
        return []
    sel = np.nonzero(d <= dmin + rho)[0]
    sel = sel[np.argsort(d[sel])]
    if max_rows:
        sel = sel[:max_rows]
    out = []
    for k in sel:
        n_hat = v[k] / max(abs(float(d[k])), 1e-9)
        if vobs_of is None:
            vo, vst = np.zeros(3), VOBS_STATIC
        else:
            vo, vst = vobs_of(obstacle, W[k] + v[k])
        out.append(list(W[k]) + list(n_hat)
                   + [float(d[k]), float(status), age, float(occ_of(W[k], v[k])),
                      float(li)] + list(pts_local[k]) + [float(rho),
                                                         float(obs_idx)]
                   + [float(vo[0]), float(vo[1]), float(vo[2]), float(vst)])
    return out


def _expand(path: str) -> str:
    """Read a URDF, expanding it first if it is still a .xacro."""
    import subprocess
    return (subprocess.check_output(['xacro', path], text=True)
            if path.endswith('.xacro') else open(path).read())


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
        # **必要配對列**：'link:obstacle'。對這些組合**另外**產生距離列，
        # 不受「只留最近障礙物」影響 —— 否則接觸例外會連帶遮掉其他物件的列。
        p('pair_rows', [''])
        p('pair_rows_exempt', [''])
        # 'points' keeps the twelve fixed detection frames, retained only so the
        # old behaviour can be reproduced; measured against the link meshes they
        # understate clearance by up to 0.238 m. 'links' is the certified
        # sampling representation.
        p('geometry', 'links')
        p('wholebody_urdf', '')
        p('rho_target', 0.015)
        p('cert_tol', 0.001)
        # History of this number, kept because the reasoning matters more than
        # the value: it was 60 when only the arm was in the barrier, and the
        # chassis pushed the largest band to 90, so a cap of 60 silently dropped
        # up to 30 rows per cycle while the coverage claim went on being made.
        # The answer is not a bigger cap, it is no cap plus an error when one is
        # imposed. The measurement below stands as the cost record.
        #
        # 100 since the chassis joined the barrier. Measured over a whole-body
        # run with the cap raised to 400 so it could not bite, the largest band
        # any link produced was base_link with 90 rows (link2 77, link4 62,
        # link1 48); total rows peaked at 392 with zero dropped. 100 sits above
        # the observed maximum, which is the only kind of cap the band argument
        # survives -- a cap that bites throws away rows the covering guarantee
        # depends on. It is an observed maximum over one scenario, not a proof
        # for every pose, which is why the dropped count is published every
        # cycle: if it is ever non-zero the guarantee is void for that cycle and
        # it says so rather than being assumed away.
        #
        # Cost: the safety solve went from p50 10.85 / p95 11.75 ms at 179 rows
        # to p50 13.31 / p95 27.44 / max 42.17 ms at 392, against a 50 ms
        # period. The deadline margin is now thin and is a known issue.
        #
        # The earlier reasoning for 60, from when only the arm was in:
        # 60, not a smaller number that looks cheap. Measured over 100 random
        # poses and commanded velocities, a cap of 8 left the solution up to
        # 32.98 mm/s worse against the full row set on 12 cycles and moved the
        # output by 1.786 rad/s; 20 still broke 8 rows that the full set held;
        # 40 broke none but moved the output by 2.9e-03; 60 reproduces the
        # uncapped solution exactly. A sample further from the obstacle can
        # still bind, because the row is n^T J v and both the normal and the
        # Jacobian change from point to point.
        # 0 = no cap, which is the default. The full band is what the covering
        # argument is stated over, so it is what gets published.
        p('max_rows_per_link', 0)
        p('publish_rate', 30.0)
        p('pose_timeout', 0.5)      # s, obstacle pose older than this is stale
        # 物件速度估值的有效期。過期 ⇒ VOBS_UNKNOWN，**不當成零速**。
        p('vobs_timeout', 0.2)
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
        self.vobs_timeout = float(g('vobs_timeout'))
        self.occl_timeout = float(g('occl_timeout'))
        self.require_occl = bool(g('require_occlusion_feed'))
        self.max_range = float(g('max_range'))

        self.obstacles = self._parse([s for s in g('obstacles') if s.strip()])
        self.geometry = str(g('geometry'))
        self.max_rows_per_link = int(g('max_rows_per_link'))
        self.samples = None
        self.link_names = []
        if self.geometry == 'links':
            from .arm_link_geometry import (arm_link_names,
                                            sample_links_certified)
            src = str(g('wholebody_urdf'))
            if not src:
                raise RuntimeError(
                    'geometry:=links needs wholebody_urdf: the barrier is built '
                    'from the collision meshes of the 9-DOF model, not from the '
                    'gz description this node would otherwise reach for')
            self.samples = sample_links_certified(
                _expand(src), rho_target=float(g('rho_target')),
                tol=float(g('cert_tol')))
            self.link_names = arm_link_names(_expand(src))
            assert self.link_names == list(self.samples), (
                'link order disagreement between sampler and name list')
            self.get_logger().info(
                'arm_link_distance: certified sampling -- '
                + ', '.join(f'{k} {len(v.points)}pt rho {v.rho*1000:.1f}mm'
                            for k, v in self.samples.items()))
        self._stamp: dict[str, float] = {}
        self._pose_prev: dict[str, tuple] = {}
        self._twist: dict[str, tuple] = {}
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
        self._obs_index = {o.name: i for i, o in enumerate(self.obstacles)}
        self._pair_rows, self._pair_exempt = expand_pair_rows(
            g('pair_rows'), g('pair_rows_exempt'),
            [o.name for o in self.obstacles])
        for _lk, _obs in self._pair_rows.items():
            self.get_logger().info(
                f'必要配對列 {_lk}：{len(_obs)} 個障礙物 {_obs}'
                f'（免列：{sorted(self._pair_exempt.get(_lk, set()))}）')
        # 障礙物索引→名稱（latched）：下游據此做配對層級規則
        from rclpy.qos import QoSProfile, DurabilityPolicy
        _lat = QoSProfile(depth=1)
        _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.names_pub = self.create_publisher(String, '~/obstacle_names', _lat)
        _nm = String(); _nm.data = json.dumps([o.name for o in self.obstacles])
        self.names_pub.publish(_nm)
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
        return parse_obstacles(specs)

    def _make_cb(self, model: str):
        def cb(msg: PoseStamped) -> None:
            q, t = msg.pose.orientation, msg.pose.position
            T = _iso(_quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))
            now = self.get_clock().now().nanoseconds * 1e-9
            # **物件 twist**：由連續兩筆位姿估，含旋轉。估不出來就留 None ——
            # 下游會因此標為 VOBS_UNKNOWN，**不會當成零速**。
            prev = self._pose_prev.get(model)
            tw = None if prev is None else model_twist(prev[0], prev[1], T, now)
            self._twist[model] = (tw, now)
            self._pose_prev[model] = (T, now)
            self._stamp[model] = now
            for o in self.obstacles:
                if o.model == model:
                    o.T_world_link = T
        return cb

    def _vobs_for(self, o, p_surf):
        """障礙物表面點的速度與可用性。

        靜態物件（設定裡沒有 model）**確為零**；動態物件缺 twist 或 twist
        過期一律 VOBS_UNKNOWN，**不以零代替**。
        """
        if not o.model:
            return np.zeros(3), VOBS_STATIC
        rec = self._twist.get(o.model)
        if rec is None or rec[0] is None:
            return np.zeros(3), VOBS_UNKNOWN
        tw, t_tw = rec
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - t_tw > self.vobs_timeout:
            return np.zeros(3), VOBS_UNKNOWN
        V = surface_point_velocity(tw, o.T_world_link[:3, 3],
                                   np.asarray(p_surf, float)[None, :])
        if V is None or not np.isfinite(V).all():
            return np.zeros(3), VOBS_UNKNOWN
        return V[0], VOBS_OK

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

        if self.geometry == 'links':
            rows, n_ok, n_unk, n_stale, n_nodata, worst_age = \
                self._rows_links(now, T_rl)
        else:
            rows, n_ok, n_unk, n_stale, n_nodata, worst_age = \
                self._rows_points(now, T_rl)

        if getattr(self, '_overflow', False) and rows:
            blank = list(rows[0])
            blank[7] = STATUS_OVERFLOW
            rows = rows + [blank]
        self._publish(rows)
        d = Float32MultiArray()
        #  0 n_points 1 ok 2 occluded 3 stale 4 nodata 5 worst_age 6 min_d
        #  7 rows dropped by max_rows_per_link this cycle
        finite = [r[6] for r in rows if r[7] == STATUS_OK]
        d.data = [float(len(rows)), float(n_ok), float(n_unk), float(n_stale),
                  float(n_nodata), float(worst_age),
                  float(min(finite)) if finite else -1.0,
                  float(getattr(self, '_dropped', 0))]
        self.diag.publish(d)


    def _rows_points(self, now, T_rl):
        """The twelve fixed detection frames. Regression path only."""
        rows = []
        worst_age = 0.0
        n_ok = n_unk = n_stale = n_nodata = 0
        for fr in self.frames:
            T = self._tf(self.report_frame, fr)
            if T is None:
                rows.append([0.0] * 6 + [0.0, STATUS_NODATA, -1.0, 0.0]
                            + [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                            + [0.0, 0.0, 0.0, float(VOBS_STATIC)])
                n_nodata += 1
                continue
            p = T[:3, 3]

            best_d, best_v, best_age, best_i = math.inf, None, 0.0, -1
            for _oi, o in enumerate(self.obstacles):
                if o.T_world_link is None:
                    continue
                age = 0.0 if not o.model else now - self._stamp.get(o.model, 0.0)
                T_wc = o.T_world_link @ o.T_link_collision
                p_loc = (_inv(T_wc) @ np.append(p, 1.0))[:3]
                surf = (T_wc @ np.append(_closest_local(o, p_loc), 1.0))[:3]
                v = surf - p
                dist = float(np.linalg.norm(v))
                if dist < best_d:
                    best_d, best_v, best_age, best_i = dist, v, age, _oi

            if best_v is None or best_d > self.max_range:
                rows.append(list(p) + [0.0, 0.0, 0.0, self.max_range,
                                       STATUS_NODATA, -1.0,
                                       1.0 if T_rl is None else 0.0]
                            + [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                            + [0.0, 0.0, 0.0, float(VOBS_STATIC)])
                n_nodata += 1
                continue

            n_hat = best_v / max(best_d, 1e-9)
            status = STATUS_OK
            if best_age > self.pose_timeout:
                status = STATUS_STALE
                n_stale += 1
            else:
                n_ok += 1
            # Observability defaults to UNKNOWN, not to clear.
            #
            # This used to read `if T_rl is not None:` with occ left at 0.0
            # otherwise, so a missing report_frame -> lidar_frame transform
            # reported every direction as observed. The occlusion test itself
            # fails closed; being unable to run it was the hole. Not knowing
            # where the sensor is means not knowing what it can see.
            if T_rl is None:
                occ = 1.0
                n_unk += 1
            else:
                p_l = (_inv(T_rl) @ np.append(p + best_v, 1.0))[:3]
                occ = 1.0 if self._occluded(p_l) else 0.0
                if occ:
                    n_unk += 1
            worst_age = max(worst_age, best_age)
            # **這條路徑先前只填 15 欄，FIELDS 卻是 16** —— 加 `obs` 欄位時漏改，
            # 會讓 geometry:=points 發出的雲寬度不符。一併補齊（obs = −1 表示
            # 此路徑不帶障礙物身分），並加上四個速度欄位。
            rows.append(list(p) + list(n_hat) + [best_d, status, best_age, occ]
                        + [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                        + [0.0, 0.0, 0.0, float(VOBS_STATIC)])

        return rows, n_ok, n_unk, n_stale, n_nodata, worst_age


    def _rows_links(self, now, T_rl):
        """One row per band-selected sample on every arm link.

        Each link's certified samples go to the report frame through TF, get
        signed distances to every live obstacle, and the band
        [d_min, d_min + rho] is kept. That band provably contains a sample
        within rho of the link's true closest point, which is the property the
        |omega| rho velocity allowance in the filter rests on; taking only the
        nearest sample does not provide it, and held-out testing of that version
        found the true approach rate beating the constrained one by 37.9 mm/s.
        """
        from .arm_link_geometry import obstacle_distances
        rows = []
        worst_age = 0.0
        n_ok = n_unk = n_stale = n_nodata = 0
        self._dropped = 0
        self._overflow = False
        live = [o for o in self.obstacles if o.T_world_link is not None]
        _by = {o.name: o for o in live}

        def _vo_row(obs_name, p_surf):
            o = _by.get(obs_name)
            if o is None:
                return (0.0, 0.0, 0.0, float(VOBS_UNKNOWN))
            vo, vst = self._vobs_for(o, p_surf)
            return (float(vo[0]), float(vo[1]), float(vo[2]), float(vst))

        for li, name in enumerate(self.link_names):
            T = self._tf(self.report_frame, name)
            S = self.samples[name]
            blank = [float(li), 0.0, 0.0, 0.0, float(S.rho), -1.0,
                     0.0, 0.0, 0.0, float(VOBS_STATIC)]
            if T is None or not live:
                rows.append([0.0] * 6 + [0.0, STATUS_NODATA, -1.0, 1.0] + blank)
                n_nodata += 1
                continue
            W = (S.points @ T[:3, :3].T) + T[:3, 3]
            d, v, which = obstacle_distances(W, live)
            age = max((0.0 if not o.model else now - self._stamp.get(o.model, 0.0))
                      for o in live)
            worst_age = max(worst_age, age)
            status = STATUS_STALE if age > self.pose_timeout else STATUS_OK
            dmin = float(d.min())
            if dmin > self.max_range:
                rows.append(list(T[:3, 3]) + [0.0, 0.0, 0.0, self.max_range,
                                              STATUS_NODATA, -1.0,
                                              1.0 if T_rl is None else 0.0]
                            + blank)   # blank 已含四個速度欄位
                n_nodata += 1
                continue
            sel = np.nonzero(d <= dmin + S.rho)[0]
            sel = sel[np.argsort(d[sel])]
            self._band_max = max(getattr(self, '_band_max', 0), len(sel))
            # Truncation breaks the covering argument the |omega| rho allowance
            # rests on, so how much of it was thrown away is published rather
            # than assumed to be nothing. A cap that never bites is the only
            # cap the band guarantee survives.
            # Truncation is now an ERROR, not a silent economy. The covering
            # argument the barrier rests on is over the WHOLE band; keeping the
            # nearest N of it and carrying on reports a coverage that no longer
            # holds. A cap of 0 means no cap, which is the default: the full band
            # is published. If a cap is set and would bite, the rows go out
            # flagged STATUS_OVERFLOW so the safety node stops, rather than the
            # rows quietly going missing.
            if self.max_rows_per_link and len(sel) > self.max_rows_per_link:
                self._dropped += len(sel) - self.max_rows_per_link
                self._overflow = True
                sel = sel[:self.max_rows_per_link]
            for k in sel:
                p_w = W[k]
                n_hat = v[k] / max(abs(float(d[k])), 1e-9)
                if T_rl is None:
                    occ = 1.0
                    n_unk += 1
                else:
                    p_l = (_inv(T_rl) @ np.append(p_w + v[k], 1.0))[:3]
                    occ = 1.0 if self._occluded(p_l) else 0.0
                    n_unk += int(occ)
                if status == STATUS_OK:
                    n_ok += 1
                else:
                    n_stale += 1
                # **障礙物身分**：由 obstacle_distances 回傳的名稱轉成索引，
                # 供下游做**配對層級**的規則（哪個連桿對哪個物件）。
                rows.append(list(p_w) + list(n_hat)
                            + [float(d[k]), float(status), age, occ,
                               float(li)] + list(S.points[k])
                            + [float(S.rho),
                               float(self._obs_index.get(which[k], -1))]
                            + list(_vo_row(which[k], p_w + v[k])))

            # **必要配對列**：即使該配對不是最近障礙物也照樣產生
            def _occ_of(p_w, vk):
                if T_rl is None:
                    return 1.0
                p_l = (_inv(T_rl) @ np.append(p_w + vk, 1.0))[:3]
                return 1.0 if self._occluded(p_l) else 0.0

            for _obn in self._pair_rows.get(name, []):
                _oi = self._obs_index[_obn]
                _ob = self.obstacles[_oi]
                if _ob.T_world_link is None:
                    continue
                _extra = forced_pair_rows(
                    W, S.points, _ob, _oi, li, S.rho, status, age, _occ_of,
                    self.max_range, self.max_rows_per_link,
                    vobs_of=self._vobs_for)
                rows.extend(_extra)
                n_ok += len(_extra) if status == STATUS_OK else 0
        return rows, n_ok, n_unk, n_stale, n_nodata, worst_age

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
