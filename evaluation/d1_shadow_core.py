#!/usr/bin/env python3
"""D1 S4 線上 shadow 的純邏輯（不依賴 ROS／Isaac，可單元測試）。

依 Codex reviews/20261005_144036_reply.md：整條資料路徑都有界、輸入契約明確、以影格識別對帳。

* PoseHistory  ：模擬器端每物理步記（world.step() 之後的實際時間、相機 optical 世界位姿、把手真值）；有界；
                 新影格以 rendering_time 找**唯一**匹配步（容差 0.25·dt），找不到就拒絕（不以讀取當下補戳）。
* PairBuffer   ：節點端把同一擷取戳的 depth／camera_info／pose／meta 四段配成一組；容量與逾時有界，淘汰附原因。
* LatestSlot   ：待處理槽最多一張；處理中來新影格 ⇒ 覆蓋舊的待處理影格並記錄被覆蓋的影格識別。
* check_contract：深度 32FC1 公尺（非有限或 ≤ 0 視為無效）、尺寸與內參一致、depth／info 的 frame_id＝相機 optical、
                 pose.header.frame_id＝世界參考框（odom），位姿描述相機 optical frame。

輸入契約（S4 事前固定）
  /wrist/aligned_depth_to_color/image_raw  sensor_msgs/Image  encoding 32FC1、單位 m、光軸深度（distance_to_image_plane）、
                                           frame_id camera_color_optical_frame、640×480、little-endian
  /wrist/color/camera_info                 frame_id 同上；K 為讀回內參
  /wrist/pose                              geometry_msgs/PoseStamped；header.frame_id = odom；pose = optical frame 在 odom 的位姿
  /wrist/capture_meta                      std_msgs/String（JSON）：n（影格序號）、stamp、rendering_frame、source_wall_t
  四者 header 戳（或 meta.stamp）＝擷取時刻（rendering_time），完全相同才配對。
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque

import numpy as np

OPT = 'camera_color_optical_frame'
WORLD = 'odom'
PARTS = ('depth', 'info', 'pose', 'meta')


# ------------------------------------------------------------------ 模擬器端：姿態歷史
class PoseHistory:
    def __init__(self, dt, horizon_s=1.0):
        self.dt = float(dt)
        self.buf = deque(maxlen=max(4, int(round(horizon_s / self.dt))))

    def add(self, t_after, pos, quat_wxyz, truth=None, step=None):
        self.buf.append({'t': float(t_after), 'pos': np.asarray(pos, float).copy(),
                         'quat': np.asarray(quat_wxyz, float).copy(), 'truth': truth, 'step': step})

    def match(self, t_render):
        """回傳 (entry, None) 或 (None, 拒絕原因)。唯一匹配：容差內只能有一步。"""
        if t_render is None or not math.isfinite(float(t_render)):
            return None, 'no_rendering_time'
        tol = 0.25 * self.dt
        hits = [e for e in self.buf if abs(e['t'] - float(t_render)) <= tol]
        if not hits:
            if self.buf and float(t_render) < self.buf[0]['t'] - tol:
                return None, 'older_than_history'
            return None, 'no_pose_at_render_time'
        if len(hits) > 1:
            return None, 'ambiguous_pose_match'
        return hits[0], None


def norm_rendering_frame(rfd):
    """Isaac 的 rendering_frame 正規化成整數串列（不可解析 ⇒ None）。

    V0 重播模式（每步 render）回傳 dict（referenceTimeNumerator／Denominator）；主迴圈 render-on-demand 時
    實測回傳**整數**（d1s4_M 2026-10-05 因 list(int) 崩潰）。也接受 tuple／list。全零視為「沒有影格」⇒ None。
    """
    if rfd is None:
        return None
    if isinstance(rfd, dict):
        v = [rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')]
    elif isinstance(rfd, (list, tuple)):
        v = list(rfd)
    else:
        v = [rfd]
    try:
        v = [int(x) for x in v]
    except (TypeError, ValueError):
        return None
    if not v or all(x == 0 for x in v):
        return None
    return v


# ------------------------------------------------------------------ 節點端：配對
def stamp_key(sec, nanosec):
    return (int(sec), int(nanosec))


class PairBuffer:
    """同戳四段配對。容量 cap 組、逾時 timeout_s（牆鐘，自第一段到達起算）；淘汰回傳 (key, 原因, 已到段)。"""

    def __init__(self, cap=3, timeout_s=1.0):
        self.cap, self.timeout = int(cap), float(timeout_s)
        self.d = OrderedDict()

    def put(self, key, part, msg, now):
        evicted = self.expire(now)
        if key not in self.d:
            while len(self.d) >= self.cap:
                k, e = self.d.popitem(last=False)
                evicted.append((k, 'pair_overflow', sorted(e['parts'])))
            self.d[key] = {'first': now, 'parts': {}}
        e = self.d[key]
        if part in e['parts']:
            evicted.append((key, f'duplicate_{part}', sorted(e['parts'])))
        e['parts'][part] = msg
        done = None
        if all(p in e['parts'] for p in PARTS):
            done = self.d.pop(key)['parts']
        return done, evicted

    def expire(self, now):
        out = []
        for k in list(self.d):
            if now - self.d[k]['first'] > self.timeout:
                out.append((k, 'pair_timeout', sorted(self.d.pop(k)['parts'])))
        return out

    def drain(self):
        out = [(k, 'shutdown_unpaired', sorted(e['parts'])) for k, e in self.d.items()]
        self.d.clear()
        return out


# ------------------------------------------------------------------ 節點端：待處理槽（最多一張）
class LatestSlot:
    def __init__(self):
        self.lock = threading.Lock()
        self.ev = threading.Event()
        self.item = None

    def put(self, item):
        """放入；若已有待處理影格則被覆蓋，回傳被覆蓋的 item（否則 None）。"""
        with self.lock:
            old, self.item = self.item, item
            self.ev.set()
            return old

    def take(self, timeout=None):
        if not self.ev.wait(timeout):
            return None
        with self.lock:
            it, self.item = self.item, None
            self.ev.clear()
            return it

    def drain(self):
        with self.lock:
            it, self.item = self.item, None
            self.ev.clear()
            return it


# ------------------------------------------------------------------ 輸入契約
def quat_to_T(pos, quat_wxyz):
    """與 d1_handle_detect.cam_T 相同的建構（wxyz_to_R、不正規化），使線上輸入與 S3 逐位元同語意；
    四元數長度偏離 1 超過 1e-6 視為無效位姿（拒絕，不默默正規化）。"""
    from wrist_v0_capture import wxyz_to_R
    q = [float(v) for v in quat_wxyz]
    n = math.sqrt(sum(v * v for v in q))
    if not (math.isfinite(n) and abs(n - 1.0) <= 1e-6):
        raise ValueError('quaternion 無效（長度偏離 1）')
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(q)
    T[:3, 3] = [float(v) for v in pos]
    return T


def check_contract(depth, info, pose, expect_wh=(640, 480)):
    """depth／info／pose 是簡化的 dict（由節點從 ROS 訊息轉出；測試可直接建）。

    depth：{'encoding','frame_id','width','height','is_bigendian','step','data'(bytes)}
    info ：{'frame_id','width','height','k'(9)}
    pose ：{'frame_id','pos'(3),'quat_wxyz'(4)}
    回傳 (dep[H,W] float32 m, K 3×3, T 4×4, None) 或 (None, None, None, 原因)。
    """
    W, H = expect_wh
    if depth['encoding'] != '32FC1':
        return None, None, None, f"depth_encoding_{depth['encoding']}"
    if depth['frame_id'] != OPT or info['frame_id'] != OPT:
        return None, None, None, 'optical_frame_id_mismatch'
    if pose['frame_id'] != WORLD:
        return None, None, None, f"pose_frame_{pose['frame_id']}"
    if (depth['width'], depth['height']) != (W, H) or (info['width'], info['height']) != (W, H):
        return None, None, None, 'size_mismatch'
    if depth['is_bigendian']:
        return None, None, None, 'bigendian'
    if depth['step'] != W * 4 or len(depth['data']) != W * H * 4:
        return None, None, None, 'step_or_length_mismatch'
    dep = np.frombuffer(bytes(depth['data']), dtype='<f4').reshape(H, W)
    try:
        kk = np.asarray(info['k'], float).reshape(-1)
    except (TypeError, ValueError):
        return None, None, None, 'bad_intrinsics'
    if kk.size != 9 or not np.isfinite(kk).all():
        return None, None, None, 'bad_intrinsics'
    K = kk.reshape(3, 3)
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        return None, None, None, 'bad_intrinsics'
    try:
        T = quat_to_T(pose['pos'], pose['quat_wxyz'])
    except ValueError:
        return None, None, None, 'bad_pose'
    if not np.isfinite(T).all():
        return None, None, None, 'bad_pose'
    return dep, K, T, None
