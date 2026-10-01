"""自由空間管線的最小通路核對（在實跑中、節點都起來之後執行）。

**為什麼需要分開核對 TF**
--------------------------
`arm_link_distance` 的產列路徑是

    if T is None or not live:
        rows.append([...STATUS_NODATA...])

所以 **TF 缺失也會產生 NODATA**，與「場景裡沒有外部障礙物」無法只憑
NODATA 區分。只看 NODATA 就宣稱空場景，等於把查不到 TF 當成淨空。

本檔因此分三件事核：
  1. **必要 TF 有效**：report_frame → 每個連桿都查得到，且以**模擬時間**判新鮮
  2. **NODATA 列真的發布**，欄位可解讀（依 ~/diag_fields 的名單）
  3. **安全層解析了那些列**（n_rows > 0）——此項**只在有命令時判**；
     尚無命令時安全層走 reason=1 閒置路徑，不會解析列

**不修改**缺資料／過期／溢位的停止規則；本檔只讀取與核對。
停止供應後仍會停止這一項，由 test_wgmpc_wg2_freespace_stop 另外以實際節點驗。
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time

import rclpy
import tf2_ros
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String

# **不重打數字**：由權威來源 import（實際值是 3.0，我一度寫死 2.0 而誤判）
import os as _os
_sys = __import__('sys')
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                 # noqa: E402
    STATUS_NODATA, STATUS_OK, STATUS_OVERFLOW, STATUS_STALE, STATUS_UNKNOWN)


class Checker(Node):
    def __init__(self, a):
        super().__init__('wgmpc_wg2_freespace_check')
        from rclpy.parameter import Parameter as P
        self.set_parameters([P('use_sim_time', P.Type.BOOL, True)])
        self.a = a
        self.buf = tf2_ros.Buffer()
        # **單一 executor 所有權**：spin_thread=False，由本檔的 executor 轉
        tf2_ros.TransformListener(self.buf, self, spin_thread=False)
        self.cloud = None
        self.cloud_t = None
        self.diag = None
        self.diag_fields = None
        self.clock_seen = 0
        _lat = QoSProfile(depth=1)
        _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 self._pts, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 self._diag, 10)
        self.create_subscription(String, '/wholebody_safety/diag_fields',
                                 self._df, _lat)
        # **自己訂閱 cmd_out**：先前用 `ros2 topic echo --once` 解析字串，
        # 取不到就誤判成「安全層沒放行」。這裡直接讀訊息。
        from std_msgs.msg import Float64MultiArray as _F64
        self.cmd_out = None
        self.create_subscription(_F64, '/wholebody_safety/cmd_out',
                                 self._cout, 10)

    def _clk(self, m):
        self.clock_seen += 1

    def _pts(self, m):
        nf = m.point_step // 4
        self.cloud = (m, nf)
        self.cloud_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def _diag(self, m):
        self.diag = list(m.data)

    def _cout(self, m):
        self.cmd_out = [float(x) for x in m.data]

    def _df(self, m):
        try:
            self.diag_fields = json.loads(m.data)
        except Exception:                                  # noqa: BLE001
            self.diag_fields = None

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--report-frame', default='odom')
    ap.add_argument('--wait-s', type=float, default=40.0)
    ap.add_argument('--max-age-s', type=float, default=0.5)
    ap.add_argument('--tf-wait-s', type=float, default=20.0,
                    help='等 TF buffer 填滿的上限（/tf_static 只發一次）')
    a = ap.parse_args()
    rclpy.init()
    nd = Checker(a)
    ex = SingleThreadedExecutor()
    ex.add_node(nd)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.wait_s:
        ex.spin_once(timeout_sec=0.05)
        if nd.cloud is not None and nd.diag is not None and nd.clock_seen > 5:
            break
    rep = {'checks': [], 'ok': True}

    def ck(name, cond, detail=''):
        rep['checks'].append({'name': name, 'ok': bool(cond),
                              'detail': str(detail)})
        rep['ok'] = rep['ok'] and bool(cond)
        print(f'  {name:52s} {"ok" if cond else "**錯**"}  {detail}',
              flush=True)

    ck('/clock 有收到', nd.clock_seen > 5, f'{nd.clock_seen} 則')
    ck('距離雲有收到', nd.cloud is not None)

    # ---------- 1 必要 TF 有效（**與 NODATA 分開**）----------
    import numpy as np
    links, ages, missing = [], [], []
    if nd.cloud is not None:
        m, nf = nd.cloud
        R = np.frombuffer(m.data, dtype=np.float32).reshape(m.width, nf)
        li = sorted({int(x) for x in R[:, 10]})
        rep['cloud'] = {'width': int(m.width), 'nf': int(nf),
                        'link_indices': li}
        # 連桿名稱由安全層的 obstacle_names 無法取得 ⇒ 改用 TF 可查性逐 frame 核
    # 直接核安全層會查的那條鏈，以及每個連桿 frame
    from ammr_wholebody_mpc.arm_link_geometry import arm_link_names
    import os
    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'models', 'omni_bot_wholebody_expanded.urdf')
    names = arm_link_names(open(urdf, encoding='utf-8').read())
    # **TF buffer 需要時間填滿**：/tf_static 只發一次、動態 TF 50 Hz。
    # 先前在雲一到就立刻查，buffer 還稀疏 ⇒ 全部 LookupException（假警報）。
    # 這裡先轉到查得到為止（有上限），再逐個核新鮮度。
    _t1 = time.monotonic()
    while time.monotonic() - _t1 < a.tf_wait_s:
        ex.spin_once(timeout_sec=0.02)
        try:
            nd.buf.lookup_transform(a.report_frame, names[-1],
                                    rclpy.time.Time())
            break
        except Exception:                                  # noqa: BLE001
            continue
    now = nd.sim_now()
    for nm in names:
        try:
            tr = nd.buf.lookup_transform(a.report_frame, nm,
                                         rclpy.time.Time())
            st = (tr.header.stamp.sec
                  + tr.header.stamp.nanosec * 1e-9)
            links.append(nm)
            ages.append(now - st)
        except Exception as e:                             # noqa: BLE001
            missing.append(f'{nm}: {type(e).__name__}')
    ck(f'**必要 TF 全部查得到**（{a.report_frame} → 各連桿）',
       not missing, f'缺 {len(missing)} 個：{missing[:3]}' if missing
       else f'{len(links)} 個')
    if missing:
        try:
            tree = nd.buf.all_frames_as_string()
        except Exception as e:                             # noqa: BLE001
            tree = f'（取不到 frame 樹：{e}）'
        print('    **實際的 TF frame 樹**：', flush=True)
        for ln in str(tree).strip().split('\n')[:14]:
            print('      ' + ln, flush=True)
        rep['tf_tree'] = str(tree)[:4000]
    if ages:
        ck('TF 以**模擬時間**判定為新鮮',
           max(ages) <= a.max_age_s,
           f'最舊 {max(ages):.3f} s（上限 {a.max_age_s}）')
    rep['tf'] = {'ok_links': links, 'missing': missing,
                 'max_age_s': (max(ages) if ages else None)}

    # ---------- 2 NODATA 列真的發布且可解讀 ----------
    if nd.cloud is not None:
        m, nf = nd.cloud
        R = np.frombuffer(m.data, dtype=np.float32).reshape(m.width, nf)
        import collections as _c
        dist_st = _c.Counter(float(x) for x in R[:, 7])
        NAME = {float(STATUS_OK): 'OK', float(STATUS_UNKNOWN): 'UNKNOWN',
                float(STATUS_STALE): 'STALE',
                float(STATUS_NODATA): 'NODATA',
                float(STATUS_OVERFLOW): 'OVERFLOW'}
        print('    狀態分布：'
              + ', '.join(f'{NAME.get(k, k)}={v}' for k, v in
                          sorted(dist_st.items())), flush=True)
        rep['status_dist'] = {NAME.get(k, str(k)): v
                              for k, v in dist_st.items()}
        n_nodata = int((R[:, 7] == float(STATUS_NODATA)).sum())
        n_ok = int((R[:, 7] == float(STATUS_OK)).sum())
        ck('距離節點**有發布列**', m.width > 0, f'{m.width} 列')
        ck('**外部障礙物為空 ⇒ 列應為 NODATA**',
           n_nodata == m.width and n_ok == 0,
           f'NODATA {n_nodata}／OK {n_ok}／共 {m.width}')
        ck('雲的時間戳以模擬時間判定為新鮮',
           nd.cloud_t is not None and (now - nd.cloud_t) <= a.max_age_s,
           f'{now - nd.cloud_t:.3f} s' if nd.cloud_t else '')
        rep['cloud'].update(n_nodata=n_nodata, n_ok=n_ok,
                            age_s=(now - nd.cloud_t) if nd.cloud_t else None)
        # **關鍵區分**：TF 齊全 ＋ 全列 NODATA ⇒ 可歸因於「外部障礙物為空」
        ck('**NODATA 可歸因於空場景（而非 TF 缺失）**',
           (not missing) and n_nodata == m.width,
           'TF 齊全且全列 NODATA' if not missing else 'TF 有缺 ⇒ 不可歸因')

    # ---------- 3 安全層解析了那些列 ----------
    if nd.diag is not None and nd.diag_fields:
        f = nd.diag_fields if isinstance(nd.diag_fields, list) else \
            nd.diag_fields.get('safety', [])
        def g(k):
            return (nd.diag[f.index(k)] if k in f and f.index(k) < len(nd.diag)
                    else None)
        rep['safety_diag'] = {k: g(k) for k in
                              ('n_rows', 'n_active', 'n_nodata', 'reason',
                               'min_d', 'speed_cap', 'fallback', 'unresolved')}
        ck('安全層 diag 以**名稱**索引（取到 n_rows）',
           g('n_rows') is not None, f"n_rows={g('n_rows')}")
        # **分階段判定**：`reason = 1.0` 是「從未收到命令」，安全層在
        # `_points()` 之前就短路（wholebody_safety_node.py:450），`res` 保持
        # 空值 ⇒ 閒置時 n_rows 結構上不可能 > 0。因此只在**有命令**時要求
        # 已解析列；閒置時改為確認走的是閒置路徑、而不是缺資料停止。
        _rsn, _nr = g('reason'), (g('n_rows') or 0)
        if _rsn == 1.0:
            ck('安全層處於**閒置**路徑（尚無命令 ⇒ n_rows=0 為預期）',
               _nr == 0, f"reason=1.0 n_rows={_nr}"
                         "　（這一項不證明列已被解析，需有命令時才判）")
        else:
            ck('安全層**解析了列**（n_rows > 0）', _nr > 0,
               f"reason={_rsn} n_rows={_nr} n_nodata={g('n_nodata')}")
        ck('安全層**未因缺資料停止**（reason != 5）', g('reason') != 5.0,
           f"reason={g('reason')}")
    else:
        ck('安全層 diag 與欄位名單都收到', False,
           f'diag={nd.diag is not None} fields={nd.diag_fields is not None}')

    if nd.cmd_out is not None:
        _nz = any(abs(v) > 1e-9 for v in nd.cmd_out)
        rep['cmd_out'] = {'v': [round(v, 8) for v in nd.cmd_out],
                          'nonzero': _nz}
        print(f'    cmd_out = {[round(v, 5) for v in nd.cmd_out[:3]]}… '
              f'（非零 {_nz}）', flush=True)
    else:
        rep['cmd_out'] = None

    print(f'\n自由空間通路核對：{"全部通過" if rep["ok"] else "**有項目未通過**"}',
          flush=True)
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
    ex.shutdown()
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0 if rep['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
