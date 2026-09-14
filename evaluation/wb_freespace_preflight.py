"""自由空間求解器趟次的起動前檢查：場景、TF、以及**執行期生效設定的讀回**。

三件事，任一不成立就不放行：

1. **空場景的兩個獨立事實**
   * 距離節點的 `obstacles` 參數為空 —— 由**參數服務**讀回，不是看列
   * 每個手臂連桿的 TF **可得、有限、且新鮮** ——
     距離節點在「場景無障礙物」與「該連桿缺 TF」兩種情況下產生
     **完全相同**的列，所以缺 TF 必須另外查，不能從列推論
2. **執行端場景檢查** —— E2 的 `solver_freespace` 模式會實際掃描 stage；
   本檔只確認那一關已通過（`obstacles=[]` 不等於模擬場景沒有障礙物）
3. **求解端與下游用同一份低速框** —— 讀回 `wholebody_safety` 的
   **執行期生效值**（參數服務），與本地 `lowspeed_cfg` 逐項比對；
   只在離線設定裡改成功不算
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_qp_lowspeed import lowspeed_cfg                              # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.parameter import Parameter                               # noqa: E402
from rosgraph_msgs.msg import Clock                                 # noqa: E402
from rclpy.parameter_client import AsyncParameterClient             # noqa: E402
from tf2_ros import Buffer, TransformListener                       # noqa: E402
from rclpy.duration import Duration                                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--urdf', default=os.path.join(
        HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--report-frame', default='odom')
    ap.add_argument('--dist-node', default='/arm_link_distance')
    ap.add_argument('--safety-node', default='/wholebody_safety')
    ap.add_argument('--endpoint-scene-ok', action='store_true',
                    help='E2 的 solver_freespace 場景掃描已通過')
    ap.add_argument('--tf-max-age-s', type=float, default=0.5)
    ap.add_argument('--timeout-s', type=float, default=30.0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'install',
                                    'ammr_wholebody_mpc', 'lib',
                                    'python3.12', 'site-packages'))
    from ammr_wholebody_mpc.arm_link_geometry import arm_link_names
    links = arm_link_names(open(a.urdf).read())

    rclpy.init()
    # **必須用模擬時間**：TF 的時戳是 sim time，拿牆鐘去減會得到 1.8e9 s 的
    # 假年齡（趟次 wb_solver_iso_094246 即因此把 11 個合格的 TF 全判為不新鮮）。
    n = Node('wb_freespace_preflight',
             parameter_overrides=[Parameter('use_sim_time', value=True)])
    buf = Buffer()
    TransformListener(buf, n)
    # 等 /clock 真的有值，否則 now() 會是 0
    got = []
    n.create_subscription(Clock, '/clock',
                          lambda m: got.append(m.clock.sec), 10)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20.0 and len(got) < 5:
        rclpy.spin_once(n, timeout_sec=0.05)
    if len(got) < 5:
        print('[preflight] **收不到 /clock**，不放行'); rclpy.try_shutdown(); return 1
    print(f'[preflight] 模擬時間已就緒（{got[-1]} s）；TF 新鮮度以 sim time 計算')
    out = {'schema': 'wb_freespace_preflight/1',
           'report_frame': a.report_frame, 'n_links': len(links),
           'links': links, 'endpoint_scene_ok': bool(a.endpoint_scene_ok)}
    fails = []

    # ---- 1a 距離節點的 obstacles 參數 ----
    pc = AsyncParameterClient(n, a.dist_node)
    ok = pc.wait_for_services(timeout_sec=15.0)
    if not ok:
        fails.append(f'{a.dist_node} 參數服務無回應')
        out['obstacles_configured'] = -1
    else:
        fut = pc.get_parameters(['obstacles'])
        rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
        try:
            vals = fut.result().values[0]
            obs = [s for s in (vals.string_array_value or []) if s.strip()]
        except Exception as e:                       # noqa: BLE001
            fails.append(f'讀不到 {a.dist_node} 的 obstacles：{e}')
            obs = None
        out['obstacles_configured'] = -1 if obs is None else len(obs)
        out['obstacles_value'] = obs
        if obs is None or len(obs) != 0:
            fails.append(f'obstacles 不為空：{obs}')

    # ---- 1b 逐連桿 TF：可得、有限、新鮮 ----
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    tf_rows, tf_ok = [], 0
    now = n.get_clock().now()
    for name in links:
        rec = {'link': name}
        try:
            tr = buf.lookup_transform(a.report_frame, name,
                                      rclpy.time.Time(),
                                      timeout=Duration(seconds=1.0))
            t = tr.transform.translation
            r = tr.transform.rotation
            v = [t.x, t.y, t.z, r.x, r.y, r.z, r.w]
            age = (now - rclpy.time.Time.from_msg(tr.header.stamp)).nanoseconds * 1e-9
            rec.update({'found': True,
                        'frame_id': tr.header.frame_id,
                        'child_frame_id': tr.child_frame_id,
                        'finite': bool(np.isfinite(v).all()),
                        'age_s': round(float(age), 4),
                        'xyz': [round(float(x), 6) for x in v[:3]]})
            # **身分**：父必須是 report_frame，子必須是該連桿
            ident = (tr.header.frame_id.lstrip('/') == a.report_frame
                     and tr.child_frame_id.lstrip('/') == name)
            rec['identity_ok'] = bool(ident)
            rec['fresh'] = bool(abs(age) <= a.tf_max_age_s)
            if rec['finite'] and ident and rec['fresh']:
                tf_ok += 1
            else:
                fails.append(f'{name} TF 不合格：finite={rec["finite"]} '
                             f'identity={ident} age={age:.3f}s')
        except Exception as e:                       # noqa: BLE001
            rec.update({'found': False, 'error': str(e)[:120]})
            fails.append(f'{name} TF 查不到')
        tf_rows.append(rec)
    out['tf'] = tf_rows
    out['tf_ok_links'] = tf_ok

    # ---- 2 執行端場景檢查 ----
    if not a.endpoint_scene_ok:
        fails.append('執行端場景檢查未標記通過'
                     '（obstacles=[] 不等於模擬場景沒有障礙物）')

    # ---- 3 下游安全層的**執行期生效**速度框讀回 ----
    ps = AsyncParameterClient(n, a.safety_node)
    want = lowspeed_cfg()
    exp = {'vmax_base_lin': float(want.vmax[0]),
           'vmax_base_ang': float(want.vmax[2]),
           'vmax_arm': float(want.vmax[3])}
    out['expected_vmax'] = exp
    if not ps.wait_for_services(timeout_sec=15.0):
        fails.append(f'{a.safety_node} 參數服務無回應')
        out['safety_vmax_effective'] = None
    else:
        fut = ps.get_parameters(list(exp))
        rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
        got = {}
        try:
            for k, v in zip(exp, fut.result().values):
                got[k] = float(v.double_value)
        except Exception as e:                       # noqa: BLE001
            fails.append(f'讀不到 {a.safety_node} 的速度框參數：{e}')
        out['safety_vmax_effective'] = got
        for k, want_v in exp.items():
            g = got.get(k)
            if g is None or not math.isclose(g, want_v, rel_tol=1e-9,
                                             abs_tol=1e-9):
                fails.append(f'下游 {k}={g} 與求解端 {want_v} 不一致 —— '
                             f'求解時滿足的界限會在下游被放寬回去')

    out['fails'] = fails
    out['passed'] = not fails
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'[preflight] obstacles={out.get("obstacles_configured")}、'
          f'TF {tf_ok}/{len(links)}、執行端場景 {a.endpoint_scene_ok}')
    print(f'[preflight] 下游速度框 {out.get("safety_vmax_effective")} '
          f'vs 求解端 {exp}')
    if fails:
        print('[preflight] **未通過**：')
        for f in fails:
            print(f'    - {f}')
    else:
        print('[preflight] 全部通過')
    print(f'[preflight] -> {a.out}')
    rclpy.try_shutdown()
    return 0 if not fails else 1


if __name__ == '__main__':
    raise SystemExit(main())
