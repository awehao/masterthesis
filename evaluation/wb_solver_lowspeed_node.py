"""**低速 QP 配置的求解節點**：沿用既有 `WholeBody`，只換約束集與速度框。

`evaluation/wholebody_pregrasp.py` **位元不變**。本檔以子類別覆寫兩處：

* `self.cfg` → `wb_qp_lowspeed.lowspeed_cfg()`（把 E2 執行界限放進框）
* `_constraints()` → `wb_qp_lowspeed.constraints_lowspeed()`
  （允許障礙物列為零，**但只有在空場景經確認時**；
  關節限位、速度框與既有加速度框一律組裝）

**不是 B 基線、不是 WGMPC。** B 凍結於 `baseline_B_frozen_20260909.md`。
本趟名稱：**低速 QP 配置的 Isaac 自由空間閉迴路驗證**。

空場景的確認**不看列、看設定**，且必須由起動前檢查
（`wb_freespace_preflight.py`）產出的檔案提供 ——
本檔拒絕自行推論，也拒絕在檔案不存在時啟動。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wb_qp_lowspeed import (NOT_B_BASELINE, SceneFacts, check_e2_bounds,  # noqa
                            constraints_lowspeed, lowspeed_cfg)

RUN_LABEL = '低速 QP 配置的 Isaac 自由空間閉迴路驗證'


def load_base():
    path = os.path.join(HERE, 'wholebody_pregrasp.py')
    spec = importlib.util.spec_from_file_location('wbp_base', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['wbp_base'] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', required=True,
                    help='wb_freespace_preflight.py 產出的 JSON')
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, 0.0, 0.55])
    ap.add_argument('--timeout-s', type=float, default=60.0)
    cl, rest = ap.parse_known_args()

    if not os.path.exists(cl.preflight):
        print(f'[solver] **起動前檢查檔不存在** {cl.preflight} —— 不啟動',
              file=sys.stderr)
        return 6
    pf = json.load(open(cl.preflight))
    facts = SceneFacts(obstacles_configured=pf['obstacles_configured'],
                       tf_ok_links=pf['tf_ok_links'],
                       n_links=pf['n_links'],
                       rows_total=pf.get('rows_total', -1),
                       rows_status_ok=pf.get('rows_status_ok', -1))
    if not facts.empty_scene_confirmed:
        print(f'[solver] **空場景未獲確認**：{facts.why_not()} —— 不啟動',
              file=sys.stderr)
        return 7
    if not pf.get('endpoint_scene_ok'):
        print('[solver] **執行端場景檢查未通過** —— 不啟動'
              '（距離節點沒設障礙物 ≠ 模擬場景沒有障礙物）', file=sys.stderr)
        return 7
    print(f'[solver] {RUN_LABEL}', flush=True)
    print(f'[solver] {NOT_B_BASELINE}', flush=True)
    print(f'[solver] 空場景已確認：障礙物設定 {facts.obstacles_configured}、'
          f'TF {facts.tf_ok_links}/{facts.n_links}、執行端場景檢查通過',
          flush=True)

    M = load_base()

    class LowSpeed(M.WholeBody):
        """只覆寫約束集與速度框；其餘完全沿用。"""

        def __init__(self, a):
            super().__init__(a)
            self.cfg = lowspeed_cfg(self.cfg)
            self.con_info = None
            self.n_nonfinite = 0
            self.n_e2_violation = 0
            self.e2_first_violation = None
            self.get_logger().info(
                f'低速 QP 配置 vmax base_lin={self.cfg.vmax[0]:.6f} '
                f'base_ang={self.cfg.vmax[2]:.6f} arm={self.cfg.vmax[3]:.6f}')

        def _constraints(self, q, v_lin):
            A, b, nb, info = constraints_lowspeed(
                self.K, q, v_lin, [], self.cfg, facts,
                v_prev=getattr(self, 'v_prev', None), dt=self.cfg.dt)
            self.con_info = info
            return A, b, nb

        def solve(self, T_des):
            v, T, ep, er = super().solve(T_des)
            # **求解結果仍要檢查**：先擋非有限值，再比界限與約束殘差
            ok, why = check_e2_bounds(v)
            if not np.isfinite(np.asarray(v, float)).all():
                self.n_nonfinite += 1
                raise RuntimeError(f'求解結果含非有限值：{why}')
            if not ok:
                self.n_e2_violation += 1
                if self.e2_first_violation is None:
                    self.e2_first_violation = why
                # 不裁切、不放行 —— 交由既有失效路徑處理
                raise RuntimeError(f'求解結果不符 E2 執行界限：{why}')
            return v, T, ep, er

    M.WholeBody = LowSpeed
    argv = ([sys.argv[0], '--solver', 'qp', '--out', cl.out,
             '--target', *[str(x) for x in cl.target],
             '--timeout-s', str(cl.timeout_s)] + rest)
    old = sys.argv
    sys.argv = argv
    try:
        rc = M.main()
    finally:
        sys.argv = old
    # 把本配置的標示補進輸出，避免日後被誤讀為 B 基線
    try:
        d = json.load(open(cl.out))
        d['run_label'] = RUN_LABEL
        d['config'] = 'wb_qp_lowspeed'
        d['not_b_baseline'] = NOT_B_BASELINE
        d['scene_facts'] = {'obstacles_configured': facts.obstacles_configured,
                            'tf_ok_links': facts.tf_ok_links,
                            'n_links': facts.n_links,
                            'endpoint_scene_ok': pf.get('endpoint_scene_ok')}
        d['safety_vmax_readback'] = pf.get('safety_vmax_effective')
        json.dump(d, open(cl.out, 'w'), ensure_ascii=False)
    except Exception as e:      # noqa: BLE001
        print(f'[solver] 輸出補標示失敗：{e}', file=sys.stderr)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
