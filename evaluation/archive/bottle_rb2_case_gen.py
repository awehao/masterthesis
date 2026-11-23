#!/usr/bin/env python3
"""R-B2 新區域案例（Codex 20261102_174123；事前固定、實跑前提交，不依結果換位置或種子）。
區域偏移 O＝(+0.80, −0.80) m、旋轉 0；機器人世界起點 (−3.6, 0)、第二桌 (2.4, −1.0)、原障礙方塊 (−2.0, 0) 不變；無移動物。
瓶位與第一桌高：numpy default_rng(20261102174123) 依序均勻產生 區域 x∈[−0.08, +0.08]、y∈[0.395, 0.460]、桌高∈[0.55, 0.65]（取到 0.1 mm／1 mm）。
產生值只交場景生成與安全評估（S1_ARGS 的 --bottle-dx／--bottle-dy 相對原瓶心 (0, 0.42)、--table1-z）；控制器只取得區域先驗與本趟視覺量測。
    python3 evaluation/bottle_rb2_case_gen.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261102174123
O = (0.80, -0.80)
OUT = os.path.join(HERE, 'experiments', '20261102_rb2_case.json')


def main():
    rng = np.random.default_rng(SEED)
    x = round(float(rng.uniform(-0.08, 0.08)), 4)
    y = round(float(rng.uniform(0.395, 0.460)), 4)
    z1 = round(float(rng.uniform(0.55, 0.65)), 3)
    dx, dy = round(x - 0.0, 4), round(y - 0.42, 4)
    out = dict(schema=1, ruling='reviews/20261102_174123_reply.md', seed=SEED, region_offset=list(O), rotation=0.0, bottle_region=[x, y], table1_z=z1,
               bottle_world=[round(x + O[0], 4), round(y + O[1], 4)], s1_args=f'--dyn-case none --table1-z {z1} --bottle-dx {dx} --bottle-dy {dy} --region-offset {O[0]},{O[1]}',
               map_args='--table2 2.4,-1.0', unchanged=['機器人世界起點 (−3.6, 0)', '第二桌 (2.4, −1.0)', '原障礙方塊 (−2.0, 0)', '無移動物'], run_id='bottle_rb2_gaze_1',
               isolation='產生值只交場景與安全評估；控制器不讀此檔', generated_before_runs=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
