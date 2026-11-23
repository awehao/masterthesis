#!/usr/bin/env python3
"""區域導引在線 MotM：三個未參與設計的新位置／第一桌高度（Codex 20261102_072116／072949；事前固定種子，實跑前提交，不依結果換位置）。
本批設計域（第一批 8 案＋B1 涵蓋的範圍；只稱「本批設計域」，不稱已核可的連續區間）：
  瓶位偏移（相對第一桌原瓶心 (0, 0.42)）dx ∈ [−0.08, +0.08] m、dy ∈ [−0.025, +0.04] m；第一桌高 z1 ∈ [0.55, 0.65] m；無動態物；第二桌不變。
抽樣：numpy default_rng(SEED) 均勻抽 3 組，dx／dy 取到 0.1 mm、z1 取到 1 mm。輸出只交給場景生成（S1_ARGS 的 --bottle-dx／--bottle-dy／--table1-z）；
控制器不讀此檔、不收這些值（目標與支撐高度只來自本趟視覺）。
    python3 evaluation/bottle_online_unseen_gen.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261102
DOMAIN = dict(dx=(-0.08, 0.08), dy=(-0.025, 0.04), z1=(0.55, 0.65))
OUT = os.path.join(HERE, 'experiments', '20261102_online_unseen_positions.json')


def main():
    rng = np.random.default_rng(SEED)
    cases = []
    for k in range(3):
        dx, dy, z1 = (float(rng.uniform(*DOMAIN[n])) for n in ('dx', 'dy', 'z1'))
        cases.append(dict(id=f'U{k + 1}', bottle_dx_m=round(dx, 4), bottle_dy_m=round(dy, 4), table1_z_m=round(z1, 3), dyn_case='none',
                          run_id_proposed=f'bottle_online_u{k + 1}_1'))
    out = dict(schema=1, seed=SEED, domain=DOMAIN, domain_name='本批設計域（非已核可連續區間）', cases=cases, order=['U1', 'U2', 'U3'],
               isolation='只交給場景生成；控制器不讀、不收；精確值不預先告知控制器', generated_before_runs=True,
               video='U1 先交（手腕＋全景），再跑 U2、U3')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(cases, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
