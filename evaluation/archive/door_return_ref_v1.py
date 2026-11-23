#!/usr/bin/env python3
"""返程參考包 v1：A（door_return_rev_v2c 列）＋B（door_return_close_v3b 列）合併成操作段 20 Hz 表＋進場目標。
進場 R0＋S1、C0＋C1 由既有滾動進場實現，不入操作段；park／q_unfold＝S2、C2 首列。實測門角差補償由任務端當趟算（本表為 q0＝1.527° 名目）。
    python3 evaluation/door_return_ref_v1.py <ra.json> <rb.json>   # 寫 results/door/door_return_ref_v1.json（不覆寫）
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'door', 'door_return_ref_v1.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    ra, rb = (json.load(open(p)) for p in sys.argv[1:3])
    A = [r for r in ra if r['stage'][:2] in ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8')]
    B = [r for r in rb if r['stage'][:2] in ('C2', 'C3', 'C4', 'C5', 'C6', 'C7')]
    pk = {'schema': 'door_return_ref_v1', 'dt': 0.05, 'A': A, 'B': B,
          'entry': {'A': {'park': A[0]['x'][:3], 'q_unfold': A[0]['x'][3:], 'bar': 'in', 'q_nominal_deg': A[0]['q_deg']},
                    'B': {'park': B[0]['x'][:3], 'q_unfold': B[0]['x'][3:], 'bar': 'out', 'q_nominal_deg': B[0]['q_deg'], 'wp': [[0.90, -1.55], [0.55, -1.86], [0.2535, -1.8864], [0.15, -1.50]]}},
          'home': [-0.6, -2.8, 1.570796], 'source': ['door_return_rev_v2.py --slide-stretch 2.5', 'door_return_close_v3.py --c2-stretch 2.5'],
          'note': '離線候選：無力學；S4／C4 受力未驗證'}
    json.dump(pk, open(OUT, 'w'), ensure_ascii=False)
    print(f"A {len(A)} 列、B {len(B)} 列；{hashlib.sha256(open(OUT, 'rb').read()).hexdigest()[:16]}；A park {[round(v, 3) for v in A[0]['x'][:3]]}；B park {[round(v, 3) for v in B[0]['x'][:3]]}；B 末 {[round(v, 3) for v in B[-1]['x'][:3]]}")


if __name__ == '__main__':
    main()
