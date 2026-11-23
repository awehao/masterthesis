#!/usr/bin/env python3
"""由 door_k3m_offline_replay_r2.json（free 候選）建走廊參考表 results/door/door_k3m_corridor_table.json（已存在則拒絕覆寫）。
來源列：free 重播每個控制週期的推得門角與底盤相對錨點的前進／側向／偏航；同門角取平均（door_k3m_corridor.Corridor 規則）。

    python3 evaluation/door_k3m_corridor_build.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_corridor as CO                         # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_corridor_table.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    rp = json.load(open(SRC))
    rows = next(c for c in rp['configs'] if c['name'] == 'free')['rows']
    C = CO.Corridor([{'q_door_deg': r['q_door_deg'], 'fwd_m': r['base_fwd_m'], 'lat_m': r['base_lat_m'], 'dyaw_deg': r['base_dyaw_deg']}
                     for r in rows], rp['anchor'])
    res = {'schema': 'door_k3m_corridor_table/1', 'source': os.path.relpath(SRC, HERE),
           'source_sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest(), 'n_source_rows': len(rows), 'n_table': len(C.q),
           'q_range_deg': [float(C.q[0]), float(C.q[-1])], 'table': C.to_json(),
           'note': '離線運動學候選（無接觸）；只作監看參考，不作開迴路命令。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"表 {res['n_table']} 列、門角 {res['q_range_deg']}°")


if __name__ == '__main__':
    main()
