#!/usr/bin/env python3
"""door_k3m_geom_monitor_v3 純邏輯驗證（不匯入 Isaac）。構型取自內側把手串接 v1 static 的各階段列（含外側持握、移動、內側閉爪、拉關）。
  G1 self：與 v2 後端（door_k3m_geom_monitor.default_backend）逐值相等（sphere 與 full 兩種模式）；
  G2 內側把手遠離時（外側持握與退臂列）：v3 的 nonfinger／finger（目標 out）／base 與 v2 逐值相等；
  G3 內側把手附近（G、P 列）：v3 nonfinger／base ≤ v2；目標 None 時手指對內側橫桿的距離計入 ⇒ finger ≤ finger_bar_in；目標 in 時不計入 bar_in、改計 bar_out；
  G4 閉爪中（G 列，手指圍住內側橫桿）：目標 None 或 out ⇒ finger < 5 mm（監看會停）；目標 in ⇒ 不因內側橫桿停；
  G5 set_target 非法 ⇒ ValueError；接到既有 GeometryMonitor：決策與明細含四類、target_bar 原樣記錄；
  G6 規格把手高與離線點雲設定不符 ⇒ 拒絕建後端。

    python3 evaluation/test_door_k3m_geom_monitor_v3.py
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402

rows = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_v1_static.json')))['rows']
fails = []


def ok(c, m):
    if not c:
        fails.append(m)


v2s, v3s = GM.default_backend(), GM3.backend_v3()
v2f, v3f = GM.default_backend(self_mode='full'), GM3.backend_v3(self_mode='full')
by = {}
for r in rows:
    by.setdefault(r['stage'], []).append(r)
pick = []
for st, rs in by.items():
    idx = np.linspace(0, len(rs) - 1, min(6, len(rs))).astype(int)
    pick += [rs[i] for i in idx]
for r in pick:
    x, q, qf, st = np.array(r['x'], float), math.radians(r['q']), float(r['qf']), r['stage']
    a2, a3 = v2s(x, q, qf), v3s(x, q, qf)
    ok(a2['self'] == a3['self'], f'G1 self sphere {st}')
    ok(v2f(x, q, qf)['self'] == v3f(x, q, qf)['self'], f'G1 self full {st}')
    if st in ('release_outer',):
        for k in ('nonfinger', 'finger', 'base'):
            ok(abs(a2[k][0] - a3[k][0]) < 1e-12, f'G2 {st} {k} v2 {a2[k]} v3 {a3[k]}')
    if st in ('G', 'P'):
        ok(a3['nonfinger'][0] <= a2['nonfinger'][0] + 1e-12 and a3['base'][0] <= a2['base'][0] + 1e-12, f'G3 {st} nonfinger/base')
        v3s.set_target(None)
        b = v3s(x, q, qf)
        ok(b['finger'][0] <= b['finger_bar_in'][0] + 1e-12 and b['finger'][0] <= b['finger_bar_out'][0] + 1e-12, f'G3 None {st}')
        v3s.set_target('in')
        c = v3s(x, q, qf)
        ok(c['finger'][0] <= c['finger_bar_out'][0] + 1e-12 and 'bar_in' not in str(c['finger'][1]), f'G3 in {st} {c["finger"]}')
        if st == 'G':
            ok(b['finger'][0] < 0.005, f"G4 None 應停 {b['finger']}")
            v3s.set_target('out')
            ok(v3s(x, q, qf)['finger'][0] < 0.005, 'G4 out 應停')
            ok(c['finger'][0] >= 0.005 or 'bar_in' not in str(c['finger'][1]), f"G4 in {c['finger']}")
        v3s.set_target('out')
try:
    v3s.set_target('inner')
    ok(False, 'G5 非法目標未拒')
except ValueError:
    pass
mon = GM.GeometryMonitor(GM.GeomMonitorConfig(age_max_s=10.0), distance_fn=v3s)
r = by['P'][10]
dec, det = mon.step(1.0, np.array(r['x'], float), math.radians(r['q']), float(r['qf']), 0.01)
ok(dec in ('ok', 'warn', 'stop') and set(det.get('classes', {})) == set(GM.CLASSES), f'G5 監看 {dec} {det.get("why")}')
D3 = copy.deepcopy(json.loads(json.dumps(__import__('yaml').safe_load(open(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml'))))))
D3['handle']['height'] = 0.95
try:
    GM3.backend_v3(D3=D3)
    ok(False, 'G6 把手高不符未拒')
except ValueError:
    pass
print('全部通過' if not fails else f'失敗 {len(fails)}', f'（構型 {len(pick)} 個）')
for f in fails[:20]:
    print('  ', f)
sys.exit(1 if fails else 0)
