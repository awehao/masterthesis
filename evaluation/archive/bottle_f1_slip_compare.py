#!/usr/bin/env python3
"""F1 各輪抓持滑移時間線比較（只讀紀錄、真值只供事後分析）：以抓持成立時的瓶對 TCP 位姿為基準，量各時刻滑移、滑移向量（TCP 系與世界系）、停到放置位後的滑移速率、指力中位數。不判定成因。
    python3 evaluation/bottle_f1_slip_compare.py RUN_ID [RUN_ID...]
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = sys.argv[1:]
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402,F401  先載入：它會設好 ammr_wholebody_mpc 的路徑
import bottle_s1_geom as G                              # noqa: E402


def one(col, tr):
    t = col['t']
    def rel(j):
        q = np.array([col[n][j] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        Tt = G.C.K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, 3] = [col['bottle_base_x'][j], col['bottle_base_y'][j], col['bottle_base_z'][j]]
        Tb[:3, :3] = Rot.from_quat([col['bottle_qx'][j], col['bottle_qy'][j], col['bottle_qz'][j], col['bottle_qw'][j]]).as_matrix()
        return Tt, np.linalg.inv(Tt) @ Tb
    at = lambda x: int(np.argmin(np.abs(t - x)))
    _, R0 = rel(at(tr['grasp']))
    d_at = lambda x: round(float(np.linalg.norm(rel(at(x))[1][:3, 3] - R0[:3, 3]) * 1e3), 2)
    Te, Re = rel(at(tr['release'] - 0.02))
    _, Rp = rel(at(tr['parked']))
    dt = float(t[at(tr['release'] - 0.02)] - t[at(tr['parked'])])
    med = lambda n, a, b: round(float(np.median(col[n][(t >= a) & (t <= b)])), 2)
    return dict(T_rel_mm=np.round(R0[:3, 3] * 1e3, 2).tolist(), slip_mm={n: d_at(x) for n, x in (('lift_ok', tr['lift_ok']), ('lift_ok+3', tr['lift_ok'] + 3), ('lift_ok+6', tr['lift_ok'] + 6), ('parked-2', tr['parked'] - 2), ('parked', tr['parked']), ('release', tr['release'] - 0.02))},
                vec_tcp_mm=np.round((Re[:3, 3] - R0[:3, 3]) * 1e3, 2).tolist(), vec_world_mm=np.round(Te[:3, :3] @ (Re[:3, 3] - R0[:3, 3]) * 1e3, 2).tolist(), rate_after_parked_mm_s=round(float(np.linalg.norm(Re[:3, 3] - Rp[:3, 3]) * 1e3 / dt), 3),
                finger_N=dict(lift=[med('f_finger1_bottle', tr['lift_start'], tr['lift_ok']), med('f_finger2_bottle', tr['lift_start'], tr['lift_ok'])], carry=[med('f_finger1_bottle', tr['lift_ok'], tr['parked']), med('f_finger2_bottle', tr['lift_ok'], tr['parked'])], parked=[med('f_finger1_bottle', tr['parked'], tr['release']), med('f_finger2_bottle', tr['parked'], tr['release'])]))


out = {}
for run in RUNS:
    o = subprocess.run([sys.executable, os.path.join(HERE, 'bottle_f1_run_report.py'), run], capture_output=True, text=True).stdout
    rep = json.loads(o[o.index('{'):])
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'), allow_pickle=True)
    col = {str(n): Z['rows'][:, i] for i, n in enumerate(Z['cols'])}
    out[run] = [dict(round=r['round'], t=r['t'], slip_max_mm=round(r['slip_max_mm_deg'][0], 2), slip_warn_t_first=(r['slip_warn'] or {}).get('t_first'), **one(col, r['t'])) for r in rep['rounds']]
print(json.dumps(out, ensure_ascii=False, default=float))
