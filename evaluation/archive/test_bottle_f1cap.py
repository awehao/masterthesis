#!/usr/bin/env python3
"""F1 視覺採集控制器的假世界核對（bottle_w1_f1cap_ctrl；假受控對象、假候選，非物理）：
  ok：有候選 ⇒ 承接、平移抓取窗，停在邊界前。  none：一直沒有候選 ⇒ 由既有 no_fix 類停止，也不越過漸變起點。
核對：實際最遠底盤 x < 該趟 x_blend；未進 INSERT、未閉爪；命令有限。    python3 evaluation/test_bottle_f1cap.py
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 2:
    res = {}
    for k in ('ok', 'none'):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=1800)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        res[k] = json.loads(line[-1]) if line else dict(error=p.stderr[-1500:])
        print(k, res[k])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'f1cap_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    sys.exit(0 if all(v.get('pass') for v in res.values()) else 1)
CASE = sys.argv[1]
sys.argv = [sys.argv[0], '0.600', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.15, 0.42
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_f1cap_ctrl as FC                       # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
TP.DX, TP.DY = 2.4, -1.0
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', 'bottle_online5_u3_1', 's1_report.json')))['plan']['joined'], float)
LOGP = os.path.join(HERE, 'results', 'bottle', f'f1cap_{CASE}_worker.log')
T = TV.T


class Ctl(FC.W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)


def main():
    scene = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    orig, orig_ml = T.W1.W1PlaceController, SC.make_lidar
    T.W1.W1PlaceController = Ctl
    SC.make_lidar = lambda **kw: orig_ml(scan_fn=lambda t, q: NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z), **kw)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, SC.make_lidar = orig, orig_ml
    c.path = NP.PlannedPath(J_RUN, -0.9)
    c.s_ref = c.sig_robot = c.path.sig0
    f.q[0], f.q[1] = J_RUN[0][0], J_RUN[0][1]
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None)
    step0 = c.step

    def step(t, q, **kw):
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if CASE == 'ok' and t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + 3.6 - 1e-9:
                xy = np.array([0.15, 0.42]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=0.60, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=40.0)
    Q = np.array([h['q'] for h in f.hist])
    ph = sorted({h['phase'] for h in f.hist})
    xmax = float(Q[:, 0].max())
    why = (r['stop'] or {}).get('why')
    ok = xmax < float(c.x_blend) and 'INSERT' not in ph and 'CLOSE' not in ph and T.finite_cmds(f) and why is not None and (CASE != 'ok' or why == 'f1_capture_boundary')
    print(json.dumps(dict(case=CASE, **{'pass': bool(ok)}, why=why, t_stop=(r['stop'] or {}).get('t'), x_max=round(xmax, 4), x_blend=round(float(c.x_blend), 4), phases=ph, cap=(r.get('f1_capture') or {}).get('cap'), look=(r.get('gaze') or {}).get('look'), vfix=None if c.vfix is None else 'set'), ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
