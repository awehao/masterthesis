"""距離節點 `_tick` 的耗時拆解（離線，不開 Isaac、不改行為）。

**不預設瓶頸**：TF／位姿、幾何變換、距離計算、G2 下界、必要配對列、去重、
訊息組裝逐段量。用既有趟次的實際位姿與完整障礙物集合，列數與線上相同。
"""
from __future__ import annotations
import os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                        # noqa
    expand_pair_rows, forced_pair_rows, pair_distance_bound, parse_obstacles,
    STATUS_OK, VOBS_STATIC)
from ammr_wholebody_mpc.arm_link_geometry import (                        # noqa
    arm_link_names, link_collision_tris, obstacle_distance_matrix,
    sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from coman_zero_cmd_feasible import hold_state                             # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
PAIR_ROWS = ['link4:*', 'link5:*', 'link6:*', 'uflite_finger1:*',
             'uflite_finger2:*', 'uflite_gripper_link:*']
EXEMPT = ['uflite_finger1:handle_bar', 'uflite_finger2:handle_bar']
TIGHT = [('uflite_gripper_link', 'handle_bar')]


class T:
    def __init__(self):
        self.d = {}

    def __call__(self, k):
        self.k = k
        return self

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *a):
        self.d[self.k] = self.d.get(self.k, 0.0) + (time.perf_counter()
                                                    - self.t0) * 1e3


def main() -> int:
    st = hold_state()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    xml = open(URDF, encoding='utf-8').read()
    q = np.zeros(len(K.dof_names))
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']
    sp = subprocess.run([sys.executable, os.path.join(HERE,
                         'coman_obstacle_specs.py'), str(st['unit'][0]),
                         str(st['unit'][1])], capture_output=True, text=True,
                        check=True)
    obs = parse_obstacles([x for x in sp.stdout.split() if x.strip()])
    T_db = np.eye(4)
    T_db[0, 3] = st['unit'][0]
    T_db[1, 3] = st['drawer_y0'] - st['opening']
    for o in obs:
        if o.model == 'drawer_body':
            o.T_world_link = T_db
    names = [o.name for o in obs]
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    links = arm_link_names(xml)
    pairs, _ = expand_pair_rows(PAIR_ROWS, EXEMPT, names)
    tris = link_collision_tris(xml, links=[l for l, _ in TIGHT])

    N = 10
    t = T()
    n_rows = n_dup = 0
    for _ in range(N):
        rows = []
        seen_all = set()
        for li, nm in enumerate(links):
            with t('1_fk_tf'):
                Tw = K.fk(q, nm)
            Sx = S[nm]
            with t('2_transform'):
                W = (Sx.points @ Tw[:3, :3].T) + Tw[:3, 3]
            with t('3_distance_matrix'):
                D, V = obstacle_distance_matrix(W, obs)
                jm = np.argmin(D, axis=1)
                ixr = np.arange(len(W))
                dd, vv = D[ixr, jm], V[ixr, jm]
                wh = [names[k] for k in jm]
            with t('4_band_select'):
                dmin = float(dd.min())
                sel = np.nonzero(dd <= dmin + Sx.rho)[0]
                sel = sel[np.argsort(dd[sel])]
            lb = None
            if (nm, 'handle_bar') in TIGHT:
                with t('5_g2_bound'):
                    tw = (tris[nm].reshape(-1, 3) @ Tw[:3, :3].T
                          + Tw[:3, 3]).reshape(-1, 3, 3)
                    lb = pair_distance_bound(tw, obs[names.index('handle_bar')],
                                             tol=5e-4, max_tris=60000,
                                             budget_s=0.020)['lb']
            with t('6_band_rows'):
                seen = set()
                for k in sel:
                    nh = vv[k] / max(abs(float(dd[k])), 1e-9)
                    rows.append(list(W[k]) + list(nh)
                                + [float(dd[k]), float(STATUS_OK), 0.0, 0.0,
                                   float(li)] + list(Sx.points[k])
                                + [float(Sx.rho), float(names.index(wh[k])),
                                   0.0, 0.0, 0.0, float(VOBS_STATIC), 0.0, 0.0])
                    seen.add((nm, wh[k], tuple(np.round(Sx.points[k], 12))))
            for obn in pairs.get(nm, []):
                oi = names.index(obn)
                with t('7_forced_rows'):
                    ex = forced_pair_rows(W, Sx.points, obs[oi], oi, li, Sx.rho,
                                          STATUS_OK, 0.0, lambda p, v: 0.0, 3.0,
                                          0, d_col=D[:, oi], v_col=V[:, oi],
                                          d_lb=lb if obn == 'handle_bar' else None)
                with t('8_dedup'):
                    kept = []
                    for r in ex:
                        key = (nm, obn, tuple(np.round(np.array(r[11:14]), 12)))
                        if key in seen:
                            n_dup += 1
                            continue
                        seen.add(key)
                        kept.append(r)
                rows.extend(kept)
        with t('9_msg_assembly'):
            arr = np.asarray(rows, dtype=np.float32)
            _blob = arr.tobytes()
        n_rows = len(rows)

    tot = sum(t.d.values())
    print(f'列數 {n_rows}（去重丟棄 {n_dup // N}）；{N} 次平均\n')
    print(f'{"段":26s}{"ms/週期":>10s}{"占比":>8s}')
    print('-' * 46)
    for k in sorted(t.d, key=lambda x: -t.d[x]):
        ms = t.d[k] / N
        print(f'{k:26s}{ms:10.2f}{100*t.d[k]/tot:7.1f}%')
    print('-' * 46)
    print(f'{"合計":26s}{tot/N:10.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
