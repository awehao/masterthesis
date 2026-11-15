#!/usr/bin/env python3
"""手臂摩擦探針三案（0／0.2／1.0 N·m）的離線比較（以零摩擦案為基線）。只讀紀錄。
    python3 evaluation/arm_fric_probe_offline.py OUT.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
KP, KD, STILL = 600.0, 60.0, 2e-3
O = {}
for tag, val in (('000', 0.0), ('020', 0.2), ('100', 1.0)):
    D = os.path.join(HERE, 'runs', f'm8house_arm_fric_probe_{tag}')
    Z = np.load(os.path.join(D, 'probe_steps.npz'), allow_pickle=True)
    P, c = Z['P'], [str(x) for x in Z['cols']]
    R = json.load(open(os.path.join(D, 'result.json')))
    i = c.index
    t = P[:, 0]
    tg, q, qd, eff, G = P[:, i('target_1'):i('target_6') + 1], P[:, i('q_1'):i('q_6') + 1], P[:, i('qd_1'):i('qd_6') + 1], P[:, i('meas_effort_1'):i('meas_effort_6') + 1], P[:, i('G_model_1'):i('G_model_6') + 1]
    r = KP * (tg - q) - KD * qd + G
    Tm, H = 1.2, 3.0

    def tail(t1):
        m = (t > t1 - 0.3) & (t <= t1 + 1e-9)
        return dict(settled=bool(np.abs(qd[m]).max() < STILL), qd_abs_max=round(float(np.abs(qd[m]).max()), 6), qd_last=[round(float(x), 5) for x in qd[m][-1]],
                    target_minus_q_mrad=[round(float(x) * 1e3, 3) for x in (tg - q)[m][-1]], r_Nm=[round(float(x), 3) for x in r[m][-1]], meas_effort_Nm=[round(float(x), 3) for x in eff[m].mean(axis=0)],
                    meas_plus_G_Nm=[round(float(x), 3) for x in (eff[m].mean(axis=0) + G[m].mean(axis=0))])
    cr = (t > 0.25) & (t <= 0.95)
    cb = (t > Tm + H + 0.25) & (t <= Tm + H + 0.95)
    O[tag] = dict(fric_value=val, backend=R['arm_backend'].get('friction_static_dynamic_viscous'), legacy=R['arm_backend'].get('legacy_friction_coefficients'), hold_check_before=R['settle']['hold_check'],
                  cruise_fwd=dict(lag4_mrad=round(float((tg - q)[cr, 3].mean()) * 1e3, 3), qd4=round(float(qd[cr, 3].mean()), 5), meas_effort4=round(float(eff[cr, 3].mean()), 3), r4=round(float(r[cr, 3].mean()), 3)),
                  cruise_back=dict(lag4_mrad=round(float((tg - q)[cb, 3].mean()) * 1e3, 3), qd4=round(float(qd[cb, 3].mean()), 5), meas_effort4=round(float(eff[cb, 3].mean()), 3), r4=round(float(r[cb, 3].mean()), 3)),
                  after_fwd=tail(Tm + H), after_back=tail(2 * (Tm + H)), contact_max_n=float(np.nanmax(P[:, i('contact_max_n')])), n_err=R['probe']['n_err'],
                  other_joint_drift_mrad=[round(float(x) * 1e3, 3) for x in (q[-1] - q[0])])
b = O['000']
for tag in ('020', '100'):
    O[tag]['vs_zero'] = dict(d_lag4_fwd_mrad=round(O[tag]['cruise_fwd']['lag4_mrad'] - b['cruise_fwd']['lag4_mrad'], 3), d_lag4_back_mrad=round(O[tag]['cruise_back']['lag4_mrad'] - b['cruise_back']['lag4_mrad'], 3),
                             d_effort4_fwd=round(O[tag]['cruise_fwd']['meas_effort4'] - b['cruise_fwd']['meas_effort4'], 3), d_effort4_back=round(O[tag]['cruise_back']['meas_effort4'] - b['cruise_back']['meas_effort4'], 3),
                             d_r_after_fwd=[round(x - y, 3) for x, y in zip(O[tag]['after_fwd']['r_Nm'], b['after_fwd']['r_Nm'])], d_r_after_back=[round(x - y, 3) for x, y in zip(O[tag]['after_back']['r_Nm'], b['after_back']['r_Nm'])])
json.dump(O, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
for k, v in O.items():
    print(k, 'backend', v['backend'][3], 'legacy', v['legacy'][3], '| hold r', v['hold_check_before']['r_Nm'])
    print('   cruise fwd', v['cruise_fwd'], 'back', v['cruise_back'])
    print('   after_fwd', v['after_fwd']['settled'], v['after_fwd']['qd_abs_max'], 'r', v['after_fwd']['r_Nm'], 'meas+G', v['after_fwd']['meas_plus_G_Nm'])
    print('   after_back', v['after_back']['settled'], v['after_back']['qd_abs_max'], 'r', v['after_back']['r_Nm'], 'qd_last', v['after_back']['qd_last'])
    print('   vs_zero', v.get('vs_zero'), 'contact', v['contact_max_n'])
