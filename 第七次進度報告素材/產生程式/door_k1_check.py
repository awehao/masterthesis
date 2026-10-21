#!/usr/bin/env python3
"""K1 判定器：只讀 door_k1_sim.py 的輸出 door_k1.json，依**事前登錄**的條件判定（數值不依結果調整）。

登錄條件（角度單位度；±1° 只是本次限位診斷容差，不是後續操作精度門檻）：
  讀回    USD 讀回 15 項（同 door_k0_readback.checks）＋ PhysX view 質量與規格相對差 ≤ 1e-4。
  方向 D  S2 施力期間每步開門力矩 > 0；施力結束時 q − q_init ≥ 0.5°。⇒ verified／fail
  下限 L  S3：q_min ≥ −1° 且施力＋觀察期間門扇對兩側牆接觸力皆 ≤ 0.1 N ⇒ verified；
          q_min ≥ −1° 但有牆接觸 ⇒ not_separable（無法單憑 q 宣稱關節限位生效）；q_min < −1° ⇒ fail。
  上限 U  S4：曾到 ≥ 上限 − 1° 時，q_max ≤ 上限 + 1° ⇒ verified，超過 ⇒ fail；施力時間內未到 ⇒ unverified（不加力、不延長）。
  被動 P  S1：10 s 內 |q| 最大 ≤ 0.5°。S5（初始化 45°、無外力）：全程 |q − q_init| ≤ 0.5°，且門角介於 [1°, 上限 − 1°]（離開兩限位）。
          S2 撤力後：|q̇| < 0.01 rad/s 之後到結束的門角變化 ≤ 0.5°（若從未低於 0.01 rad/s ⇒ 此子項 unverified）。
          三子項皆 verified ⇒ verified；任一 fail ⇒ fail；否則 unverified。
  阻尼    只報：S2 撤力後 q̇ 對 exp(−c·t) 的擬合（c＝規格角阻尼）；有牆接觸、到達限位或睡眠設定失敗時標記「不可解讀」。
  完整    五項短測都在、無牆鐘逾時、無 fatal ⇒ complete；否則整體 incomplete（已得子項照報）。

    python3 evaluation/door_k1_check.py <run_dir>     # 寫 <run_dir>/analysis/door_k1_check.json；結束碼 0＝complete 且四項 verified
"""
import json
import math
import os
import sys

import numpy as np

CONTACT_N = 0.1
TOL_LIMIT_DEG = 1.0
PASSIVE_DEG = 0.5
DIR_MIN_DEG = 0.5
REST_QDOT = 0.01


def _rows(rep, name):
    t = (rep.get('tests') or {}).get(name)
    return (t or {}).get('rows') or [], t


def judge(rep, spec):
    out = {'complete': bool(rep.get('complete')) and not rep.get('fatal') and not rep.get('wall_timeouts')}
    lo, hi = [float(x) for x in spec['leaf']['limit_q_deg']]
    # 讀回
    try:
        mass = float(np.asarray(rep['readback_physx']['masses']).ravel()[0])
        out['readback_physx_mass'] = abs(mass - float(spec['physics']['leaf_mass_kg'])) / float(spec['physics']['leaf_mass_kg']) <= 1e-4
    except Exception:                                    # noqa: BLE001
        out['readback_physx_mass'] = False
    # 方向
    r2, t2 = _rows(rep, 'S2')
    app = [r for r in r2 if r['applying']]
    if not app:
        out['direction'] = 'unverified'
    else:
        torque_ok = all(r['open_torque_nm'] > 0 for r in app)
        idx = r2.index(app[-1])
        gain = (r2[idx + 1]['q_deg'] if idx + 1 < len(r2) else app[-1]['q_deg']) - t2['q_init_deg']
        out['direction'] = 'verified' if (torque_ok and gain >= DIR_MIN_DEG) else 'fail'
        out['direction_detail'] = {'all_open_torque_positive': torque_ok, 'gain_deg': gain}
    # 下限
    r3, _ = _rows(rep, 'S3')
    if not r3:
        out['lower_limit'] = 'unverified'
    else:
        qmin = min(r['q_deg'] for r in r3)
        cmax = max((max(r['contact_walls_n'] or [0.0]) for r in r3), default=0.0)
        torque_neg = all(r['open_torque_nm'] < 0 for r in r3 if r['applying'])
        if qmin < lo - TOL_LIMIT_DEG:
            v = 'fail'
        elif cmax > CONTACT_N:
            v = 'not_separable'
        else:
            v = 'verified' if torque_neg else 'unverified'
        out['lower_limit'] = v
        out['lower_limit_detail'] = {'q_min_deg': qmin, 'wall_contact_max_n': cmax, 'push_torque_negative': torque_neg}
    # 上限
    r4, t4 = _rows(rep, 'S4')
    if not r4:
        out['upper_limit'] = 'unverified'
    else:
        qmax = max(r['q_deg'] for r in r4)
        reached = qmax >= hi - TOL_LIMIT_DEG
        out['upper_limit'] = ('fail' if qmax > hi + TOL_LIMIT_DEG else 'verified') if reached else 'unverified'
        out['upper_limit_detail'] = {'q_max_deg': qmax, 'reached': reached, 'force_stopped_at_s': t4.get('force_stopped_at_s')}
    # 被動
    sub = {}
    r1, _ = _rows(rep, 'S1')
    sub['S1_rest_at_zero'] = ('verified' if max(abs(r['q_deg']) for r in r1) <= PASSIVE_DEG else 'fail') if r1 else 'unverified'
    r5, t5 = _rows(rep, 'S5')
    if r5:
        dq = max(abs(r['q_deg'] - t5['q_init_deg']) for r in r5)
        inside = all(lo + TOL_LIMIT_DEG <= r['q_deg'] <= hi - TOL_LIMIT_DEG for r in r5)
        sub['S5_nonzero_hold'] = 'verified' if (dq <= PASSIVE_DEG and inside) else 'fail'
        sub['S5_detail'] = {'max_abs_dq_deg': dq, 'inside_limits': inside}
    else:
        sub['S5_nonzero_hold'] = 'unverified'
    post = [r for r in r2 if not r['applying']][1:] if r2 else []
    rest = next((i for i, r in enumerate(post) if abs(r['qdot_rad_s']) < REST_QDOT), None)
    if rest is None:
        sub['S2_after_release'] = 'unverified'
    else:
        seg = post[rest:]
        drift = max(r['q_deg'] for r in seg) - min(r['q_deg'] for r in seg)
        inside = all(lo + TOL_LIMIT_DEG <= r['q_deg'] <= hi - TOL_LIMIT_DEG for r in seg)
        sub['S2_after_release'] = ('verified' if drift <= PASSIVE_DEG else 'fail') if inside else 'unverified'
        sub['S2_after_release_detail'] = {'drift_deg': drift, 'inside_limits': inside, 'from_t': seg[0]['t']}
    vals = [sub[k] for k in ('S1_rest_at_zero', 'S5_nonzero_hold', 'S2_after_release')]
    out['passivity'] = 'fail' if 'fail' in vals else ('verified' if all(v == 'verified' for v in vals) else 'unverified')
    out['passivity_detail'] = sub
    # 阻尼（只報）
    c = float(spec['physics']['angular_damping'])
    seg = [r for r in r2 if not r['applying']][1:] if r2 else []
    seg = [r for r in seg if abs(r['qdot_rad_s']) > 1e-4]
    flags = []
    if rep.get('sleep_threshold_zero') is not True:
        flags.append('sleep_threshold_not_zero')
    if any(max(r['contact_walls_n'] or [0.0]) > CONTACT_N for r in r2):
        flags.append('wall_contact')
    if any(not (lo + TOL_LIMIT_DEG <= r['q_deg'] <= hi - TOL_LIMIT_DEG) for r in seg):
        flags.append('at_limit')
    if len(seg) >= 10 and all(r['qdot_rad_s'] > 0 for r in seg):
        t = np.array([r['t'] for r in seg]) - seg[0]['t']
        k_fit = -float(np.polyfit(t, np.log([r['qdot_rad_s'] for r in seg]), 1)[0])
        out['damping_report'] = {'fit_decay_per_s': k_fit, 'spec_angular_damping': c, 'interpretable': not flags, 'flags': flags}
    else:
        out['damping_report'] = {'fit_decay_per_s': None, 'interpretable': False, 'flags': flags + ['insufficient_monotone_segment']}
    out['all_verified'] = out['complete'] and out['readback_physx_mass'] and all(
        out[k] == 'verified' for k in ('direction', 'lower_limit', 'upper_limit', 'passivity'))
    return out


def main():
    import yaml
    run = sys.argv[1]
    rep = json.load(open(os.path.join(run, 'door_k1.json')))
    ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = yaml.safe_load(open(os.path.join(ws, rep['spec'])))
    res = judge(rep, spec)
    p = os.path.join(run, 'analysis', 'door_k1_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('_detail')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['all_verified'] else 1)


if __name__ == '__main__':
    main()
