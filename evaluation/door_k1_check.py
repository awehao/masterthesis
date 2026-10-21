#!/usr/bin/env python3
"""K1 判定器（**缺證據不通過**）：只讀 door_k1_sim.py 的輸出 door_k1.json，依事前登錄條件判定（數值不依結果調整）。

一、完整性（任一不成立 ⇒ integrity fail，整體不可能 all_verified）
  schema door_k1/2；規格雜湊＝目前規格檔；登錄計畫＝door_k1_plan.plan(spec)；五項皆 status done；
  每項列數＝登錄步數、k 連續、t_pre＝k·dt、t_post＝t_pre＋dt；實際 q0 與登錄差 ≤ 0.5°、|q̇0| ≤ 0.01 rad/s；
  施力列：S1／S5 為 0，S2／S3 恰為登錄步數且自 k=0 連續，S4 自 k=0 連續且 ≤ 登錄步數；
  所有必要數值有限；接觸為長度 2、有限、非負（None／NaN／形狀錯一律不補零）；
  每列 τ_open 與由 p_apply、F、τ_extra 重算者差 ≤ 1e-9。
二、讀回：USD 15 項（door_k0_readback.checks）全真；PhysX 質量相對差 ≤ 1e-4；質心與規格差 ≤ 1e-4 m；
  慣量對角相對差 ≤ 1e-3、非對角 ≤ 1e-6·max。格式無法解析 ⇒ 該項 unverified（不冒稱完成）。
三、四項（±1° 只是本次限位診斷容差）：
  方向 D  S2 每個施力列 τ_open ≥ 0.9·F·r（r＝把手到軸水平距離）；最後施力列的 q_post − q0 ≥ 0.5°。
  下限 L  S3 每個施力列 τ_open ≤ −0.9·|F|·r；q_min（含 q_post）≥ −1° 且接觸全程 ≤ 0.1 N ⇒ verified；
          有接觸 ⇒ not_separable；q_min < −1° ⇒ fail。
  上限 U  S4 施力列 τ_open > 0；曾到 ≥ 上限 − 1° 時 q_max（含 q_post）≤ 上限 + 1° ⇒ verified，否則 fail；未到 ⇒ unverified。
  被動 P  S1：|q| ≤ 0.5°。S5：|q − q0| ≤ 0.5° 且離兩限位，且該項睡眠門檻設 0 成功（否則靜止可能是睡眠 ⇒ unverified）。
          S2 撤力後：|q̇_post| < 0.01 起至結束漂移 ≤ 0.5° 且離限位、睡眠設定成功（未靜止 ⇒ unverified）。三者皆 verified 才 verified。
  阻尼    只報；睡眠設定失敗、牆接觸或在限位 ⇒ 不可解讀。

    python3 evaluation/door_k1_check.py <run_dir>     # 寫 <run_dir>/analysis/door_k1_check.json；結束碼 0＝all_verified
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k1_force as KF                             # noqa: E402
import door_k1_plan as KP                              # noqa: E402

CONTACT_N, TOL_LIMIT_DEG, PASSIVE_DEG, DIR_MIN_DEG, REST_QDOT = 0.1, 1.0, 0.5, 0.5, 0.01
NUM = ('q_pre_deg', 'q_post_deg', 'qdot_pre', 'qdot_post', 'tau_open_nm')


def _finite(x):
    try:
        a = np.asarray(x, float)
    except (TypeError, ValueError):
        return False
    return a.size > 0 and bool(np.isfinite(a).all())


def integrity(rep, spec, spec_sha_now):
    why = []
    if rep.get('schema') != 'door_k1/2':
        why.append('schema')
    if rep.get('spec_sha256') != spec_sha_now:
        why.append('spec_sha_mismatch')
    plan = KP.plan(spec)
    if rep.get('plan') != plan:
        why.append('plan_mismatch')
    if rep.get('fatal') or rep.get('aborted') or rep.get('interrupted') or not rep.get('complete'):
        why.append('run_not_complete')
    hinge = np.array([spec['placement']['pose']['x'], spec['placement']['pose']['y'], 0.0])
    for name, P in plan.items():
        T = (rep.get('tests') or {}).get(name)
        if not T:
            why.append(f'{name}:missing')
            continue
        if T.get('status') != 'done':
            why.append(f'{name}:status:{T.get("status")}')
        rows = T.get('rows') or []
        n_force, n_total = KP.n_steps(P)
        if len(rows) != n_total:
            why.append(f'{name}:rows {len(rows)}!={n_total}')
        if not (_finite(T.get('q0_actual_deg')) and _finite(T.get('qdot0_actual'))):
            why.append(f'{name}:q0_nonfinite')
        elif abs(T['q0_actual_deg'] - P['q0_deg']) > 0.5 or abs(T['qdot0_actual']) > 0.01:
            why.append(f'{name}:q0_mismatch')
        app = []
        for i, r in enumerate(rows):
            if r.get('k') != i or abs(r.get('t_pre', -1) - i * KP.DT) > 1e-9 or abs(r.get('t_post', -1) - (i + 1) * KP.DT) > 1e-9:
                why.append(f'{name}:time_discontinuity@{i}')
                break
            if not all(_finite(r.get(k)) for k in NUM) or not all(_finite(r.get(k)) for k in ('p_apply', 'F', 'tau_extra')):
                why.append(f'{name}:nonfinite@{i}')
                break
            c = r.get('contact_walls_n')
            if not (isinstance(c, list) and len(c) == 2 and _finite(c) and min(c) >= 0.0):
                why.append(f'{name}:contact_invalid@{i}')
                break
            if abs(KF.hinge_moment(r['p_apply'], r['F'], r['tau_extra'], hinge, KP.A_OPEN) - r['tau_open_nm']) > 1e-9:
                why.append(f'{name}:tau_recompute_mismatch@{i}')
                break
            if r.get('applying'):
                app.append(i)
        if app != list(range(len(app))):
            why.append(f'{name}:force_window_not_contiguous_from_0')
        if name in ('S1', 'S5') and app:
            why.append(f'{name}:unexpected_force')
        if name in ('S2', 'S3') and len(app) != n_force:
            why.append(f'{name}:force_steps {len(app)}!={n_force}')
        if name == 'S4' and len(app) > n_force:
            why.append(f'{name}:force_steps_exceed')
    return why


def readback(rep, spec):
    import door_asset as DA
    import door_k0_readback as K0
    out = {}
    try:
        c = K0.checks(spec, rep['readback_usd'])
        out['usd'] = 'verified' if all(c.values()) else 'fail'
        out['usd_detail'] = c
    except Exception as e:                               # noqa: BLE001
        out['usd'], out['usd_detail'] = 'unverified', f'{type(e).__name__}: {e}'
    m, com, Idiag = DA.leaf_inertia(spec)
    px = rep.get('readback_physx') or {}
    try:
        mass = float(np.asarray(px['masses'], float).ravel()[0])
        out['physx_mass'] = 'verified' if abs(mass - m) / m <= 1e-4 else 'fail'
    except Exception:                                    # noqa: BLE001
        out['physx_mass'] = 'unverified'
    try:
        pos = np.asarray(px['coms'][0], float).reshape(-1, 3)[0]
        out['physx_com'] = 'verified' if np.abs(pos - np.asarray(com)).max() <= 1e-4 else 'fail'
    except Exception:                                    # noqa: BLE001
        out['physx_com'] = 'unverified'
    try:
        I = np.asarray(px['inertias'], float).reshape(-1, 9)[0].reshape(3, 3)
        dg, off = np.diag(I), I - np.diag(np.diag(I))
        ok = np.all(np.abs(dg - Idiag) / np.asarray(Idiag) <= 1e-3) and np.abs(off).max() <= 1e-6 * max(Idiag)
        out['physx_inertia'] = 'verified' if ok else 'fail'
    except Exception:                                    # noqa: BLE001
        out['physx_inertia'] = 'unverified'
    return out


def _q_all(rows):
    return [r['q_pre_deg'] for r in rows] + [r['q_post_deg'] for r in rows]


def judge(rep, spec, spec_sha_now):
    out = {'integrity_issues': integrity(rep, spec, spec_sha_now)}
    out['integrity'] = not out['integrity_issues']
    out['readback'] = readback(rep, spec)
    lo, hi = [float(x) for x in spec['leaf']['limit_q_deg']]
    r_h = KP.handle_radius(spec)
    plan = KP.plan(spec)
    T = rep.get('tests') or {}
    rows = {n: (T.get(n) or {}).get('rows') or [] for n in plan}
    sleep_ok = {n: (T.get(n) or {}).get('sleep_threshold_zero') is True for n in plan}
    q0 = {n: (T.get(n) or {}).get('q0_actual_deg') for n in plan}
    usable = out['integrity']                            # 完整性不成立時各項只報 unverified，不做判讀
    # 方向
    r2 = rows['S2']
    app = [r for r in r2 if r.get('applying')]
    if not usable or not app:
        out['direction'] = 'unverified'
    else:
        tq = all(r['tau_open_nm'] >= 0.9 * plan['S2']['force_n'] * r_h for r in app)
        gain = app[-1]['q_post_deg'] - q0['S2']
        out['direction'] = 'verified' if (tq and gain >= DIR_MIN_DEG) else 'fail'
        out['direction_detail'] = {'torque_sufficient_positive': tq, 'gain_deg': gain}
    # 下限
    r3 = rows['S3']
    app3 = [r for r in r3 if r.get('applying')]
    if not usable or not app3:
        out['lower_limit'] = 'unverified'
    else:
        tq = all(r['tau_open_nm'] <= -0.9 * abs(plan['S3']['force_n']) * r_h for r in app3)
        qmin = min(_q_all(r3))
        cmax = max(max(r['contact_walls_n']) for r in r3)
        if not tq:
            v = 'unverified'
        elif qmin < lo - TOL_LIMIT_DEG:
            v = 'fail'
        elif cmax > CONTACT_N:
            v = 'not_separable'
        else:
            v = 'verified'
        out['lower_limit'] = v
        out['lower_limit_detail'] = {'q_min_deg': qmin, 'wall_contact_max_n': cmax, 'push_torque_sufficient_negative': tq}
    # 上限
    r4 = rows['S4']
    if not usable or not r4:
        out['upper_limit'] = 'unverified'
    else:
        tq = all(r['tau_open_nm'] > 0 for r in r4 if r.get('applying'))
        qmax = max(_q_all(r4))
        reached = qmax >= hi - TOL_LIMIT_DEG
        out['upper_limit'] = ('unverified' if not tq else ('fail' if qmax > hi + TOL_LIMIT_DEG else 'verified')) if reached else 'unverified'
        out['upper_limit_detail'] = {'q_max_deg': qmax, 'reached': reached, 'torque_positive': tq,
                                     'force_stopped_at_s': (T.get('S4') or {}).get('force_stopped_at_s')}
    # 被動
    sub = {}
    if not usable:
        sub = {'S1_rest_at_zero': 'unverified', 'S5_nonzero_hold': 'unverified', 'S2_after_release': 'unverified'}
    else:
        sub['S1_rest_at_zero'] = 'verified' if max(abs(x) for x in _q_all(rows['S1'])) <= PASSIVE_DEG else 'fail'
        qs5 = _q_all(rows['S5'])
        dq = max(abs(x - q0['S5']) for x in qs5)
        inside = all(lo + TOL_LIMIT_DEG <= x <= hi - TOL_LIMIT_DEG for x in qs5)
        sub['S5_nonzero_hold'] = ('verified' if (dq <= PASSIVE_DEG and inside) else 'fail') if sleep_ok['S5'] else 'unverified'
        sub['S5_detail'] = {'max_abs_dq_deg': dq, 'inside_limits': inside, 'sleep_threshold_zero': sleep_ok['S5']}
        post = [r for r in r2 if not r.get('applying')]
        rest = next((i for i, r in enumerate(post) if abs(r['qdot_post']) < REST_QDOT), None)
        if rest is None or not sleep_ok['S2']:
            sub['S2_after_release'] = 'unverified'
        else:
            seg = _q_all(post[rest:])
            drift = max(seg) - min(seg)
            inside = all(lo + TOL_LIMIT_DEG <= x <= hi - TOL_LIMIT_DEG for x in seg)
            sub['S2_after_release'] = ('verified' if drift <= PASSIVE_DEG else 'fail') if inside else 'unverified'
            sub['S2_after_release_detail'] = {'drift_deg': drift, 'inside_limits': inside, 'from_t': post[rest]['t_post']}
    vals = [sub[k] for k in ('S1_rest_at_zero', 'S5_nonzero_hold', 'S2_after_release')]
    out['passivity'] = 'fail' if 'fail' in vals else ('verified' if all(v == 'verified' for v in vals) else 'unverified')
    out['passivity_detail'] = sub
    # 阻尼（只報）
    c = float(spec['physics']['angular_damping'])
    seg = [r for r in r2 if not r.get('applying') and _finite(r.get('qdot_post')) and abs(r['qdot_post']) > 1e-4]
    flags = [] if sleep_ok['S2'] else ['sleep_threshold_not_zero']
    if any(isinstance(r.get('contact_walls_n'), list) and _finite(r['contact_walls_n']) and max(r['contact_walls_n']) > CONTACT_N for r in r2):
        flags.append('wall_contact')
    if any(not (lo + TOL_LIMIT_DEG <= r['q_post_deg'] <= hi - TOL_LIMIT_DEG) for r in seg if _finite(r.get('q_post_deg'))):
        flags.append('at_limit')
    if usable and len(seg) >= 10 and all(r['qdot_post'] > 0 for r in seg):
        t = np.array([r['t_post'] for r in seg]) - seg[0]['t_post']
        k_fit = -float(np.polyfit(t, np.log([r['qdot_post'] for r in seg]), 1)[0])
        out['damping_report'] = {'fit_decay_per_s': k_fit, 'spec_angular_damping': c, 'interpretable': not flags, 'flags': flags}
    else:
        out['damping_report'] = {'fit_decay_per_s': None, 'interpretable': False, 'flags': flags + ['insufficient_or_unusable']}
    rb = out['readback']
    out['all_verified'] = (out['integrity'] and all(rb[k] == 'verified' for k in ('usd', 'physx_mass', 'physx_com', 'physx_inertia'))
                           and all(out[k] == 'verified' for k in ('direction', 'lower_limit', 'upper_limit', 'passivity')))
    return out


def main():
    import yaml
    run = sys.argv[1]
    rep = json.load(open(os.path.join(run, 'door_k1.json')))
    ws = os.path.dirname(HERE)
    sp = os.path.join(ws, KP.SPEC_REL)
    spec = yaml.safe_load(open(sp))
    res = judge(rep, spec, hashlib.sha256(open(sp, 'rb').read()).hexdigest())
    p = os.path.join(run, 'analysis', 'door_k1_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('_detail')}, ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['all_verified'] else 1)


if __name__ == '__main__':
    main()
