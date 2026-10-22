#!/usr/bin/env python3
"""K3-A 接觸承載取證報告（只讀；依 K3_contact_evidence_spec.md §4／§10 事前登錄）。

輸入：<run>/evidence.jsonl、room_run.json、task.json。每物理步、每個 keep 剛體：
  點數、fn 帶號和、|fn| 和、|fn| 加權接觸點與法向（把手座標：x 沿桿、y 法向、z 鉛直；原點＝把手 prim 原點；m）、
  切向合量（把手座標）、代理比 |Σft| / Σ|fn|（**只是 pair 層級代理指標，不代表局部摩擦錐或滑移**）。
另列 j6 反作用旋量原始值（子關節座標、參考點在 joint6；**不與接觸合量比較分量符號**）與 DOF 方向量測力。
窗口：HOLD、OPEN→首次單指 < 0.5 N（物理步，依 room 的 bar 分類力）、首次低接觸→中止、中止後。
結果只稱「取證觀察」；法向作用側未核清 ⇒ 不宣稱物理合力矩。

    python3 evaluation/door_k3a_evidence_report.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_evidence_report.{json,png}（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3_evidence as EV                         # noqa: E402

FINGERS = ('uflite_finger1', 'uflite_finger2')


def handle_frame(T):
    T = np.asarray(T, float).reshape(4, 4)
    return T[:3, :3], T[:3, 3]


def per_step(row):
    R, c = handle_frame(row['T_WH'])
    out = {}
    summ = EV.summarize_contacts(row.get('contacts') or {}, row['T_WH'])
    for nm, s in summ.items():
        out[nm] = dict(s)
    for nm, lst in (row.get('friction') or {}).items():
        ft = np.array([x['ft'] for x in lst if None not in x['ft']] or [[0.0, 0.0, 0.0]])
        v = R.T @ ft.sum(axis=0)
        out.setdefault(nm, {})['ft_handle_sum'] = v.tolist()
        out[nm]['n_friction'] = len(lst)
    for nm, d in out.items():
        fa = d.get('fn_abs_sum')
        ft = d.get('ft_handle_sum')
        d['proxy_ratio'] = (float(np.linalg.norm(ft)) / fa) if (fa and ft is not None and fa > 0) else None
    return out


def main():
    run = sys.argv[1]
    out_j = os.path.join(run, 'analysis', 'door_k3a_evidence_report.json')
    out_p = os.path.join(run, 'analysis', 'door_k3a_evidence_report.png')
    for p in (out_j, out_p):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    t_ab = ab['t'] if ab else math.inf
    room = {s[I['step']]: s for s in r['steps']}
    t_low = next(s[I['sim_t']] for s in r['steps'] if s[I['sim_t']] >= t_open
                 and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5)
    rows = []
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        x = json.loads(ln)
        if x['t'] is None or x['t'] < t_hold - 0.5:
            continue
        s = room.get(x['step'])
        rows.append({'step': x['step'], 't': x['t'], 'q_deg': None if s is None else math.degrees(s[I['door_q_rad']]),
                     'bodies': per_step(x), 'j6': x.get('j6_reaction_wrench'), 'eff': x.get('dof_measured_efforts')})
    wins = [('HOLD', t_hold, t_open), ('OPEN→first_low', t_open, t_low), ('first_low→abort', t_low, t_ab), ('post_abort', t_ab, math.inf)]

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.array(vals, float)
        return {'start': a[0].tolist(), 'end': a[-1].tolist(), 'min': a.min(axis=0).tolist(), 'max': a.max(axis=0).tolist(), 'n': len(a)}
    W = {}
    for name, lo, hi in wins:
        w = [x for x in rows if lo <= x['t'] < hi]
        d = {'n_steps': len(w), 't0': w[0]['t'] if w else None, 't1': w[-1]['t'] if w else None}
        for b in FINGERS + ('uflite_gripper_link', 'link_eef', 'link6'):
            g = [x['bodies'].get(b) for x in w]
            d[b] = {'n_steps_in_contact': sum(1 for v in g if v and v.get('n')),
                    'n_points': stat([v.get('n') for v in g if v]),
                    'fn_signed_sum': stat([v.get('fn_signed_sum') for v in g if v]),
                    'fn_abs_sum': stat([v.get('fn_abs_sum') for v in g if v]),
                    'p_handle_absw_mm': stat([None if (not v or v.get('p_handle_absw') is None) else (np.array(v['p_handle_absw']) * 1e3).tolist() for v in g]),
                    'n_handle_absw': stat([v.get('n_handle_absw') for v in g if v]),
                    'ft_handle_sum': stat([v.get('ft_handle_sum') for v in g if v]),
                    'proxy_ratio': stat([v.get('proxy_ratio') for v in g if v])}
        d['j6_reaction_wrench_raw'] = stat([x['j6'] for x in w])
        d['dof_measured_efforts'] = stat([[x['eff'][k] for k in EV.ARM_DOF] for x in w if x['eff']])
        W[name] = d
    res = {'run': os.path.basename(os.path.normpath(run)), 't_hold': t_hold, 't_open': t_open, 't_first_low_phys': t_low, 'abort': ab,
           'windows': W,
           'note': '取證觀察（只讀）。把手座標 x 沿桿、y 門扇法向、z 鉛直，原點＝把手 prim 原點。fn 可帶負號；幾何權重＝|fn|。'
                   '代理比＝|Σft|/Σ|fn|（pair 層級），不代表局部摩擦錐或滑移。j6 旋量為子關節座標、參考點 joint6，不與接觸合量比分量符號。'
                   '法向作用側未核 ⇒ 不宣稱物理合力矩。',
           'series_every_5': [{'t': x['t'], 'q_deg': x['q_deg'], 'bodies': {b: x['bodies'].get(b) for b in FINGERS}, 'j6': x['j6']} for x in rows[::5]]}
    os.makedirs(os.path.dirname(out_j), exist_ok=True)
    json.dump(res, open(out_j, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tt = np.array([x['t'] for x in rows])

    def ser(b, key, i=None, scale=1.0):
        out = []
        for x in rows:
            v = x['bodies'].get(b) or {}
            y = v.get(key)
            out.append(np.nan if y is None else (y[i] if i is not None else y) * scale)
        return np.array(out, float)
    fig, ax = plt.subplots(5, 1, figsize=(8.5, 12), sharex=True)
    for b, ls in zip(FINGERS, ('-', '--')):
        ax[0].plot(tt, ser(b, 'fn_abs_sum'), ls, label=f'{b} sum|fn| (N)')
        ax[0].plot(tt, ser(b, 'n'), ls, alpha=0.4, label=f'{b} n points')
        for i, n in enumerate('yz'):
            ax[1].plot(tt, ser(b, 'p_handle_absw', i + 1, 1e3), ls, label=f'{b} contact {n} (mm, handle)')
        ax[2].plot(tt, ser(b, 'p_handle_absw', 0, 1e3), ls, label=f'{b} contact x along bar (mm)')
        for i, n in enumerate('yz'):
            ax[3].plot(tt, ser(b, 'n_handle_absw', i + 1), ls, label=f'{b} normal {n}')
        ax[4].plot(tt, ser(b, 'proxy_ratio'), ls, label=f'{b} |sum ft|/sum|fn| (proxy)')
    ax[4].set_xlabel('sim time (s)')
    for a_ in ax:
        for tv, c in ((t_open, 'k'), (t_low, 'm'), (t_ab, 'r')):
            if math.isfinite(tv):
                a_.axvline(tv, color=c, lw=0.8)
        a_.legend(fontsize=6, loc='upper left', ncol=2)
        a_.grid(alpha=0.3)
    ax[0].set_title(f"{res['run']} contact evidence: OPEN (black), first finger <0.5N (magenta), abort (red)")
    fig.tight_layout()
    fig.savefig(out_p, dpi=120)
    brief = {}
    for name in ('HOLD', 'OPEN→first_low', 'first_low→abort'):
        d = W[name]
        brief[name] = {b: {k: d[b][k] for k in ('n_steps_in_contact', 'n_points', 'fn_abs_sum', 'p_handle_absw_mm', 'n_handle_absw', 'ft_handle_sum', 'proxy_ratio')}
                       for b in FINGERS}
    print(json.dumps({'t_open': t_open, 't_first_low_phys': t_low, 'abort_t': t_ab, 'windows': brief}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
