#!/usr/bin/env python3
"""OPEN 期間偏載—位姿誤差的同門角分箱定位（只讀既有取證資料；依 Codex 20261022_123917）。

對每個 run（OPEN → 中止），每物理步依**門角**分箱（0.05°），每箱取中位：
  - 位姿誤差：已發布目標（task trace，依來源時間）相對 FK 的平移（把手座標 mm）與旋轉（旋轉向量，把手座標 °）；
    抓取相對位姿 E＝H(q)⁻¹·T_tcp·G_T_H_grasp 的平移（mm）與旋轉（°）。
  - 承載：每指 Σ|fn|、|fn| 加權接觸重心（把手座標 mm）、回報切向合量 Σft（把手座標 N）。
  - 力矩**幾何代理量**：每指 Σ (p − o) × F，F＝fn·n（接觸點）與 ft（摩擦錨點）分開計，o＝把手 prim 原點或 TCP；把手座標 N·m。
    作用側：法向 fn·n 依夾合一致性推論為作用於門扇；切向作用側仍未完全確認 ⇒ 只稱代理量，不套用局部摩擦錐。
任一指資料缺失 ⇒ 該步該量 NA（不補零），並報每箱有效筆數。

    python3 evaluation/door_k3a_open_bins.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
from door_k3a_diag2 import rotvec                     # noqa: E402

FINGERS = ('uflite_finger1', 'uflite_finger2')
BIN = 0.05


def per_step(x, s, I, tr, tr_t, G):
    T_WH = np.array(x['T_WH']).reshape(4, 4)
    R, c = T_WH[:3, :3], T_WH[:3, 3]
    H = KC._H_of(s[I['door_q_rad']])
    Tm = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
    j = int(np.searchsorted(tr_t, x['t'] + 1e-9, side='right') - 1)
    out = {'t': x['t'], 'q_deg': math.degrees(s[I['door_q_rad']])}
    if j >= 0:
        Tg = np.array(tr[j]['target']).reshape(4, 4)
        out['tgt_fk_p_mm'] = (H[:3, :3].T @ (Tg[:3, 3] - Tm[:3, 3]) * 1e3).tolist()
        out['tgt_fk_r_deg'] = (H[:3, :3].T @ np.degrees(rotvec(Tm[:3, :3].T @ Tg[:3, :3]))).tolist() if False else \
            np.degrees(rotvec(H[:3, :3].T @ Tg[:3, :3] @ (H[:3, :3].T @ Tm[:3, :3]).T)).tolist()
    E = np.linalg.inv(H) @ Tm @ G
    out['grasp_p_mm'] = (E[:3, 3] * 1e3).tolist()
    out['grasp_r_deg'] = np.degrees(rotvec(E[:3, :3])).tolist()
    con_ok = x.get('contacts') is not None and not x.get('contacts_err')
    fri_ok = x.get('friction') is not None and not x.get('friction_err')
    for b in FINGERS:
        d = {}
        cl = (x.get('contacts') or {}).get(b, []) if con_ok else None
        fl = (x.get('friction') or {}).get(b, []) if fri_ok else None
        if cl is None or any(cc.get('fn') is None or cc.get('n') is None or None in cc['n'] or None in cc['p'] for cc in cl):
            d['valid_n'] = False
        else:
            d['valid_n'] = True
            fa = np.array([abs(cc['fn']) for cc in cl]) if cl else np.zeros(0)
            d['fn_abs'] = float(fa.sum())
            if cl and fa.sum() > 0:
                P = np.array([cc['p'] for cc in cl])
                d['centroid_mm'] = (R.T @ ((fa / fa.sum()) @ (P - c)) * 1e3).tolist()
            Fn = [cc['fn'] * np.array(cc['n']) for cc in cl]
            Pn = [np.array(cc['p']) for cc in cl]
            d['M_n_handle'] = (R.T @ np.sum([np.cross(p - c, f) for p, f in zip(Pn, Fn)], axis=0)).tolist() if cl else [0.0, 0.0, 0.0]
            d['M_n_tcp'] = (R.T @ np.sum([np.cross(p - Tm[:3, 3], f) for p, f in zip(Pn, Fn)], axis=0)).tolist() if cl else [0.0, 0.0, 0.0]
        if fl is None or any(ff.get('ft') is None or None in ff['ft'] or None in ff['p'] for ff in fl):
            d['valid_t'] = False
        else:
            d['valid_t'] = True
            Ft = [np.array(ff['ft']) for ff in fl]
            Pt = [np.array(ff['p']) for ff in fl]
            d['ft_handle'] = (R.T @ np.sum(Ft, axis=0)).tolist() if fl else [0.0, 0.0, 0.0]
            d['M_t_handle'] = (R.T @ np.sum([np.cross(p - c, f) for p, f in zip(Pt, Ft)], axis=0)).tolist() if fl else [0.0, 0.0, 0.0]
            d['M_t_tcp'] = (R.T @ np.sum([np.cross(p - Tm[:3, 3], f) for p, f in zip(Pt, Ft)], axis=0)).tolist() if fl else [0.0, 0.0, 0.0]
        out[b] = d
    return out


def run_bins(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ph = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    t_open, t_end = ph['OPEN'], (ab['t'] if ab else math.inf)
    G = np.array(t['G_T_H'])
    room = {s[I['step']]: s for s in r['steps']}
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    rows = []
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        x = json.loads(ln)
        if x['t'] is None or not (t_open <= x['t'] < t_end):
            continue
        s = room.get(x['step'])
        if s is not None:
            rows.append(per_step(x, s, I, tr, tr_t, G))
    bins = {}
    for rw in rows:
        k = round(math.floor(max(rw['q_deg'], 0.0) / BIN) * BIN, 3)
        bins.setdefault(k, []).append(rw)

    def med(lst, f):
        v = [f(x) for x in lst]
        v = [y for y in v if y is not None]
        return (None if not v else np.median(np.array(v, float), axis=0).tolist()), len(v)
    out = {}
    for k in sorted(bins):
        L = bins[k]
        d = {'n': len(L), 't0': L[0]['t'], 't1': L[-1]['t']}
        for key in ('tgt_fk_p_mm', 'tgt_fk_r_deg', 'grasp_p_mm', 'grasp_r_deg'):
            d[key], d[key + '_n'] = med(L, lambda x, key=key: x.get(key))
        for b in FINGERS:
            for key in ('fn_abs', 'centroid_mm', 'ft_handle', 'M_n_handle', 'M_t_handle', 'M_n_tcp', 'M_t_tcp'):
                d[f'{b}.{key}'], d[f'{b}.{key}_n'] = med(L, lambda x, b=b, key=key: (x.get(b) or {}).get(key))
        out[f'{k:.2f}'] = d
    return {'run': os.path.basename(os.path.normpath(run)), 't_open': t_open, 't_end': t_end, 'n_rows': len(rows), 'bins': out}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'bin_deg': BIN, 'runs': [run_bins(r) for r in sys.argv[2:]],
           'note': '只讀定位。把手座標 x 沿桿、y 門扇法向（+y 指向門內）、z 鉛直。力矩為幾何代理量（法向作用側由夾合一致性推論；切向作用側未完全確認）。'
                   'tgt_fk_r＝目標相對 FK 的旋轉（把手座標旋轉向量）。不判定、不推因果。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    for R in res['runs']:
        print('==', R['run'], 'rows', R['n_rows'])
        for k, d in R['bins'].items():
            f = lambda v, s=1: None if v is None else [round(x * s, 2) for x in v]
            print(f"  q {k}° n{d['n']:3d} tgt-fk p{f(d['tgt_fk_p_mm'])} r{f(d['tgt_fk_r_deg'])} grasp p{f(d['grasp_p_mm'])} r{f(d['grasp_r_deg'])}")
            for b in FINGERS:
                print(f"      {b[-7:]} |fn| {None if d[b+'.fn_abs'] is None else round(d[b+'.fn_abs'],2)} cen{f(d[b+'.centroid_mm'])} ft{f(d[b+'.ft_handle'])} "
                      f"Mn_h{f(d[b+'.M_n_handle'], 1e3)} Mt_h{f(d[b+'.M_t_handle'], 1e3)} (mN·m)")


if __name__ == '__main__':
    main()
