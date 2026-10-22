#!/usr/bin/env python3
"""OPEN 旋轉偏差的來源分解（純離線、只讀；依 Codex 20261022_130555）：這些偏轉是目標本身要求的，還是執行時才產生？

**資料配對規則**
  - 時間基準＝求解器同快照：solve_in.snap_sim_t_exact（快照時刻）；同列 solve_in.q_meas（底盤＋手臂）、solve_in.s_meas、
    solve_in.offset_d_hat、solve_in.T_cyc（求解器**實際採用**的目標）。不用 task trace.t。
  - 門角：room 物理步中 sim_t 與快照時刻差 ≤ 1e-4 s 的那一步（room sim_t 為 4 位小數）；找不到 ⇒ 該快照 NA。
  - 窗口：OPEN 步（trace 第一列 OPEN 的 step）≤ 快照所配物理步 < 觸發步（trace 第一列 ABORTED 的 step）；觸發步另列。
  - 把手位姿 H(q)：門規格模型（door_geom_v2；K1 已核對與被動門實測 ≤ 0.02 mm）；G＝夾持成立時量測的 G_T_H。
**分解（全部以把手座標的旋轉向量表示，度）**
  - 機構應有轉動：夾爪與把手剛性夾持 ⇒ 相對把手的應有轉動＝0（以此為零點）。
  - cmd＝rotvec(H⁻¹·T_cyc·G)：目標相對當下把手的轉動（命令所要求）。依設計只應有繞鉸鏈（−z）的超前角，x／y 應≈初值。
  - sp＝rotvec(H⁻¹·FK(s＋d̂)·G)：設定點加偏移估計後的靜態平衡參考（模型允許一階落後，不是動態預測）。
  - meas＝rotvec(H⁻¹·FK(q_meas)·G)：實測相對轉動。
  - exec＝rotvec((H⁻¹T_cycG)⁻¹·(H⁻¹FK(q)G))：命令 → 實測的執行偏差（同快照）。
  回答：若 |cmd_x|、|cmd_y| 遠小於 |meas_x|、|meas_y| 且 exec≈meas ⇒ 偏轉**非目標要求**、在執行時產生；
  再以 sp−cmd 與 meas−sp 分出「設定點已偏」與「設定點到實測之間」兩段（後者含負載下順應與模型允許的落後，未區分）。
  只描述，不推因果；不調觀測器、不提出補償。

    python3 evaluation/door_k3a_rot_source.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
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

BIN = 0.05


def rel(H, T, G):
    return np.linalg.inv(H) @ T @ G


def run_one(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])
    tr = t.get('trace') or []
    s_open = next(e['step'] for e in tr if e.get('phase') == 'OPEN')
    s_trig = next((e['step'] for e in tr if e.get('phase') == 'ABORTED'), None)
    G = np.array(t['G_T_H'])
    rows, n_unpaired, trig = [], 0, None
    for L in a['log']:
        si = L.get('solve_in')
        if not si or si.get('T_cyc') is None:
            continue
        ts = si['snap_sim_t_exact']
        k = int(np.argmin(np.abs(st_t - ts)))
        if abs(st_t[k] - ts) > 1e-4:
            n_unpaired += 1
            continue
        step = st[k][I['step']]
        if step < s_open:
            continue
        q = st[k][I['door_q_rad']]
        H = KC._H_of(q)
        qm = np.array(si['q_meas'])
        Tc = np.array(si['T_cyc']).reshape(4, 4)
        Tm = KC._tcp(qm[:3], qm[3:9])
        Ts = KC._tcp(qm[:3], np.array(si['s_meas']) + np.array(si.get('offset_d_hat') or np.zeros(6)))
        Ec, Es, Em = rel(H, Tc, G), rel(H, Ts, G), rel(H, Tm, G)
        row = {'snap_t': ts, 'step': step, 'q_deg': math.degrees(q),
               'cmd_deg': np.degrees(rotvec(Ec[:3, :3])).tolist(), 'sp_deg': np.degrees(rotvec(Es[:3, :3])).tolist(),
               'meas_deg': np.degrees(rotvec(Em[:3, :3])).tolist(),
               'exec_deg': np.degrees(rotvec(Ec[:3, :3].T @ Em[:3, :3])).tolist(),
               'cmd_p_mm': (Ec[:3, 3] * 1e3).tolist(), 'meas_p_mm': (Em[:3, 3] * 1e3).tolist()}
        if s_trig is not None and step >= s_trig:
            if trig is None:
                trig = row
            continue
        rows.append(row)
    bins = {}
    for rw in rows:
        kq = round(math.floor(max(rw['q_deg'], 0.0) / BIN) * BIN, 3)
        bins.setdefault(f'{kq:.2f}', []).append(rw)
    out = {}
    for k in sorted(bins):
        L = bins[k]
        med = lambda key: np.median(np.array([x[key] for x in L]), axis=0).tolist()
        out[k] = {'n': len(L), **{key: med(key) for key in ('cmd_deg', 'sp_deg', 'meas_deg', 'exec_deg', 'cmd_p_mm', 'meas_p_mm')}}
    first = rows[0] if rows else None
    return {'run': os.path.basename(os.path.normpath(run)), 'step_open': s_open, 'step_trigger': s_trig, 'n_snapshots': len(rows),
            'n_unpaired': n_unpaired, 'first_snapshot': first, 'trigger_snapshot': trig, 'bins': out}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'bin_deg': BIN, 'runs': [run_one(r) for r in sys.argv[2:]],
           'note': '純離線分解（只讀）。旋轉向量為把手座標（x 沿桿、y 門扇法向、z 鉛直），度；夾持剛性下的機構應有相對轉動＝0。'
                   'cmd＝目標相對把手、sp＝設定點加 d̂ 的靜態參考、meas＝實測、exec＝命令到實測。配對以求解快照時刻與物理步 sim_t（≤ 1e-4 s）。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    f = lambda v: '[' + ', '.join(f'{x:+.3f}' for x in v) + ']'
    for R in res['runs']:
        print(f"== {R['run']} 快照 {R['n_snapshots']}、未配對 {R['n_unpaired']}、OPEN 步 {R['step_open']}、觸發 {R['step_trigger']}")
        for k in ('0.00', '0.10', '0.25', '0.40', '0.45', '0.50'):
            d = R['bins'].get(k)
            if d:
                print(f"  q {k}° n{d['n']:2d} cmd {f(d['cmd_deg'])} sp {f(d['sp_deg'])} meas {f(d['meas_deg'])} exec {f(d['exec_deg'])}")


if __name__ == '__main__':
    main()
