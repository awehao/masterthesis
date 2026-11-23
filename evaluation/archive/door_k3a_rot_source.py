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
  - （r1，已由下方 r2 取代）exec＝rotvec((H⁻¹T_cycG)⁻¹·(H⁻¹FK(q)G))，並以 sp−cmd、meas−sp 分段。
  回答：若 |cmd_x|、|cmd_y| 遠小於 |meas_x|、|meas_y| ⇒ 偏轉**非目標要求**、在執行時產生。
  只描述，不推因果；不調觀測器、不提出補償。
r2（Codex 20261022_131054）：
  - 殘差改在**共同把手座標**以旋轉矩陣閉合：R_c、R_s、R_m 為 Ec、Es、Em 的旋轉部分，
    D_SC＝R_s·R_cᵀ、D_MS＝R_m·R_sᵀ、D_MC＝R_m·R_cᵀ（左乘＝空間座標），逐筆驗證 D_MC＝D_MS·D_SC（閉合誤差另存）後各取旋轉向量。
    舊版的 exec＝rotvec(R_cᵀR_m) 是命令局部座標，作廢。旋轉向量不相減、不以分箱中位數閉合。
  - 移除物理來源百分比分攤：sp 為靜態平衡參考、d̂ 由量測更新 ⇒ 只陳述「目標沒有要求增加的 x／y 偏轉；靜態平衡參考也有部分偏轉，
    但與實測仍有差距；此差距包含動態落後、接觸負載、模型及估計誤差等尚未分離的因素」。
  - 缺 offset_d_hat ⇒ sp 相關量記 NA（不默認零）。觸發快照只收步號**恰等於**觸發步者；之後的快照另列 post_trigger。
  - 先保存逐筆分解與閉合誤差，再產生分箱摘要。

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


def decompose(Rc, Rs, Rm):
    """共同（空間＝把手）座標的旋轉殘差與閉合誤差。Rs 可為 None（缺 d̂ ⇒ NA）。回傳度。"""
    D_MC = Rm @ Rc.T
    out = {'D_MC_deg': np.degrees(rotvec(D_MC)).tolist(), 'D_SC_deg': None, 'D_MS_deg': None, 'closure_err': None}
    if Rs is not None:
        D_SC, D_MS = Rs @ Rc.T, Rm @ Rs.T
        out['D_SC_deg'] = np.degrees(rotvec(D_SC)).tolist()
        out['D_MS_deg'] = np.degrees(rotvec(D_MS)).tolist()
        out['closure_err'] = float(np.abs(D_MS @ D_SC - D_MC).max())
    return out


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
    rows, post, n_unpaired, trig = [], [], 0, None
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
        dh = si.get('offset_d_hat')
        Ts = None if dh is None else KC._tcp(qm[:3], np.array(si['s_meas']) + np.array(dh))
        Ec, Em = rel(H, Tc, G), rel(H, Tm, G)
        Es = None if Ts is None else rel(H, Ts, G)
        dec = decompose(Ec[:3, :3], None if Es is None else Es[:3, :3], Em[:3, :3])
        row = {'snap_t': ts, 'step': step, 'q_deg': math.degrees(q),
               'cmd_deg': np.degrees(rotvec(Ec[:3, :3])).tolist(),
               'sp_deg': None if Es is None else np.degrees(rotvec(Es[:3, :3])).tolist(),
               'meas_deg': np.degrees(rotvec(Em[:3, :3])).tolist(), **dec,
               'cmd_p_mm': (Ec[:3, 3] * 1e3).tolist(), 'meas_p_mm': (Em[:3, 3] * 1e3).tolist()}
        if s_trig is not None and step == s_trig:
            trig = row
            continue
        if s_trig is not None and step > s_trig:
            post.append(row)
            continue
        rows.append(row)
    bins = {}
    for rw in rows:
        kq = round(math.floor(max(rw['q_deg'], 0.0) / BIN) * BIN, 3)
        bins.setdefault(f'{kq:.2f}', []).append(rw)
    out = {}
    for k in sorted(bins):
        L = bins[k]
        def med(key):
            v = [x[key] for x in L if x[key] is not None]
            return (None if not v else np.median(np.array(v), axis=0).tolist()), len(v)
        d = {'n': len(L)}
        for key in ('cmd_deg', 'sp_deg', 'meas_deg', 'D_MC_deg', 'D_SC_deg', 'D_MS_deg', 'cmd_p_mm', 'meas_p_mm'):
            d[key], d[key + '_n'] = med(key)
        ce = [x['closure_err'] for x in L if x['closure_err'] is not None]
        d['closure_err_max'] = max(ce) if ce else None
        out[k] = d
    first = rows[0] if rows else None
    ce = [x['closure_err'] for x in rows if x['closure_err'] is not None]
    return {'run': os.path.basename(os.path.normpath(run)), 'step_open': s_open, 'step_trigger': s_trig, 'n_snapshots': len(rows),
            'n_unpaired': n_unpaired, 'n_sp_na': sum(1 for x in rows if x['sp_deg'] is None), 'closure_err_max': max(ce) if ce else None,
            'first_snapshot': first, 'trigger_snapshot': trig, 'post_trigger_snapshots': post, 'rows': rows, 'bins': out}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'bin_deg': BIN, 'runs': [run_one(r) for r in sys.argv[2:]],
           'note': '純離線分解（只讀）。旋轉向量為把手座標（x 沿桿、y 門扇法向、z 鉛直），度；夾持剛性下的機構應有相對轉動＝0。'
                   'cmd＝目標相對把手、sp＝設定點加 d̂ 的靜態平衡參考、meas＝實測。殘差 D_SC＝R_sR_cᵀ、D_MS＝R_mR_sᵀ、D_MC＝R_mR_cᵀ（共同把手座標），'
                   '逐筆閉合 D_MC＝D_MS·D_SC。不作物理來源百分比分攤。配對以求解快照時刻與物理步 sim_t（≤ 1e-4 s）。', 'version': 2}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    f = lambda v: '[' + ', '.join(f'{x:+.3f}' for x in v) + ']'
    for R in res['runs']:
        print(f"== {R['run']} 快照 {R['n_snapshots']}、未配對 {R['n_unpaired']}、sp NA {R['n_sp_na']}、閉合誤差最大 {R['closure_err_max']:.2e}、"
              f"觸發 {'有' if R['trigger_snapshot'] else '無'}、觸發後 {len(R['post_trigger_snapshots'])}")
        for k in ('0.00', '0.10', '0.25', '0.40', '0.45', '0.50'):
            d = R['bins'].get(k)
            if d:
                print(f"  q {k}° n{d['n']:2d} cmd {f(d['cmd_deg'])} meas {f(d['meas_deg'])} D_MC {f(d['D_MC_deg'])} D_SC {f(d['D_SC_deg'])} D_MS {f(d['D_MS_deg'])}")


if __name__ == '__main__':
    main()
