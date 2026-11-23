#!/usr/bin/env python3
"""V2 離線誤差預算（Codex 20261101_052133）：視覺抓取目標相對真瓶的偏移 Δ，換到工具閉合軸（世界 x）與插入軸（世界 +y）核對幾何餘裕。
幾何：用 bottle_p1_3 實際走過的 INSERT／CLOSE 軌跡（已含手臂追蹤誤差，不另加），把瓶子相對軌跡移動 −Δ（＝目標偏 +Δ），算
  手指↔瓶（閉爪前，開爪）最小距離、非手指↔瓶最小距離（線上拒絕線 5.5 mm）、閉爪時瓶軸在 TCP 後方的深度（指片 1.3–21.3 mm）。
估計誤差：bottle_v1_1 的逐格誤差（相機到瓶距離 0.5–2.0 m 的有效格；距離用估計值與相機位姿算，不用真值），分成穩定偏差（平均）與隨機（標準差、5 筆中位數後的離散）。
不相加重複計入、不拿同趟偏差校正後宣稱精度提升；只比較「偏差＋隨機」落在幾何允許範圍的哪裡。
r2（Codex 20261101_071255）：估計誤差改用 bottle_v1_2 同趟影格以半像素約定（CP1）重估（bottle_v1_reest.per_frame），v1_1 只當對照；另列觀察範圍（平均與標準差不是上界）。
舊版結果 v2_budget.json（v1_1 線上、整數約定）保留不覆寫，r2 寫 v2_budget_r2.json。
    python3 evaluation/bottle_v2_budget.py [--r1]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

K = G.C.K
PAD = (0.0013, 0.0213)                                  # 瓶軸須在 TCP 後方的指片範圍（bottle_s0_ctrl GRASP_DEPTH 註解）
REJECT = 0.0055


def traj(run='bottle_p1_3'):
    r = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    names = [str(x) for x in Z['phases']]
    ph = np.array([names[int(v)] for v in c['phase_index']])
    t = c['t']
    m = (ph == 'INSERT') | ((ph == 'PREGRASP')) | ((ph == 'CLOSE') & (t < r['t_close'] + 1e-9))
    Q = np.column_stack([c[k] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
    i_close = int(np.argmin(np.abs(t - r['t_close'])))
    b0 = np.array([c['bottle_base_x'][0], c['bottle_base_y'][0], c['bottle_base_z'][0]])
    return Q[m][::2], c['finger1_q'][m][::2], Q[i_close], b0


def margins(Qi, Fi, Qc, b0, d):
    """d：目標偏移（世界 x, y），等同瓶子相對軌跡移動 −d。"""
    bb = b0 - np.array([d[0], d[1], 0.0])
    f_min, n_min = 9.9, 9.9
    for q, fq in zip(Qi, Fi):
        for n, P in G.link_world(q, fq).items():
            dd = float(G.cyl_dist(P - bb, 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min())
            if n in G.C.FINGERS:
                f_min = min(f_min, dd)
            else:
                n_min = min(n_min, dd)
    Tt = K.fk(Qc, 'link_tcp')
    depth = float(np.dot(Tt[:3, 3] - bb, Tt[:3, 2]))                       # 瓶軸在 TCP 後方的距離（沿工具 z）
    lat = float(np.dot(Tt[:3, 3] - bb, Tt[:3, 1]))                         # 閉合軸偏心
    return f_min, n_min, depth, lat


def est_errors(run='bottle_v1_1'):
    Z = np.load(os.path.join(HERE, 'runs', run, 'v1_log.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    val = c['valid'] == 1
    # 距離用估計中心與相機位姿：v1_log 只存真值距離 ⇒ 用估計中心對擷取時刻相機位置重算（相機位置＝底盤量測×外參，存於 s1_steps）
    S = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    cs = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
    rep = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    T_m = np.array(rep['v1']['meta']['T_mount'])
    dist_est = np.full(len(val), np.nan)
    for i in np.where(val)[0]:
        j = int(np.argmin(np.abs(cs['t'] - c['task_t'][i])))
        q = np.array([cs[k][j] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        p = (K.fk(q, 'base_link') @ T_m)[:3, 3]
        dist_est[i] = float(np.linalg.norm(p - np.array([c['cx'][i], c['cy'][i], c['z_support'][i] + 0.1])))
    m = val & (dist_est >= 0.5) & (dist_est <= 2.0)
    ex, ey = (c['cx'] - c['tb_x'])[m] * 1e3, (c['cy'] - c['tb_y'])[m] * 1e3
    tt = c['stamp'][m]
    med = []
    for k in range(len(tt) - 4):                                          # 連續 5 筆不同戳的中位數（同一段接近）
        if tt[k + 4] - tt[k] <= 1.0 + 1e-6:
            med.append((np.median(ex[k:k + 5]), np.median(ey[k:k + 5])))
    med = np.array(med)
    return dict(n=int(m.sum()), bias_x=float(ex.mean()), bias_y=float(ey.mean()), std_x=float(ex.std()), std_y=float(ey.std()),
                n_med=len(med), med_std_x=float(med[:, 0].std()) if len(med) else None, med_std_y=float(med[:, 1].std()) if len(med) else None,
                med_range_x=[float(med[:, 0].min()), float(med[:, 0].max())] if len(med) else None, med_range_y=[float(med[:, 1].min()), float(med[:, 1].max())] if len(med) else None)


def est_errors_reest(run='bottle_v1_2', ofs=0.5):
    """同趟影格重估（估計距離 0.5–2.0 m）：偏差、標準差、觀察範圍、5 筆中位數（1 s 內）的離散。"""
    import bottle_v1_reest as RE
    res, _, src, _ = RE.per_frame(run, (ofs,))
    R = np.array(res[ofs], float)
    R = R[(R[:, 1] >= 0.5) & (R[:, 1] <= 2.0)]
    R = R[np.argsort(R[:, 0])]
    tt, ex, ey = R[:, 0], R[:, 2], R[:, 3]
    med = np.array([(np.median(ex[k:k + 5]), np.median(ey[k:k + 5])) for k in range(len(tt) - 4) if tt[k + 4] - tt[k] <= 1.0 + 1e-6])
    return dict(run=run, pix_ofs=ofs, pose_source=src, n=len(R), bias_x=float(ex.mean()), bias_y=float(ey.mean()), std_x=float(ex.std()), std_y=float(ey.std()),
                range_x=[float(ex.min()), float(ex.max())], range_y=[float(ey.min()), float(ey.max())], n_med=len(med),
                med_std_x=float(med[:, 0].std()) if len(med) else None, med_std_y=float(med[:, 1].std()) if len(med) else None,
                med_range_x=[float(med[:, 0].min()), float(med[:, 0].max())] if len(med) else None, med_range_y=[float(med[:, 1].min()), float(med[:, 1].max())] if len(med) else None)


def main():
    r1 = '--r1' in sys.argv
    Qi, Fi, Qc, b0 = traj()
    f0, n0, dep0, lat0 = margins(Qi, Fi, Qc, b0, (0.0, 0.0))
    Tt = K.fk(Qc, 'link_tcp')
    print(f'工具軸（閉爪時）：閉合軸 y_tool＝{np.round(Tt[:3, 1], 3).tolist()}、插入軸 z_tool＝{np.round(Tt[:3, 2], 3).tolist()}')
    print(f'名目（Δ＝0）：手指↔瓶 {f0 * 1e3:.2f} mm、非手指↔瓶 {n0 * 1e3:.2f} mm、深度 {dep0 * 1e3:.2f} mm、閉合軸偏心 {lat0 * 1e3:.2f} mm')
    grid = np.arange(-5.0, 5.01, 0.5) * 1e-3
    res = dict(nominal=dict(finger=f0, nonfinger=n0, depth=dep0, lat=lat0))
    for axis, k in (('閉合軸（世界 x）', 0), ('插入軸（世界 y）', 1)):
        rows = []
        for v in grid:
            d = [0.0, 0.0]
            d[k] = v
            f, n, dep, lat = margins(Qi, Fi, Qc, b0, d)
            ok = f > 0.0 and n >= REJECT and PAD[0] <= dep <= PAD[1]
            rows.append((round(v * 1e3, 1), round(f * 1e3, 2), round(n * 1e3, 2), round(dep * 1e3, 2), ok))
        okv = [r_[0] for r_ in rows if r_[4]]
        print(f'{axis}：允許 Δ ∈ [{min(okv)}, {max(okv)}] mm（手指>0、非手指≥5.5、深度在 1.3–21.3）')
        for r_ in rows:
            print(f'   Δ {r_[0]:+5.1f} mm：手指 {r_[1]:6.2f}、非手指 {r_[2]:6.2f}、深度 {r_[3]:6.2f} mm {"" if r_[4] else "✗"}')
        res[axis] = dict(rows=rows, allowed=[min(okv), max(okv)])
    if r1:
        e = est_errors()
        print(f"\n估計誤差（v1_1，估計距離 0.5–2.0 m，{e['n']} 格）：偏差 x {e['bias_x']:+.2f}、y {e['bias_y']:+.2f} mm；單筆標準差 x {e['std_x']:.2f}、y {e['std_y']:.2f} mm；"
              f"5 筆中位數 {e['n_med']} 組，標準差 x {e['med_std_x']}、y {e['med_std_y']}，範圍 x {e['med_range_x']}、y {e['med_range_y']} mm")
        res['est'] = e
        json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'v2_budget.json'), 'w'), ensure_ascii=False, indent=1, default=float)
        return 0
    res['est'] = {}
    for run in ('bottle_v1_2', 'bottle_v1_1'):
        e = est_errors_reest(run)
        res['est'][run] = e
        print(f"\n估計誤差（{run} 半像素重估，{e['pose_source']}，估計距離 0.5–2.0 m，{e['n']} 格{'；主要' if run == 'bottle_v1_2' else '；對照'}）：偏差 x {e['bias_x']:+.2f}、y {e['bias_y']:+.2f} mm；"
              f"標準差 x {e['std_x']:.2f}、y {e['std_y']:.2f}；觀察範圍 x [{e['range_x'][0]:+.2f}, {e['range_x'][1]:+.2f}]、y [{e['range_y'][0]:+.2f}, {e['range_y'][1]:+.2f}] mm；"
              f"5 筆中位數 {e['n_med']} 組，範圍 x {None if e['med_range_x'] is None else np.round(e['med_range_x'], 2).tolist()}、y {None if e['med_range_y'] is None else np.round(e['med_range_y'], 2).tolist()} mm")
    res['note'] = '平均與標準差不是誤差上界；觀察範圍只來自同一場景、同一相機路徑（n=2 趟），不構成安全保證。插入軸實際餘裕另見 Codex 20261101_053807（預測停止淨空約 +1.3 mm）'
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'v2_budget_r2.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
