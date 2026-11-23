#!/usr/bin/env python3
"""下滑事件與夾爪受力重新分配／手指微動的逐事件對應（Codex 20261102_035741 (a)；只讀既有紀錄，一次完成、不掃門檻）。
規則（看結果前固定）：
  訊號（0.01 s）：滑移 x＝夾持相對位姿對 t_grasp 的 TCP x（世界鉛直，負＝下滑）；受力差 fd＝f1−f2（contact force matrix 接觸法向力範數，不含切向）；手指速度 v＝finger1_q 差分。
  差分雜訊：懸持趟 13 s 靜態懸持段的 0.1 s 窗差分（滑移 Δs、受力差 Δfd）與手指速度 |v| 的 99.9 百分位 n_s、n_f、n_v。
  門檻：下滑 thr_s＝max(0.05 mm, 5·n_s)；受力重新分配 thr_f＝max(0.3 N, 5·n_f)；手指速度峰 thr_v＝max(2 mm/s, 5·n_v)。
  事件：下滑＝0.1 s 窗內滑移 x 減少 ≥ thr_s；重新分配＝0.1 s 窗內 |Δfd| ≥ thr_f；速度峰＝|v| ≥ thr_v；同類事件相距 ≤ 0.2 s 合併為一次（取區間中最大變化時刻）。
  對應窗 ±0.2 s。分析範圍：各趟持瓶行進開始（CARRY_NAV）到停止（或鬆爪）——運動段；懸持段只用來定雜訊。
報：每趟事件數；正向＝下滑事件中 ±0.2 s 內有重新分配／速度峰的比例與時間差（下滑時刻 − 對應事件時刻）中位；反向＝重新分配事件中伴隨下滑的比例；
  機率基準＝分析範圍內被重新分配（或速度峰）事件 ±0.2 s 覆蓋的時間比例（隨機時刻落入的機率）；無下滑的運動時段中重新分配事件密度 vs 下滑事件附近的密度。
時間對應只是線索，不是因果證明。輸出 results/bottle/slip_event_match.json。
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

RUNS = ('bottle_w1_vhigh_z060_p1_1', 'bottle_w1_vhigh_hold_z060_p1_1', 'bottle_w1_vhigh_outslow_z060_p1_1')
W, MERGE, MATCH = 10, 0.2, 0.2                          # 0.1 s 窗（10 步）、合併 0.2 s、對應 ±0.2 s


def load(run):
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    return c, json.load(open(os.path.join(D, 's1_report.json')))


def slip_x(c, r, idx):
    def rel(k):
        q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = Rot.from_quat([c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k], c['bottle_qw'][k]]).as_matrix()
        Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
        return (np.linalg.inv(Tt) @ Tb)[:3, 3]
    kg = int(np.argmin(np.abs(c['t'] - float(r['t_grasp']))))
    x0 = rel(kg)[0]
    return np.array([(rel(k)[0] - x0) * 1e3 for k in idx])


def sig(run, a, b):
    c, r = load(run)
    t = c['t']
    idx = np.nonzero((t >= a) & (t <= b))[0]
    s = slip_x(c, r, idx)
    fd = c['f_finger1_bottle'][idx] - c['f_finger2_bottle'][idx]
    v = np.gradient(c['finger1_q'][idx], 0.01) * 1e3
    return t[idx], s, fd, v


def diffw(x):
    d = np.full(len(x), np.nan)
    d[W:] = x[W:] - x[:-W]
    return d


def events(t, score, thr):
    """score ≥ thr 的時刻，相距 ≤ MERGE 合併；回傳各合併區間中 score 最大的時刻。"""
    k = np.nonzero(np.nan_to_num(score, nan=-np.inf) >= thr)[0]
    if not len(k):
        return np.zeros(0)
    groups, cur = [], [k[0]]
    for i in k[1:]:
        if t[i] - t[cur[-1]] <= MERGE:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return np.array([t[g[int(np.argmax(score[g]))]] for g in groups])


def match(ea, eb):
    if not len(ea):
        return 0, 0, None
    hits, dts = 0, []
    for x in ea:
        if len(eb):
            d = eb - x
            j = int(np.argmin(np.abs(d)))
            if abs(d[j]) <= MATCH:
                hits += 1
                dts.append(float(x - eb[j]))
    return hits, len(ea), (None if not dts else round(float(np.median(dts)), 3))


def coverage(t, ev):
    if not len(ev):
        return 0.0
    m = np.zeros(len(t), bool)
    for x in ev:
        m |= np.abs(t - x) <= MATCH
    return float(m.mean())


def main():
    c, r = load('bottle_w1_vhigh_hold_z060_p1_1')
    th = next(e['t'] for e in r['events'] if e['what'] == 'hold_start')
    tH, sH, fdH, vH = sig('bottle_w1_vhigh_hold_z060_p1_1', th, th + 13.0)
    n_s, n_f, n_v = (float(np.nanpercentile(np.abs(diffw(sH)), 99.9)), float(np.nanpercentile(np.abs(diffw(fdH)), 99.9)), float(np.percentile(np.abs(vH), 99.9)))
    thr = dict(slip_mm=max(0.05, 5 * n_s), fdiff_N=max(0.3, 5 * n_f), v_mm_s=max(2.0, 5 * n_v))
    out = dict(noise_p999=dict(slip_0p1s_mm=n_s, fdiff_0p1s_N=n_f, v_mm_s=n_v), thresholds=thr, window_s=W * 0.01, merge_s=MERGE, match_s=MATCH, runs={})
    for run in RUNS:
        c, r = load(run)
        ev = {}
        for e in r['events']:
            if e['what'] == 'place_phase':
                ev.setdefault(e['to'], e['t'])
        a = ev.get('CARRY_NAV')
        b = min(v for v in ((r.get('stop') or {}).get('t'), ev.get('RELEASE')) if v is not None)
        t, s, fd, v = sig(run, a, b)
        e_s = events(t, -diffw(s), thr['slip_mm'])
        e_f = events(t, np.abs(diffw(fd)), thr['fdiff_N'])
        e_v = events(t, np.abs(v), thr['v_mm_s'])
        hf, ns_, dtf = match(e_s, e_f)
        hv, _, dtv = match(e_s, e_v)
        rf, nf_, _ = match(e_f, e_s)
        cov_f, cov_v = coverage(t, e_f), coverage(t, e_v)
        far = np.ones(len(t), bool)                          # 遠離下滑事件（> 0.5 s）的運動時段
        for x in e_s:
            far &= np.abs(t - x) > 0.5
        near = ~far
        dens_far = float(sum(1 for x in e_f if far[int(np.argmin(np.abs(t - x)))]) / max(far.sum() * 0.01, 1e-9))
        dens_near = float(sum(1 for x in e_f if near[int(np.argmin(np.abs(t - x)))]) / max(near.sum() * 0.01, 1e-9))
        out['runs'][run] = dict(range_s=[round(a, 2), round(b, 2)], dur_s=round(b - a, 2), n_slip=len(e_s), n_redistribution=len(e_f), n_vpeak=len(e_v),
                                slip_down_total_mm=round(float(s[0] - s[-1]), 3),
                                forward_redistribution=dict(hits=hf, of=ns_, frac=None if not ns_ else round(hf / ns_, 3), dt_median_s=dtf, chance=round(cov_f, 3)),
                                forward_vpeak=dict(hits=hv, of=ns_, frac=None if not ns_ else round(hv / ns_, 3), dt_median_s=dtv, chance=round(cov_v, 3)),
                                reverse_redistribution_with_slip=dict(hits=rf, of=nf_, frac=None if not nf_ else round(rf / nf_, 3), chance=round(coverage(t, e_s), 3)),
                                redistribution_density_per_s=dict(near_slip=round(dens_near, 3), far_from_slip=round(dens_far, 3)),
                                slip_times=[round(float(x), 2) for x in e_s])
    out['note'] = '時間對應只是線索、不是因果；機率基準＝隨機時刻落入對應窗的比例；門檻依懸持段雜訊一次固定'
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'slip_event_match.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'runs'}, ensure_ascii=False, default=float))
    for run, v in out['runs'].items():
        print(run, {k: v[k] for k in v if k != 'slip_times'})
    return 0


if __name__ == '__main__':
    sys.exit(main())
