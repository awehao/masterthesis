#!/usr/bin/env python3
"""換抓參考時間化 v4（Codex r54 §2、§4）：依既有預測淨空判定逐段降速，直到 20 Hz 取樣全部滿足（不改監看門檻）。
來源：door_regrasp_replan_v5.json 的串接列（release_outer＋M＋A＋I＋G），最前補 bench_6 110.0 s 實測初態列（同 retime_v2）。
時間參數化同 door_coopen_pass_close_inner_retime_v3（經過點版 LSPB、有號速度、接點混合、底盤車體座標取偏航區間精確最大、規劃 0.995 倍加速度），
差異：**速度比例逐段可調**（初值 0.5＝v3 自由段），反覆：
  取樣 20 Hz → 每點以 door_k3m_geom_monitor_v3 後端（目標：release_outer 的 W1 段 out、其後 None；A 未到預抓取 None、到後與 I、G in）算四類距離，
  接近率＝相鄰取樣差分（**跨模式交界不重置**），預測 d − max(接近率, v_floor)·T_plan − ds_err，T_plan＝0.29 s（規劃壓力值；有效結果上界 0.27）；
  不足 d_min＋warn 的取樣，所在段與前後各一段的速度比例 ×0.7，重新時間化；另每點核 door_coopen_pass_close_inner_v1.check（分階段規則、關節餘裕、底盤 20 mm）。
  任何段速度比例 < 0.02 仍不足 ⇒ 報阻擋（類別、位置），不改其他參數。
報：違規取樣點數與各類違規數分開；各類最小 d 與最小預測值；總時長；各段速度比例分佈；20 Hz 參考（供參考包 v2）。

    python3 -u evaluation/door_regrasp_retime_v4.py   # 寫 results/door/door_regrasp_retime_v4.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_retime_v3 as RT3   # noqa: E402
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v5.json')
REL = os.path.join(HERE, 'results', 'door', 'door_coopen_release_bench6_v4.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_retime_v4.json')
C = PI.C
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
WARN, T_PLAN, DT_CTRL = 0.001, 0.29, 0.05
CL = ('nonfinger', 'finger', 'base', 'self')
F0 = (0.001686, 0.001704)


def seg_dt(P, Q, F, k, ratio):
    d = P[k + 1] - P[k]
    bx, by = RT3.body_xy_max(d, float(P[k][2]), float(P[k + 1][2]))
    db = np.abs(d)
    db[0], db[1] = bx, by
    return max(float(np.max(db / (RT3.VMAX * ratio[k]))), abs(Q[k + 1] - Q[k]) / (RT3.VQ * RT3.TIGHT), abs(F[k + 1] - F[k]) / RT3.VF, 1e-3)


def plan(P, Q, F, ratio):
    n = len(P)
    dt = np.array([seg_dt(P, Q, F, k, ratio) for k in range(n - 1)])
    for _ in range(5000):
        v = [(P[k + 1] - P[k]) / dt[k] for k in range(n - 1)]
        vq = [(Q[k + 1] - Q[k]) / dt[k] for k in range(n - 1)]
        vext = [np.zeros(9)] + v + [np.zeros(9)]
        tau = np.zeros(n)
        for j in range(n):
            dv = RT3.body(vext[j + 1] - vext[j], float(P[j][2]))
            tau[j] = float(np.max(np.abs(dv) / (RT3.AMAX * RT3.TIGHT)))
        bad = [k for k in range(n - 1) if 0.5 * (tau[k] + tau[k + 1]) > dt[k] * (1 - 1e-9)]
        if not bad:
            return dt, tau, v, vq
        for k in bad:
            dt[k] *= 1.1
    raise RuntimeError('時間化未收斂')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    raw = open(SRC, 'rb').read()
    src = json.loads(raw)
    if not src.get('ok'):
        raise SystemExit('重規劃結果不成立，無串接可時間化')
    rel = json.load(open(REL))['start']
    rows = [{'stage': 'release_outer', 'x': list(rel['base']) + list(rel['qa']), 'q': rel['q_deg'], 'qf': float(np.mean(F0))}] + src['rows']
    P = [np.array(r['x'], float) for r in rows]
    Q = [float(r['q']) for r in rows]
    F = [float(r['qf']) for r in rows]
    stages = [r['stage'] for r in rows]
    for k in range(1, len(P)):
        P[k][2] = P[k - 1][2] + math.remainder(P[k][2] - P[k - 1][2], 2 * math.pi)
    T0 = C.K.fk([float(v) for v in P[0]], 'link_tcp')
    p0, z0 = T0[:3, 3], T0[:3, 2]
    i_rel = [i for i, s in enumerate(stages) if s == 'release_outer']
    i_w1_end = next(i for i in i_rel if float((p0 - C.K.fk([float(v) for v in P[i]], 'link_tcp')[:3, 3]) @ z0) >= 0.06 - 1e-4)
    Tg = np.asarray(C.K.fk([float(v) for v in P[max(i for i, s in enumerate(stages) if s == 'I')]], 'link_tcp'))
    Tpre = PI.tz(Tg, -0.06)
    fn = GM3.backend_v3(self_mode='full')
    ratio = np.full(len(P) - 1, 0.5)
    it_log = []
    for it in range(60):
        TJ.cool()
        dt, tau, v, vq = plan(P, Q, F, ratio)
        S, tw = RT3.sample(P, Q, F, dt, tau, v, vq, stages)
        prev, viol_pts, viol_cls, bad_segs, first = None, 0, {c: 0 for c in CL}, set(), None
        mins = {c: {'d': math.inf, 'd_pred': math.inf} for c in CL}
        pi_fail = []
        for s in S:
            x, st = np.asarray(s['x'], float), s['stage']
            if st == 'release_outer':
                tgt = 'out' if s['seg'] < i_w1_end else None
            elif st == 'A':
                tgt = 'in' if float(np.linalg.norm(C.K.fk(list(x), 'link_tcp')[:3, 3] - Tpre[:3, 3])) <= 0.005 else None
            elif st in ('I', 'G'):
                tgt = 'in'
            else:
                tgt = None
            fn.set_target(tgt)
            d = fn(x, math.radians(s['q']), s['qf'])
            vp = False
            for c in CL:
                dc = float(d[c][0])
                rate = 0.0 if prev is None else max(0.0, (prev[c] - dc) / DT_CTRL)
                dp = dc - max(rate, VF[c]) * T_PLAN - DS[c]
                mins[c]['d'], mins[c]['d_pred'] = min(mins[c]['d'], dc), min(mins[c]['d_pred'], dp)
                if dp < DMIN[c] + WARN:
                    viol_cls[c] += 1
                    vp = True
                    if first is None:
                        first = {'t': s['t'], 'stage': st, 'class': c, 'd_mm': dc * 1e3, 'd_pred_mm': dp * 1e3, 'rate': rate, 'who': d[c][1]}
            if vp:
                viol_pts += 1
                for kk in (s['seg'] - 1, s['seg'], s['seg'] + 1):
                    if 0 <= kk < len(ratio):
                        bad_segs.add(kk)
            ok_pi, why_pi, _ = PI.check(x, s['q'], s['qf'], 'release_outer_c' if st == 'release_outer' else st)
            if not ok_pi:
                pi_fail.append({'t': s['t'], 'stage': st, 'why': why_pi})
            prev = {c: float(d[c][0]) for c in CL}
        it_log.append({'it': it, 't_end': S[-1]['t'], 'viol_points': viol_pts, 'viol_by_class': viol_cls, 'first': first, 'n_pi_fail': len(pi_fail),
                       'ratio_min': float(ratio.min())})
        print(f"it {it}：總時長 {S[-1]['t']:.1f} s、違規點 {viol_pts}（{viol_cls}）、PI 失敗 {len(pi_fail)}、比例最小 {ratio.min():.3f}；首 {first}", flush=True)
        if viol_pts == 0 and not pi_fail:
            break
        if pi_fail:
            break
        for kk in bad_segs:
            ratio[kk] *= 0.7
        if ratio.min() < 0.02:
            break
    ok = viol_pts == 0 and not pi_fail
    out = {'schema': 'door_regrasp_retime/4', 'source': os.path.relpath(SRC, HERE), 'ok': ok, 'T_plan_s': T_PLAN, 'warn_m': WARN,
           'iterations': it_log, 'pi_fail_head': pi_fail[:20], 'min_m': mins, 't_end_s': S[-1]['t'], 'n_samples': len(S),
           'ratio_hist': np.histogram(ratio, bins=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.51])[0].tolist(),
           'stage_t_start': {st: next((float(tw[i - 1]) for i, s in enumerate(stages) if s == st and i > 0), None) for st in ('M', 'A', 'I', 'G')},
           'reference_20hz': {'cols': ['t', 'x9', 'q_door_deg', 'q_finger', 'stage'],
                              'rows': [[s['t'], [float(z) for z in s['x']], s['q'], s['qf'], s['stage']] for s in S]},
           'wall_s': time.time() - t0w}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('ok', 't_end_s', 'min_m', 'ratio_hist')}, ensure_ascii=False, default=float), flush=True)


if __name__ == '__main__':
    main()
