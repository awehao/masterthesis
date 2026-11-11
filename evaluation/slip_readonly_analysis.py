#!/usr/bin/env python3
"""只讀滑移分析（Codex 20261111_152343）：bottle_motm_obs_A_2 與 bottle_motm_obs_nom_3。不加診斷趟、不調夾力或摩擦。
滑移定義沿用判定器：抓持成立時刻 t_grasp 的 TCP→瓶相對位姿為固定基準（T_rel_base），之後每步 Δ＝(當下 TCP→瓶) 的平移 − 基準平移（TCP 座標分量）、
|Δ| 即滑移；不事後排除 LOWER、不重設零點。TCP 由實測底盤位姿（世界）＋實測手臂關節以控制器同一運動學 FK；瓶位姿為模擬讀回（世界）。
先以重算的最大滑移與報告值比對（驗證本重算與判定器一致），再列關鍵時刻：抓持成立、離桌（t_lift_ok）、PARKED、首次第二桌支撐、首次超標、鬆爪（RELEASE）。
各時刻同列：滑移向量、指力（兩指對瓶）、第二桌支撐力 fz、TCP 世界 z 速度、放置子相位。
    python3 evaluation/slip_readonly_analysis.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_r0_approach as R0                                         # noqa: E402

RUNS = ('bottle_motm_obs_A_2', 'bottle_motm_obs_nom_3')
SLIP_MM = 5.0


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    s = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(s['cols']))}
    R = s['rows']
    pl = np.load(os.path.join(D, 'place_log.npz'), allow_pickle=True)
    pc = {n: i for i, n in enumerate(list(pl['cols']))}
    P = pl['rows']
    t = R[:, 0]
    tg = float(r['t_grasp'])
    rel = []
    for k in range(len(R)):
        q9 = np.r_[R[k, c['base_x']], R[k, c['base_y']], R[k, c['base_yaw']], R[k, c['q1']:c['q6'] + 1]]
        Tt = R0.K.fk(q9, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = quat_R(R[k, c['bottle_qw']], R[k, c['bottle_qx']], R[k, c['bottle_qy']], R[k, c['bottle_qz']])
        Tb[:3, 3] = [R[k, c['bottle_base_x']], R[k, c['bottle_base_y']], R[k, c['bottle_base_z']]]
        rel.append((np.linalg.inv(Tt) @ Tb, Tt[2, 3]))
    k0 = int(np.argmin(np.abs(t - tg)))
    base = rel[k0][0]
    dv = np.array([rel[k][0][:3, 3] - base[:3, 3] for k in range(len(R))]) * 1e3        # mm，TCP 座標
    d = np.linalg.norm(dv, axis=1)
    zt = np.array([z for _, z in rel])
    vz = np.r_[0.0, np.diff(zt) / np.maximum(np.diff(t), 1e-9)]
    after = t >= tg - 1e-9
    ev = {e['to']: e['t'] for e in r['events'] if e['what'] in ('place_phase',)}
    sw = (r.get('slip_warn') or {}).get('warn') or {}
    rel_t = sw.get('t_first') if sw else None
    fz = P[:, pc['fz']]
    sup = np.where((P[:, 0] > tg) & (np.nan_to_num(fz) > 0.5))[0]                       # 首次第二桌支撐（fz > 0.5 N）
    t_sup = float(P[sup[0], 0]) if len(sup) else None
    marks = [('抓持成立 t_grasp', tg), ('離桌 t_lift_ok', r.get('t_lift_ok')), ('PARKED', ev.get('PARKED')), ('首次第二桌支撐 fz>0.5 N', t_sup),
             ('首次超標（報告）', rel_t), ('鬆爪 RELEASE', ev.get('RELEASE'))]
    rows = []
    for nm, tm in marks:
        if tm is None:
            rows.append(dict(mark=nm, t=None))
            continue
        k = int(np.argmin(np.abs(t - tm)))
        j = int(np.argmin(np.abs(P[:, 0] - tm)))
        rows.append(dict(mark=nm, t=round(float(t[k]), 3), slip_mm=round(float(d[k]), 3), slip_vec_tcp_mm=[round(float(v), 3) for v in dv[k]],
                         f_finger_N=[round(float(R[k, c['f_finger1_bottle']]), 2), round(float(R[k, c['f_finger2_bottle']]), 2)],
                         fz_table2_N=None if not np.isfinite(fz[j]) else round(float(fz[j]), 2), tcp_vz_mps=round(float(vz[k]), 4), pph=int(P[j, pc['pph']])))
    km = int(np.argmax(np.where(after & (t <= (ev.get('RELEASE') or t[-1]) + 1e-9), d, -1)))
    # 分段：各段內滑移增量（段末 − 段初）
    seg = []
    bounds = [('抓持→離桌', tg, r.get('t_lift_ok')), ('離桌→PARKED（運送）', r.get('t_lift_ok'), ev.get('PARKED')), ('PARKED→首次支撐（LOWER 前段）', ev.get('PARKED'), t_sup),
              ('首次支撐→鬆爪（LOWER 末段）', t_sup, ev.get('RELEASE'))]
    for nm, a, b in bounds:
        if a is None or b is None:
            seg.append(dict(seg=nm, a=a, b=b))
            continue
        ka, kb = int(np.argmin(np.abs(t - a))), int(np.argmin(np.abs(t - b)))
        w = slice(ka, kb + 1)
        seg.append(dict(seg=nm, a=round(float(a), 3), b=round(float(b), 3), d_start=round(float(d[ka]), 3), d_end=round(float(d[kb]), 3), d_max=round(float(d[w].max()), 3),
                        dvec_change_mm=[round(float(v), 3) for v in dv[kb] - dv[ka]]))
    prof = []                                                           # 運送與放置段逐 1 s：滑移、底盤速度／加速度、轉向率（世界）
    bx, by, byaw = R[:, c['base_x']], R[:, c['base_y']], R[:, c['base_yaw']]
    vb = np.r_[0.0, np.hypot(np.diff(bx), np.diff(by)) / np.maximum(np.diff(t), 1e-9)]
    wb = np.r_[0.0, np.diff(np.unwrap(byaw)) / np.maximum(np.diff(t), 1e-9)]
    t_a, t_b = r.get('t_lift_ok'), ev.get('RELEASE')
    if t_a is not None and t_b is not None:
        for tm in np.arange(float(t_a), float(t_b) + 1e-9, 1.0):
            w = (t >= tm - 0.5) & (t < tm + 0.5)
            k = int(np.argmin(np.abs(t - tm)))
            prof.append(dict(t=round(float(tm), 2), slip_mm=round(float(d[k]), 3), dx_tcp_mm=round(float(dv[k][0]), 3), v_base=round(float(np.median(vb[w])), 3),
                             w_base_abs=round(float(np.median(np.abs(wb[w]))), 3), f_min_N=round(float(np.min(R[w][:, [c['f_finger1_bottle'], c['f_finger2_bottle']]])), 2)))
    return dict(run=run, carry_profile=prof, report_slip_max_mm=round(float(r['slip_max_mm_deg'][0]), 4), recomputed_max_mm_to_release=round(float(d[km]), 4), t_recomputed_max=round(float(t[km]), 3),
                recomputed_vec_at_max_tcp_mm=[round(float(v), 3) for v in dv[km]], hold_ok=(r.get('slip_warn') or {}).get('hold_ok'), marks=rows, segments=seg,
                note='Δ 為 TCP 座標下瓶相對位置對抓持基準的變化（mm）；指力為兩指對瓶接觸力大小；fz 為瓶對第二桌支撐力（放置判定器量測）')


def main():
    out = dict(rule='沿用原抓持基準 T_rel_base（t_grasp），不排除 LOWER、不重設零點；只讀；成因無法分辨即寫未定', runs=[one(r) for r in RUNS])
    for x in out['runs']:
        print(x['run'], '報告', x['report_slip_max_mm'], '重算', x['recomputed_max_mm_to_release'], '@', x['t_recomputed_max'], x['recomputed_vec_at_max_tcp_mm'], flush=True)
        for m in x['marks']:
            print('  ', m, flush=True)
        for sg in x['segments']:
            print('  段', sg, flush=True)
        print('  運送剖面', [(p_['t'], p_['slip_mm'], p_['v_base'], p_['w_base_abs'], p_['f_min_N']) for p_ in x['carry_profile']], flush=True)
    json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
