#!/usr/bin/env python3
"""動作流暢度：既有紀錄的停走／減速／轉向／手臂收放定位與分層歸因（Howard 11/11「影片來看還滿不順暢的」；reviews/20261111_howard_smooth_motion_and_right_side.md A 1–2）。
只讀 bottle_motm_obs_nom_3、bottle_motm_obs_A_2 的實錄；不調參、不改控制器。
四層分開（每控制拍 0.05 s）：
  參考 v_ref＝控制器路徑參考速度（s1_steps v_b）；命令 u＝控制器送出的底盤本體速度（cmd_chain u）；Shield 後 out（cmd_chain out）；
  實測 v_meas＝實測底盤位置差分（世界，0.05 s 間隔，未濾波；位置來自模擬讀回）。轉向率＝實測偏航差分。手臂＝實測關節差分最大值。
分段：接近（0→閉爪請求）、抓取（閉爪→抬升成立）、運送（CARRY_NAV→PARKED）、放置（PARKED→DONE）。
每段：時長、實測速度中位／最低、近停累計（< 0.02 m/s）、減速再加速循環數（實測速度 0.25 s 中位序列的局部谷，前後落差都 ≥ 0.03 m/s）、
  最大轉向率、手臂關節最大速度；各拍歸因（取第一個成立者）：
  ref_limited（參考本身低於該段名目速度 80%）→ wheel_saturated（u < 0.8·v_ref 且輪級額度 (max(|vx|,|vy|)+L|ωz|)/0.2775 ≥ 0.97）→ mpc_limited（其餘 u < 0.8·v_ref）→ shield_limited（out < 0.8·u）→ plant_lag（v_meas < 0.8·out）→ nominal。
  各低速拍同時標當拍相位、放置子相位，以及參考受限時的規則線索（觀察序列、x_slow 後 v_c、x_pre 等手臂、航向角速度上限、運送限速）。
加速度、jerk 若由位置差分取得，量化與無濾波都會放大；本檔只報速度與轉向率，不報差分加速度。
    python3 evaluation/smooth_motion_analysis.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ('bottle_motm_obs_nom_3', 'bottle_motm_obs_A_2')
WHEEL_CAP, WHEEL_L = 0.05 * 5.55, 0.245                                # 輪半徑×輪速上限＝0.2775 m/s；L＝0.245（gmpc_wb 預設）


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    s = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(s['cols']))}
    R, PH = s['rows'], list(s['phases'])
    cc = np.load(os.path.join(D, 'cmd_chain.npz'), allow_pickle=True)
    pl = np.load(os.path.join(D, 'place_log.npz'), allow_pickle=True)
    t_s = R[:, 0]
    tick = np.round(np.arange(0.0, float(t_s[-1]) - 0.05, 0.05), 3)
    idx = np.searchsorted(t_s, tick - 1e-9)
    idx = np.clip(idx, 0, len(t_s) - 1)
    i2 = np.clip(np.searchsorted(t_s, tick + 0.05 - 1e-9), 0, len(t_s) - 1)
    dt = np.maximum(t_s[i2] - t_s[idx], 1e-6)
    vm = np.hypot(R[i2, c['base_x']] - R[idx, c['base_x']], R[i2, c['base_y']] - R[idx, c['base_y']]) / dt
    wm = np.abs(np.angle(np.exp(1j * (R[i2, c['base_yaw']] - R[idx, c['base_yaw']])))) / dt
    qa = np.max(np.abs(R[i2, c['q1']:c['q6'] + 1] - R[idx, c['q1']:c['q6'] + 1]), axis=1) / dt
    vref = R[idx, c['v_b']]
    ph = [PH[int(p)] for p in R[idx, c['phase_index']]]
    ct = np.round(cc['t_send'], 3)
    def at_cmd(arr):
        j = np.clip(np.searchsorted(ct, tick - 1e-9), 0, len(ct) - 1)
        ok = np.abs(ct[j] - tick) < 1e-6
        out = np.full(len(tick), np.nan)
        out[ok] = arr[j[ok]]
        return out
    u = at_cmd(np.hypot(cc['u'][:, 0], cc['u'][:, 1]))
    wheel = at_cmd((np.maximum(np.abs(cc['u'][:, 0]), np.abs(cc['u'][:, 1])) + WHEEL_L * np.abs(cc['u'][:, 2])) / WHEEL_CAP)   # 輪級額度使用率：max(|vx|,|vy|)＋L|ωz| ≤ 0.2775（gmpc_wb wheel_matrix；Codex 161442 更正）
    out = at_cmd(np.hypot(cc['out'][:, 0], cc['out'][:, 1]))
    pt = pl['rows'][:, 0]
    pph = pl['rows'][np.clip(np.searchsorted(pt, tick - 1e-9), 0, len(pt) - 1), list(pl['cols']).index('pph')]
    ev = r['events']
    E = {}
    for e in ev:
        E.setdefault(e['what'] + ('' if 'to' not in e else ':' + str(e['to'])), e['t'])
    t_close, t_lift = float(r['t_close']), float(r['t_lift_ok'])
    t_carry, t_park, t_done = E.get('place_phase:CARRY_NAV'), E.get('place_phase:PARKED'), E.get('place_phase:DONE')
    obs_u0 = [e['t'] for e in ev if e['what'] == 'vhigh_seq' and e.get('ph') == 'U']
    obs_h = [e['t'] for e in ev if e['what'] == 'vhigh_seq' and e.get('ph') == 'H']
    t_sw = [e['t'] for e in ev if e['what'] == 'tpa_switch']
    segs = [('接近（0→閉爪請求）', 0.0, t_close, 0.35), ('抓取（閉爪→抬升成立）', t_close, t_lift, 0.10), ('運送（CARRY_NAV→PARKED）', t_carry, t_park, 0.25),
            ('放置（PARKED→DONE）', t_park, t_done, 0.0)]
    out_segs = []
    for nm, a, b, vnom in segs:
        if a is None or b is None:
            continue
        w = (tick >= a) & (tick < b)
        if not w.any():
            continue
        vv = vm[w]
        k = max(1, 5)
        sm = np.array([np.median(vv[max(0, i - 2):i + 3]) for i in range(len(vv))])
        cyc, cyc_t = 0, []
        tw = tick[w]
        for i in range(k, len(sm) - k):
            if sm[i] == sm[max(0, i - k):i + k + 1].min() and sm[max(0, i - 20):i].max() - sm[i] >= 0.03 and sm[i:i + 20].max() - sm[i] >= 0.03:
                if not cyc_t or tw[i] - cyc_t[-1][0] > 0.5:
                    cyc += 1
                    jj = np.where(w)[0][i]
                    cyc_t.append((round(float(tw[i]), 2), round(float(sm[i]), 3), round(float(vref[jj]), 3), None if not np.isfinite(wheel[jj]) else round(float(wheel[jj]), 2),
                                  round(float(wm[jj]), 3), ph[jj]))
        lab = []
        for j in np.where(w)[0]:
            if vnom > 0 and vref[j] < 0.8 * vnom:
                lab.append('ref_limited')
            elif np.isfinite(u[j]) and u[j] < 0.8 * vref[j] - 1e-6:
                lab.append('wheel_saturated' if np.isfinite(wheel[j]) and wheel[j] >= 0.97 else 'mpc_limited')
            elif np.isfinite(out[j]) and np.isfinite(u[j]) and out[j] < 0.8 * u[j] - 1e-6:
                lab.append('shield_limited')
            elif np.isfinite(out[j]) and vm[j] < 0.8 * out[j] - 1e-6:
                lab.append('plant_lag')
            else:
                lab.append('nominal')
        frac = {k_: round(lab.count(k_) * 0.05, 2) for k_ in ('ref_limited', 'wheel_saturated', 'mpc_limited', 'shield_limited', 'plant_lag', 'nominal')}
        out_segs.append(dict(seg=nm, t=[round(a, 2), round(b, 2)], dur_s=round(b - a, 2), v_nominal=vnom, v_meas_median=round(float(np.median(vv)), 3),
                             v_meas_min=round(float(vv.min()), 3), near_stop_s=round(float((vv < 0.02).sum() * 0.05), 2), decel_accel_cycles=cyc, cycles_at=cyc_t,
                             yaw_rate_max=round(float(wm[w].max()), 3), arm_joint_speed_max=round(float(qa[w].max()), 3), time_by_cause_s=frac))
    # 參考受限的規則線索（接近段）：依時間窗對到事件
    cues = []
    wa = (tick < t_close)
    rl = wa & (vref < 0.8 * 0.35)
    runs_ = []
    j0 = None
    for j in range(len(tick)):
        if rl[j] and j0 is None:
            j0 = j
        if (not rl[j] or j == len(tick) - 1) and j0 is not None:
            runs_.append((tick[j0], tick[j]))
            j0 = None
    for a, b in runs_:
        if b - a < 0.2:
            continue
        why = []
        if obs_u0 and a < (obs_h[0] if obs_h else 1e9) + 0.5 and b > obs_u0[0]:
            why.append('觀察抬臂序列期間')
        if t_sw and a >= t_sw[0] - 0.1:
            why.append('切換後插入段（v_c／heuristic）')
        relw = r.get('relwin', {}).get('new', {})
        xi = R[idx[(tick >= a) & (tick < b)], c['base_x']] - 0.6
        if relw and len(xi) and xi.max() >= float(relw.get('x_slow', 9)):
            why.append('過 x_slow 後慢速段 v_c 0.10')
        if a < 0.6:
            why.append('起步加速')
        cues.append(dict(t=[round(float(a), 2), round(float(b), 2)], v_ref_min=round(float(vref[(tick >= a) & (tick < b)].min()), 3), cue=why or ['未對到既有規則（待查）']))
    arm = []                                                            # 手臂大動作（實測關節速度 > 0.3 rad/s 持續 ≥ 0.3 s）
    hi = qa > 0.3
    j0 = None
    for j in range(len(tick)):
        if hi[j] and j0 is None:
            j0 = j
        if (not hi[j] or j == len(tick) - 1) and j0 is not None:
            if tick[j] - tick[j0] >= 0.3:
                arm.append(dict(t=[round(float(tick[j0]), 2), round(float(tick[j]), 2)], phase=ph[j0], pph=int(pph[j0]), peak=round(float(qa[j0:j + 1].max()), 3)))
            j0 = None
    T = r.get('timing', {})
    pub = np.diff(np.asarray(cc['t_send'], float))
    return dict(run=run, t_done=t_done, segments=out_segs, ref_limited_cues_approach=cues, arm_motions=arm,
                compute=dict(ctrl_tick_solve=T.get('ctrl_tick_solve'), cmd_send_interval=dict(p50=round(float(np.median(pub)), 3), max=round(float(pub.max()), 3))),
                note='速度為位置差分（0.05 s、未濾波）；歸因取第一個成立的層；ref_limited 的段名目速度：接近 0.35、抓取 0.10、運送 0.25')


def main():
    res = dict(runs=[one(x) for x in RUNS])
    for x in res['runs']:
        print('==', x['run'], 'DONE', x['t_done'])
        for sg in x['segments']:
            print('  ', sg)
        print('  參考受限線索（接近段）', x['ref_limited_cues_approach'])
        print('  手臂大動作', x['arm_motions'])
        print('  計算', x['compute'])
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
