#!/usr/bin/env python3
"""W1c（原保持基準）＋W1d（短窗對照）暫停期：門角、TCP—把手相對運動、接觸承載與底盤命令／實測運動對齊（只讀；Codex 20261025_013926 (a)）。

回答三件事：
  Q1 PAUSE 各段（SETTLE、PAUSE→ZERO_PUB、ZERO_PUB→STOPPED、STOPPED→中止；W1d 另分新命令套用前／套用後至 FAIL）門角變化（角度差，不用 raw 門速）、
     TCP 相對實測把手（evidence T_WH，以 PAUSE 為基準，mm）、兩指法向承載（room_run leaf_contacts bar）與底盤命令／實測位移速度如何對齊；
  Q2 現有接觸資料能否算繞鉸鏈的接觸力矩：以「接觸點 p × (fn·n)」與「摩擦錨點 p × ft」（原樣、不翻號）算 τ_z，對照門的被動模型在
     角度差得到的 ω、ω̇ 下所需的力矩量級（讀回：joint_friction 0、drive 0、角阻尼 c＝2.0 只作用繞質心、linear_damping 0）：
     τ_req ≈ c·I_zz·ω ＋ I_h·ω̇；兩者量級不符 ⇒ 接觸力矩 NA（作用側與正負約定未經本專案核對，不猜符號）；
  Q3 被動設定讀回能支持哪些解釋、仍缺哪些證據。
底盤：實測速度＝世界位姿 0.1 s 位移轉本體（不是 meas_vb_body 單步差分）；偏差＝實測 − 套用命令（base_cmd_body）。單趟描述，不推因果。
輸出：results/door/w1cd_pause_door_analysis.json、第七次素材／圖表／w1cd_暫停期門與底盤對齊.png。

    python3 evaluation/door_w1cd_pause_door_analysis.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

RUNS = {'w1c': os.path.join(HERE, 'runs', 'door_wide_w1c_30_1'), 'w1d': os.path.join(HERE, 'runs', 'door_wide_w1d_30_1')}
OUT = os.path.join(HERE, 'results', 'door', 'w1cd_pause_door_analysis.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1cd_暫停期門與底盤對齊.png')
FING = ('uflite_finger1', 'uflite_finger2')


def load(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    s = json.load(open(os.path.join(run, 'align_solver.json')))
    pe = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    ta = next(e['t'] for e in t['policy_events'] if e.get('abort'))
    ev = {}
    for line in open(os.path.join(run, 'evidence.jsonl')):
        d = json.loads(line)
        if pe['SETTLE'] - 0.5 <= d['t'] <= ta + 1e-6:
            ev[d['step']] = d
    tr = s['stats']['k3m_pause']['supervisor_outcome']['transitions']
    return t, r, s, pe, ta, ev, tr


def body(v, yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    return [v[0] * c + v[1] * s_, -v[0] * s_ + v[1] * c]


def tau_z(d):
    tn = tf = 0.0
    for L in (d.get('contacts') or {}).values():
        for c in L:
            if c.get('fn') is None:
                continue
            F = np.array(c['n'], float) * float(c['fn'])
            tn += c['p'][0] * F[1] - c['p'][1] * F[0]
    for L in (d.get('friction') or {}).values():
        for c in L:
            F = np.array(c['ft'], float)
            tf += c['p'][0] * F[1] - c['p'][1] * F[0]
    return tn, tf


def analyse(k, run):
    t, r, s, pe, ta, ev, tr = load(run)
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    dcfg = r['config']['door']
    st = [x for x in r['steps'] if pe['SETTLE'] - 0.5 - 1e-9 <= x[I['sim_t']] <= ta + 1e-9]
    j_of = {int(x[I['step']]): j for j, x in enumerate(r['steps'])}
    rel0 = None
    rows = []
    for x in st:
        tt = float(x[I['sim_t']])
        j = j_of[int(x[I['step']])]
        p = r['steps'][j - 10]
        dt = tt - float(p[I['sim_t']])
        b, bp = x[I['base_xyth']], p[I['base_xyth']]
        vb = body([(b[0] - bp[0]) / dt, (b[1] - bp[1]) / dt], b[2])
        e = ev.get(int(x[I['step']]))
        rel = tn = tf = None
        if e and e.get('T_WH'):
            Trel = np.linalg.inv(np.array(e['T_WH'], float).reshape(4, 4)) @ KC._tcp(b, x[I['q_arm_meas']])
            if rel0 is None and tt >= pe['PAUSE'] - 1e-9:
                rel0 = Trel[:3, 3].copy()
            if rel0 is not None:
                rel = ((Trel[:3, 3] - rel0) * 1e3).tolist()
            tn, tf = tau_z(e)
        lc = x[I['leaf_contacts']] or {}
        rows.append({'t': tt, 'q': float(x[I['door_q_rad']]), 'xyth': list(b), 'v_body_mm_s': [v * 1e3 for v in vb],
                     'cmd_body_mm_s': [float(c) * 1e3 for c in x[I['base_cmd_body']][:2]], 'rel_mm': rel, 'tau_n': tn, 'tau_f': tf,
                     'fb': [float((lc.get(f) or {}).get('bar', 0.0)) for f in FING]})
    # 分段
    segs = [('SETTLE', pe['SETTLE'], pe['PAUSE']), ('PAUSE→ZERO_PUB', pe['PAUSE'], tr['ZERO_PUB']),
            ('ZERO_PUB→STOPPED', tr['ZERO_PUB'], tr['STOPPED'])]
    if k == 'w1d':
        rec = s['stats']['pause_reanchor']['record']
        seq1 = rec['first_published']['source_seq']
        t_apply = next(float(x[I['sim_t']]) for x in r['steps'] if x[I['applied_src']][2] >= seq1)
        segs = [segs[0], segs[1], ('ZERO_PUB→新命令套用', tr['ZERO_PUB'], t_apply), ('套用→FAIL', t_apply, tr['FAIL']),
                ('FAIL→任務中止', tr['FAIL'], ta)]
    else:
        segs.append(('STOPPED→中止', tr['STOPPED'], ta))
    Izz = float(dcfg['diagonal_inertia'][2])
    m, com = float(dcfg['mass_kg']), dcfg['center_of_mass']
    Ih = Izz + m * (com[0] ** 2 + com[1] ** 2)
    c_ang = float(dcfg['angular_damping'])
    out = []
    for nm, a, b_ in segs:
        w = [x for x in rows if a - 1e-9 <= x['t'] <= b_ + 1e-9]
        if len(w) < 2:
            continue
        T = w[-1]['t'] - w[0]['t']
        om = (w[-1]['q'] - w[0]['q']) / T if T > 0 else float('nan')
        half = [x for x in w if x['t'] <= w[0]['t'] + T / 2]
        om1 = (half[-1]['q'] - w[0]['q']) / max(half[-1]['t'] - w[0]['t'], 1e-9)
        om2 = (w[-1]['q'] - half[-1]['q']) / max(w[-1]['t'] - half[-1]['t'], 1e-9)
        omd = (om2 - om1) / (T / 2) if T > 0 else float('nan')
        dx, dy = w[-1]['xyth'][0] - w[0]['xyth'][0], w[-1]['xyth'][1] - w[0]['xyth'][1]
        vmean = body([dx / T * 1e3, dy / T * 1e3], w[0]['xyth'][2]) if T > 0 else [float('nan')] * 2
        cmean = np.mean([x['cmd_body_mm_s'] for x in w], 0).tolist()
        tz = [x['tau_n'] + x['tau_f'] for x in w if x['tau_n'] is not None]
        rl = [x['rel_mm'] for x in w if x['rel_mm'] is not None]
        req = c_ang * Izz * abs(om) + Ih * abs(omd)
        out.append({'seg': nm, 't': [a, b_], 'dur_s': T, 'door_deg_change': math.degrees(w[-1]['q'] - w[0]['q']),
                    'door_omega_mrad_s_from_angle': om * 1e3, 'base_mean_v_body_mm_s': vmean, 'base_cmd_mean_body_mm_s': cmean,
                    'v_minus_cmd_body_mm_s': [vmean[0] - cmean[0], vmean[1] - cmean[1]],
                    'v_minus_cmd_norm_mm_s': math.hypot(vmean[0] - cmean[0], vmean[1] - cmean[1]),
                    'tcp_rel_handle_change_mm': (np.array(rl[-1]) - np.array(rl[0])).tolist() if len(rl) > 1 else None,
                    'fb_mean_n': np.mean([x['fb'] for x in w], 0).tolist(),
                    'tau_contact_z_Nm': None if not tz else {'median': float(np.median(tz)), 'p05': float(np.percentile(tz, 5)),
                                                              'p95': float(np.percentile(tz, 95))},
                    'tau_required_model_Nm': req,
                    'ratio_median_abs_contact_over_required': None if not tz or req <= 0 else float(np.median(np.abs(tz)) / req)})
    hold = [e for e in s['stats']['k3m_pause']['events'] if e.get('stage') == 'hold']
    anc = hold[0]['anchor'] if hold else None
    return rows, out, {'transitions': tr, 'pause': pe['PAUSE'], 'settle': pe['SETTLE'], 'abort': ta, 'hold_anchor': anc,
                       'Izz': Izz, 'I_hinge': Ih, 'c_ang': c_ang}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUNS['w1c'], 'room_run.json')))
    dcfg = r['config']['door']
    passive = {k: dcfg.get(k) for k in ('joint_friction', 'drive_stiffness', 'drive_damping', 'drive_max_force', 'angular_damping',
                                        'linear_damping', 'mass_kg', 'diagonal_inertia', 'center_of_mass', 'limit_lower_deg', 'limit_upper_deg')}
    res = {'schema': 'w1cd_pause_door/1', 'passive_readback': passive,
           'contact_semantics': 'get_contact_force_data：感測端＝門扇、法向正負約定未經本專案核對；get_friction_data 錨點與接觸點不保證一一對應（door_k3_evidence）',
           'runs': {}}
    allrows = {}
    for k, run in RUNS.items():
        rows, segs, meta = analyse(k, run)
        allrows[k] = (rows, meta)
        res['runs'][k] = {'meta': meta, 'segments': segs}
    # P 伺服穩態：|錨點誤差| 預測＝|命令|／k_p（k_p 2.0）對照實測
    for k in RUNS:
        rows, meta = allrows[k]
        tz = meta['transitions']['ZERO_PUB']
        x = min(rows, key=lambda y: abs(y['t'] - tz))
        a = meta['hold_anchor']
        err = math.hypot(x['xyth'][0] - a[0], x['xyth'][1] - a[1]) * 1e3
        res['runs'][k]['servo_steady'] = {'anchor_err_mm_at_zero_pub': err, 'cmd_norm_mm_s_at_zero_pub': math.hypot(*x['cmd_body_mm_s']),
                                          'pred_err_mm_cmd_over_kp2': math.hypot(*x['cmd_body_mm_s']) / 2.0}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    sty = {'w1c': dict(color='#444', lw=1.1), 'w1d': dict(color='#2a6fdb', lw=1.4)}
    for k, (rows, meta) in allrows.items():
        t0 = meta['pause']
        tt = np.array([x['t'] - t0 for x in rows])
        q0 = next(x['q'] for x in rows if x['t'] >= t0 - 1e-9)
        ax[0].plot(tt, [math.degrees(x['q'] - q0) for x in rows], label=f'{k}', **sty[k])
        rl = [(x['t'] - t0, x['rel_mm']) for x in rows if x['rel_mm'] is not None]
        ax[1].plot([u for u, _ in rl], [v[1] for _, v in rl], **sty[k], label=f'{k} y')
        ax[1].plot([u for u, _ in rl], [v[2] for _, v in rl], ls='--', **{kk: vv for kk, vv in sty[k].items()}, label=f'{k} z')
        ax[2].plot(tt, [x['fb'][0] for x in rows], **sty[k], label=f'{k} F1')
        ax[2].plot(tt, [x['fb'][1] for x in rows], ls='--', **sty[k], label=f'{k} F2')
        # 底盤：沿 W1d 重設後位移方向（本體 (−0.81, 0.59)）投影
        u = np.array([-0.811, 0.585])
        ax[3].plot(tt, [np.dot(x['cmd_body_mm_s'], u) for x in rows], ls='--', **sty[k], label=f'{k} 命令')
        ax[3].plot(tt, [np.dot(x['v_body_mm_s'], u) for x in rows], **sty[k], label=f'{k} 實測')
        ax[4].plot(tt, [np.dot(x['v_body_mm_s'], u) - np.dot(x['cmd_body_mm_s'], u) for x in rows], **sty[k], label=f'{k} 實測−命令')
        for nm, tv, cl in (('ZERO_PUB', meta['transitions']['ZERO_PUB'], '#4a9a5a'), ('STOPPED', meta['transitions']['STOPPED'], '#d9822b'),
                           ('中止', meta['abort'], '#c0392b')):
            for a_ in ax:
                a_.axvline(tv - t0, color=cl, ls=':' if k == 'w1c' else '-.', lw=0.8)
    ax[0].set_ylabel('門角變化（°，相對 PAUSE）')
    ax[1].set_ylabel('TCP 相對實測把手（mm）')
    ax[2].set_ylabel('兩指法向承載（N）')
    ax[3].set_ylabel('底盤沿 d 速度（mm/s）')
    ax[4].set_ylabel('實測−命令（mm/s）')
    ax[4].set_xlabel('相對 PAUSE 時間（s）；綠＝ZERO_PUB、橙＝STOPPED、紅＝中止（點線 w1c、點劃線 w1d）；d＝W1d 重設後位移方向（本體）')
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.legend(fontsize=7, frameon=False, ncol=4)
    fig.suptitle('暫停期對齊：W1c（原保持基準，灰）與 W1d（短窗對照，藍）', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    for k in res['runs']:
        print(k, json.dumps(res['runs'][k]['servo_steady'], ensure_ascii=False, default=float))
        for sg in res['runs'][k]['segments']:
            print('  ', json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in sg.items() if kk not in ('t',)},
                                   ensure_ascii=False, default=lambda o: round(o, 4) if isinstance(o, float) else o))


if __name__ == '__main__':
    main()
