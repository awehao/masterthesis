#!/usr/bin/env python3
"""K1（低速介面放寬決策資料）步驟 3：R3、S8 兩個固定案例在框 ×1／×1.5／×2 下離線重產參考，核逐點速度／加速度與完整幾何監看。
只產出到 results/k1/（不接線、不改正式參考包與跑法）。加速度上限、keep 窗、路徑幾何一律沿用現值，只放大速度上限與框。
  R3：door_ref_r3_retime2 的流程（來源 pb60m 段 41 尾段），v_max、w_pk、r3_w_cap 與框同乘 k。
  S8：door_ref_overlap_v1.retime（來源 door_return_ref_v1 A 段 S8 前 800 列路徑），V_MAX 乘 k（W_PEAK 0.0738 沿用；現值偏航不受限）。
×1 須重現現行 pb60n（sha 前 16 a70f5fe5751af996）與 v3 s8_overlap.T 28.957 s，否則判工具不一致、不採用 ×k 結果。
    python3 evaluation/k1_ref_scale.py
"""
import contextlib
import hashlib
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_ref_overlap_v1 as OV                       # noqa: E402
import door_ref_r3_retime2 as R3T                      # noqa: E402

OUT = os.path.join(HERE, 'results', 'k1')
PB60N_SHA16 = 'a70f5fe5751af996'
KS = (1.0, 1.5, 2.0)
V0_BOX, W0_BOX = R3T.V_BOX, R3T.W_BOX


def run_r3(k):
    out = os.path.join(OUT, f'r3_x{k:g}.json')
    if os.path.exists(out):
        os.remove(out)                                  # 本腳本自己的離線產物，重跑即重產
    R3T.V_BOX, R3T.W_BOX = V0_BOX * k, W0_BOX * k
    argv = ['k1', '--v-max', f'{0.030 * k}', '--w-pk', f'{0.15 * k}', '--r3-w-cap', f'{0.19 * k}', '--out', out]
    buf = io.StringIO()
    old = sys.argv
    sys.argv = argv
    rc = 'ok'
    try:
        with contextlib.redirect_stdout(buf):
            R3T.main()
    except SystemExit as e:
        rc = f'rejected: {e}'
    finally:
        sys.argv = old
        R3T.V_BOX, R3T.W_BOX = V0_BOX, W0_BOX
    txt = buf.getvalue()
    res = {'k': k, 'rc': rc, 'log': txt.strip().splitlines()}
    if rc == 'ok':
        P = json.load(open(out))
        rr = P['r3_retime2']
        s = P['segments']['41']
        res.update(T_seg41=len(s['t']) * float(P['dt']), T_tail=rr['T_tail'], T_tail_src=rr['T_tail_src'],
                   offline=rr['offline'], sha16=hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16],
                   monitor=mon_full(np.c_[np.asarray(s['base'], float), np.asarray(s['arm'], float)], float(P['q_ref_deg'])))
    return res


def s8_case(k):
    V0 = OV.V_MAX
    OV.V_MAX = V0 * k
    try:
        R = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v1.json')))
        A = R['A']
        i8 = [i for i, r in enumerate(A) if r['stage'].startswith('S8')]
        ie = i8[0] + 800
        X = np.array([r['x'] for r in A])
        DT = OV.DT
        v_e = float(np.hypot(*(X[ie + 1, :2] - X[ie, :2]) / DT))
        w_e = float((X[ie + 1, 2] - X[ie, 2]) / DT)
        new, T, vc, wc = OV.retime(X[i8[0]:ie + 1, :2], (float(X[i8[0], 2]), float(X[ie, 2])), 0.0, v_e, 0.0, w_e)
        seq = np.vstack([X[i8[0], :3], new, X[ie + 1:ie + 40, :3]])
        vel, acc = OV.rate(seq)
        armA = X[i8[0], 3:]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = mon_full(np.c_[new, np.tile(armA, (len(new), 1))], 59.9)
        L = float(np.sum(np.linalg.norm(np.diff(X[i8[0]:ie + 1, :2], axis=0), axis=1)))
        return {'k': k, 'T': T, 'vc': vc, 'wc': wc, 'L_m': L, 'body_vel_peak': vel, 'world_acc_peak': acc,
                'end_row': len(new) - 1, 'monitor': st, 'v_e': v_e, 'w_e': w_e}
    finally:
        OV.V_MAX = V0


def mon_full(rows, q_deg):
    """同 door_ref_overlap_v1.monitor，回傳 stop／warn 計數、各類真最小 d_pred（mm、列）與速度項超出下限的最大扣減（mm）。
    註：既有重時間化工具的「最小 d_pred」以公尺值比對已存的 mm 值，實際印出的是末列值；本函式以公尺比較。"""
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = OV.GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = OV.GM3.backend_v3(self_mode='full')
    mon = OV.GS.SwitchAwareGeom(OV.GM.GeometryMonitor(cfg, fn), fn)
    t, st, wn, mn, vt = 0.0, 0, 0, {}, {}
    for i, x in enumerate(rows):
        t += OV.DT
        dec, det = mon.step(t, x, math.radians(q_deg), 0.0089, 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        T = det.get('T_pred_s', 0.0)
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9.0,))[0]:
                mn[c] = (v['d_pred'], v['who'], i)
            vt[c] = max(vt.get(c, 0.0), (v['v_used'] - cfg.v_floor[c]) * T)
    return {'stop': st, 'warn': wn, 'n': len(rows), 'min_mm': {c: [round(d * 1e3, 2), w, i] for c, (d, w, i) in mn.items()},
            'v_term_max_mm': {c: round(v * 1e3, 2) for c, v in vt.items()}}


def main():
    os.makedirs(OUT, exist_ok=True)
    v3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v3.json')))
    res = {'r3': [], 's8': [], 'ref': {'pb60n_sha16': PB60N_SHA16, 'v3_s8_T': v3['s8_overlap']['T']}}
    for k in KS:
        r = run_r3(k)
        res['r3'].append(r)
        print(f"R3 ×{k:g}: {r['rc']}" + (f"；段 41 {r['T_seg41']:.2f} s（尾段 {r['T_tail_src']:.2f}→{r['T_tail']:.2f}）；監看 stop {r['offline']['stop']} warn {r['offline']['warn']}；"
                                         f"sha {r['sha16']}" if r['rc'] == 'ok' else ''))
        for ln in r['log']:
            if '速度峰' in ln or '接點' in ln:
                print('   ', ln.strip()[:230])
        if r['rc'] == 'ok':
            m = r['monitor']
            print(f"    監看（改正）stop {m['stop']} warn {m['warn']}；真最小 {m['min_mm']}；速度項最大扣減 {m['v_term_max_mm']}")
    for k in KS:
        s = s8_case(k)
        res['s8'].append(s)
        print(f"S8 ×{k:g}: {s['L_m']:.3f} m，T {s['T']:.2f} s（vc {s['vc']:.4f}、wc {s['wc']:.4f}）；本體速度峰 {s['body_vel_peak']}；世界加速度峰 {s['world_acc_peak']}；"
              f"監看 stop {s['monitor']['stop']} warn {s['monitor']['warn']} 真最小 {s['monitor']['min_mm']}；速度項最大扣減 {s['monitor']['v_term_max_mm']}")
    r1 = res['r3'][0]
    res['repro'] = {'r3_x1_matches_pb60n': r1.get('sha16') == PB60N_SHA16, 's8_x1_T_matches_v3': abs(res['s8'][0]['T'] - v3['s8_overlap']['T']) < 1e-6}
    print('重現核對：', res['repro'])
    json.dump(res, open(os.path.join(OUT, 'ref_scale.json'), 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(os.path.join(OUT, 'ref_scale.json'), os.path.dirname(HERE)))


if __name__ == '__main__':
    raise SystemExit(main())
