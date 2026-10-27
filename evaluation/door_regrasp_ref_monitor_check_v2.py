#!/usr/bin/env python3
"""換抓「實際發布參考」的預測淨空與運動限值核對 v2（Codex r54 §4）。對象＝door_regrasp_ref_v2 經 door_regrasp_policy3.RefRuntime3 產生的構型序列。
序列：mode 3（名目：起始實測＝參考起點）→ 交界保持 1.0 s（等待提交；參考不推進）→ mode 4（含構型空間起始銜接）→ 保持 1.0 s → mode 5（含門角修正）→ 保持 1.0 s → mode 6。
r3（r2 結果改名 *_r2_jumpbugs.json）：r2 在 mode 4 第一個樣本人為製造跳變（B：偏差只加在 mode 4 起點；Q：門角在交界突變）——
  改為 B：偏差自 mode 3 起恆定、mode 4 起點＝mode 3 實際終點；Q：門角整段偏 Δq，mode 3 夾持構型隨門繞鉸鏈剛體轉，mode 4 以轉後終點做起始銜接（範圍外即起始被拒＝安全停止）。
r2（測試錯誤修正，r1 結果改名 *_r1_testbugs.json）：B 角點改取上限 0.999 倍；Q 的門角偏差只套 mode 4 起（mode 3 仍夾外側把手，構型隨門，不能以參考構型配偏門角）。
情境：N 名目（Δ0＝0、Δq＝0）；B 起始銜接角點：mode 4 起點 Δ0＝(±0.02/√2, ±0.02/√2, ±2° 偏航) × 手臂全軸 ±0.05 rad（16 組；底盤合偏差 0.02 m＝登錄上限）；
  Q 門角 Δq＝±1°（mode 5／6）。mode 3 只核名目（其目標相對實測 TCP，非名目時構型需 IK，另由閉迴路核）。
每 20 Hz 樣本：door_k3m_geom_monitor_v3 後端（目標 mode 3 out、4 None、5 None→距預抓取 ≤ 5 mm 後 in、6 in），接近率＝相鄰樣本差分（**跨交界不重置**），
  預測 d − max(接近率, v_floor)·0.29 − ds_err ≥ d_min＋warn（1 mm）；分報違規樣本點數與各類違規數。
  運動：發布構型的相鄰差分速度（底盤車體座標每軸、偏航、手臂）與二階差分加速度對求解器上限（0.035255、0.1999、0.9999；0.5、2.0、19.984）的比值。

    python3 -u evaluation/door_regrasp_ref_monitor_check_v2.py   # 寫 results/door/door_regrasp_ref_monitor_check_v2.json（不覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_policy3 as P3                      # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v2.json')))
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_monitor_check_v2.json')
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
CL = ('nonfinger', 'finger', 'base', 'self')
T_PLAN, WARN, DT, HOLD_S = 0.29, 0.001, 0.05, 1.0
VLIM = np.array([0.035255, 0.035255, 0.1999] + [0.9999] * 6)
ALIM = np.array([0.5, 0.5, 2.0] + [19.984] * 6)
KIN = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))


def rot_cfg(x, a):
    """構型繞鉸鏈軸剛體轉 a（rad）：底盤位置旋轉、偏航加 a、手臂不變。"""
    c, s_ = math.cos(a), math.sin(a)
    y = x.copy()
    y[0], y[1], y[2] = c * x[0] - s_ * x[1], s_ * x[0] + c * x[1], x[2] + a
    return y


def sequence(d0_4=None, dq_deg=0.0):
    rr = P3.RefRuntime3(json.loads(json.dumps(PKG)), KIN)
    seq = []
    s3 = PKG['segments']['3']
    off = np.zeros(9) if d0_4 is None else d0_4
    for i in range(len(s3['t'])):                                       # r3：mode 3 夾持中構型隨門剛體轉（−Δq）並帶恆定偏差（B 情境）
        seq.append((3, rot_cfg(np.r_[s3['base'][i], s3['arm'][i]], -math.radians(dq_deg)) + off, float(s3['qf'][i])))
    for _ in range(int(HOLD_S / DT)):
        seq.append((3, seq[-1][1].copy(), seq[-1][2]))
    x_start4 = seq[-1][1].copy()                                        # r3：mode 4 起點＝mode 3 實際終點（不另加跳變）
    s4 = PKG['segments']['4']
    ok, why, det = rr.start(4, x_start4[:3], x_start4[3:], PKG['q_ref_deg'] + dq_deg, None)
    if not ok:
        return None, why
    for i in range(len(s4['t'])):
        seq.append((4, rr.cfg(4, i), float(s4['qf'][i])))
    for _ in range(int(HOLD_S / DT)):
        seq.append((4, seq[-1][1].copy(), seq[-1][2]))
    s5 = PKG['segments']['5']
    x5 = seq[-1][1]
    ok, why, det = rr.start(5, x5[:3], x5[3:], PKG['q_ref_deg'] + dq_deg, None)
    if not ok:
        return None, why
    for i in range(len(s5['t'])):
        seq.append((5, rr.cfg(5, i), float(s5['qf'][i])))
    for _ in range(int(HOLD_S / DT)):
        seq.append((5, seq[-1][1].copy(), seq[-1][2]))
    s6 = PKG['segments']['6']
    for i in range(len(s6['t'])):
        seq.append((6, seq[-1][1].copy(), float(s6['qf'][i])))
    return seq, rr


def check(seq, rr, dq_deg, fn):
    q_rel = q = math.radians(PKG['q_ref_deg'] + dq_deg)                  # r3：門角整段偏 Δq（mode 3 構型已隨門轉）
    pre = P3.P1.Rz4(-math.radians(dq_deg)) @ np.asarray(PKG['pregrasp_T'], float).reshape(4, 4)
    prev, reached = None, False
    vpts, vcls, first = 0, {c: 0 for c in CL}, None
    mins = {c: {'d': math.inf, 'd_pred': math.inf} for c in CL}
    X = np.array([x for _, x, _ in seq])
    for j, (k, x, qf) in enumerate(seq):
        if k == 5 and not reached:
            reached = float(np.linalg.norm(KIN.fk(x, 'link_tcp')[:3, 3] - pre[:3, 3])) <= 0.005
        fn.set_target({3: 'out', 4: None, 5: ('in' if reached else None), 6: 'in'}[k])
        d = fn(x, q_rel if k == 3 else q, qf)
        bad = False
        for c in CL:
            dc = float(d[c][0])
            rate = 0.0 if prev is None else max(0.0, (prev[c] - dc) / DT)
            dp = dc - max(rate, VF[c]) * T_PLAN - DS[c]
            mins[c]['d'], mins[c]['d_pred'] = min(mins[c]['d'], dc), min(mins[c]['d_pred'], dp)
            if dp < DMIN[c] + WARN:
                vcls[c] += 1
                bad = True
                if first is None:
                    first = {'j': j, 'mode': k, 'class': c, 'd_mm': dc * 1e3, 'd_pred_mm': dp * 1e3, 'rate': rate, 'who': d[c][1]}
        vpts += int(bad)
        prev = {c: float(d[c][0]) for c in CL}
    V = np.diff(X, axis=0) / DT
    Vb = np.array([np.r_[math.cos(X[i, 2]) * V[i, 0] + math.sin(X[i, 2]) * V[i, 1], -math.sin(X[i, 2]) * V[i, 0] + math.cos(X[i, 2]) * V[i, 1], V[i, 2:]]
                   for i in range(len(V))])
    A = np.diff(Vb, axis=0) / DT
    return {'n': len(seq), 'viol_points': vpts, 'viol_by_class': vcls, 'first': first,
            'min_mm': {c: {'d': mins[c]['d'] * 1e3, 'd_pred': mins[c]['d_pred'] * 1e3} for c in CL},
            'v_ratio_max': float(np.max(np.abs(Vb) / VLIM)), 'a_ratio_max': float(np.max(np.abs(A) / ALIM))}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    fn = GM3.backend_v3(self_mode='full')
    res = {}
    cases = [('N', None, 0.0), ('Q-1', None, -1.0), ('Q+1', None, 1.0)]
    r2 = 0.999 * 0.02 / math.sqrt(2)                      # r2：角點取上限 0.999 倍（r1 恰在上限，浮點略超 ⇒ 全被拒）
    for sx, sy, sw, sa in itertools.product((1, -1), (1, -1), (1, -1), (1, -1)):
        cases.append((f'B{sx:+d}{sy:+d}{sw:+d}{sa:+d}', np.r_[sx * r2, sy * r2, sw * math.radians(0.999 * 2.0), [sa * 0.999 * 0.05] * 6], 0.0))
    for name, d0, dq in cases:
        seq, rr = sequence(d0, dq)
        if seq is None:
            res[name] = {'start_rejected': rr, 'note': '起始銜接範圍外 ⇒ 任務端於 mode 4 起點停止並報偏差（安全停止，非違規）'}
            print(name, '起始被拒', rr, flush=True)
            continue
        r = check(seq, rr, dq, fn)
        res[name] = r
        print(f"{name}：違規點 {r['viol_points']}（{r['viol_by_class']}）、速度比 {r['v_ratio_max']:.3f}、加速度比 {r['a_ratio_max']:.3f}；首 {r['first']}", flush=True)
    out = {'schema': 'door_regrasp_ref_monitor_check/2', 'T_plan_s': T_PLAN, 'hold_s': HOLD_S, 'cases': res,
           'all_ok': all(('start_rejected' in v) or (v.get('viol_points') == 0 and v.get('v_ratio_max', 9) <= 1.0 and v.get('a_ratio_max', 9) <= 1.0)
                         for v in res.values()),
           'n_start_rejected': sum(1 for v in res.values() if 'start_rejected' in v)}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('all_ok', out['all_ok'], flush=True)


if __name__ == '__main__':
    main()
