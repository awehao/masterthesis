#!/usr/bin/env python3
"""往後拉開 48° 的換抓參考包 pb1（Codex 20261028_195306：mode 3／4 由新末態更新，mode 5–8 演算法與候選沿用、須接得上新 mode 4 終點）。
起點：door_wide_w1f_60_3（正面抓法、往後拉開）門角最接近 Q0 的實測物理步（底盤、手臂、手指）。規劃門角取 Q0＝ref v4 的 q_ref 47.888663°
（實測差 ≤ 0.01°，揭露），使 mode 5–8（B1、內側抓取 φ＋30°、拉關、放手）以同一門角、同一後端重算而與 ref v5 一致（逐點核對，不只看門角相同）。
  P1 放手列（stage release_outer）：張爪 3 點 → 沿起點工具 −z 退 0.06 m（12 點、全身最小範數 PI.track）→ 手臂關節內插到 STOW（40 點、底盤固定）；
     每點以 replan_v2.margins（full 後端＋PI.check）核名目門角 Q0（張爪／退開目標 out、收臂 None）。r2：r1 對 Q0±0.7° 固定手臂只轉門 ⇒ 閉爪第 1 點
     夾爪殼穿入橫桿 1.7 mm（夾持中門與手剛體相連，此情境不成立；r1 輸出改名 *_r1_fixedarm_dq.json 保留）；±0.7° 改由 P3 retime_v7 三門角發布序列（mode 3 隨門剛體轉）核；
  P2 重規劃：door_regrasp_replan_v8.run_variant(φ＋30°)，只把放手列來源換成 P1（N0／N1／M 段整機 A*＋門角包絡／經過點規則不變）；
  P3 時間化：door_regrasp_retime_v7（起點列＝W1 實測、手指＝實測開度；其餘規則不變）；
  P4 建包：door_regrasp_ref_build_v4（外側抓取基準改正面 phf_01_M）→ door_regrasp_ref_build_v5（加 mode 7／8；銜接核 ≤ 1e-9）；
  另核 mode 5–8 與 ref v5 的構型差（最大絕對值）。
輸出（皆不覆寫）：results/door/door_regrasp_pb1_release.json、door_regrasp_replan_pb1.json、door_regrasp_retime_pb1.json、door_regrasp_ref_pb1_v4.json、door_regrasp_ref_pb1.json。

    python3 -u evaluation/door_regrasp_pb1.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_regrasp_replan_v2 as R2                    # noqa: E402

RD = os.path.join(HERE, 'results', 'door')
RUN = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3', 'room_run.json')
Q0 = 47.888663359359974
DQ = (0.0,)                                            # r2：只核名目（見說明）
OUTER_GB = 'evaluation/runs/phf_01_M/task.json'
F = {k: os.path.join(RD, f) for k, f in (('rel', 'door_regrasp_pb1_release.json'), ('replan', 'door_regrasp_replan_pb1.json'), ('retime', 'door_regrasp_retime_pb1.json'),
                                          ('v4', 'door_regrasp_ref_pb1_v4.json'), ('v5', 'door_regrasp_ref_pb1.json'))}


def worst(x, q, qf, tgt):
    best = None
    for dq in DQ:
        ok, m, det = R2.margins(x, q + dq, qf, tgt, 'release_outer')
        if not ok:
            return False, dict(det, dq_deg=dq)
        best = m if best is None else min(best, m)
    return True, best


def release():
    r = json.load(open(RUN))
    S, C = r['steps'], r['steps_cols']
    I = {c: i for i, c in enumerate(C)}
    k = min(range(len(S)), key=lambda i: abs(math.degrees(S[i][I['door_q_rad']]) - Q0))
    s = S[k]
    x = np.r_[np.asarray(s[I['base_xyth']], float), np.asarray(s[I['q_arm_meas']], float)]
    qf0 = [float(v) for v in s[I['q_finger']]]
    start = {'sim_t': s[1], 'step': s[0], 'base': x[:3].tolist(), 'qa': x[3:].tolist(), 'q_deg': Q0, 'q_deg_meas': math.degrees(s[I['door_q_rad']]),
             'q_finger': qf0, 'finger_contact_n': s[I['finger_contact_n']], 'source': os.path.relpath(RUN, HERE)}
    rows, mins = [], {}
    for j in range(3):
        qf = float(np.mean(qf0)) + (PI.OPEN - float(np.mean(qf0))) * (j + 1) / 3
        ok, m = worst(x, Q0, qf, 'out')
        if not ok:
            return start, None, {'stage': 'open', 'j': j, **m}
        mins['open'] = min(mins.get('open', 9), m)
        rows.append({'stage': 'release_outer', 'x': x.tolist(), 'q': Q0, 'qf': qf})
    T0 = PI.C.K.fk(list(x), 'link_tcp')
    for j in range(12):
        tr = []
        xn, why = PI.track(x, [PI.tz(T0, -0.06 * (j + 1) / 12)], lambda _: Q0, PI.OPEN, 'release_outer', tr, None)
        if why:
            return start, None, {'stage': 'retreat', 'j': j, 'why': why}
        ok, m = worst(xn, Q0, PI.OPEN, 'out')
        if not ok:
            return start, None, {'stage': 'retreat', 'j': j, **m}
        mins['retreat'] = min(mins.get('retreat', 9), m)
        x = xn
        rows.append({'stage': 'release_outer', 'x': x.tolist(), 'q': Q0, 'qf': PI.OPEN})
    qa0 = x[3:].copy()
    for j in range(1, 41):
        xn = x.copy()
        xn[3:] = (1 - j / 40) * qa0 + (j / 40) * np.asarray(TJ.STOW, float)
        ok, m = worst(xn, Q0, PI.OPEN, None)
        if not ok:
            return start, None, {'stage': 'stow', 'j': j, **m}
        mins['stow'] = min(mins.get('stow', 9), m)
        rows.append({'stage': 'release_outer', 'x': xn.tolist(), 'q': Q0, 'qf': PI.OPEN})
    return start, rows, mins


def main():
    for k, p in F.items():
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    t0 = time.time()
    start, rows, info = release()
    json.dump({'schema': 'door_regrasp_pb1_release/1', 'ok': rows is not None, 'start': start, 'dq_env_deg': list(DQ),
               ('min_norm_margin' if rows else 'block'): info, 'rows': rows or []}, open(F['rel'], 'w'), ensure_ascii=False, indent=1, default=float)
    print('P1 放手列', 'OK' if rows else '不成立', json.dumps(info, ensure_ascii=False, default=float), f'起點 t {start["sim_t"]:.2f} q {start["q_deg_meas"]:.3f}°', flush=True)
    if rows is None:
        return
    import door_regrasp_replan_v8 as RV8
    RV8.R2.SRC = F['rel']
    rp = RV8.run_variant(30.0, F['replan'])
    print('P2 重規劃', rp.get('ok'), rp.get('block'), json.dumps(rp.get('N2'), ensure_ascii=False, default=float)[:300], flush=True)
    if not rp.get('ok'):
        return
    import door_regrasp_retime_v7 as RT7
    RT7.SRC, RT7.REL, RT7.OUT, RT7.F0 = F['replan'], F['rel'], F['retime'], tuple(start['q_finger'])
    RT7.main()
    if not json.load(open(F['retime'])).get('ok'):
        print('P3 時間化不成立', flush=True)
        return
    import door_regrasp_ref_build_v4 as RB4
    RB4.SRC, RB4.REPLAN, RB4.OUTER_GB, RB4.OUT = F['retime'], F['replan'], OUTER_GB, F['v4']
    RB4.main()
    import door_regrasp_ref_build_v5 as RB5
    RB5.V4, RB5.OUT = F['v4'], F['v5']
    RB5.main()
    new, old = json.load(open(F['v5'])), json.load(open(os.path.join(RD, 'door_regrasp_ref_v5.json')))
    cmp = {}
    for sg in ('5', '6', '7', '8'):
        a, b = new['segments'][sg], old['segments'][sg]
        same_n = len(a['t']) == len(b['t'])
        cmp[sg] = {'n_new': len(a['t']), 'n_old': len(b['t']), 'T_new': a['t'][-1], 'T_old': b['t'][-1],
                   'max_abs_x_end': float(np.max(np.abs(np.r_[a['base'][-1], a['arm'][-1]] - np.r_[b['base'][-1], b['arm'][-1]]))),
                   'max_abs_x_all': float(np.max(np.abs(np.c_[a['base'], a['arm']] - np.c_[b['base'], b['arm']]))) if same_n else None}
    print('mode 5–8 對 ref v5', json.dumps(cmp, ensure_ascii=False), flush=True)
    print({k: round(v['t'][-1], 2) for k, v in new['segments'].items()}, f'牆鐘 {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
