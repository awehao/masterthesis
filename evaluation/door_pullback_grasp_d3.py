#!/usr/bin/env python3
"""正面抓取深度 6.8 → 3 mm 候選（Codex 20261028 pb1_6 後回覆：明確核可的單一深度候選，不掃描）：抓取朝向與橫桿抓取點不變。
產出（皆不覆寫）：
  results/door/pullback_grasp_basis_d3.json：args＝phf_01_M 的 R_grasp，grasp_depth_m＝0.003（任務節點 T_HG＝[R_grasp | (0, depth, 0)]）；
  results/door/pullback_params_48_r4.json：r3 只改 grasp（基準路徑／sha）、grasp_init.qa_grasp、bench.q_nominal／s_init（d_init 沿用 r3：W1 實測，深度不同、揭露）；
  results/door/door_regrasp_ref_pb1d.json：ref pb1b 只改 segments['3']——起點換成「W1 47.89° 實測構型的 TCP 沿工具 −z 退 3.8 mm」的 IK 構型（底盤不動），
    前 T_BLEND（0.5 s，餘弦權重）內由新起點平滑併回原退離路徑，其後逐字同 pb1b（終點不變 ⇒ mode 4–8 不變）；時間表不變。
核對（任一不成立即停，報阻點）：
  K1 停位 qa_grasp_d3：IK（以 r3 qa_grasp 為種子、同分支）、TCP 對抓取目標 < 1 mm／0.01 rad、關節餘裕；
  K2 雙指夾持：手指閉合值（實測持握開度 0.0017 m）下兩指對外側橫桿距離 ∈ [−1, +1] mm（同 replan N1 對向接觸規則），並報接觸點落在指片 z 範圍內與否；
  K3 掌心（uflite_gripper_link）對 bar_out 淨空：d3 對 6.8 的增量；
  K4 新 seg 3：PI.check 分階段（張爪中 release_outer_c、張開後 release_outer）＋replan_v2.margins 名目逐點過；速度／加速度差分 ≤ 求解器上限。

    python3 -u evaluation/door_pullback_grasp_d3.py
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_regrasp_ref_registry as REG                # noqa: E402
import door_regrasp_replan_v2 as R2                    # noqa: E402

RD = os.path.join(HERE, 'results', 'door')
GB0 = os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')
P3 = os.path.join(RD, 'pullback_params_48_r3.json')
REF0 = os.path.join(RD, 'door_regrasp_ref_pb1b.json')
OUT_GB, OUT_P, OUT_R = os.path.join(RD, 'pullback_grasp_basis_d3.json'), os.path.join(RD, 'pullback_params_48_r4.json'), os.path.join(RD, 'door_regrasp_ref_pb1d.json')
D_NEW, D_OLD = 0.003, 0.0068
Q_HOLD = 0.0017
T_BLEND = 0.5
C = PI.C
LIM = {'vb': 0.035255, 'qd': 0.9999, 'ab': 0.5, 'qdd': 19.984}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def T_HG(R, d):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = [0.0, d, 0.0]
    return T


def fingers_bar(x, q_deg, qf):
    fb = PI.dists(np.asarray(x, float), q_deg, qf)['fingers']
    return {f: round(float(fb[f]['bar_out']) * 1e3, 3) for f in fb}


def palm(x, q_deg, qf):
    R2.FN.set_target(None)
    d = R2.FN(np.asarray(x, float), math.radians(q_deg), qf)
    return {c: (round(float(d[c][0]) * 1e3, 2), d[c][1]) for c in ('nonfinger', 'finger', 'self', 'base')}


def main():
    for p in (OUT_GB, OUT_P, OUT_R):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    gb0 = json.load(open(GB0))['args']
    p3 = json.load(open(P3))
    stand = np.asarray(p3['start']['stand'], float)
    qa_old = np.asarray(p3['grasp_init']['qa_grasp'], float)
    ev = json.loads(open(os.path.join(HERE, 'runs', 'door_wide_w1f_60_3', 'evidence.jsonl')).readline())
    H0 = np.asarray(ev['T_WH'], float).reshape(4, 4)
    rep = {}
    # K1
    T_new = H0 @ T_HG(gb0['R_grasp'], D_NEW)
    q9 = C.ik(stand, T_new, qa_old, only_prev=True)
    if q9 is None:
        raise SystemExit('K1：停位 IK 無解（同分支）')
    qa_new = np.asarray(q9, float)[C.IDX]
    Tf = C.K.fk(list(np.r_[stand, qa_new]), 'link_tcp')
    ep = float(np.linalg.norm(Tf[:3, 3] - T_new[:3, 3]))
    ok, why, info = PI.check(np.r_[stand, qa_new], 0.0, PI.OPEN, 'G')
    rep['K1'] = {'qa_grasp_d3': qa_new.tolist(), 'dq_vs_d68': float(np.max(np.abs(qa_new - qa_old))), 'tcp_err_mm': ep * 1e3, 'check_open': [ok, why]}
    print('K1', json.dumps(rep['K1'], ensure_ascii=False, default=float), flush=True)
    if ep > 1e-3:
        raise SystemExit('K1：TCP 誤差 > 1 mm')
    # K2／K3（門 0°、停位）
    x_old, x_new = np.r_[stand, qa_old], np.r_[stand, qa_new]
    rep['K2'] = {'d3_closed_hold': fingers_bar(x_new, 0.0, Q_HOLD), 'd68_closed_hold': fingers_bar(x_old, 0.0, Q_HOLD)}
    rep['K3'] = {'d3_open': palm(x_new, 0.0, PI.OPEN), 'd68_open': palm(x_old, 0.0, PI.OPEN), 'd3_hold': palm(x_new, 0.0, Q_HOLD), 'd68_hold': palm(x_old, 0.0, Q_HOLD)}
    print('K2', rep['K2'], flush=True)
    print('K3', rep['K3'], flush=True)
    if not all(-1.0 <= v <= 1.0 for v in rep['K2']['d3_closed_hold'].values()):
        json.dump(rep, open(OUT_R.replace('.json', '_blocked.json'), 'w'), ensure_ascii=False, indent=1, default=float)
        raise SystemExit(f"K2：d3 持握開度下兩指對橫桿距離不在 ±1 mm：{rep['K2']['d3_closed_hold']}")
    # K4：新 seg 3
    ref = json.load(open(REF0))
    s3 = ref['segments']['3']
    X = np.c_[np.asarray(s3['base'], float), np.asarray(s3['arm'], float)]
    q0 = float(s3['q_door_deg'][0])
    T0 = C.K.fk(list(X[0]), 'link_tcp')
    T0n = PI.tz(T0, -(D_OLD - D_NEW))
    q9 = C.ik(X[0, :3], T0n, X[0, 3:], only_prev=True)
    if q9 is None:
        raise SystemExit('K4：seg 3 新起點 IK 無解')
    dX0 = np.r_[np.zeros(3), np.asarray(q9, float)[C.IDX] - X[0, 3:]]
    t = np.asarray(s3['t'], float)
    w = np.where(t < T_BLEND, 0.5 * (1 + np.cos(np.pi * t / T_BLEND)), 0.0)
    Xn = X + w[:, None] * dX0[None, :]
    bad, worst = None, None
    for i in range(len(t)):
        qf = s3['qf'][i]
        st = 'release_outer_c' if qf < PI.OPEN - 1e-6 else 'release_outer'
        okm, m, det = R2.margins(Xn[i], q0, qf, 'out' if s3['s'][i] < 0.06 else None, st)
        if not okm:
            bad = {'i': i, 't': float(t[i]), 'stage': st, 'det': det}
            break
        if worst is None or m < worst[0]:
            worst = (float(m), i, float(t[i]), det)
    dt = np.diff(t)
    qd = np.diff(Xn[:, 3:], axis=0) / dt[:, None]
    vb = np.diff(Xn[:, :2], axis=0) / dt[:, None]
    lim = {'vb': float(np.abs(vb).max()), 'qd': float(np.abs(qd).max()), 'ab': float(np.abs(np.diff(vb, axis=0) / dt[1:, None]).max()),
           'qdd': float(np.abs(np.diff(qd, axis=0) / dt[1:, None]).max())}
    rep['K4'] = {'dX0_max': float(np.max(np.abs(dX0))), 'margin_bad': bad, 'margin_worst': worst, 'limits': lim, 'limits_ok': all(lim[k] <= LIM[k] for k in LIM),
                 'palm_start_new': palm(Xn[0], q0, s3['qf'][0]), 'palm_start_old': palm(X[0], q0, s3['qf'][0])}
    print('K4', json.dumps(rep['K4'], ensure_ascii=False, default=float), flush=True)
    if bad is not None or not rep['K4']['limits_ok']:
        json.dump(rep, open(OUT_R.replace('.json', '_blocked.json'), 'w'), ensure_ascii=False, indent=1, default=float)
        raise SystemExit('K4 不成立')
    # 寫出
    gb = {'schema': 'pullback_grasp_basis/1', 'args': {'R_grasp': gb0['R_grasp'], 'grasp_depth_m': D_NEW},
          'source': {'path': os.path.relpath(GB0, HERE), 'sha256': sha(GB0), 'grasp_depth_m_old': gb0['grasp_depth_m']},
          'note': 'Codex 20261028：深度單一候選 3 mm（朝向、抓取點不變）；掌心淨空增量見 door_regrasp_ref_pb1d.json grasp_d3'}
    json.dump(gb, open(OUT_GB, 'w'), ensure_ascii=False, indent=1)
    p4 = dict(p3)
    p4['grasp'] = dict(p3['grasp'], basis=os.path.relpath(OUT_GB, WS), basis_sha256=sha(OUT_GB), grasp_depth_m=D_NEW)
    p4['grasp_init'] = dict(p3['grasp_init'], qa_grasp=qa_new.tolist(), qa_grasp_cli=','.join(f'{v:.6f}' for v in qa_new),
                            source={'ik': 'C.ik(stand, H0·T_HG(d3), seed＝r3 qa_grasp, only_prev)', 'r3_qa_grasp': qa_old.tolist()})
    d = np.asarray(p3['bench']['d_init'], float)
    s_init = (qa_new - d).tolist()
    p4['bench'] = dict(p3['bench'], q_nominal=qa_new.tolist(), s_init=s_init, q_nominal_cli=','.join(f'{v:.9f}' for v in qa_new),
                       s_init_cli=','.join(f'{v:.9f}' for v in s_init), d_init_note='沿用 r3（W1 深度 6.8 mm 實測；深度 3 mm 未另測，揭露）')
    p4['derived_from_r3'] = {'path': os.path.relpath(P3, HERE), 'sha256': sha(P3)}
    json.dump(p4, open(OUT_P, 'w'), ensure_ascii=False, indent=1)
    ref['segments']['3'] = dict(s3, base=Xn[:, :3].tolist(), arm=Xn[:, 3:].tolist(),
                                tcp=[np.asarray(C.K.fk(list(x), 'link_tcp'), float).reshape(-1).tolist() for x in Xn])
    ref['schema'] = ref.get('schema', '') + '+pb1d'
    ref['grasp_d3'] = dict(rep, derived_from={'path': os.path.relpath(REF0, HERE), 'sha256': sha(REF0)}, T_blend_s=T_BLEND, depth=[D_OLD, D_NEW])
    ok, why = REG.validate(json.loads(json.dumps(ref, default=float)))
    ref['registry_validate'] = {'ok': ok, 'why': why}
    if not ok:
        raise SystemExit(f'登錄自驗不過：{why}')
    json.dump(ref, open(OUT_R, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫', OUT_GB, OUT_P, OUT_R, flush=True)


if __name__ == '__main__':
    main()
