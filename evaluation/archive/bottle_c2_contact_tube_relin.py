#!/usr/bin/env python3
"""C2 候選：一次重新線性化核對（Codex 20261102_215118；同一短窗、同一組參數，不掃參數、不反覆迭代，不接入任務、不跑物理）。
口徑同 bottle_c2_contact_tube_check：簡化約束下的探索重解；沒有接觸與滑移物理；TCP 誤差不等於瓶子滑移。
1 線性化中心核對：線性殘差（列在 δ＝0、U＝U0 的值）對同一構型的外部 FK 殘差（K.fk 對 T_seq[k+1]），排除索引、座標、時間對齊錯誤；另報中心構型對模型前推（參考底盤命令＋U0）的差。
2 u_arm_nom＝上一拍可取得的手臂計畫移位一步（末步重複）。首拍來源：20.80 s 拍以紀錄輸入重解同一候選（u_arm_nom 未給）的計畫；不使用未來紀錄。
3 每拍只做一次候選重解。另在同一狀態解一次 u_arm_nom 未給的版本，只供比較線性－FK 差，命令不施加。
4 三者分開判：線性化是否改善（線性－FK 差）／誤差是否降低（生效時刻 TCP）／是否通過管道（原判準，FK 逐軸 ≤ 3 mm＋0.2 mm）。
5 接受流程 flow：accept_fail 或 need_stop 就結束。強制續走診斷 forced：accept_fail 仍施加命令到 need_stop 或窗尾，另列，不混入接受流程。
    python3 evaluation/bottle_c2_contact_tube_relin.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_c2_contact_tube_check as T                # noqa: E402

Z, K, N, N_TUBE = T.Z, T.K, T.N, T.N_TUBE
OUT = os.path.join(HERE, 'results', 'bottle', 'c2_contact_tube_relin.json')
SRC0 = '20.80 s 拍以紀錄輸入重解同一候選（u_arm_nom 未給）的計畫，移位一步'


def shift(r):
    U = np.asarray(r.u_arm_seq, float)
    return np.vstack([U[1:], U[-1:]])


def center_check(i, qp, sp, U0, ck):
    """中心一致性：列的中心殘差、K.fk 外部殘差、模型前推（δ＝0、U0）。"""
    ct = ck['ct']
    qc = np.array(ct['q_center'])
    ec, er = np.array(ct['e_center']), np.array(ct['e_center_rows']).reshape(-1, 3)
    ef = np.array([Z['T_seq'][i][k + 1][:3, 3] - K.fk(qc[k], 'link_tcp')[:3, 3] for k in range(len(qc))])
    q, s = np.asarray(qp, float).copy(), np.asarray(sp, float).copy()
    gb, ga, gp = 0.0, 0.0, 0.0
    for k in range(len(qc)):
        q, s = T.SP.plant_phys_step(q, s, np.r_[Z['xi_ref'][i][k], U0[k]], T.CFG_M, T.C.NPH)
        gb = max(gb, float(np.linalg.norm(q[:2] - qc[k][:2])))
        ga = max(ga, float(np.abs(q[3:] - qc[k][3:]).max()))
        gp = max(gp, float(np.abs(K.fk(q, 'link_tcp')[:3, 3] - K.fk(qc[k], 'link_tcp')[:3, 3]).max()))
    return dict(rows_vs_center_mm=float(np.abs(er - ec).max() * 1e3), center_vs_fk_mm=float(np.abs(ec - ef).max() * 1e3), rollout_base_mm=round(gb * 1e3, 4), rollout_arm_rad=float(ga), rollout_tcp_mm=round(gp * 1e3, 4))


def lin_fk(row):
    L, F = np.array(row['lin_vs_fk_mm']['lin']), np.array(row['lin_vs_fk_mm']['fk'])
    d = L - F
    return dict(axis_max_mm=np.round(np.abs(d).max(axis=0), 3).tolist(), max_mm=round(float(np.abs(d).max()), 3), by_k_mm=np.round(np.abs(d).max(axis=1), 3).tolist(),
                fk_axis_max_mm=np.round(np.abs(F).max(axis=0), 3).tolist(), fk_viol_mm=round(max(float(np.abs(F).max()) - T.E_C * 1e3, 0.0), 3), fk_steps_over=int((np.abs(F).max(axis=1) > T.E_C * 1e3).sum()))


def roll(forced):
    i0 = int(np.argmin(np.abs(T.T - T.T_ROLL0)))
    _, r_prev = T.tick('ct', i0 - 1, Z['qp'][i0 - 1], Z['sp'][i0 - 1], Z['u_prev'][i0 - 1], T.record_path(i0 - 1), stop=False)
    U0, src = shift(r_prev), SRC0
    qp, sp, up = Z['qp'][i0].copy(), Z['sp'][i0].copy(), Z['u_prev'][i0].copy()
    path = T.record_path(i0)
    rows = []
    for i in range(i0, len(T.T)):
        row, r = T.tick('ct', i, qp, sp, up, path, stop=False, u_nom=U0)
        row['u_nom_source'] = src
        ck = r.arm_check or {}
        if 'ct' in ck:
            row['center'] = center_check(i, qp, sp, U0, ck)
        if 'lin_vs_fk_mm' in row:
            row['lin_fk'] = lin_fk(row)
            base, _ = T.tick('ct', i, qp, sp, up, path, stop=False)
            if 'lin_vs_fk_mm' in base:
                row['same_state_u0_none'] = dict(lin_fk=lin_fk(base), u=base['u'], cls=base['class'])
        rows.append(row)
        if row['class'] == 'need_stop' or (row['class'] != 'feasible_forward' and not forced) or T.T[i] >= T.WIN[1]:
            break
        u9 = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]
        tt = float(T.T[i]) + T.TAU
        for _ in range(T.C.NPH):
            qp, sp = T.SP.plant_phys_step(qp, sp, u9, T.CFG_M, 1)
            tt += T.C.PHYS
            path.append((tt, qp.copy()))
        up = u9
        U0, src = shift(r), '上一拍候選計畫移位一步'
    return rows


def main():
    flow, forced = roll(False), roll(True)
    conflict = [r for r in forced if any(abs(r['t'] - x) < 1e-6 for x in T.T_CONFLICT)]
    verdict = dict(flow_end=dict(t=flow[-1]['t'], cls=flow[-1]['class'], why=flow[-1]['why']), flow_ticks=len(flow), conflict_pass_flow=bool(len(flow) >= len(T.T_CONFLICT) and all(r['class'] == 'feasible_forward' for r in flow[:len(T.T_CONFLICT)])),
                   lin_fk_max_u_nom=[r.get('lin_fk', {}).get('max_mm') for r in conflict], lin_fk_max_u0_none=[(r.get('same_state_u0_none') or {}).get('lin_fk', {}).get('max_mm') for r in conflict],
                   fk_axis_max=[max(r['lin_fk']['fk_axis_max_mm']) if 'lin_fk' in r else None for r in conflict], tcp_eff=[r['tcp_eff_norm_mm'] for r in conflict])
    json.dump(dict(run=T.RUN, ruling='reviews/20261102_215118_reply.md', scope='簡化約束下的探索重解；無接觸與滑移物理；n＝1 會遇；forced 是強制續走診斷，不是接受流程', judge=T.JUDGE, verdict=verdict, flow=flow, forced=forced),
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('接受流程', len(flow), '拍，結束', verdict['flow_end'], '通過' if verdict['conflict_pass_flow'] else '未過')
    for r in forced:
        lf, b = r.get('lin_fk') or {}, (r.get('same_state_u0_none') or {}).get('lin_fk') or {}
        print(f"{r['t']:.2f} {r['class']:18s} why {r['why']} u {r['u']} eps0 {r['eps0']} TCP生效 {r['tcp_eff_norm_mm']} 線性-FK {lf.get('axis_max_mm')}（未給 {b.get('axis_max_mm')}） FK逐軸 {lf.get('fk_axis_max_mm')} 超出 {lf.get('fk_viol_mm')} 超出步數 {lf.get('fk_steps_over')} h {(r.get('h_plan') or {}).get('dyn_min')}/{(r.get('h_plan') or {}).get('static_min')} ms {r['solve_ms']} bind {(r.get('ct_rows') or {}).get('bind')} 中心 {r.get('center')}")
    print(verdict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
