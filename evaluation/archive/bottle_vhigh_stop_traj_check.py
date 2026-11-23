#!/usr/bin/env python3
"""V_HIGH 接近序列的「完整停止軌跡」核對＋控制命令變化率（Codex 20261102_025919；假世界閉迴路，非物理；不改控制器、不放寬門檻）。
停止軌跡：接近段（U／H／R）每個控制拍，在 _advance_inner（控制拍上、本拍求解前；相位 0，同 compactcarry21 收近閘的呼叫點與狀態）呼叫
  compactcarry21.stop_traj(q, s, pipe, u_last, cfg, extra_ticks)：extra_ticks＝0「現在停止」、1「延後一拍停止」。軌跡含在途命令、設定點積分、
  底盤 A_BASE／手臂 A_ARM 有界減速與收斂尾段（連續 SETTLE_STEPS 物理步無命令才結束，上限 2000 步），每 SAMPLE_EVERY（4）物理步＝0.04 s 取一個構型。
  每個構型核：第一桌（Z1）與第二桌、場景方塊、瓶（P1）≥ 0.05 m；自碰（開爪）≥ 已驗證名目展臂開爪最小值 8.017 mm；關節餘裕 ≥ 0.05；非有限；未停完（到上限仍有命令）。
命令變化率：每個控制拍求解後的 u_last（底盤 3＋手臂 6），相鄰控制拍差分 /0.05 s；報手臂各關節最大值（對 A_ARM）與發生相位、底盤各軸最大值（對 A_BASE），全程與接近段分開。
  （離線 acc_peak 39.19 是 0.01 s 設定點網格的二階差分，不是控制命令加速度。）
    python3 evaluation/bottle_vhigh_stop_traj_check.py 0.60   （0.75：候選不送，接近段走完於 x_blend no_fix）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z1 = sys.argv[1] if len(sys.argv) > 1 else '0.60'
sys.argv = [sys.argv[0], Z1, 'nom' if float(Z1) < 0.7 else 'never']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import test_bottle_place as TP                          # noqa: E402

SELF_FLOOR, CLEAR = 0.008017, 0.05
OBST = (np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))
Z = float(Z1)
TB = [(c_, s_) for _, c_, s_ in G.table_boxes()] + [(c_, s_) for _, c_, s_ in TP.table2_boxes()]
BC, BS = np.array([0.0, 0.42, Z + 0.10]), np.array([0.03, 0.03, 0.20])
N_MAX = 2000 // C21.SAMPLE_EVERY + 1
ST = dict(n=0, n_cfg=0, worst={}, where={}, nonfinite=0, unfinished=0, len_max=0)
U = []


def clear(q9):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for c_, s_ in TB), obst=float(G.box_dist(P, *OBST).min()), bottle=float(G.box_dist(P, BC, BS).min()),
                self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=BB.margin(q9[3:]))


class Rec(TV.VH.W1PlaceController):
    def _advance_inner(self, T_tcp, close_done, q):
        if self.st.phase == 'CRUISE' and not self.blending and self.judge.stop is None and self.vh_seq['ph'] in ('U', 'H', 'R') and np.isfinite(q).all():
            ST['n'] += 1
            for e in (0, 1):
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                ST['len_max'] = max(ST['len_max'], len(Q))
                if len(Q) >= N_MAX:
                    ST['unfinished'] += 1
                if not np.isfinite(Q).all():
                    ST['nonfinite'] += 1
                    continue
                for qq in Q:
                    ST['n_cfg'] += 1
                    for k, v in clear(qq).items():
                        key = f'e{e}:{k}'
                        if v < ST['worst'].get(key, 9.9):
                            ST['worst'][key], ST['where'][key] = v, (round(float(self.t), 3), self.vh_seq['ph'], round(float(q[0]), 4))
        super()._advance_inner(T_tcp, close_done, q)

    def _solve(self, q):
        u = super()._solve(q)
        if u is not None:
            U.append((float(self.t), self.st.phase, self.vh_seq['ph'] if (self.st.phase == 'CRUISE' and not self.blending) else self.st.phase, np.asarray(u, float).copy()))
        return u


def main():
    TV.VH.W1PlaceController = Rec
    c, f, r, st = TV.run(**TV.CASES[sys.argv[2]][1])
    lim = dict(table=CLEAR, obst=CLEAR, bottle=CLEAR, self=SELF_FLOOR - 1e-6, margin=0.05)
    fails = sorted(k for k, v in ST['worst'].items() if v < lim[k.split(':')[1]]) + (['nonfinite'] if ST['nonfinite'] else []) + (['unfinished'] if ST['unfinished'] else [])
    du = []
    for i in range(1, len(U)):
        dt = U[i][0] - U[i - 1][0]
        if dt <= 0.0 or dt > 0.051:
            continue                                               # 只比相鄰控制拍
        du.append((U[i][0], U[i][2], np.abs(U[i][3] - U[i - 1][3]) / dt))

    def mx(rows, sl):
        if not rows:
            return None
        k = int(np.argmax([float(x[2][sl].max()) for x in rows]))
        return dict(max=round(float(rows[k][2][sl].max()), 3), joint=int(np.argmax(rows[k][2][sl])), t=round(rows[k][0], 3), phase=rows[k][1])
    appr = [x for x in du if x[1] in ('TUCK', 'U', 'H', 'R', 'DONE')]
    by_ph = {}
    for x in du:
        b = by_ph.setdefault(x[1], 0.0)
        by_ph[x[1]] = max(b, float(x[2][3:].max()))
    out = dict(z1=Z, case=sys.argv[2], outcome=(r['stop'] or {}).get('why') or ('done' if r['mission_done'] else None), n_ticks=ST['n'], n_cfg=ST['n_cfg'],
               sample_dt_s=C21.SAMPLE_EVERY * C0.PHYS, settle_steps=C21.SETTLE_STEPS, traj_len_max=ST['len_max'], nonfinite=ST['nonfinite'], unfinished=ST['unfinished'],
               worst={k: round(v, 5) for k, v in ST['worst'].items()}, where=ST['where'], limits=lim, fails=fails,
               du_dt=dict(grid_s=B.DT, arm_all=mx(du, slice(3, 9)), arm_approach=mx(appr, slice(3, 9)), base_all=mx(du, slice(0, 3)), arm_max_by_phase={k: round(v, 3) for k, v in by_ph.items()},
                          A_ARM=C0.A_ARM, A_BASE=B.A_BASE.tolist()),
               seq=r['vhigh_seq']['log'], note='compactcarry21.stop_traj（控制拍上、相位 0、現在停／延後一拍停）；命令變化率＝相鄰控制拍 u_last 差分；假世界閉迴路')
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', f'vhigh_stop_traj_z{int(round(Z * 100)):03d}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'where'}, ensure_ascii=False, default=float))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
