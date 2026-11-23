#!/usr/bin/env python3
"""V_HIGH 接近序列的停止核對（Codex 20261102_023000 第 4 點：依實際停止模型，不以「關節直線構型都取樣過」代替；假世界閉迴路，非物理）。
在 test_bottle_w1_segtarget_vhigh 的閉迴路中（控制器＝bottle_w1_segtarget_vhigh_ctrl 的子類，只加紀錄），接近段（CRUISE、未漸變：TUCK／U／H／R／DONE）每個控制拍呼叫
控制器自己的 _stop_rest(q)：在途命令佇列走完的構型 qp、立即停止點 q_rest（底盤各軸 A_BASE、手臂 A_ARM 有界減速＋穩態偏移）、下一拍才停的 q_next（照 MPC 計畫走一拍再減速）；
連同實測 q 共四種構型，核：第一桌（Z1）與第二桌（桌面＋桌腳）、場景方塊、瓶（P1、Z1）、自碰（開爪）、關節餘裕。底盤圓盤對環境由導航 CBF／Shield 另管，不在此。
門檻：桌／方塊／瓶 ≥ 0.05 m（同視點庫）；自碰不低於已驗證名目展臂開爪最小值（8.017 mm，Codex 023000 重算；與閉爪 8.39 mm 分開）；關節餘裕 ≥ 0.05。
    python3 evaluation/bottle_vhigh_stop_check.py 0.60   （或 0.75：候選不送，接近段走完到 x_blend 以 no_fix 停止）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z1 = sys.argv[1] if len(sys.argv) > 1 else '0.60'
sys.argv = [sys.argv[0], Z1, 'nom' if float(Z1) < 0.7 else 'never']
sys.path.insert(0, HERE)
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402  （依 argv 的 Z1 在匯入時改第一桌幾何）
import bottle_branch as BB                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402

SELF_FLOOR, CLEAR = 0.008017, 0.05
OBST = (np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))
Z = float(Z1)
TB = [(c_, s_) for _, c_, s_ in G.table_boxes()] + [(c_, s_) for _, c_, s_ in TP.table2_boxes()]
BC, BS = np.array([0.0, 0.42, Z + 0.10]), np.array([0.03, 0.03, 0.20])
REC = []


def clear(q9):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for c_, s_ in TB), obst=float(G.box_dist(P, *OBST).min()), bottle=float(G.box_dist(P, BC, BS).min()),
                self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=BB.margin(q9[3:]))


class Rec(TV.VH.W1PlaceController):
    def step(self, t, q, **kw):
        out = super().step(t, q, **kw)
        if self.st.phase == 'CRUISE' and not self.blending and self.judge.stop is None and self.n_tick % 2 == 0 and np.isfinite(q).all() and self.vh_seq['ph'] != 'TUCK':
            qp, q_rest, q_next = self._stop_rest(np.asarray(q, float))
            row = dict(t=round(float(t), 3), ph=self.vh_seq['ph'], x=round(float(q[0]), 4))
            for nm, qq in (('meas', q), ('inflight', qp), ('rest', q_rest), ('next', q_next)):
                row[nm] = clear(np.asarray(qq, float))
            REC.append(row)
        return out


def main():
    TV.VH.W1PlaceController = Rec
    c, f, r, st = TV.run(**TV.CASES[sys.argv[2]][1])
    worst, where = {}, {}
    for row in REC:
        for nm in ('meas', 'inflight', 'rest', 'next'):
            for k, v in row[nm].items():
                key = f'{nm}:{k}'
                if v < worst.get(key, 9.9):
                    worst[key], where[key] = v, (row['t'], row['ph'], row['x'])
    lim = dict(table=CLEAR, obst=CLEAR, bottle=CLEAR, self=SELF_FLOOR - 1e-6, margin=0.05)
    fails = sorted(k for k, v in worst.items() if v < lim[k.split(':')[1]])
    by_ph = {}
    for row in REC:
        b = by_ph.setdefault(row['ph'], dict(n=0, min_table=9.9, min_obst=9.9, min_self=9.9))
        b['n'] += 1
        for nm in ('meas', 'inflight', 'rest', 'next'):
            b['min_table'] = min(b['min_table'], row[nm]['table'])
            b['min_obst'] = min(b['min_obst'], row[nm]['obst'])
            b['min_self'] = min(b['min_self'], row[nm]['self'])
    out = dict(z1=Z, case=sys.argv[2], n=len(REC), outcome=(r['stop'] or {}).get('why') or ('done' if r['mission_done'] else None), worst={k: round(v, 5) for k, v in worst.items()},
               where=where, fails=fails, by_phase={k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in by_ph.items()}, limits=lim, seq=r['vhigh_seq']['log'],
               note='控制器自身 _stop_rest（在途命令、A_BASE／A_ARM 有界減速、下一拍停止點）；假世界閉迴路，非物理')
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', f'vhigh_stop_check_z{int(round(Z * 100)):03d}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('where',)}, ensure_ascii=False, default=float))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
