#!/usr/bin/env python3
"""gmpc_wb 短閉迴路 r3 軌跡的幾何核對（Codex 20261031_192829 (1)：用實際軌跡，不只三個目標位姿的離散 IK）。
純運動學、無 Isaac。取 bottle_offline_wb 的逐週期紀錄，對三種構型計算距離：
  meas＝實測關節與底盤；sp＝實測底盤＋手臂設定點；pred＝求解時的延遲前推狀態（qp, sp）以回傳手臂序列與 QP 底盤序列（ξ_ref＋δξ）沿設定點模型前推到 k＝5、10、20。
項目：非手指連桿／手指 ↔ 桌面、桌腳、瓶（瓶在桌上時）；底盤圓盤 ↔ 桌腳；手臂自碰（door_offline_check_v1 全點雲與配對規則）；
  持瓶（LIFT 起固定相對 TCP）↔ 桌面、桌腳、非手指連桿。環境距離逐週期算；自碰每 0.5 s 一點，相位接點前後 ±0.3 s 逐週期。
結果只代表此模型軌跡與此場景；未含 S1 障礙。手指↔瓶在 INSERT 後是指定接觸，不列。
    python3 evaluation/bottle_wb_geom.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_offline_wb as W                           # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

C, K, OPEN_PH = G.C, W.K, ('CRUISE', 'PREGRASP', 'INSERT')
BZ0, BOT = G.S1_TABLE['top_z'], G.BOTTLE
PRED_K = (5, 10, 20)
DT, NPH = W.DT, W.NPH


def pred_cfgs(e, cfg):
    """回傳 [(k, q9)]：沿回傳手臂序列與 QP 底盤序列前推（手臂序列補償的是後者，用參考速度會造出假穿透）。"""
    q, s, out = e['qp'].copy(), e['sp'].copy(), []
    U, V = e['u_arm_seq'], e['xi_seq']
    for j in range(max(PRED_K)):
        u = np.r_[V[j], U[j]]
        for _ in range(NPH):
            q, s = W.S.plant_phys_step(q, s, u, cfg, 1)
        if j + 1 in PRED_K:
            out.append((j + 1, q.copy()))
    return out


def env_scan(q, ph, t, q_f, mins, boxes, legs, kind, bottle_on_table, Tb=None, BP=None):
    def upd(k, d):
        key = f'{kind}:{k}'
        if key not in mins or d < mins[key][0]:
            mins[key] = (float(d), round(float(t), 2), str(ph))
    Wl = G.link_world(q, q_f)
    for n, P in Wl.items():
        fing = n in C.FINGERS
        for nm, c, sz in boxes:
            upd(('finger' if fing else 'nonfinger') + '↔' + ('table_top' if nm == 'table_top' else 'leg'), G.box_dist(P, c, sz).min())
        if bottle_on_table:
            d = G.cyl_dist(P, BOT['x'], BOT['y'], BZ0, BZ0 + BOT['h'], BOT['r']).min()
            if not fing:
                upd('nonfinger↔bottle', d)
            elif ph in ('CRUISE', 'PREGRASP'):
                upd('finger↔bottle（INSERT 前）', d)
    for nm, c, sz in legs:
        dx, dy = max(abs(q[0] - c[0]) - sz[0] / 2, 0.0), max(abs(q[1] - c[1]) - sz[1] / 2, 0.0)
        upd('base_disc↔leg', float(np.hypot(dx, dy)) - C.RB)
    if Tb is not None:
        Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
        for nm, c, sz in boxes:
            upd('held_bottle↔' + ('table_top' if nm == 'table_top' else 'leg') + f'（{ph}）', G.box_dist(Pw, c, sz).min())
        for n, P in Wl.items():
            if n not in C.FINGERS and n != 'uflite_gripper_link':
                Pb = (np.linalg.inv(Tb) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
                upd('held_bottle↔nonfinger', G.cyl_dist(Pb, 0.0, 0.0, 0.0, BOT['h'], BOT['r']).min())


def scan(v, pa, label):
    tr = []
    r = W.run(v, pa, 1.0, label=label, trace=tr)
    cfg = W.BO.make_cfg(1.0)
    boxes = G.table_boxes()
    legs = [b for b in boxes if b[0].startswith('leg')]
    BP = G.bottle_points()
    ph = [e['phase'] for e in tr]
    near = set()
    for i in range(1, len(ph)):
        if ph[i] != ph[i - 1]:
            near |= set(range(max(i - 6, 0), min(i + 7, len(ph))))
    mins, rel, self_min, n_self = {}, None, (float('inf'), None, None, None), 0
    for i, e in enumerate(tr):
        p, t = e['phase'], e['t']
        q_f = G.Q_OPEN if p in OPEN_PH else 0.0038
        held = p in ('LIFT', 'CARRY')
        q_sp = np.r_[e['q'][:3], e['s']]
        Tb = None
        if held:
            Tt = K.fk(e['q'], 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [BOT['x'], BOT['y'], BZ0]
                rel = np.linalg.inv(Tt) @ T0
            Tb = Tt @ rel
        env_scan(e['q'], p, t, q_f, mins, boxes, legs, 'meas', not held, Tb, BP)
        env_scan(q_sp, p, t, q_f, mins, boxes, legs, 'sp', not held, (K.fk(q_sp, 'link_tcp') @ rel) if held else None, BP)
        if i % 5 == 0 or i in near:
            for k, qk in pred_cfgs(e, cfg):
                env_scan(qk, p, t, q_f, mins, boxes, legs, 'pred', not held, (K.fk(qk, 'link_tcp') @ rel) if held else None, BP)
        if i % 10 == 0 or i in near:
            for tag, qq in (('meas', e['q']), ('sp', q_sp)):
                d, who = C.self_clear(qq, q_f)
                n_self += 1
                if d < self_min[0]:
                    self_min = (float(d), round(float(t), 2), f'{tag} {p}', who)
    return r, mins, self_min, n_self


def main():
    res = []
    for v, pa, lab in ((0.03, 1.0, '正常'), (0.05, 1.0, '正常'), (0.07, 1.0, '正常'), (0.05, 0.6, '伺服慢未知')):
        r, mins, sm, ns = scan(v, pa, lab)
        print(f"\n{lab} v {v}：{r['result']} 事件 {r['events']}；自碰最小 {sm[0] * 1e3:.1f} mm @ {sm[1]} s {sm[2]} {sm[3]}（{ns} 點）")
        for k, (d, t, p) in sorted(mins.items()):
            expected = 'held_bottle↔table_top（LIFT）' in k               # 抬升起點瓶子本來就貼桌
            fl = '' if expected else ('  ← 穿透' if d < 0 else ('  ← <10 mm' if d < 0.01 else ''))
            print(f'   {k:<42} {d * 1e3:8.1f} mm @ {t:6.2f} s {p}{fl}', flush=True)
        res.append(dict(label=lab, v_c=v, result=r['result'], events=r['events'], self_min=dict(mm=round(sm[0] * 1e3, 2), t=sm[1], where=sm[2], pair=sm[3], n=ns),
                        min={k: dict(mm=round(d * 1e3, 2), t=t, phase=p) for k, (d, t, p) in mins.items()}))
    out = os.path.join(HERE, 'results', 'bottle', 'wb_geom_r3.json')
    json.dump(dict(scene=dict(table=G.S1_TABLE, bottle=BOT), note='模型軌跡；無 S1 障礙；held_bottle↔table_top（LIFT）起點貼桌為預期', runs=res), open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    sys.exit(main())
