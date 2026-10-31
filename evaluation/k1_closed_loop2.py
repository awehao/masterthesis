#!/usr/bin/env python3
"""步驟 2 離線閉迴路（Codex 20261031_153959：使用各段實際權重）：比較 rt_43 原配置與 roundtrip3 配置。
受控對象與延遲同 k1_closed_loop（手臂設定點模型＋底盤運動學增益 1.0；發布 0.6＋命令 1.0 週期，補償 1.6；無接觸、非 Isaac；d̂ 觀測器未模擬）。
  R3（門段求解端實際 coord：w_vref 1、w_p 50、w_a 0.05、w_qn 0）：
    原＝pb60o 段 41、引導 policy1._coord（世界逐軸 v_cap 0.035、w_cap r3_w_cap 0.19）、框 ×1；
    新＝pb60p 段 41、引導 _coord_body（本體逐軸 r3_v_cap_body 0.0525、w_cap 0.285）、base_box_scale 2.0。
  S8（返程 A 求解端實際 coord：w_vref 0.03、w_p 200、w_a 0.01、w_qn 0.8、q_nom＝列手臂）：引導＝列差分（door_return_task_node.pub_coord，無截斷）、
    TCP 目標＝列 FK（S8 列 _f7＝0，無退離偏移）；原＝v3 S8 前綴、框 ×1；新＝v4 S8 前綴、base_box_scale 2.0。
    python3 evaluation/k1_closed_loop2.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import wgmpc_closed_loop_sim as CL                     # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

DT = 0.05
DOOR = os.path.join(HERE, 'results', 'door')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
OUT = os.path.join(HERE, 'results', 'k1', 'closed_loop2.json')
SETTLE_S = 4.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def r3_rows(pack):
    s = json.load(open(os.path.join(DOOR, pack)))['segments']['41']
    return np.asarray(s['base'], float), np.asarray(s['arm'], float), np.asarray(s['base_v'], float)


def s8_rows(pack):
    R = json.load(open(os.path.join(DOOR, pack)))
    A = R['A']
    j8 = [i for i, r in enumerate(A) if r['stage'].startswith('S8')][0]
    n = int(R['s8_overlap']['end_row']) + 1
    X = np.array([r['x'] for r in A[j8:j8 + n]], float)
    return X[:, :3], X[:, 3:], None


def run(K, ident, B, Ar, BV, *, scale, coord, guide):
    n = len(B)
    tt = np.arange(n) * DT
    Bu = B.copy()
    Bu[:, 2] = np.unwrap(Bu[:, 2])

    def ref_at(t):
        b = np.array([np.interp(t, tt, Bu[:, c]) for c in range(3)])
        a = np.array([np.interp(t, tt, Ar[:, c]) for c in range(Ar.shape[1])])
        if BV is not None:
            ff = np.array([np.interp(t, tt, BV[:, c]) for c in range(3)])
        else:                                                               # 列差分（任務節點 pub_coord：x_i − x_{i−1}）
            i = min(int(round(t / DT)), n - 1)
            ff = (Bu[i] - Bu[max(i - 1, 0)]) / DT
        return b, a, ff
    cfg = CL.make_cfg(ident, w_s=1e-3, w_a=coord['w_a'], w_s_arm=0.05)
    cfg.w_p, cfg.w_vref, cfg.w_qn = coord['w_p'], coord['w_vref'], coord['w_qn']
    cfg.base_box_scale = scale

    def T_fn(t):
        b, a, _ = ref_at(t)
        return K.fk(np.r_[b, a], 'link_tcp')

    def pre(t, q, c):
        b, a, ff = ref_at(t)
        x, y, th = float(q[0]), float(q[1]), float(q[2])
        cs, sn = math.cos(th), math.sin(th)
        if guide['kind'] == 'ff':
            vw = ff
            c.base_vref = (cs * vw[0] + sn * vw[1], -sn * vw[0] + cs * vw[1], float(vw[2]))
        else:
            e = np.array([b[0] - x, b[1] - y, wrap(b[2] - th)])
            vw = ff + e
            if guide['kind'] == 'world':
                vw = np.array([np.clip(vw[0], -guide['v'], guide['v']), np.clip(vw[1], -guide['v'], guide['v']), np.clip(vw[2], -guide['w'], guide['w'])])
                c.base_vref = (cs * vw[0] + sn * vw[1], -sn * vw[0] + cs * vw[1], float(vw[2]))
            else:
                c.base_vref = (float(np.clip(cs * vw[0] + sn * vw[1], -guide['v'], guide['v'])), float(np.clip(-sn * vw[0] + cs * vw[1], -guide['v'], guide['v'])),
                               float(np.clip(vw[2], -guide['w'], guide['w'])))
        if coord['w_qn'] > 0.0:
            c.arm_q_nom = tuple(float(v) for v in a)
    b0, a0, _ = ref_at(0.0)
    res = CL.run(cfg, K, np.r_[b0, a0], a0.copy(), T_fn, t_end=tt[-1] + SETTLE_S, reach_p=-1.0, reach_r=-1.0,
                 delay_cycles=1.0, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', pre_solve=pre)
    r = res['rec']
    Q, T, ep = np.asarray(r['q']), np.asarray(r['t']), np.asarray(r['ep'])
    on = T <= tt[-1] + 1e-9
    eb = np.array([np.hypot(*(ref_at(t)[0][:2] - q[:2])) for t, q in zip(T, Q)])
    ey = np.array([abs(wrap(ref_at(t)[0][2] - q[2])) for t, q in zip(T, Q)])
    bad = np.where(~on & (ep > 0.005))[0]
    settle = None if (bad.size and bad[-1] == len(T) - 1) else (0.0 if not bad.size else float(T[bad[-1]] - tt[-1] + DT))
    U = np.asarray(r['u_req'])
    pc = lambda z, p: round(float(np.percentile(z, p)) * 1e3, 2)          # noqa: E731
    return {'T_ref_s': round(float(tt[-1]), 2), 'n_fail': int(res['n_fail']), 'n_cycles': int(len(T)),
            'tcp_err_mm': {'p50': pc(ep[on], 50), 'p95': pc(ep[on], 95), 'max': round(float(ep[on].max()) * 1e3, 2)},
            'settle_to_5mm_s': settle, 'base_err_mm': {'p95': pc(eb[on], 95), 'max': round(float(eb[on].max()) * 1e3, 2)},
            'yaw_err_mrad': {'p95': pc(ey[on], 95), 'max': round(float(ey[on].max()) * 1e3, 2)},
            'u_base_peak': np.abs(U[:, :3]).max(0).round(4).tolist(), 'vbox': cfg.vbox()[:3].round(6).tolist()}


def main():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ident = json.load(open(IDENT))
    cR3 = {'w_vref': 1.0, 'w_p': 50.0, 'w_a': 0.05, 'w_qn': 0.0}
    cS8 = {'w_vref': 0.03, 'w_p': 200.0, 'w_a': 0.01, 'w_qn': 0.8}
    p = json.load(open(os.path.join(DOOR, 'door_regrasp_ref_pb60p.json')))['m4r']
    cases = [('R3 原（rt_43）', r3_rows('door_regrasp_ref_pb60o.json'), 1.0, cR3, {'kind': 'world', 'v': 0.035, 'w': 0.19}),
             ('R3 新（roundtrip3）', r3_rows('door_regrasp_ref_pb60p.json'), 2.0, cR3, {'kind': 'body', 'v': p['r3_v_cap_body'], 'w': p['r3_w_cap']}),
             ('S8 原（rt_43）', s8_rows('door_return_ref_v3.json'), 1.0, cS8, {'kind': 'ff'}),
             ('S8 新（roundtrip3）', s8_rows('door_return_ref_v4.json'), 2.0, cS8, {'kind': 'ff'})]
    out = {'model': '手臂設定點模型（free4）＋底盤運動學（增益 1.0）；延遲 發布 0.6＋命令 1.0 週期、補償 1.6；無接觸、d̂ 未模擬', 'cases': {}}
    for name, rows, sc, co, g in cases:
        r = run(K, ident, *rows, scale=sc, coord=co, guide=g)
        r.update(coord=co, guide=g, box_scale=sc)
        out['cases'][name] = r
        print(f"{name}：參考 {r['T_ref_s']} s；失敗 {r['n_fail']}/{r['n_cycles']}；TCP mm {r['tcp_err_mm']}；末後收斂 {r['settle_to_5mm_s']} s；底盤 mm {r['base_err_mm']}；"
              f"偏航 mrad {r['yaw_err_mrad']}；u 底盤峰 {r['u_base_peak']}（框 {r['vbox']}）", flush=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(OUT, WS))


if __name__ == '__main__':
    sys.exit(main())
