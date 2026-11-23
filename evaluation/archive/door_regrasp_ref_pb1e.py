#!/usr/bin/env python3
"""ref pb1e（Codex 20261029 extf_leader_1 回覆）：ref pb1 只改 segments['3']／['4'] 的交界——**以來源退離終點節點切段**，不再以 TCP 投影首次 ≥ 59.9 mm 推斷。
缺陷：ref pb1 的放手列退離在第 14 列正好 60.0 mm、其後為收臂；20 Hz 時間化在退離終點以 LSPB 轉角混合，取樣 s 最高 59.3 mm，收臂時 TCP 回擺再外擺至 61.4 mm 才越過門檻
⇒ seg 3 混入收臂前段 24 個取樣（1.2 s、j1／j3／j4／j5 合計變化至 0.58 rad），執行端追 TCP 由底盤實現、關節不動 ⇒ mode 4 銜接手臂 0.591 rad 擋下（extf_leader_1）。
處置（只此交界；其他段、欄位、門檻不變）：
  - 交界節點＝放手列第 14 列（退離終點）構型 x_node；在原 seg 3 的 20 Hz 取樣中取 TCP 最接近 x_node 的取樣 k*（t 3.6 s；TCP 0.72 mm、關節 0.53 mrad）。
  - 新 seg 3＝原 seg 3 的 [0, t_k*]，以時間重映射 τ(t) 讓速度在末段 T_D 內平滑降為 0（τ' 由 1 線性降到 0；不跨界、不帶入收臂），末樣本換成 x_node（停穩）。
  - 新 seg 4＝自 x_node 靜止起步：原 seg 3 的 [t_k*, 末] 接原 seg 4 全段，以 τ(t) 讓速度在首段 T_A 內由 0 平滑升至原速，首樣本＝x_node；之後同原時間表。
  - mode 3 終點＝mode 4 起點＝同一完整構型 x_node；0.05 rad 銜接限制不變。
核對：重映射段的差分速度／加速度 ≤ 求解器上限（底盤本體每軸 0.035255 m/s、偏航 0.1999 rad/s、線加速度 0.5 m/s²；手臂 0.9999 rad/s、19.984 rad/s²）；
修改段（新 seg 3 末 T_D、新 seg 4 首 T_A＋原混入的收臂前段）逐點 replan_v2.margins 名目（release_outer 規則；s < 0.06 目標 out，其後 None）；
以 extf_leader_1 實際退離末態（84.9 s）對新 seg 4 起點算銜接偏差（只報，不放寬）。登錄自驗須通過。

    python3 evaluation/door_regrasp_ref_pb1e.py   # 寫 results/door/door_regrasp_ref_pb1e.json（不覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_regrasp_ref_registry as REG                # noqa: E402
import door_regrasp_replan_v2 as R2                    # noqa: E402

RD = os.path.join(HERE, 'results', 'door')
SRC = os.path.join(RD, 'door_regrasp_ref_pb1.json')
REL = os.path.join(RD, 'door_regrasp_pb1_release.json')
OUT = os.path.join(RD, 'door_regrasp_ref_pb1e.json')
RUN = os.path.join(HERE, 'runs', 'door_pullback_extf_leader_1', 'room_run.json')
NODE_ROW = 14
DT = 0.05
T_D, T_A = 0.6, 0.6
LIM = {'vb': 0.035255, 'w': 0.1999, 'qd': 0.9999, 'ab': 0.5, 'qdd': 19.984}
FIELDS_VEC = ('tcp', 'base', 'arm')
FIELDS_SCAL = ('qf', 'q_door_deg', 's')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def interp_seg(seg, tq):
    """以原時間 seg['t'] 對任意 τ 線性內插各欄位（stage／src_index 取最近）。"""
    t = np.asarray(seg['t'], float)
    out = {}
    for f in FIELDS_VEC:
        if f in seg:
            A = np.asarray(seg[f], float)
            out[f] = np.stack([np.interp(tq, t, A[:, j]) for j in range(A.shape[1])], axis=1)
    for f in FIELDS_SCAL:
        if f in seg:
            out[f] = np.interp(tq, t, np.asarray(seg[f], float))
    near = np.clip(np.searchsorted(t, tq), 0, len(t) - 1)
    for f in ('stage', 'src_index'):
        if f in seg:
            out[f] = [seg[f][int(i)] for i in near]
    return out


def warp_end(T, Td):
    """新時間軸 0..T'：τ' 在最後 Td 內由 1 線性降至 0（τ 總長＝T）。回傳 τ(t_new) 取樣。"""
    T_new = T + Td / 2.0                       # 末段平均速率 1/2 ⇒ 多 Td/2
    tn = np.arange(0.0, T_new + 1e-9, DT)
    tn[-1] = T_new
    t0 = T_new - Td
    tau = np.where(tn <= t0, tn, t0 + (tn - t0) - (tn - t0) ** 2 / (2 * Td))
    return tn, np.minimum(tau, T)


def warp_start(T, Ta):
    T_new = T + Ta / 2.0
    tn = np.arange(0.0, T_new + 1e-9, DT)
    tn[-1] = T_new
    tau = np.where(tn <= Ta, tn ** 2 / (2 * Ta), tn - Ta / 2.0)
    return tn, np.minimum(tau, T)


def concat(a, b):
    """seg 片段 a 接 b（b 的時間平移接續）。"""
    out = {}
    for k in a:
        if isinstance(a[k], np.ndarray):
            out[k] = np.concatenate([a[k], b[k]])
        else:
            out[k] = list(a[k]) + list(b[k])
    return out


def limits(seg):
    t = np.asarray(seg['t'], float)
    B, A = np.asarray(seg['base'], float), np.asarray(seg['arm'], float)
    dt = np.diff(t)
    ok = dt > 1e-9
    vb = np.diff(B, axis=0)[ok] / dt[ok, None]
    th = B[:-1, 2][ok]
    vbx, vby = np.cos(th) * vb[:, 0] + np.sin(th) * vb[:, 1], -np.sin(th) * vb[:, 0] + np.cos(th) * vb[:, 1]
    qd = np.diff(A, axis=0)[ok] / dt[ok, None]
    dt2 = dt[ok][1:]
    ab = np.diff(np.c_[vbx, vby], axis=0) / dt2[:, None]
    qdd = np.diff(qd, axis=0) / dt2[:, None]
    return {'vb': float(max(np.abs(vbx).max(), np.abs(vby).max())), 'w': float(np.abs(vb[:, 2]).max()), 'qd': float(np.abs(qd).max()),
            'ab': float(np.abs(ab).max()), 'qdd': float(np.abs(qdd).max())}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    s3, s4 = p['segments']['3'], p['segments']['4']
    rel = json.load(open(REL))
    x_node = np.asarray(rel['rows'][NODE_ROW]['x'], float)
    Tn = np.asarray(PI.C.K.fk(list(x_node), 'link_tcp'), float)
    X3 = np.c_[np.asarray(s3['base'], float), np.asarray(s3['arm'], float)]
    d = [float(np.linalg.norm(np.asarray(PI.C.K.fk(list(x), 'link_tcp'))[:3, 3] - Tn[:3, 3])) for x in X3]
    k = int(np.argmin(d))
    t3 = np.asarray(s3['t'], float)
    tk = float(t3[k])
    # 新 seg 3：[0, tk] 末段減速至停
    tn3, tau3 = warp_end(tk, T_D)
    n3 = interp_seg(s3, tau3)
    n3['t'] = tn3
    # 末樣本換成精確節點（差 ≤ 0.53 mrad／0.72 mm），自該差值以餘弦在末段 T_D 內混入
    w = np.clip((tn3 - (tn3[-1] - T_D)) / T_D, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    dX = x_node - np.r_[n3['base'][-1], n3['arm'][-1]]
    n3['base'] = n3['base'] + w[:, None] * dX[None, :3]
    n3['arm'] = n3['arm'] + w[:, None] * dX[None, 3:]
    # 新 seg 4：原 seg 3 [tk, 末] ＋ 原 seg 4 全段，自靜止起步
    tail3 = {kk: (v[k:] if isinstance(v, list) else v) for kk, v in s3.items()}
    tail3['t'] = [float(x - tk) for x in t3[k:]]
    t4 = np.asarray(s4['t'], float)
    off = tail3['t'][-1] + DT
    cat = {kk: list(tail3[kk]) + list(s4[kk]) for kk in s4 if kk in tail3}
    cat['t'] = list(tail3['t']) + [float(x + off) for x in t4]
    T4 = cat['t'][-1]
    tn4, tau4 = warp_start(T4, T_A)
    n4 = interp_seg(cat, tau4)
    n4['t'] = tn4
    w4 = np.clip(1 - tn4 / T_A, 0, 1)
    w4 = 0.5 - 0.5 * np.cos(np.pi * w4)
    dX4 = x_node - np.r_[n4['base'][0], n4['arm'][0]]
    n4['base'] = n4['base'] + w4[:, None] * dX4[None, :3]
    n4['arm'] = n4['arm'] + w4[:, None] * dX4[None, 3:]
    # tcp 重算
    for seg in (n3, n4):
        seg['tcp'] = np.stack([np.asarray(PI.C.K.fk(list(np.r_[b, a]), 'link_tcp'), float).reshape(-1) for b, a in zip(seg['base'], seg['arm'])])
    join = float(np.max(np.abs(np.r_[n3['base'][-1], n3['arm'][-1]] - np.r_[n4['base'][0], n4['arm'][0]])))
    lim = {'seg3': limits(n3), 'seg4_first_3s': limits({kk: (v[:int(3.0 / DT)] if not isinstance(v, (int, float)) else v) for kk, v in n4.items()})}
    bad = {sg: {kk: v for kk, v in L.items() if v > LIM[kk]} for sg, L in lim.items()}
    # 淨空（修改段）
    q0 = float(s3['q_door_deg'][0])
    mins, fail = [], None
    for name, seg, rng in (('seg3_end', n3, range(max(0, len(n3['t']) - int(T_D / DT) - 2), len(n3['t']))),
                           ('seg4_start', n4, range(0, min(len(n4['t']), int((T_A + (t3[-1] - tk) + 0.5) / DT))))):
        for i in rng:
            x = np.r_[n4['base'][i] if seg is n4 else n3['base'][i], seg['arm'][i]]
            qf = float(seg['qf'][i])
            sv = float(seg['s'][i]) if 's' in seg else 0.06
            okm, m, det = R2.margins(x, q0, qf, 'out' if sv < 0.06 else None, 'release_outer')
            if not okm:
                fail = {'seg': name, 'i': i, 't': float(seg['t'][i]), 'det': det}
                break
            mins.append((float(m), name, i))
        if fail:
            break
    # 本趟實際退離末態對新 seg 4 起點
    rr = json.load(open(RUN))
    I = {c: i for i, c in enumerate(rr['steps_cols'])}
    sx = min(rr['steps'], key=lambda s: abs(s[1] - 84.9))
    xb, xa = np.asarray(sx[I['base_xyth']], float), np.asarray(sx[I['q_arm_meas']], float)
    bridge = {'base_dev_m': float(np.linalg.norm(xb[:2] - n4['base'][0][:2])), 'yaw_dev_deg': math.degrees(abs(math.remainder(xb[2] - n4['base'][0][2], 2 * math.pi))),
              'arm_dev_rad': float(np.max(np.abs(xa - n4['arm'][0]))), 'limits': {'base_m': 0.02, 'base_deg': 2.0, 'arm_rad': 0.05}}
    rep = {'k_star': k, 't_k_star': tk, 'node_tcp_dist_mm': d[k] * 1e3, 'join_max_abs': join, 'limits': lim, 'limits_bad': bad,
           'margins_fail': fail, 'margins_min': min(mins)[0] if mins else None, 'bridge_vs_extf_leader_1_84p9': bridge,
           'seg3_T': [float(t3[-1]), float(n3['t'][-1])], 'seg4_T': [float(t4[-1]), float(n4['t'][-1])]}
    print(json.dumps(rep, ensure_ascii=False, default=float, indent=0)[:1800], flush=True)
    if any(bad.values()) or fail is not None or join > 1e-12:
        raise SystemExit('核對不成立（不寫檔）')
    def pack(seg, with_s):
        o = {'t': [float(x) for x in seg['t']], 'tcp': [list(map(float, r)) for r in seg['tcp']], 'base': [list(map(float, r)) for r in seg['base']],
             'arm': [list(map(float, r)) for r in seg['arm']], 'qf': [float(x) for x in seg['qf']], 'q_door_deg': [float(x) for x in seg['q_door_deg']],
             'src_index': [int(x) for x in seg['src_index']], 'stage': list(seg['stage'])}
        if with_s:
            o['s'] = [float(x) for x in seg['s']]
        return o
    p['segments']['3'] = pack(n3, True)
    p['segments']['4'] = pack(n4, False)
    p['mode3_s_end_m'] = float(n3['s'][-1])
    p['schema'] = p.get('schema', '') + '+pb1e'
    p['pb1e'] = dict(rep, derived_from={'path': os.path.relpath(SRC, HERE), 'sha256': sha(SRC)}, node_row=NODE_ROW, T_D=T_D, T_A=T_A,
                     note='以來源退離終點節點切段；mode 3 終點＝mode 4 起點＝x_node（停穩）；mode 5–8 不變')
    ok, why = REG.validate(json.loads(json.dumps(p, default=float)))
    p['registry_validate'] = {'ok': ok, 'why': why}
    if not ok:
        raise SystemExit(f'登錄自驗不過：{why}')
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫', OUT)


if __name__ == '__main__':
    main()
