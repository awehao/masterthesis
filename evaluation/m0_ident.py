#!/usr/bin/env python3
"""M0 手臂設定點追蹤模型辨識與驗證：m0_ident.py [批次=m0] [--json out.json]

資料：results/<批次>/<區域>_r<k>/m0.npz（isaac_m0_excite.py；q_i＝寫入前量測、s_i＝支配 i→i+1 的設定點）。
模型（同 wgmpc_core_sp.ArmSetpointModel）：每物理步  q_{i+1} = q_i + α⊙(s_i − q_i) + b。
  * 區域模型：只用該區域 train 段估 α、b（逐軸最小平方）；
  * 共同模型：三區 train 段合併估一組 α、b（WB-MPC 實際用一組參數）；
  * 對照「直接積分」：q(+H) = q + H·u（u＝延遲後實際積分的速率 u_eff；不含設定點狀態、不含線上補償）。
評估（validation／test 段，同一段的視窗不混入訓練）：
  * 一步（10 ms）預測 RMSE；
  * 50 ms（5 步）開迴路預測 RMSE——設定點由 u_eff 在視界內自行遞推（MPC 實際用法）；
  * 逐區、逐軸；另報穩態偏差（保持段 s − q 平均）。
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.01
H = 5                                                                # 50 ms 控制拍


def load(batch):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'results', batch, '*_r*', 'm0.npz'))):
        d = os.path.dirname(f); meta = json.load(open(os.path.join(d, 'meta.json')))
        z = np.load(f)
        out[os.path.basename(d)] = dict(meta=meta, **{k: z[k] for k in z.files})
    return out


def fit(recs):
    """逐軸最小平方：Δq = α(s − q) + b。"""
    A_, Y_ = [], []
    for r in recs:
        q, s = r['q'], r['s']
        A_.append(s[:-1] - q[:-1]); Y_.append(q[1:] - q[:-1])
    A_, Y_ = np.vstack(A_), np.vstack(Y_)
    al, b = np.zeros(6), np.zeros(6)
    for j in range(6):
        X = np.c_[A_[:, j], np.ones(len(A_))]
        sol, *_ = np.linalg.lstsq(X, Y_[:, j], rcond=None)
        al[j], b[j] = sol
    return al, b


def eval_rec(r, al, b):
    q, s, u = r['q'], r['s'], r['u_eff']
    n = len(q)
    one = q[:-1] + al * (s[:-1] - q[:-1]) + b - q[1:]
    e_sp, e_di = [], []
    for i in range(0, n - H, 1):
        qq, ss = q[i].copy(), s[i - 1].copy() if i > 0 else s[0].copy()
        for k in range(H):                                           # 設定點自行遞推：s_{i+k} = s_{i+k−1} + dt·u_{i+k}
            ss = ss + DT * u[i + k]
            qq = qq + al * (ss - qq) + b
        e_sp.append(qq - q[i + H])
        e_di.append(q[i] + DT * u[i:i + H].sum(axis=0) - q[i + H])
    rms = lambda e: [float(1e3 * math.sqrt(np.mean(np.asarray(e)[:, j] ** 2))) for j in range(6)]   # noqa: E731
    hold = r['hold'].astype(bool)
    hb = [float(1e3 * np.mean((s - q)[hold & (np.arange(n) % 100 > 50)][:, j])) for j in range(6)] if hold.any() else None
    return dict(one_step_rms_mrad=rms(one), h50_setpoint_rms_mrad=rms(e_sp), h50_direct_rms_mrad=rms(e_di), hold_offset_mrad=hb)


def main():
    av = sys.argv
    batch = av[1] if len(av) > 1 and not av[1].startswith('--') else 'm0'
    D = load(batch)
    if not D:
        print('沒有資料'); return 1
    regs = sorted({v['meta']['region'] for v in D.values()})
    tr = {g: [v for v in D.values() if v['meta']['region'] == g and v['meta']['split'] == 'train'] for g in regs}
    models = {g: fit(tr[g]) for g in regs if tr[g]}
    models['pooled'] = fit([r for g in regs for r in tr[g]])
    res = dict(models={k: dict(alpha=v[0].tolist(), bias_rad=v[1].tolist(), tau_s=[(-DT / math.log(1 - x)) if 0 < x < 1 else None for x in v[0]]) for k, v in models.items()}, eval={})
    for name, r in sorted(D.items()):
        g, sp = r['meta']['region'], r['meta']['split']
        if sp == 'train':
            continue
        res['eval'][name] = dict(region=g, split=sp, regional=eval_rec(r, *models[g]), pooled=eval_rec(r, *models['pooled']))
    for k, v in res['models'].items():
        print(k, 'α', np.round(v['alpha'], 4).tolist(), 'τ', [round(x, 3) if x else None for x in v['tau_s']])
    for name, e in res['eval'].items():
        print(name, e['split'], '50ms 設定點模型（區域）', np.round(e['regional']['h50_setpoint_rms_mrad'], 2).tolist(),
              '共同', np.round(e['pooled']['h50_setpoint_rms_mrad'], 2).tolist(), '直接積分', np.round(e['regional']['h50_direct_rms_mrad'], 2).tolist())
    if '--json' in av:
        json.dump(res, open(av[av.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
