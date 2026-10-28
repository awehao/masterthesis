#!/usr/bin/env python3
"""換抓 20 Hz 參考 → 各模式段（自 door_regrasp_ref_build_v3 抽出；切段規則不變，供時間化 v6 於記憶體建包與參考包建置 v4 共用）。
  r2：每點加 stage（來源階段；mode 5 預抓取節點閘控用）。
  mode 3：release_outer 自起點到 TCP 沿起點工具 −z 退開量第一次達 0.06 m；mode 4：release_outer 其餘＋M；mode 5：A＋I；mode 6：G。
"""
from __future__ import annotations

import numpy as np

S_W1 = 0.06


def split(R, fk):
    """R：reference_20hz 列 [t, x9, q_deg, qf, stage]；fk(x)→4×4。回傳 (seg, info)。"""
    X = np.array([r[1] for r in R], float)
    st = [r[4] for r in R]
    tt = np.array([r[0] for r in R], float)
    TCP = [np.asarray(fk([float(v) for v in x]), float) for x in X]
    i_rel = [i for i, s in enumerate(st) if s == 'release_outer']
    p0, z0 = TCP[0][:3, 3], TCP[0][:3, 2]
    s_all = np.array([float((p0 - TCP[i][:3, 3]) @ z0) for i in i_rel])
    if not np.any(s_all >= S_W1 - 1e-4):
        raise ValueError('release_outer 未達 W1 終點 0.06 m')
    i3_end = i_rel[int(np.argmax(s_all >= S_W1 - 1e-4))]
    seg = {3: list(range(0, i3_end + 1)), 4: [i for i in range(i3_end + 1, len(R)) if st[i] in ('release_outer', 'M')],
           5: [i for i, s in enumerate(st) if s in ('A', 'I')], 6: [i for i, s in enumerate(st) if s == 'G']}
    assert seg[4][0] == i3_end + 1 and seg[5][0] == seg[4][-1] + 1 and seg[6][0] == seg[5][-1] + 1
    return seg, {'X': X, 'st': st, 'tt': tt, 'TCP': TCP, 'p0': p0, 'z0': z0, 's_all': s_all, 'i_rel': i_rel, 'i3_end': i3_end}


def pack(R, seg, info):
    X, tt, TCP, p0, z0 = info['X'], info['tt'], info['TCP'], info['p0'], info['z0']

    def one(idx, with_s):
        tb = tt[idx[0]]
        d = {'t': [float(tt[i] - tb) for i in idx], 'tcp': [TCP[i].reshape(-1).tolist() for i in idx], 'base': [X[i, :3].tolist() for i in idx],
             'arm': [X[i, 3:].tolist() for i in idx], 'qf': [float(R[i][3]) for i in idx], 'q_door_deg': [float(R[i][2]) for i in idx],
             'src_index': [int(i) for i in idx], 'stage': [R[i][4] for i in idx]}
        if with_s:
            d['s'] = [float((p0 - TCP[i][:3, 3]) @ z0) for i in idx]
        return d
    return {str(k): one(v, k == 3) for k, v in seg.items()}
