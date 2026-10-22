#!/usr/bin/env python3
"""K3 接觸承載**取證**的純邏輯（擷取與整理；不判定、不控制）。執行端複本 isaac_door_k2_sim_ev.py 逐物理步呼叫。
規格：evaluation/results/specs/K3_contact_evidence_spec.md。

資料來源與語意（Isaac Sim 6.0.1 isaacsim.core.prims；以 API 文件為準）：
  - get_contact_force_data(dt)：感測端＝門扇剛體、過濾對象＝機器人各剛體。逐接觸點：**法向力**（純量）、接觸點、法向、分離距離；
    pair 由 (count, start) 索引。法向的正負約定未經本專案核對 ⇒ 原樣保存，不自行翻號。
  - get_friction_data(dt)：逐錨點**切向（摩擦）力向量**與位置；pair 由 (count, start) 索引。與接觸點不保證一一對應 ⇒ 分開保存。
  - get_measured_joint_forces()[joint6 列]：連桿入射關節的**反作用力與力矩**（6 維，子關節座標）；**不是馬達輸出**。
  - get_measured_joint_efforts()：物理解算器在 DOF 方向的關節力（含驅動及其他約束在 DOF 方向的投影）；**不是馬達命令值**。
讀不到任何一項 ⇒ 該欄記 None 並附原因字串，不寫成 0。
"""
from __future__ import annotations

import math

import numpy as np

KEEP_BODIES = ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef', 'link6')
ARM_DOF = tuple(f'joint{i}' for i in range(1, 7))
FINGER_DOF = ('finger_joint1', 'finger_joint2')


def _arr(x):
    return np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x, dtype=float)


def _r(v, n=7):
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def _pair_slices(cnt, start, names, keep):
    cnt = np.asarray(cnt).reshape(-1)
    start = np.asarray(start).reshape(-1)
    if len(cnt) != len(names) or len(start) != len(names):
        raise ValueError(f'pair 數 {len(cnt)}／{len(start)} 與剛體名 {len(names)} 不符')
    for j, nm in enumerate(names):
        if nm in keep and int(cnt[j]) > 0:
            yield nm, int(start[j]), int(cnt[j])


def extract_contacts(fn, pts, nrm, sep, cnt, start, names, keep=KEEP_BODIES):
    """回傳 {body: [{'p':[3],'n':[3],'fn':x,'sep':x}, …]}（只含 keep 中有接觸者）。"""
    fn, pts, nrm, sep = _arr(fn), _arr(pts), _arr(nrm), _arr(sep)
    fn = fn.reshape(fn.shape[0], -1)[:, 0] if fn.ndim > 1 else fn.reshape(-1)
    pts, nrm = pts.reshape(-1, 3), nrm.reshape(-1, 3)
    sep = sep.reshape(-1)
    out = {}
    for nm, s0, k in _pair_slices(cnt, start, names, keep):
        out[nm] = [{'p': [_r(x) for x in pts[i]], 'n': [_r(x) for x in nrm[i]], 'fn': _r(fn[i]), 'sep': _r(sep[i])}
                   for i in range(s0, s0 + k)]
    return out


def extract_friction(ff, fp, cnt, start, names, keep=KEEP_BODIES):
    """回傳 {body: [{'p':[3],'ft':[3]}, …]}。"""
    ff, fp = _arr(ff).reshape(-1, 3), _arr(fp).reshape(-1, 3)
    out = {}
    for nm, s0, k in _pair_slices(cnt, start, names, keep):
        out[nm] = [{'p': [_r(x) for x in fp[i]], 'ft': [_r(x) for x in ff[i]]} for i in range(s0, s0 + k)]
    return out


def wrench_row(F, row):
    F = _arr(F)
    F = F.reshape(-1, F.shape[-1]) if F.ndim > 2 else F
    v = F[row]
    if v.shape[-1] < 6:
        raise ValueError(f'反作用力列長 {v.shape[-1]} < 6')
    return [_r(x) for x in v[:6]]


def efforts_by_name(E, dof_names, wanted):
    E = _arr(E).reshape(-1)
    if len(E) != len(dof_names):
        raise ValueError(f'efforts 長 {len(E)} 與 DOF 名 {len(dof_names)} 不符')
    idx = {n: i for i, n in enumerate(dof_names)}
    missing = [w for w in wanted if w not in idx]
    if missing:
        raise ValueError(f'找不到 DOF {missing}')
    return {w: _r(E[idx[w]]) for w in wanted}


def safe(fn, *a, **k):
    """呼叫擷取函式；任何例外 ⇒ (None, 原因字串)。"""
    try:
        return fn(*a, **k), None
    except Exception as e:                                    # noqa: BLE001 — 取證不得中斷模擬
        return None, f'{type(e).__name__}: {e}'[:200]


def summarize_contacts(contacts, T_WH):
    """供離線分析：把接觸點轉到把手座標（x 沿橫桿、y 法向、z 鉛直），回傳 {body: {'n','fn_sum','p_handle_mean','n_handle_mean'}}。"""
    T = np.asarray(T_WH, float).reshape(4, 4)
    R, c = T[:3, :3], T[:3, 3]
    out = {}
    for nm, lst in (contacts or {}).items():
        ok = [x for x in lst if x['fn'] is not None and None not in x['p'] and None not in x['n']]
        if not ok:
            continue
        P = np.array([x['p'] for x in ok]) - c
        N = np.array([x['n'] for x in ok])
        f = np.array([x['fn'] for x in ok])
        w = f / f.sum() if f.sum() > 0 else np.full(len(f), 1.0 / len(f))
        out[nm] = {'n': len(ok), 'fn_sum': float(f.sum()), 'p_handle_mean': (R.T @ (w @ P)).tolist(), 'n_handle_mean': (R.T @ (w @ N)).tolist()}
    return out
