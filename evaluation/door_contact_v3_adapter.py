#!/usr/bin/env python3
"""door_contact_v3 的執行端轉接層（Codex r47 §3；純函式，不匯入 Isaac／SimulationApp）。供版本化執行端旁路呼叫：舊分類與舊控制欄位原義不變，
新分類另存，不改換抓准入或中止規則。
資料語意（Isaac Sim 6.0.1 isaacsim.core.prims RigidPrim.get_contact_force_data(dt) 文件）：
  回傳 (forces, points, normals, distances, pair_contacts_count, start_indices)；forces 為 (max_contact_count, 1) 的**法向接觸力純量**（可正可負，
  底層衝量 ÷ dt）、points／normals (max_contact_count, 3)、distances (max_contact_count, 1) 分離量；pair 由 (count, start) 索引。
  ⇒ 力大小＝|forces[:,0]|，是法向力大小；不含切向摩擦、也不是合力向量。原始有號值、法向、分離量與 dt 原樣保存。
緩衝完整性：四個緩衝長度一致；forces 第二維須為 1；count／start 為有限非負整數、長度＝機器人剛體名數；有資料切片落在緩衝內且互不重疊。
  任一不符 ⇒ 該步 status query_failed（附原因），不記零力。
每步紀錄＝door_contact_v3.classify_step 的輸出＋dt_s、force_semantics、names_sha（剛體名清單）。
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import door_contact_v3 as DC
import door_k3_evidence as KEV

FORCE_SEMANTICS = '法向接觸力大小（|get_contact_force_data 第一緩衝|＝底層法向衝量 ÷ dt）；不含切向摩擦；非合力向量'


def slice_all(fn, pts, nrm, sep, cnt, start, names):
    """回傳 dict(P, fn_signed, normal, sep, links)；緩衝不完整 ⇒ ValueError。"""
    fn, pts, nrm, sep = KEV._arr(fn), KEV._arr(pts), KEV._arr(nrm), KEV._arr(sep)
    if fn.ndim == 2:
        if fn.shape[1] != 1:
            raise ValueError(f'法向力緩衝形狀 {fn.shape}（應為 (M,1)）')
        fn = fn[:, 0]
    elif fn.ndim != 1:
        raise ValueError(f'法向力緩衝維度 {fn.ndim}')
    sep = sep.reshape(-1) if sep.ndim == 1 or (sep.ndim == 2 and sep.shape[1] == 1) else None
    if sep is None:
        raise ValueError('分離量緩衝形狀不符')
    if pts.ndim != 2 or pts.shape[1] != 3 or nrm.ndim != 2 or nrm.shape[1] != 3:
        raise ValueError(f'點／法向緩衝形狀 {pts.shape}／{nrm.shape}')
    m = len(fn)
    if not (len(pts) == len(nrm) == len(sep) == m):
        raise ValueError(f'緩衝長度不一致 fn {m}、點 {len(pts)}、法向 {len(nrm)}、分離 {len(sep)}')
    c = KEV._nonneg_int_array(cnt, 'pair count')
    s = KEV._nonneg_int_array(start, 'pair start')
    if len(c) != len(names) or len(s) != len(names):
        raise ValueError(f'pair 數 {len(c)}／{len(s)} 與剛體名 {len(names)} 不符')
    used = np.zeros(m, bool)
    idx, links = [], []
    for j, nm in enumerate(names):
        k, s0 = int(c[j]), int(s[j])
        if k == 0:
            continue
        if s0 + k > m:
            raise ValueError(f'pair {nm} 切片 [{s0}, {s0 + k}) 超出緩衝長度 {m}')
        if used[s0:s0 + k].any():
            raise ValueError(f'pair {nm} 切片 [{s0}, {s0 + k}) 與其他切片重疊')
        used[s0:s0 + k] = True
        idx.extend(range(s0, s0 + k))
        links.extend([nm] * k)
    idx = np.asarray(idx, int)
    return {'P': pts[idx] if len(idx) else np.zeros((0, 3)), 'fn_signed': fn[idx], 'normal': nrm[idx] if len(idx) else np.zeros((0, 3)),
            'sep': sep[idx], 'links': links}


def names_sha(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False).encode()).hexdigest()[:16]


def step_record(get_data, get_q, get_T, D, tol, dt, names, step=None, t=None, spec_id=None):
    """同一物理步：取接觸緩衝、門角（rad）、門根變換，分類。任何讀取／核對失敗 ⇒ status query_failed＋原因（不拋例外、不記零）。"""
    base = {'dt_s': float(dt), 'force_semantics': FORCE_SEMANTICS, 'names_sha': names_sha(names)}
    why = None
    try:
        data = get_data()
    except Exception as e:                                # noqa: BLE001
        data, why = None, f'讀取接觸緩衝失敗：{type(e).__name__}: {e}'[:200]
    q = T = None
    if why is None:
        try:
            q = float(get_q())
        except Exception as e:                            # noqa: BLE001
            why = f'讀取門角失敗：{type(e).__name__}: {e}'[:200]
    if why is None:
        try:
            T = np.asarray(get_T(), float)
        except Exception as e:                            # noqa: BLE001
            why = f'讀取門根變換失敗：{type(e).__name__}: {e}'[:200]
    sl = None
    if why is None:
        try:
            if len(data) != 6:
                raise ValueError(f'接觸緩衝回傳 {len(data)} 項（應為 6）')
            sl = slice_all(*data, names)
        except ValueError as e:
            why = f'緩衝不完整：{e}'[:200]
    if why is not None:
        r = DC.classify_step([], [], q, D, tol, T_world_door=None, status='query_failed', step=step, t=t, spec_id=spec_id)
        r['why'] = why
        r.update(base)
        return r
    r = DC.classify_step(sl['P'], np.abs(sl['fn_signed']), q, D, tol, robot_links=sl['links'], T_world_door=T, status='ok', step=step, t=t,
                         source='isaac_contact_force_data:normal', spec_id=spec_id,
                         extras={'fn_signed': sl['fn_signed'], 'normal': sl['normal'], 'sep': sl['sep']})
    r.update(base)
    return r


class Summary:
    """逐步累計（寫入 room_run.json 用）；不判定、不控制。"""

    def __init__(self):
        self.s = {'n_steps': 0, 'n_ok': 0, 'n_complete': 0, 'n_query_failed': 0, 'n_invalid_points': 0, 'why_first': None,
                  'max_sum_by_class': {k: 0.0 for k in DC.CLASSES}, 'n_steps_with': {k: 0 for k in DC.CLASSES}, 'all_reconciled': True}

    def add(self, r):
        s = self.s
        s['n_steps'] += 1
        if r['status'] != 'ok':
            s['n_query_failed'] += 1
            s['why_first'] = s['why_first'] or r.get('why')
            return
        s['n_ok'] += 1
        s['n_complete'] += int(bool(r['complete']))
        s['n_invalid_points'] += int(r['n_invalid'])
        s['all_reconciled'] = s['all_reconciled'] and bool(r['reconciled'])
        for k, v in r['sums'].items():
            if math.isfinite(v):
                s['max_sum_by_class'][k] = max(s['max_sum_by_class'][k], v)
                s['n_steps_with'][k] += int(v > 0)

    def as_dict(self):
        d = dict(self.s)
        d['complete_all_steps'] = bool(d['n_steps'] > 0 and d['n_complete'] == d['n_steps'])
        return d
