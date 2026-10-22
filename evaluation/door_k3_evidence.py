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

import json
import math
import time

import numpy as np

KEEP_BODIES = ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef', 'link6')
ARM_DOF = tuple(f'joint{i}' for i in range(1, 7))
FINGER_DOF = ('finger_joint1', 'finger_joint2')


def _arr(x):
    return np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x, dtype=float)


def _r(v, n=7):
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


def _nonneg_int_array(x, what):
    a = _arr(x).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError(f'{what} 含非有限值')
    if (a < 0).any():
        raise ValueError(f'{what} 含負值')
    if (a != np.floor(a)).any():
        raise ValueError(f'{what} 含非整數值')
    return a.astype(np.int64)


def _pair_slices(cnt, start, names, keep, n_buf):
    """(count, start) 必須是有限非負整數；有資料的切片須完全落在緩衝 [0, n_buf) 內（Codex 20261022_115319）。不符 ⇒ ValueError。"""
    cnt = _nonneg_int_array(cnt, 'pair count')
    start = _nonneg_int_array(start, 'pair start')
    if len(cnt) != len(names) or len(start) != len(names):
        raise ValueError(f'pair 數 {len(cnt)}／{len(start)} 與剛體名 {len(names)} 不符')
    for j, nm in enumerate(names):
        k, s0 = int(cnt[j]), int(start[j])
        if k > 0 and s0 + k > n_buf:
            raise ValueError(f'pair {nm} 切片 [{s0}, {s0 + k}) 超出緩衝長度 {n_buf}')
        if nm in keep and k > 0:
            yield nm, s0, k


def extract_contacts(fn, pts, nrm, sep, cnt, start, names, keep=KEEP_BODIES):
    """回傳 {body: [{'p':[3],'n':[3],'fn':x,'sep':x}, …]}（只含 keep 中有接觸者）。"""
    fn, pts, nrm, sep = _arr(fn), _arr(pts), _arr(nrm), _arr(sep)
    fn = fn.reshape(fn.shape[0], -1)[:, 0] if fn.ndim > 1 else fn.reshape(-1)
    pts, nrm = pts.reshape(-1, 3), nrm.reshape(-1, 3)
    sep = sep.reshape(-1)
    n_buf = min(len(fn), len(pts), len(nrm), len(sep))
    out = {}
    for nm, s0, k in _pair_slices(cnt, start, names, keep, n_buf):
        out[nm] = [{'p': [_r(x) for x in pts[i]], 'n': [_r(x) for x in nrm[i]], 'fn': _r(fn[i]), 'sep': _r(sep[i])}
                   for i in range(s0, s0 + k)]
    return out


def extract_friction(ff, fp, cnt, start, names, keep=KEEP_BODIES):
    """回傳 {body: [{'p':[3],'ft':[3]}, …]}。"""
    ff, fp = _arr(ff).reshape(-1, 3), _arr(fp).reshape(-1, 3)
    out = {}
    for nm, s0, k in _pair_slices(cnt, start, names, keep, min(len(ff), len(fp))):
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
    """供離線分析：接觸點轉到把手座標（x 沿橫桿、y 法向、z 鉛直；原點＝把手 prim 原點；單位 m）。
    SDK 的法向力可帶負號 ⇒ 原始符號保留：fn_signed_sum；幾何摘要明示採**幅值權重** |fn|：fn_abs_sum、p_handle_absw、n_handle_absw。
    |fn| 全零 ⇒ 加權中心標 None（未定），不以等權冒充力加權。法向作用側未核清 ⇒ n 只作幾何方向描述。"""
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
        fa = np.abs(f)
        if fa.sum() > 0:
            w = fa / fa.sum()
            p_w, n_w = (R.T @ (w @ P)).tolist(), (R.T @ (w @ N)).tolist()
        else:
            p_w = n_w = None
        out[nm] = {'n': len(ok), 'fn_signed_sum': float(f.sum()), 'fn_abs_sum': float(fa.sum()), 'weight': 'abs(fn)',
                   'p_handle_absw': p_w, 'n_handle_absw': n_w,
                   'p_handle_unweighted': (R.T @ P.mean(axis=0)).tolist()}
    return out


class EvidenceWriter:
    """取證 JSONL 的開檔／寫出／flush／關檔，每一步都隔離故障（Codex 20261022_115319）：任何失敗只記入 status，不拋例外。
    status['complete'] 只在「有開檔、無任何錯誤、已正常關檔」時為 True；其餘一律 False 並列出原因，不靜默當成功。"""

    def __init__(self, path, flush_every=200, opener=open):
        self.path = path
        self.flush_every = int(flush_every)
        self.f = None
        self.status = {'path': path or None, 'enabled': bool(path), 'opened': False, 'closed': False, 'n': 0, 'n_err': {},
                       'errors': [], 'write_s_total': 0.0, 'write_s_max': 0.0, 'complete': False}
        if not path:
            return
        try:
            self.f = opener(path, 'w')
            self.status['opened'] = True
        except Exception as e:                            # noqa: BLE001
            self._err('open', e)

    def _err(self, k, e):
        self.status['n_err'][k] = self.status['n_err'].get(k, 0) + 1
        if len(self.status['errors']) < 20:
            self.status['errors'].append(f'{k}: {type(e).__name__}: {e}'[:200])

    @property
    def active(self):
        return self.f is not None

    def count_err(self, k):
        self.status['n_err'][k] = self.status['n_err'].get(k, 0) + 1

    def write(self, rec):
        if self.f is None:
            return False
        t0 = time.perf_counter()
        try:
            self.f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False, default=float) + '\n')
            self.status['n'] += 1
            if self.flush_every > 0 and self.status['n'] % self.flush_every == 0:
                self.f.flush()
            ok = True
        except Exception as e:                            # noqa: BLE001
            self._err('write', e)
            ok = False
        dt = time.perf_counter() - t0
        self.status['write_s_total'] += dt
        self.status['write_s_max'] = max(self.status['write_s_max'], dt)
        return ok

    def add_time(self, dt):
        """記錄整個取證讀取＋寫出一步的耗時（秒）。"""
        self.status.setdefault('step_s_total', 0.0)
        self.status.setdefault('step_s_max', 0.0)
        self.status['step_s_total'] += dt
        self.status['step_s_max'] = max(self.status['step_s_max'], dt)

    def close(self):
        if self.f is not None:
            try:
                self.f.flush()
            except Exception as e:                        # noqa: BLE001
                self._err('flush', e)
            try:
                self.f.close()
                self.status['closed'] = True
            except Exception as e:                        # noqa: BLE001
                self._err('close', e)
            self.f = None
        self.status['complete'] = bool(self.status['enabled'] and self.status['opened'] and self.status['closed'] and not self.status['n_err'])
        return dict(self.status)
