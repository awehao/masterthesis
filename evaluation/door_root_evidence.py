#!/usr/bin/env python3
"""根與門扇逐步三點讀取取證的純邏輯（規格 K3_root_evidence_spec_draft2.md；Codex 20261023_233900／234952）。

給執行端複本 isaac_door_k2_sim_ev2r.py 用；本檔不 import Isaac，可離線測試。檔案 I/O 沿用 door_k3_evidence.EvidenceWriter（故障隔離）。

  - grab(fn)：呼叫讀取函式，**立即複製**成 Python float 串列（不保存可能被後續更新的 view／張量）；
    例外 ⇒ (None, 原因)；非有限值 ⇒ 該欄記 None，原因 'nonfinite'（不寫 NaN、不補零）。
  - Row：一個物理步的一列；put(key, fn) 逐欄擷取，失敗原因收進 row['errors'][key]；finish() 補上缺欄清單。
  - physics_snapshot(usd_items, runtime_items)：啟動時一次性快照，分「USD 宣告值」「執行期回讀值」「未知」三類；
    讀不到或不存在 ⇒ 列入 unknown 並附原因，**不把預設值猜成實際值**。
  - module_info(objs)：執行期模組來源（__module__、原始檔路徑、sha256）與套件版本。
"""
from __future__ import annotations

import hashlib
import inspect
import math

SCHEMA = 'door_root_evidence/1'
POINTS = ('A_root_pre', 'api_args', 'B_root_post_write', 'C_leaf_pre_step', 'D_root_post_step', 'D_leaf_post_step')


def _flat(x):
    """任意巢狀數值（numpy／torch／list）⇒ 立即複製的 float 串列；失敗拋例外。"""
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    if hasattr(x, 'tolist'):
        x = x.tolist()
    out = []

    def rec(v):
        if hasattr(v, 'cpu'):
            v = v.cpu()
        if hasattr(v, 'numpy') and not isinstance(v, (list, tuple)):
            v = v.numpy()
        if hasattr(v, 'tolist'):
            v = v.tolist()
        if isinstance(v, (list, tuple)):
            for w in v:
                rec(w)
        else:
            if isinstance(v, bool):
                raise TypeError('bool 不是數值')
            out.append(float(v))
    rec(x)
    return out


def grab(fn):
    """回傳 (值或 None, 原因或 None)。值為新建的 float 串列（與來源記憶體無關）。"""
    try:
        v = _flat(fn())
    except Exception as e:                                   # noqa: BLE001 — 取證不得中斷模擬
        return None, f'{type(e).__name__}: {e}'[:200]
    if not all(math.isfinite(x) for x in v):
        return None, 'nonfinite'
    return v, None


def grab_bool(fn):
    try:
        v = fn()
    except Exception as e:                                   # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'[:200]
    if not isinstance(v, bool):
        try:
            v = bool(v)
        except Exception as e:                               # noqa: BLE001
            return None, f'not_bool: {e}'[:200]
    return v, None


class Row:
    def __init__(self, step, t):
        self.d = {'schema': SCHEMA, 'step': int(step), 't': float(t), 'errors': {}}

    def put(self, key, fn, kind='num'):
        v, why = (grab_bool(fn) if kind == 'bool' else grab(fn))
        self.d[key] = v
        if why is not None:
            self.d['errors'][key] = why
        return v

    def set(self, key, value):
        """直接放入已計算好的值（例如實際傳給 API 的向量）；非有限 ⇒ None＋原因。"""
        try:
            v = _flat(value) if not isinstance(value, (str, bool)) and value is not None else value
        except Exception as e:                               # noqa: BLE001
            self.d[key] = None
            self.d['errors'][key] = f'{type(e).__name__}: {e}'[:200]
            return
        if isinstance(v, list) and not all(math.isfinite(x) for x in v):
            self.d[key] = None
            self.d['errors'][key] = 'nonfinite'
            return
        self.d[key] = v

    def finish(self, required):
        self.d['missing'] = [k for k in required if self.d.get(k) is None]
        if not self.d['errors']:
            del self.d['errors']
        return self.d


def physics_snapshot(usd_items, runtime_items):
    """usd_items / runtime_items：{名稱: 無參數函式}。回傳三分類快照；None、例外、非可序列化 ⇒ unknown（附原因）。"""
    snap = {'usd_declared': {}, 'runtime_readback': {}, 'unknown': {}}
    for cat, items in (('usd_declared', usd_items), ('runtime_readback', runtime_items)):
        for name, fn in items.items():
            try:
                v = fn()
            except Exception as e:                           # noqa: BLE001
                snap['unknown'][f'{cat}:{name}'] = f'{type(e).__name__}: {e}'[:200]
                continue
            if v is None:
                snap['unknown'][f'{cat}:{name}'] = 'none（未宣告或讀不到；不以預設值代替）'
                continue
            try:
                if hasattr(v, 'tolist'):
                    v = v.tolist()
                if isinstance(v, tuple):
                    v = list(v)
                if not isinstance(v, (bool, int, float, str, list, dict)):
                    v = str(v)
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError('nonfinite')
            except Exception as e:                           # noqa: BLE001
                snap['unknown'][f'{cat}:{name}'] = f'{type(e).__name__}: {e}'[:200]
                continue
            snap[cat][name] = v
    return snap


def module_info(objs, packages=('isaacsim', 'isaacsim-core')):
    """objs：{名稱: 類別或函式或模組}。回傳執行期來源與版本；讀不到的欄位記原因。"""
    out = {'objects': {}, 'packages': {}}
    for name, o in objs.items():
        rec = {}
        try:
            rec['module'] = getattr(o, '__module__', None) or getattr(o, '__name__', None)
            path = inspect.getsourcefile(o)
            rec['file'] = path
            with open(path, 'rb') as f:
                rec['sha256'] = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:                               # noqa: BLE001
            rec['error'] = f'{type(e).__name__}: {e}'[:200]
        out['objects'][name] = rec
    try:
        import importlib.metadata as M
        for p in packages:
            try:
                out['packages'][p] = M.version(p)
            except Exception as e:                           # noqa: BLE001
                out['packages'][p] = f'unknown: {type(e).__name__}'
    except Exception as e:                                   # noqa: BLE001
        out['packages_error'] = str(e)[:200]
    return out
