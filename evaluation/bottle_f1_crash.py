#!/usr/bin/env python3
"""F1 例外保存與獨立收尾（Codex 20261103 裁定；純函式，Isaac 與假世界共用）。原則：
1 先存例外、時刻、輪次與相位（crash.json，原子寫入），再盡力保存逐步資料；存檔失敗只記在 save_errors，不掩蓋原例外。
2 收尾走獨立停止：只把底盤命令有界減速到零、手臂保持最後目標，不再進控制器或規劃流程。
3 停止有效性要量：實測底盤與手臂都靜止才算 ok；量不到或未靜止就是 False／None（未驗證），照實報。
"""
import json
import os
import traceback

import numpy as np


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, path)


def record(out_dir, exc, ctx):
    """先把例外落盤；回傳 crash dict（之後補欄位再寫一次）。"""
    c = dict(type=type(exc).__name__, msg=str(exc), traceback=traceback.format_exception(type(exc), exc, exc.__traceback__), save_errors=[], stop=None, **ctx)
    atomic_json(os.path.join(out_dir, 'crash.json'), c)
    return c


def guarded(c, name, fn):
    """盡力保存：失敗只記錄，不掩蓋原例外。回傳 fn 的結果或 None。"""
    try:
        return fn()
    except Exception as e:                                    # noqa: BLE001
        c['save_errors'].append(dict(step=name, type=type(e).__name__, msg=str(e)))
        return None


def independent_stop(step_fn, read_q, u0, a_base, dt, t_tail=3.0, v_still=0.01, w_still=0.02, arm_still=1e-3):
    """step_fn(u_base)：送底盤命令並推進一個物理步（手臂保持由呼叫端做）；read_q()：回傳 9 維量測。回傳 dict(ok, v, w, arm, n, why)。"""
    u = np.asarray(u0, float).copy()
    da = np.asarray(a_base, float) * float(dt)
    n = int(round(t_tail / float(dt)))
    q_prev, last = None, None
    for k in range(n):
        u = u + np.clip(-u, -da, da)
        step_fn(u.copy())
        if k >= n - 6:                                        # 最後 5 個物理步的差分
            q = np.asarray(read_q(), float)
            if q_prev is not None and np.isfinite(q).all() and np.isfinite(q_prev).all():
                dy = float(np.arctan2(np.sin(q[2] - q_prev[2]), np.cos(q[2] - q_prev[2])))
                last = (float(np.hypot(*(q[:2] - q_prev[:2]))) / dt, abs(dy) / dt, float(np.abs(q[3:] - q_prev[3:]).max()) / dt)
            q_prev = q
    if last is None:
        return dict(ok=None, why='no_measurement', n=n)
    return dict(ok=bool(not np.any(u) and last[0] <= v_still and last[1] <= w_still and last[2] <= arm_still), v=last[0], w=last[1], arm=last[2], n=n, why=None)
