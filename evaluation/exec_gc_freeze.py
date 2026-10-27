#!/usr/bin/env python3
"""執行端一次性 gc.freeze() 候選（Codex r30 §2；純邏輯、不依賴 Isaac）。
freeze_once()：在物理尚凍結、台架正式釋放之前呼叫**一次**。把當時受追蹤的物件移到永久世代，之後的自動回收忽略它們。
  - 不關閉自動 GC、不改回收門檻、不呼叫 gc.collect；不循環 freeze。
  - 記錄：Python 版本、freeze 耗時、freeze 前後的 gc.get_count／get_freeze_count、受追蹤物件數、原 GC 設定（isenabled、get_threshold）、
    行程記憶體（/proc/self/status VmRSS／VmHWM、resource ru_maxrss）。
  - 揭露：被 freeze 的物件（含當時已成為垃圾者）之後不再被回收 ⇒ 記憶體不會因回收釋放；不保證新物件不再造成停頓。
  - 任何錯誤都記下並回傳（不拋），不得影響控制。
final_state()：結束時再記一次 GC 設定、freeze 數與記憶體，供前後對照。

    python3 evaluation/test_exec_gc_freeze.py
"""
from __future__ import annotations

import gc
import resource
import sys
import time

NOTE = ('gc.freeze 把當時受追蹤物件移到永久世代，之後回收忽略它們；不是關閉 GC，不保證新物件不再造成停頓；'
        '被 freeze 的物件（含當時的垃圾）不再被回收')


def proc_mem():
    out = {}
    try:
        with open('/proc/self/status') as f:
            for ln in f:
                if ln.startswith(('VmRSS:', 'VmHWM:')):
                    k, v = ln.split(':', 1)
                    out[k + '_kB'] = int(v.split()[0])
    except (OSError, ValueError):
        out['proc_status'] = None
    try:
        out['ru_maxrss_kB'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:                                              # noqa: BLE001
        out['ru_maxrss_kB'] = None
    return out


def gc_state():
    return {'enabled': gc.isenabled(), 'threshold': list(gc.get_threshold()), 'count': list(gc.get_count()), 'freeze_count': gc.get_freeze_count()}


def freeze_once(clock=time.monotonic, count_objects=True):
    rec = {'python': sys.version, 'note': NOTE, 'ok': False, 'error': None}
    try:
        rec['before'] = dict(gc_state(), mem=proc_mem())
        if count_objects:
            objs = gc.get_objects()
            rec['before']['n_tracked'] = len(objs)
            del objs
        t0 = clock()
        gc.freeze()
        rec['freeze_s'] = clock() - t0
        rec['after'] = dict(gc_state(), mem=proc_mem())
        if count_objects:
            objs = gc.get_objects()
            rec['after']['n_tracked'] = len(objs)
            del objs
        rec['ok'] = True
    except Exception as e:                                         # noqa: BLE001  不得影響控制
        rec['error'] = f'{type(e).__name__}: {e}'
    return rec


def final_state():
    try:
        return dict(gc_state(), mem=proc_mem())
    except Exception as e:                                         # noqa: BLE001
        return {'error': f'{type(e).__name__}: {e}'}
