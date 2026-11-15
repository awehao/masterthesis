#!/usr/bin/env python3
"""門任務程序的 CPU 排程配置（T2，Codex 20261115）。只改排程與數值執行緒數；不改視窗解析度、算圖頻率、工作程序數。
機器：8 實體核心×2（SMT；邏輯 CPU i 與 i+8 同核心）。
  主控制執行緒：CPU 0（其 SMT 同核 8 不分配給本任務的任何程序）。
  Kit（模擬器其餘執行緒）＋背景規劃程序（nice 10）：CPU 1, 2, 9, 10。
  閉鏈一致停法的兩個模擬程序：CPU 3, 11（各一）。
  淨空計算池 8 個工作程序：CPU 4–7, 12–15（各一）。
Linux 親和性是逐執行緒屬性：工作程序在 fork 後、單執行緒時設定；主程序在啟動 Kit 前先設成 Kit 的集合（Kit 執行緒繼承），Kit 起來後只把主執行緒移到 CPU 0。
之後才由主執行緒建立的執行緒會繼承 CPU 0——readback() 逐執行緒讀回，照實記錄落在 CPU 0 的非主執行緒數。
BLAS：模擬器 Python 的 NumPy 內附多執行緒 OpenBLAS（系統 Python 是單執行緒參考 BLAS）；匯入 NumPy 前設 *_NUM_THREADS＝1（fork 的工作程序繼承）。"""
import os

MAIN, KIT, CSTOP, POOL = [0], [1, 2, 9, 10], [3, 11], [4, 5, 6, 7, 12, 13, 14, 15]
BLAS_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def blas_single():
    """須在匯入 NumPy 之前呼叫。"""
    for k in BLAS_ENV:
        os.environ[k] = '1'


def pin(pid, cpus, nice=None):
    try:
        os.sched_setaffinity(pid, set(cpus))
        if nice is not None:
            os.setpriority(os.PRIO_PROCESS, pid, nice)
        return True
    except Exception:                                                 # noqa: BLE001
        return False


def pin_workers(pool=None, csw=None, planners=()):
    out = {}
    if pool is not None:
        out['pool'] = [pin(p.pid, [POOL[i % len(POOL)]]) for i, p in enumerate(pool.procs)]
    if csw is not None:
        out['cstop'] = [pin(p.pid, [CSTOP[i % len(CSTOP)]]) for i, p in enumerate(csw.procs)]
    out['planners'] = [pin(pl.proc.pid, KIT, nice=10) for pl in planners]
    return out


def pre_kit():
    return pin(0, KIT)


def post_kit():
    return pin(0, MAIN)


def readback(pool=None, csw=None, planners=()):
    def aff(pid):
        try:
            return sorted(os.sched_getaffinity(pid))
        except Exception:                                             # noqa: BLE001
            return None
    th = []
    try:
        me = os.getpid()
        for tid in os.listdir(f'/proc/{me}/task'):
            try:
                nm = open(f'/proc/{me}/task/{tid}/comm').read().strip()
            except Exception:                                         # noqa: BLE001
                nm = '?'
            th.append((int(tid), nm, aff(int(tid))))
    except Exception:                                                 # noqa: BLE001
        pass
    main_tid = os.getpid()
    on0 = [t for t in th if t[0] != main_tid and t[2] is not None and 0 in t[2]]
    hist = {}
    for _, _, a in th:
        hist[str(a)] = hist.get(str(a), 0) + 1
    r = dict(layout=dict(main=MAIN, kit_and_planners=KIT, cstop=CSTOP, pool=POOL, smt='邏輯 CPU i 與 i+8 同實體核心；CPU 8 不分配'), blas_env={k: os.environ.get(k) for k in BLAS_ENV},
             main_thread=aff(main_tid), n_threads=len(th), thread_affinity_hist=hist, n_other_threads_allowed_on_cpu0=len(on0), other_threads_on_cpu0_names=sorted({t[1] for t in on0})[:12])
    if pool is not None:
        r['pool'] = [aff(p.pid) for p in pool.procs]
    if csw is not None:
        r['cstop'] = [aff(p.pid) for p in csw.procs]
    r['planners'] = [dict(aff=aff(pl.proc.pid), nice=os.getpriority(os.PRIO_PROCESS, pl.proc.pid)) for pl in planners]
    try:
        import numpy as np
        r['numpy'] = np.__version__
        try:
            import threadpoolctl
            r['threadpools'] = [dict(api=x.get('internal_api'), n=x.get('num_threads'), lib=os.path.basename(str(x.get('filepath')))) for x in threadpoolctl.threadpool_info()]
        except Exception:                                             # noqa: BLE001
            r['threadpools'] = 'threadpoolctl 不可用（以環境變數為準）'
    except Exception:                                                 # noqa: BLE001
        pass
    return r
