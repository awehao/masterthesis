#!/usr/bin/env python3
"""G3：停止掃掠淨空的平行計算池（Codex 20261113_120619 核可條件）。
  工作程序在控制器建立時預先 fork（之後不再建立）；每個工作訊息自帶完整快照——把手位姿 T_WH0、手指開度、門角基準 PHI0、樣本構型——
  工作程序每次依訊息設定自己的模組值後才算，不共享任何會變動的全域設定。
  每次派工有版本號；收集時只收同版本的結果，舊版本（例如上一個逾時工作遲到的結果）一律丟棄並計數，不得用來放行。
  逾時、工作程序死亡、工作程序例外、結果缺漏 ⇒ 回 (None, 原因)：呼叫端必須當作「未通過」，依既定失效政策處理。
  淨空函式＝door_clear_door_fast.clear_door_fast（與 door_wbopt.clear_door 3362 構型逐值相同）；樣本與覆蓋不變，只是分給多個程序。
  時間分項：準備（呼叫端）、派工（序列化＋送出）、等待、彙整，各自記錄。"""
import multiprocessing as mp
import time

import numpy as np

_OUT_KEYS = ('env', 'base', 'self', 'shell_bar')


class _FB:
    """door_exit_ctrl.DoorExitCtrl.finger_bar 只讀 T_WH_bind 與 finger_q：用同一函式（同式）算兩指—橫桿距離。"""
    def __init__(self, T, fq):
        self.T_WH_bind, self.finger_q = T, fq


def _die_with_parent():
    """父程序死亡（含段錯誤）時本程序收 SIGTERM（Linux prctl PR_SET_PDEATHSIG），避免孤兒程序佔住 Isaac 互斥檢查。"""
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass


def _worker(conn):
    _die_with_parent()
    import door_wbopt as W
    import door_clear_door_fast as F
    import door_exit_ctrl as XE
    import door_clear_fast as FAST
    for n in W.DOC.CL:                                               # 預熱：自碰撞 KD 樹（與延遲建立的是同一棵：同點雲同建法）、靜態幾何快取
        if n not in FAST._TREE:
            FAST._TREE[n] = FAST.cKDTree(W.DOC.CL[n])
    W._static()
    FAST._pairs()
    conn.send(('ready',))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        ver, w, T, fq, phi0, X, delay, fb = msg
        try:
            if delay:
                time.sleep(delay)                                    # 測試用：注入延遲
            W.FINGER_Q, W.PHI0 = float(fq), float(phi0)
            res = []
            fbo = None if fb is None else _FB(T, float(fb))
            for x in X:
                r = F.clear_door_fast(x, T)
                res.append((r['env'], r['base'], r['self'], r['shell_bar'], r['who']['env'], r['who']['base'],
                            None if fbo is None else XE.DoorExitCtrl.finger_bar(fbo, x[:9], float(x[9]))))
            conn.send((ver, w, 'ok', res))
        except Exception as e:                                        # noqa: BLE001
            conn.send((ver, w, 'error', repr(e)))


class SweepPool:
    def __init__(self, n):
        ctx = mp.get_context('fork')
        self.n = int(n)
        self.conns, self.procs = [], []
        for _ in range(self.n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p)
        t_w = time.perf_counter()
        for c in self.conns:                                           # 等工作程序預熱完成（建池時，不在控制拍內）
            if not c.poll(30.0) or c.recv() != ('ready',):
                raise RuntimeError('sweep worker warm-up failed')
        self.warm_ms = (time.perf_counter() - t_w) * 1e3
        self.ver = 0
        self.stats = dict(jobs=0, samples=0, timeouts=0, dead=0, errors=0, stale_dropped=0)
        self.inject_delay = {}                                       # 測試用：{worker 索引: 秒}

    def submit(self, T_WH0, fq, phi0, X, finger_bar_q=None):
        t0 = time.perf_counter()
        self.ver += 1
        X = np.asarray(X, float)
        chunks = [c for c in np.array_split(np.arange(len(X)), self.n)]
        T = np.asarray(T_WH0, float)
        sent = []
        for w, idx in enumerate(chunks):
            if len(idx) == 0:
                continue
            try:
                self.conns[w].send((self.ver, w, T, float(fq), float(phi0), X[idx], self.inject_delay.get(w, 0.0), finger_bar_q))
                sent.append(w)
            except (BrokenPipeError, OSError):
                return dict(ver=self.ver, chunks=chunks, sent=sent, t_submit=t0, t_dispatch_ms=(time.perf_counter() - t0) * 1e3, n=len(X), send_failed=w)
        self.stats['jobs'] += 1
        self.stats['samples'] += len(X)
        return dict(ver=self.ver, chunks=chunks, sent=sent, t_submit=t0, t_dispatch_ms=(time.perf_counter() - t0) * 1e3, n=len(X), send_failed=None)

    def collect(self, h, timeout):
        """回 (結果列表或 None, 原因, 時間分項 ms)。結果依原樣本順序，每筆 dict(env, base, self, shell_bar, who)。"""
        t0 = time.perf_counter()
        if h.get('send_failed') is not None:
            self.stats['dead'] += 1
            return None, 'worker_dead', dict(dispatch=h['t_dispatch_ms'], wait=0.0, gather=0.0)
        deadline = h['t_submit'] + float(timeout)
        got = {}
        for w in h['sent']:
            conn = self.conns[w]
            while True:
                rem = deadline - time.perf_counter()
                try:
                    ready = rem > 0 and conn.poll(rem)
                except (EOFError, OSError):
                    self.stats['dead'] += 1
                    return None, 'worker_dead', dict(dispatch=h['t_dispatch_ms'], wait=(time.perf_counter() - t0) * 1e3, gather=0.0)
                if not ready:
                    if not self.procs[w].is_alive():
                        self.stats['dead'] += 1
                        return None, 'worker_dead', dict(dispatch=h['t_dispatch_ms'], wait=(time.perf_counter() - t0) * 1e3, gather=0.0)
                    self.stats['timeouts'] += 1
                    return None, 'timeout', dict(dispatch=h['t_dispatch_ms'], wait=(time.perf_counter() - t0) * 1e3, gather=0.0)
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    self.stats['dead'] += 1
                    return None, 'worker_dead', dict(dispatch=h['t_dispatch_ms'], wait=(time.perf_counter() - t0) * 1e3, gather=0.0)
                if msg[0] != h['ver']:
                    self.stats['stale_dropped'] += 1                 # 舊版本結果：丟棄，不得放行
                    continue
                if msg[2] != 'ok':
                    self.stats['errors'] += 1
                    return None, 'worker_error:' + str(msg[3])[:200], dict(dispatch=h['t_dispatch_ms'], wait=(time.perf_counter() - t0) * 1e3, gather=0.0)
                got[w] = msg[3]
                break
        t1 = time.perf_counter()
        out = [None] * h['n']
        for w in h['sent']:
            idx = h['chunks'][w]
            if len(got[w]) != len(idx):
                self.stats['errors'] += 1
                return None, 'missing_results', dict(dispatch=h['t_dispatch_ms'], wait=(t1 - t0) * 1e3, gather=0.0)
            for j, r in zip(idx, got[w]):
                out[int(j)] = dict(env=r[0], base=r[1], self=r[2], shell_bar=r[3], who=dict(env=r[4], base=r[5]))
                if r[6] is not None:
                    out[int(j)]['finger_bar'] = r[6]
        if any(o is None for o in out):
            self.stats['errors'] += 1
            return None, 'missing_results', dict(dispatch=h['t_dispatch_ms'], wait=(t1 - t0) * 1e3, gather=0.0)
        return out, 'ok', dict(dispatch=h['t_dispatch_ms'], wait=(t1 - t0) * 1e3, gather=(time.perf_counter() - t1) * 1e3)

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
