#!/usr/bin/env python3
"""有界、非阻塞的逐列紀錄（JSONL）＋迴圈時序統計（Codex 20261025_034022：任務端最小紀錄修正；r2＝Codex 20261025_144446）。

用途：取代任務節點迴圈內「每 100 筆同步重寫整份 task.json」（w1f_60_1 第 1000 筆後出現 0.30 s 任務端更新間隙的候選原因；未確認根因）。

主迴圈只做：
  put(row)：在呼叫端執行緒把 row **深拷貝**（入列後主迴圈再改 row 也不影響已入列內容；成本只與單列大小有關、與已記筆數無關），
             附入列序號 seq 與入列單調時間，put_nowait 進有界佇列；佇列滿 ⇒ 丟棄、計數（不阻塞、不重試）。
背景執行緒做：序列化（json.dumps）、append 寫入、定期 flush、輕量心跳檔覆寫，**以及收尾的最後 flush／close**——全部不在主迴圈。
r2：
  - close(deadline_s)：呼叫端**只做有期限等待**（通知停止＋join(timeout)），不做任何 I/O；最後 flush／close 由寫入端執行緒完成。
    逾期 ⇒ drain_timeout True、complete False，立即返回（寫入端若卡在 I/O 不影響呼叫端時間上限）。
  - 寫入、flush、close 失敗都計數並記第一筆原因，任一 > 0 ⇒ complete False。
  - 心跳寫入失敗計數並記第一筆原因（不影響 complete，但留狀態）。
注意：背景執行緒仍可能競爭 GIL／CPU，put_nowait 不等於已證明時序有界——由慢寫入、溢出、寫入／flush 例外、flush 卡住、主迴圈卡住等
      測試驗證，並以每列的入列單調時間與迴圈時序統計事後核對。

LoopTimer：記每輪迴圈起迄單調時間的耗時統計（最大、p99、超過門檻次數），供區分「來源停更／迴圈延遲／寫入延遲」。
"""
from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time


class BoundedTraceWriter:
    def __init__(self, path, capacity=4096, flush_s=0.5, heartbeat_path=None, heartbeat_s=1.0,
                 dumps=None, opener=open, clock=time.monotonic, write_delay_s=0.0):
        self.path, self.capacity = path, int(capacity)
        self.flush_s, self.heartbeat_path, self.heartbeat_s = float(flush_s), heartbeat_path, float(heartbeat_s)
        self._dumps = dumps or (lambda o: json.dumps(o, ensure_ascii=False, default=float))
        self._opener, self._clock, self._write_delay = opener, clock, float(write_delay_s)   # write_delay_s 只供測試模擬慢寫入
        self._q = queue.Queue(maxsize=self.capacity)
        self._stop = threading.Event()
        self._th = None
        self._fh = None
        self.n_enqueued = self.n_written = self.n_dropped = self.n_write_fail = 0
        self.n_flush_fail = self.n_close_fail = self.n_hb_fail = 0
        self.first_drop = self.first_write_error = self.first_flush_error = self.close_error = self.first_hb_error = None
        self.open_error = None
        self.final_closed = False
        self.max_depth = 0
        self.max_put_s = 0.0
        self.closed = None

    def start(self):
        try:
            self._fh = self._opener(self.path, 'a', encoding='utf-8')
        except Exception as e:                                                  # noqa: BLE001  開檔失敗：照常入列計數，收尾判 incomplete
            self.open_error = f'{type(e).__name__}: {e}'
        self._th = threading.Thread(target=self._run, name='bounded_trace_writer', daemon=True)
        self._th.start()
        return self

    def put(self, row):
        """主迴圈呼叫：不阻塞。回傳是否入列。"""
        c0 = self._clock()
        seq = self.n_enqueued + self.n_dropped + 1
        item = (seq, c0, copy.deepcopy(row))
        try:
            self._q.put_nowait(item)
            self.n_enqueued += 1
            ok = True
        except queue.Full:
            self.n_dropped += 1
            if self.first_drop is None:
                self.first_drop = {'seq': seq, 'wall_mono': c0}
            ok = False
        self.max_depth = max(self.max_depth, self._q.qsize())
        self.max_put_s = max(self.max_put_s, self._clock() - c0)
        return ok

    def _write_one(self, item):
        seq, c_enq, row = item
        try:
            if self._fh is None:
                raise OSError(f'檔案未開啟（{self.open_error}）')
            line = self._dumps({'enq_seq': seq, 'enq_wall_mono': c_enq, 'row': row})
            if self._write_delay:
                time.sleep(self._write_delay)
            self._fh.write(line + '\n')
            self.n_written += 1
        except Exception as e:                                                  # noqa: BLE001  寫入例外只計數、不中斷主迴圈
            self.n_write_fail += 1
            if self.first_write_error is None:
                self.first_write_error = {'seq': seq, 'error': f'{type(e).__name__}: {e}'}

    def _flush(self):
        if self._fh is None:
            return
        try:
            self._fh.flush()
        except Exception as e:                                                  # noqa: BLE001
            self.n_flush_fail += 1
            if self.first_flush_error is None:
                self.first_flush_error = f'{type(e).__name__}: {e}'

    def _heartbeat(self):
        if not self.heartbeat_path:
            return
        try:
            tmp = self.heartbeat_path + '.tmp'
            with self._opener(tmp, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'wall_mono': self._clock(), 'n_enqueued': self.n_enqueued, 'n_written': self.n_written,
                                    'n_dropped': self.n_dropped, 'n_write_fail': self.n_write_fail, 'depth': self._q.qsize()}))
            os.replace(tmp, self.heartbeat_path)
        except Exception as e:                                                  # noqa: BLE001  留狀態，不影響 complete
            self.n_hb_fail += 1
            if self.first_hb_error is None:
                self.first_hb_error = f'{type(e).__name__}: {e}'

    def _run(self):
        last_flush = last_hb = self._clock()
        while True:
            try:
                item = self._q.get(timeout=0.05)
            except queue.Empty:
                item = None
            if item is not None:
                self._write_one(item)
            now = self._clock()
            if now - last_flush >= self.flush_s:
                self._flush()
                last_flush = now
            if now - last_hb >= self.heartbeat_s:
                self._heartbeat()
                last_hb = now
            if self._stop.is_set() and self._q.empty():
                break
        self._flush()                                                           # r2：最後 flush／close 在寫入端執行緒
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception as e:                                              # noqa: BLE001
                self.n_close_fail += 1
                self.close_error = f'{type(e).__name__}: {e}'
        self._heartbeat()
        self.final_closed = True

    def close(self, deadline_s=5.0):
        """呼叫端只做有期限等待（不做 I/O）；回傳統計（complete 為嚴格 bool）。重複呼叫回傳同一結果。"""
        if self.closed is not None:
            return self.closed
        c0 = self._clock()
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=max(0.0, float(deadline_s)))
        timeout = bool(self._th is not None and self._th.is_alive())
        left = self._q.qsize()
        self.closed = {'path': self.path, 'capacity': self.capacity, 'n_enqueued': self.n_enqueued, 'n_written': self.n_written,
                       'n_dropped': self.n_dropped, 'n_write_fail': self.n_write_fail, 'n_flush_fail': self.n_flush_fail,
                       'n_close_fail': self.n_close_fail, 'n_hb_fail': self.n_hb_fail, 'n_left_in_queue': left,
                       'first_drop': self.first_drop, 'first_write_error': self.first_write_error,
                       'first_flush_error': self.first_flush_error, 'close_error': self.close_error, 'first_hb_error': self.first_hb_error,
                       'open_error': self.open_error, 'final_closed': bool(self.final_closed and not timeout),
                       'max_depth': self.max_depth, 'max_put_s': self.max_put_s, 'drain_timeout': timeout,
                       'close_wall_s': self._clock() - c0, 'deadline_s': float(deadline_s),
                       'complete': bool(not timeout and self.final_closed and self.open_error is None and self.n_dropped == 0
                                        and self.n_write_fail == 0 and self.n_flush_fail == 0 and self.n_close_fail == 0
                                        and left == 0 and self.n_written == self.n_enqueued)}
        return self.closed


class LoopTimer:
    """每輪迴圈起迄單調時間；耗時統計（秒）。"""

    def __init__(self, over_s=0.1, keep=20000):
        self.over_s, self.keep = float(over_s), int(keep)
        self.durs = []
        self.n = self.n_over = 0
        self.max_s = 0.0
        self.worst = None

    def record(self, w_start, w_end, seq=None):
        d = float(w_end - w_start)
        self.n += 1
        if len(self.durs) < self.keep:
            self.durs.append(d)
        if d > self.max_s:
            self.max_s, self.worst = d, {'seq': seq, 'w_start': w_start, 'dur_s': d}
        if d > self.over_s:
            self.n_over += 1

    def stats(self):
        s = sorted(self.durs)
        p99 = s[min(len(s) - 1, int(0.99 * len(s)))] if s else None
        return {'n': self.n, 'max_s': self.max_s, 'p99_s': p99, 'over_s': self.over_s, 'n_over': self.n_over, 'worst': self.worst}
