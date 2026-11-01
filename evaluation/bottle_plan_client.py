#!/usr/bin/env python3
"""在線規劃工作者的用戶端（控制器端；Codex 20261102_072949 第 5 項：有界、非阻塞 IPC）。
子程序：系統 python（最小環境變數，不繼承 Isaac 的 PYTHONHOME／PYTHONPATH／LD_LIBRARY_PATH），close_fds＝True（不留住 Isaac 互斥鎖 fd），
stdin／stdout 管道只走 JSON 協定，子程序 stderr 寫到日誌檔。
控制拍只呼叫 submit()／poll()：submit 放進有界佇列（滿或子程序已死 ⇒ 回 None，不阻塞），寫入由寫入執行緒負責；
讀取執行緒逐行解析，schema、id（必須是未結案的請求）、kind、有限值不符 ⇒ 計入 bad／stale，不交給控制器。close() 關 stdin、等 1 s、再 terminate／kill。
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, 'bottle_plan_worker.py')
SCHEMA = 1


def _finite(x):
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def _jsonable(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class PlanClient:
    def __init__(self, py='/usr/bin/python3', log_path=None, max_queue=8):
        self.py, self.log_path, self.max_queue = py, log_path, max_queue
        self.proc, self.q, self.res, self.open = None, None, {}, {}
        self.lock = threading.Lock()
        self.st = dict(n_sent=0, n_recv=0, n_bad=0, n_stale=0, n_drop_full=0, n_drop_dead=0, t_start_wall=None, t_ping_wall=None, exit_code=None, worker_error=[])
        self._next = 1

    def start(self):
        env = dict(PATH='/usr/local/bin:/usr/bin:/bin', HOME=os.environ.get('HOME', '/tmp'), LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONIOENCODING='utf-8')
        err = open(self.log_path, 'a') if self.log_path else subprocess.DEVNULL
        self.st['t_start_wall'] = time.time()
        self.proc = subprocess.Popen([self.py, '-u', WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env, close_fds=True, text=True, bufsize=1, cwd=HERE)
        self.q = queue.Queue(maxsize=self.max_queue)
        threading.Thread(target=self._writer, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()
        self.ping_id = self.submit('ping', {})
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def submit(self, kind, inputs):
        """非阻塞：回請求 id；佇列滿或子程序已死 ⇒ None（呼叫端照未送出處置）。"""
        if not self.alive():
            self.st['n_drop_dead'] += 1
            return None
        with self.lock:
            rid = self._next
            self._next += 1
            self.open[rid] = dict(kind=kind, t_wall=time.time())
        try:
            self.q.put_nowait(json.dumps(dict(schema=SCHEMA, id=rid, kind=kind, inputs=_jsonable(inputs))) + '\n')
        except queue.Full:
            with self.lock:
                self.open.pop(rid, None)
            self.st['n_drop_full'] += 1
            return None
        self.st['n_sent'] += 1
        return rid

    def poll(self, rid):
        """非阻塞：結果到了就取出（dict），否則 None。"""
        with self.lock:
            return self.res.pop(rid, None)

    def _writer(self):
        while True:
            line = self.q.get()
            if line is None:
                break
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
            except Exception as e:                                       # 子程序已死：之後的 submit 由 alive() 擋下
                self.st['worker_error'].append(f'write:{e!r}')
                break

    def _reader(self):
        for line in self.proc.stdout:
            t = time.time()
            try:
                m = json.loads(line)
                ok = int(m.get('schema', -1)) == SCHEMA and isinstance(m.get('id'), int) and isinstance(m.get('result'), dict) and _finite(m.get('result'))
            except Exception:
                ok = False
            if not ok:
                self.st['n_bad'] += 1
                continue
            with self.lock:
                o = self.open.pop(m['id'], None)
                if o is None or o['kind'] != m.get('kind'):
                    self.st['n_stale'] += 1
                    continue
                m['t_wall_submit'], m['t_wall_recv'] = o['t_wall'], t
                if m['kind'] == 'ping':
                    self.st['t_ping_wall'] = t
                    self.st['ping'] = m.get('result')
                    continue
                self.res[m['id']] = m
            self.st['n_recv'] += 1
        self.st['exit_code'] = self.proc.poll()

    def close(self):
        if self.proc is None:
            return
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.st['exit_code'] = self.proc.poll()

    def stats(self):
        s = dict(self.st)
        if s['t_start_wall'] and s['t_ping_wall']:
            s['cold_start_wall_s'] = s['t_ping_wall'] - s['t_start_wall']
        s['n_open'] = len(self.open)
        s['alive'] = self.alive()
        return s
