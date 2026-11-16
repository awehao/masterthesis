#!/usr/bin/env python3
"""非阻塞執行（Howard 11/17 主線第 2 段 C；Codex 20261117 執行契約）：物理不為控制器計算暫停。

控制器（MovingGraspCtrl 等，每物理步呼叫 step、每 NPH 步一個控制拍）放在 fork 的工作程序裡，執行器的物理迴圈獨立推進。
「提前一拍」：
  拍 k（物理步 n_k）執行器送實測快照 q(n_k) 後，立刻繼續執行已提交的名目段 S_k（n_k..n_k+NPH−1 每步的設定點與底盤命令）。
  工作程序以控制器自己的受控對象模型（S.plant_phys_step）把 q(n_k) 沿 S_k 推到 n_{k+1}，在拍 k+1 用預測狀態呼叫控制器，
  再推 NPH−1 步，得名目段 S_{k+1}；停止尾段 tail_{k+1}＝控制器在該拍「候選提交後下一拍起停」（STOP_EXTRAS 的 extra＝1）停止模擬的逐步輸出
  ——該停止模擬本來就是控制器每拍放行候選前驗過的（淨空、限位、停穩）；前 NPH 步必須等於 S_{k+1}（逐值核對，否則視為不一致）。
  控制器已進停止模式時，尾段由工作程序在名目段結束處另做一次停止模擬（e＝0）。
執行器在 n_{k+1}：S_{k+1} 已到、版本相符、從送快照到收到的牆鐘 ≤ 期限 ⇒ 提交並執行；否則執行 tail_k（上一段附帶的已核停止尾段），
  控制器即作廢（它的內部狀態假設了 S_{k+1} 會被執行），照實記為任務中斷。不在晚到後插入結果、不重設管線。
時間：送快照、工作程序開始／完成、執行器收到、提交步分別記（模擬步序號與牆鐘）；計算、通訊、8 步管線延遲分開。"""
import multiprocessing as mp
import os
import time

import numpy as np

PHYS = 0.01


def _plant(S, cfg, q, s_prev, s_next, base):
    """由控制器輸出還原彈出管線的命令並推一步（控制器自己的受控對象模型）。"""
    u = np.r_[np.asarray(base, float), (np.asarray(s_next, float) - np.asarray(s_prev, float)) / PHYS]
    q2, s2 = S.plant_phys_step(q, np.asarray(s_prev, float), u, cfg, 1)
    return q2


def _worker(conn, ctrl, S, nph, phi0, finger0, late_ticks):
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass
    seg_prev, s_before = None, None
    k_tick = 0
    while True:
        msg = conn.recv()
        if msg is None:
            break
        op = msg[0]
        if op == 'call':                                              # 同步呼叫控制器方法（交接核對等）；呼叫端自己決定是否在時間帳內
            _, name, args = msg
            try:
                conn.send(('call', getattr(ctrl, name)(*args)))
            except Exception as e:                                    # noqa: BLE001
                conn.send(('call_error', repr(e)[:300]))
            continue
        if op == 'get':
            conn.send(('get', {k: getattr(ctrl, k, None) for k in msg[1]}))
            continue
        _, snap = msg
        w_start = time.perf_counter()
        try:
            q = np.asarray(snap['q'], float).copy()
            t_next = float(snap['t_next'])
            phi = float(snap.get('phi', phi0))
            fm = snap.get('finger', finger0)
            if snap.get('first'):                                     # 初始化：第一段以實測狀態在 n_0 直接算（執行器尚未起跑）
                qp = q
                s_cur = np.asarray(ctrl.s, float).copy()
            else:                                                     # 沿已提交的 S_k 預測到 n_{k+1}
                qp = q
                s_cur = s_before.copy()
                for j in range(nph):
                    qp = _plant(S, ctrl.cfg, qp, s_cur, seg_prev[j]['arm_target'], seg_prev[j]['base_twist'])
                    s_cur = np.asarray(seg_prev[j]['arm_target'], float)
            if k_tick in late_ticks:                                  # 核對用：注入延遲（模擬計算晚到）
                time.sleep(late_ticks[k_tick])
            ctrl._rec_tail = {1: []}
            seg, qj, sj = [], qp.copy(), s_cur.copy()
            for j in range(nph):
                o = ctrl.step(t_next + j * PHYS, qj.copy(), phi, finger_meas=fm)
                seg.append(dict(arm_target=np.asarray(o['arm_target'], float).copy(), base_twist=np.asarray(o['base_twist'], float).copy(), finger=float(o['finger']),
                                ready=bool(o.get('ready')), ready_ver=o.get('ready_ver'), stopped=bool(o.get('stopped')), abort=bool(o.get('abort'))))
                qj = _plant(S, ctrl.cfg, qj, sj, seg[-1]['arm_target'], seg[-1]['base_twist'])
                sj = seg[-1]['arm_target']
            rec = ctrl._rec_tail.get(1) or []
            ctrl._rec_tail = None
            tail, tail_src, mism = None, None, None
            if ctrl.stop is None and len(rec) >= nph:
                mism = max(max(float(np.abs(rec[j][0] - seg[j]['arm_target']).max()), float(np.abs(rec[j][1] - seg[j]['base_twist']).max())) for j in range(nph))
                if mism <= 1e-9:
                    tail, tail_src = [(a.copy(), b.copy()) for a, b in rec[nph:]], 'candidate_extra1'
            if tail is None and ctrl.abort is None:                   # 停止模式（或候選尾段不一致）：在名目段結束處另做停止模擬
                rt = {0: []}
                ctrl._rec_tail = rt
                try:
                    _, vtr, _ = ctrl._stop_traj(qj, ctrl.u_last, 0, ctrl.ramp_j, ctrl.u_from_base)
                finally:
                    ctrl._rec_tail = None
                if vtr is None and rt[0]:
                    tail, tail_src = [(a.copy(), b.copy()) for a, b in rt[0]], 'explicit_stop_e0'
            seg_prev, s_before = seg, s_cur.copy()                    # 下一拍預測：由 S_{k+1} 之前的設定點沿 S_{k+1} 推
            out = dict(ver=snap['ver'], seg=seg, tail=tail, tail_src=tail_src, tail_mismatch=mism, w_start=w_start, w_done=time.perf_counter(),
                       stop=ctrl.stop, abort=ctrl.abort, phase=getattr(ctrl, 'phase', None), k_tick=k_tick, q_pred=qp.tolist())
            conn.send(('seg', out))
        except Exception as e:                                        # noqa: BLE001
            import traceback
            conn.send(('error', dict(ver=snap.get('ver'), err=traceback.format_exc()[-1500:])))
        k_tick += 1


class AheadRunner:
    """執行器端：持有工作程序、已提交段、尾段、時間帳。"""

    def __init__(self, ctrl, S, nph, deadline_ms=50.0, phi0=0.0, finger0=None, late_ticks=None):
        ctx = mp.get_context('fork')
        self.conn, c2 = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(c2, ctrl, S, nph, phi0, finger0, dict(late_ticks or {})), daemon=True)
        self.proc.start()
        c2.close()
        self.nph, self.deadline_ms = nph, float(deadline_ms)
        self.ver, self.pending, self.seg, self.tail, self.mode = 0, None, None, None, 'init'
        self.log, self.n_commit, self.late = [], 0, None
        self.j = 0                                                    # 目前段內步序
        self.got = None                                               # 已收到但尚未到提交步的結果（tag, out, 收到牆鐘）

    def _send(self, q, t_next, n_send, first=False, phi=None, finger=None):
        self.ver += 1
        snap = dict(ver=self.ver, q=np.asarray(q, float).tolist(), t_next=float(t_next), first=first)
        if phi is not None:
            snap['phi'] = float(phi)
        if finger is not None:
            snap['finger'] = finger
        self.pending = dict(ver=self.ver, n_send=int(n_send), w_send=time.perf_counter())
        self.conn.send(('tick', snap))

    def start(self, q, t0, phi=None, finger=None, timeout=5.0):
        """初始化（執行器起跑前）：第一段以實測狀態直接算。"""
        self._send(q, t0, 0, first=True, phi=phi, finger=finger)
        if not self.conn.poll(timeout):
            raise RuntimeError('ahead worker init timeout')
        tag, out = self.conn.recv()
        if tag != 'seg':
            raise RuntimeError('ahead worker init error: ' + str(out)[:800])
        self.seg, self.tail, self.mode, self.j = out['seg'], out['tail'], 'nominal', 0
        self.pending = None
        self.log.append(dict(kind='init', ver=out['ver'], tail_src=out['tail_src'], compute_ms=round((out['w_done'] - out['w_start']) * 1e3, 2)))
        return out

    def poll(self):
        """每個物理步呼叫：結果一到就收下並記收到時刻（不等拍邊界）。"""
        if self.pending is not None and self.got is None and self.conn.poll(0):
            tag, out = self.conn.recv()
            self.got = (tag, out, time.perf_counter())

    def tick_boundary(self, q, t_next, n, phi=None, finger=None):
        """在拍邊界 n_k（套用本步輸出之前）呼叫：決定 S_k 是否提交（第一次呼叫時 S_0 已在 start() 準備好），並送下一拍快照。"""
        rec = dict(n=int(n))
        if self.mode == 'nominal' and self.pending is not None:
            self.poll()
            got, self.got = self.got, None
            p = self.pending
            if got is not None and got[0] == 'seg' and got[1]['ver'] == p['ver']:
                o = got[1]
                rt_ms = (got[2] - p['w_send']) * 1e3
                rec.update(ver=o['ver'], comm_in_ms=round((o['w_start'] - p['w_send']) * 1e3, 3), compute_ms=round((o['w_done'] - o['w_start']) * 1e3, 2),
                           comm_out_ms=round((got[2] - o['w_done']) * 1e3, 3), roundtrip_ms=round(rt_ms, 2), steps_since_send=int(n - p['n_send']), tail_src=o['tail_src'], tail_mismatch=o['tail_mismatch'])
                if rt_ms <= self.deadline_ms and o['seg'] is not None and o['abort'] is None:
                    self.seg, self.j = o['seg'], 0
                    if o['tail'] is not None:
                        self.tail = o['tail']
                    self.n_commit += 1
                    rec['commit'] = True
                else:
                    rec['commit'] = False
                    rec['why'] = 'deadline' if rt_ms > self.deadline_ms else 'abort_or_empty'
            else:
                rec['commit'] = False
                rec['why'] = 'not_ready' if got is None else ('error:' + str(got[1])[:300] if got[0] == 'error' else 'version_mismatch')
                rec['steps_since_send'] = int(n - p['n_send'])
                rec['wall_since_send_ms'] = round((time.perf_counter() - p['w_send']) * 1e3, 2)
            if not rec['commit']:                                     # 晚到／錯誤：執行上一段附帶的已核停止尾段；控制器作廢
                self.mode = 'tail' if self.tail is not None else 'no_tail'
                self.late = dict(rec)
                self.seg, self.j = None, 0
                self.pending = None
        if self.mode == 'nominal':
            self._send(q, t_next, n, phi=phi, finger=finger)
        self.log.append(rec)
        return rec

    def output(self):
        """本步要套用的輸出：dict(arm_target, base_twist, finger, ready, ready_ver, stopped) 或 None（尾段已執行完＝停住）。"""
        if self.mode == 'nominal':
            o = self.seg[self.j]
            self.j += 1
            return o
        if self.mode == 'tail' and self.tail:
            a, b = self.tail.pop(0)
            return dict(arm_target=a, base_twist=b, finger=None, ready=False, ready_ver=None, stopped=True, abort=False)
        return None

    def call(self, name, *args, timeout=2.0):
        self.conn.send(('call', name, args))
        if not self.conn.poll(timeout):
            return ('timeout', None)
        return self.conn.recv()

    def get(self, names, timeout=2.0):
        self.conn.send(('get', list(names)))
        if not self.conn.poll(timeout):
            return None
        return self.conn.recv()[1]

    def close(self):
        try:
            self.conn.send(None)
        except Exception:                                             # noqa: BLE001
            pass
        self.proc.join(timeout=1.0)
        if self.proc.is_alive():
            self.proc.terminate()
