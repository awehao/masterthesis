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
import math
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


def _seg_entry(o, fm=None):
    return dict(arm_target=np.asarray(o['arm_target'], float).copy(), base_twist=np.asarray(o['base_twist'], float).copy(), finger=None if o.get('finger') is None else float(o['finger']),
                ready=bool(o.get('ready')), ready_ver=o.get('ready_ver'), stopped=bool(o.get('stopped')), abort=bool(o.get('abort')), done=bool(o.get('done')))


def _worker(conn, ctrl, S, nph, phi0, finger0, late_ticks, door=None, builder=None):
    """door＝door_ahead_open.DoorOpenPhase（可選）：抓持 READY 後在本程序內交接給開門執行層，之後同一套提前一拍協定。"""
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass
    import gc as _gc
    if os.environ.get('AHEAD_GC_FREEZE') == '1':                      # 執行期最佳化（不改任何判定）：起跑前把既有物件移到永久代，減少第二代回收停頓
        _gc.collect()
        _gc.freeze()
    seg_prev, s_before, cfg_prev = None, None, None
    mode = 'grasp'
    k_tick, k_open = 0, 0
    while True:
        msg = conn.recv()
        if msg is None:
            break
        op = msg[0]
        if op == 'call':                                              # 同步呼叫控制器方法（交接核對等）；呼叫端自己決定是否在時間帳內
            _, name, args = msg
            try:
                tgt = door if name.startswith('door.') else ctrl
                conn.send(('call', getattr(tgt, name.split('.', 1)[-1])(*args)))
            except Exception as e:                                    # noqa: BLE001
                conn.send(('call_error', repr(e)[:300]))
            continue
        if op == 'build':                                             # D2：交權請求到才建控制器（builder(packet) → 控制器；工作程序內建、不經 fork 傳物件）
            w_b = time.perf_counter()
            try:
                ctrl = builder(msg[1])
                seg_prev, s_before, cfg_prev, mode, k_tick, k_open = None, None, None, 'grasp', 0, 0
                if os.environ.get('AHEAD_GC_FREEZE') == '1':          # 起跑後才建的控制器也移到永久代（同起跑前的凍結；不改判定）
                    _gc.collect()
                    _gc.freeze()
                conn.send(('built', dict(ok=True, ms=round((time.perf_counter() - w_b) * 1e3, 2), ver=msg[1].get('ver'))))
            except Exception as e:                                    # noqa: BLE001
                import traceback
                conn.send(('built', dict(ok=False, err=traceback.format_exc()[-1200:], ver=msg[1].get('ver'))))
            continue
        if op == 'get':
            conn.send(('get', dict({k: getattr(ctrl, k, None) for k in msg[1]}, _mode=mode)))
            continue
        _, snap = msg
        w_start = time.perf_counter()
        try:
            q = np.asarray(snap['q'], float).copy()
            t_next = float(snap['t_next'])
            if snap.get('door_rebind') is not None and door is not None:  # D2：首段結果到達後隨快照改綁（計入本拍計算時間）
                door.rebind_first(**snap['door_rebind'])
            phi = float(snap.get('phi', phi0))
            fm = snap.get('finger', finger0)
            if snap.get('first'):                                     # 初始化：第一段以實測狀態在 n_0 直接算（執行器尚未起跑）
                qp = q
                s_cur = np.asarray(ctrl.s, float).copy()
                if door is not None and getattr(door, 'pre_k2', False):
                    door.prerequest()                                 # 第一個續段資料已齊 ⇒ 抓持段起點即送規劃程序
            else:                                                     # 沿已提交的 S_k 預測到 n_{k+1}（用產生 S_k 的那個控制器的受控對象模型）
                qp = q
                s_cur = s_before.copy()
                for j in range(nph):
                    qp = _plant(S, cfg_prev, qp, s_cur, seg_prev[j]['arm_target'], seg_prev[j]['base_twist'])
                    s_cur = np.asarray(seg_prev[j]['arm_target'], float)
            if k_tick in late_ticks:                                  # 核對用：注入延遲（模擬計算晚到）
                time.sleep(late_ticks[k_tick])
            if mode == 'open' and ('o%d' % k_open) in late_ticks:      # 開門段第 k_open 拍（交接後起算）
                time.sleep(late_ticks['o%d' % k_open])
            events, o0, bind = [], None, None
            if mode == 'grasp' and door is not None and ctrl.ready_now and ctrl.ready_ver is not None and ctrl.stop is None:
                ex, o0, hrec = door.handover(ctrl, qp, t_next)        # 試跑即正式：通過 ⇒ 第一步已算完
                events.append(hrec)
                if ex is not None:
                    ctrl, mode = ex, 'open'
                    bind = dict(T_EH=ex.T_EH.copy(), ang_bind=float(ex.ang_bind), phi_bind=float(ex.phi_bind))
            if mode == 'open' and o0 is None:
                phi_p = ctrl.phi_of(qp)[0]                            # 綁定中：門角由預測構型反算（控制器自己的理想閉鏈模型）
                ctrl, o0, ev_ = door.pre_tick(ctrl, t_next, qp, phi_p)
                events += ev_
            seg, qj, sj = [], qp.copy(), s_cur.copy()
            prop = None
            if mode == 'grasp':
                ctrl._rec_tail = {1: []}
            elif o0 is None:
                ctrl._cand_out = None
            for j in range(nph):
                if j == 0 and o0 is not None:
                    o = o0
                elif mode == 'open':
                    o = ctrl.step(t_next + j * PHYS, qj.copy(), ctrl.phi_of(qj)[0])
                else:
                    o = ctrl.step(t_next + j * PHYS, qj.copy(), phi, finger_meas=fm)
                seg.append(_seg_entry(o))
                if j == 0:                                            # 本拍提案（進管線的拍命令）：發 /cmd_vel_grasp 供安全鏈放行核對
                    prop = np.asarray(ctrl.u_last[:3], float).copy()
                qj = _plant(S, ctrl.cfg, qj, sj, seg[-1]['arm_target'], seg[-1]['base_twist'])
                sj = seg[-1]['arm_target']
            tail, tail_src, mism = None, None, None
            if mode == 'open':
                tail, tail_src, mism = door.tail(ctrl, seg, qj, nph)
            else:
                rec = ctrl._rec_tail.get(1) or []
                ctrl._rec_tail = None
                if ctrl.stop is None and len(rec) >= nph:
                    mism = max(max(float(np.abs(rec[j][0] - seg[j]['arm_target']).max()), float(np.abs(rec[j][1] - seg[j]['base_twist']).max())) for j in range(nph))
                    if mism <= 1e-9:
                        tail, tail_src = [(a.copy(), b.copy()) for a, b in rec[nph:]], 'candidate_extra1'
                if tail is None and getattr(ctrl, 'tail_cert_mode', False):
                    tail_src = 'none_cert_mode'                       # 尾段證書模式（Codex 20261117_023209）：不補生未經完整淨空掃掠的 e＝0 尾段；沒有合格新證書 ⇒ 不提交、執行器沿用上一份已核尾段
                elif tail is None and ctrl.abort is None:             # 停止模式（或候選尾段不一致）：在名目段結束處另做停止模擬
                    rt = {0: []}
                    ctrl._rec_tail = rt
                    try:
                        _, vtr, _ = ctrl._stop_traj(qj, ctrl.u_last, 0, ctrl.ramp_j, ctrl.u_from_base)
                    finally:
                        ctrl._rec_tail = None
                    if vtr is None and rt[0]:
                        tail, tail_src = [(a.copy(), b.copy()) for a, b in rt[0]], 'explicit_stop_e0'
            seg_prev, s_before, cfg_prev = seg, s_cur.copy(), ctrl.cfg    # 下一拍預測：由 S_{k+1} 之前的設定點沿 S_{k+1} 推
            parts = None
            if mode == 'grasp' and getattr(ctrl, 'tick_log', None):      # 診斷：抓持控制器本拍耗時與計算池最近一次分項
                tl = ctrl.tick_log[-1]
                pm = ctrl.pool_ms[-1] if getattr(ctrl, 'pool_ms', None) else None
                parts = dict(tick_ms=round(float(tl[1]), 2), phase=getattr(ctrl, 'phase', None), pool=pm if not isinstance(pm, dict) else {k_: round(float(v_), 2) for k_, v_ in pm.items() if isinstance(v_, (int, float))})
                if getattr(ctrl, '_prof', None) is not None:          # AHEAD_PROF：本段（5 步）各方法累計 ms
                    parts['prof'] = {k_: round(v_, 2) for k_, v_ in ctrl._prof.items()}
                    ctrl._prof = {}
            if mode == 'open':                                        # 診斷：控制器本拍各分項（ms）與工作程序額外開銷
                parts = {k_: round(float(getattr(ctrl, k_)[-1]), 2) for k_ in ('tick_ms', 'solve_ms', 'sweep_ms', 'cl_ms', 'cand_ms', 'mon_ms') if getattr(ctrl, k_, None)}
                parts['cand'] = getattr(ctrl, '_cand_diag', None)
                parts['shield_min'] = None if door is None or not math.isfinite(getattr(door, 'shield_min', float('inf'))) else round(float(door.shield_min), 4)   # 累計最小 Shield 約束殘差（m/s）
                parts['qp'] = [round(float(x), 2) for x in ctrl.qp_log[-1]] if getattr(ctrl, 'qp_log', None) else None   # t、已停止、前置、mpc.solve、組裝、GI、OSQP、求解器(0=gi/1=osqp)
            out = dict(ver=snap['ver'], seg=seg, prop=prop, tail=tail, tail_src=tail_src, tail_mismatch=mism, w_start=w_start, w_done=time.perf_counter(), mode=mode, bind=bind, events=events, parts=parts, gc_n=_gc.get_count()[2] if _gc else None,
                       stop=ctrl.stop, abort=ctrl.abort, done=bool(getattr(ctrl, 'done', False)), phase=getattr(ctrl, 'phase', None), k_tick=k_tick, q_pred=qp.tolist(),
                       sigma=float(getattr(ctrl, 'sigma', float('nan'))) if mode == 'open' else None, stop_mode=getattr(ctrl, 'stop_mode', None) if mode == 'open' else None)
            if tail is not None:                                      # 尾段打包成單一陣列（每列＝設定點 6＋底盤命令 3），減少傳輸
                out['tail'] = np.array([np.r_[a_, b_] for a_, b_ in tail], float)
            conn.send(('seg', out))
            if mode == 'grasp' and door is not None and ctrl.ready_now and ctrl.ready_ver is not None and ctrl.stop is None and ctrl.abort is None:
                door.prebuild(ctrl, qj)                               # 空檔預建（下一快照到達前）；交接拍再以該拍預測構型重設依賴構型的兩項
        except Exception as e:                                        # noqa: BLE001
            import traceback
            conn.send(('error', dict(ver=snap.get('ver'), err=traceback.format_exc()[-1500:])))
        k_tick += 1
        if mode == 'open':
            k_open += 1


class AheadRunner:
    """執行器端：持有工作程序、已提交段、尾段、時間帳。"""

    def __init__(self, ctrl, S, nph, deadline_ms=50.0, phi0=0.0, finger0=None, late_ticks=None, door=None, builder=None):
        ctx = mp.get_context('fork')
        self.conn, c2 = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(c2, ctrl, S, nph, phi0, finger0, dict(late_ticks or {}), door, builder), daemon=True)
        self.proc.start()
        c2.close()
        self.nph, self.deadline_ms = nph, float(deadline_ms)
        self.ver, self.pending, self.seg, self.tail, self.tail_i, self.mode = 0, None, None, None, 0, 'init'
        self.log, self.n_commit, self.late = [], 0, None
        self.j = 0                                                    # 目前段內步序
        self.got = None                                               # 已收到但尚未到提交步的結果（tag, out, 收到牆鐘）
        self.early, self.accepted = False, None                       # early：結果到達即提交（預設關）；accepted＝(結果, 提交步)
        self.last = None                                              # 最近一次已提交結果的摘要（mode、bind、done、stop…）
        self.events = []

    def start_rx(self):
        """ROS 執行器用：獨立接收執行緒，結果一到就記收到牆鐘（主迴圈只在物理步才輪詢會把收到時刻量化到步）。之後不可再用 call()/get()。"""
        import threading
        self._rx_lock, self._rx_stop = threading.Lock(), False

        def _loop():
            while not self._rx_stop:
                try:
                    if not self.conn.poll(0.02):
                        continue
                    tag, out = self.conn.recv()
                except (EOFError, OSError):
                    break
                t_ = time.perf_counter()
                with self._rx_lock:
                    if tag == 'built':
                        self.built = out
                    elif tag in ('seg', 'error'):
                        self._rx_box = (tag, out, t_)
        self._rx_box = None
        self._rx = threading.Thread(target=_loop, daemon=True)
        self._rx.start()

    def tail_props(self, tail, d_steps):
        """尾段的逐拍提案：第 m 拍（尾段起點後 m·NPH 步）的拍命令在其斜坡末步（j＝NPH−1）進管線、D 步後出現在尾段輸出 ⇒ u_m＝tail[m·NPH＋NPH−1＋D] 的底盤分量。"""
        out, m = [], 0
        while m * self.nph + self.nph - 1 + d_steps < len(tail):
            out.append(np.asarray(tail[m * self.nph + self.nph - 1 + d_steps][6:9], float).copy())
            m += 1
        return out

    def _send(self, q, t_next, n_send, first=False, phi=None, finger=None):
        self.ver += 1
        snap = dict(ver=self.ver, q=np.asarray(q, float).tolist(), t_next=float(t_next), first=first)
        if phi is not None:
            snap['phi'] = float(phi)
        if finger is not None:
            snap['finger'] = finger
        if getattr(self, 'extra', None):                              # D2：下一拍快照附帶的資料（如首段改綁），送出即清
            snap.update(self.extra); self.extra = None
        self.pending = dict(ver=self.ver, n_send=int(n_send), w_send=time.perf_counter())
        self.conn.send(('tick', snap))

    def prepare_takeover(self, packet, q_pred, t_h, n_h, phi=None, finger=None):
        """D2（Codex 20261117 交權契約）：非阻塞準備。工作程序以 packet 建控制器、用預測的接手狀態 q_pred 算接手拍 n_h 的第一段＋尾段。
        只送出、不等；結果由 poll()／接收執行緒收下。之後在 n_h 呼叫 commit_takeover()。"""
        self.conn.send(('build', packet))
        self.ver += 1
        snap = dict(ver=self.ver, q=np.asarray(q_pred, float).tolist(), t_next=float(t_h), first=True)
        if phi is not None:
            snap['phi'] = float(phi)
        if finger is not None:
            snap['finger'] = finger
        self.pending = dict(ver=self.ver, n_send=int(n_h), w_send=time.perf_counter(), takeover=True)
        self.mode, self.took = 'prep', None
        self.conn.send(('tick', snap))
        return self.ver

    def commit_takeover(self, n_h, extra_ok=True, why_not=None):
        """在接手步 n_h（套用本步輸出之前）呼叫：第一段結果已到、計算 ≤ 期限、附尾段、無中止，且呼叫端的前綴／版本／狀態驗收（extra_ok）成立 ⇒ 原子切換為名目；
        否則 mode＝'nack'（遲到結果之後丟棄），回 (ok, 紀錄)。"""
        rec = dict(n=int(n_h), kind='takeover')
        self.poll()
        got, self.got = self.got, None
        p = self.pending
        ok = False
        if got is not None and got[0] == 'seg' and p is not None and got[1]['ver'] == p['ver']:
            o = got[1]
            comp = (o['w_done'] - o['w_start']) * 1e3
            rec.update(compute_ms=round(comp, 2), wall_since_request_ms=round((got[2] - p['w_send']) * 1e3, 2), tail_src=o['tail_src'], abort=o['abort'])
            ok = bool(extra_ok and comp <= self.deadline_ms and o['seg'] is not None and o['abort'] is None and o['tail'] is not None)
            if ok:
                self.seg, self.tail, self.tail_i, self.mode, self.j = o['seg'], o['tail'], 0, 'nominal', 0
                self.prop0 = None if o.get('prop') is None else np.asarray(o['prop'], float).copy()
                self.n_commit += 1
        else:
            rec['why'] = 'not_ready' if got is None else ('error:' + str(got[1])[:300] if got[0] == 'error' else 'version_mismatch')
        if not ok:
            rec.setdefault('why', why_not or ('extra_check' if not extra_ok else 'deadline_or_tail'))
            self.mode, self.seg = 'nack', None
        self.pending = None
        rec['ok'] = ok
        self.log.append(rec)
        return ok, rec

    def start(self, q, t0, phi=None, finger=None, timeout=5.0):
        """初始化（執行器起跑前）：第一段以實測狀態直接算。"""
        self._send(q, t0, 0, first=True, phi=phi, finger=finger)
        if not self.conn.poll(timeout):
            raise RuntimeError('ahead worker init timeout')
        tag, out = self.conn.recv()
        if tag != 'seg':
            raise RuntimeError('ahead worker init error: ' + str(out)[:800])
        self.pending = None
        if out.get('abort') is not None or out.get('tail') is None or out.get('seg') is None:   # 起始放行條件（Codex 20261117_003351）：首段無中止、附已核尾段才進名目
            self.mode = 'start_rejected'
            self.log.append(dict(kind='init', ver=out['ver'], ok=False, abort=out.get('abort'), tail_src=out.get('tail_src')))
            return out
        self.seg, self.tail, self.tail_i, self.mode, self.j = out['seg'], out['tail'], 0, 'nominal', 0
        self.prop0 = None if out.get('prop') is None else np.asarray(out['prop'], float).copy()
        self.log.append(dict(kind='init', ver=out['ver'], tail_src=out['tail_src'], compute_ms=round((out['w_done'] - out['w_start']) * 1e3, 2)))
        return out

    def poll(self):
        """每個物理步呼叫：結果一到就收下並記收到時刻（不等拍邊界）。接收執行緒模式下從其盒子取。"""
        if getattr(self, '_rx', None) is not None:
            if self.pending is not None and self.got is None:
                with self._rx_lock:
                    if self._rx_box is not None:
                        self.got, self._rx_box = self._rx_box, None
            return
        while self.pending is not None and self.got is None and self.conn.poll(0):
            tag, out = self.conn.recv()
            if tag == 'built':                                        # D2 建控制器的回報（只記錄）
                self.built = out
                continue
            self.got = (tag, out, time.perf_counter())

    def tick_boundary(self, q, t_next, n, phi=None, finger=None):
        """在拍邊界 n_k（套用本步輸出之前）呼叫：決定 S_k 是否提交（第一次呼叫時 S_0 已在 start() 準備好），並送下一拍快照。"""
        rec = dict(n=int(n))
        if self.mode == 'nominal' and self.pending is not None:
            n_acc = None
            if self.accepted is not None:                             # 已在結果到達時提交（early 模式）
                got, n_acc = self.accepted
                self.accepted = None
            else:
                self.poll()
                got, self.got = self.got, None
            p = self.pending
            if got is not None and got[0] == 'seg' and got[1]['ver'] == p['ver']:
                o = got[1]
                rt_ms = (got[2] - p['w_send']) * 1e3
                rec.update(ver=o['ver'], comm_in_ms=round((o['w_start'] - p['w_send']) * 1e3, 3), compute_ms=round((o['w_done'] - o['w_start']) * 1e3, 2),
                           comm_out_ms=round((got[2] - o['w_done']) * 1e3, 3), roundtrip_ms=round(rt_ms, 2), steps_since_send=int(n - p['n_send']), tail_src=o['tail_src'], tail_mismatch=o['tail_mismatch'])
                rec.update(mode=o.get('mode'), events=o.get('events') or None, sigma=o.get('sigma'), stop_mode=o.get('stop_mode'), parts=o.get('parts'))
                self.events += [dict(e, n=int(n)) for e in (o.get('events') or [])]
                if rt_ms <= self.deadline_ms and o['seg'] is not None and o['abort'] is None and o['tail'] is not None:
                    self.seg, self.j = o['seg'], 0
                    self.tail, self.tail_i = o['tail'], 0             # 每次提交都必須附本段的已核停止尾段（沒有就不提交）
                    rec['prop'] = None if o.get('prop') is None else [float(x) for x in o['prop']]
                    self.n_commit += 1
                    rec['commit'] = True
                    rec['commit_step'] = int(n if n_acc is None else n_acc)
                    rec['bind'] = o.get('bind') is not None
                    self.last = dict(mode=o.get('mode'), bind=o.get('bind'), done=o.get('done'), stop=o.get('stop'), stop_mode=o.get('stop_mode'), n=int(n))
                else:
                    rec['commit'] = False
                    rec['why'] = 'deadline' if rt_ms > self.deadline_ms else ('abort' if o['abort'] is not None else ('no_tail:' + str(o.get('tail_src')) if o['tail'] is None else 'empty'))
                    rec['ctrl_done'] = bool(o.get('done'))
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

    def poll_commit(self, n_now):
        """early 模式（預設關）：結果一到、準時（來回 ≤ 期限）、附尾段、無中止 ⇒ 當下即提交（之後拍首必用它，只有執行器工程中止能取消），
        回本拍提案供立刻發布；否則回 None（拍首照原規則判定）。提交判定與拍首判定同一條件。"""
        if not self.early or self.mode != 'nominal' or self.pending is None or self.accepted is not None:
            return None
        self.poll()
        g = self.got
        if g is None or g[0] != 'seg' or g[1]['ver'] != self.pending['ver']:
            return None
        o, rt_ms = g[1], (g[2] - self.pending['w_send']) * 1e3
        if rt_ms <= self.deadline_ms and o['seg'] is not None and o['abort'] is None and o['tail'] is not None:
            self.accepted, self.got = (g, int(n_now)), None
            return None if o.get('prop') is None else np.asarray(o['prop'], float).copy()
        return None

    def output(self):
        """本步要套用的輸出：dict(arm_target, base_twist, finger, ready, ready_ver, stopped) 或 None（尾段已執行完＝停住）。"""
        if self.mode in ('start_rejected', 'prep', 'nack'):
            return None
        if self.mode == 'nominal':
            o = self.seg[self.j]
            self.j += 1
            return o
        if self.mode == 'tail' and self.tail is not None and self.tail_i < len(self.tail):
            r = self.tail[self.tail_i]
            self.tail_i += 1
            return dict(arm_target=r[:6].copy(), base_twist=r[6:9].copy(), finger=None, ready=False, ready_ver=None, stopped=True, abort=False)
        return None

    def call(self, name, *args, timeout=2.0):
        self.conn.send(('call', name, args))
        if not self.conn.poll(timeout):
            return ('timeout', None)
        return self.conn.recv()

    def get(self, names, timeout=2.0):
        self.conn.send(('get', list(names)))
        t_end = time.perf_counter() + timeout
        while self.conn.poll(max(t_end - time.perf_counter(), 0.0)):
            tag, out = self.conn.recv()
            if tag == 'get':                                          # 先到的遲到段結果（tick）丟棄
                return out
        return None

    def close(self):
        if getattr(self, '_rx', None) is not None:
            self._rx_stop = True
            self._rx.join(0.2)
        try:
            self.conn.send(None)
        except Exception:                                             # noqa: BLE001
            pass
        self.proc.join(timeout=1.0)
        if self.proc.is_alive():
            self.proc.terminate()
