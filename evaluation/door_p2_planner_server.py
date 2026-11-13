#!/usr/bin/env python3
"""P2 背景規劃程序（Codex 20261113_141451 §3：背景規劃只用隔離的模型快照）。執行器在啟動模擬器前 fork（之後模擬器程序不再 fork），
規劃在獨立程序裡算：每個請求自帶完整快照（把手位姿 T_WH0、門角基準、手指開度、T_HE、起點與起始速度、H、目標、是否試末段），
程序內自設模組值，不與控制／掃掠共用任何全域設定。回 (req_id, 密集參考或 None, 可行, 失敗項, 是否末段, 規劃 wall 秒)。
    srv = PlannerServer(); srv.submit(req_id, snap); srv.poll() → 結果或 None"""
import multiprocessing as mp
import time

import numpy as np


def _die_with_parent():
    """父程序死亡（含段錯誤）時本程序收 SIGTERM（Linux prctl PR_SET_PDEATHSIG），避免孤兒程序佔住 Isaac 互斥檢查。"""
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass


def _serve(conn):
    _die_with_parent()
    import os
    os.environ.setdefault('P2_TERM', '1')
    os.environ.setdefault('P2_FINAL', '1')
    import door_motm_p2 as P2
    WB = P2.WB
    conn.send(('ready',))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        rid, sn = msg
        t0 = time.perf_counter()
        try:
            WB.set_model(phi0=sn['phi_model'], finger_q=sn['fq'])
            out, fin = None, False
            if sn.get('try_final', True):
                try:
                    ft = P2.plan_final(sn['T_WH0'], sn['T_HE'], sn['x0'], sn['v0'], sn['goal'], sn['fq'], sn['phi_model'])
                except ValueError:
                    ft = None
                if ft is not None and ft[2]['feasible']:
                    out, fin = ft, True
            if out is None:
                out = P2.plan_seg(sn['T_WH0'], sn['T_HE'], sn['x0'], sn['v0'], sn['H'], sn['goal'], sn['fq'], sn['phi_model'])
            Xd, h, _ = out[0].dense_reference(4)
            conn.send((rid, Xd, bool(out[2]['feasible']), out[2]['failed'], fin, time.perf_counter() - t0, dict(why=out[1]['why'], iters=out[1]['iters'], phi_end_deg=out[2]['phi_end_deg'], min_clear=out[2]['min_clear_m'])))
        except Exception as e:                                        # noqa: BLE001
            conn.send((rid, None, False, ['exception:' + repr(e)[:200]], False, time.perf_counter() - t0, None))


class PlannerServer:
    def __init__(self):
        ctx = mp.get_context('fork')
        self.conn, b = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(b,), daemon=True)
        self.proc.start()
        b.close()
        if not self.conn.poll(120.0) or self.conn.recv() != ('ready',):
            raise RuntimeError('planner server start failed')
        self.busy = None

    def submit(self, rid, snap):
        self.conn.send((rid, snap))
        self.busy = rid

    def poll(self):
        if self.busy is None or not self.conn.poll(0):
            return None
        r = self.conn.recv()
        self.busy = None
        return r

    def close(self):
        try:
            self.conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.terminate()
