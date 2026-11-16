#!/usr/bin/env python3
"""非阻塞執行的開門段（Howard 11/17 主線第 2 段 C；Codex 20261117 執行契約）：在 door_ahead_exec 的工作程序裡，
抓持控制器 READY 後交接給開門執行層（DoorOpenExec），之後 P2 續段在背景規劃程序算、暫存、在最早影響命令的那一拍接上。

與阻塞版（test_door_motm_chain_fake.py／Isaac 執行器）的差別只在「計算放在哪、算幾次」，判定不變：
  - 交接：阻塞版先用複本試跑第一拍、通過才接受，再讓本體重算同一拍（兩次）。這裡複本試跑那一拍就是正式那一拍：
    通過 ⇒ 接受交接並直接採用該複本（狀態與「本體重算」相同）；不通過 ⇒ 丟棄、抓持控制器照常算本拍。
  - P2 接縫：同上，最早影響拍的複本試跑通過即採用複本（staged_step 的判定原樣照抄），不通過 ⇒ 本體不接新參考、照常算本拍。
  - P2 規劃：規劃程序（PlannerServer）與控制並行；結果到達的那一拍（模擬時間）才處理，不用「計算時間記到時間線」的替代。
  - 尾段：候選提交前核對的閉鏈一致停法模擬（hold＝1，DoorOpenExec.rec_cand）逐步輸出，前 NPH 步須等於名目段；
    控制器已在備援停法（cstop）中 ⇒ 名目段末另做一次停法模擬（hold＝0）並送計算池核淨空；其他停止模式（path 以外的 settled）沒有尾段 ⇒ 執行器不提交。
所有時間（交接建構、試跑、規劃結果處理）都在該拍的計算時間內，由執行器的來回期限檢查。"""
import math
import time

import numpy as np


class DoorOpenPhase:
    def __init__(self, X, P2, K, first, helper, planner, ident, goal, fq, phi_b, H, L_LAT, g2, deadline_ms=50.0, stream=True, tail_cert=False):
        self.stream = bool(stream)
        self.tail_cert = bool(tail_cert)
        self.X, self.P2, self.K = X, P2, K
        self.first, self.helper, self.planner = first, helper, planner
        self.ident, self.goal, self.fq, self.phi_b, self.H, self.L_LAT = ident, float(goal), float(fq), float(phi_b), float(H), float(L_LAT)
        self.g2 = dict(g2)
        self.deadline_ms = float(deadline_ms)
        self.pending = None                                          # 已送規劃程序、尚未回的請求
        self.t_next = None                                           # 下一次請求時刻
        self.k = 1
        self.req_log, self.decisions = [], []
        self.warm_ms = None

    # ---- 起跑前預熱（對應遠距接近期間的背景工作；不在任何控制拍內）
    def prewarm(self, gc, q_pred):
        """以預測 READY 構型建一個丟棄用開門執行層並空跑一拍：匯入、求解器第一次呼叫、停法工作程序的模組載入都在這裡發生。"""
        w0 = time.perf_counter()
        ex = self._build(gc, q_pred)
        ex.warmup(q_pred, self.phi_b)
        tmp = ex._light_copy()
        tmp.step(0.0, np.asarray(q_pred, float).copy(), self.phi_b)  # 經停法工作程序（init＋模擬）與計算池走一遍
        self.warm_ms = round((time.perf_counter() - w0) * 1e3, 1)
        return dict(prewarm_ms=self.warm_ms, stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'))

    def _build(self, gc, q):
        X = self.X
        Xd0, h0, _ = self.first['o'].dense_reference(4)
        T_HE_act = np.linalg.inv(self.first['T_plan']) @ self.K.fk(q, 'link_tcp')
        ho = dict(s=gc.s.copy(), pipe=[u_.copy() for u_ in gc.pipe], u_last=gc.u_last.copy(), u_prev=gc.u_prev.copy(), bias=np.asarray(gc.cfg.arm_model.bias, float).copy(), moving=True,
                  off_static=gc.off(q) if getattr(gc, 'G0', None) is not None else np.asarray(q, float)[3:] - gc.s)   # 起跑前預熱時抓持控制器尚未錨定（G0 未定）⇒ 用 q−s（只供丟棄用實例）
        ex = X.DoorOpenExec(self.ident, Xd0, h0, self.first['T_plan'], self.phi_b, T_HE_act, self.goal, ho, self.fq, arm_mode='closure', g2=self.g2)
        ex.p2_open = True
        ex.rec_cand = True
        ex.stream_cand = self.stream                                  # A1：候選停法模擬分批送淨空池（判定不變）
        ex.tail_cert_mode = self.tail_cert                            # A2：尾段證書模式（只在非阻塞執行器）
        return ex

    @staticmethod
    def _adopt(base, tmp):
        """複本試跑通過 ⇒ 採用複本：接回紀錄、期限、最小值統計（複本建立時清空的大型紀錄與期限設定）。"""
        big = ('step_log', 'pipe_log', 'cmd_hist', 'events', 'rec', 'tick_ms', 'solve_ms', 'sweep_ms', 'pool_ms', 'prep_ms', 'p2_log', 'fb_log', 'mon_ms', 'cl_ms', 'cand_ms', 'sn_ms', 'qp_log', 'pr_ts', 'pr_log', 'hold_log')
        for k in big:
            if hasattr(base, k) and hasattr(tmp, k):
                setattr(tmp, k, list(getattr(base, k)) + list(getattr(tmp, k)))
        tmp.g2 = dict(tmp.g2, deadline_ms=base.g2['deadline_ms'])
        tmp.mins['stop_closure_pos'] = max(base.mins['stop_closure_pos'], tmp.mins['stop_closure_pos'])
        tmp.mins['stop_closure_rot'] = max(base.mins['stop_closure_rot'], tmp.mins['stop_closure_rot'])
        return tmp

    def _valid(self, tmp, okx=True):
        X = self.X
        fb = tmp._fb_ok if not getattr(tmp, 'tail_cert_mode', False) else (tmp._cand_out is not None)   # A2：尾段證書模式＝本拍候選的完整停止尾段已核（不以 _fb_ok 代替）
        return bool(okx and tmp.stop is None and tmp.abort is None and tmp._cl_fail is None and tmp.mins['stop_closure_pos'] <= X.CL_TOL['pos_m'] and tmp.mins['stop_closure_rot'] <= X.CL_TOL['rot_deg']
                    and (tmp._pool is None or fb))

    @staticmethod
    def _ho_key(gc):
        return [np.asarray(gc.s, float).copy(), [np.asarray(u_, float).copy() for u_ in gc.pipe], np.asarray(gc.u_last, float).copy(), np.asarray(gc.u_prev, float).copy(), np.asarray(gc.cfg.arm_model.bias, float).copy()]

    def prebuild(self, gc, q_guess):
        """抓持控制器剛送出 READY 段後的空檔（下一個快照到達前）先建開門執行層：交接狀態（設定點、在途管線、上一命令、偏置）此後不再變（抓持控制器不會再被呼叫），
        只有依賴交接拍構型的兩項（靜態偏置 off_static、抓持變換 T_HE）在交接拍用該拍預測構型重設。"""
        w0 = time.perf_counter()
        self._pre = (self._build(gc, q_guess), self._ho_key(gc))
        return round((time.perf_counter() - w0) * 1e3, 1)

    def _take_prebuilt(self, gc, q):
        pre, self._pre = getattr(self, '_pre', None), None
        if pre is None:
            return None, 'none'
        ex_c, key = pre
        now = self._ho_key(gc)
        same = all(np.array_equal(a, b) for a, b in zip([key[0], key[2], key[3], key[4]], [now[0], now[2], now[3], now[4]])) and len(key[1]) == len(now[1]) and all(np.array_equal(a, b) for a, b in zip(key[1], now[1]))
        if not same:
            return None, 'state_changed'
        ex_c.off_static = gc.off(q).copy()                            # 交接拍預測構型（同 _build 的定義）
        ex_c.T_HE = np.linalg.inv(self.first['T_plan']) @ self.K.fk(q, 'link_tcp')
        ex_c.T_EH = np.linalg.inv(ex_c.T_HE)
        return ex_c, 'prebuilt'

    # ---- 交接（抓持控制器上一拍 READY、本拍首）
    def handover(self, gc, q, t):
        """回 (ex 或 None, 第一步輸出或 None, 紀錄)。ex 已算完本拍第一步（試跑即正式）。"""
        w0 = time.perf_counter()
        ex_c, src = self._take_prebuilt(gc, q)
        if ex_c is None:
            ex_c = self._build(gc, q)
        w1 = time.perf_counter()
        ex_c._cand_out = None
        tmp = ex_c._light_copy()
        tmp.mins = dict(tmp.mins, stop_closure_pos=0.0, stop_closure_rot=0.0)
        import door_cstop as _DCS
        _ci = {}
        _CS0, _init0 = _DCS.CStop.__init__, type(tmp._csw).init

        def _cs_init(obj, P, _f=_CS0):
            w_ = time.perf_counter(); _f(obj, P); _ci['cstop_ctor_ms'] = round((time.perf_counter() - w_) * 1e3, 2)

        def _w_init(obj, P, _f=_init0):
            w_ = time.perf_counter(); _f(obj, P); _ci['csw_init_ms'] = round((time.perf_counter() - w_) * 1e3, 2)
        _DCS.CStop.__init__, type(tmp._csw).init = _cs_init, _w_init      # 只計時（診斷用；行為不變）
        try:
            o0 = tmp.step(float(t), np.asarray(q, float).copy(), self.phi_b)
        finally:
            _DCS.CStop.__init__, type(tmp._csw).init = _CS0, _init0
        valid = self._valid(tmp)
        _ci.update({k: round(float(getattr(tmp, k)[-1]), 2) for k in ('tick_ms', 'solve_ms', 'sweep_ms', 'cl_ms', 'cand_ms', 'sn_ms', 'mon_ms', 'prep_ms') if getattr(tmp, k, None)})
        rec = dict(t=round(float(t), 3), what='handover_trial_is_tick', ok=valid, ready_ver=gc.ready_ver, build_src=src, build_ms=round((w1 - w0) * 1e3, 1), tick_ms=round((time.perf_counter() - w1) * 1e3, 1),
                   stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail, fb_ok=bool(tmp._fb_ok), tail_cert=tmp._cand_out is not None,
                   stop_closure_pos_mm=round(tmp.mins['stop_closure_pos'] * 1e3, 3), start_base_mm=round(float(np.linalg.norm(ex_c.Xd[0, :2] - np.asarray(q)[:2])) * 1e3, 3), parts_ms=_ci)
        if not valid:
            return None, None, rec
        if not gc.accept_handover(gc.ready_ver):
            rec.update(ok=False, why='accept_refused')
            return None, None, rec
        ex = self._adopt(ex_c, tmp)
        self.t_next = float(t) + self.P2.T_RE
        rec['base_speed_mps'] = round(float(np.linalg.norm(gc.u_last[:2])), 4)
        return ex, o0, rec

    # ---- 開門段每拍首（step 之前）：暫存參考的採用試跑、規劃結果處理、新請求
    def pre_tick(self, ex, t, q, phi):
        """回 (ex, 第一步輸出或 None, 紀錄 list)。第一步輸出非 None ⇒ 本拍第一步已由採用的複本算完。"""
        X, P2 = self.X, self.P2
        out, o0 = [], None
        st = ex.staged
        if st is not None:
            if ex.stop is not None or ex.abort is not None:
                ex.staged = None
                out.append(dict(t=round(t, 3), what='staged', ver=st['ver'], ok=False, why='exec_stopped'))
            elif ex.sigma > st['sigma_s'] + 1e-9:
                ex.staged = None
                out.append(dict(t=round(t, 3), what='staged', ver=st['ver'], ok=False, why='expired_before_influence', sigma=round(ex.sigma, 3), sigma_s=round(st['sigma_s'], 3)))
            elif not (ex.sigma + X.D_STEPS * X.PHYS + X.N * X.DT < st['sigma_s'] - 1e-9 and ex.sigma < ex.sig_end - 1e-9):
                w0 = time.perf_counter()
                ex._cand_out = None
                tmp = ex._light_copy()
                tmp.mins = dict(tmp.mins, stop_closure_pos=0.0, stop_closure_rot=0.0)
                okx = tmp.extend_reference(st['Xd'], st['sigma_s'], final=st['final'])
                fq_save, phi0_save = X.W.FINGER_Q, X.W.PHI0
                o_t = None
                try:
                    if okx:
                        o_t = tmp.step(float(t), np.asarray(q, float).copy(), float(phi))
                finally:
                    X.W.FINGER_Q, X.W.PHI0 = fq_save, phi0_save
                valid = self._valid(tmp, okx)
                ex.staged = None
                rec = dict(t=round(t, 3), what='staged', ver=st['ver'], ok=valid, why=None if valid else 'dry_run_rejected', trial_is_tick=valid, wall_ms=round((time.perf_counter() - w0) * 1e3, 1),
                           extend_ok=okx, stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail, fb_ok=bool(tmp._fb_ok), tail_cert=tmp._cand_out is not None,
                           sigma=round(ex.sigma, 3), sigma_s=round(st['sigma_s'], 3))
                if valid:
                    ex._last_ver = st['ver']
                    ex = self._adopt(ex, tmp)
                    o0 = o_t
                else:
                    ex.p2_log.append(dict(rec))
                out.append(rec)
                self.decisions.append(rec)
        # 規劃結果（背景規劃程序；到達後的第一個控制拍處理）
        if self.pending is not None and self.planner.busy is not None:
            r = self.planner.poll()
            if r is not None:
                pend, self.pending = self.pending, None
                rid, Xd, feas, failed, fin, wall_s, info = r
                rec = pend['rec']
                rec.update(t_avail=round(t, 3), wall_s=round(wall_s, 3), feasible=bool(feas), final=bool(fin), failed=failed, info=info)
                xr = ex.ref(ex.sigma)
                rec['track'] = dict(yaw_deg=round(abs(math.degrees(q[2] - xr[2])), 3), door_deg=round(abs(math.degrees(phi - xr[9])), 3), base_mm=round(float(np.linalg.norm(q[:2] - xr[:2])) * 1e3, 2), g=round(ex.g, 3))
                if ex.stop is not None:
                    rec['result'] = 'exec_stopped'
                elif Xd is None or not feas:
                    rec['result'] = 'infeasible'
                elif ex.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
                    rec['result'] = 'missed_deadline'
                elif ex.staged is not None:
                    rec['result'] = 'staged_busy'
                else:
                    okS, inf = ex.stage_reference(Xd, pend['sigma_s'], final=bool(fin), ver=rec['k'])
                    rec['result'] = 'staged' if okS else 'stage_rejected'
                    if not okS:
                        rec['why'] = inf.get('why')
                out.append(dict(what='p2_result', why=rec.get('why'), seam=(inf.get('seam') or {}).get('failed') if rec.get('result') == 'stage_rejected' else None, **{k: rec[k] for k in ('k', 'result', 'wall_s', 'feasible', 'final')}))
        # 新請求
        if self.pending is None and self.planner.busy is None and ex.stop is None and ex.p2_open and self.t_next is not None and t >= self.t_next - 1e-9:
            sig_s = math.ceil((ex.sigma + self.L_LAT) / ex.h - 1e-9) * ex.h
            rest = sig_s >= ex.sig_end - 1e-9
            still_ok = ex.rate['arm'] <= X.END['arm_rps'] and ex.rate['base'] <= X.END['base_mps'] and ex.rate['door'] <= X.END['door_rps']
            if rest and not still_ok:
                self.t_next = t + 0.05
            elif not rest and sig_s <= getattr(self, 'last_sig_s', -1.0) + 1e-9:
                self.t_next = t + 0.05                                # 同一接縫已請求過（例如提前請求）⇒ 不重複請求
            else:
                if rest:
                    sig_s = ex.sig_end
                i = min(int(round(sig_s / ex.h)), len(ex.Xd) - 1)
                xs = ex.Xd[i].copy()
                if rest or i >= len(ex.Xd) - 1:
                    vs = np.zeros(10)
                else:
                    vv = (ex.Xd[min(i + 1, len(ex.Xd) - 1)] - ex.Xd[max(i - 1, 0)]) / (2 * ex.h)
                    X.W.set_model(phi0=self.phi_b, finger_q=self.fq)
                    vs = np.r_[vv[:3], self.helper.closure_velocity(xs, vv[:3], vv[9]), vv[9]]
                self.k += 1
                rec = dict(k=self.k, t_req=round(t, 3), sigma=round(ex.sigma, 3), sigma_s=round(sig_s, 3), rest=bool(rest))
                self.req_log.append(rec)
                sn = dict(T_WH0=self.first['T_plan'], T_HE=self.first['T_HE_pred'], x0=xs, v0=vs, H=self.H, goal=self.goal, fq=self.fq, phi_model=self.phi_b, try_final=True,
                          warm_ref=None if rest else (ex.Xd.copy(), ex.h, sig_s))
                self.planner.submit(self.k, sn)
                self.pending = dict(sigma_s=sig_s, rest=bool(rest), rec=rec)
                self.last_sig_s = sig_s
                self.t_next = t + P2.T_RE
                out.append(dict(what='p2_request', k=self.k, sigma_s=round(sig_s, 3), rest=bool(rest)))
        return ex, o0, out

    # ---- 第一個續段提前請求（抓持段起點即送；輸入只取決於首段參考）
    def prerequest(self):
        """k＝2 的接縫與起點照交接後第一次請求的排程算（σ_req＝σ₀＋T_RE，σ_s＝⌈(σ_req＋L_LAT)/h⌉·h），起點狀態／速度取首段參考——這些在交接前就已確定，
        所以資料到齊即開算（規劃耗時照計：結果到達後的第一個開門拍才處理，接縫過期照拒）。D1（RTF 1）實測 k＝2 規劃牆鐘 1.2 s，交接後才請求會晚於接縫。"""
        X, P2 = self.X, self.P2
        Xd0, h0, _ = self.first['o'].dense_reference(4)
        sig0 = -X.D_STEPS * X.PHYS if X.START_LAT_ALIGN else 0.0
        sig_req = sig0 + P2.T_RE
        sig_s = math.ceil((sig_req + self.L_LAT) / h0 - 1e-9) * h0
        i = min(int(round(sig_s / h0)), len(Xd0) - 1)
        xs = Xd0[i].copy()
        vv = (Xd0[min(i + 1, len(Xd0) - 1)] - Xd0[max(i - 1, 0)]) / (2 * h0)
        X.W.set_model(phi0=self.phi_b, finger_q=self.fq)
        vs = np.r_[vv[:3], self.helper.closure_velocity(xs, vv[:3], vv[9]), vv[9]]
        self.k += 1
        rec = dict(k=self.k, t_req='grasp_start', sigma=round(sig_req, 3), sigma_s=round(sig_s, 3), rest=False, pre=True)
        self.req_log.append(rec)
        self.planner.submit(self.k, dict(T_WH0=self.first['T_plan'], T_HE=self.first['T_HE_pred'], x0=xs, v0=vs, H=self.H, goal=self.goal, fq=self.fq, phi_model=self.phi_b, try_final=True, warm_ref=(Xd0.copy(), h0, sig_s)))
        self.pending = dict(sigma_s=sig_s, rest=False, rec=rec)
        self.last_sig_s = sig_s
        return rec

    # ---- 尾段
    def tail(self, ex, seg, q_end, nph):
        """回 (尾段 list[(設定點, 底盤命令)] 或 None, 來源, 前 nph 步不一致量)。"""
        if ex.abort is not None:
            return None, None, None
        co = ex._cand_out
        if ex.stop is None or ex.stop_mode == 'path':
            if co is not None and len(co) >= nph:
                mism = max(max(float(np.abs(co[j][0] - seg[j]['arm_target']).max()), float(np.abs(co[j][1] - seg[j]['base_twist']).max())) for j in range(nph))
                if mism <= 1e-9:
                    return [(a.copy(), b.copy()) for a, b in co[nph:]], 'cstop_candidate_hold1', mism
                return None, 'cstop_candidate_mismatch', mism
            return None, 'no_candidate_out', None
        if ex.stop_mode == 'cstop' and ex._cs_on():                   # 已在備援停法：名目段末另做一次停法模擬（hold＝0）＋淨空核對
            ex._cs_obj()
            ver = ex._csw.submit(ex._cs_state(np.asarray(q_end, float), rec=True))
            r = ex._csw.collect(ver, 0.2)
            if r is None or not r['ok'] or r.get('out') is None:
                return None, 'explicit_cstop_failed', None
            res, why, _ = ex._pool.collect(ex._pool.submit(ex.T_WH_bind, self.X.W.FINGER_Q, self.X.W.PHI0, r['X']), self.X.POOL_TIMEOUT_S)
            if res is None:
                return None, 'explicit_cstop_pool_' + str(why), None
            mn = {k: min(float(c[k]) for c in res) for k in self.X.KEYS}
            if any(mn[k] < self.X.W.THR[k] for k in self.X.KEYS):
                return None, 'explicit_cstop_clear_bad', None
            return [(a.copy(), b.copy()) for a, b in r['out']], 'explicit_cstop_e0', None
        return None, 'no_tail_mode_' + str(ex.stop_mode), None
