#!/usr/bin/env python3
"""閉鏈一致備援停法（Codex 20261115_005224 §1）：綁定中控制期限晚到時用；不依賴全身 QP。
停法：沿用當下的開門參考，進度率 g 以每拍 DT/T_STOP 降到 0（σ 隨之停住 ⇒ 門角依參考減速到停；只追隨實測門角不代表門會停，所以門角目標取參考）；
  底盤命令＝參考前饋×g（本體座標），每拍變化量受 A_BASE·DT 限制；手臂＝以「管線清空後＋本拍底盤命令（拍內斜坡）」的預測底盤與拍末參考門角解閉鏈 IK，
  設定點 s＝q＋τ·q̇−off（off＝起點實測偏置＋重力變化／kp，同開門執行層 closure 手臂），命令＝(s−s_pred)/DT。
  命令限制（手臂速率、相鄰命令加速度、輪速）不過或 IK 失敗 ⇒ 回失敗（呼叫端模擬中止；不退回各軸減速）。
執行（step_cmd，每拍一次）與預測（sim，逐拍向前到停穩）共用同一函式、同一受控對象模型（真管線、拍內斜坡、偏置）。
CStopWorker：啟動模擬器前 fork 的獨立程序，收參數與狀態快照做 sim（與主程序求解並行）。"""
import math
import multiprocessing as mp
import time

import numpy as np
import door_cstop_def as DEF

T_STOP = 0.5
K_HOLD = 2.0                                                         # 1/s：底盤保持比例增益（管線延遲 0.08 s ⇒ 增益×延遲＝0.16）
HOLD_V = np.array([0.02, 0.02, 0.05])                                # 保持命令上限（m/s, m/s, rad/s）
HOLD_DEAD = 1e-5                                                     # 保持命令死區（數值零）
ARM_CMD_SETTLE = 0.005


class CStop:
    def __init__(self, P):
        import door_open_exec_ctrl as XO
        import door_fast_kin as DFK
        self.XO, self.DFK = XO, DFK
        self.K, self.S, self.C0, self.B, self.W = XO.K, XO.S, XO.C0, XO.B, XO.W
        self.cfg = XO.C0.make_cfg(P['ident'])
        self.cfg.base_fixed = False
        self.cfg.arm_model.bias = np.asarray(P['bias'], float).copy()
        al = np.asarray(self.cfg.arm_model.alpha, float)
        self.tau = XO.PHYS / al
        self.va = float(XO.B.W_ARM['v_arm'])
        self.T_WH_bind, self.phi_bind, self.T_HE = np.asarray(P['T_WH_bind'], float), float(P['phi_bind']), np.asarray(P['T_HE'], float)
        self.off0, self.G0, self.kp = np.asarray(P['off_cl0'], float), np.asarray(P['G0'], float), float(P['joint_kp'])
        self.grav = (DFK.FastGravity3 if P.get('gravity3') else DFK.FastGravity2)(self.K, XO.BM.arm_bodies(self.K))   # 與執行層同一個重力函式
        self.h = float(P['h'])
        import door_fric_ff as FF_
        self.FF, self.fric_ff = FF_, P.get('fric_ff')
        self.lim_lo, self.lim_hi = np.asarray(P['lim_lo'], float), np.asarray(P['lim_hi'], float)     # r2：設定點限位（同執行層）
        self.q_lo, self.q_hi = np.asarray(P['q_lo'], float), np.asarray(P['q_hi'], float)             # 關節限位（同閉鏈手臂驗收）
        self.set_ref(P['Xd'])

    def set_ref(self, Xd):
        self.Xd = np.asarray(Xd, float)
        self.sig_end = (len(self.Xd) - 1) * self.h

    def ref(self, sig):
        x = min(max(sig, 0.0), self.sig_end) / self.h
        i = min(int(x), len(self.Xd) - 2)
        a = x - i
        return (1 - a) * self.Xd[i] + a * self.Xd[i + 1]

    def dref(self, sig):
        if sig <= 0.0 or sig >= self.sig_end:
            return np.zeros(10)
        i = min(int(sig / self.h), len(self.Xd) - 2)
        return (self.Xd[i + 1] - self.Xd[i]) / self.h

    def _ik(self, b, phi, seed):
        Tg = self.W.Rz4(-(float(phi) - self.phi_bind)) @ self.T_WH_bind @ self.T_HE
        return self.DFK.ik_closure_fc(Tg, np.asarray(b, float), seed)

    def step_cmd(self, q, s, pipe, u_prev, sigma, g, seed, se_end, hold=None):
        """一拍的備援命令。回 dict(ok, u, sigma, g, q1, why)。q＝實測（或模擬）構型；pipe＝在途命令；u_prev＝上一拍提交命令。"""
        XO = self.XO
        DT, PHYS, NPH, D = XO.DT, XO.PHYS, XO.NPH, XO.D_STEPS
        g2 = max(g - DT / T_STOP, 0.0)
        # r2：參考進度對命令連續——本拍命令自「上一個真正已提交命令的參考終點」se_end 起、前進 DT·g2。
        #     se_end 由呼叫端保存（只在命令提交時更新；不從當下的 g 回推——主控制器每拍先重算 g，回推會回拉或前跳）。
        #     （最初的式子 se＝σ＋D·PHYS·g2 每拍把起點回拉 D·PHYS·Δg，交接處參考門速快時預測閉鏈偏差到 8–11 mm。）
        se = float(se_end)
        se1 = min(se + DT * g2, self.sig_end)
        ge = g2
        x, dx = self.ref(se), self.dref(0.5 * (se + se1))
        c, s_ = math.cos(x[2]), math.sin(x[2])
        ub = np.array([ge * (c * dx[0] + s_ * dx[1]), ge * (-s_ * dx[0] + c * dx[1]), ge * dx[2]])
        dab = np.asarray(self.B.A_BASE, float) * DT
        qp, sp = np.asarray(q, float).copy(), np.asarray(s, float).copy()
        for u_in in pipe:
            qp, sp = self.S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        # r3（B1，Codex 20261115）：底盤保持——減速尾端（g2＝0、參考前饋為零）且在途底盤命令已清空時，以當下實測位姿一次錨定（之後不重錨）；
        #   保持命令＝K_HOLD·(錨點−管線清空後的預測位姿)（本體座標），速度上限 HOLD_V、加速度同 A_BASE、輪速上限同原式；
        #   它是底盤命令本身 ⇒ 同樣進手臂閉鏈預測、管線與停止預測。不瞬移、不硬鎖。
        if hold is None and g2 <= 0.0 and not np.any(ub) and not any(np.any(np.asarray(u_in)[:3]) for u_in in pipe) and not np.any(np.asarray(u_prev)[:3]):
            hold = np.asarray(q[:3], float).copy()
        if hold is not None:
            e = np.asarray(hold, float) - qp[:3]
            e[2] = (e[2] + math.pi) % (2 * math.pi) - math.pi
            ch, sh = math.cos(qp[2]), math.sin(qp[2])
            ub = K_HOLD * np.array([ch * e[0] + sh * e[1], -sh * e[0] + ch * e[1], e[2]])
            ub = np.clip(ub, -HOLD_V, HOLD_V)
            ub[np.abs(ub) < HOLD_DEAD] = 0.0
        ub = np.clip(ub, u_prev[:3] - dab, u_prev[:3] + dab)
        b = qp[:3].copy()
        uf = np.asarray(pipe[-1][:3], float)
        for j in range(NPH):
            v = uf + min(j + 1, NPH) / NPH * (ub - uf)
            cb, sb = math.cos(b[2]), math.sin(b[2])
            b = b + PHYS * np.array([cb * v[0] - sb * v[1], sb * v[0] + cb * v[1], v[2]])
        ph1, ph0 = float(self.ref(se1)[9]), float(x[9])
        q1, r1, sm1 = self._ik(b, ph1, seed)
        q0, r0, _ = self._ik(qp[:3], ph0, q1)
        if max(r1, r0) > 1e-6 or float(np.abs(q1 - seed).max()) > 0.5 or not np.isfinite(q1).all():
            return dict(ok=False, why='ik_fail')
        off = self.off0 + (self.grav(np.r_[b, q1]) - self.G0) / self.kp
        S1 = q1 + self.tau * (q1 - q0) / DT - off + self.FF.ff((q1 - q0) / DT, self.fric_ff, self.kp)   # 同執行層：庫倫摩擦前饋（參考速度歸零時自動收回）
        ua = (S1 - sp) / DT
        u = np.r_[ub, ua]
        if float(np.abs(ua).max()) > self.va + 1e-9:
            return dict(ok=False, why='arm_rate', v=float(np.abs(ua).max()))
        if float(np.abs(ua - u_prev[3:]).max()) / DT > self.C0.A_ARM + 1e-9:
            return dict(ok=False, why='arm_acc', v=float(np.abs(ua - u_prev[3:]).max()) / DT)
        if float(np.abs(self.W.WM @ ub).max()) > self.W.W_LIM + 1e-9:
            return dict(ok=False, why='wheel_v')
        if np.any(S1 < self.lim_lo) or np.any(S1 > self.lim_hi):       # r2：同閉鏈手臂命令的驗收（設定點限位、關節限位、可操作度）
            return dict(ok=False, why='setpoint_limit')
        if np.any(q1 < self.q_lo) or np.any(q1 > self.q_hi):
            return dict(ok=False, why='ik_joint_limit')
        if sm1 < self.W.SIG_FLOOR:
            return dict(ok=False, why='manipulability', v=float(sm1))
        if g2 <= 0.0 and ge <= 0.0 and not np.any(ub) and float(np.abs(ua).max()) <= ARM_CMD_SETTLE:
            u = np.zeros(9)                                        # 停穩：命令歸零（設定點保持在閉鏈一致位置）
            return dict(ok=True, u=u, sigma=se1 - D * PHYS * g2, g=g2, q1=q1, zeroed=True, se_end=se1, hold=hold)
        return dict(ok=True, u=u, sigma=se1 - D * PHYS * g2, g=g2, q1=q1, zeroed=False, se_end=se1, hold=hold)

    def sim(self, st, max_ticks=80):
        """自狀態 st（q, s, pipe, u_prev, sigma, g, seed；可含 u_first, hold）起向前模擬到停穩。
        hold>0：先提交候選命令 u_first 共 hold 拍（σ 照常前進），之後才以本停法減速（候選提交前核對用）。
        停穩（r2）＝命令與管線為零，且預測手臂／底盤／偏航／門（參考門角）速率低於 STOP_STILL、連續 hold_s——與執行層停止驗收同門檻。
        回 dict(ok, X(取樣構型＋參考門角), why, n_ticks)；self._cl＝取樣點最大預測閉鏈偏差。"""
        XO = self.XO
        NPH, DT = XO.NPH, XO.DT
        SS = XO.STOP_STILL
        need = int(math.ceil(SS['hold_s'] / DT - 1e-9))
        q, s = np.asarray(st['q'], float).copy(), np.asarray(st['s'], float).copy()
        pipe = [np.asarray(u, float).copy() for u in st['pipe']]
        u_prev, sigma, g, seed = np.asarray(st['u_prev'], float).copy(), float(st['sigma']), float(st['g']), np.asarray(st['seed'], float).copy()
        hold = int(st.get('hold', 0))
        se_end = float(st['se_end'])
        bh = None if st.get('base_hold') is None else np.asarray(st['base_hold'], float).copy()
        zeroed = False
        bh_steps = 0
        X, quiet = [np.r_[q, self.ref(sigma)[9]]], 0
        self._cl = [0.0, 0.0]
        self._tr = []                                                # 逐拍（拍、g、預測閉鏈位置偏差 mm、手臂命令最大值）：診斷用
        for n in range(max_ticks):
            if n < hold:
                u = np.asarray(st['u_first'], float).copy()
                sigma = min(sigma + DT * g, self.sig_end)
                se_end = float(st['se_first'])                       # 候選命令自己的參考終點（主控制器求解時算出）
            elif zeroed:
                u = np.zeros(9)                                      # 已歸零後不再重算命令：g＝0、參考與預測底盤終點不變、管線末設定點不變 ⇒ 本停法下一拍輸出仍是同一個（已歸零的）命令
            else:
                r = self.step_cmd(q, s, pipe, u_prev, sigma, g, seed, se_end, bh)
                if not r['ok']:
                    return dict(ok=False, why=r['why'], n_ticks=n, X=np.array(X))
                u, sigma, g, seed, zeroed, se_end, bh = r['u'], r['sigma'], r['g'], r['q1'], r['zeroed'], r['se_end'], r['hold']
            uf = np.asarray(pipe[-1][:3], float).copy()
            q_a = q.copy()
            for j in range(NPH):
                cmd = XO.ramp_cmd(uf, u, j)
                pipe.append(cmd)
                ua = pipe.pop(0)
                q, s = self.S.plant_phys_step(q, s, ua, self.cfg, 1)
            u_prev = u
            X.append(np.r_[q, self.ref(sigma)[9]])
            if n >= hold:                                            # 閉鏈偏差只計停法段（候選段由候選自己的閉鏈驗收負責）
                Tc = self.W.FC.fk_jac(q)[0]
                Td = self.W.Rz4(-(float(X[-1][9]) - self.phi_bind)) @ self.T_WH_bind @ self.T_HE
                self._cl[0] = max(self._cl[0], float(np.linalg.norm(Tc[:3, 3] - Td[:3, 3])))
                self._cl[1] = max(self._cl[1], self.XO.rot_deg(Td, Tc))
                self._tr.append((n, round(g, 2), round(float(np.linalg.norm(Tc[:3, 3] - Td[:3, 3])) * 1e3, 2), round(float(np.abs(u[3:]).max()), 3), round(float(np.hypot(u[0], u[1])), 4), round(sigma, 3)))
            if bh is not None:
                bh_steps += NPH
            pipe_ok = (not np.any(u) and not any(np.any(x_) for x_ in pipe)) or (bh is not None and DEF.motion_cleared(bh_steps - NPH))   # 同執行層定義：零命令，或保持啟用且錨定前命令已排完
            still = (pipe_ok
                     and float(np.abs(q[3:] - q_a[3:]).max()) / DT < SS['arm_rps'] and float(np.hypot(*(q[:2] - q_a[:2]))) / DT < SS['base_mps']
                     and abs(q[2] - q_a[2]) / DT < SS['yaw_rps'] and abs(float(X[-1][9] - X[-2][9])) / DT < SS['door_rps'])
            quiet = quiet + 1 if still else 0
            if quiet >= need:
                return dict(ok=True, why=None, n_ticks=n + 1, X=np.array(X))   # 全部拍末樣本都送淨空核對（不做近似去重）
        return dict(ok=False, why='not_settled', n_ticks=max_ticks, X=np.array(X))


def _die_with_parent():
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass


def _serve(conn):
    _die_with_parent()
    import door_wbopt as WB
    import door_clear_door_fast as _F
    WB.clear_door = _F.clear_door_fast
    cs = None
    conn.send(('ready',))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            if msg[0] == 'init':
                cs = CStop(msg[1])
                conn.send(('init_ok',))
            else:
                _, ver, st = msg
                t0 = time.perf_counter()
                cs.set_ref(st['Xd'])
                r = cs.sim(st)
                conn.send((ver, r['ok'], r['why'], r['n_ticks'], r['X'], (time.perf_counter() - t0) * 1e3, list(cs._cl), list(cs._tr)))
        except Exception as e:                                        # noqa: BLE001
            conn.send((msg[1] if msg[0] != 'init' else 'init', False, 'exception:' + repr(e)[:200], 0, None, 0.0, None))


class CStopWorker:
    """r2：n 個獨立程序（預設 2：候選核對的 hold＝1／2 兩個模擬並行）。結果以版本號對應請求；遲到的舊版本結果丟棄。"""
    def __init__(self, n=2):
        ctx = mp.get_context('fork')
        self.conns, self.procs = [], []
        for _ in range(int(n)):
            a, b = ctx.Pipe()
            p_ = ctx.Process(target=_serve, args=(b,), daemon=True)
            p_.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p_)
        for c in self.conns:
            if not c.poll(120.0) or c.recv() != ('ready',):
                raise RuntimeError('cstop worker start failed')
        self.ver = 0
        self.n = int(n)

    def init(self, P):
        for c in self.conns:
            c.send(('init', P))
        for c in self.conns:                                           # 先前請求的遲到結果（未被收走）一律丟棄，等到 init_ok 為止
            t_end = time.perf_counter() + 30.0
            while True:
                if not c.poll(max(t_end - time.perf_counter(), 0.0)):
                    raise RuntimeError('cstop worker init failed')
                if c.recv() == ('init_ok',):
                    break

    def submit(self, st, w=0):
        self.ver += 1
        self.conns[w].send(('sim', self.ver, st))
        return self.ver

    def collect(self, ver, timeout, w=0):
        """回結果 dict；逾時／程序死亡 ⇒ None（呼叫端當作「結果缺失」，不是「模型判不可行」）。"""
        t_end = time.perf_counter() + timeout
        conn = self.conns[w]
        while True:
            rem = t_end - time.perf_counter()
            try:
                if rem <= 0 or not conn.poll(rem):
                    return None
                m = conn.recv()
            except (EOFError, OSError):
                return None
            if m[0] == ver:
                return dict(ok=m[1], why=m[2], n_ticks=m[3], X=m[4], sim_ms=m[5], closure=m[6] if len(m) > 6 else None, tr=m[7] if len(m) > 7 else None)

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p_ in self.procs:
            p_.join(timeout=2.0)
            if p_.is_alive():
                p_.terminate()
