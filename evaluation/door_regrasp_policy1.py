#!/usr/bin/env python3
"""換抓任務端策略（純邏輯；Codex r49–r51；規格 REGRASP_inner_contact_prep_draft2.md §3–§8）。door_k3m_policy_w2 以上全部不改。

繼承 door_k3m_policy_w2.K3MW2Policy：開門 → PAUSE → 求解端 HELD（10 s 觀察通過）→ mode 1 測試保持 test_hold_s（全部沿用 W2）。
差異：mode 1 保持滿後**不 DONE**，改依序請求 mode 2–6（請求、提交、套用確認一律以執行端權威 /door/w2_mode_state 為準；權威類別 door_regrasp_mode1）。
相位：RG2 RELEASE_OPEN → RG3 RETREAT_WB → RG4 STOW_MOVE → RG5 APPROACH_INNER → RG6 CLOSE_INNER → DONE（＝內側實際夾持＋2 s 短保持確認；不是拉關或通過）。
每個模式 k：
  - 請求條件：前一模式已提交且 applied_confirmed、且其完成證據成立——
      2：mode 1 保持滿 test_hold_s；3：執行端 gate3（解除確認）CONFIRMED；4：mode 3 參考走完＋gate4 CONFIRMED；
      5：mode 4 參考走完＋gate5 CONFIRMED；6：mode 5 參考走完＋gate6 CONFIRMED。gate 狀態只讀執行端權威狀態的 gates 欄（任務端不自行判定）。
  - 三個時刻分記：權威提交 t_commit（權威 commits 列）、任務端看到 applied_confirmed 的 t_seen、參考起算 t_ref0（＝t_seen 當輪）；之前目標保持上一模式終點。
  - 逾時：請求後 commit_timeout_s 未提交 ⇒ 中止；參考段走完後 gate 未成立 ⇒ 等到「段長＋gate_wait_s」仍未成立 ⇒ 中止 rg{k}_gate_timeout。
目標（door_regrasp_ref_v1 參考包，起始銜接見 RefRuntime）：TCP 4×4、底盤引導 coord（τ 查表前饋＋k(參考−實測)，世界每軸 0.035、偏航 0.10，w_vref 1；
  保持形模式 2、6 不送底盤引導）。v3 淨空監看目標：mode 2–3 out、4 None、5 None→（TCP 距預抓取 ≤ pre_tol 5 mm 後）in、6 in。
換抓相位的監看（父類 ACTIVE_MONITOR 的開門監看與外側夾持檢查**不用**——其「非設計接觸」以外側把手座標分類，內側夾持會被誤判）：
  資料年齡（state、contacts、v3 摘要 ≤ stale_s 0.2）、v3 摘要 complete；門角相對解除時 ≤ door_dev_deg 1.0°（起始銜接登錄值）、|q̇| ≤ door_abort_rate 0.05；
  牆接觸 ≤ contact_abort_n 0.1；腕力 < wrist_abort_n 30；求解失敗／狀態停更；v3 淨空監看（door_k3m_geom_monitor_v3＋既有 GeometryMonitor 判定）；
  v3 指定接觸（窗口最大值，逐部位逐類）：非手指任何類別 > 0.1 N ⇒ 中止；手指：mode 2 只允許外側類（bar_out、post_out*、ambiguous）；
  mode 3 退開量 s ≥ S_EXIT 後外側類 > 0.1 N ⇒ 中止（重新接觸）；mode 4、5 任何類別 > 0.1 N ⇒ 中止（閉爪前不允許碰觸）；
  mode 6 只允許 bar_in，其他類別（含 ambiguous、立柱）> 0.1 N ⇒ 中止。執行端 gate／短保持 FAILED（權威狀態）⇒ 中止。
"""
from __future__ import annotations

import json
import math

import numpy as np

import door_k3m_policy_w2 as PW2
import door_regrasp_mode1 as RG

VERSION = 'door_regrasp_policy1/r1'
Meas = PW2.Meas
OUTER = ('bar_out', 'post_out0', 'post_out1', 'ambiguous')
ALL = ('bar_out', 'bar_in', 'post_out0', 'post_out1', 'post_in0', 'post_in1', 'panel', 'ambiguous')
FINGERS = ('uflite_finger1', 'uflite_finger2')
REG = {'door_dev_deg': 1.0, 'stale_s': 0.2, 'contact_abort_n': 0.1, 'wrist_abort_n': 30.0, 'door_abort_rate': 0.05,
       'commit_timeout_s': 2.0, 'gate_wait_s': 10.0, 'pre_tol_m': 0.005, 's_exit_m': 0.02,
       'k_lin': 1.0, 'k_yaw': 1.0, 'v_cap': 0.035, 'w_cap': 0.10, 'w_vref': 1.0}


def Rz4(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RefRuntime:
    """參考包的執行期取用（起始銜接、門角修正）。所有超出登錄範圍 ⇒ 回 (None, why)，呼叫端停止並報偏差。"""

    def __init__(self, pkg):
        self.p = pkg
        self.dt = float(pkg['dt'])
        self.seg = {int(k): v for k, v in pkg['segments'].items()}
        self.br = pkg['bridge']
        self.ctx = {}

    def length_s(self, k):
        return float(self.seg[k]['t'][-1])

    def start(self, k, base, arm, q_door_deg, T_tcp):
        """模式 k 參考起算；核起始銜接範圍並建立銜接變換。回傳 (ok, why, detail)。"""
        sg = self.seg[k]
        b0, a0 = np.asarray(sg['base'][0], float), np.asarray(sg['arm'][0], float)
        b, a = np.asarray(base, float), np.asarray(arm, float)
        dq = float(q_door_deg) - float(self.p['q_ref_deg'])
        det = {'k': k, 'base_dev_m': float(np.linalg.norm(b[:2] - b0[:2])), 'yaw_dev_deg': math.degrees(abs(wrap(b[2] - b0[2]))),
               'arm_dev_rad': float(np.max(np.abs(a - a0))), 'door_dev_deg': dq}
        if abs(dq) > self.br['door_deg'] + 1e-12:
            return False, 'bridge_door', det
        if k in (3, 4):
            if det['base_dev_m'] > self.br['base_m'] or det['yaw_dev_deg'] > self.br['base_deg']:
                return False, 'bridge_base', det
            if det['arm_dev_rad'] > self.br['arm_rad']:
                return False, 'bridge_arm', det
        if k == 3:
            T0 = np.asarray(T_tcp, float)
            self.ctx[3] = {'T0': T0, 'b_off': b - b0}
        if k in (5, 6):
            ok_dq = self.p.get('door_bridge_check', {}).get('dq_ok_deg', [])
            if not ok_dq or dq < min(ok_dq) - 1e-12 or dq > max(ok_dq) + 1e-12:
                return False, 'bridge_door_unverified', det
            self.ctx[k] = {'dq_rad': math.radians(dq)}
        return True, None, det

    def at(self, k, tau):
        """回傳 (T_tcp 4×4, base_ref 3, base_ff_world 3)。tau＞段長 ⇒ 停在段末、前饋 0。"""
        sg = self.seg[k]
        n = len(sg['t'])
        i = min(int(round(max(0.0, tau) / self.dt)), n - 1)
        j = min(i + 1, n - 1)
        T = np.asarray(sg['tcp'][i], float).reshape(4, 4)
        b = np.asarray(sg['base'][i], float)
        ff = np.zeros(3) if (i >= n - 1) else (np.asarray(sg['base'][j], float) - b) / self.dt
        if i < n - 1:
            ff[2] = wrap(sg['base'][j][2] - sg['base'][i][2]) / self.dt
        if k == 3:
            c = self.ctx[3]
            s = float(sg['s'][i])
            Tz = np.eye(4)
            Tz[2, 3] = -s
            T = c['T0'] @ Tz
            b = b + c['b_off']
        if k in (5, 6):
            dq = self.ctx[k]['dq_rad']
            w = self._w5(i) if k == 5 else 1.0
            R = Rz4(-w * dq)
            T = R @ T
            bxy = R[:2, :2] @ b[:2]
            ffxy = R[:2, :2] @ ff[:2]
            b = np.array([bxy[0], bxy[1], b[2] - w * dq])
            ff = np.array([ffxy[0], ffxy[1], ff[2]])
        return T, b, ff

    def _w5(self, i):
        sg = self.seg[5]
        nA = self.p.get('mode5_n_approach')
        if nA is None:
            pre = np.asarray(self.p['pregrasp_T'], float)
            nA = 1 + min(range(len(sg['tcp'])), key=lambda j: float(np.max(np.abs(np.asarray(sg['tcp'][j]) - pre))))
            self.p['mode5_n_approach'] = nA
        return min(1.0, i / max(1, nA - 1)) if i < nA else 1.0


class RegraspPolicy(PW2.K3MW2Policy):
    # r3（行為不變的表驅動重構，供關門版子類換表）：模式表、提交後幾何目標、各模式完成所看的 gate、終止模式、無參考（保持形）模式
    RGT = RG
    GEOM_FOR = {2: 'out', 3: 'out', 4: None, 5: None, 6: 'in'}
    GATE_FOR = {2: 'gate3', 3: 'gate4', 4: 'gate5', 5: 'gate6', 6: 'hold'}
    FINAL_MODE = 6
    HOLD_REF = (2, 6)

    def __init__(self, *a, ref_pkg, geom_v3_fn=None, reg=None, **kw):
        super().__init__(*a, **kw)
        self.r = dict(REG, **(reg or {}))
        self.ref = RefRuntime(ref_pkg)
        self.geom_v3_fn = geom_v3_fn
        self.rg = {'k': 1, 'req': {}, 't_req': {}, 't_commit': {}, 't_seen': {}, 't_ref0': {}, 'bridge': {}, 'events': [],
                   'q_rel_deg': None, 'last_T': None, 'geom_target': 'out', 'gate_wait_t0': None}
        self.rg_outbox = []
        self.c3 = None
        self.c3_t = None

    # ---- 輸入：v3 接觸窗口摘要（節點每週期組裝：{'complete': bool, 'max': {部位: {類別: N}}, 'src_step': int}）
    def on_c3(self, msg, t_recv):
        if isinstance(msg, dict):
            self.c3, self.c3_t = msg, t_recv

    def take_w2_outbox(self):
        out = super().take_w2_outbox()
        if self.phase != 'ABORTED':
            out += self.rg_outbox
        self.rg_outbox = []
        return out

    def _act(self):
        return {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False, 'stop_solver': False,
                'finish': False, 'coord': None, 'geom_target': self.rg['geom_target'], 'rg': None}

    def _rg_abort(self, why, m, act, **kw):
        self.rg['events'].append(dict({'t': m.t, 'abort': why, 'k': self.rg['k']}, **kw))
        self._abort(why, m.t, rg_k=self.rg['k'], **kw)
        self.rg_outbox.clear()
        return self._aborted_act(act, m)

    def step(self, m: Meas, dt: float):
        if self.phase == 'PAUSE':
            if self.w2_state == 'committed' and m.t - self.w2_t['committed'] >= self.w2_test_hold_s:
                return self._enter_rg(m, dt)                             # 父類此刻會 DONE；改轉換抓（不送 finish／stop_solver）
            return super().step(m, dt)
        if self.phase.startswith('RG'):
            return self._rg_step(m, dt)
        return super().step(m, dt)

    # ---- 進入換抓
    def _enter_rg(self, m, dt):
        act = self._act()
        st = self.w2_last_state
        if not isinstance(st, dict) or st.get('mode_seq') != 1 or st.get('aborted'):
            return self._rg_abort('rg_enter_state_invalid', m, act)
        self.rg['q_rel_deg'] = math.degrees(m.q_door)
        self.rg['last_T'] = np.asarray(self.last_target, float) if self.last_target is not None else np.asarray(m.T_tcp, float)
        self._go('RG2', m.t, q_rel_deg=self.rg['q_rel_deg'])
        self._request(2, m)
        act['target'] = self.rg['last_T']
        return act

    def _request(self, k, m):
        req = {'run_id': self.w2_run_id, 'req_seq': k, 'from_mode': self.RGT.MODE_OF[k - 1], 'to_mode': self.RGT.MODE_OF[k], 'mode_seq': k}
        self.rg_outbox.append(dict(req))
        self.rg['req'][k], self.rg['t_req'][k] = req, m.t
        self.rg['events'].append({'t': m.t, 'request': k})

    def _commit_of(self, st, k):
        for c, rs in zip(st.get('commits') or [], st.get('commit_req_seqs') or []):
            if isinstance(c, dict) and c.get('mode_seq') == k and rs == k:
                return c
        return None

    def _monitor_rg(self, m, act):
        r = self.r
        if not all(v is not None and math.isfinite(v) for v in (m.state_age, m.contacts_age)) or m.state_age > r['stale_s'] \
                or m.contacts_age > r['stale_s']:
            return self._rg_abort('rg_monitor_stale', m, act)
        if self.c3 is None or self.c3_t is None or m.t - self.c3_t > r['stale_s'] or self.c3.get('complete') is not True:
            return self._rg_abort('rg_contact_v3_stale_or_incomplete', m, act)
        if m.q_door is None or m.qdot_door is None or not math.isfinite(m.q_door) or not math.isfinite(m.qdot_door):
            return self._rg_abort('rg_door_nonfinite', m, act)
        if abs(math.degrees(m.q_door) - self.rg['q_rel_deg']) > r['door_dev_deg'] or abs(m.qdot_door) > r['door_abort_rate']:
            return self._rg_abort('rg_door_moved', m, act, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
        if m.wall_max is None or not math.isfinite(m.wall_max) or m.wall_max > r['contact_abort_n']:
            return self._rg_abort('rg_wall_contact', m, act, n=m.wall_max)
        if m.wrist_max is None or not math.isfinite(m.wrist_max) or m.wrist_max >= r['wrist_abort_n']:
            return self._rg_abort('rg_wrist_force', m, act, n=m.wrist_max)
        if m.solver_failed:
            return self._rg_abort('rg_solver_failed', m, act, detail=m.solver_failed)
        if m.solver_up and (m.solver_status_age is None or m.solver_status_age > self.c.solver_stale_s):
            return self._rg_abort('rg_solver_status_stale', m, act)
        if getattr(m, 'executor_violation', None) is not None:
            return self._rg_abort('rg_executor_violation', m, act, detail=m.executor_violation)
        mx = self.c3.get('max') or {}
        for link, cl in mx.items():
            for c, f in (cl or {}).items():
                if not (isinstance(f, (int, float)) and math.isfinite(f)):
                    return self._rg_abort('rg_contact_v3_nonfinite', m, act)
                if f <= r['contact_abort_n']:
                    continue
                if link not in FINGERS:
                    return self._rg_abort('rg_nonfinger_contact', m, act, link=link, cls=c, n=f)
                if c not in self._allowed():
                    return self._rg_abort('rg_contact_not_designated', m, act, link=link, cls=c, n=f)
        if self.geom_v3_fn is not None:
            self.geom_v3_fn.set_target(self.rg['geom_target'])
            dec, det = self.geom.step(m.t, getattr(m, 'qf', None), m.q_door, getattr(m, 'q_finger', None), m.state_age)
            self.geom_last, self.geom_t = (dec, det), m.t
            if dec == 'stop':
                return self._rg_abort(f"rg_geometry:{det.get('why')}", m, act, detail=det)
        return None

    def _extra_ready(self, k, m):
        """子類可加的請求下一模式前置（預設無）。回傳 (ok, why)。"""
        return True, None

    def _allowed(self):
        """手指允許接觸的類別：依**目前已提交、正在執行**的模式（請求中未提交仍沿用目前模式）。"""
        kc = self.rg.get('k_committed', 1)
        if kc in (1, 2):
            return OUTER
        if kc == 3:
            return OUTER if (self.rg.get('s_meas') or 0.0) < self.r['s_exit_m'] else ()
        if kc == 6:
            return ('bar_in',)
        return ()

    def _coord(self, m, b_ref, ff):
        r = self.r
        bp = getattr(m, 'base_pose', None)
        if bp is None or not all(math.isfinite(float(v)) for v in bp):
            return None
        x, y, th = (float(v) for v in bp)
        e = np.array([b_ref[0] - x, b_ref[1] - y, wrap(b_ref[2] - th)])
        vw = np.asarray(ff, float) + np.array([r['k_lin'], r['k_lin'], r['k_yaw']]) * e
        vw = np.array([np.clip(vw[0], -r['v_cap'], r['v_cap']), np.clip(vw[1], -r['v_cap'], r['v_cap']), np.clip(vw[2], -r['w_cap'], r['w_cap'])])
        c, s = math.cos(th), math.sin(th)
        vb = (c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], float(vw[2]))
        return [1.0, r['w_vref'], vb[0], vb[1], vb[2], 0.0] + [float('nan')] * 6 + [float('nan'), float('nan')]

    def _rg_step(self, m, dt):
        act = self._act()
        k = self.rg['k_target'] if 'k_target' in self.rg else 2
        self.rg['k'] = k
        st = self.w2_last_state
        if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
            return self._rg_abort('rg_state_stale', m, act)
        if isinstance(st, dict) and st.get('aborted'):
            return self._rg_abort(f"rg_authority_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
        gates = (st or {}).get('gates') or {}
        for g, gs in gates.items():
            if isinstance(gs, dict) and gs.get('state') == 'FAILED':
                return self._rg_abort(f"rg_exec_{g}_failed:{gs.get('why')}", m, act)
        res = self._monitor_rg(m, act)
        if res is not None:
            return res
        # 提交與套用確認
        kc = self.rg.get('k_committed', 1)
        if kc < k:
            rj = (st or {}).get('last_reject')
            if isinstance(rj, dict) and rj.get('req_seq') == k:
                return self._rg_abort(f'rg{k}_request_rejected:{rj.get("why")}', m, act)
            c = self._commit_of(st or {}, k)
            if c is not None and c.get('applied_confirmed') is False:
                return self._rg_abort(f'rg{k}_commit_not_applied', m, act)
            if c is not None and c.get('applied_confirmed') is True and (st or {}).get('mode_seq') == k:
                bp, arm = getattr(m, 'base_pose', None), getattr(m, 'qf', None)
                ok, why, det = self.ref.start(k, bp, None if arm is None else list(arm)[3:9], math.degrees(m.q_door), m.T_tcp) \
                    if k >= 3 else (True, None, None)
                self.rg['bridge'][k] = det
                if not ok:
                    return self._rg_abort(f'rg{k}_{why}', m, act, detail=det)
                self.rg['k_committed'] = k
                self.rg['t_commit'][k], self.rg['t_seen'][k], self.rg['t_ref0'][k] = c.get('t'), m.t, m.t
                self.rg['geom_target'] = self.GEOM_FOR[k]
                self.rg['events'].append({'t': m.t, 'committed': k, 't_commit': c.get('t'), 'bridge': det})
                self._go(f'RG{k}', m.t)
            elif m.t - self.rg['t_req'][k] > self.r['commit_timeout_s']:
                return self._rg_abort(f'rg{k}_commit_timeout', m, act)
            act['target'] = self.rg['last_T']                               # 未提交：保持上一模式終點（參考不推進）
            act['rg'] = {'k': k, 'state': 'requested'}
            return act
        # 已提交：目標與完成判斷
        tau = m.t - self.rg['t_ref0'][k]
        if k in self.RGT.MOTION_MODES or k == 6:
            T, b_ref, ff = self.ref.at(k, tau)
            act['target'] = T
            if k in self.RGT.MOTION_MODES:
                act['coord'] = self._coord(m, b_ref, ff)
            if k == 3 and m.T_tcp is not None:
                c3 = self.ref.ctx[3]
                self.rg['s_meas'] = float((c3['T0'][:3, 3] - np.asarray(m.T_tcp, float)[:3, 3]) @ c3['T0'][:3, 2])
            if k == 5 and self.rg['geom_target'] is None and m.T_tcp is not None:
                pre = np.asarray(self.ref.p['pregrasp_T'], float).reshape(4, 4)
                pre = Rz4(-self.ref.ctx[5]['dq_rad']) @ pre
                if float(np.linalg.norm(np.asarray(m.T_tcp, float)[:3, 3] - pre[:3, 3])) <= self.r['pre_tol_m']:
                    self.rg['geom_target'] = 'in'
                    self.rg['events'].append({'t': m.t, 'geom_target': 'in'})
        else:
            act['target'] = self.rg['last_T']
        self.rg['last_T'] = np.asarray(act['target'], float)
        act['geom_target'] = self.rg['geom_target']
        done_ref = (k in self.HOLD_REF) or tau >= self.ref.length_s(k) - 1e-9
        gk = self.GATE_FOR[k]
        gs = gates.get(gk) if isinstance(gates.get(gk), dict) else {}
        act['rg'] = {'k': k, 'state': 'running', 'tau': tau, 'gate': gs.get('state')}
        if done_ref:
            xr, xwhy = self._extra_ready(k, m)
            if gs.get('state') == 'CONFIRMED' and gs.get('fresh') is True and xr:
                if k == self.FINAL_MODE:
                    self._go('DONE', m.t, rg='inner_grasp_hold_confirmed', hold=gs)
                    self.done = True
                    act.update(finish=True, stop_solver=False)
                    self.rg['events'].append({'t': m.t, 'to': 'DONE'})
                    return act
                self.rg['k_target'] = k + 1
                self._request(k + 1, m)
                self.rg['gate_wait_t0'] = None
                return act
            self.rg['gate_wait_t0'] = self.rg['gate_wait_t0'] if self.rg['gate_wait_t0'] is not None else m.t
            if m.t - self.rg['gate_wait_t0'] > self.r['gate_wait_s']:
                return self._rg_abort(f'rg{k}_gate_timeout', m, act, gate=gs)
        return act

    def rg_report(self):
        def j(x):
            return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        return {'version': VERSION, 'reg': dict(self.r), 'k_committed': self.rg.get('k_committed'), 't_req': self.rg['t_req'],
                't_commit': self.rg['t_commit'], 't_seen': self.rg['t_seen'], 't_ref0': self.rg['t_ref0'], 'bridge': j(self.rg['bridge']),
                'q_rel_deg': self.rg['q_rel_deg'], 'events': j(self.rg['events'][:300])}
