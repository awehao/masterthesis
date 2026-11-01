#!/usr/bin/env python3
"""區域導引在線 MotM（Howard 20261102_howard_region_guided_online_motm、Codex 072116／072949；版本化子類，基線＝bottle_w1_vhigh_relwin_slipwarn_ctrl〔v2w〕）。
任務輸入＝已知區域（第一桌所在、導航路徑、接近車道）＋目標類別（同型直立瓶）；物品精確位置與支撐高度只來自本趟視覺（known_type 快照）。
與 v2w／v3 的差別只在「計畫從哪裡來」：
1 抓取計畫（背景工作者，bottle_plan_worker／bottle_plan_client；控制拍不阻塞）：首次合格快照 ⇒ 送出請求（目標 xy、支撐高度、抓持帶與抬升先驗、
  相對窗 [−0.35, +0.15]、實際執行的備妥構型為唯一種子）。v2 是在控制拍內同步呼叫多種子 window_branch（約 0.5 s 牆鐘，模擬同時凍結）。
  承接（每拍輪詢，結果到才核）：結果有限且可行；最新合格觀測距今 ≤ OBS_FRESH_S 且與計畫目標差 ≤ dev_tol（2 mm）、支撐高度差 ≤ 2 mm；
  底盤仍在計畫有效起始範圍：x < x_latest＝min(平移後回備妥觸發點, 名目觸發點) − 在途距離 − 煞停距離 − 0.02 m（車速取參考車速、不低於 0.30 m/s）。
  分支核對是沿抓取車道（y＝0、偏航 0，任務先驗）在抓取窗內做；採用時底盤仍在規劃路徑上併向車道（物理 A1 當時 y≈−0.85），所以不以當下 y／偏航判有效。
  承接後做 v2 相同的排程平移與目標施加（不重算）。計畫未到可繼續接近；到 x_latest 仍未承接 ⇒ 受控停止 grasp_plan_late（不凍結世界、不延期限）。
  目標更新：等待中新快照與請求目標差 > dev_tol ⇒ 舊請求作廢、重送（最多 GP_MAX 次）。
  第一桌（收近閘／外伸閘）高度改用本趟量測支撐高度（實例參數 cc_t1）；桌面 xy 範圍仍是任務先驗（資產），不是由深度重建。
2 轉回計畫（同一工作者）：liftb 切換、側持姿態確定時送出（側持 T_side、當下量測 T_rel、手臂構型、第二桌先驗）⇒ z_back、三段時長（IK 延續 NS_RET 步、
  所需×1.05 進位 0.05 s、不低於原 3.8／2.0／3.8 s）、路徑餘裕。承接＝收近開始時（收近閘的前置）：計畫已到、側持姿態與計畫相同（≤ 1 mm）、
  當下 T_rel 與計畫差 ≤ 3 mm／2°；不符 ⇒ 有界重送（最多 RP_MAX 次）再不符受控停止。未承接不開始收近。
  最晚處置點：運送路徑進度距直線段入口 ≤ v_carry×(收近＋升高下限時長)＋在途＋煞停距離 時仍未開始收近 ⇒ return_plan_late（每拍先查，等待中也查；
  初版只在結果到後才查且只看 PLACE_APPROACH，假世界 envelope 反例因此帶著抬升高度接近第二桌而碰桌，已改）。
  時長以實例參數 cc_t_to／cc_t_up／cc_t_out 給（compactcarry2／3 的 _ccd 存取：參考、收近閘剩餘時間、提前升高前置量、外伸閘與報告同一來源），不換模組常數。
  收近完成時升高／外伸目標換成 z_back（同 v3），外伸完成後 TCP 對目標 ≤ 2 mm 才累計停車靜止（同 v3）。
3 工作者已死、結果不可行或逾期：受控停止並記原因；真值安全網（判定器離桌、淨空拒絕、禁止接觸）照舊，只負責拒絕，不修正計畫。
延遲紀錄：首次合格觀測、請求、結果收到、承接（模擬時間與牆鐘分開），冷啟動另列。
"""
import atexit
import math
import os

import numpy as np

import bottle_plan_client as PCL
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_w1_compactcarry2_ctrl as C2m
import bottle_w1_vhigh_relwin_ctrl as RW
import bottle_w1_vhigh_relwin_slipwarn_ctrl as SW

C3, C2 = SW.C3, SW.C2
H_CLEAR = 0.029
XH_TCP_TOL = 0.002
OBS_FRESH_S = 2.0
GP_MAX, RP_MAX = 3, 3
NS_RET = 20
V_STOP_FLOOR = 0.30
DUR_FLOOR = {'to_compact': 3.8, 'raise': 2.0, 'out': 3.8}
RET_LIM = dict(self=0.0, bottle_link=0.0, deck=0.0, margin=0.05)


def _rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class W1PlaceController(SW.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE'

    def __init__(self, *a, planner_log=None, planner_py='/usr/bin/python3', **kw):
        super().__init__(*a, **kw)
        self.q_ready_nom = np.asarray(a[4], float).copy()         # 實際執行的備妥構型（建構參數 q_ready）＝抓取計畫的唯一種子
        self.pc = PCL.PlanClient(py=planner_py, log_path=planner_log).start()
        atexit.register(self.pc.close)
        self.gp, self.gp_n, self.gp_hist = None, 0, []
        self.rp, self.rp_n, self.rp_hist = None, 0, []
        self.obs_last, self.obs_first = None, None
        self.xh, self.xh_conv, self.t_first_gate = None, None, None
        self.cc_t1 = None

    # ---- 抓取計畫
    def _seg_freeze(self, xy, zs, seq):
        xy = np.asarray(xy, float)
        self.obs_last = dict(t=float(self.t), seq=int(seq), xy=xy.copy(), zs=float(zs))
        if self.obs_first is None:
            self.obs_first = dict(t=float(self.t), seq=int(seq), xy=xy.tolist(), zs=float(zs), wall=_wall())
            self._ev('online_obs_first', seq=int(seq))
        g, tol = self.gp, self.seg['dev_tol_mm'] * 1e-3
        if g is not None and g['state'] == 'pending':
            if float(np.linalg.norm(xy - g['xy'])) <= tol and abs(zs - g['zs']) <= tol:
                return False
            g['state'] = 'superseded'
            self.gp = None
        if self.gp is None:
            self._gp_submit(xy, zs, seq)
        return False

    def _gp_submit(self, xy, zs, seq):
        if self.gp_n >= GP_MAX:
            self.judge._stop(self.t, 'grasp_plan_retry_exceeded', n=self.gp_n)
            return
        p = getattr(self, 'q_prev2', None)
        x_now = float(p[0]) if p is not None else float('nan')
        v = max(float(getattr(self, 'v_b', 0.0)), V_STOP_FLOOR)            # 煞停距離用的車速：參考車速，不低於 0.30 m/s（高於指令上限 0.2775，保守）
        dx = float(xy[0]) - self.x_nom
        d_stop = v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0])) + 0.02
        x_latest = min(self.x_dep + dx, self.x_dep) - d_stop
        inp = dict(xy=xy.tolist(), z_support=float(zs), band=0.10, pre_m=float(self.task.pre_m), lift_m=float(self.task.lift_m),
                   win_rel=[float(self.win[0]) - self.x_nom, float(self.win[1]) - self.x_nom], q_seed=self.q_ready_nom.tolist(), x_nom=self.x_nom)
        rid = self.pc.submit('grasp', inp)
        self.gp_n += 1
        self.gp = dict(state='pending' if rid is not None else 'not_sent', id=rid, xy=np.asarray(xy, float).copy(), zs=float(zs), seq=int(seq), t_req=float(self.t), wall_req=_wall(),
                       x_req=x_now, v_req=v, d_stop=d_stop, x_latest=x_latest, dx=dx)
        self.gp_hist.append(self.gp)
        self._ev('grasp_plan_request', id=rid, seq=int(seq), x_req=round(x_now, 4), x_latest=round(x_latest, 4))
        if rid is None:
            self.judge._stop(self.t, 'planner_unavailable', kind='grasp', alive=self.pc.alive())

    def _gp_poll(self, q):
        g = self.gp
        if g is None or g['state'] != 'pending':
            return
        m = self.pc.poll(g['id'])
        if m is None:
            if not self.pc.alive():
                self.judge._stop(self.t, 'planner_dead', kind='grasp')
            elif q[0] >= g['x_latest']:
                self.judge._stop(self.t, 'grasp_plan_late', x=float(q[0]), x_latest=g['x_latest'], waited_s=float(self.t - g['t_req']))
            return
        g.update(t_res=float(self.t), wall_recv=m['t_wall_recv'], compute_wall=m['wall']['t1'] - m['wall']['t0'], ok=m['ok'], error=m['error'])
        res = m['result']
        if not m['ok']:
            g['state'] = 'infeasible'
            self.judge._stop(self.t, 'segtarget_unreachable', source='online_grasp_plan', error=m['error'], seq=g['seq'])
            return
        o, tol = self.obs_last, self.seg['dev_tol_mm'] * 1e-3
        why = None
        if o is None or self.t - o['t'] > OBS_FRESH_S:
            why = 'obs_stale'
        elif float(np.linalg.norm(o['xy'] - g['xy'])) > tol or abs(o['zs'] - g['zs']) > tol:
            why = 'obs_inconsistent'
        elif q[0] >= g['x_latest']:
            why = 'late'
        T_g = np.array(res['T_g'], float)
        if why is None and float(np.abs(T_g - C0.grasp_target(float(g['xy'][0]), float(g['xy'][1]), g['zs'] + 0.10)).max()) > 1e-9:
            why = 'target_mismatch'
        if why is not None:
            g['state'] = 'rejected:' + why
            self._ev('grasp_plan_reject', id=g['id'], why=why)
            if why in ('late', 'target_mismatch'):
                self.judge._stop(self.t, 'grasp_plan_' + why, id=g['id'])
            elif o is not None:
                self.gp = None
                self._gp_submit(o['xy'], o['zs'], o['seq'])
            return
        self._gp_apply(g, res, q)

    def _gp_apply(self, g, res, q):
        xy, zs, seq, dx = g['xy'], g['zs'], g['seq'], g['dx']
        x_now = float(q[0])
        old = {k: float(getattr(self, k)) for k in RW.SHIFT_KEYS}
        new = {k: v + dx for k, v in old.items()}
        self.relwin = dict(dx=dx, t=float(self.t), x_now=x_now, old=dict(old, win=list(self.win)), new=dict(new, win=[self.win[0] + dx, self.win[1] + dx]), seq=seq, plan_id=g['id'])
        for k, v in new.items():
            setattr(self, k, v)
        self.win = (self.win[0] + dx, self.win[1] + dx)
        self._ev('relwin_shift', dx=round(dx, 5), x_now=round(x_now, 4), win=[round(v, 4) for v in self.win], x_blend=round(self.x_blend, 4), x_dep=round(self.x_dep, 4))
        self.seg_z = zs
        self.task.T_grasp = np.array(res['T_g'], float)
        self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=0.0, n=1, margin=float(res['margin']), seq=seq, z_support=zs)
        self.w_goal = np.array(xy, float)
        self.seg_seq = seq
        self.seg_adopt.append((self.t, seq, list(map(float, xy)), zs))
        self.cc_t1 = (float(zs), np.array(C2m.TABLE1_C, float), np.array(C2m.TABLE1_HALF, float))   # 第一桌高度＝本趟量測；xy 範圍＝任務先驗
        g.update(state='accepted', t_acc=float(self.t), wall_acc=_wall(), x_acc=x_now, margin=float(res['margin']), q_mid=res['q_mid'])
        self._ev('segtarget_freeze', xy=[round(float(v), 5) for v in xy], z_support=round(zs, 5), seq=seq, margin=float(res['margin']), plan_id=g['id'])

    # ---- 轉回計畫
    def _rp_submit(self, q):
        if self.rp_n >= RP_MAX:
            self.judge._stop(self.t, 'return_plan_retry_exceeded', n=self.rp_n)
            return
        T_side = np.array(self.lb['T_fin'], float)
        T_rel = np.array(self.judge.T_rel, float)
        inp = dict(T_side=T_side.tolist(), T_rel=T_rel.tolist(), q_arm=np.asarray(q, float)[3:].tolist(), top2=float(self.place['top']), z_pad=C2m.Z_PAD, h_clear=H_CLEAR,
                   tcp_c=np.asarray(C2m.TCP_C, float).tolist(), ns=NS_RET, scale=1.05, step=0.05, floor=dict(DUR_FLOOR), limits=RET_LIM)
        rid = self.pc.submit('return', inp)
        self.rp_n += 1
        self.rp = dict(state='pending' if rid is not None else 'not_sent', id=rid, t_req=float(self.t), wall_req=_wall(), T_side=T_side, T_rel=T_rel)
        self.rp_hist.append(self.rp)
        self._ev('return_plan_request', id=rid, n=self.rp_n)
        if rid is None:
            self.judge._stop(self.t, 'planner_unavailable', kind='return', alive=self.pc.alive())

    def _rp_late(self):
        """轉回最晚處置點：收近必須在運送路徑進度距直線段入口 ≥ v_carry×(收近＋升高下限時長)＋煞停距離 時已開始；否則手臂仍在抬升高度就會接近第二桌。"""
        if self.cc is not None:
            return False
        if self.pph == 'PLACE_APPROACH':
            return True
        if self.pph != 'CARRY_NAV' or self.cpath is None or self.csig is None:
            return False
        v = max(float(getattr(self, 'v_carry', 0.25)), 0.05)
        need = v * (DUR_FLOOR['to_compact'] + DUR_FLOOR['raise']) + v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0]))
        return float(self.cpath.x_lane) - float(self.csig) <= need

    def _rp_poll(self, q):
        if self._rp_late():
            self.judge._stop(self.t, 'return_plan_late', state=None if self.rp is None else self.rp['state'], rem=None if self.csig is None else float(self.cpath.x_lane) - float(self.csig))
            return
        if self.rp is None:
            if getattr(self, 'lb', None) is not None and self.judge.T_rel is not None and self.judge.t_grasp is not None:
                self._rp_submit(q)
            return
        r = self.rp
        if r['state'] == 'pending':
            m = self.pc.poll(r['id'])
            if m is None:
                if not self.pc.alive():
                    self.judge._stop(self.t, 'planner_dead', kind='return')
                return
            r.update(t_res=float(self.t), wall_recv=m['t_wall_recv'], compute_wall=m['wall']['t1'] - m['wall']['t0'], ok=m['ok'], error=m['error'], res=m['result'])
            if not m['ok']:
                r['state'] = 'infeasible'
                self.judge._stop(self.t, 'return_plan_infeasible', error=m['error'])
                return
            r['state'] = 'ready'
            self._ev('return_plan_ready', id=r['id'], dur={k: round(v, 2) for k, v in m['result']['dur'].items()}, z_back=round(m['result']['z_back'], 4))

    def _rp_accept(self):
        r = self.rp
        if r is None or r['state'] != 'ready':
            return False
        Tc = np.asarray(self.st.T_carry_rel, float) if self.st.T_carry_rel is not None else None
        Tn = np.asarray(self.judge.T_rel, float)
        why = None
        if Tc is None or float(np.linalg.norm(Tc[:3, 3] - r['T_side'][:3, 3])) > 0.001 or _rot_deg(Tc[:3, :3], r['T_side'][:3, :3]) > 0.5:
            why = 'side_pose_changed'
        elif float(np.linalg.norm(Tn[:3, 3] - r['T_rel'][:3, 3])) > 0.003 or _rot_deg(Tn[:3, :3], r['T_rel'][:3, :3]) > 2.0:
            why = 'envelope_changed'
        if why is not None:
            r['state'] = 'rejected:' + why
            self._ev('return_plan_reject', id=r['id'], why=why)
            self.rp = None                                            # 下一拍有界重送（_rp_poll）
            return False
        d = r['res']['dur']
        self.cc_t_to, self.cc_t_up, self.cc_t_out = float(d['to_compact']), float(d['raise']), float(d['out'])
        r.update(state='accepted', t_acc=float(self.t), wall_acc=_wall())
        self._ev('return_plan_accept', id=r['id'], T_TO=self.cc_t_to, T_UP=self.cc_t_up, T_OUT=self.cc_t_out)
        return True

    def _gate(self, q, t_rem):
        if self.cc is None:                                           # 收近開始前：先有已核可的轉回計畫才承接
            if self.t_first_gate is None:
                self.t_first_gate = float(self.t)
            if self.cc_t1 is None:
                self.judge._stop(self.t, 'table1_measured_missing')
                return None, 0
            if self.rp is None or self.rp['state'] != 'accepted':
                if not self._rp_accept():
                    return None, 0
            t_rem = self._ccd('T_TO')
        return super()._gate(q, t_rem)

    # ---- 每拍
    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None:
            self._gp_poll(q)
        if self.judge.stop is None and self.vfix is not None:
            self._rp_poll(q)
        if self.judge.stop is not None:
            return
        super()._advance_inner(T_tcp, close_done, q)
        if self.judge.stop is not None or self.cc is None:
            return
        f = self.cc
        if self.xh is None and f['stage'] in ('compact', 'raise', 'raised', 'out', 'side'):
            res = self.rp['res']
            T_side = np.array(f['T_side'], float)
            T_back = T_side.copy()
            T_back[2, 3] = float(res['z_back'])
            self.xh = dict(t=float(self.t), z_side_lift=float(T_side[2, 3]), z_back=float(res['z_back']), zmin_rel=float(res['zmin_rel']), top2=float(self.place['top']),
                           h_clear=H_CLEAR, bottom_at_back=float(res['z_back']) + float(res['zmin_rel']))
            f['T_side_lift'], f['T_side'] = T_side, T_back
            self._ev('xh_back', z_side_lift=round(float(T_side[2, 3]), 4), z_back=round(float(res['z_back']), 4), zmin_rel=round(float(res['zmin_rel']), 4), top2=float(self.place['top']))
        if f['stage'] == 'side' and self.pph == 'PLACE_APPROACH' and self.xh is not None:
            Tt = self.K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(q, float)[3:]], 'link_tcp')
            e = float(np.linalg.norm(Tt[:3, 3] - np.asarray(f['T_side'], float)[:3, 3]))
            if e > XH_TCP_TOL:
                self.n_still_p = 0
            elif self.xh_conv is None:
                self.xh_conv = dict(t=float(self.t), err=e)
                self._ev('xh_tcp_converged', err_mm=round(e * 1e3, 2))

    def report(self):
        r = super().report()
        self.pc.close()

        def gsum(g):
            return {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in g.items() if k not in ('T_side', 'T_rel', 'res')}
        rp_res = [dict(gsum(x), res={k: v for k, v in (x.get('res') or {}).items() if k not in ('q_side', 'T_side', 'T_rel')}) for x in self.rp_hist]
        lat = None
        ga = next((g for g in self.gp_hist if g['state'] == 'accepted'), None)
        if ga is not None and self.obs_first is not None:
            lat = dict(obs_first_to_accept_sim=ga['t_acc'] - self.obs_first['t'], obs_first_to_accept_wall=ga['wall_acc'] - self.obs_first['wall'],
                       request_to_result_sim=ga['t_res'] - ga['t_req'], request_to_result_wall=ga['wall_recv'] - ga['wall_req'], compute_wall=ga['compute_wall'])
        ra = next((x for x in self.rp_hist if x['state'] == 'accepted'), None)
        rl = None
        if ra is not None:
            rl = dict(request_to_ready_sim=ra['t_res'] - ra['t_req'], request_to_ready_wall=ra['wall_recv'] - ra['wall_req'], compute_wall=ra['compute_wall'],
                      first_gate=self.t_first_gate, ready_after_first_gate_s=max(0.0, ra['t_res'] - (self.t_first_gate or ra['t_res'])), accept_t=ra['t_acc'])
        r['online'] = dict(obs_first=self.obs_first, grasp_plans=[gsum(g) for g in self.gp_hist], return_plans=rp_res, grasp_latency=lat, return_latency=rl,
                           t1_measured=None if self.cc_t1 is None else dict(top=self.cc_t1[0], center_prior=self.cc_t1[1].tolist(), half_prior=self.cc_t1[2].tolist()),
                           durations=dict(T_TO=getattr(self, 'cc_t_to', None), T_UP=getattr(self, 'cc_t_up', None), T_OUT=getattr(self, 'cc_t_out', None)),
                           xh=self.xh, xh_conv=self.xh_conv, client=self.pc.stats(),
                           note='目標與計畫只來自本趟量測＋任務先驗（區域、車道、備妥、抓持帶、第二桌）；真值安全網只負責拒絕')
        return r


def _wall():
    import time
    return time.time()
