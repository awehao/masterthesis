#!/usr/bin/env python3
"""接近末段排程提速（bottle_w1_approachsched_ctrl）對 carrygov 基線的最小離線對照（Codex 20261102_193444；假世界、假候選，非物理；不掃參數）。
兩案（R-B1、R-B2）× 兩版（base＝carrygov、cand＝approachsched）。報：漸變起點、提早 R、INSERT→離桌確立、實際抓取段速度（插入到抓持成立的實測車速）、完整任務時間、
接近末段（R 開始到漸變起點）的實測最低淨空（手臂／底盤對第一桌、第二桌、原障礙、瓶；自碰；關節餘裕）與每控制拍現在停／延後一拍停的完整取樣停止軌跡、
近停（承接後 CRUISE 參考車速 < 0.02）與等手臂（抵達漸變起點到漸變開始）時間、接近末段最高實測車速。
    python3 evaluation/test_bottle_w1_approachsched.py all      （依序，寫 results/bottle/approachsched_check.json）
以下為沿用的 carrygov 核對說明。
運送進度協調（bottle_w1_carrygov_ctrl）的最小核對（Codex 20261102_175049；假世界、假候選，非物理；不做速度或位置掃描）。
案例：
  rb2    R-B2 短路徑（接近＝rb2_offline_plan 離線規劃、O＝(0.8, −0.8)、第一桌 0.645、瓶 區域 (0.0607, 0.4048)）：期望完整抓放；記持守位置、減速過程。
  rb1    R-B1 原路徑（接近＝bottle_rb1_1 實跑規劃、第一桌 0.58）：期望持守點從不限速（無 bind 事件）、完成時刻同視向版（48.4 s）。
  delay  R-B2＋轉回計畫結果晚 3.0 s 交付（模擬時間）：期望底盤在直線段入口前有界減速、停在持守點等待，收到後照常完成。
  fail   R-B2＋轉回計畫結果永不交付（工作者仍活）：期望持守點等待到 HOLD_MAX 後 carry_hold_timeout 受控停止，持瓶、無接觸、底盤停在持守點前。
  recede 延遲案＋收近開始後 0.6 s 把控制器的第二桌先驗（place['c']）沿 −x 移動，使幾何危險邊界落在底盤目前進度前 0.04 m（持守點突然後退，來不及停）：
         期望停止模型監看觸發 carry_hold_overrun 受控停止；持瓶段停止軌跡對實際兩桌照核。
  recede_mild 同上但邊界在前 0.12 m：期望停止模型監看觸發，或來得及停住且停止模型餘裕始終 ≥ 0。
  nonfinite 延遲案＋收近開始後 0.6 s 起持守包絡回傳非有限：期望 carry_gov_invalid 受控停止、不放行到停車位。
每案：持瓶段（離桌確立 t_lift_ok＋0.2 s 起，同實際瓶對桌；到鬆爪或停止）每 5 控制拍＋持守 bind／release、收近、外伸等事件 ±5 拍，做現在停（e0）與延後一拍停（e1）的完整取樣停止軌跡（compactcarry21.stop_traj），
  核非手指／手指對兩桌、持瓶對兩桌（放置相位排除第二桌桌面）、自碰、關節餘裕；另記實際瓶對兩桌最小距離、持守紀錄、參考車速最小值與減速度。
假世界節流 RTF 0.6、落後重新錨定（同 test_bottle_w1_gaze）。
    python3 evaluation/test_bottle_w1_carrygov.py all       （依序，寫 results/bottle/carrygov_check.json）
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'approachsched_check.json')
ORDER = ('rb1:base', 'rb1:cand', 'rb2:base', 'rb2:cand')
GEO = dict(rb2=dict(run=None, plan='rb2_offline_plan.json', z1=0.645, b=(0.0607, 0.4048), O=(0.80, -0.80)),
           rb1=dict(run='bottle_rb1_1', z1=0.58, b=(-0.03, 0.44), O=(0.60, 0.80)))

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res = {}
    for k in ORDER:
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        res[k] = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:])
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('outcome', 't_done', 'blend', 'early_R', 'insert_to_liftok', 'grasp_speed', 'approach_stop_fails', 'approach_min', 'near_stop_s', 'arm_wait_s', 'v_max_approach', 'gate', 'error')}, flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    sys.exit(0)

CASE, VER = sys.argv[1].split(':')
CF = GEO[CASE]
Z1, D_U = CF['z1'], 3.6
sys.argv = [sys.argv[0], f'{Z1:.3f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = CF['b']
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_ctrl as B_                            # noqa: E402
import math                                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_carrygov_ctrl as CG                    # noqa: E402
import bottle_w1_approachsched_ctrl as AS               # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import test_bottle_place as TP                          # noqa: E402

T = TV.T
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_region as BRG                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
REG = BRG.RegionFrame(*CF['O'])
PLACE_R = REG.place_to_region(PC.PLACE)
PC.PLACE.clear()
PC.PLACE.update(PLACE_R)
TP.DX, TP.DY = 2.4 - CF['O'][0], -1.0 - CF['O'][1]
SC.X0 = -3.6 - CF['O'][0]
SC.OBST['c'] = np.array([-2.0 - CF['O'][0], 0.0 - CF['O'][1], 0.25])
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', CF['run'], 's1_report.json')))['plan']['joined'] if CF.get('run') else
                 json.load(open(os.path.join(HERE, 'results', 'bottle', CF['plan'])))['approach']['joined_region'], float)
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
EV_HOT = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned', 'xh_back', 'xh_tcp_converged', 'place_phase',
          'return_plan_accept', 'carry_gov_bind', 'carry_gov_release')
REC, TBLOG = [], []
RTF_CAP = float(os.environ.get('CG_RTF', '0.6'))
DBG = os.environ.get('CG_DEBUG')
LOGP = os.path.join(HERE, 'results', 'bottle', f'approachsched_{CASE}_{VER}_worker.log')
N_MAX = 2000 // C21.SAMPLE_EVERY + 1


def chk(q9, T_rel, f_cmd, placing):
    W = G.link_world(q9, f_cmd)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for _, c_, s_ in TBN), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for _, c_, s_ in TBN),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        Pb = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
        out['held_bottle_table'] = min(float(G.box_dist(Pb, c_, s_).min()) for n, c_, s_ in TBN if not (placing and n == 't2:table_top'))
    return out


OBX = (SC.OBST['c'], SC.OBST['sz'])
BOXC, BOXS = np.array([CF['b'][0], CF['b'][1], Z1 + G.BOTTLE['h'] / 2]), np.array([2 * G.BOTTLE['r'], 2 * G.BOTTLE['r'], G.BOTTLE['h']])
ALIM = dict(table=0.05, obst=0.05, bottle=0.05, self=0.008017 - 1e-6, margin=0.05)
AST = dict(n=0, worst={}, where={}, nonfinite=0, unfinished=0, meas={})
GATE = []


def achk(q9):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for _, c_, s_ in TBN), obst=float(G.box_dist(P, *OBX).min()), bottle=float(G.box_dist(P, BOXC, BOXS).min()),
                self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=BB.margin(q9[3:]))


BASE = AS.W1PlaceController if VER == 'cand' else CG.W1PlaceController


class Ctl(BASE):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _advance_inner(self, T_tcp, close_done, q):
        ok_ = self.judge.stop is None and np.isfinite(q).all()
        if ok_ and self.st.phase == 'CRUISE' and not self.blending and self.vh_seq['ph'] in ('R', 'DONE'):
            AST['n'] += 1
            for kk, v in achk(np.asarray(q, float)).items():
                AST['meas'][kk] = min(AST['meas'].get(kk, 9.9), v)
            for e in (0, 1):
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                if len(Q) >= N_MAX:
                    AST['unfinished'] += 1
                if not np.isfinite(Q).all():
                    AST['nonfinite'] += 1
                    continue
                for qq in Q:
                    for kk, v in achk(qq).items():
                        key = f'e{e}:{kk}'
                        if v < AST['worst'].get(key, 9.9):
                            AST['worst'][key], AST['where'][key] = v, (round(float(self.t), 3), self.vh_seq['ph'], round(float(q[0]), 4))
        if ok_ and self.st.phase == 'CRUISE' and not self.blending and float(q[0]) >= self.x_blend and len(GATE) < 400:   # 到漸變起點後、漸變開始前：管道條件
            GATE.append((round(float(self.t), 3), round(float(q[1]), 4), round(math.degrees(float(q[2])), 3), bool(self._unfolded()),
                         bool(self._in_tube(np.r_[max(float(q[0]), self.x_ws), np.asarray(q, float)[1:3]]))))
        super()._advance_inner(T_tcp, close_done, q)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    c.path = NP.PlannedPath(J_RUN, -0.9)
    c.s_ref = c.sig_robot = c.path.sig0
    f.q[0], f.q[1] = J_RUN[0][0], J_RUN[0][1]
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, wall0=None)
    step0 = c.step
    import time as _time

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        else:
            st['wall0'] = _time.time() - t / RTF_CAP
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=85.0)
    return c, f, r


def main():
    c, f, r = run()
    ev = r['events']
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    ph = {}
    for e in ev:
        if e['what'] in ('phase', 'place_phase'):
            ph.setdefault(e['to'], e['t'])
    t_blend = next((e['t'] for e in ev if e['what'] == 'blend_start'), None)
    vh = {e[1]: e[0] for e in r['vhigh_seq']['log']}
    th = np.array([h['t'] for h in f.hist])
    Qh = np.array([h['q'][:3] for h in f.hist])
    vm = np.r_[0.0, np.linalg.norm(np.diff(Qh[:, :2], axis=0), axis=1) / np.maximum(np.diff(th), 1e-9)]
    vm = np.convolve(vm, np.ones(10) / 10, mode='same')
    vbh = np.array([h['v_b'] for h in f.hist])
    out = dict(case=CASE, ver=VER, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               blend=t_blend, x_blend=round(float(c.x_blend), 4), vhigh=r['vhigh_seq']['log'], early_R=(r.get('approach_sched') or {}).get('early_R'),
               t_acc=next((g.get('t_acc') for g in (r.get('online') or {}).get('grasp_plans', []) if g.get('state') == 'accepted'), None), phases=ph,
               t_grasp=r.get('t_grasp'), t_lift_ok=r.get('t_lift_ok'), place_mm=round(((r.get('place') or {}).get('metrics') or {}).get('offset', float('nan')) * 1e3, 2))
    if 'INSERT' in ph and r.get('t_lift_ok') is not None:
        out['insert_to_liftok'] = round(float(r['t_lift_ok']) - ph['INSERT'], 3)
    if 'INSERT' in ph and r.get('t_grasp') is not None:
        m_ = (th >= ph['INSERT']) & (th <= float(r['t_grasp']))
        out['grasp_speed'] = dict(mean=round(float(vm[m_].mean()), 4), min=round(float(vm[m_].min()), 4), max=round(float(vm[m_].max()), 4))
    tr = vh.get('R')
    if tr is not None and t_blend is not None:
        m_ = (th >= tr) & (th <= t_blend)
        out['approach_window'] = [tr, t_blend]
        out['v_max_approach'] = round(float(vm[m_].max()), 4) if m_.any() else None
        xb = np.array([h['x'] for h in f.hist])
        k_at = np.flatnonzero((xb >= c.x_blend) & (th <= t_blend))
        out['arm_wait_s'] = round(float(t_blend - th[k_at[0]]), 3) if len(k_at) else 0.0
    t_acc = out['t_acc']
    if t_acc is not None and t_blend is not None:
        m_ = (th >= t_acc) & (th <= t_blend) & (vbh < 0.02)
        out['near_stop_s'] = round(float(m_.sum() * C0.PHYS), 3)
    out['approach_stop_n'] = AST['n']
    out['approach_min'] = {k: round(v, 4) for k, v in AST['meas'].items()}
    out['approach_stop_worst'] = {k: round(v, 5) for k, v in AST['worst'].items()}
    out['approach_stop_where'] = AST['where']
    out['approach_stop_fails'] = sorted(k for k, v in AST['worst'].items() if v < ALIM[k.split(':')[1]]) + (['nonfinite'] if AST['nonfinite'] else []) + (['unfinished'] if AST['unfinished'] else [])
    out['sched'] = None if VER != 'cand' else dict(wait=(r.get('approach_sched') or {}).get('wait'), log_tail=((r.get('approach_sched') or {}).get('log') or [])[-3:], n_log=len((r.get('approach_sched') or {}).get('log') or []))
    out['gate'] = dict(n=len(GATE), first=GATE[:1], last=GATE[-1:], y_range=[min((g[1] for g in GATE), default=None), max((g[1] for g in GATE), default=None)],
                       yaw_absmax=max((abs(g[2]) for g in GATE), default=None), n_not_unfolded=sum(1 for g in GATE if not g[3]), n_out_tube=sum(1 for g in GATE if not g[4]),
                       y_tube=list(B_.Y_TUBE), yaw_tube_deg=math.degrees(B_.YAW_TUBE))
    out['finite'] = bool(T.finite_cmds(f))
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
