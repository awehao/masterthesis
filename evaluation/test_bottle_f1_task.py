#!/usr/bin/env python3
"""F1 任務層假世界整合（Codex 20261102 F1 裁定；假受控對象、假候選，非物理，不用 Isaac、不跑 ROS 視覺節點）。遠瓶先抓：+0.15 → 遠端 (2.55, −0.58)，返回原起點，−0.15 → 近端 (2.25, −0.58)。
控制器 bottle_w1_f1_ctrl＋任務層 bottle_f1_task。持瓶與返回路徑＝離線 Smac（results/bottle/f1_offline_plan_rev.json；首末點對齊當趟）。同一個假受控對象貫穿兩輪，關節與設定點不重置。
案例：
  nominal   名目兩輪。掃掠核對（三項缺口）：① 抓持成立到離桌確認之間的持瓶停止掃掠（此段瓶仍受第一桌支撐，對第一桌桌面不判、其餘照判）；② 返回段全身與停止；③ 切換條件與連續性。
  stale     舊候選與舊完成通知在切換後到達（來源早於 t_round0）⇒ 拒收，不建立目標。之後送正常候選照常完成。
  noobs     第二輪沒有任何新觀測 ⇒ 不沿用第一輪快照，由既有規則停止，outcome＝round1_aborted。
  abort0    第一輪持瓶段禁止接觸 ⇒ 真正中止，不進第二輪。
  slipwarn  第一輪持瓶段平移滑移 6 mm（超過 5 mm 只警告）、仍正常放置保持 ⇒ 進第二輪，品質紀錄保留（grip_ok[0]＝False）。
  badsettle 校正窗 d̂ 無效 ⇒ 不放行切換。
  moving    命令與管線皆零、校正中量測仍在動（0.5 s）⇒ 累計清空重來，未切換；之後靜止才切換，d̂ 不被污染。
  movingall 命令與管線皆零但量測一直在動 ⇒ return_settle_timeout，不切換。
  deadline0 第一輪完成、返回未完成時到達總期限 ⇒ 受控停止並走尾段，不等迴圈次數用完。
  defect    兩輪共用規劃接口、沒有 PlanPort（bottle_f1_1 的實際情況）⇒ 重現 KeyError: 'start'。只用來確認假世界抓得到原缺陷。
  planlate  切換後才到的舊回覆（舊序號）⇒ 不交付；第二輪照常用自己的請求完成。
  crash     第二輪持瓶段注入例外 ⇒ 先存 crash.json 與 ctx，獨立停止實測靜止，再存逐步資料與報告；另核存檔失敗不掩蓋原例外、停止未驗證照實報。
  dyn       名目兩輪＋原已固定的移動物 f1_return_pass1（世界軌跡與任務時鐘排程不改）。障礙只經離線射線→追蹤器→控制器；真值只用於劇本與距離評估。核會遇窗全身與停止掃掠（障礙隨取樣時間推進，連桿、手指、持瓶、底盤圓盤都要 > 0），分開報 tracked／in_solver／reaction。
  cont      連續交接（Codex 20261103_133750）：第一瓶放好後走一條路徑直接接第二瓶，到換目標點換新實例不停車；場景同 lchan。cont_stale／cont_noobs／cont_planlate 是舊觀測、第二瓶沒取得、舊規劃晚到的反例。
  lchan     兩瓶＋L 型走廊＋第二桌東移（Codex 20261103_065813；無移動物，座標凍結）。真追蹤器＋射線掃描＋GMPC＋Shield；牆進掃描、地圖與全身核對。交進出走廊的手臂排程、全身對牆淨空與停止掃掠、路徑連續性、牆的動態誤判、鬆弛、Shield、期限與第一阻點。假世界無代價圖（不含 path_blocked）、無接觸與滑移物理。
  origin    名目兩輪，重啟點在原起點（對照 bottle_f1_2 的做法；其餘案例都是短返回：重啟點在接近路徑上 x＝RESTART_X）。
規劃接口與 Isaac 相同：兩輪共用同一個 SharedPlan（舊結果會留著、請求序號只增），外面包 PlanPort。
    python3 evaluation/test_bottle_f1_task.py all
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'f1_task_check.json')
CASES = ('nominal', 'dyn', 'lchan', 'cont', 'contdyn', 'cont_stale', 'cont_noobs', 'cont_planlate', 'origin', 'stale', 'noobs', 'abort0', 'slipwarn', 'badsettle', 'moving', 'movingall', 'deadline0', 'defect', 'planlate', 'crash')
if len(sys.argv) < 2 or sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=3000)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:]))
    with ThreadPoolExecutor(2) as ex:
        res = dict(ex.map(one, CASES))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, r in res.items():
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 'stop', 't_done', 'checks', 'error')})
    sys.exit(0 if all(r.get('pass') for r in res.values()) else 1)

CASE = sys.argv[1]
ROUNDS = [dict(b=(0.15, 0.42), target=(2.55, -0.58), x_look=0.15), dict(b=(-0.15, 0.42), target=(2.25, -0.58), x_look=-0.15)]
PLACE_DEV, LIM_OTHER, GATE = 0.030, 0.05, 0.60
sys.path.insert(0, HERE)
import bottle_f1_lchan_scene as LS                      # noqa: E402
CONT = CASE.startswith('cont')                          # 連續交接（不回西端、不停車）；場景同 lchan
BASE_CASE = CASE[5:] if CASE.startswith('cont_') else CASE
SCN = LS.load() if CASE == 'lchan' or CONT else None
FB_FAIL = CASE == 'lchan' and bool(os.environ.get('LCHAN_FB_FAIL'))   # 只供核對：注入後備失敗，應照原規則停止
if SCN is not None:
    ROUNDS[0]['target'], ROUNDS[1]['target'] = SCN['targets']
sys.argv = [sys.argv[0], '0.600', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = ROUNDS[0]['b']
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_f1_crash as FX                            # noqa: E402
import bottle_s1_ctrl as C2                             # noqa: E402
import bottle_f1_dyn_report as FD                       # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_f1_task as FT                             # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_f1_ctrl as F1C                         # noqa: E402
import bottle_w1_f1l_ctrl as F1L                        # noqa: E402
import bottle_w1_f1c_ctrl as F1CC                       # noqa: E402
import bottle_f1c_path as CP                            # noqa: E402
import bottle_monitor_progress as MPG                   # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
TP.DX, TP.DY = (SCN['table2_x'] if SCN is not None else 2.4), -1.0
if SCN is not None:                                     # 假世界的接觸模型（FakePlace）讀 PC.PLACE['c']：第二桌東移要同步，否則瓶子放在舊桌位之外、量不到支撐
    PC_ = sys.modules.get('bottle_place_ctrl') or __import__('bottle_place_ctrl')
    PC_.PLACE['c'] = SCN['place_c']
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', 'bottle_online5_u3_1', 's1_report.json')))['plan']['joined'], float)
PLANS = json.load(open(os.path.join(HERE, 'results', 'bottle', SCN['plan_file'].replace('.json', '_cont.json') if CONT else SCN['plan_file'] if SCN is not None else 'f1_offline_plan_rev_box.json')))   # 原障礙方塊進地圖的離線路徑（實跑靠掃描進代價圖；無方塊版的返回路徑貼著方塊走，與實跑不符）
RTF_CAP = 0.6
LOGP = os.path.join(HERE, 'results', 'bottle', f'f1_task_{CASE}_worker.log')
W = TV.T
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
WALLS = [] if SCN is None else [(c_, s_) for _, c_, s_ in SCN['walls']]
TBN += [] if SCN is None else [('wall:' + n, c_, s_) for n, c_, s_ in SCN['walls']]
TB = [(c_, s_) for _, c_, s_ in TBN]
TB_NO_T2TOP = [(c_, s_) for n, c_, s_ in TBN if n != 't2:table_top']
TB_NO_T1TOP = [(c_, s_) for n, c_, s_ in TBN if n != 't1:table_top']
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
REC = []
CUR = dict(id=0, t0=0.0, q_tuck=None)
CA = None
if CASE in ('dyn', 'contdyn'):
    CA = dict(json.load(open(os.path.join(HERE, 'experiments', '20261103_f1c_dyn_case.json' if CASE == 'contdyn' else '20261103_f1_dyn_case.json')))['case'])
    for k_ in ('start', 'end', 'meet_robot', 'robot_dir'):
        CA[k_] = tuple(CA[k_])
Z_BOT, H_BOT, R_BOT = 0.60, G.BOTTLE['h'], G.BOTTLE['r']
SECOND = (ROUNDS[1]['b'][0], ROUNDS[1]['b'][1], R_BOT)
FIRST_PLACED = (ROUNDS[0]['target'][0], ROUNDS[0]['target'][1], R_BOT + PLACE_DEV)


def others(x):
    if x['round'] == 0:
        return [SECOND, FIRST_PLACED] if x['ret'] is not None else [SECOND]
    return [FIRST_PLACED]


def d_cyl(P, L):
    return min(float(G.cyl_dist(P, o[0], o[1], Z_BOT, Z_BOT + H_BOT, o[2]).min()) for o in L)


def split(W_):
    return np.vstack([v for k, v in W_.items() if k not in G.C.FINGERS]), np.vstack([v for k, v in W_.items() if k in G.C.FINGERS])


class SharedPlan:
    """離線 Smac 路徑依請求順序取用；介面與語意同 RosCarryPlanner＋PlanRos：結果留著直到下一次請求，序號只增；poll 比對序號。只換首末點對齊當趟實際起點與目標。"""

    def __init__(self, names):
        self.names, self.i, self.info = list(names), 0, []
        self.id, self.result, self.req_id, self.pending = None, None, 0, None

    def request(self, start, goal, t):
        self.result = None
        self.req_id += 1
        self.id = self.req_id
        self.pending = (self.req_id, self.names[self.i], np.array(start[:2], float), np.array(goal, float))
        self.i += 1

    def invalidate(self):
        self.id, self.result = None, None
        self.req_id += 1

    def tick(self):
        pass

    def inject_late(self, rid):
        """舊請求的回覆晚到（PlanRos 的 stale 判定之前就寫進 result 的最壞情形）。"""
        self.result = dict(id=rid, ok=True, P=np.array(PLANS['return_far']['P_region'], float), frame='map')

    def poll(self):
        if self.pending is not None:
            rid, nm, a, b = self.pending
            self.pending = None
            if rid == self.req_id:
                P0 = np.array(PLANS[nm]['P_region'], float)
                d0, d1 = float(np.linalg.norm(P0[0] - a)), float(np.linalg.norm(P0[-1] - b))
                if max(d0, d1) > 0.07:
                    raise SystemExit(f'{nm} 離線路徑首末點差太大 {d0:.3f}／{d1:.3f} m ⇒ 拒跑')
                P = P0.copy()
                P[0], P[-1] = a, b
                self.info.append(dict(name=nm, d_start=round(d0, 4), d_goal=round(d1, 4), L=PLANS[nm]['length_m']))
                self.result = dict(id=rid, ok=True, P=P, frame='map', name=nm)
        r = self.result
        if self.id is None or r is None or r.get('id') != self.id:
            return None
        return dict(ok=True, P=r['P'], frame='map', grid=TP.empty_grid(), info=dict(planned=True, name=r.get('name')), monitor=None)


class Ctl(F1CC.W1PlaceController if CONT else (F1L.W1PlaceController if SCN is not None else F1C.W1PlaceController)):
    def __init__(self, *a, **kw):
        if CUR['q_tuck'] is not None:
            kw['q_tuck'] = CUR['q_tuck']
        super().__init__(*a, x_look=ROUNDS[CUR['id']]['x_look'], t_round0=CUR['t0'], round_id=CUR['id'], place=dict(target=ROUNDS[CUR['id']]['target'], **({} if SCN is None else dict(c=SCN['place_c']))), planner_log=LOGP, return_heading='forward', **kw)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all():
            j = self.judge
            T_b = getattr(self, 'T_b_now', None)
            T_rel, pre = None, False
            if self.t_release is None and j.t_grasp is not None:
                if j.t_lift_ok is not None and j.T_rel is not None:
                    T_rel = np.asarray(j.T_rel, float).copy()
                elif T_b is not None:                         # 抓持成立到離桌確認：用當下實測瓶對 TCP（缺口 ①）
                    T_rel, pre = np.linalg.inv(self.K.fk(q, 'link_tcp')) @ np.asarray(T_b, float), True
            REC.append(dict(round=self.round_id, t=float(self.t), ph=self.st.phase, pph=getattr(self, 'pph', None), q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(), pipe=[np.asarray(u, float).copy() for u in self.pipe],
                            u_last=np.asarray(self.u_last, float).copy(), v_b=float(self.v_b), f_cmd=copy.deepcopy(self.f_cmd), T_rel=T_rel, pre_lift=pre, ret=None if self.ret is None else self.ret['state'], cfg=self.cfg))
        super()._advance_inner(T_tcp, close_done, q)


AUDIT = []


def audit_fk(c):
    """F1C_AUDIT：σ 只能用於進度判定。FK 與手腕視向一旦拿到被改寫的 q 就記下呼叫位置。"""
    import traceback
    K_ = c.K
    fk0 = K_.fk

    def fk(q_, *a_, **k_):
        e_ = getattr(c, '_q_eff', None)
        if e_ is not None and np.asarray(q_).shape == e_.shape and np.array_equal(np.asarray(q_), e_):
            AUDIT.append(''.join(traceback.format_stack(limit=4)[:-1])[-400:])
        return fk0(q_, *a_, **k_)
    K_.fk = fk
    q0 = c._q_for_gaze

    def qg(q_):
        out_ = q0(q_)
        if getattr(c, '_q_eff', None) is not None and out_ is c._q_eff:
            AUDIT.append('gaze 拿到改寫的 q')
        return out_
    c._q_for_gaze = qg


def build_ctrl():
    orig, orig_ml = W.W1.W1PlaceController, SC.make_lidar
    W.W1.W1PlaceController = Ctl
    scene = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top'] + WALLS
    def scan_fn(t, q):
        r_ = NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z)
        if CA is not None:
            r_ = np.minimum(r_, DC.ray_cyl(q[:2], NC.scan_angles() + float(q[2]), DC.state(t, CA)[0], CA['radius'], NC.RANGE_MAX))
        return r_
    SC.make_lidar = lambda **kw: orig_ml(scan_fn=scan_fn, **dict(kw, extra_boxes=list(kw.get('extra_boxes', [])) + WALLS))
    try:
        return W.build()
    finally:
        W.W1.W1PlaceController, SC.make_lidar = orig, orig_ml


def lchan_report(rec, reps, c0, placed, wall_sw, wh, shared):
    """走廊核對：實際排程（不延後階段）、全身對牆淨空、路徑連續性、動態誤判、鬆弛、Shield。"""
    out = dict(scene=SCN['name'], widths=SCN['widths'], corridor={}, wall_actual={}, wall_stop={k_: round(v_, 4) for k_, v_ in wall_sw.items()}, wall_stop_where={k_: wh.get(k_) for k_ in wall_sw})
    segs = [('carry_r1', [x for x in rec if x['round'] == 0 and x['ret'] is None and x['pph'] is not None], reps[0]), ('return', [x for x in rec if x['round'] == 0 and x['ret'] is not None], None), ('carry_r2', [x for x in rec if x['round'] == 1 and x['pph'] is not None], reps[-1])]
    for nm, L_, rep in segs:
        if not L_:
            out['corridor'][nm] = dict(reached=False)
            continue
        Q_ = np.array([x['q'] for x in L_])
        tt_ = np.array([x['t'] for x in L_])
        ins = LS.in_corridor(Q_[:, :2])
        d_ = dict(reached=bool(ins.any()))
        if ins.any():
            k_ = np.flatnonzero(ins)
            d_.update(t_in=round(float(tt_[k_[0]]), 2), t_out=round(float(tt_[k_[-1]]), 2), dur=round(float(tt_[k_[-1]] - tt_[k_[0]]), 2))
            v_ = np.linalg.norm(np.diff(Q_[k_, :2], axis=0), axis=1) / np.maximum(np.diff(tt_[k_]), 1e-6)
            d_.update(v_mean=round(float(v_.mean()), 3), v_min=round(float(v_.min()), 3), v_max=round(float(v_.max()), 3))
            if rep is not None:
                ev = {e_['what']: e_['t'] for e_ in rep['events'] if str(e_.get('what', '')).startswith('compact_')}
                d_['compact_events'] = {a_: round(float(b_), 2) for a_, b_ in ev.items()}
                d_['in_compact'] = ev.get('compact_reached', 1e9) <= tt_[k_[0]]
                d_['raise_after_out'] = ev.get('compact_raise_start', -1e9) >= tt_[k_[-1]]
                d_['out_after_out'] = ev.get('compact_out_start', -1e9) >= tt_[k_[-1]]
                d_['margin_in_s'] = round(float(tt_[k_[0]] - ev.get('compact_reached', 1e9)), 2)
                d_['margin_raise_s'] = round(float(ev.get('compact_raise_start', -1e9) - tt_[k_[-1]]), 2)
        out['corridor'][nm] = d_
    for x in rec[::2]:
        P_n, P_f = split(G.link_world(x['q'], x['f_cmd']))
        for k_, P_ in (('nonfinger', P_n), ('finger', P_f)):
            out['wall_actual'][k_] = min(out['wall_actual'].get(k_, 9.9), min(float(G.box_dist(P_, c_, s_).min()) for c_, s_ in WALLS))
        out['wall_actual']['base_disc'] = min(out['wall_actual'].get('base_disc', 9.9), min(float(LS.d2_box(x['q'][:2], c_, s_)[0]) for c_, s_ in WALLS) - 0.39)
        if x['T_rel'] is not None:
            Tb_ = G.C.K.fk(x['q'], 'link_tcp') @ x['T_rel']
            out['wall_actual']['held'] = min(out['wall_actual'].get('held', 9.9), min(float(G.box_dist((Tb_[:3, :3] @ BPT.T).T + Tb_[:3, 3], c_, s_).min()) for c_, s_ in WALLS))
    out['wall_actual'] = {k_: round(v_, 4) for k_, v_ in out['wall_actual'].items()}
    rows_ = list(c0.compact or []) + (list(placed['c1'].compact or []) if 'c1' in placed else [])
    dyn_t = [float(e_['t']) for e_ in rows_ if FD._dyn(e_['obs'])]
    def _is_wall_dyn(e_, o_):                                  # 靠牆且不是移動圓柱本身（圓柱會經過牆端，不能算牆被誤判）
        if min(float(LS.d2_box(np.array([o_['x'], o_['y']]), c_, s_)[0]) for c_, s_ in WALLS) >= 0.6:
            return False
        if CA is None:
            return True
        oc_ = DC.state(float(e_['t']), CA)[0]
        return float(np.hypot(o_['x'] - oc_[0], o_['y'] - oc_[1])) - CA['radius'] > 0.5
    near_w = sum(1 for e_ in rows_ if any(_is_wall_dyn(e_, o_) for o_ in FD._dyn(e_['obs'])))
    eps_ = np.array([float(e_.get('eps0', 0.0)) for e_ in rows_])
    out['false_dynamic'] = dict(n_ticks_near_wall=near_w, n_ticks=len(dyn_t), n_rows=len(rows_), t_first=dyn_t[0] if dyn_t else None, t_last=dyn_t[-1] if dyn_t else None, max_items=max((len(FD._dyn(e_['obs'])) for e_ in rows_), default=0), note='場景沒有移動物，任何動態項都算誤判')
    i_e = int(np.argmax(eps_)) if len(eps_) else None
    in_c = np.array([bool(LS.in_corridor(np.asarray(e_['qp'], float)[:2])[0]) for e_ in rows_])
    t_over = [float(rows_[i_]['t']) for i_ in np.flatnonzero(eps_ > 1e-4)]
    out['slack'] = dict(eps0_max=float(eps_.max()) if len(eps_) else None, at=None if i_e is None else dict(t=round(float(rows_[i_e]['t']), 2), xy=[round(float(v_), 3) for v_ in np.asarray(rows_[i_e]['qp'], float)[:2]], in_corridor=bool(in_c[i_e]), phase=int(rows_[i_e]['phase'])),
                        n_over_1e4=len(t_over), n_over_1e4_in_corridor=int(((eps_ > 1e-4) & in_c).sum()), eps0_max_in_corridor=float(eps_[in_c].max()) if in_c.any() else None, t_over_range=[t_over[0], t_over[-1]] if t_over else None, n_ticks_corridor=int(in_c.sum()), min_h_min=float(min((float(e_['min_h']) for e_ in rows_), default=float('nan'))))
    out['shield'] = [(r_.get('nav_chain') or {}).get('shield') for r_ in reps]
    # 路徑連續性：實際底盤到當趟承接路徑（規劃路徑＋放置直線段）的橫向誤差；直線段銜接前後的命令變化
    out['tracking'] = {}
    for nm, L_, rep in segs:
        if rep is None or not L_ or nm == 'return':
            continue
        Q_ = np.array([x['q'] for x in L_ if x['pph'] in ('CARRY_NAV', 'PLACE_APPROACH')])
        if not len(Q_):
            continue
        rows_c_ = [e2_ for e2_ in (c0.compact if nm == 'carry_r1' else (placed['c1'].compact if 'c1' in placed else [])) if float(e2_['w_p']) != 0.0 and float(e2_['t']) >= L_[0]['t'] and float(e2_['t']) <= L_[-1]['t']]
        e_ = np.array([float(np.linalg.norm(np.asarray(e2_['qp'], float)[:2] - np.asarray(e2_['X_ref'], float)[0][:2, 2])) for e2_ in rows_c_]) if rows_c_ else np.zeros(1)
        v_ = np.diff(Q_[:, :2], axis=0) / 0.05
        a_ = np.linalg.norm(np.diff(v_, axis=0), axis=1) / 0.05
        Lc_ = [x for x in L_ if x['pph'] in ('CARRY_NAV', 'PLACE_APPROACH')]
        vr_ = np.array([x['v_b'] for x in Lc_])
        vm_ = np.linalg.norm(v_, axis=1)
        ic_ = LS.in_corridor(Q_[:, :2])
        prof_ = ((rep.get('place') or {}).get('carry') or {}).get('profile')
        out['tracking'][nm] = dict(duration_s=round(float(Lc_[-1]['t'] - Lc_[0]['t']), 2), ref_v=dict(mean=round(float(vr_.mean()), 3), min=round(float(vr_.min()), 3), max=round(float(vr_.max()), 3), corridor_mean=round(float(vr_[ic_].mean()), 3) if ic_.any() else None), meas_v=dict(mean=round(float(vm_.mean()), 3), min=round(float(vm_.min()), 3), max=round(float(vm_.max()), 3), corridor_mean=round(float(vm_[ic_[1:]].mean()), 3) if ic_[1:].any() else None), accept_profile=prof_, ref_err_max_m=round(float(e_.max()), 4), ref_err_mean_m=round(float(e_.mean()), 4), a_max_est_mps2=round(float(a_.max()), 3), v_max=round(float(np.linalg.norm(v_, axis=1).max()), 3), yaw_range=[round(float(Q_[:, 2].min()), 3), round(float(Q_[:, 2].max()), 3)])
    # 中止時保留該拍輸入、狀態、殘差與耗時（Codex 083722）；求解紀錄存檔，不改求解器
    out['ret_solver'] = [(r_.get('f1l') or {}).get('ret_solver') for r_ in reps]
    rd_ = os.path.join(HERE, 'runs', 'lchan_fake_' + SCN['name'])
    os.makedirs(rd_, exist_ok=True)
    C2.save_compact(c0, os.path.join(rd_, 's1_cbf_r0.npz'))
    if 'c1' in placed:
        C2.save_compact(placed['c1'], os.path.join(rd_, 's1_cbf.npz'))
    out['cbf_dir'] = rd_
    why_ = ' '.join(str((r_.get('stop') or {}).get('why')) for r_ in reps)
    if 'wb_fallback' in why_ and rows_ and float(rows_[-1]['w_p']) == 0.0:
        e_ = rows_[-1]
        from ammr_wholebody_mpc.se2 import from_xytheta as fx_
        c_ = placed.get('c1') or c0
        c_.t = float(e_['t'])
        r_ = c_.mpc_nav.solve(obstacles=list(e_['obs'] or []), X_now=fx_(*np.asarray(e_['qp'], float)[:3]), X_ref_win=e_['X_ref'], xi_ref_win=e_['xi_ref'], xi_prev=np.asarray(e_['u_prev'], float)[:3], arm=c_._ret_arm())
        out['stop_tick'] = dict(hold_fb=r_.hold_fb, logged_eps0=float(e_['eps0']), logged_resid_noslack=float(e_['resid_ns']), logged_resid_slack=float(e_['resid_s']), t=float(e_['t']), qp=np.asarray(e_['qp'], float)[:3].tolist(), logged_u=np.asarray(e_['u'], float)[:3].tolist(), re_u=np.asarray(r_.u_opt, float).tolist(), status=str(r_.status), accept=str(r_.accept_action), eps0=float(r_.eps0), resid_noslack=float(r_.cbf_resid_noslack), resid_slack=float(r_.cbf_resid_slack), solve_ms=float(r_.solve_time_s) * 1e3, min_h_m2=float(r_.min_h), n_obs=len(e_['obs'] or []), note='重解用同一輸入；Shield 列取控制器最後一幀掃描，近似')
    return out


def cont_report(f, r, R, task):
    """連續交接主表：第一瓶放好到第二瓶 INSERT 的時間與路程、靜止時間、換目標時的連續性、兩次抓取車速。"""
    H = f.hist
    tt, Q, U = np.array([h['t'] for h in H]), np.array([h['q'] for h in H]), np.array([h['u'] for h in H])
    k = np.flatnonzero(np.abs(tt * 20 - np.round(tt * 20)) < 1e-6)
    t0 = r['t_done0']
    ins = [next((e_['t'] for e_ in x_['events'] if e_.get('what') == 'phase' and e_.get('to') == 'INSERT'), None) for x_ in R]
    gr = [next((e_['t'] for e_ in x_['events'] if e_.get('what') == 'contact_first' or e_.get('what') == 'close_request'), None) for x_ in R]
    done = [next((e_['t'] for e_ in x_['events'] if e_.get('what') == 'mission_done'), None) for x_ in R]
    out = dict(t_done0=t0, t_switch=r['t_switch'], t_done=[d_ for d_ in done], t_insert=ins, outcome=r['outcome'], stop=[(x_.get('stop') or {}).get('why') for x_ in R])

    def seg(a, b):
        m = k[(tt[k] >= a) & (tt[k] <= b)]
        v = np.hypot(np.diff(Q[m, 0]), np.diff(Q[m, 1])) / 0.05
        wz = np.abs(np.arctan2(np.sin(np.diff(Q[m, 2])), np.cos(np.diff(Q[m, 2])))) / 0.05
        return dict(dur=round(float(tt[m[-1]] - tt[m[0]]), 2), L=round(float(np.hypot(np.diff(Q[m, 0]), np.diff(Q[m, 1])).sum()), 3), still_s=round(float((v < 0.01).sum() * 0.05), 2), still_not_rot_s=round(float(((v < 0.01) & (wz < 0.05)).sum() * 0.05), 2), v_min=round(float(v.min()), 3), v_mean=round(float(v.mean()), 3))
    if t0 is not None and ins[-1] is not None and r['stage'] == 1:
        out['done0_to_insert'] = seg(t0, ins[-1])
    if r['t_switch'] is not None:
        i_ = int(np.argmin(np.abs(tt - r['t_switch'])))
        j0, j1 = max(i_ - 10, 0), min(i_ + 10, len(tt) - 1)
        out['v_at_switch'] = round(float(np.hypot(*(Q[j1, :2] - Q[j0, :2])) / max(tt[j1] - tt[j0], 1e-6)), 3)
        du = np.abs(np.diff(U[max(i_ - 30, 0):min(i_ + 30, len(U))], axis=0))
        out['du_near_switch_max'] = np.round(du.max(axis=0), 5).tolist()
        out['q_at_switch'] = np.round(Q[i_, :3], 3).tolist()
        out['ret_info'] = (R[0].get('f1c') or {}).get('ret_info')
    if len(R) > 1:
        out['events2'] = [(round(e_['t'], 2), e_.get('what'), e_.get('ph') or e_.get('to') or '') for e_ in R[-1]['events'] if e_.get('what') in ('rolling_adopt', 'vhigh_seq', 'online_obs_first', 'relwin_shift', 'blend_start', 'phase', 'stop')][:20]
    return out


def run_crash(e, task, f):
    """與 Isaac 相同的收尾順序；另核存檔失敗與停止未驗證。"""
    import tempfile
    d = tempfile.mkdtemp(prefix='f1crash_')
    c = FX.record(d, e, dict(t=float(f.t), stage=task.stage, phase=task.st.phase, pph=task.pph))
    first = json.load(open(os.path.join(d, 'crash.json')))
    u0 = np.asarray(f.hist[-1]['u'], float)
    s_hold = np.asarray(f.hist[-1]['s'], float)

    def step_fn(u):
        c_, s_ = np.cos(f.q[2]), np.sin(f.q[2])
        f.q[:3] += 0.01 * np.array([c_ * u[0] - s_ * u[1], s_ * u[0] + c_ * u[1], u[2]])
        f.q[3:] = f.q[3:] + f.alpha * (s_hold - f.q[3:] + W.T.D_TRUE)
    c['stop'] = FX.independent_stop(step_fn, lambda: f.q.copy(), u0, F1C.B.A_BASE, 0.01)
    FX.guarded(c, 'report', lambda: FX.atomic_json(os.path.join(d, 's1_report_crash.json'), task.c.report()))
    FX.guarded(c, 'steps', lambda: np.savez_compressed(os.path.join(d, 's1_steps.npz'), rows=np.array([h['q'] for h in f.hist], float)))
    FX.guarded(c, 'boom', lambda: (_ for _ in ()).throw(OSError('注入存檔失敗')))
    FX.atomic_json(os.path.join(d, 'crash.json'), c)
    final = json.load(open(os.path.join(d, 'crash.json')))
    moving = FX.independent_stop(lambda u: f.q.__setitem__(0, f.q[0] + 0.001), lambda: f.q.copy(), np.zeros(3), F1C.B.A_BASE, 0.01, t_tail=0.2)   # 一直在動 ⇒ ok False
    noq = FX.independent_stop(lambda u: None, lambda: np.full(9, np.nan), np.zeros(3), F1C.B.A_BASE, 0.01, t_tail=0.2)                          # 量不到 ⇒ None
    chk = {'例外先落盤且含時刻輪次相位': first['type'] == 'RuntimeError' and first['stage'] == 1 and first['phase'] == 'CARRY' and first['pph'] == 'CARRY_NAV' and first['stop'] is None,
           '獨立停止實測靜止': final['stop']['ok'] is True, '報告與逐步資料寫出': os.path.exists(os.path.join(d, 's1_report_crash.json')) and os.path.exists(os.path.join(d, 's1_steps.npz')),
           '存檔失敗不掩蓋原例外': final['type'] == 'RuntimeError' and final['save_errors'][0]['step'] == 'boom', '仍在動不算停止有效': moving['ok'] is False, '量不到是未驗證': noq['ok'] is None}
    return dict(case=CASE, **{'pass': all(chk.values())}, outcome='crash', stop=final['stop'], t_done=None, checks=chk, grip_ok=None, t=final['t'])


def main():
    if FB_FAIL:
        from ammr_wholebody_mpc import gmpc_wb as GW_
        GW_.HOLD_GI = lambda *a_, **k_: (None, None, dict(ok=False, why='injected', steps=0))
    c0, f = build_ctrl()
    c0.path = NP.PlannedPath(J_RUN, -0.9)
    c0.s_ref = c0.sig_robot = c0.path.sig0
    PLAN_SEQ = ['carry_far', 'transit_far' if CONT else ('return_far' if CASE == 'origin' else 'return_short'), 'carry_near']
    shared = SharedPlan(PLAN_SEQ)
    pp0 = NP.PlannedPath(J_RUN, -0.9)
    sig_r, xy_r = (pp0.sig0, J_RUN[0].tolist()) if CASE == 'origin' else FT.restart_point(pp0)
    port = shared if CASE == 'defect' else FT.PlanPort(shared)
    if CASE == 'defect':                                     # 下面沒走到 KeyError 就判未過（main 結尾）
        shared.invalidate_real, shared.invalidate = shared.invalidate, (lambda: None)     # 重現 bottle_f1_1：切換與取用後都不清
    else:
        port.bind(c0)
    c0.carry_planner = port
    c0.compact = [] if CASE in ('dyn', 'lchan') or CONT else None
    f.q[0], f.q[1] = J_RUN[0][0], J_RUN[0][1]
    placed = {}

    def make_round(i, s_now, d, t):
        CUR.update(id=i, t0=t, q_tuck=np.asarray(s_now, float).copy())
        SC.X0 = float(f.q[0])                                  # 建構起點＝實際位置，不是原起點
        c, _ = build_ctrl()
        c.path = NP.PlannedPath(J_RUN, -0.9)
        c.s_ref = c.sig_robot = c.path.project(f.q[:2], sig_r)
        c.carry_planner = port
        c.compact = [] if CASE in ('dyn', 'lchan') or CONT else None
        placed['c1'] = c
        if CASE != 'defect':
            port.bind(c)
        c.obs.d = np.clip(d, -c.obs.max, c.obs.max)
        c.cfg.arm_model.bias = c.bias0 + np.asarray(c.cfg.arm_model.alpha, float) * c.obs.d
        if CONT:                                               # 同 Isaac dynsr38 的接線：路徑與監看都換成前一實例規劃好的那一條，σ 由它的進度換算
            prev_ = task.c
            pn_ = NP.PlannedPath(prev_.ret['J'], CP.X_LANE)
            c.path = pn_
            c.s_ref = c.sig_robot = float(pn_.sig0 + float(prev_.ret['sig']))
            mon_ = NP.PathMonitor(pn_)
            mon_.on_grid(TP.empty_grid(), c.sig_robot)
            mprog_ = MPG.MonitorProgress(pn_, C2.B.Y_TUBE)
            mprog_.sig = float(c.sig_robot)
            placed.update(mon=mon_, mprog=mprog_, sig_start=float(c.sig_robot), path_new=pn_)
            if os.environ.get('F1C_AUDIT'):
                audit_fk(c)
        placed['first'] = np.array(f.Tb, float).copy()
        S0B.BX, S0B.BY = ROUNDS[i]['b']
        f.Tb, f.grasped, f.released, f.Trel = S0B.bottle_T(), False, False, None
        return c
    ckpt = {}
    task = (FT.F1TaskC if CONT else FT.F1Task)(c0, CP.G_TRANSIT if CONT else xy_r, make_round, on_checkpoint=lambda kind, r_: ckpt.update({kind: json.loads(json.dumps(dict(mission_done=r_.get('mission_done'), round=r_['f1']['round']), default=float))}))
    f.c = task
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, wall0=None, stale_done=False, inj=False, guard=None, late_done=False)
    import time as _time

    def cand(c, t, xy, seq, t_last):
        c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(seq), snap_t_last=t_last, snap_cx=float(xy[0]), snap_cy=float(xy[1]), snap_z_support=0.60, snap_ik=2.0, snap_local=2.0, snap_dev_mm=0.0, snap_consistent=1.0))

    def fault(t, f_):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        else:
            st['wall0'] = _time.time() - t / RTF_CAP
        c = task.c
        ov = {}
        f_.q[2] = F1C.wrap(float(f_.q[2]))                     # 偏航量測與 Isaac 一樣落在 (−π, π]
        if CONT and 'mprog' in placed:
            placed['mprog'].update(np.asarray(f_.q, float), float(t))
        if BASE_CASE == 'planlate' and task.stage == 1 and not st['late_done']:      # 切換後第一拍：舊回覆才到
            st['late_done'] = True
            shared.inject_late(shared.req_id - 1)
            st['late_poll'] = port.poll()
            st['late_poll_inner'] = shared.poll()
        if CASE == 'crash' and task.stage == 1 and c.pph == 'CARRY_NAV':
            raise RuntimeError('注入例外：第二輪持瓶段')
        if task.stage == 1 and BASE_CASE == 'stale' and not st['stale_done'] and t >= task.t_switch + 2.0:     # 第一輪的候選與完成通知這時才到
            st['stale_done'] = True
            cand(c, t, ROUNDS[0]['b'], 999, task.t_switch - 1.0)
            st['stale_vfix'] = c.vfix is not None
            st['stale_gp'] = c.gp is not None
            st['stale_rej'] = c.f1_rej['old_round']
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
            allow = not (BASE_CASE == 'noobs' and task.stage == 1)
            if allow and c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + 3.6 - 1e-9:
                xy = np.array(ROUNDS[task.stage]['b']) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                cand(c, t, st['snap'], st['seq'], t - 0.05)
        if task.stage == 0 and c.ret is not None and c.ret['state'] == 'READY' and st['guard'] is None:      # 切換條件的反例（同一拍內改回）
            g = {}
            u0_, p0_, d0_ = c.u_last.copy(), [u_.copy() for u_ in c.pipe], c.d_settle
            c.u_last = u0_ + 1e-3
            g['cmd_not_zero'] = c.switch_ready()[1]
            c.u_last = u0_
            c.pipe[0] = p0_[0] + 1e-3
            g['pipe_not_zero'] = c.switch_ready()[1]
            c.pipe[0] = p0_[0]
            c.d_settle = np.full(6, np.nan)
            g['d_nan'] = c.switch_ready()[1]
            c.d_settle = d0_
            s0_ = c.still_now
            c.still_now = False
            g['not_still'] = c.switch_ready()[1]
            c.still_now = s0_
            st['guard'] = g
        if CASE == 'abort0' and task.stage == 0 and c.pph == 'CARRY_NAV':
            ov['forbidden'] = {k: (1.0 if k == 'robot_table' else 0.0) for k in W.T.REQ}
        if CASE == 'slipwarn' and task.stage == 0 and c.pph == 'CARRY_NAV' and not st['inj'] and f_.Trel is not None:
            st['inj'] = True
            f_.Trel = f_.Trel.copy()
            f_.Trel[0, 3] += 0.006
        if CASE == 'badsettle' and task.stage == 0 and c.ret is not None and c.ret['state'] == 'SETTLE':
            c.obs.max = np.zeros(6)
        if CASE in ('moving', 'movingall') and task.stage == 0 and c.ret is not None and c.ret['state'] == 'SETTLE' and c.ret['t_drained'] is not None:   # 命令與管線全零，量測還在動
            if 't_mv' not in st:
                st['t_mv'] = t
            if CASE == 'movingall' or t - st['t_mv'] <= 0.5:
                f_.q[0] += 0.0005                             # 0.05 m/s（高於 park_v）
        task.check_deadline(t, 60.0 if CASE == 'deadline0' else (240.0 if SCN is not None else 170.0))
        return ov
    crash = None
    try:
        f.run(245.0 if SCN is not None else 175.0, fault, tail=2.6)
    except KeyError as e:
        if CASE != 'defect':
            raise
        print(json.dumps(dict(case=CASE, **{'pass': str(e) == "'start'" and task.stage == 1}, outcome='重現原缺陷', stop=None, t_done=None, checks={'重現 KeyError start': str(e) == "'start'", '發生在第二輪持瓶段': task.stage == 1}, t=float(f.t), phase=task.st.phase, grip_ok=None), ensure_ascii=False, default=float))
        return 0
    except RuntimeError as e:
        if CASE != 'crash':
            raise
        crash = run_crash(e, task, f)
    if crash is not None:
        print(json.dumps(crash, ensure_ascii=False, default=float))
        return 0
    r = task.report()
    R = r['rounds']
    chk = {}
    out = dict(case=CASE, outcome=r['outcome'], stage=r['stage'], t_done0=r['t_done0'], t_switch=r['t_switch'], t_done=None, switch=r['switch'], grip_ok=r['grip_ok'], stop=[(x['stop'] or {}).get('why') for x in R],
               rounds=[dict(round=x['f1']['round'], done=x['mission_done'], slip_warn=(x.get('slip_warn') or {}).get('warn'), slip_max=x['slip_max_mm_deg'], rej=x['f1']['rej'], place=x['place'].get('metrics'), ret=x['f1']['ret'], phases={e['to']: e['t'] for e in x['events'] if e['what'] in ('phase', 'place_phase')},
                            events=[(e['t'], e['what'], e.get('to')) for e in x['events'] if e['what'] in ('return_request', 'return_state', 'mission_done', 'stop', 'relwin_shift', 'online_obs_first')], solve=x['solve_ms'], n_over_50ms=x['n_over_50ms'], n_fallback=x['n_fallback'], plans=[g.get('state') for g in (x.get('online') or {}).get('grasp_plans', [])]) for x in R],
               finite=bool(W.finite_cmds(f)), guard=st['guard'])
    if CASE == 'lchan' and not any(x['ret'] is not None for x in REC):      # 第一輪沒走到返回：交第一阻點，不算掃掠
        Lr = [x for x in REC if x['round'] == 0]
        Qr = np.array([x['q'] for x in Lr])
        Rw = {n_: float(min(float(LS.d2_box(Qr[-1, :2], c_, s_)[0]) for m_, c_, s_ in SCN['walls'] if m_ == n_)) for n_, _, _ in SCN['walls']}
        rows0 = list(c0.compact or [])
        out.update(first_blocker=dict(stop=R[0].get('stop'), t_last=float(Lr[-1]['t']), ph=Lr[-1]['ph'], pph=Lr[-1]['pph'], q_last=Qr[-1, :3].tolist(), wall_center_dist=Rw, n_dyn_ticks=sum(1 for e_ in rows0 if FD._dyn(e_['obs'])), events_tail=R[0]['events'][-14:], shield=(R[0].get('nav_chain') or {}).get('shield'), eps0_max=max((float(e_['eps0']) for e_ in rows0), default=None), info=shared.info), checks={'兩輪完成': False}, **{'pass': False})
        print(json.dumps(out, ensure_ascii=False, default=float))
        return 0
    if CASE == 'lchan' or CONT:                              # 掃描要算很久：先留下結果與中止原因，後面報告出事也不會丟
        print('#EARLY ' + json.dumps(dict(outcome=r['outcome'], stage=r['stage'], stop=out['stop'], t_switch=r['t_switch'], t_done0=r['t_done0'], t_last=float(REC[-1]['t']), last=dict(ph=REC[-1]['ph'], pph=REC[-1]['pph'], ret=REC[-1]['ret'], q=REC[-1]['q'][:3].tolist()), events_tail=[e_ for e_ in (R[-1].get('events') or [])[-10:]], ret=R[0].get('f1', {}).get('ret')), ensure_ascii=False, default=float), flush=True)
    if CASE == 'lchan' and os.environ.get('LCHAN_DIAG'):     # 只讀診斷：不算掃描；存求解紀錄、把中止前幾拍用同一輸入重解一次（掃描用當拍位姿重新射線，年齡近似）
        C2.save_compact(c0, os.environ['LCHAN_DIAG'])
        rows_ = list(c0.compact)
        i_stop = max((i_ for i_, e_ in enumerate(rows_) if float(e_['w_p']) == 0.0), default=None)
        diag = []
        for e_ in rows_[max(i_stop - 3, 0):i_stop + 1]:
            c0.t = float(e_['t'])
            c0.scan_last = (float(e_['t']), np.array(c0.scan_fn(float(e_['t']), np.asarray(e_['qp'], float)), float)) if False else c0.scan_last
            Xn_ = TP.from_xytheta(*np.asarray(e_['qp'], float)[:3]) if hasattr(TP, 'from_xytheta') else __import__('ammr_wholebody_mpc.se2', fromlist=['from_xytheta']).from_xytheta(*np.asarray(e_['qp'], float)[:3])
            r_ = c0.mpc_nav.solve(obstacles=list(e_['obs'] or []), X_now=Xn_, X_ref_win=e_['X_ref'], xi_ref_win=e_['xi_ref'], xi_prev=np.asarray(e_['u_prev'], float)[:3], arm=dict(mode='hold'))
            p_ = np.asarray(e_['qp'], float)[:2]
            ds_ = sorted(round(float(np.hypot(o_['x'] - p_[0], o_['y'] - p_[1])), 3) for o_ in (e_['obs'] or []))
            diag.append(dict(t=round(float(e_['t']), 2), xy=[round(float(v_), 3) for v_ in p_], yaw=round(float(e_['qp'][2]), 3), logged_u=np.round(e_['u'][:3], 4).tolist(), u_prev=np.round(e_['u_prev'][:3], 4).tolist(), xi0=np.round(e_['xi_ref'][0], 4).tolist(), ref_err=[round(float(v_), 4) for v_ in (np.asarray(e_['X_ref'][0][:2, 2]) - p_)], re_u=np.round(r_.u_opt, 4).tolist(), re_status=str(r_.status), re_accept=str(r_.accept_action), re_eps0=float(r_.eps0), re_active=int(r_.cbf_active), re_min_h=float(r_.min_h), n_obs=len(e_['obs'] or []), n_dyn=len(FD._dyn(e_['obs'])), obs_dist=ds_[:8]))
        print('#DIAG ' + json.dumps(diag, ensure_ascii=False, default=float), flush=True)
        print('#EPS ' + json.dumps([dict(t=round(float(e_['t']), 2), eps0=float(e_['eps0']), xy=[round(float(v_), 3) for v_ in np.asarray(e_['qp'], float)[:2]], w_p=float(e_['w_p']), n_dyn=len(FD._dyn(e_['obs']))) for e_ in rows_ if float(e_['eps0']) > 1e-4], ensure_ascii=False), flush=True)
        print('#DYN ' + json.dumps([dict(t=round(float(e_['t']), 2), xy=[round(float(v_), 3) for v_ in np.asarray(e_['qp'], float)[:2]], d=[[round(float(o_['x']), 2), round(float(o_['y']), 2), round(float(o_.get('vx', 0.0)), 2), round(float(o_.get('vy', 0.0)), 2)] for o_ in FD._dyn(e_['obs'])]) for e_ in rows_ if FD._dyn(e_['obs'])][::3], ensure_ascii=False), flush=True)
        print('#Q ' + json.dumps(dict(cpath=None if c0.cpath is None else np.round(c0.cpath.P[::20], 3).tolist()), ensure_ascii=False), flush=True)
        return 0
    if CONT and 'mprog' in placed:
        out['monitor'] = dict(sig_start=placed['sig_start'], mprog=placed['mprog'].report(), why=placed['mon'].why, n_check=placed['mon'].n_check, path_L=placed['path_new'].L)
    if os.environ.get('F1C_AUDIT'):
        print('#AUDIT ' + json.dumps(dict(n=len(AUDIT), first=AUDIT[:3]), ensure_ascii=False), flush=True)
    if CASE in ('cont', 'contdyn') or BASE_CASE in ('stale', 'noobs', 'planlate') and CONT:
        out['cont'] = cont_report(f, r, R, task)
        print('#CONT ' + json.dumps(out['cont'], ensure_ascii=False, default=float), flush=True)
    if CONT:                                                  # 軌跡留檔：移動物案例由它事前產生，之後不再依成敗調整
        rd0_ = os.path.join(HERE, 'runs', 'lchan_fake_' + SCN['name'])
        os.makedirs(rd0_, exist_ok=True)
        np.savez_compressed(os.path.join(rd0_, f'traj_{CASE}.npz'), t=np.array([h['t'] for h in f.hist]), q=np.array([h['q'] for h in f.hist]), ret=np.array([[e_['t'], e_['to']] for e_ in R[0]['events'] if e_.get('what') == 'return_state'], dtype=object), t_switch=r['t_switch'], t_done0=r['t_done0'])
    if CONT and os.environ.get('CONT_QUICK'):                 # 只要流程與主表，不算掃描
        return 0
    if CASE in ('nominal', 'origin', 'dyn', 'lchan', 'cont', 'contdyn'):
        H = f.hist
        tt, Q = np.array([h['t'] for h in H]), np.array([h['q'] for h in H])
        out['t_done'] = float(tt[-1])
        act, sw, sst, wh, nonfin, unfin = {}, {}, {}, {}, 0, 0
        wall_sw = {}
        dt_s = C21.SAMPLE_EVERY * 0.01

        def upd(D, k, v, w):
            if v < D.get(k, 9.9):
                D[k], wh[k] = v, w
        for x in REC:
            L = others(x)
            W_ = G.link_world(x['q'], x['f_cmd'])
            P_n, P_f = split(W_)
            tag = (x['round'], round(x['t'], 2), x['pph'] or x['ph'], x['ret'])
            upd(act, 'nonfinger', d_cyl(P_n, L), tag)
            upd(act, 'finger', d_cyl(P_f, L), tag)
            near = d_cyl(np.vstack([P_n, P_f]), L) < GATE
            x['near'] = near
            x['wall_near'] = bool(WALLS) and min(float(LS.d2_box(x['q'][:2], c_, s_)[0]) for c_, s_ in WALLS) < 1.2
            if x['T_rel'] is not None:
                Tb = G.C.K.fk(x['q'], 'link_tcp') @ x['T_rel']
                upd(act, 'held', d_cyl((Tb[:3, :3] @ BPT.T).T + Tb[:3, 3], L), tag)
        idx = [i for i, x in enumerate(REC) if x['near'] or x['pre_lift'] or (x['pph'] in PLACING and x['ret'] is None) or (x['ret'] in ('NAV', 'ARRIVE', 'ROT0', 'ROT1') and i % 5 == 0) or (x['wall_near'] and i % 5 == 0)]
        sel = set()
        for i in idx:
            sel.update(range(max(i - 2, 0), min(i + 3, len(REC))))
        for i in sorted(sel):
            x = REC[i]
            L = others(x)
            for e in (0, 1):
                Qs = C21.stop_traj(x['q'], x['s'], x['pipe'], x['u_last'], x['cfg'], e)
                if len(Qs) >= 2000 // C21.SAMPLE_EVERY + 1:
                    unfin += 1
                if not np.isfinite(Qs).all():
                    nonfin += 1
                    continue
                for j in range(0, len(Qs), 2):
                    qq = Qs[j]
                    W_ = G.link_world(qq, x['f_cmd'])
                    P_n, P_f = split(W_)
                    tag = (x['round'], round(x['t'], 2), x['pph'] or x['ph'], x['ret'], round(j * dt_s, 2))
                    pfx = 'pre_lift:' if x['pre_lift'] else ('return:' if x['ret'] in ('NAV', 'ARRIVE', 'ROT0', 'ROT1') else 'other:')
                    upd(sw, pfx + 'nonfinger', d_cyl(P_n, L), tag)
                    upd(sw, pfx + 'finger', d_cyl(P_f, L), tag)
                    P_b = None
                    if x['T_rel'] is not None:
                        Tb = G.C.K.fk(qq, 'link_tcp') @ x['T_rel']
                        P_b = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
                        upd(sw, pfx + 'held', d_cyl(P_b, L), tag)
                    placing = (x['pph'] or '') in PLACING
                    if WALLS:
                        upd(wall_sw, pfx + 'nonfinger', min(float(G.box_dist(P_n, c_, s_).min()) for c_, s_ in WALLS), tag)
                        upd(wall_sw, pfx + 'finger', min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in WALLS), tag)
                        upd(wall_sw, pfx + 'base_disc', min(float(LS.d2_box(qq[:2], c_, s_)[0]) for c_, s_ in WALLS) - 0.39, tag)
                        if P_b is not None:
                            upd(wall_sw, pfx + 'held', min(float(G.box_dist(P_b, c_, s_).min()) for c_, s_ in WALLS), tag)
                    upd(sst, pfx + 'nonfinger_table', min(float(G.box_dist(P_n, c_, s_).min()) for c_, s_ in TB), tag)
                    upd(sst, pfx + 'finger_table', min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB), tag)
                    upd(sst, pfx + 'self', float(G.C.self_clear(qq, x['f_cmd'])[0]), tag)
                    upd(sst, pfx + 'margin', BB.margin(qq[3:]), tag)
                    if P_b is not None:
                        bx = TB_NO_T1TOP if x['pre_lift'] else (TB_NO_T2TOP if placing else TB)
                        upd(sst, pfx + 'held_bottle_table', min(float(G.box_dist(P_b, c_, s_).min()) for c_, s_ in bx), tag)
        fails = sorted(k for k, v in act.items() if v < LIM_OTHER) + sorted(k for k, v in sw.items() if v < LIM_OTHER)
        sfails = sorted(k for k, v in sst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
        rt = [x for x in REC if x['ret'] in ('ROT0', 'NAV', 'ROT1', 'ARRIVE', 'SETTLE', 'READY')]
        rq = np.array([x['q'] for x in rt])
        arm_move = float(np.abs(rq[:, 3:] - rq[0, 3:]).max())
        spd = np.linalg.norm(np.diff(rq[:, :2], axis=0), axis=1) / 0.05
        boxes_base = [(c_, s_) for n, c_, s_ in TBN if 'leg' in n or 'wall:' in n] + [(SC.OBST['c'], SC.OBST['sz'])]
        Pb = np.c_[rq[:, :2], np.full(len(rq), 0.2)]
        base_c = min(float(np.hypot(np.maximum(np.abs(Pb[:, 0] - c_[0]) - s_[0] / 2, 0), np.maximum(np.abs(Pb[:, 1] - c_[1]) - s_[1] / 2, 0)).min()) for c_, s_ in boxes_base) - 0.39
        i_sw = next((k for k, h in enumerate(H) if task.t_switch is not None and h['t'] >= task.t_switch), None)
        Hs = np.array([h['s'] for h in H])
        ds_sw = float(np.abs(Hs[i_sw + 1] - Hs[i_sw - 1]).max()) if i_sw else None
        j_q = float(np.abs(np.diff(Q[:, 3:], axis=0)).max())
        ret = R[0]['f1']['ret']
        dyaw = np.abs(np.arctan2(np.sin(np.diff(rq[:, 2])), np.cos(np.diff(rq[:, 2])))) / 0.05
        tst = {}
        for x in rt:
            tst.setdefault(x['ret'], [x['t'], x['t']])[1] = x['t']
        out.update(actual=act, stop=dict(other=sw, static=sst), stop_where=wh, fails=fails, static_fails=sfails, n_stop_ticks=len(sel), return_seg=dict(arm_move_rad=arm_move, v_max=float(spd.max()), w_max=float(dyaw.max()), t_state={k_: [round(v_[0], 2), round(v_[1], 2)] for k_, v_ in tst.items()}, yaw_end=float(rq[-1, 2]), pos_end=rq[-1, :2].tolist(), base_clear_disc_m=round(base_c, 4), t_nav=[x['t'] for x in rt][0], t_end=[x['t'] for x in rt][-1], L=ret['info']['L'], plans=[shared.info]),
                   switch_cont=dict(s_step_rad=ds_sw, q_step_max_rad=j_q, first_placed=None if 'first' not in placed else placed['first'][:3, 3].tolist(), second_placed=np.array(f.Tb)[:3, 3].tolist()), stop_why=out['stop'] if isinstance(out['stop'], list) else None)
        if CA is not None:
            tH = np.array([h['t'] for h in H])
            QH = np.array([h['q'] for h in H])
            out['dyn'] = dict(case=CA, actual=FD.distances(tH, QH, CA, DC.state))

            def d_ob(P, t_):
                oc_ = DC.state(t_, CA)[0]
                P = P[P[:, 2] <= CA['height']]
                return float((np.hypot(P[:, 0] - oc_[0], P[:, 1] - oc_[1]) - CA['radius']).min()) if len(P) else 9.9
            nearob = [i_ for i_, x_ in enumerate(REC) if float(np.hypot(*(x_['q'][:2] - DC.state(x_['t'], CA)[0]))) - CA['radius'] - 0.39 < 1.5]
            sel_d = set()
            for i_ in nearob:
                sel_d.update(range(max(i_ - 5, 0), min(i_ + 6, len(REC))))
            wd, whd = {}, {}
            for i_ in sorted(sel_d):
                x_ = REC[i_]
                W_ = G.link_world(x_['q'], x_['f_cmd'])
                P_n, P_f = split(W_)
                for nm_, v_ in (('act:nonfinger', d_ob(P_n, x_['t'])), ('act:finger', d_ob(P_f, x_['t']))):
                    if v_ < wd.get(nm_, 9.9):
                        wd[nm_], whd[nm_] = v_, (round(x_['t'], 2), x_['ret'] or x_['pph'] or x_['ph'])
                for e_ in (0, 1):
                    Qs_ = C21.stop_traj(x_['q'], x_['s'], x_['pipe'], x_['u_last'], x_['cfg'], e_)
                    for j_ in range(0, len(Qs_), 2):
                        tj_ = x_['t'] + j_ * C21.SAMPLE_EVERY * 0.01
                        qq_ = Qs_[j_]
                        W_ = G.link_world(qq_, x_['f_cmd'])
                        P_n, P_f = split(W_)
                        oc_ = DC.state(tj_, CA)[0]
                        vals_ = [(f'e{e_}:nonfinger', d_ob(P_n, tj_)), (f'e{e_}:finger', d_ob(P_f, tj_)), (f'e{e_}:base_disc', float(np.hypot(*(qq_[:2] - oc_))) - CA['radius'] - 0.39)]
                        if x_['T_rel'] is not None:
                            Tb_ = G.C.K.fk(qq_, 'link_tcp') @ x_['T_rel']
                            vals_.append((f'e{e_}:held', d_ob((Tb_[:3, :3] @ BPT.T).T + Tb_[:3, 3], tj_)))
                        for nm_, v_ in vals_:
                            if v_ < wd.get(nm_, 9.9):
                                wd[nm_], whd[nm_] = v_, (round(x_['t'], 2), x_['ret'] or x_['pph'] or x_['ph'], round(j_ * C21.SAMPLE_EVERY * 0.01, 2))
            out['dyn'].update(sweep={k_: round(v_, 4) for k_, v_ in wd.items()}, sweep_where=whd, n_ticks=len(sel_d))
            nav_ev = {e_['to']: e_['t'] for e_ in R[0]['events'] if e_['what'] == 'return_state'}
            nav_w = (nav_ev.get('NAV', 0.0), nav_ev.get('ROT1', r['t_switch'] if CONT else 1e9))
            rows_ = list(c0.compact) + (list(placed['c1'].compact) if 'c1' in placed else [])
            out['dyn']['interaction'] = FD.analyse(rows_, c0.mpc_nav, CA['t_start'], nav_w, arm=(c0._ret_arm() if CONT else None))
            # 實跑走 save_compact→load_rows：混合手臂主動與返回 hold 兩種紀錄列必須能存能讀，存讀後的統計要與記憶體版相同
            rows_rt_ = []
            for c_rt_ in (c0, placed.get('c1')):
                if c_rt_ is not None:
                    with tempfile.TemporaryDirectory() as td_:
                        p_rt_ = os.path.join(td_, 's1_cbf.npz')
                        C2.save_compact(c_rt_, p_rt_)
                        rows_rt_ += FD.load_rows(p_rt_)
            chk['求解紀錄存讀一致'] = FD.analyse(rows_rt_, c0.mpc_nav, CA['t_start'], nav_w, arm=(c0._ret_arm() if CONT else None)) == out['dyn']['interaction']
            out['dyn']['nav_window'] = nav_w
            k_nav = np.flatnonzero((tH >= nav_w[0]) & (tH <= nav_w[1]) & (np.abs(tH * 20 - np.round(tH * 20)) < 1e-6))
            v_ = np.hypot(np.diff(QH[k_nav, 0]), np.diff(QH[k_nav, 1])) / 0.05
            w_ = np.abs(np.arctan2(np.sin(np.diff(QH[k_nav, 2])), np.cos(np.diff(QH[k_nav, 2])))) / 0.05
            near_ = np.array([float(np.hypot(*(QH[k_, :2] - DC.state(tH[k_], CA)[0]))) - CA['radius'] for k_ in k_nav[1:]]) < 1.0
            out['dyn']['reaction_motion'] = dict(v_min_near=round(float(v_[near_].min()), 4) if near_.any() else None, v_mean_near=round(float(v_[near_].mean()), 4) if near_.any() else None, w_max_near=round(float(w_[near_].max()), 4) if near_.any() else None)
            chk['移動物全身與停止掃掠 > 0'] = all(v_ > 0.0 for v_ in wd.values()) and FD.distances(tH, QH, CA, DC.state)['base_disc'] > 0.0
        if CASE == 'lchan' or CONT:
            out['lchan'] = lchan_report(REC, R, c0, placed, wall_sw, wh, shared)
            chk['牆無動態誤判'] = out['lchan']['false_dynamic']['n_ticks_near_wall'] == 0
            chk['對牆全身與停止掃掠 > 0'] = all(v_ > 0.0 for v_ in out['lchan']['wall_actual'].values()) and all(v_ > 0.0 for v_ in out['lchan']['wall_stop'].values())
            chk['期限內完成'] = bool(out.get('t_done') is not None and out['t_done'] < 240.0)
        chk['全程淨空'] = not fails
        Q2 = np.array([x['q'] for x in REC if x['round'] == 1])
        out['round2_start'] = None if (not len(Q2) or CONT) else dict(sig_r=round(float(sig_r), 4), xy_r=[round(float(v), 4) for v in xy_r], q_start=Q2[0, :3].tolist(), err_to_path_mm=round(float(np.min(np.linalg.norm(pp0.xy(np.arange(pp0.sig0, pp0.x_lane, 0.005)) - Q2[0, :2], axis=1))) * 1e3, 2), events=[(round(e[0], 2), e[1]) for e in out['rounds'][1]['events'] if e[1] in ('online_obs_first', 'relwin_shift')], vhigh=[(round(e_['t'], 2), e_.get('ph'), round(e_.get('x', 0.0), 3)) for e_ in R[1]['events'] if e_['what'] == 'vhigh_seq'], plan=[{k_: g_.get(k_) for k_ in ('t_req', 't_acc', 'x_acc', 'x_latest')} for g_ in (R[1].get('online') or {}).get('grasp_plans', [])], t_u_minus_switch=None)
        chk['第二輪未拿到舊規劃'] = port.n_blocked >= 1 and [x['name'] for x in shared.info] == PLAN_SEQ
        chk['檢查點落盤'] = ckpt.get('round0_done', {}).get('mission_done') is True and ckpt.get('switch', {}).get('round') == 0
        chk['靜態判準'] = not sfails
        chk['兩輪完成'] = r['outcome'] == 'complete'
        chk['返回段手臂未動'] = arm_move < 0.002
        if CONT:
            sw_ = r['switch'] or {}
            chk['換目標連續：設定點、上一命令、在途管線都承接'] = bool(sw_.get('new_s_equal') and sw_.get('pipe_equal') and sw_.get('u_prev_equal') and sw_.get('u_last_equal') and sw_.get('rolling'))
            chk['換目標時沒有停車'] = (out.get('cont') or {}).get('v_at_switch', 0.0) > 0.05
        else:
            chk['切換新實例承接設定點'] = bool(r['switch'] and r['switch']['new_s_equal'] and r['switch']['pipe_zero'] and r['switch']['u_prev_zero'])
            chk['反例擋下'] = bool(st['guard'] and 'cmd_not_zero' in st['guard']['cmd_not_zero'] and 'cmd_not_zero' in st['guard']['pipe_not_zero'] and 'd_invalid' in st['guard']['d_nan'] and 'not_still' in st['guard']['not_still'])
    elif BASE_CASE == 'stale':
        chk['舊候選被拒'] = st.get('stale_rej', 0) >= 1 and st.get('stale_vfix') is False and st.get('stale_gp') is False
        chk['之後照常完成'] = r['outcome'] == 'complete'
    elif BASE_CASE == 'noobs':
        chk['第二輪中止'] = r['outcome'] == 'round1_aborted'
        chk['未建立目標'] = R[-1]['f1']['rej']['old_round'] == 0 and not (R[-1].get('online') or {}).get('grasp_plans')
        chk['未抓持的輪次品質是未成立'] = r['grip_ok'] == [True, None]
    elif CASE == 'abort0':
        chk['第一輪中止不進第二輪'] = r['outcome'] == 'round0_aborted' and r['stage'] == 0 and r['t_switch'] is None
        chk['未確認離桌的輪次品質不是合格'] = r['grip_ok'] == [None]
    elif CASE == 'slipwarn':
        chk['警告後進第二輪'] = r['stage'] == 1 and r['outcome'] == 'complete'
        chk['品質紀錄保留'] = r['grip_ok'][0] is False
    elif CASE == 'badsettle':
        chk['校正無效不切換'] = r['outcome'] == 'round0_aborted' and r['t_switch'] is None and 'return_settle_invalid' in str(out['stop'])
    elif BASE_CASE == 'planlate':
        chk['舊回覆不交付'] = st.get('late_poll') is None and st.get('late_poll_inner') is None
        chk['第二輪照常完成'] = r['outcome'] == 'complete'
    elif CASE == 'moving':
        ret0 = R[0]['f1']['ret']
        d_ = (r['switch'] or {}).get('d') or [float('nan')] * 6
        chk['校正累計清空重來'] = ret0['n_reset'] >= 1
        chk['靜止後才切換且完成'] = r['outcome'] == 'complete'
        chk['d̂ 未被污染'] = bool(np.allclose(d_, np.asarray(W.T.D_TRUE, float), atol=1e-6))
    elif CASE == 'movingall':
        chk['一直在動不切換'] = r['outcome'] == 'round0_aborted' and r['t_switch'] is None and 'return_settle_timeout' in str(out['stop'])
    elif CASE == 'deadline0':
        H_ = f.hist
        chk['返回中到期受控停止'] = r['outcome'] == 'round0_aborted' and r['t_switch'] is None and 'sim_limit' in str(out['stop'])
        chk['走完停止尾段'] = float(H_[-1]['t']) >= 60.0 + 2.5 and float(H_[-1]['t']) < 66.0
        chk['尾段底盤已停'] = bool(np.allclose(H_[-1]['u'], 0.0, atol=1e-6))
    if CASE in ('defect', 'crash'):
        chk['走到注入點'] = False                              # 正常結束代表沒走到注入點，不能算通過
    chk['命令有限'] = out['finite']
    out['checks'] = chk
    out['pass'] = bool(all(chk.values()))
    out['note'] = '假受控對象、假候選；不含 ROS 視覺節點、接觸與滑移物理；取樣核對，非全流程保證；另一瓶不進 QP'
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
