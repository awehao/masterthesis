#!/usr/bin/env python3
"""P2 第一包：短段可行參考＋安全續接（Codex 20261113_120619 §5）。離線，規劃層。
每段＝door_wbopt r8 segment 模式（N＝H/dt 個節點；B 樣條末端重複控制點 ⇒ 段尾減速到靜止、閉鏈與各限制整段照非線性驗收 ⇒ 每段自帶機構相容停止）。段終不要求 85°。
續接：每 T_RE 秒發一次規劃請求，新段起點＝目前執行段在 t_s＝t_req＋L 的狀態（位置、速度；手臂投影回閉鏈、手臂速度取閉鏈一致解）＋登錄的偏差；
  單一規劃程序、請求依序處理；**計算時間換成模擬時間**（請求在 max(t_req, 上一個完成) 開始、wall 秒後才可用，不暫停）。
  新段只在「可用時刻 ≤ t_s」且完整驗收可行時於 t_s 接管；否則繼續舊段（必要時走到段尾停穩＝停滯，記錄），之後的請求從當時狀態（可能已靜止）續接。
  例外（靜止重啟）：t_s 時舊段已停穩 ⇒ 狀態到可用時刻都不變，新段（起點＝該靜止狀態、v0＝0）可在可用時刻接管（只是晚，不是放寬驗收）。
  已閉爪／已綁定時不自動解綁、不繞行重來（此層沒有這些動作）。
量：每段 wall（首段冷啟、之後）、期限達成、驗收可行、停滯次數與時長、門角進展（到 84.5°／85° 的時間）、續接點位置／速度跳變、執行過的段的最小淨空。
    python3 evaluation/door_motm_p2.py measure READY_DIR OUT.json         —— 量短段求解時間（登錄 L 用）
    python3 evaluation/door_motm_p2.py run READY_DIR H L CASE OUT.json     —— 續接迴圈"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
DT = 0.2
T_RE = 0.4
MAX_ITER = 45


def load_start(rd, tag):
    src = open(os.path.join(HERE, 'door_motm_reuse_check.py')).read()
    argv = sys.argv
    sys.argv = ['x', rd]
    ns = dict(__file__=os.path.join(HERE, 'door_motm_reuse_check.py'), __name__='rc_head')
    exec(compile(src[:src.index("Zn = np.load(")], 'reuse_head', 'exec'), ns)
    sys.argv = argv
    Z = np.load(os.path.join(rd, f'{tag}_ready.npz'))
    T_e, x0, v0, T_HE = ns['start_of'](Z)
    return ns, Z, T_e, x0, v0, T_HE


class Seg:
    """一段規劃結果＋其 B 樣條（控制點 P＝[x₋₁, X_0…X_N, X_N]）。t0＝接管的絕對時刻。"""
    def __init__(self, o, t0):
        self.o, self.t0 = o, float(t0)
        self.P = np.vstack([o.Xm1, o.X, o.X[-1]])
        self.T = (o.N + 1) * o.dt

    def state(self, t_abs):
        t = min(max(t_abs - self.t0, 0.0), self.T - 1e-9)
        k = min(int(t / self.o.dt), self.o.N)
        u = t / self.o.dt - k
        a, b, c = self.P[k], self.P[k + 1], self.P[k + 2]
        x = 0.5 * (1 - u) ** 2 * a + (-u * u + u + 0.5) * b + 0.5 * u * u * c
        v = ((u - 1) * a + (1 - 2 * u) * b + u * c) / self.o.dt
        if t_abs - self.t0 >= self.T - 1e-9:
            v = np.zeros_like(v)
        xp = self.o.project(x, iters=30)
        va = self.o.closure_velocity(xp, v[:3], v[9])
        return xp, np.r_[v[:3], va, v[9]]

    def at_rest(self, t_abs):
        return t_abs - self.t0 >= self.T - 1e-9


TERM = os.environ.get('P2_TERM', '0') == '1'                        # (i) 段終剩餘代價（Codex 20261113_124213）


def term_nodes(x0, goal, N):
    """M＝整段規劃節點數估計 N_full（同整段規劃的 N 公式：ceil((T_ramp＋2)/dt)，T_ramp 以段起點門角與底盤到鉸鏈距離算）減去本段 N；不小於 0。"""
    b = x0[:3]
    Tr = 1.875 * max(goal - x0[9], 0.0) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
    return max(0, int(math.ceil((Tr + 2.0) / DT)) - N)


def plan_seg(T_e, T_HE, x0, v0, H, goal, fq, phi_model, warm=None, t_s=None):
    WB.set_model(phi0=phi_model, finger_q=fq)
    N = int(round(H / DT))
    o = WB.DoorWBOpt(T_e, np.eye(4), x0, goal, N=N, dt=DT, T_HE=T_HE, project_start=False, v0=v0, grad='analytic', segment=True, term_nodes=term_nodes(x0, goal, N) if TERM else 0)
    if warm is None:
        b = x0[:3]
        Tr = 1.875 * (goal - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
        o.init_corotate(max(Tr, H), v0_tau=0.4 if np.any(v0) else None)
    else:                                                            # 熱啟：前一段在新節點對應時刻 t_s＋(k＋½)dt 的值（超出段尾取靜止終點），手臂投影
        for k in range(1, N + 1):
            o.X[k] = o.project(warm.state(t_s + (k + 0.5) * DT)[0], iters=30)
        o.init = dict(kind='warm_shift')
    t0 = time.perf_counter()
    info = o.solve(max_iter=MAX_ITER)
    v = o.verify()
    return o, info, v, time.perf_counter() - t0


def measure(rd, out):
    ns, Z, T_e, x0, v0, T_HE = load_start(rd, 'nominal_0.05')
    res = {}
    for H in (1.0, 2.0):
        o, info, v, w = plan_seg(T_e, T_HE, x0, v0, H, ns['GOAL'], ns['FQ'], float(Z['phi']))
        r = dict(cold=dict(wall_s=round(w, 2), why=info['why'], iters=info['iters'], feasible=v['feasible'], failed=v['failed'], phi_end_deg=v['phi_end_deg'], dphi_deg=round(math.degrees(o.X[-1, 9] - x0[9]), 3)))
        s = Seg(o, 0.0)
        warm = []
        for k in range(1, 6):
            t_s = k * T_RE
            xs, vs = s.state(t_s)
            o2, i2, v2, w2 = plan_seg(T_e, T_HE, xs, vs, H, ns['GOAL'], ns['FQ'], float(Z['phi']), warm=s, t_s=t_s)
            warm.append(dict(t_s=t_s, wall_s=round(w2, 2), why=i2['why'], iters=i2['iters'], feasible=v2['feasible'], failed=v2['failed'], dphi_deg=round(math.degrees(o2.X[-1, 9] - xs[9]), 3)))
            if v2['feasible']:
                s = Seg(o2, t_s)
        r['warm_chain'] = warm
        res[f'H{H:g}'] = r
        print(H, json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


L_REG = {1.0: 0.4, 2.0: 1.2}                                         # 事前登錄（量測後、跑續接前）：熱啟 wall 最大 0.31／0.85 s 的約 1.3–1.4 倍
CASES = dict(ideal={}, dev=dict(dev_base_mm=3.0, dev_phi_deg=0.2), est=dict(phi_jump_deg=0.5, phi_jump_t=2.0), timeout=dict(timeout_req=(3, 4)), fail=dict(fail_req=(3,)),
             slow=dict(wall_scale=1.5), v008=dict(start='nominal_0.08'))


def run(rd, H, L, case, out):
    C = CASES[case]
    ns, Z, T_e, x0, v0, T_HE = load_start(rd, C.get('start', 'nominal_0.05'))
    goal, fq, phm = ns['GOAL'], ns['FQ'], float(Z['phi'])
    rng = np.random.default_rng(20261113)
    ws = float(C.get('wall_scale', 1.0))
    # 首段：READY 實際狀態冷啟（首段延遲照實記；此層把首段視為 t＝0 已可用，見報告）
    o, info, v, w = plan_seg(T_e, T_HE, x0, v0, H, goal, fq, phm)
    log = dict(first=dict(wall_s=round(w, 3), why=info['why'], iters=info['iters'], feasible=v['feasible'], failed=v['failed']), req=[], switches=[])
    if not v['feasible']:
        json.dump(log, open(out, 'w'), ensure_ascii=False, indent=1, default=float); print('first infeasible'); return
    cur = Seg(o, 0.0)
    segs = [dict(t0=0.0, min_clear=v['min_clear_m'], v=v)]
    t_free, t_last_req, k_req = 0.0, -1e9, 0
    t_end = 25.0
    ph_jump_done = False
    n_good, t_done = 0, None
    t = 0.0
    while t < t_end:
        # 發請求：規劃程序空閒且距上次請求 ≥ T_RE
        if t >= t_free - 1e-9 and t - t_last_req >= T_RE - 1e-9:
            k_req += 1
            t_req, t_s = t, t + L
            xs, vs = cur.state(t_s)
            if C.get('dev_base_mm'):
                d = rng.normal(size=2); d /= np.linalg.norm(d)
                xs = xs.copy(); xs[:2] += d * C['dev_base_mm'] * 1e-3
            if C.get('dev_phi_deg'):
                xs = xs.copy(); xs[9] += math.radians(C['dev_phi_deg']) * rng.choice([-1.0, 1.0])
            jump_now = bool(C.get('phi_jump_deg') and t_s >= C['phi_jump_t'] and not ph_jump_done)
            if jump_now:
                xs = xs.copy(); xs[9] += math.radians(C['phi_jump_deg'])   # 量測門角一次性比參考多 0.5°（新段從量測值起算；之後的段已含此偏移，不重複加）
                ph_jump_done = True
            if C.get('dev_base_mm') or C.get('dev_phi_deg') or jump_now:
                WB.set_model(phi0=phm, finger_q=fq)
                tmp = WB.DoorWBOpt(T_e, np.eye(4), np.r_[xs], goal, N=2, dt=DT, T_HE=T_HE, project_start=True)
                xs = tmp.x0
                vs = np.r_[vs[:3], tmp.closure_velocity(xs, vs[:3], vs[9]), vs[9]]
            on = None
            try:
                o2, i2, v2, w2 = plan_seg(T_e, T_HE, xs, vs, H, goal, fq, phm, warm=cur, t_s=t_s)
                on = (o2, i2, v2)
            except ValueError as e:
                w2, i2, v2 = 0.0, dict(why='start_invalid:' + str(e)[:80], iters=0), dict(feasible=False, failed=['start'])
            w2 *= ws
            if k_req in C.get('timeout_req', ()):
                w2 = 1e9                                             # 注入：求解逾時（結果永不在期限前可用）
            if k_req in C.get('fail_req', ()):
                v2 = dict(v2, feasible=False, failed=list(v2.get('failed', [])) + ['injected_fail'])
            t_avail = max(t_req, t_free) + w2
            ok_deadline = t_avail <= t_s + 1e-9
            rest_restart = bool((not ok_deadline) and w2 < 1e8 and cur.at_rest(t_s))
            if rest_restart:
                t_s = t_avail                                          # 靜止重啟：舊段已停穩，晚接管不改變起點狀態
            rec = dict(k=k_req, t_req=round(t_req, 3), t_s=round(t_s, 3), wall_s=round(w2, 3) if w2 < 1e8 else 'timeout', why=i2['why'], iters=i2['iters'], feasible=v2['feasible'], failed=v2.get('failed'),
                       deadline_met=bool(ok_deadline), rest_restart=rest_restart, cur_at_rest_at_ts=bool(cur.at_rest(t_s)))
            if (ok_deadline or rest_restart) and v2['feasible'] and on is not None:
                xo, vo = cur.state(t_s)
                new = Seg(on[0], t_s)
                xn, vn = new.state(t_s)
                log['switches'].append(dict(k=k_req, t_s=round(t_s, 3), jump_pos=round(float(np.abs(xn[:3] - xo[:3]).max()) * 1e3, 3), jump_phi_deg=round(math.degrees(abs(xn[9] - xo[9])), 4),
                                            jump_vel=round(float(np.abs(vn[[0, 1, 2, 9]] - vo[[0, 1, 2, 9]]).max()), 5), dphi_seg_deg=round(math.degrees(on[0].X[-1, 9] - xs[9]), 3)))
                pend = (t_s, new, on[2]['min_clear_m'], on[2])
            else:
                pend = None
            rec['switched'] = pend is not None
            log['req'].append(rec)
            t_free = t_avail if w2 < 1e8 else t + T_RE                  # 逾時：此層把該次求解在下個週期放棄（不阻塞後續請求）
            t_last_req = t
            # 推進到 t_s（期間依舊段）並換段
            if pend is not None:
                samples_until(log, cur, t, pend[0])
                t = pend[0]
                cur = pend[1]
                segs.append(dict(t0=round(t, 3), min_clear=pend[2], v=pend[3]))
                continue
        nt = t + 0.05
        samples_until(log, cur, t, nt)
        t = nt
        xr, vr = cur.state(t)
        # 完成判準＝開門執行層 END（門角 ≥ 目標−0.5°、手臂／底盤／偏航／門角速度低於門檻、連續 10 拍）；在參考上判
        good = xr[9] >= goal - math.radians(0.5) and float(np.abs(vr[3:9]).max()) <= 0.02 and float(np.hypot(vr[0], vr[1])) <= 0.005 and abs(vr[2]) <= 0.01 and abs(vr[9]) <= 0.01
        n_good = n_good + 1 if good else 0
        if n_good >= 10 and t_done is None:
            t_done = t
            break
    P = np.array(log.pop('samples'))
    ph = P[:, 1]
    rest = P[:, 2] > 0.5
    stall = rest & (ph < goal - math.radians(0.5))
    n_st, dur = 0, 0.0
    for i in range(len(stall)):
        if stall[i]:
            dur += 0.05
            if i == 0 or not stall[i - 1]:
                n_st += 1
    t84 = next((float(P[i, 0]) for i in range(len(P)) if ph[i] >= goal - math.radians(0.5)), None)
    mc = {k: min(s_['min_clear'][k] for s_ in segs) for k in segs[0]['min_clear']}
    walls = [r['wall_s'] for r in log['req'] if isinstance(r['wall_s'], float)]
    log['summary'] = dict(H=H, L=L, case=case, term=TERM, ref_len_s=round((int(round(H / DT)) + 1) * DT, 2), t_done_s=None if t_done is None else round(t_done, 2), t_phi_84p5_s=t84, t_end_s=round(float(P[-1, 0]), 2), phi_end_deg=round(math.degrees(ph[-1]), 3), n_req=len(log['req']), n_switch=len(log['switches']),
                          n_deadline_miss=sum(1 for r in log['req'] if not r['deadline_met']), n_infeasible=sum(1 for r in log['req'] if not r['feasible']),
                          wall_p50=round(float(np.median(walls)), 3) if walls else None, wall_max=round(float(max(walls)), 3) if walls else None,
                          n_stall=n_st, stall_s=round(dur, 2), min_clear_executed_segments=mc,
                          margins_executed_segments=dict(wheel_max=max(s_['v']['wheel_max'] for s_ in segs), wheel_lim=round(WB.W_LIM, 4),
                                                         body_acc_max=[max(s_['v']['body_acc_max'][i] for s_ in segs) for i in range(3)], body_acc_lim=WB.A_BODY.tolist(),
                                                         setpoint_rate_max=max(s_['v']['setpoint_rate_needed_max'] for s_ in segs), setpoint_rate_lim=WB.V_ARM,
                                                         sigma_min=min(s_['v']['sigma_min'] for s_ in segs), joint_margin_min=min(s_['v']['joint_margin_min_rad'] for s_ in segs),
                                                         phi_rate_max=max(s_['v']['phi_rate_max'] for s_ in segs), phi_rate_lim=WB.PHI_RATE),
                          jump_pos_mm_max=max((s_['jump_pos'] for s_ in log['switches']), default=None), jump_vel_max=max((s_['jump_vel'] for s_ in log['switches']), default=None),
                          phi_curve=[[round(float(a), 2), round(math.degrees(float(b)), 2)] for a, b in P[::10, :2]])
    json.dump(log, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v_ for k, v_ in log['summary'].items() if k != 'phi_curve'}, ensure_ascii=False, default=float))


def samples_until(log, seg, ta, tb):
    log.setdefault('samples', [])
    t = ta
    while t < tb - 1e-9:
        x, _ = seg.state(t)
        log['samples'].append([t, float(x[9]), float(seg.at_rest(t))])
        t += 0.05


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        measure(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'run':
        run(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])
