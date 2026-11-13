#!/usr/bin/env python3
"""P2 執行層整合（假世界；Codex 20261113_124213 §3、20261113_130916 §5）：開門執行層 r12＋短段續接（door_motm_p2：(i) 段終剩餘代價＋末段模式）。
起點：綁定後靜止（ex4 plan_nominal 的起點；閉鏈一致）。首段＝冷啟規劃，計算時間 w0 進時間線（開門晚 w0 才開始；此起點已綁定靜止，等待期間不動）。
續接：每 T_RE 秒（模擬時間）請求一次；σ_s＝目前 σ＋L（參考時間，取密集網格）；新段起點＝執行層目前參考在 σ_s 的點（閉鏈一致）與速度
  （底盤／門角中央差分、手臂取閉鏈一致解）⇒ 接上時 C1 連續、不跳參考。單一規劃程序；結果在 t_req＋wall（模擬時間）才可用。
接管前核對：σ 還沒過 σ_s；實測構型對目前參考（底盤 ≤10 mm、偏航 ≤1°、量測門角 ≤1°，事前登錄）；不過 ⇒ 拒接（舊段走到段尾停穩）。
  不改寫、不投影實測構型。執行層提交的每個命令照樣走真管線停止掃掠（執行層本身）。
靜止重啟：參考已到盡頭且門未到目標 ⇒ 實測停穩（END 速率門檻連續 10 拍）才請求；新段從舊參考終點（閉鏈一致）起算，接管前同樣核對實測對參考。
案：ideal、plant（受控對象 a×0.8、底盤速度×0.95）、meas_bias（量測門角 +0.3°，受控對象不變）、timeout（第 3、4 次請求逾時）、deadline50（執行層 50 ms 期限）。
報告分列：任務完成（執行層 END）、安全拒絕／停止、連續性（停滯時間、拒接、逾期）、控制期限。
    python3 evaluation/test_door_p2_exec.py PLAN_DIR CASE H L OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('P2_TERM', '1'); os.environ.setdefault('P2_FINAL', '1')
import door_motm_p2 as P2
WB = P2.WB
import test_door_open_exec as T
X = T.X
plan_dir, case, H, L, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
P = np.load(os.path.join(plan_dir, 'plan_nominal.npz'))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
T_e, phb, T_HE, h = P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], float(P['h'])
GOAL, FQ = T.PHI_GOAL, T.FINGER_Q
TAKE = dict(base_m=0.010, yaw_deg=1.0, door_deg=1.0)
T_RE = P2.T_RE
x0 = np.asarray(P['Xd'][0], float).copy()
WB.set_model(phi0=phb, finger_q=FQ)
helper = WB.DoorWBOpt(T_e, np.eye(4), x0, GOAL, N=2, dt=0.2, T_HE=T_HE, project_start=False)
log = dict(case=case, H=H, L=L, take=TAKE, req=[], splices=[])

# ---- 首段（冷啟；時間線上的延遲）
o0, i0, v0_, w0 = P2.plan_seg(T_e, T_HE, x0, np.zeros(10), H, GOAL, FQ, phb)
log['first'] = dict(wall_s=round(w0, 3), why=i0['why'], feasible=v0_['feasible'], failed=v0_['failed'])
if not v0_['feasible']:
    json.dump(log, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float); sys.exit('first segment infeasible')
Xd0, h0, _ = o0.dense_reference(4)
assert abs(h0 - h) < 1e-12
g2 = dict(gravity2=True, ik_fc=True, pool=6, prep_fc=True, deadline_ms=50.0 if case == 'deadline50' else None)
c = X.DoorOpenExec(ident, Xd0, h, T_e, phb, T_HE, GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), FQ, arm_mode='closure', g2=g2)
c.p2_open = True


class FakeB(T.Fake):
    def door(self):
        phi, pH = super().door()
        return (phi + (math.radians(0.3) if case == 'meas_bias' else 0.0)), pH


F = FakeB(c, q0, a_scale=0.8 if case == 'plant' else 1.0, v_scale=0.95 if case == 'plant' else 1.0)
c.warmup(q0, F.door()[0])
st = dict(pending=None, t_next=T_RE, k=0, still=0, final_sent=False)
sam = []


def ref_state(sig):
    """執行層密集參考在 σ 的點（網格）與速度（底盤／門角中央差分，手臂閉鏈一致解）。"""
    i = int(round(sig / h))
    i = min(max(i, 0), len(c.Xd) - 1)
    x = c.Xd[i].copy()
    if i >= len(c.Xd) - 1 or sig >= c.sig_end - 1e-9:
        return x, np.zeros(10), True
    j0, j1 = max(i - 1, 0), min(i + 1, len(c.Xd) - 1)
    v = (c.Xd[j1] - c.Xd[j0]) / ((j1 - j0) * h)
    WB.set_model(phi0=phb, finger_q=FQ)
    va = helper.closure_velocity(x, v[:3], v[9])
    return x, np.r_[v[:3], va, v[9]], False


def track_err(q, phi_meas):
    xr = c.ref(c.sigma)
    return float(np.linalg.norm(q[:2] - xr[:2])), abs(math.degrees(q[2] - xr[2])), abs(math.degrees(phi_meas - xr[9]))


def hook(f):
    t = f.t
    if c.n_step % 5 == 0:
        sam.append([t, float(f.door()[0]), c.sigma, c.g, float(c.sigma >= c.sig_end - 1e-9)])
        ok_still = c.rate['arm'] <= X.END['arm_rps'] and c.rate['base'] <= X.END['base_mps'] and c.rate['yaw'] <= X.END['yaw_rps'] and c.rate['door'] <= X.END['door_rps']
        st['still'] = st['still'] + 1 if ok_still else 0
    if c.stop is not None or c.abort is not None:
        return
    pend = st['pending']
    if pend is not None and t >= pend['t_avail'] - 1e-9:
        st['pending'] = None
        rec = pend['rec']
        phi_m = float(f.door()[0])
        eb, ey, ed = track_err(f.q, phi_m)
        rec.update(t_avail=round(pend['t_avail'], 3), sigma_at_avail=round(c.sigma, 3), track_err=dict(base_mm=round(eb * 1e3, 3), yaw_deg=round(ey, 3), door_deg=round(ed, 3)))
        if not pend['feasible']:
            rec['result'] = 'infeasible'
        elif c.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
            rec['result'] = 'missed_deadline'
        elif eb > TAKE['base_m'] or ey > TAKE['yaw_deg'] or ed > TAKE['door_deg']:
            rec['result'] = 'takeover_rejected'
        else:
            sig_s = pend['sigma_s'] if not pend['rest'] else max(pend['sigma_s'], math.ceil(c.sigma / h - 1e-9) * h)
            okx = c.extend_reference(pend['Xd'], sig_s, final=pend['final'])
            rec['result'] = 'spliced' if okx else 'extend_refused'
            if okx:
                log['splices'].append(dict(k=rec['k'], t=round(t, 3), sigma_s=round(sig_s, 3), final=pend['final'], rest=pend['rest']))
    if st['pending'] is None and t >= st['t_next'] - 1e-9 and c.p2_open:
        sig_s = math.ceil((c.sigma + L) / h - 1e-9) * h
        rest = sig_s >= c.sig_end - 1e-9
        if rest and st['still'] < 10:                                    # 靜止重啟：要實測停穩
            st['t_next'] = t + 0.05
            return
        if rest:
            sig_s = c.sig_end
        xs, vs, _ = ref_state(sig_s)
        st['k'] += 1
        k = st['k']
        on, fin, w = None, False, 0.0
        try:
            ft = P2.plan_final(T_e, T_HE, xs, vs, GOAL, FQ, phb, warm=None, t_s=None)
        except ValueError:
            ft = None
        if ft is not None:
            w += ft[3]
            if ft[2]['feasible']:
                on, fin = ft, True
        if on is None:
            try:
                on = P2.plan_seg(T_e, T_HE, xs, vs, H, GOAL, FQ, phb)
            except ValueError as e:
                on = None
            if on is not None:
                w += on[3]
        if case == 'timeout' and k in (3, 4):
            w = 1e9
        rec = dict(k=k, t_req=round(t, 3), sigma_now=round(c.sigma, 3), sigma_s=round(sig_s, 3), rest=rest, final_try=ft is not None, final=fin, wall_s=round(w, 3) if w < 1e8 else 'timeout',
                   feasible=bool(on is not None and on[2]['feasible']), failed=None if on is None else on[2]['failed'])
        log['req'].append(rec)
        if w < 1e8 and on is not None:
            Xn, hn, _ = on[0].dense_reference(4)
            st['pending'] = dict(t_avail=t + w, sigma_s=sig_s, Xd=Xn, final=fin, feasible=on[2]['feasible'], rest=rest, rec=rec)
        else:
            rec['result'] = 'timeout' if w >= 1e8 else 'no_plan'
        st['t_next'] = t + T_RE


t0w = time.perf_counter()
F.run(t_end=40.0, hook=hook)
c.close_pool()
S = np.array(sam)
stall = (S[:, 4] > 0.5) & (S[:, 1] < GOAL - math.radians(0.5))
n_st, dur = 0, 0.0
for i in range(len(stall)):
    if stall[i]:
        dur += 0.05
        if i == 0 or not stall[i - 1]:
            n_st += 1
tk = np.array(c.tick_ms)
st_t = None if c.stop is None else float(c.stop['t'])
normal = np.array([m for i, m in enumerate(tk) if st_t is None or i * 0.05 < st_t - 1e-9])
res_count = {}
for r in log['req']:
    res_count[r.get('result', 'pending_at_end')] = res_count.get(r.get('result', 'pending_at_end'), 0) + 1
log['summary'] = dict(
    task=dict(done=c.done, stop=c.stop, t_open_done_s=None if not c.done else round(w0 + float(c.stop['t']), 2), first_segment_latency_s=round(w0, 3),
              phi_end_meas_deg=round(math.degrees(float(S[-1, 1])), 3)),
    safety=dict(abort=c.abort, stop_ok=c.stop_ok, min_clear_actual=c.mins['act'], min_clear_stop_sweep=c.mins['stop'], closure_stats={k: v for k, v in c.cl_stats.items() if k != 'fails'},
                after_stop_closure_pos_mm=round(F.after_stop['closure_pos'] * 1e3, 3), n_cmd_after_abort=F.n_cmd_after_abort),
    continuity=dict(n_req=len(log['req']), results=res_count, n_splice=len(log['splices']), n_stall=n_st, stall_s=round(dur, 2), splices=log['splices'][-3:], p2_log_tail=c.p2_log[-3:]),
    deadline=dict(deadline_ms=g2['deadline_ms'], n_miss=c.n_deadline_miss, normal_ticks=dict(n=int(normal.size), p50=round(float(np.median(normal)), 1), p95=round(float(np.percentile(normal, 95)), 1), max=round(float(normal.max()), 1), n_over_50=int((normal > 50).sum())) if normal.size else None),
    phi_curve=[[round(float(a), 2), round(math.degrees(float(b)), 2)] for a, b in S[::20, :2]], wall_total_s=round(time.perf_counter() - t0w, 1))
json.dump(log, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in log['summary'].items() if k != 'phi_curve'}, ensure_ascii=False, default=float)[:1800])
