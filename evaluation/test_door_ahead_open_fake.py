#!/usr/bin/env python3
"""（非阻塞執行版，抓持＋交接＋開門）test_door_ahead_fake.py 延伸：同一工作程序在 READY 後交接給開門執行層（door_ahead_open），P2 續段由背景規劃程序並行算；
物理以固定 RTF 推進，不為任何計算暫停。到開門完成後的停止尾段執行完（或晚到後尾段執行完）為止。
  AHEAD_LATE=o20 ⇒ 開門段第 20 拍注入延遲（AHEAD_LATE_S，預設 0.08 s）；預熱（丟棄用開門執行層空跑一拍）在起跑前，對應遠距接近期間的背景工作。
原說明：MotM 整條鏈假世界名目一趟（Howard 20261113 13:49：收斂成可看的一趟；這是 Isaac 起跑前的放行核對，不是測試矩陣）。
1 移動抓持（MovingGraspCtrl r8：pool 6、fastkin、期限 50 ms）：底盤自可抓窗口起點以 v＝0.05 m/s 沿門面 −x 通過、不停。
2 移動接近中（進入可抓窗口前 T_LEAD＝1.0 s；此假世界沒有接近段的受控對象，只用那時刻可取得的影格）以當時估計與「預測 READY 狀態」（底盤照 v 前推到排程的 READY 時刻、手臂為該底盤對抓取目標的 IK、閉鏈一致速度）
  請求 P2 首段（door_motm_p2.plan_seg，H＝2、段終剩餘代價）；計算時間進模擬時間線；閉爪閘門＝首段可用且可行，未就緒不閉爪。
3 READY 那拍首：以首段建開門執行層 r13（移動承接：設定點、在途管線、上一命令、上游靜態偏置估計），複本試跑第一拍
  （候選、閉鏈驗收、停止掃掠、預測停止閉鏈、備援）通過才 accept_handover（當拍版本）並理想綁定；否則不接受（抓持端逾時接觸停法）。
4 開門：P2 續接（每 0.4 s 請求；σ_s＝σ＋1.2；暫存＋首次影響那拍試跑；末段模式），期限 50 ms。
受控對象：底盤照命令積分；手臂一階（α）＋靜態偏置 d(q)；門綁定前靜止、綁定後由 TCP 反算（理想閉鏈）。
記：每拍時間序列（控制器、底盤速度、門角、TCP 對真值抓取位姿）、事件、各段計算時間（模擬時間線上的等待）、完成／停止／中止。
    python3 evaluation/test_door_motm_chain_fake.py OUT_PREFIX"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from door_ahead_case import *                                       # noqa: F401,F403  共用案例設定（同 ROS 乾跑 D1）
import door_ahead_case as CASE
import door_ahead_exec as AE
OUT = sys.argv[1]
runner = AE.AheadRunner(gc, G.S, NPH, deadline_ms=DL_MS, phi0=phi_b, late_ticks=LATE, door=door)
if CPU_LAYOUT:
    CPU_PIN['ahead_worker'] = CPUL.pin(runner.proc.pid, [0, 8])     # 控制器工作程序：整個實體核心 0（邏輯 0＋8；背景執行緒可用同核另一邏輯 CPU）
    CPU_PIN['readback'] = dict(main=sorted(os.sched_getaffinity(0)), worker=sorted(os.sched_getaffinity(runner.proc.pid)), pool=[sorted(os.sched_getaffinity(p_.pid)) for p_ in SHARED.procs], cstop=[sorted(os.sched_getaffinity(p_.pid)) for p_ in CSW.procs], planner=sorted(os.sched_getaffinity(BG.proc.pid)))
init = runner.start(q, 0.0, phi=phi_b)
if os.environ.get('AHEAD_RX_THREAD', '1') == '1':
    runner.start_rx()                                                # 同 ROS 乾跑執行器：獨立接收執行緒（結果一到即記收到牆鐘；主迴圈忙於物理步時不延後）
ev.append(dict(t=0.0, what='ahead_init', tail_src=init['tail_src'], phase=init['phase'], mode=runner.mode))
exec_abort, meas_min = None, {k_: 9.0 for k_ in ('env', 'base', 'self', 'shell_bar')}
SS = X.STOP_STILL                                                    # 停穩判準（同開門執行層：手臂、底盤、偏航、門角速率連續低於門檻 hold_s；逾時 timeout_s）
hold = None                                                          # 尾段播完後：保持最後設定點、底盤零命令，繼續推受控對象並以停穩判準確認
t, n, rows, overrun = 0.0, 0, [], 0
t_ready, stopped_at, bindinfo, t_bind = None, None, None, None
phi_now = phi_b
w0 = time.perf_counter()


def phi_bound(q9):
    pH = (K.fk(q9, 'link_tcp') @ bindinfo['T_EH'])[:3, 3]
    d = math.atan2(pH[1], pH[0]) - bindinfo['ang_bind']
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return bindinfo['phi_bind'] - d


T_SIM = float(os.environ.get('AHEAD_TSIM', '30'))
Tt = T_at(T_true, 0.0)                                               # 抓取目標（常數；只算一次）
while t < T_SIM:
    if n % NPH == 0:
        rec = runner.tick_boundary(q, t + NPH * PHYS, n, phi=phi_now)
        if rec.get('commit') and rec.get('bind') and bindinfo is None:   # 交接段已提交 ⇒ 理想綁定（假世界：門角由 TCP 反算）
            bindinfo, t_bind = runner.last['bind'], round(t, 3)
            ev.append(dict(t=t_bind, what='bind_committed', n=n))
        if rec.get('commit') is False and not any(e['what'] == 'tail_started' for e in ev):
            ev.append(dict(t=round(t, 3), what='tail_started', rec={k_: rec.get(k_) for k_ in ('why', 'roundtrip_ms', 'compute_ms', 'steps_since_send', 'wall_since_send_ms', 'mode', 'ctrl_done', 'stop_mode')}, last=None if runner.last is None else {k_: runner.last.get(k_) for k_ in ('mode', 'done', 'stop_mode')}))
    runner.poll()
    o = runner.output() if hold is None else None
    if o is None:
        if hold is None:
            if runner.mode == 'start_rejected':
                ev.append(dict(t=round(t, 3), what='start_rejected')); break
            stopped_at = round(t, 3)
            ev.append(dict(t=round(t, 3), what='tail_done'))
            hold = dict(t0=t, arm=np.asarray(last_arm, float).copy(), still=0, q_prev=q.copy(), phi_prev=phi_now, verified=None)
        o = dict(arm_target=hold['arm'], base_twist=np.zeros(3), stopped=True, ready=False)
    last_arm = np.asarray(o['arm_target'], float).copy()
    tw = o['base_twist']
    cs, sn = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
    q[3:] = q[3:] + al * dead(o['arm_target'] - q[3:] + dq(q))
    phi_now = phi_bound(q) if bindinfo is not None else phi_b
    if not (np.isfinite(q).all() and math.isfinite(phi_now)):        # 執行器端實測監看（Codex 20261117_004610）：安全前提失效 ⇒ 立即工程中止，不降級成播尾段；尾段與保持期間照常
        ev.append(dict(t=round(t, 3), what='engineering_abort', why='state_nonfinite')); exec_abort = 'state_nonfinite'; break
    if n % NPH == 0:
        c_ = P2.WB.clear_door(np.r_[q, phi_now], T_true)
        bad_ = [k_ for k_ in ('env', 'base', 'self', 'shell_bar') if c_[k_] < P2.WB.THR[k_]]
        for k_ in meas_min:
            meas_min[k_] = min(meas_min[k_], float(c_[k_]))
        if bad_:
            ev.append(dict(t=round(t, 3), what='engineering_abort', why='measured_clearance', bad=bad_, at={k_: round(float(c_[k_]), 4) for k_ in bad_})); exec_abort = 'measured_clearance'; break
    rows.append([t, *q, phi_now, float(np.linalg.norm(K.fk(q, 'link_tcp')[:3, 3] - Tt[:3, 3])), float(o['stopped']), runner.mode == 'nominal', bindinfo is not None])
    if hold is not None:
        qa_, ph_ = hold['q_prev'], hold['phi_prev']
        st_ok = (float(np.abs(q[3:] - qa_[3:]).max()) / PHYS < SS['arm_rps'] and float(np.hypot(*(q[:2] - qa_[:2]))) / PHYS < SS['base_mps'] and abs(q[2] - qa_[2]) / PHYS < SS['yaw_rps'] and abs(phi_now - ph_) / PHYS < SS['door_rps'])
        hold['still'] = hold['still'] + 1 if st_ok else 0
        hold['q_prev'], hold['phi_prev'] = q.copy(), phi_now
        if hold['still'] * PHYS >= SS['hold_s']:
            hold['verified'] = True
            ev.append(dict(t=round(t, 3), what='settled_verified', since_tail_done_s=round(t - hold['t0'], 3), criteria=SS)); break
        if t - hold['t0'] >= SS['timeout_s']:
            hold['verified'] = False
            ev.append(dict(t=round(t, 3), what='not_settled_after_tail', criteria=SS)); break
    if o.get('ready') and t_ready is None and runner.mode == 'nominal':
        t_ready = round(t, 3)
        ev.append(dict(t=t_ready, what='first_ready', ready_ver=o.get('ready_ver')))
    n += 1
    t = n * PHYS
    lag = (w0 + t / RTF) - time.perf_counter()
    if lag < -0.001:
        overrun += 1
    while lag > 0:                                                 # 定速推進期間持續收結果直到步界（不阻塞物理；舊版最後 2 ms 不收 ⇒ 收到時刻最多晚 2 ms）
        runner.poll()
        time.sleep(min(lag, 0.0005))
        lag = (w0 + t / RTF) - time.perf_counter()
R_ = np.array(rows, float)
LG = [r for r in runner.log if 'compute_ms' in r]


def st_(a):
    a = np.asarray(a, float)
    return None if not len(a) else dict(n=int(len(a)), p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1))


by = {}
for m_ in ('grasp', 'open'):
    L_ = [r for r in LG if r.get('mode') == m_]
    by[m_] = dict(compute_ms=st_([r['compute_ms'] for r in L_]), roundtrip_ms=st_([r['roundtrip_ms'] for r in L_]), n_commit=sum(1 for r in L_ if r.get('commit')), tail_src=sorted({str(r.get('tail_src')) for r in L_}),
                  tail_mismatch_max=max([r['tail_mismatch'] for r in L_ if r.get('tail_mismatch') is not None] or [None]) if L_ else None)
bs = np.r_[0.0, np.linalg.norm(np.diff(R_[:, 1:3], axis=0), axis=1) / PHYS] if len(R_) > 1 else np.zeros(1)
i_b = int(np.argmax(R_[:, 14] > 0.5)) if len(R_) and (R_[:, 14] > 0.5).any() else None
p2ev = [e for e in runner.events if e.get('what') in ('p2_request', 'p2_result', 'staged')]
ctrl_final = runner.get(['stop', 'abort', 'done', 'stop_mode', 'stop_ok', 'n_deadline_miss', 'sigma', 'sig_end']) if getattr(runner, '_rx', None) is None else dict(note='接收執行緒模式不讀工作程序狀態；以 last_commit 與事件為準')
res = dict(exec_abort=exec_abort, meas_clear_min={k_: round(v_, 4) for k_, v_ in meas_min.items()}, rx_thread=getattr(runner, '_rx', None) is not None, tail_cert=door.tail_cert, stream=door.stream, stream_chunk=X.STREAM_CHUNK, pool_n=SHARED.n, settled_after_tail=None if hold is None else hold['verified'], cpu_layout=CPU_PIN if CPU_LAYOUT else None, deadline_ms=DL_MS, diagnostic_only=DL_MS != 50.0, rtf_target=RTF, wall_s=round(time.perf_counter() - w0, 2), sim_s=round(t, 3), physics_overrun_steps=overrun, n_commit=runner.n_commit, late=runner.late, t_ready=t_ready, t_bind=t_bind, stopped_at=stopped_at,
           by_mode=by, handover=[e for e in runner.events if e.get('what') == 'handover_trial_is_tick'],
           p2=dict(n_request=sum(1 for e in p2ev if e['what'] == 'p2_request'), results=[{k_: e.get(k_) for k_ in ('k', 'result', 'why', 'seam', 'wall_s', 'feasible', 'final', 'n')} for e in p2ev if e['what'] == 'p2_result'],
                   staged=[{k_: e.get(k_) for k_ in ('ver', 'ok', 'why', 'trial_is_tick', 'wall_ms', 'n')} for e in p2ev if e['what'] == 'staged']),
           last_commit=None if runner.last is None else {k_: (v_ if k_ != 'bind' else (v_ is not None)) for k_, v_ in runner.last.items()},
           phi_end_deg=round(math.degrees(float(R_[-1, 10])), 3) if len(R_) else None,
           base_speed=dict(min_start_to_bind=None if i_b is None else round(float(bs[1:i_b + 3].min()), 4), at_bind=None if i_b is None else round(float(bs[i_b]), 4), max=round(float(bs.max()), 4), end=round(float(bs[-1]), 5)),
           arm_rate_end=round(float(np.abs(R_[-1, 4:10] - R_[-2, 4:10]).max()) / PHYS, 5) if len(R_) > 1 else None,
           events=ev, ctrl_final={k_: v_ for k_, v_ in (ctrl_final or {}).items()}, first_segment=dict(ok=first['ok'], t_avail=first['t_avail']))
json.dump(res, open(OUT + '.json', 'w'), ensure_ascii=False, indent=1, default=str)
np.savez(OUT + '_rows.npz', rows=R_, cols=np.array(['t', 'x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'phi', 'tcp_err_grasp', 'stopped', 'nominal', 'bound']), log=np.array([json.dumps(r, default=str) for r in runner.log]))
print(json.dumps({k_: res[k_] for k_ in ('settled_after_tail', 'sim_s', 'wall_s', 'physics_overrun_steps', 'n_commit', 't_ready', 't_bind', 'stopped_at', 'by_mode', 'handover', 'last_commit', 'phi_end_deg', 'base_speed', 'arm_rate_end', 'ctrl_final')}, ensure_ascii=False, default=str)[:4000])
print('P2', json.dumps(res['p2'], ensure_ascii=False, default=str)[:2000])
if runner.late: print('LATE', json.dumps(runner.late, default=str)[:1500])
runner.close()
BG.close()
