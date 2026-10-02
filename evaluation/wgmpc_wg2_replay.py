#!/usr/bin/env python3
"""保持窗求解器離線重播。

把錄到的那一輪求解輸入逐輪餵回 `solve_sp`，先確認能重現錄下的
`request_body_presolve`（整形前輸出）。重現不了就表示仍缺求解輸入或內部狀態
—— 那本身就是結論，不以「差不多」帶過。

輸入全部來自既有檔案，不開 Isaac：
  wg2_out.json    每輪 snap 時刻、u_prev（stages.applied.v，body 系）、
                  request_body（整形後，body 系，即節點放進 _cmd_hist 的值）、
                  publish_sim_t、n_missed_slot、args、target_tcp
  cmd_env.jsonl   solver 段 stamp_sim_t -> source_seq；endpoint 段 -> 實際套用值
  sim/wb_run.json 每物理步 base_x/base_y/base_yaw 與 joint*_act、joint*_sp
"""
import argparse, json
import numpy as np

import sys
sys.path.insert(0, 'evaluation')
sys.path.insert(0, 'src/ammr_wholebody_mpc')

from ammr_wholebody_mpc.wgmpc_core import NU, task_error
from ammr_wholebody_mpc.wgmpc_core_sp import (
    ArmSetpointModel, WGMPCConfigSP, make_z, plant_phys_step, solve_sp)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

ARMJ = ['joint%d' % (i + 1) for i in range(6)]


def build(run):
    w = json.load(open(run + '/wg2_out.json'))
    s = json.load(open(run + '/sim/wb_run.json'))
    env = [json.loads(l) for l in open(run + '/cmd_env.jsonl')]
    env = [r for r in env if r.get('type') == 'env']
    a = w['args']
    ident = json.load(open(a['arm_ident']))
    am = ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'],
                          phys_dt=ident['phys_dt_measured_s'])
    cfg = WGMPCConfigSP(N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'],
                        arm_model=am, row_scaling=not a['no_row_scaling'])
    K = WholeBodyKinematics.from_urdf_file(a['urdf'])
    return w, s, env, a, cfg, K


def state_at(L, ci, t, tol=1e-6):
    """取 sim log 中時間最接近 t 的那一個物理步，回傳 (q9, s6, 實際時間, 時間差)。"""
    k = int(np.argmin(np.abs(L[:, ci['t']] - t)))
    q = np.array([L[k, ci['base_x']], L[k, ci['base_y']], L[k, ci['base_yaw']]]
                 + [L[k, ci['%s_act' % j]] for j in ARMJ], float)
    sp = np.array([L[k, ci['%s_sp' % j]] for j in ARMJ], float)
    return q, sp, float(L[k, ci['t']]), float(abs(L[k, ci['t']] - t))


def predict_delay(q, s, t_snap, cmd_hist, applied_by_seq, cfg, d_state, d_cmd,
                  use_applied, t_avail=None):
    """與節點 `_predict_delay` 同一套流程（同樣的查表規則與推進次數）。

    `applied_by_seq` 的值是 (u, 回報時刻)。節點只有在**回報已經到達**時才會用
    實際套用值，否則退回請求值；重播手上有全趟資料，若不加 `t_avail` 條件就會
    100% 用套用值，與節點的 37% 不符。
    """
    D_state = d_state * cfg.dt
    D_cmd = d_cmd * cfg.dt
    if D_state <= 0.0 or not cmd_hist:
        return q, s, 0, 0, 0
    dtp = cfg.arm_model.phys_dt
    n = int(round(D_state / dtp))
    qq, ss = np.asarray(q, float).copy(), np.asarray(s, float).copy()
    n_app = n_req = 0
    for j in range(n):
        tau = t_snap + j * dtp - D_cmd
        uu, seq = None, None
        for (tp, up, sq) in cmd_hist:
            if tp <= tau + 1e-12:
                uu, seq = up, sq
            else:
                break
        if uu is None:
            uu = np.zeros(NU)
        elif (use_applied and seq in applied_by_seq
              and (t_avail is None
                   or applied_by_seq[seq][1] <= t_avail + 1e-12)):
            uu = applied_by_seq[seq][0]
            n_app += 1
        else:
            n_req += 1
        qq, ss = plant_phys_step(qq, ss, uu, cfg, 1)
    return qq, ss, n, n_app, n_req


def acf(x, K=8):
    y = np.asarray(x, float); y = y - y.mean(); d = (y ** 2).sum()
    if d <= 0:
        return np.full(K + 1, np.nan)
    return np.array([float(y[:len(y) - k] @ y[k:] / d) for k in range(K + 1)])


def cycle_period(x, kmax=6):
    """自相關在 lag>=2 的最大值 ⇒「幾個控制週期一輪」，另回傳峰值強度。"""
    a = acf(x, kmax)
    if np.isnan(a).any():
        return 0, float('nan')
    k = int(np.argmax(a[2:kmax + 1])) + 2
    return k, float(a[k])


def diagnose(rows, K, T_des, cfg, ch=5):
    """第2步：哪些量也呈現同一個週期。第3步：單一輸入敏感度。

    第3步是**開迴路**的 —— 輸入維持實錄、只換一個通道，所以它指出
    振盪經由哪個通道進入輸出，**不等於因果已證明**。
    """
    from ammr_wholebody_mpc.wgmpc_core import task_error
    print('\n=== 第2步：各量的週期（保持窗內，20 Hz 序列）===')
    print(f"{'量':>26} {'RMS':>11} {'週期':>5} {'峰':>6}  L1..L6")

    def row(lbl, v, sc=1.0):
        v = np.asarray(v, float) * sc
        p, pk = cycle_period(v); a = acf(v)
        print(f'{lbl:>26} {np.std(v):11.5f} {p:5d} {pk:+6.2f}  '
              + ' '.join(f'{z:+5.2f}' for z in a[1:7]))

    j = ch
    row('u0 整形前輸出 [qd%d]' % (j - 2), [r['u0'][j] for r in rows])
    row('q_pred 預測關節角', [r['qp'][j] for r in rows])
    row('q0 實測關節角', [r['q0'][j] for r in rows])
    row('s_pred 預測設定點', [r['spp'][j - 3] for r in rows])
    row('s 實測設定點', [r['s6'][j - 3] for r in rows])
    row('u_prev', [r['up'][j] for r in rows])
    row('任務誤差 |e_p| 實測 mm',
        [np.linalg.norm(task_error(K, r['q0'], T_des, cfg.tcp)[:3]) for r in rows], 1e3)
    row('任務誤差 |e_p| 預測 mm',
        [np.linalg.norm(task_error(K, r['qp'], T_des, cfg.tcp)[:3]) for r in rows], 1e3)
    row('任務誤差 e_p[x] 預測 mm',
        [task_error(K, r['qp'], T_des, cfg.tcp)[0] for r in rows], 1e3)

    print('\n=== 第3步：單一輸入敏感度（開迴路）===')
    up_mean = np.mean([r['up'] for r in rows], 0)
    variants = {
        'A 基準（全部照實錄）': lambda r: (r['qp'], r['spp'], r['up']),
        'B 關掉延遲補償': lambda r: (r['q0'], r['s6'], r['up']),
        'C 只不預測設定點': lambda r: (r['qp'], r['s6'], r['up']),
        'D 只不預測 q': lambda r: (r['q0'], r['spp'], r['up']),
        'E u_prev 固定為窗內平均': lambda r: (r['qp'], r['spp'], up_mean),
    }
    print(f"{'變體':>24} {'RMS':>11} {'對基準':>8} {'週期':>5} {'峰':>6}  L1..L6")
    b0 = None
    for nm, f in variants.items():
        out = []
        for r in rows:
            q_, s_, u_ = f(r)
            rr = solve_sp(K, make_z(q_, s_), u_, T_des, cfg, U_warm=None)
            out.append(float(rr.u0[ch]))
        p, pk = cycle_period(out); a = acf(out); sd = float(np.std(out))
        if b0 is None:
            b0 = sd
        print(f'{nm:>24} {sd:11.5f} {sd/b0:8.1%} {p:5d} {pk:+6.2f}  '
              + ' '.join(f'{z:+5.2f}' for z in a[1:7]))
    print('  輸入維持實錄、只換一個通道 ⇒ 指出振盪經由哪個通道進入輸出，'
          '**不等於因果已證明**。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--diagnose', action='store_true',
                    help='重現判定之後，另做第2步（各量週期）與第3步（單一輸入敏感度）')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--lead-cycles', type=int, default=8,
                    help='保持窗之前額外重播幾輪，用來把 U_warm 帶到窗口起點')
    g = ap.parse_args()

    w, s, env, a, cfg, K = build(g.run)
    # T_des：位置取 target_tcp，姿態取**起始 start_q 的 FK 旋轉**（節點只給位置目標）
    T_des = np.eye(4)
    T_des[:3, 3] = np.array(w['target_tcp'], float)
    T_des[:3, :3] = K.fk(np.array(w['start_q'], float), a['tcp'])[:3, :3]
    ci = {c: i for i, c in enumerate(s['log_cols'])}
    L = np.array(s['log'], float)
    pub = [x for x in w['log'] if x.get('published')]

    # 合成係數要與趟次當時一致
    P, Q, G, h, kp = cfg.composed()
    assert np.allclose(G, w['composed_G']) and kp == w['composed_kp'], \
        '合成係數與趟次紀錄不符'
    print(f'合成係數吻合：kp={kp}, G[0]={G[0]:.9f}')

    # ---- source_seq：solver 封裝 stamp_sim_t 與節點 publish_sim_t 以 1 us 配對
    sol = [r for r in env if r['stage_name'] == 'solver' and r['derived']]
    sst = np.array([r['stamp_sim_t'] for r in sol])
    # 值與**回報時刻**一起存：節點要等回報到了才用得到實際套用值
    applied_by_seq = {r['source_seq']: (np.array(r['u'], float),
                                        float(r['stamp_sim_t']))
                      for r in env if r['stage_name'] == 'endpoint' and r['derived']}

    # log 裡的 publish_sim_t / snap_sim_t 都已四捨五入到 1e-6，但節點內部用的是
    # 未捨入值；_predict_delay 的查表是 `tp <= tau + 1e-12`，4.7e-7 的差就足以
    # 選到不同的那一筆。未捨入值取自 solver 封裝的 stamp_sim_t（發布時刻）
    # 與 log 的 paired.t_arm（快照時刻）。
    seq_of, pubt_exact = {}, {}
    for i, x in enumerate(pub):
        k = int(np.argmin(np.abs(sst - x['publish_sim_t'])))
        if abs(sst[k] - x['publish_sim_t']) <= 1e-6:
            seq_of[i] = sol[k]['source_seq']
            pubt_exact[i] = float(sst[k])

    def snap_t(x):
        # 節點的 `snap.sim_t = key * 1e-6`，key = round(t*1e6) ⇒ **µs 量化值**，
        # 正是 log 的 timing.snap_sim_t；不是各來源自己的 t_arm/t_base/t_sp
        #（三者與它最大相差 4.7e-7 s）。_predict_delay 的 tau 由它算起。
        return float(x['timing']['snap_sim_t'])

    def row_t(x):
        # 取 sim log 物理步時用來源自己的未量化時間，取最近一列即可
        return float(x['paired']['t_arm'])

    # ---- 保持窗
    ep = np.array([x['err_p'] for x in pub]); er = np.array([x['err_r'] for x in pub])
    I = np.where((ep <= a['reach_pos_m']) & (er <= a['reach_rot_rad']))[0]
    seg = max(np.split(I, np.where(np.diff(I) > 1)[0] + 1), key=len)
    i0, i1 = int(seg[0]), int(seg[-1])
    j0 = max(0, i0 - g.lead_cycles)
    print(f'保持窗 = 已發布輪次 [{i0}, {i1}]（{i1-i0+1} 輪），'
          f'重播自輪次 {j0}（含 {i0-j0} 輪前置以建立 U_warm）')

    # ---- 閘門 A：狀態重建是否正確（用 FK 誤差比對節點紀錄）
    print('\n=== 閘門 A：由 sim log 重建 q0，FK 誤差對比節點紀錄 ===')
    dp = dr = dt_max = 0.0
    for i in range(j0, i1 + 1):
        x = pub[i]
        q0, sp6, t_act, dt = state_at(L, ci, row_t(x))
        e = task_error(K, q0, T_des, cfg.tcp)
        dp = max(dp, abs(float(np.linalg.norm(e[:3])) - x['err_p']))
        dr = max(dr, abs(float(np.linalg.norm(e[3:])) - x['err_r']))
        dt_max = max(dt_max, dt)
    print(f'  err_p 最大差 {dp:.3e} m；err_r 最大差 {dr:.3e} rad；'
          f'snap 時刻對齊最大差 {dt_max:.3e} s')
    # 門檻由紀錄位數決定：sim log 以 1e-6 記錄位姿與關節角，TCP 力臂約 0.5 m，
    # 故 FK 誤差的重建底限就是 ~1e-6；比這更嚴的門檻無意義。
    gateA = dp < 5e-6 and dr < 5e-6
    print('  門檻 5e-6（sim log 以 1e-6 記錄 ⇒ 重建底限約 1e-6，不可能更嚴）')
    print('  -> 狀態重建%s' % ('通過' if gateA else '**未通過**，以下重播結果不可採信'))

    # ---- 重播
    print('\n=== 重播 ===')
    cmd_hist, U_warm, out, diag_rows = [], None, [], []
    # 前置：把窗口之前所有已發布輪次的命令都放進歷史（節點保留 128 筆）
    for i, x in enumerate(pub):
        if i < j0 and i in seq_of:
            cmd_hist.append((pubt_exact[i], np.array(x['request_body'], float),
                             seq_of[i]))
    cmd_hist = cmd_hist[-128:]
    prev_miss = pub[j0]['n_missed_slot'] if j0 < len(pub) else 0

    for i in range(j0, i1 + 1):
        x = pub[i]
        if x['n_missed_slot'] > prev_miss:
            U_warm = None          # 與節點一致：跨時槽就丟棄暖啟動
        prev_miss = x['n_missed_slot']
        q0, sp6, _, _ = state_at(L, ci, row_t(x))
        u_prev = np.array(x['stages']['applied']['v'], float)   # body 系
        qp, spp, n_used, n_app, n_req = predict_delay(
            q0, sp6, snap_t(x), cmd_hist, applied_by_seq, cfg,
            a['delay_comp_state_cycles'], a['delay_comp_cmd_cycles'],
            a['use_applied_for_predict'],
            t_avail=float(x['timing']['solve_start_sim_t']))
        # ---- 趟次若有 `solve_in`（節點新增的完整求解輸入紀錄）----
        # 就**直接用節點記下的那一組**，並把自行重建的值拿來比對：
        # 重建誤差從此可以量化，不必再從 dq 的純量摘要反推。
        si = x.get('solve_in')
        if si is not None:
            qp_rec = np.array(si['q_pred'], float)
            spp_rec = np.array(si['s_pred'], float)
            up_rec = np.array(si['u_prev'], float)
            recon_err = dict(
                q_pred=float(np.abs(qp - qp_rec).max()),
                s_pred=float(np.abs(spp - spp_rec).max()),
                u_prev=float(np.abs(u_prev - up_rec).max()))
            qp, spp, u_prev = qp_rec, spp_rec, up_rec
        else:
            recon_err = None
        r = solve_sp(K, make_z(qp, spp), u_prev, T_des, cfg, U_warm=U_warm)
        rec = np.array(x['request_body_presolve'], float)
        d = np.abs(r.u0 - rec) if r.u0 is not None else np.full(NU, np.nan)
        dc = x['delay_comp']
        dq_arm = float(np.abs(qp[3:] - q0[3:]).max())
        dq_pos = float(np.linalg.norm(qp[:2] - q0[:2]))
        out.append(dict(i=i, sim_t=x['sim_t'], seq=seq_of.get(i),
                        ok=bool(r.ok), max_abs_diff=float(np.nanmax(d)),
                        diff_base=float(np.nanmax(d[:3])), diff_arm=float(np.nanmax(d[3:])),
                        rec_arm_linf=float(np.abs(rec[3:]).max()),
                        n_sqp_rec=x['n_sqp'], n_sqp_replay=r.n_sqp_used,
                        sqp_stop_rec=x['sqp_stop'], sqp_stop_replay=r.sqp_stop_reason,
                        residual_rec=x['residual'], residual_replay=r.max_residual,
                        n_pred_used=n_used, n_pred_applied=n_app, n_pred_requested=n_req,
                        dq_arm_replay=dq_arm, dq_arm_rec=dc['dq_arm_max_rad'],
                        dq_pos_replay=dq_pos, dq_pos_rec=dc['dq_pos_m'],
                        dq_arm_match=bool(abs(dq_arm - dc['dq_arm_max_rad']) < 5e-7),
                        recon_err=recon_err, used_solve_in=bool(si is not None),
                        u0=(r.u0.tolist() if r.u0 is not None else None),
                        rec_presolve=rec.tolist()))
        if i >= i0:
            diag_rows.append(dict(q0=q0, s6=sp6, qp=qp, spp=spp, up=u_prev,
                                  u0=(r.u0 if r.u0 is not None else np.full(NU, np.nan))))
        if i >= j0:
            cmd_hist.append((pubt_exact.get(i, x['publish_sim_t']),
                             np.array(x['request_body'], float), seq_of.get(i, -1)))
            cmd_hist = cmd_hist[-128:]
        U_warm = r.U

    W = [o for o in out if o['i'] >= i0]
    md = np.array([o['max_abs_diff'] for o in W])
    print(f"  窗內 {len(W)} 輪：重現誤差 max {md.max():.3e}  p50 {np.median(md):.3e}  "
          f"min {md.min():.3e}")
    print(f"  相對錄到的手臂命令幅度（L_inf p50 "
          f"{np.median([o['rec_arm_linf'] for o in W]):.4f} rad/s）："
          f"{md.max()/max(np.median([o['rec_arm_linf'] for o in W]),1e-12):.1%}")
    nsi = sum(1 for o in W if o['used_solve_in'])
    if nsi:
        re_ = [o['recon_err'] for o in W if o['recon_err']]
        print(f"  趟次含 solve_in：{nsi}/{len(W)} 輪直接採用節點記下的求解輸入")
        for k in ('q_pred', 's_pred', 'u_prev'):
            v = [d[k] for d in re_]
            print(f"    自行重建 vs 節點紀錄 {k}: max {max(v):.3e} p50 {np.median(v):.3e}")
    else:
        print('  趟次不含 solve_in（舊趟次）⇒ 求解輸入全為重建，'
              '殘差無法與重建誤差分開')
    nm = sum(1 for o in W if o['dq_arm_match'])
    print(f"  延遲補償交叉檢查：dq_arm 與節點紀錄吻合 {nm}/{len(W)} 輪（門檻 5e-7 rad）")
    na = sum(o['n_pred_applied'] for o in W); nr = sum(o['n_pred_requested'] for o in W)
    print(f"  預測步用套用值/請求值 {na}/{nr}（節點全趟 "
          f"{w['stats']['n_pred_applied']}/{w['stats']['n_pred_requested']}）")
    ns = sum(1 for o in W if o['n_sqp_rec'] != o['n_sqp_replay'])
    nst = sum(1 for o in W if o['sqp_stop_rec'] != o['sqp_stop_replay'])
    print(f"  SQP 疊代數不符 {ns}/{len(W)}；停止理由不符 {nst}/{len(W)}")
    print(f"  前 8 輪逐輪 max|Δ|: " + ' '.join(f'{o["max_abs_diff"]:.2e}' for o in W[:8]))

    # 波形是否重現（與逐筆數值是否吻合分開判定）
    rec_a = np.array([o['rec_presolve'] for o in W])
    rep_a = np.array([o['u0'] for o in W])
    pr, pkr = cycle_period(rec_a[:, 5]); pp, pkp = cycle_period(rep_a[:, 5])
    cc = float(np.corrcoef(rec_a[:, 5], rep_a[:, 5])[0, 1])
    resid = float(np.std(rep_a[:, 5] - rec_a[:, 5]) / np.std(rec_a[:, 5]))
    print(f'\n  波形：錄到 週期 {pr} 峰 {pkr:+.2f} RMS {np.std(rec_a[:,5]):.4f}；'
          f'重播 週期 {pp} 峰 {pkp:+.2f} RMS {np.std(rep_a[:,5]):.4f}')
    print(f'       相關係數 {cc:+.5f}，殘差佔訊號 {resid:.1%}')
    gateW = (pr == pp) and abs(pkr - pkp) < 0.05 and cc > 0.99
    print(f'  波形重現判定：{"通過" if gateW else "**未通過**"}')

    gateB = md.max() < 1e-6
    print(f"\n重現判定：{'通過' if gateB else '**未通過**'}"
          f"（門檻 1e-6 rad/s，與 presolve 的紀錄位數相稱）")
    if g.diagnose:
        diagnose(diag_rows, K, T_des, cfg)
    if g.json_out:
        json.dump(dict(run=g.run, gateA=bool(gateA), gateB=bool(gateB),
                       i0=i0, i1=i1, j0=j0, cycles=out),
                  open(g.json_out, 'w'), indent=1)
        print(f'寫出 {g.json_out}')
    return 0 if (gateA and gateB) else 1


if __name__ == '__main__':
    raise SystemExit(main())
