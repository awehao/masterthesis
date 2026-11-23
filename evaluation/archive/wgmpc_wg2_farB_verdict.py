#!/usr/bin/env python3
"""遠目標 B 趟次的四項判定（A–D）＋ 三個觀察項。

判準在實跑**之前**已寫定於
evaluation/results/wgmpc_wg2_far_target_B_spec.yaml，本檔只讀資料、不改判準。

  A 到達並保持      二分
  B 底盤淨位移      二分（>= 認證下界；**取 TCP 進入容差的那一物理步**）
  C 移動期間同動    二分 + 描述
  D 保持窗振動      只記錄，**不設門檻**
  觀察項：j3 起始裕度、網格碰撞（由 Isaac 接觸回報）、Isaac TCP 與節點 FK 的差 δ
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

SPEC = os.path.join(HERE, 'results', 'wgmpc_wg2_far_target_B_spec.yaml')
ARMJ = ['joint%d' % (i + 1) for i in range(6)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--json-out', default=None)
    g = ap.parse_args()
    import yaml
    sp = yaml.safe_load(open(SPEC))
    acc = sp['acceptance']
    lb = float(acc['B_base_participation']['certified_lower_bound_with_tol_m'])
    lb_exact = float(acc['B_base_participation']['certified_lower_bound_exact_m'])

    w = json.load(open(g.run + '/wg2_out.json'))
    s = json.load(open(g.run + '/sim/wb_run.json'))
    a = w['args']
    K = WholeBodyKinematics.from_urdf_file(a['urdf'])
    ci = {c: i for i, c in enumerate(s['log_cols'])}
    L = np.array(s['log'], float)
    ts = L[:, ci['t']]
    pub = [x for x in w['log'] if x.get('published')]
    out = {'run': os.path.basename(g.run.rstrip('/')),
           'target': w['target_tcp'], 'gamma': w['stats'].get('near_target_gamma'),
           'stop_why': w['stats'].get('stop_why')}
    print(f"趟次 {out['run']}   目標 {np.round(w['target_tcp'],4).tolist()}   "
          f"γ={out['gamma']}   stop_why={out['stop_why']}")
    print(f"已發布週期 {len(pub)}；物理步 {len(L)}；sim {ts[0]:.2f}–{ts[-1]:.2f} s")

    # ---------------- A 到達並保持 ----------------
    ep = np.array([x['err_p'] for x in pub]); er = np.array([x['err_r'] for x in pub])
    tc = np.array([x['sim_t'] for x in pub])
    ok = (ep <= a['reach_pos_m']) & (er <= a['reach_rot_rad'])
    I = np.where(ok)[0]
    A = {'reached_held_flag': bool(w.get('reached_held'))}
    if len(I):
        seg = max(np.split(I, np.where(np.diff(I) > 1)[0] + 1), key=len)
        t0, t1 = float(tc[seg[0]]), float(tc[seg[-1]])
        A.update(hold_t0=t0, hold_t1=t1, hold_s=t1 - t0, n_cycles=int(len(seg)))
        hw = (ts >= t0) & (ts <= t1)
        A['n_physics_steps'] = int(hw.sum())
        A['pass'] = bool(t1 - t0 >= a['hold_s'] and hw.sum() >= 200)
    else:
        t0 = t1 = None; hw = np.zeros(len(ts), bool)
        A.update(hold_s=0.0, n_cycles=0, n_physics_steps=0, pass_=False)
        A['pass'] = False
    out['A_reach_and_hold'] = A
    print(f"\n[A] 到達並保持：{'**通過**' if A['pass'] else '**未通過**'}"
          f"  保持 {A.get('hold_s',0):.3f} s / {A.get('n_physics_steps',0)} 物理步"
          f"（需 >= {a['hold_s']} s 且 >= 200 步）")
    if t0 is not None:
        print(f"    保持窗 sim {t0:.3f}–{t1:.3f}；窗內誤差 "
              f"位置 p50 {1e3*np.median(ep[seg]):.2f} mm max {1e3*ep[seg].max():.2f}；"
              f"姿態 p50 {1e3*np.median(er[seg]):.2f} mrad max {1e3*er[seg].max():.2f}")

    # ---------------- 觀察項：δ (c) Isaac TCP vs 節點 FK ----------------
    q_all = np.column_stack([L[:, ci['base_x']], L[:, ci['base_y']], L[:, ci['base_yaw']]]
                            + [L[:, ci['%s_act' % j]] for j in ARMJ])
    fin = np.all(np.isfinite(q_all), axis=1) & np.all(
        np.isfinite(L[:, [ci['tcp_x'], ci['tcp_y'], ci['tcp_z']]]), axis=1)
    idx = np.where(fin)[0]
    idx = idx[::max(1, len(idx) // 400)]
    d_fk = np.array([np.linalg.norm(
        K.fk(q_all[k], a['tcp'])[:3, 3] - L[k, [ci['tcp_x'], ci['tcp_y'], ci['tcp_z']]])
        for k in idx])
    delta_c = float(d_fk.max())
    delta = 1e-6 + 1e-9 + delta_c
    out['model_sync'] = {'n_sampled': int(len(idx)), 'fk_vs_isaac_max_m': delta_c,
                         'fk_vs_isaac_p50_m': float(np.median(d_fk))}
    out['delta_m'] = delta
    print(f"\n[觀察] Isaac TCP vs 節點 FK（{len(idx)} 取樣）："
          f"max {delta_c*1e3:.3f} mm、p50 {np.median(d_fk)*1e3:.3f} mm")
    print(f"    δ = 1e-6（紀錄） + 1e-9（界） + {delta_c:.6f}（模型差） = {delta:.6f} m")

    # ---------------- B 底盤淨位移 ----------------
    B = {'certified_lower_bound_with_tol_m': lb,
         'certified_lower_bound_exact_m': lb_exact, 'delta_m': delta}
    if t0 is not None:
        k = int(np.argmin(np.abs(ts - t0)))          # TCP 進入容差的那一物理步
        st = np.array(w['start_q'][:2], float)
        disp = float(np.hypot(L[k, ci['base_x']] - st[0], L[k, ci['base_y']] - st[1]))
        path = float(np.sum(np.linalg.norm(
            np.diff(L[:k + 1][:, [ci['base_x'], ci['base_y']]], axis=0), axis=1)))
        dend = float(np.hypot(L[-1, ci['base_x']] - st[0], L[-1, ci['base_y']] - st[1]))
        B.update(t_eval_step=k, t_eval_sim_t=float(ts[k]),
                 start_base_xy=st.tolist(),
                 base_xy_at_t_eval=[float(L[k, ci['base_x']]), float(L[k, ci['base_y']])],
                 net_displacement_m=disp, path_length_to_t_eval_m=path,
                 net_displacement_at_end_m=dend,
                 base_yaw_at_t_eval=float(L[k, ci['base_yaw']]))
        B['pass'] = bool(disp >= lb)
        B['below_bound_beyond_delta'] = bool(disp < lb - delta)
        print(f"\n[B] 底盤淨位移：{'**通過**' if B['pass'] else '**未通過**'}")
        print(f"    t_eval = 物理步 {k}，sim {ts[k]:.3f} s")
        print(f"    淨位移 {disp:.6f} m   （認證下界 {lb:.6f}，精確到達 {lb_exact:.6f}）")
        print(f"    路徑長至 t_eval {path:.6f} m（**非判準**）；"
              f"趟末淨位移 {dend:.6f} m（**非判準**）")
        if disp < lb:
            print(f"    低於下界 {lb-disp:.6f} m；δ = {delta:.6f} ⇒ "
                  + ('**啟動幾何不一致調查**' if B['below_bound_beyond_delta']
                     else '差距未超過 δ，**不構成不一致**'))
    else:
        B['pass'] = None
        print('\n[B] 底盤淨位移：**無法判定**（未進入容差，沒有 t_eval）')
    out['B_base_participation'] = B

    # ---------------- C 移動期間同動 ----------------
    vb, va = a.get('vmax_base_lin', 0.035255), a.get('vmax_arm', 0.999900)
    vb, va = 0.035255, 0.999900
    tr = [x for x in pub if (t0 is None or x['sim_t'] <= t0)]
    C = {'n_transit_cycles': len(tr)}
    if len(tr) >= 4:
        U = np.array([x['request_body'] for x in tr], float)
        bb = np.hypot(U[:, 0], U[:, 1]) / vb
        aa = np.abs(U[:, 3:]).max(1) / va
        both = (bb > 0.05) & (aa > 0.05)
        C.update(both_moving_frac=float(both.mean()),
                 base_moving_frac=float((bb > 0.05).mean()),
                 arm_moving_frac=float((aa > 0.05).mean()),
                 base_axis_saturated_frac=float(
                     ((np.abs(U[:, 0]) / vb > 0.98) | (np.abs(U[:, 1]) / vb > 0.98)).mean()),
                 arm_saturated_frac=float((aa > 0.98).mean()),
                 arm_cmd_p50=float(np.median(aa)),
                 arm_cmd_iqr=[float(np.percentile(aa, 25)), float(np.percentile(aa, 75))])
        C['pass'] = bool(C['both_moving_frac'] >= 0.80)
        print(f"\n[C] 移動期間同動：{'**通過**' if C['pass'] else '**未通過**'}"
              f"  同動 {C['both_moving_frac']:.0%}（需 >= 80%），{len(tr)} 週期")
        print(f"    底盤動 {C['base_moving_frac']:.0%}、手臂動 {C['arm_moving_frac']:.0%}；"
              f"底盤逐軸飽和 {C['base_axis_saturated_frac']:.0%}、"
              f"手臂飽和 {C['arm_saturated_frac']:.0%}")
        print(f"    手臂命令佔框 p50 {C['arm_cmd_p50']:.2f} "
              f"(IQR {C['arm_cmd_iqr'][0]:.2f}–{C['arm_cmd_iqr'][1]:.2f})"
              f" —— 飽和期間是「底盤全速、手臂補其餘」，同動不等於非瑣碎分配")
    else:
        C['pass'] = None
        print('\n[C] 移動期間同動：**無法判定**（移動期間週期數不足）')
    out['C_simultaneous_motion'] = C

    # ---------------- 觀察項：j3 起始裕度 ----------------
    lim = np.array(K.joint_limits())
    q_arm0 = np.array(w['start_q'][3:], float)
    m = 0.05
    sl = np.minimum(q_arm0 - (lim[0, 3:] + m), (lim[1, 3:] - m) - q_arm0)
    j3_run = None
    if t0 is not None and len(idx):
        j3 = L[fin, ci['joint3_act']]
        j3_run = {'min_rad': float(j3.min()),
                  'slack_to_effective_lower_rad': float(j3.min() - (lim[0, 5] + m))}
    out['j3'] = {'start_slack_rad': float(sl[2]), 'start_q3': float(q_arm0[2]),
                 'run': j3_run}
    print(f"\n[觀察] j3：起始裕度 {sl[2]:.6f} rad（起始角 {q_arm0[2]:+.6f}）")
    if j3_run:
        print(f"    趟中最小 j3 = {j3_run['min_rad']:+.6f}；"
              f"距有效下界 {j3_run['slack_to_effective_lower_rad']:+.6f} rad"
              + ('  <- **貼住該界**' if j3_run['slack_to_effective_lower_rad'] < 1e-3 else ''))

    # ---------------- C2 移動段運動品質（原 C 量不到的部分）----------------
    C2 = {}
    if len(tr) >= 8:
        U = np.array([x['request_body_presolve'] for x in tr], float)
        kk = [int(np.argmin(np.abs(ts - x['sim_t']))) for x in tr]
        qq = np.column_stack([L[kk][:, ci['base_x']], L[kk][:, ci['base_y']],
                              L[kk][:, ci['base_yaw']]]
                             + [L[kk][:, ci['%s_act' % j]] for j in ARMJ])
        tcp = np.array([K.fk(q, a['tcp'])[:3, 3] for q in qq])
        path = float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1)))
        net = float(np.linalg.norm(tcp[-1] - tcp[0]))
        epv = np.array([x['err_p'] for x in tr])
        d = np.diff(epv)
        # **以實際時間間隔相除**，不是固定 0.05：漏槽會造成 0.25 s 的缺口，
        # 除以 0.05 會把跨缺口的位移灌水 5 倍（實測曾報出 1535 mm/s 的假峰值）。
        dtc = np.diff([x['sim_t'] for x in tr])
        sp = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(dtc, 1e-9)
        # 另以 100 Hz 物理步計算，週期不均時以此為準
        kk2 = np.where((ts >= tr[0]['sim_t']) & (ts <= tr[-1]['sim_t']))[0]
        tcp_p = np.array([K.fk(np.concatenate([
            [L[i, ci['base_x']], L[i, ci['base_y']], L[i, ci['base_yaw']]],
            [L[i, ci['%s_act' % j]] for j in ARMJ]]), a['tcp'])[:3, 3]
            for i in kk2])
        sp_p = (np.linalg.norm(np.diff(tcp_p, axis=0), axis=1)
                / np.maximum(np.diff(ts[kk2]), 1e-9))
        C2 = dict(tcp_path_m=path, tcp_net_m=net, path_ratio=path / max(net, 1e-9),
                  arm_flip_rate=float(np.mean([
                      np.mean(np.sign(U[:-1, j]) * np.sign(U[1:, j]) < 0)
                      for j in range(3, 9)])),
                  err_increase_frac=float(np.mean(d > 0)),
                  wasted_frac=float(d[d > 0].sum() / max(epv[0] - epv[-1], 1e-9)),
                  tcp_speed_max_mm_s=float(sp.max() * 1e3),
                  tcp_speed_p50_phys_mm_s=float(np.median(sp_p) * 1e3),
                  tcp_speed_max_phys_mm_s=float(sp_p.max() * 1e3),
                  n_cycle_gaps=int((np.diff([x['sim_t'] for x in tr]) > 0.06).sum()),
                  arm_at_box_frac=float(np.mean(np.abs(U[:, 3:]).max(1) > 0.99 * 0.9999)),
                  n_sqp_p50=float(np.median([x['n_sqp'] for x in tr])),
                  no_progress_frac=float(np.mean(
                      [x['sqp_stop'] == 'no_progress' for x in tr])))
        print('\n[C2] 移動段運動品質（**原 C 量不到的部分**，無事前門檻，先如實記錄）')
        print(f"     TCP 路徑/淨位移 {C2['path_ratio']:.3f}"
              f"（{C2['tcp_path_m']:.4f} / {C2['tcp_net_m']:.4f} m）")
        print(f"     手臂命令翻號率 {C2['arm_flip_rate']:.1%}；"
              f"貼速度框 {C2['arm_at_box_frac']:.1%}")
        print(f"     誤差變大的週期 {C2['err_increase_frac']:.1%}；"
              f"無效來回佔淨降量 {C2['wasted_frac']:.1%}")
        print(f"     TCP 速度（100 Hz 物理步）p50 {C2['tcp_speed_p50_phys_mm_s']:.1f} "
              f"max {C2['tcp_speed_max_phys_mm_s']:.1f} mm/s"
              + (f"；控制週期有 {C2['n_cycle_gaps']} 個 >0.06 s 的缺口"
                 if C2['n_cycle_gaps'] else ''))
        print(f"     SQP 疊代 p50 {C2['n_sqp_p50']:.0f}；"
              f"no_progress {C2['no_progress_frac']:.1%}")
    out['C2_transit_motion_quality'] = C2

    print('\n[D] 保持窗振動：**不設門檻**，另以 '
          'evaluation/wgmpc_wg2_hold_vibration.py 計算並如實呈現。')
    print('    換遠目標不代表振動已解決；終端構型與近目標基準不同，不可逐項對比。')

    if g.json_out:
        json.dump(out, open(g.json_out, 'w'), indent=1, ensure_ascii=False)
        print(f'\n寫出 {g.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
