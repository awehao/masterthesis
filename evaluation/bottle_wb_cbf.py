#!/usr/bin/env python3
"""CBF 實際介入且手臂作用的離線閉迴路（Codex 20261031_195839：少量案例、不開矩陣、不跑 Isaac）。
控制器是實跑 S0b 那份 S0bController＋gmpc_wb，受控對象是 test_bottle_s0b.Fake（理想底盤、設定點模型、合成接觸）。結果只驗控制邏輯與約束介入，不驗物理。
CBF 參數同 run_mission_roundtrip4.sh（α 0.5、動態邊距 0.69、靜態 0.47、剪枝 1.2、near 6、stride 2）；代價權重沿用 r3 的 nav_cfg，與任務導航不同（須揭露）。無 Raw-scan 護盾。
介入證據不看「有加入 CBF」：逐週期記回傳命令的 CBF 殘差與 slack、實測最小距離，並於相同狀態移除 CBF 重解，比較命令差。
hold 對照：相同初態、限制與重播的 v_b 參考，手臂不動，只比較底盤。
案例：S1 靜態障礙在路徑上迫使減速；D1 反向來車貼近路徑迫使改變底盤運動；F1 窗內突現障礙且 k0 硬約束不可行；F2 CBF 作用中求解失敗。
    python3 evaluation/bottle_wb_cbf.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s0b_path_check as PC                      # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

X0, V_C = -0.75, 0.05
DU_BOUND = 1e-3         # 命令差 ≥ 此值（m/s 範數）才算 CBF 確實改變命令；事前指定
R_PHYS = 0.39           # 護盾保護半徑（僅報實體距離）


def static_on_path(t):
    return [dict(x=0.37, y=0.0, radius=0.10, margin=0.47, static=True)]


def oncoming(t):
    return [dict(x=-0.30 - 0.25 * (t - 10.0), y=-0.70, radius=0.10, vx=-0.25, vy=0.0)]


def sudden(t):
    return [dict(x=0.10, y=0.0, radius=0.10, margin=0.47, static=True)] if t >= 9.5 else []


def make(obs, arm_mode='active', v_replay=None, cbf_extra=None, log=True):
    return B.S0bController(T.K, T.ID, T.T_G, T.T_READY, T.Q_READY, T.TOP, T.R_B, X0, T.WIN, T.X_END, v_c=V_C, required=T.REQ,
                           obstacles_fn=obs, cbf_cfg=dict(B.CBF_MISSION, **(cbf_extra or {})), arm_mode=arm_mode, v_replay=v_replay, log_cbf=log)


def clearance(f, obs):
    """實測底盤中心對障礙：回傳 (對 r_eff 的餘裕 m, 對實體 radius＋R_PHYS 的餘裕 m, 時刻)。"""
    best = (float('inf'), float('inf'), None)
    for h in f.hist:
        for o in obs(h['t']):
            d = math.hypot(h['q'][0] - o['x'], h['q'][1] - o['y'])
            m = d - (o['radius'] + o.get('margin', B.CBF_MISSION['cbf_safe_margin']))
            if m < best[0]:
                best = (m, d - o['radius'] - R_PHYS, h['t'])
    return best


def summarize(name, f, obs):
    rep, L = f.c.report(), f.c.cbf_log
    m_eff, m_phys, t_m = clearance(f, obs)
    active = [e for e in L if e['cbf_active'] > 0]
    bound = [e for e in active if e['du'] >= DU_BOUND]
    ph_bound = {}
    for e in bound:
        ph_bound[e['phase']] = ph_bound.get(e['phase'], 0) + 1
    rs, ns = [e['resid_s'] for e in active if math.isfinite(e['resid_s'])], [e['resid_ns'] for e in active if math.isfinite(e['resid_ns'])]
    xs = np.array([h['q'][:3] for h in f.hist])
    s = dict(case=name, arm_mode=rep['arm_mode'], success=rep['success'], stop=(rep['stop'] or {}).get('why'), exit=B.exit_code(rep), events=[(e['t'], e.get('to', e['what'])) for e in rep['events'] if e['what'] in ('phase', 'done', 'stop', 'wb_fallback')],
             n_fallback=rep['n_fallback'], n_ticks=len(L), n_cbf_active=len(active), n_cbf_bound=len(bound), phase_bound=ph_bound, du_max=round(max([e['du'] for e in L], default=0.0), 4),
             eps0_max=round(max([e['eps0'] for e in L], default=0.0), 5), resid_slack_min=round(min(rs), 6) if rs else None, resid_noslack_min=round(min(ns), 6) if ns else None,
             clear_eff_mm=round(m_eff * 1e3, 1), clear_phys_mm=round(m_phys * 1e3, 1), t_clear=t_m, x_end=round(float(xs[-1, 0]), 4), y_peak_mm=round(float(np.abs(xs[:, 1]).max()) * 1e3, 1), yaw_peak_deg=round(math.degrees(float(np.abs(xs[:, 2]).max())), 2),
             vx_max=round(float(np.max(np.gradient(xs[:, 0], 0.01))), 4), vx_min=round(float(np.min(np.gradient(xs[:, 0], 0.01))), 4),
             ms=dict(p50=round(float(np.percentile(rep['solve_ms_all'], 50)), 1), p95=round(float(np.percentile(rep['solve_ms_all'], 95)), 1), max=round(float(np.max(rep['solve_ms_all'])), 1)) if rep['solve_ms_all'] else None,
             n_over_50ms=rep['n_over_50ms'], stop_effect=rep['stop_effect'], slip_mm=round(rep['slip_max_mm_deg'][0], 3))
    s['n_gate'], s['done_t'] = rep['n_gate'], next((e['t'] for e in rep['events'] if e['what'] == 'done'), None)
    if s['done_t'] is not None:
        vm = np.gradient(xs[:, 0], 0.01)
        tt = np.array([h['t'] for h in f.hist])
        s['back_s_after_done'] = round(float(np.sum((vm < -0.002) & (tt >= s['done_t'])) * 0.01), 2)
        s['back_s_before_done'] = round(float(np.sum((vm < -0.002) & (tt < s['done_t'])) * 0.01), 2)
    if rep['arm_mode'] == 'active' and rep['t_close'] is not None:
        mins, sm, _, _, marg = PC.check(f)
        keys = [k for k in mins if not k.endswith('（LIFT）') or 'table_top' not in k]
        k0 = min(keys, key=lambda k_: mins[k_][0])
        s['geom'] = dict(min_key=k0, min_mm=round(mins[k0][0] * 1e3, 1), self_mm=round(sm[0] * 1e3, 1), self_pair=sm[3], joint_margin=round(marg, 3))
    tcp = np.array([h['tcp'] for h in f.hist])
    car = [i for i, h in enumerate(f.hist) if h['phase'] == 'CARRY']
    if car:                                                            # 持瓶相對底盤漂移：對 CARRY 起點
        rel = np.array([[np.linalg.norm(T.K.fk(np.r_[0.0, 0.0, 0.0, f.hist[i]['q'][3:]], 'link_tcp')[:3, 3] - T.K.fk(np.r_[0.0, 0.0, 0.0, f.hist[car[0]]['q'][3:]], 'link_tcp')[:3, 3])] for i in car[::5]])
        s['carry_rel_drift_mm'] = round(float(rel.max()) * 1e3, 2)
    tc, tl = rep['t_close'], rep['t_lift_start']
    if tc is not None and tl is not None:
        mk = [tc <= h['t'] <= tl for h in f.hist]
        s['tcp_drift_mm'] = round(float(np.ptp(tcp[mk], axis=0).max()) * 1e3, 2)
        s['base_move_grasp_mm'] = round(float(np.ptp(xs[mk, 0])) * 1e3, 2)
        v = np.gradient(xs[:, 0], 0.01)
        s['v_grasp_minmax'] = [round(float(v[mk].min()), 4), round(float(v[mk].max()), 4)]
    return s


def show(s):
    print(f"\n[{s['case']}／{s['arm_mode']}] success {s['success']} stop {s['stop']} 離開碼 {s['exit']} fallback {s['n_fallback']} 事件 {s['events']}")
    print(f"   CBF 週期 {s['n_ticks']}、作用列 >0 {s['n_cbf_active']}、確實改變命令 {s['n_cbf_bound']} {s['phase_bound']}（|Δu|峰 {s['du_max']}）；slack 峰 {s['eps0_max']}；回傳殘差 無slack {s['resid_noslack_min']}／含slack {s['resid_slack_min']}")
    print(f"   實測最小餘裕：對 r_eff {s['clear_eff_mm']} mm、對實體 {s['clear_phys_mm']} mm @ {s['t_clear']}；x 末 {s['x_end']}、y 峰 {s['y_peak_mm']} mm、偏航峰 {s['yaw_peak_deg']}°；vx {s['vx_min']}…{s['vx_max']}")
    print(f"   耗時 {s['ms']}、>50 ms {s['n_over_50ms']}；停止有效 {s['stop_effect']}；滑移 {s['slip_mm']} mm；夾持期 TCP 漂移 {s.get('tcp_drift_mm')} mm、底盤移動 {s.get('base_move_grasp_mm')} mm、速度 {s.get('v_grasp_minmax')}；管外擋 INSERT {s['n_gate']} 週期；倒退 完成前 {s.get('back_s_before_done')} s／後 {s.get('back_s_after_done')} s；持瓶相對底盤漂移 {s.get('carry_rel_drift_mm')} mm；幾何 {s.get('geom')}", flush=True)


def main():
    res = []
    for name, obs, t_end in (('S1 靜態路徑障礙', static_on_path, 30.0), ('D1 反向來車貼近路徑', oncoming, 30.0)):
        f = T.Fake(make(obs), x0=X0).run(t_end)
        s = summarize(name, f, obs)
        show(s)
        res.append(s)
        hist_v = list(f.c.v_b_hist)
        fh = T.Fake(make(obs, arm_mode='hold', v_replay=hist_v), x0=X0).run(min(t_end, f.t), tail=0.0)
        sh = summarize(name, fh, obs)
        show(sh)
        res.append(sh)
    f = T.Fake(make(sudden, cbf_extra=dict(cbf_hard_k0=True)), x0=X0).run(30.0)
    s = summarize('F1 窗內突現障礙（k0 硬約束）', f, sudden)
    show(s)
    res.append(s)
    c = make(static_on_path)
    c.fail_inject = lambda t: t >= 10.5
    f = T.Fake(c, x0=X0).run(30.0)
    s = summarize('F2 CBF 作用中求解失敗', f, static_on_path)
    show(s)
    res.append(s)
    out = os.path.join(HERE, 'results', 'bottle', 'wb_cbf_r3.json')
    json.dump(dict(note='假受控對象、理想底盤、合成接觸；CBF 參數同任務、代價權重同 r3；無護盾；耦合 QP 主求解器 GI；CARRY 目標跟實際底盤、S_LEAD 0.02、INSERT 放行管（r1 OSQP、r2 未修持瓶參考，皆保留）', du_bound=DU_BOUND, cases=res), open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫入', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    sys.exit(main())
