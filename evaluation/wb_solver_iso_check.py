"""低速 QP 配置 Isaac 趟次的離線判定：只讀趟次輸出，不開模擬器。

到達必須由**兩個獨立來源**同時成立：

1. Isaac 記錄的 `link_tcp` 世界位姿（E2.1 新增的欄位）
2. 由**實測底盤位姿與關節角**重算的 FK

**求解器自報的誤差不被接受。** 兩來源不一致即判定不成立。

同動由**實測運動**判定，與到達分項；停止觀察窗取**窗內最大原始值**，
不扣漂移。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'install',
                                'ammr_wholebody_mpc', 'lib',
                                'python3.12', 'site-packages'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from wb_wheel_limit import wheel_matrix                              # noqa: E402

DEFAULT_SPEC = 'evaluation/results/specs/wb_solver_iso_criteria_v1.yaml'
WANT = 'wb_solver_iso_criteria/1'
ARM = [f'joint{i}' for i in range(1, 7)]


def rot_err(Rc, Rd):
    Re = Rc.T @ Rd
    c = float(np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return 0.0
    return float(abs(th))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--spec', default=DEFAULT_SPEC)
    ap.add_argument('--urdf', default=os.path.join(
        HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--allow-recheck', action='store_true')
    a = ap.parse_args()

    raw = open(a.spec, 'rb').read()
    spec_sha = hashlib.sha256(raw).hexdigest()
    S = yaml.safe_load(raw.decode('utf-8'))
    if S.get('schema') != WANT:
        print(f"**規格 schema 不符**：{S.get('schema')}"); return 3
    if not S.get('prospective') or not S.get('not_retroactive'):
        print('**規格未宣告 prospective／不追溯 —— 拒絕判定**'); return 3
    SP = S['S_post_arrival']
    if not SP.get('drift_subtraction_forbidden') or not SP.get('use_max_not_endpoint'):
        print('**規格未禁止扣漂移／未要求取最大值 —— 拒絕判定**'); return 3
    R, C, B, L, P, D = (S['R_arrival'], S['C_simultaneity'], S['B_bounds'],
                        S['L_wheel'], S['P_preflight'], S['D_protections'])

    OUT = os.path.join(a.run, 'solver_iso_check.json')
    if os.path.exists(OUT) and not a.allow_recheck:
        print(f'**拒絕覆寫**：{OUT} 已存在。加 --allow-recheck'); return 5
    shutil.copyfile(a.spec, os.path.join(a.run, 'criteria_used.yaml'))

    def rd(n):
        p = os.path.join(a.run, n)
        return json.load(open(p)) if os.path.exists(p) else None

    run = rd(os.path.join('sim', 'wb_run.json'))
    pf = rd('freespace_preflight.json')
    sol = rd('solver_out.json')
    missing = [n for n, v in [('sim/wb_run.json', run),
                              ('freespace_preflight.json', pf),
                              ('solver_out.json', sol)] if v is None]
    if missing:
        print(f'**缺少輸出：{missing} —— 判定不成立**（缺少 ≠ 通過）'); return 3
    if run.get('execution_version') != 'E2.1':
        print(f"**執行版本 {run.get('execution_version')} 不是 E2.1**"); return 3

    cols = run['log_cols']
    Lg = np.array(run['log'], dtype=float)
    i = {c: k for k, c in enumerate(cols)}
    t = Lg[:, i['t']]
    DT = 0.01
    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    tgt = np.array(S['profile']['target_odom_m'], float)
    zc = np.array([1.0, 0.0, 0.0]); xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc; xc /= np.linalg.norm(xc)
    R_des = np.column_stack([xc, np.cross(zc, xc), zc])

    res = []
    def chk(k, ok, d):
        res.append({'key': k, 'pass': bool(ok), 'detail': d})

    # ---- P 起動前檢查 ----
    chk('P1_起動前檢查通過', pf.get('passed') is True,
        f"fails={pf.get('fails')}")
    chk('P2_障礙物設定為 0',
        pf.get('obstacles_configured') == P['P2_obstacles_configured_eq'],
        f"obstacles_configured={pf.get('obstacles_configured')}")
    chk('P3_每連桿 TF 合格',
        pf.get('tf_ok_links') == pf.get('n_links') and pf.get('n_links', 0) > 0,
        f"{pf.get('tf_ok_links')}/{pf.get('n_links')}")
    chk('P4_執行端場景檢查', pf.get('endpoint_scene_ok') is True,
        f"endpoint_scene_ok={pf.get('endpoint_scene_ok')}")
    got, exp = pf.get('safety_vmax_effective') or {}, pf.get('expected_vmax') or {}
    same = bool(exp) and all(
        got.get(k) is not None and math.isclose(got[k], v, abs_tol=1e-9)
        for k, v in exp.items())
    chk('P5_下游與求解端同一份低速框', same, f'下游 {got} vs 求解端 {exp}')

    # ---- R 到達：兩個獨立來源 ----
    have_tcp = 'tcp_x' in i
    chk('R0_Isaac 有記錄 link_tcp', have_tcp,
        'E2.1 欄位 tcp_x/tcp_y/tcp_z + 旋轉矩陣' if have_tcp else '缺欄位')
    ep_iso = np.full(len(t), np.nan)
    er_iso = np.full(len(t), np.nan)
    ep_fk = np.full(len(t), np.nan)
    agree = np.full(len(t), np.nan)
    if have_tcp:
        for k in range(len(t)):
            p_iso = Lg[k, [i['tcp_x'], i['tcp_y'], i['tcp_z']]]
            Riso = np.array([[Lg[k, i[f'tcp_r{r}{c}']] for c in range(3)]
                             for r in range(3)])
            ep_iso[k] = np.linalg.norm(tgt - p_iso)
            er_iso[k] = rot_err(Riso, R_des)
            q9 = np.zeros(9)
            q9[0], q9[1], q9[2] = Lg[k, i['base_x']], Lg[k, i['base_y']], \
                Lg[k, i['base_yaw']]
            q9[3:] = [Lg[k, i[f'{j}_act']] for j in ARM]
            T = K.fk(q9, 'link_tcp')
            ep_fk[k] = np.linalg.norm(tgt - T[:3, 3])
            agree[k] = np.linalg.norm(T[:3, 3] - p_iso)
    ok_both = (ep_iso <= R['R1_pos_err_m_max']) & (er_iso <= R['R2_rot_err_rad_max']) \
        & (ep_fk <= R['R1_pos_err_m_max'])
    best = cur = 0
    for v in ok_both:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    sustain = best * DT
    chk('R1_兩來源同時達標並持續', sustain >= R['R3_sustain_s_min'],
        f"最長連續 {sustain:.2f}s（≥ {R['R3_sustain_s_min']}）；"
        f'Isaac **全程最小**誤差 {np.nanmin(ep_iso)*1000:.3f} mm、'
        f'FK 最小 {np.nanmin(ep_fk)*1000:.3f} mm、'
        f'末值 {ep_iso[-1]*1000:.3f} mm')
    ag = float(np.nanmax(agree)) if np.isfinite(agree).any() else float('nan')
    chk('R2_兩來源一致', ag <= R['R5_source_agreement_m_max'],
        f"最大差 {ag*1000:.3f} mm（≤ {R['R5_source_agreement_m_max']*1000:.0f}）")
    chk('R3_姿態誤差（完整三維）',
        float(np.nanmin(er_iso)) <= R['R2_rot_err_rad_max'],
        f"最小 {np.nanmin(er_iso):.5f} rad（≤ {R['R2_rot_err_rad_max']}）")

    # ---- C 同動：實測運動 ----
    bv = Lg[:, i['base_lin_meas']]
    rates = np.column_stack([np.gradient(Lg[:, i[f'{j}_act']], t) for j in ARM])
    am = np.max(np.abs(rates), axis=1)
    both = (bv > C['base_moving_mps']) & (am > C['arm_moving_rps'])
    best = cur = 0
    for v in both:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    chk('C1_最長連續同動', best * DT >= C['C1_longest_continuous_s_min'],
        f"{best*DT:.2f}s（≥ {C['C1_longest_continuous_s_min']}）；"
        f'累計 {both.sum()*DT:.2f}s（**實測運動判定**）')

    # ---- B 界限 ----
    vx, vy, wz = Lg[:, i['vx_cmd']], Lg[:, i['vy_cmd']], Lg[:, i['wz_cmd']]
    f = np.isfinite(vx) & np.isfinite(vy)
    lin = np.hypot(vx[f], vy[f])
    chk('B1_線速度', float(lin.max()) <= B['B1_lin_mps_max'] + 1e-9,
        f"max {lin.max():.6f} ≤ {B['B1_lin_mps_max']}")
    chk('B2_角速度',
        float(np.nanmax(np.abs(wz[np.isfinite(wz)]))) <= B['B2_ang_rps_max'] + 1e-9,
        f"max {np.nanmax(np.abs(wz[np.isfinite(wz)])):.6f} ≤ {B['B2_ang_rps_max']}")
    cc = run['cmd_chain']
    chk('B3_手臂門檻不變',
        float(cc['config']['arm_rate_max']) == B['B3_arm_rate_max'],
        f"arm_rate_max={cc['config']['arm_rate_max']}")
    lsb = run.get('low_speed_interface_bound', {})
    chk('B4_執行端界限未動',
        float(lsb.get('lin_mps', -1)) == B['B1_lin_mps_max']
        and float(lsb.get('ang_rps', -1)) == B['B2_ang_rps_max'],
        f"{lsb.get('lin_mps')}/{lsb.get('ang_rps')}")
    chk('B5_rejected', cc.get('rejected') == B['B5_rejected_eq'],
        f"rejected={cc.get('rejected')}")
    chk('B6_fail_is_none', (cc.get('fail') is None) == B['B6_fail_is_none'],
        f"fail={cc.get('fail')}")

    # ---- L 輪級 ----
    ws = Lg[:, i['wheel_speed_max']]
    wa = Lg[:, i['wheel_accel_max']]
    tolr = float(L['L3_tolerance'])
    wsm = float(np.nanmax(ws[np.isfinite(ws)])) if np.isfinite(ws).any() else 0.0
    wam = float(np.nanmax(wa[np.isfinite(wa)])) if np.isfinite(wa).any() else 0.0
    chk('L1_輪速', wsm <= L['L1_wheel_speed_max_mps'] + tolr,
        f"max {wsm:.6f} ≤ {L['L1_wheel_speed_max_mps']}")
    chk('L2_輪加速度', wam <= L['L2_wheel_accel_max_mps2'] + tolr,
        f"max {wam:.6f} ≤ {L['L2_wheel_accel_max_mps2']}")

    # ---- S 到達後觀察 ----
    idx = np.where(ok_both)[0]
    t_arr = float(t[idx[0]]) if idx.size else float('nan')
    t_end = t_arr + float(SP['S1_observe_after_arrival_s'])
    covers = np.isfinite(t_arr) and t[-1] >= t_end
    chk('S1_log 涵蓋到達後觀察窗', bool(covers),
        f'到達 {t_arr:.2f}s，窗至 {t_end:.2f}s，log 至 {t[-1]:.2f}s')
    if covers:
        m = (t >= t_arr) & (t <= t_end)
        bx, by = Lg[m, i['base_x']], Lg[m, i['base_y']]
        bmax = float(np.max(np.hypot(bx - bx[0], by - by[0])))
        qa = np.column_stack([Lg[m, i[f'{j}_act']] for j in ARM])
        amax = float(np.max(np.abs(qa - qa[0])))
        chk('S2_觀察窗內底盤最大位移', bmax <= SP['S3_base_max_move_m'],
            f"{bmax*1000:.3f} mm（≤ {SP['S3_base_max_move_m']*1000:.0f}；未扣漂移）")
        chk('S3_觀察窗內關節最大變化', amax <= SP['S4_arm_max_move_rad'],
            f"{amax*1000:.3f} mrad（≤ {SP['S4_arm_max_move_rad']*1000:.0f}）")
    else:
        chk('S2_觀察窗內底盤最大位移', False, '觀察窗未涵蓋')
        chk('S3_觀察窗內關節最大變化', False, '觀察窗未涵蓋')

    chk('D1_pregrasp 由獨立條件禁止',
        run.get('pregrasp_preconditions_met') is False,
        f"preconditions={run.get('pregrasp_preconditions_met')}")
    chk('D2_熱中止門檻',
        float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
        f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")

    npass = sum(r['pass'] for r in res)
    ok = npass == len(res)
    print(f"\n=== {S['run_label']} ===")
    print(f"  規格 {a.spec}  {S['schema']} {S['version']} sha256 {spec_sha[:16]}")
    print(f"  **{S['not_b_baseline'].strip()}**")
    for r in res:
        print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:26} {r['detail']}")
    print(f'\n  {npass}/{len(res)} 項通過')
    print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修、不換目標'}")
    json.dump({'schema': 'wb_solver_iso_check/1', 'run': a.run,
               'run_label': S['run_label'], 'not_b_baseline': S['not_b_baseline'],
               'spec_version': S['version'], 'spec_sha256': spec_sha,
               'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
               'measured': {
                   'arrival_sim_t': t_arr,
                   'sustain_s': sustain,
                   'ep_iso_min_m': float(np.nanmin(ep_iso)),
                   'ep_iso_final_m': float(ep_iso[-1]),
                   'ep_fk_min_m': float(np.nanmin(ep_fk)),
                   'ep_fk_final_m': float(ep_fk[-1]),
                   'er_iso_min_rad': float(np.nanmin(er_iso)),
                   'source_agreement_max_m': ag,
                   'simultaneous_longest_s': best * DT,
                   'lin_max': float(lin.max()), 'wheel_speed_max': wsm,
                   'wheel_accel_max': wam},
               'solver_self_report': sol.get('completed'),
               'solver_self_report_not_used_for_acceptance': True,
               'not_claimed': ('上游全身安全性在命令被修改後未重新論證；'
                               '停止掃掠範圍無避碰保證；接觸操作不在範圍內')},
              open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f'  -> {OUT}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
