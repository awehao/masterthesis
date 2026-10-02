#!/usr/bin/env python3
"""離線挑選**需要底盤參與**的自由空間目標，並核對整段預測路徑。

動機：目前的自由空間目標水平半徑只有 0.4705 m，遠低於「底盤固定時」手臂
的水平可達上限 0.7000 m —— 也就是說**不動底盤本來就能到**。即使到達並
保持，也證明不了 W-GMPC 在分配底盤與手臂的運動。

本檔只做離線挑選與核對，不開 Isaac、不改任何控制參數。
離線通過**不等於** Isaac 會通過；最後仍須實跑確認（見 wgmpc_closed_loop_sim 的同一條界線）。

核對四項：
  R1 固定底盤不可達 —— 水平半徑超過可達上限（與姿態無關的充分條件），
                       另以多起點 IK 確認求不到解
  R2 底盤可動可達   —— 9 自由度 IK 有解，關節與底盤都在限位內且留裕度
  R3 整段路徑合規   —— 關節限位、輪速、離地、手臂不進底盤圓柱
  R4 途中同時運動   —— 底盤與手臂在移動期間確實同時動
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))


from ammr_wholebody_mpc.wgmpc_core import task_error       # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (      # noqa: E402
    WholeBodyKinematics)
import wgmpc_closed_loop_sim as CL                         # noqa: E402

# base_link 的碰撞圓柱（URDF）：半徑 0.300 m、高 0.28 m、原點 z=0.14
BASE_R, BASE_TOP = 0.300, 0.28
# 手臂側要核對的連桿（略過輪、輪輥與相機等不受手臂命令影響者）
ARM_LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6',
             'link_eef', 'uflite_gripper_link', 'uflite_finger1',
             'uflite_finger2', 'link_tcp']
FLOOR_MARGIN = 0.02        # 連桿原點離地至少這麼高
JOINT_MARGIN = 0.05        # 與 cfg.joint_margin 同值


def certified_bound(K, urdf, tcp, z=None):
    """**可認證**的水平半徑上界（三角不等式），與各關節角無關。

    串聯鏈：p_tcp − p_mount = Σ R_i(q) t_i ⇒ ||p_tcp − p_mount|| <= Σ|t_i|
    （旋轉不改變各段固定平移的長度）。手臂底座 link_base 剛接於底盤，
    底盤只平移與繞 z 轉 ⇒ 其水平偏移與高度**恆定**。
    給定 z 時可再收緊：水平分量 <= sqrt(Σ|t_i|² − (z − z_mount)²)。

    這個界**成立但不緊**（實測搜尋最大值約 0.70，界約 0.94）。
    它是唯一可以拿來說「必要條件」的數字；`reach_bound` 的搜尋最大值不行。
    """
    import xml.etree.ElementTree as ET
    r = ET.parse(urdf).getroot()
    par = {}
    for j in r.findall('joint'):
        o = j.find('origin'); xyz = np.zeros(3)
        if o is not None and o.get('xyz'):
            xyz = np.array([float(v) for v in o.get('xyz').split()])
        par[j.find('child').get('link')] = (j.find('parent').get('link'), xyz)
    cur, L = tcp, 0.0
    while cur != 'link_base':
        p, xyz = par[cur]; L += float(np.linalg.norm(xyz)); cur = p
    m = K.fk(np.zeros(9), 'link_base')[:3, 3]
    r0, z0 = float(np.hypot(m[0], m[1])), float(m[2])
    out = {'sum_abs_t_m': L, 'mount_radius_m': r0, 'mount_z_m': z0,
           'ub_any_z_m': r0 + L}
    if z is not None:
        dz = abs(float(z) - z0)
        out['z'] = float(z)
        out['ub_at_z_m'] = (r0 + float(np.sqrt(L ** 2 - dz ** 2))
                            if dz < L else r0)
    return out


def reach_bound(K, tcp, n_start=60, seed=1):
    """底盤固定時 TCP 水平半徑的**搜尋最大值**（多起點最佳化，不限姿態）。

    **這不是上界。** 多起點最佳化只給「目前搜尋到的最大值」，不保證不存在
    更遠的構型；超過它**不能**據以宣稱不可達。可認證的上界見 `certified_bound`。
    """
    lim = np.array(K.joint_limits()); lo, hi = lim[0, 3:], lim[1, 3:]
    q = np.zeros(9)

    def neg(x):
        q[3:] = x
        T = K.fk(q, tcp)
        return -float(np.hypot(T[0, 3], T[1, 3]))

    rng = np.random.default_rng(seed); best = 0.0
    for _ in range(n_start):
        r = minimize(neg, lo + (hi - lo) * rng.random(6),
                     bounds=list(zip(lo, hi)), method='L-BFGS-B')
        best = max(best, -r.fun)
    return float(best)


def ik(K, T_des, tcp, q_seed, free_base, lim, tol_p=1e-4, tol_r=1e-3,
       w_rot=0.05):
    """有界非線性最小平方 IK。`free_base=False` 時把前三維的界夾成定值。

    先前用固定步長的阻尼最小平方會發散：位置（m）與姿態（rad）被同等
    對待，姿態誤差一長大線性化就失效（實測 5 次疊代後 |er| 由 0 漲到
    2.8 rad）。這裡改用 `least_squares`（trf，帶界），並以 `w_rot`（m/rad）
    把姿態殘差換算到與位置同量級。
    """
    from scipy.optimize import least_squares
    q_seed = np.asarray(q_seed, float).copy()
    lo, hi = lim[0].copy(), lim[1].copy()
    if not free_base:
        for k in range(3):                     # 凍結底盤：上下界夾成起始值
            lo[k] = hi[k] = q_seed[k]
        lo[:3] -= 1e-9; hi[:3] += 1e-9
    q_seed = np.clip(q_seed, lo, hi)

    def resid(x):
        e = task_error(K, x, T_des, tcp)       # [dp(3) m, dr(3) rad]
        return np.concatenate([e[:3], w_rot * e[3:]])

    r = least_squares(resid, q_seed, bounds=(lo, hi), method='trf',
                      xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
    q = np.clip(r.x, lim[0], lim[1])
    e = task_error(K, q, T_des, tcp)
    ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
    return bool(ep < tol_p and er < tol_r), q, ep, er


def ik_multi(K, T_des, tcp, q0, free_base, n_seed=200, seed=0):
    lim = np.array(K.joint_limits())
    rng = np.random.default_rng(seed)
    best = (False, None, np.inf, np.inf)
    for k in range(n_seed):
        qs = np.asarray(q0, float).copy()
        if k:                                   # 第 0 顆用起始構型
            qs[3:] = lim[0, 3:] + (lim[1, 3:] - lim[0, 3:]) * rng.random(6)
            if free_base:
                qs[:2] = np.asarray(T_des[:2, 3]) + rng.uniform(-0.6, 0.6, 2)
                qs[2] = rng.uniform(-np.pi, np.pi)
        ok, q, ep, er = ik(K, T_des, tcp, qs, free_base, lim)
        if ok:
            return True, q, ep, er
        if ep < best[2]:
            best = (False, q, ep, er)
    return best


def path_check(rec, K, cfg, tcp):
    """整段路徑的限位／輪速／離地／自碰代理核對。"""
    Q = np.asarray(rec['q']); U = np.asarray(rec['u_req'])
    lim = np.array(K.joint_limits())
    out = {}
    # 1 關節限位（留 JOINT_MARGIN）
    lo, hi = lim[0, 3:] + JOINT_MARGIN, lim[1, 3:] - JOINT_MARGIN
    slack = np.minimum(Q[:, 3:] - lo, hi - Q[:, 3:])
    k = np.unravel_index(np.argmin(slack), slack.shape)
    out['joint_margin_min_rad'] = float(slack.min())
    out['joint_limit_ok'] = bool(slack.min() > 0)
    out['joint_worst'] = int(k[1]) + 1
    out['joint_worst_at_cycle'] = int(k[0])
    out['joint_margin_at_start_rad'] = float(slack[0].min())
    out['joint_margin_tightest_is_start'] = bool(k[0] <= 1)
    # 2 底盤平移限位
    bl = np.minimum(Q[:, :2] - lim[0, :2], lim[1, :2] - Q[:, :2])
    out['base_margin_min_m'] = float(bl.min())
    out['base_limit_ok'] = bool(bl.min() > 0)
    # 3 輪速（與核心同一個映射與上限）
    W = cfg.wheel_matrix(); wlim = cfg.wheel_radius * cfg.wheel_w_max
    ws = np.abs(U[:, :3] @ W.T)
    out['wheel_speed_max_mps'] = float(ws.max())
    out['wheel_speed_limit_mps'] = float(wlim)
    out['wheel_ok'] = bool(ws.max() <= wlim)
    # 4 離地與自碰代理（連桿**原點**，非網格）
    zmin = np.inf; dmin = np.inf; worst_z = worst_d = ''
    for L in ARM_LINKS:
        P = np.array([K.fk(q, L)[:3, 3] for q in Q])
        if P[:, 2].min() < zmin:
            zmin, worst_z = P[:, 2].min(), L
        rel = P[:, :2] - Q[:, :2]
        inside = P[:, 2] <= BASE_TOP
        if inside.any():
            d = np.hypot(rel[inside, 0], rel[inside, 1]).min() - BASE_R
            if d < dmin:
                dmin, worst_d = d, L
    out['floor_clear_min_m'] = float(zmin)
    out['floor_ok'] = bool(zmin > FLOOR_MARGIN)
    out['floor_worst_link'] = worst_z
    # **未觸發不等於通過**：若手臂連桿原點全程都高於底盤圓柱頂，這項條件
    # 根本沒被測到，要照實標示，不能當成已核對。
    out['base_cyl_tested'] = bool(np.isfinite(dmin))
    out['base_cyl_clear_min_m'] = (float(dmin) if np.isfinite(dmin) else None)
    out['self_collision_ok'] = (bool(dmin > 0) if np.isfinite(dmin) else None)
    out['base_cyl_worst_link'] = worst_d or None
    out['geometry_note'] = '以連桿**原點**核對，非網格層級碰撞；實跑由 Isaac 的接觸回報確認'
    return out


def simultaneity(rec, cfg, t_reach, v_base_frac=0.05, v_arm_frac=0.05):
    """移動期間底盤與手臂是否確實同時運動。"""
    t = np.asarray(rec['t']); U = np.asarray(rec['u_req'])
    m = t <= (t_reach if t_reach is not None else t[-1])
    if m.sum() < 4:
        return {'n_transit_cycles': int(m.sum())}
    bb = np.hypot(U[m, 0], U[m, 1]) / cfg.v_base_lin
    aa = np.abs(U[m, 3:]).max(1) / cfg.v_arm
    both = (bb > v_base_frac) & (aa > v_arm_frac)
    # 飽和：底盤的框是**逐軸** L_inf（|vx|,|vy| 各自 <= v_base_lin），
    # 所以 hypot 最大可到 sqrt(2)；逐軸飽和另外算。
    bx = np.abs(U[m, 0]) / cfg.v_base_lin
    by = np.abs(U[m, 1]) / cfg.v_base_lin
    return {'n_transit_cycles': int(m.sum()),
            'base_moving_frac': float((bb > v_base_frac).mean()),
            'arm_moving_frac': float((aa > v_arm_frac).mean()),
            'both_moving_frac': float(both.mean()),
            'base_cmd_p50_frac_of_box': float(np.median(bb)),
            'arm_cmd_p50_frac_of_box': float(np.median(aa)),
            'base_axis_saturated_frac': float(
                ((bx > 0.98) | (by > 0.98)).mean()),
            'arm_saturated_frac': float((aa > 0.98).mean()),
            'arm_cmd_frac_iqr': [float(np.percentile(aa, 25)),
                                 float(np.percentile(aa, 75))]}


def evaluate(K, cfg, q0, s0, tcp, R0, target, bound, t_end, ik_seeds, cb_at_z):
    T = np.eye(4); T[:3, 3] = np.asarray(target, float); T[:3, :3] = R0
    r_xy = float(np.hypot(target[0], target[1]))
    res = {'target': list(map(float, target)), 'radius_xy_m': r_xy,
           'searched_max_m': bound}
    # R1：**兩個層級分開報**
    #   beyond_searched_max  —— 只支持「預期不可達」，不足以宣稱必要條件
    #   beyond_certified_ub  —— 可認證，才能說「固定底盤必不可達」
    res['beyond_searched_max'] = bool(r_xy > bound)
    res['certified_ub_at_z_m'] = cb_at_z
    res['beyond_certified_ub'] = bool(r_xy > cb_at_z)
    res['certified_min_base_travel_m'] = float(max(0.0, r_xy - cb_at_z))
    ok_f, _, ep_f, er_f = ik_multi(K, T, tcp, q0, False, ik_seeds)
    res['ik_base_fixed'] = {'solved': bool(ok_f), 'best_err_p_m': ep_f,
                            'best_err_r_rad': er_f}
    res['R1_level'] = ('certified' if (res['beyond_certified_ub'] and not ok_f)
                       else ('search_supported'
                             if (res['beyond_searched_max'] and not ok_f)
                             else 'not_supported'))
    res['R1_base_fixed_infeasible'] = bool(res['beyond_certified_ub'] and not ok_f)
    # R2
    ok_m, q_sol, ep_m, er_m = ik_multi(K, T, tcp, q0, True, ik_seeds)
    res['ik_base_free'] = {'solved': bool(ok_m), 'best_err_p_m': ep_m,
                           'best_err_r_rad': er_m,
                           'q': (list(map(float, q_sol)) if q_sol is not None else None)}
    lim = np.array(K.joint_limits())
    if ok_m:
        sl = float(np.minimum(q_sol[3:] - lim[0, 3:] - JOINT_MARGIN,
                              lim[1, 3:] - JOINT_MARGIN - q_sol[3:]).min())
        res['ik_base_free']['joint_margin_m'] = sl
        res['R2_base_mobile_feasible'] = bool(sl > 0)
        res['base_travel_m'] = float(np.hypot(q_sol[0] - q0[0], q_sol[1] - q0[1]))
        res['base_travel_min_s_at_box'] = res['base_travel_m'] / cfg.v_base_lin
    else:
        res['R2_base_mobile_feasible'] = False
    if not (res['R1_level'] != 'not_supported' and res['R2_base_mobile_feasible']):
        res['R3'] = None; res['R4'] = None
        return res
    # R3 / R4：以與實跑相同的設定做離線閉迴路
    out = CL.run(cfg, K, q0, s0, T, t_end=t_end, tcp=tcp,
                 delay_cycles=1.0, d_pub_cycles=0.6,
                 comp_state=1.6, comp_cmd=1.0, gamma=1.0,
                 u_prev_mode='applied')
    rec = out['rec']
    res['sim'] = {'reached_held': bool(out['reached_held']),
                  't_reach_s': out['t_reach'], 'n_cycles': int(len(rec['t'])),
                  'n_fail': int(out['n_fail']),
                  'err_p_final_m': float(rec['ep'][-1]),
                  'err_r_final_rad': float(rec['er'][-1])}
    res['R3'] = path_check(rec, K, cfg, tcp)
    res['R4'] = simultaneity(rec, cfg, out['t_reach'])
    res['_rec'] = rec
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-run',
                    default='evaluation/runs/wgmpc_wg2_g1p0_111058',
                    help='取起始構型、URDF 與手臂辨識參數的參考趟次')
    ap.add_argument('--target', nargs=3, type=float, action='append',
                    help='候選目標（世界座標，可重複）')
    ap.add_argument('--t-end', type=float, default=90.0)
    ap.add_argument('--ik-seeds', type=int, default=200)
    ap.add_argument('--json-out', default=None)
    g = ap.parse_args()

    w = json.load(open(g.ref_run + '/wg2_out.json'))
    a = w['args']
    ident = json.load(open(a['arm_ident']))
    cfg = CL.make_cfg(ident, N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'])
    K = WholeBodyKinematics.from_urdf_file(a['urdf'])
    q0 = np.array(w['start_q'], float)
    s0 = q0[3:].copy()          # 起始設定點 = 起始關節角（握手時成立）
    R0 = K.fk(q0, a['tcp'])[:3, :3]
    tcp = a['tcp']

    bound = reach_bound(K, tcp)
    cur = np.array(w['target_tcp'], float)
    cb = certified_bound(K, a['urdf'], tcp, z=None)
    print(f'參考趟次 {os.path.basename(g.ref_run.rstrip("/"))}')
    print(f'起始 TCP {np.round(K.fk(q0,tcp)[:3,3],4)}；起始構型水平半徑 '
          f'{np.hypot(*K.fk(q0,tcp)[:2,3]):.4f} m')
    print(f'底盤固定時 TCP 水平半徑的**搜尋最大值** = {bound:.4f} m'
          f'（多起點最佳化；**不是上界**）')
    print(f'**可認證上界**（三角不等式，不分 z）= {cb["ub_any_z_m"]:.6f} m'
          f'  [Σ|t_i|={cb["sum_abs_t_m"]:.6f}, 底座半徑 {cb["mount_radius_m"]:.3f},'
          f' 底座高 {cb["mount_z_m"]:.3f}]')
    print(f'目前目標 {np.round(cur,4)}：水平半徑 {np.hypot(*cur[:2]):.4f} m '
          f'=> {"超出搜尋最大值" if np.hypot(*cur[:2])>bound else "**未超出搜尋最大值**，固定底盤即可達"}')
    print(f'底盤速度框 {cfg.v_base_lin:.6f} m/s；輪速上限 '
          f'{cfg.wheel_radius*cfg.wheel_w_max:.4f} m/s\n')

    tgts = g.target or [[1.00, 0.00, 0.45], [0.80, 0.60, 0.45], [1.20, 0.00, 0.45]]
    allr = []
    for t in tgts:
        print(f'=== 候選 {np.round(t,3).tolist()} ===')
        cbz = certified_bound(K, a['urdf'], tcp, z=t[2])['ub_at_z_m']
        r = evaluate(K, cfg, q0, s0, tcp, R0, t, bound, g.t_end, g.ik_seeds, cbz)
        r.pop("_rec", None)
        allr.append(r)
        print(f"  水平半徑 {r['radius_xy_m']:.4f} m"
              f"（搜尋最大值 {bound:.4f}；該 z 的可認證上界 {r['certified_ub_at_z_m']:.6f}）")
        print(f"  R1 固定底盤不可達：**{r['R1_level']}**"
              f"（超出搜尋最大值 {r['beyond_searched_max']}；"
              f"超出可認證上界 {r['beyond_certified_ub']}；"
              f"IK 求解 {r['ik_base_fixed']['solved']}，最佳殘差 "
              f"{r['ik_base_fixed']['best_err_p_m']*1e3:.1f} mm）")
        print(f"     認證必要底盤位移 = {r['certified_min_base_travel_m']:.6f} m"
              f"{'（為 0 ⇒ 認證層級上尚未成立）' if r['certified_min_base_travel_m']<=0 else ''}")
        print(f"  R2 底盤可動可達  : {r['R2_base_mobile_feasible']}", end='')
        if r['R2_base_mobile_feasible']:
            print(f"  （關節裕度 {r['ik_base_free']['joint_margin_m']:.4f} rad；"
                  f"底盤需移動 {r['base_travel_m']:.3f} m，"
                  f"速度框下至少 {r['base_travel_min_s_at_box']:.1f} s）")
        else:
            print(f"  （IK 最佳殘差 {r['ik_base_free']['best_err_p_m']*1e3:.1f} mm）")
        if r.get('R3'):
            s_ = r['sim']; c = r['R3']; m = r['R4']
            print(f"  離線閉迴路: 到達並保持 {s_['reached_held']}"
                  f"  t_reach {s_['t_reach_s']}  週期 {s_['n_cycles']}"
                  f"  求解失敗 {s_['n_fail']}")
            print(f"  R3 關節裕度 {c['joint_margin_min_rad']:+.4f} rad (j{c['joint_worst']}"
                  f"@週期{c['joint_worst_at_cycle']}"
                  f"{'，即起始構型' if c['joint_margin_tightest_is_start'] else ''}) {c['joint_limit_ok']}"
                  f" | 底盤裕度 {c['base_margin_min_m']:.3f} m {c['base_limit_ok']}")
            print(f"     輪速 max {c['wheel_speed_max_mps']:.4f} / "
                  f"{c['wheel_speed_limit_mps']:.4f} m/s {c['wheel_ok']}"
                  f" | 離地 min {c['floor_clear_min_m']:.3f} m ({c['floor_worst_link']}) {c['floor_ok']}"
                  f" | 底盤圓柱 {'淨空 %.3f m %s' % (c['base_cyl_clear_min_m'], c['self_collision_ok']) if c['base_cyl_tested'] else '**未觸發**（手臂連桿原點全程高於 0.28 m）'}")
            print(f"  R4 途中 {m['n_transit_cycles']} 週期：底盤動 {m['base_moving_frac']:.0%}"
                  f"、手臂動 {m['arm_moving_frac']:.0%}、**同時動 {m['both_moving_frac']:.0%}**")
            print(f"     底盤逐軸飽和 {m['base_axis_saturated_frac']:.0%}、"
                  f"手臂飽和 {m['arm_saturated_frac']:.0%}；"
                  f"手臂命令佔框 p50 {m['arm_cmd_p50_frac_of_box']:.2f} "
                  f"(IQR {m['arm_cmd_frac_iqr'][0]:.2f}–{m['arm_cmd_frac_iqr'][1]:.2f})")
        print()
    if g.json_out:
        json.dump({'searched_max_m': bound, 'certified_bound': cb,
                   'current_target': cur.tolist(),
                   'candidates': allr}, open(g.json_out, 'w'),
                  indent=1, ensure_ascii=False)
        print(f'寫出 {g.json_out}')
    print('離線通過**不等於** Isaac 會通過；碰撞只核到連桿**原點**，非網格層級。')
    print('「搜尋最大值」不是上界：超過它只支持「預期不可達」；'
          '要說「必不可達」必須超過**可認證上界**。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
