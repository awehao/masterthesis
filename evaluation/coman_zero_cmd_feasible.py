"""**完整屏障**下，設計保持位姿是否允許零命令（離線核對，不開模擬器）。

指導意見要求的核對：用既有幾何資料、**距離節點實際輸出的 `d` 與 `rho`**，
套入完整屏障，確認設計保持位姿允許零命令。**不能只拿名目 27.9 mm 減 10 mm 判定。**

完整屏障一列（取樣點形式）：

    rhs = alpha * (d_eff - d_stop) - rho * |J_omega v| - velocity_error_margin
    d_eff  = d - rho                （stale 時改為 d - age * stale_speed）
    d_stop = g_pair + v_app * tau + v_app^2 / (2 a)       ← 本配對用總靜態間距
             d0     + v_app * tau + v_app^2 / (2 a) + eps ← 其餘配對照一般規則

零命令 u = 0 ⇒ 每列左式為 0，可行的條件是**每列 rhs >= 0**。
v = 0 使 v_app、|J_omega v| 皆為 0，但 `rho` 扣除與 `velocity_error_margin`
**照常生效** —— 這正是名目減法會漏掉的部分。

取樣、`rho`、障礙物解析與列的產生一律呼叫節點本身的程式碼，不另寫一份。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                     # noqa
    expand_pair_rows, forced_pair_rows, pair_distance_bound, parse_obstacles)
from ammr_wholebody_mpc.arm_link_geometry import link_collision_tris   # noqa
from ammr_wholebody_mpc.arm_link_geometry import (                     # noqa
    arm_link_names, obstacle_distances, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from ammr_wholebody_mpc.wholebody_safety_filter import (               # noqa
    DetectionPoint, SafetyConfig, STATUS_OK, _box_rows, _joint_limit_rows,
    _rows_from_points, validate_pair_config)

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
RUN = 'drawer_220102_offset20'          # 既有 20 mm 趟次；**不重跑**
# **間距一律讀定版規格**，不在此另存一份常數（避免兩處不一致）
def _load_gaps():
    import yaml
    d = yaml.safe_load(open(os.path.join(
        HERE, 'results/specs/coman_r1_pair_gaps_proposal.yaml'),
        encoding='utf-8'))
    if d['status'] != 'approved':
        raise SystemExit(f'R1 未核准（status={d["status"]}），不執行核對')
    return {p['pair']: float(p['g_pair_m']) for p in d['pairs']}


PAIR_GAP = _load_gaps()
CONTACT = {'uflite_finger1|handle_bar': ['engage', 'pull', 'hold', 'release'],
           'uflite_finger2|handle_bar': ['engage', 'pull', 'hold', 'release']}
PAIR_ROWS = sorted({k.split('|')[0] + ':*' for k in PAIR_GAP})
EXEMPT = ['uflite_finger1:handle_bar', 'uflite_finger2:handle_bar']


def hold_state():
    """既有趟次的 20 mm 保持狀態：底盤停放位姿、六軸角度、抽屜開度。"""
    d = json.load(open(os.path.join(HERE, 'runs', RUN, 'sim',
                                    'drawer_run.json'), encoding='utf-8'))
    ix = {k: j for j, k in enumerate(d['log_cols'])}
    rows = [r for r in d['log'] if str(r[ix['phase']]) == 'hold']
    if not rows:
        raise SystemExit('該趟沒有 hold 相位')
    r = rows[len(rows) // 2]            # 保持段中點
    q = np.array([float(r[ix[f'joint{i}']]) for i in range(1, 7)])
    return dict(park=list(map(float, d['park'])), unit=list(map(float, d['pose'])),
                q_arm=q, opening=float(r[ix['opening']]),
                drawer_y0=float(d['drawer_y0']), sim_t=float(r[ix['t']]))


def main() -> int:
    gaps = dict(PAIR_GAP)
    print(f'間距來源：coman_r1_pair_gaps_proposal.yaml（approved，{len(gaps)} 組）')
    st = hold_state()
    print(f'資料來源：既有趟次 {RUN} 的 hold 相位（sim {st["sim_t"]:.2f} s，'
          f'開度 {st["opening"]*1000:.2f} mm）—— **未重跑任何趟次**')

    K = WholeBodyKinematics.from_urdf_file(URDF)
    n = len(K.dof_names)
    q = np.zeros(n)
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']

    specs = subprocess.run([sys.executable,
                            os.path.join(HERE, 'coman_obstacle_specs.py'),
                            str(st['unit'][0]), str(st['unit'][1])],
                           capture_output=True, text=True, check=True)
    obs = parse_obstacles([x for x in specs.stdout.split() if x.strip()])
    names = [o.name for o in obs]
    # drawer_body 的世界位姿＝抽屜沿滑動軸位移（開度）；櫃體已是靜態世界位姿
    T_db = np.eye(4)
    T_db[0, 3] = st['unit'][0]
    T_db[1, 3] = st['drawer_y0'] - st['opening']
    for o in obs:
        if o.model == 'drawer_body':
            o.T_world_link = T_db
    print(f'障礙物 {len(obs)} 個；drawer_body 位於 y = {T_db[1,3]:.5f} m')

    samples = sample_links_certified(open(URDF, encoding='utf-8').read(),
                                     rho_target=0.015, tol=0.001)
    link_names = arm_link_names(open(URDF, encoding='utf-8').read())
    print('取樣（節點同一份、rho 經認證）：'
          + '、'.join(f'{k} {len(v.points)}pt rho {v.rho*1000:.1f}mm'
                      for k, v in samples.items()))

    pairs, exempt = expand_pair_rows(PAIR_ROWS, EXEMPT, names)
    # R2：殼—橫桿的精確下界，每次由當前位姿重算（與節點同一支函式）
    _xml = open(URDF, encoding='utf-8').read()
    _tt = link_collision_tris(_xml, links=['uflite_gripper_link'])
    _Tw = K.fk(q, 'uflite_gripper_link')
    _tris = (_tt['uflite_gripper_link'].reshape(-1, 3) @ _Tw[:3, :3].T
             + _Tw[:3, 3]).reshape(-1, 3, 3)
    _bnd = pair_distance_bound(_tris, obs[names.index('handle_bar')],
                               tol=5.0e-4, max_tris=60000, budget_s=0.020)
    TIGHT = ({('uflite_gripper_link', 'handle_bar'): float(_bnd['lb'])}
             if np.isfinite(_bnd['lb']) else {})
    print(f'精確下界（殼—橫桿）：{_bnd["lb"]*1000:.4f} mm '
          f'（上界 {_bnd["ub"]*1000:.4f}、耗時 {_bnd["elapsed_s"]*1000:.1f} ms）')

    # ---- 依節點的做法產生列：一般帶狀列（最近障礙物）＋必要配對列 ----
    rows = []
    for li, name in enumerate(link_names):
        T = K.fk(q, name)
        S = samples[name]
        W = (S.points @ T[:3, :3].T) + T[:3, 3]
        d, v, which = obstacle_distances(W, obs)
        dmin = float(d.min())
        sel = np.nonzero(d <= dmin + S.rho)[0]
        sel = sel[np.argsort(d[sel])]
        for k in sel:
            nh = v[k] / max(abs(float(d[k])), 1e-9)
            rows.append(list(W[k]) + list(nh)
                        + [float(d[k]), float(STATUS_OK), 0.0, 0.0, float(li)]
                        + list(S.points[k]) + [float(S.rho),
                                               float(names.index(which[k]))])
        for obn in pairs.get(name, []):
            oi = names.index(obn)
            rows.extend(forced_pair_rows(W, S.points, obs[oi], oi, li, S.rho,
                                         STATUS_OK, 0.0, lambda p, vv: 0.0,
                                         3.0, 0))

    pts = [DetectionPoint(link_names[int(r[10])], np.array(r[0:3]),
                          np.array(r[3:6]), float(r[6]), int(r[7]),
                          float(r[8]), r[9] >= 0.5,
                          offset=np.array(r[11:14]), rho=float(r[14]),
                          obs=names[int(r[15])],
                          d_lb=TIGHT.get((link_names[int(r[10])],
                                          names[int(r[15])])))
           for r in rows]
    print(f'屏障輸入列 {len(pts)} 條')

    cfg = SafetyConfig(g_by_pair=dict(gaps),
                       contact_pairs={k: list(v) for k, v in CONTACT.items()})
    cfg.phase = 'hold'
    cfg.freespace_confirmed = False
    validate_pair_config(cfg)
    print(f'配置：一般 d0 {cfg.d0*1000:.0f} mm + eps {cfg.eps*1000:.0f} mm；'
          f'兩指—面板改用總靜態間距 {gaps["uflite_finger1|drawer_front_panel"]*1000:.0f} mm'
          f'（取代 d0+eps）；alpha {cfg.alpha}、'
          f'velocity_error_margin {cfg.velocity_error_margin}')

    A, b, cap, owner = _rows_from_points(K, q, pts, cfg, np.zeros(n))
    A = np.array(A) if len(A) else np.zeros((0, n))
    b = np.array(b) if len(b) else np.zeros(0)
    Aj, bj = _joint_limit_rows(K, q, cfg)
    Ax, bx = _box_rows(cfg, n, cap, None, cfg.dt)

    print(f'\n屏障列 {len(b)} 條、關節限位列 {len(bj)} 條、框列 {len(bx)} 條')
    print(f'零命令下的接觸例外免列數：{getattr(cfg, "last_contact_skipped", 0)}')

    bad = []
    for label, Am, bm in (('屏障', A, b), ('關節限位', np.array(Aj), np.array(bj)),
                          ('速度／加速度框', np.array(Ax), np.array(bx))):
        if len(bm) == 0:
            print(f'{label}：無列')
            continue
        resid = bm - (Am @ np.zeros(n))          # u = 0
        worst = int(np.argmin(resid))
        ok = bool(np.all(resid >= 0.0))
        print(f'{label}：最小餘量 {resid[worst]:+.6f}（列 {worst}）'
              f'  {"零命令可行" if ok else "**零命令不可行**"}')
        if not ok:
            bad.append((label, resid, Am, bm))

    if len(b):
        by_pair = {}
        for i, pi in enumerate(owner):
            p_ = pts[pi]
            key = f'{p_.frame}|{p_.obs}'
            r_ = float(b[i])
            if key not in by_pair or r_ < by_pair[key][0]:
                by_pair[key] = (r_, float(p_.d), float(p_.rho),
                                None if p_.d_lb is None else float(p_.d_lb))
        print('\n最緊的十個配對（餘量由小到大；d 與 rho 為距離節點的實際輸出）。')
        print('「可行上限」= **實際生效的距離項**（有精確下界者用下界，'
              '否則 d − rho），即零速下該列要 rhs >= 0 所需的總靜態間距上限。')
        print('餘量單位是 rhs 本身（alpha x 距離，m/s）；「餘量距離」= 餘量 / alpha。')
        print(f'  {"配對":42s}{"餘量":>10s}{"餘量距離":>10s}{"d":>9s}{"rho":>8s}'
              f'{"可行上限":>10s}{"現行門檻":>10s}')
        print(f'  {"":42s}{"":>10s}{"(mm)":>10s}{"(mm)":>9s}{"(mm)":>8s}'
              f'{"(mm)":>10s}{"(mm)":>10s}')
        for key, (r_, d_, rho_, lb_) in sorted(by_pair.items(),
                                               key=lambda kv: kv[1][0])[:10]:
            lim = (lb_ if lb_ is not None else d_ - rho_) * 1000.0
            cur = (cfg.g_by_pair.get(key, cfg.d0 + cfg.eps)) * 1000.0
            print(f'  {key:42s}{r_:+10.5f}{1000.0*r_/cfg.alpha:+10.2f}'
                  f'{d_*1000:9.2f}{rho_*1000:8.2f}{lim:10.2f}{cur:10.1f}'
                  f'{"  ←精確下界" if lb_ is not None else ""}')

    if bad:
        print('\n**零命令不可行。** 造成的列：')
        for label, resid, Am, bm in bad:
            for i in np.argsort(resid)[:5]:
                if resid[i] < 0:
                    src = ''
                    if label == '屏障':
                        p_ = pts[owner[i]]
                        src = (f'  ← {p_.frame}|{p_.obs}'
                               f'  d {p_.d*1000:.2f} mm、rho {p_.rho*1000:.2f} mm')
                    print(f'  {label} 列 {i}：餘量 {resid[i]:+.6f}{src}')
        print('\n**不再逐項刪除餘裕**；以上即為造成不相容的項目。')
        return 1
    print('\n設計保持位姿在完整屏障下**允許零命令**（所有列餘量 >= 0）。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
