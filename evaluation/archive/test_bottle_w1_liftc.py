#!/usr/bin/env python3
"""進 CARRY 時重規劃候選的最小離線驗證（Codex 20261101_125000）。非物理。
0 掛鉤退化：bottle_place_ctrl._carry_pending 預設 False ⇒ G、LIFTB 名目與加掛鉤前的離線結果（w1_liftb_offline.json）相同（完成時刻、切換、阻點）。
1 物理軌跡重放（只驗銜接幾何）：bottle_w1_dyn2l_g1 進 CARRY（21.9 s）的實測狀態當新請求起點；路徑取 (a) 起點到目標直線、
  (b) 該趟規劃器原始路徑平移到新起點（近似規劃器形狀，非重新規劃）；照原採用核對算 join 橫向、接段轉角、參考跳動、參考加速度。
2 名目、3 原第二案：假受控對象閉迴路（FakePlanner 下一週期即回覆）；完成或受控停止、等待時長、等待中與全程近停（底盤速 < 0.01 m/s，同主線表）、
  持瓶目標連續（等待期間 TCP 對底盤座標終點誤差）、Shield／CBF 照常、切換後持瓶↔桌 ≥ 5.5 mm、禁止接觸、命令有限。
    python3 evaluation/test_bottle_w1_liftc.py
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_path as NP                             # noqa: E402
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import bottle_w1_liftb_ctrl as LB                       # noqa: E402
import bottle_w1_liftc_ctrl as LC                       # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []
RUN = 'bottle_w1_dyn2l_g1'
NEAR_STOP = 0.01


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def case_regress():
    ref = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_liftb_offline.json')))
    out = {}
    for name, cls in (('G', W1.W1PlaceController), ('LIFTB', LB.W1PlaceController)):
        c, f, r = TL.run(cls, False)
        ev = {e['what']: e['t'] for e in r['events']}
        o = ref[f'nominal_{name}']
        sw = (r.get('paired_strategy') or {}).get('switch')
        same = ev.get('mission_done') == o['t_done'] and r['stop'] == o['stop'] and (sw or {}).get('t') == (o.get('switch') or {}).get('t')
        out[name] = dict(t_done=ev.get('mission_done'), ref=o['t_done'], same=same)
        chk(f'0 掛鉤預設關閉：{name} 名目同加掛鉤前（DONE {ev.get("mission_done")} vs {o["t_done"]}）', same, (r['stop'], sw))
    return out


def case_replay():
    D = os.path.join(HERE, 'runs', RUN)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    col = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t_c = next(e['t'] for e in r['events'] if e['what'] == 'phase' and e.get('to') == 'CARRY')
    k = int(np.argmin(np.abs(col['t'] - t_c)))
    q = np.array([col['base_x'][k], col['base_y'][k], col['base_yaw'][k]])
    v_b = float(col['v_b'][k])
    park = np.array(r['place']['park'], float)
    goal = np.array([park[0] - PC.PLACE['entry_back'], park[1]])
    raw0 = np.array(r['place']['carry']['plan']['raw'], float)
    paths = dict(straight=q[:2] + (goal - q[:2]) * np.linspace(0, 1, max(int(np.linalg.norm(goal - q[:2]) / 0.05), 2) + 1)[:, None],
                 planner_shifted=raw0 + (q[:2] - raw0[0]))
    R, out = PC.REG, {}
    for nm, P in paths.items():
        dw = np.array([np.cos(q[2]), np.sin(q[2])])
        cp, info = PC.build_carry(P, q[:2], dw, float(goal[0]), float(park[1]), R, v_max=0.25)
        if cp is None:
            out[nm] = info
            chk(f'1 重放（{nm}）：銜接幾何通過', False, info)
            continue
        csig = cp.project(q[:2], cp.sig0, back=0.0, ahead=0.3)
        res = {}
        for vb in (v_b, 0.0):
            D_ = vb * C0.D_STEPS * C0.PHYS
            jump = float(np.linalg.norm(cp.xy(csig + D_)[0] - np.array([q[0] + D_, q[1]])))
            sig_k, km = PC.curvature_grid(cp, R['kappa_win'])
            vcap = lambda s_, sk=sig_k, kk=km: min(0.25, np.sqrt(R['a_lat'] / max(float(np.interp(s_, sk, kk, right=0.0)), 1e-9)))   # noqa: E731
            ok_p, prof = PC.ref_profile_check(cp, csig, vb, float(park[0]), vcap, 0.25, R)
            res[f'v_b={vb:.3f}'] = dict(jump=jump, prof_ok=ok_p, prof={kk: round(v, 3) for kk, v in prof.items()})
        ok = info['lateral'] <= R['join_lat'] and all(x['jump'] <= R['ref_jump'] and x['prof_ok'] for x in res.values())
        out[nm] = dict(lateral=info['lateral'], conn_turn_deg=info['conn_turn_deg'], turn_max_deg=info['turn_max_deg'], checks=res)
        print(f"   重放 {nm}：起點 {np.round(q, 3).tolist()}（{t_c:.2f} s，v_b {v_b:.3f}）橫向 {info['lateral'] * 1e3:.1f} mm、接段轉角 {info['conn_turn_deg']:.2f}°、{res}")
        chk(f'1 重放（{nm}）：橫向 ≤ {R["join_lat"]}、參考跳動 ≤ {R["ref_jump"]}、參考加速度核對通過', ok, out[nm])
    return out


def near_stop(f, t0, t1):
    t = np.array([h['t'] for h in f.hist])
    Q = np.array([h['q'] for h in f.hist])
    sp = np.convolve(np.hypot(np.gradient(Q[:, 0], t), np.gradient(Q[:, 1], t)), np.ones(5) / 5, mode='same')
    m = (t >= t0) & (t <= t1)
    return float((sp[m] < NEAR_STOP).sum() * 0.01), float(sp[m].min()) if m.any() else None


def case_loop(dyn):
    tag = 'dyn2' if dyn else 'nominal'
    c, f, r = TL.run(LC.W1PlaceController, dyn)
    o = TL.summarize(f'{tag} LIFTC', c, f, r, dyn)
    ps = r['paired_strategy']
    cw = ps.get('carry_wait') or {}
    t_end = o['t_done'] if o['t_done'] is not None else (o['stop'] or {}).get('t', f.hist[-1]['t'])
    o['near_run_s'], _ = near_stop(f, 0.2, t_end)
    if cw.get('t_req') is not None:
        t1 = cw.get('t_accept') if cw.get('t_accept') is not None else t_end
        o['near_wait_s'], o['v_min_wait'] = near_stop(f, cw['t_req'], t1)
        H = [h for h in f.hist if cw['t_req'] - 1e-9 <= h['t'] <= t1 + 1e-9]
        T_fin = c.lb['T_fin'] if c.lb is not None else None
        if T_fin is not None and H:
            e = [BR.pose_err(c.K.fk(h['q'], 'link_tcp'), np.linalg.inv(BR.base_inv(h['q'][:3])) @ T_fin) for h in H]
            o['wait_tcp_err_max'] = [float(max(x[0] for x in e)), float(max(x[1] for x in e))]
    o['carry_wait'] = cw
    o['carry_req'] = r['place']['carry_req']
    o['shield'] = r['nav_chain']['shield']
    print(f"   [{tag} LIFTC] 等待 {cw}；近停 等待中 {o.get('near_wait_s')} s／全程 {o['near_run_s']:.2f} s；等待期間 TCP 對底盤座標終點最大誤差 {o.get('wait_tcp_err_max')}；Shield {o['shield']}")
    ha = o.get('held_after_switch_mm')
    held_ok = ha is None or all(v[0] >= TL.REJ * 1e3 for v in ha.values())
    chk(f'{2 if not dyn else 3} {tag}：完成或受控停止、命令有限、無禁止接觸、切換後持瓶↔桌 ≥ 5.5 mm、自碰 > 0、請求於 CARRY 送出且只一次',
        (o['done'] or o['stop'] is not None) and o['finite'] and o['forb'] == 0 and held_ok and o['self_min_mm'] > 0 and (o['carry_req'] or {}).get('at') == 'CARRY',
        (o['stop'], o['forb'], ha, o['carry_req']))
    if not dyn:
        chk('2 名目：候選完成', o['done'], o['stop'])
    return o


def main():
    res = dict(t_wait_max=LC.T_WAIT_MAX, costmap_max_age=NP.COSTMAP_MAX_AGE)
    res['regress'] = case_regress()
    res['replay'] = case_replay()
    res['nominal'] = case_loop(False)
    res['dyn2'] = case_loop(True)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_liftc_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
