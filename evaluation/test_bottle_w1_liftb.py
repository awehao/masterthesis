#!/usr/bin/env python3
"""分段抬升候選的最小安全核對（Codex 20261101_123440；假受控對象，非物理）。只做名目與原第二案，不換障礙劇本。
每案：完成或停止、切換時刻與間隙、抬升進度（切換後 TCP 對底盤座標終點的誤差）、手臂設定點速率達上限的時間、
淨空（bottle_s0b_path_check：含持瓶↔桌、非手指↔瓶、自碰）、底盤中心到障礙表面、耗時；與凍結 G 同案對照。
安全核對（通過才物理實跑第二案）：完成或受控停止、命令有限；持瓶↔桌（held）淨空 ≥ 拒絕線 5.5 mm；自碰最小距離 > 0；無禁止接觸。
    python3 evaluation/test_bottle_w1_liftb.py
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_liftb_ctrl as LB                       # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402

BAD = []
REJ = S1.CLEAR_REJECT + S1.CLEAR_GUARD


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(cls, dyn):
    holder = {}

    class Cap(cls):
        def report(self):
            r = super().report()
            holder['r'] = r
            holder['c'] = self
            return r
    orig, orig_ml = T.W1.W1PlaceController, TD.SC.make_lidar
    T.W1.W1PlaceController = Cap
    if dyn:
        def scan_fn(t, q):
            r = TD.NC.ray_boxes(q[:2], float(q[2]), TD.SC.scene_boxes() + [(c_, sz) for n, c_, sz in TD.TP.table2_boxes() if n != 'table_top'], TD.G.LIDAR_Z)
            oc, _ = DC.state(t, TD.CA)
            return np.minimum(r, DC.ray_cyl(q[:2], TD.NC.scan_angles() + float(q[2]), oc, TD.CA['radius'], TD.NC.RANGE_MAX))
        TD.SC.make_lidar = lambda **kw: orig_ml(scan_fn=scan_fn, **kw)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, TD.SC.make_lidar = orig, orig_ml
    r, _ = T.run(c, f)
    return c, f, r


def held_after(f, t0):
    """持瓶↔桌（桌面、桌腳）只算 t ≥ t0：瓶以第一筆 LIFT 時刻的 TCP 相對位姿剛性附著（同 PCK.check，瓶底在桌面）。
    PCK.check 從 LIFT 起算，瓶尚貼桌時本來就是 0（G 也一樣）；切換判準要看的是切換後。"""
    boxes, BP, rel, best = PCK.G.table_boxes(), PCK.G.bottle_points(), None, {}
    for h in f.hist:
        if h['phase'] not in ('LIFT', 'CARRY'):
            continue
        Tt = PCK.T.K.fk(h['q'], 'link_tcp')
        if rel is None:
            T0 = np.eye(4)
            T0[:3, 3] = [PCK.T.BX, PCK.T.BY, PCK.T.TOP]
            rel = np.linalg.inv(Tt) @ T0
        if h['t'] < t0 - 1e-9:
            continue
        Tb = Tt @ rel
        Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
        for nm, c_, sz in boxes:
            k = 'table_top' if nm == 'table_top' else 'leg'
            d = float(PCK.G.box_dist(Pw, c_, sz).min())
            if k not in best or d < best[k][0]:
                best[k] = (d, round(float(h['t']), 2), h['phase'])
    return {k: (round(v[0] * 1e3, 2), v[1], v[2]) for k, v in best.items()}


def summarize(name, c, f, r, dyn):
    ev = {e['what']: e['t'] for e in r['events']}
    mins, sm, ns, rate, marg = PCK.check(f)
    held = {k: round(v[0] * 1e3, 2) for k, v in mins.items() if 'held' in k}
    nfb = {k: round(v[0] * 1e3, 2) for k, v in mins.items() if 'nonfinger↔bottle' in k}
    t = np.array([h['t'] for h in f.hist])
    S = np.array([h['s'] for h in f.hist])
    k = np.flatnonzero(np.abs(t * 100 - np.round(t * 100)) < 1e-6)
    rate_s = np.abs(np.diff(S[k], axis=0)) / np.diff(t[k])[:, None]
    sat_s = float((rate_s.max(axis=1) >= 0.99 * B.W_ARM['v_arm']).sum() * 0.01)
    out = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), t_lift=r.get('t_lift_start'), t_lift_ok=r.get('t_lift_ok'), t_carry=ev.get('place_phase'),
               held_min_mm=held, nonfinger_bottle_mm=nfb, self_min_mm=round(sm[0] * 1e3, 2), arm_sat_s=sat_s, finite=bool(T.finite_cmds(f)),
               forb=max(v['peak'] for v in r['forbidden_pulses'].values()), slip=r['slip_max_mm_deg'], solve_p95=float(np.percentile(c.solve_ms, 95)) if c.solve_ms else None)
    if dyn:
        Q = np.array([h['q'] for h in f.hist])
        ob = np.array([DC.state(tt, TD.CA)[0] for tt in t])
        out['base_surface_min'] = float((np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - TD.CA['radius']).min())
    ps = r.get('paired_strategy') or {}
    if ps.get('switch'):
        out['switch'] = ps['switch']
        out['lb_err_tail'] = ps.get('lb_err_log', [])[-6:]
        out['held_after_switch_mm'] = held_after(f, ps['switch']['t'])
    elif out['t_lift_ok'] is not None:
        out['held_after_liftoff_mm'] = held_after(f, out['t_lift_ok'])
    print(f"   [{name}] 完成 {out['done']} 阻點 {out['stop'] and out['stop'].get('why')} DONE {out['t_done']} 抬升 {out['t_lift']}→離桌成立 {out['t_lift_ok']} 切換 {out.get('switch') and {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['switch'].items()}} "
          f"持瓶↔桌 {held} 非手指↔瓶 {nfb} 自碰 {out['self_min_mm']} mm 手臂飽和 {sat_s:.2f} s 禁止接觸 {out['forb']} 滑移 {np.round(out['slip'], 2).tolist()} 底盤到障礙 {out.get('base_surface_min')}", flush=True)
    return out


def main():
    res = {}
    for dyn in (False, True):
        tag = 'dyn2' if dyn else 'nominal'
        for name, cls in (('G', W1.W1PlaceController), ('LIFTB', LB.W1PlaceController)):
            c, f, r = run(cls, dyn)
            res[f'{tag}_{name}'] = summarize(f'{tag} {name}', c, f, r, dyn)
        o = res[f'{tag}_LIFTB']
        ha = o.get('held_after_switch_mm')
        print(f"   [{tag} LIFTB] 切換後持瓶↔桌 {ha}；[{tag} G] 離桌成立後 {res[f'{tag}_G'].get('held_after_liftoff_mm')}")
        held_ok = ha is None or all(v[0] >= REJ * 1e3 for v in ha.values())
        chk(f'{tag}：候選完成或受控停止、命令有限、無禁止接觸、切換後持瓶↔桌 ≥ 5.5 mm、自碰 > 0', (o['done'] or o['stop'] is not None) and o['finite'] and o['forb'] == 0 and held_ok and o['self_min_mm'] > 0,
            (o['stop'], o['forb'], ha, o['self_min_mm']))
    json.dump(dict(g_switch=LB.G_SWITCH, **res), open(os.path.join(HERE, 'results', 'bottle', 'w1_liftb_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
