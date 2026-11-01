#!/usr/bin/env python3
"""緊湊持物修正版 v2（bottle_w1_compactcarry2_ctrl）的針對性離線核對（Codex 20261101_201934 兩項修正；假受控對象，非物理）。
0 參考速度連續：TCP 參考以 1 ms 取樣差分；收近、轉回 A、轉回 B 各段起訖速度 ≈ 0、相鄰取樣速度跳變 ≈ 0；同法量 v1 單段先升後伸的中點跳變（對照）。
  路徑時長：bottle_w1_compact_timing.json 的 max|dq/ds|、max|d²q/ds²| 代入 v2 各段時長 ⇒ 峰值關節速度 ≤ v_arm、加速度 ≤ a_arm。
1 淨空閘單元：非有限 q／T_rel／位姿 ⇒ 不放行；v1 名目閉迴路收近起點的狀態代入 v2 閘（底盤就地停住）⇒ 報餘裕（v1 單點閘是否不足）。
2 名目閉迴路 v2：完成且放置判準通過；收近不早於抓持與離桌判定、閘餘裕 ≥ 0；轉回完成早於 PARKED；持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限；
  收近起點之後到 LOWER 前，完整瓶身＋夾爪格點對第一桌（桌面、桌腳）三維距離 > 0；設定點速率峰 ≤ v_arm；緊湊穩態包絡水平半徑小於側持基線；閘單拍耗時。
3 停止路徑（收近開始後 0.5×T_to 起注入）：
  a 掃描不發布 ⇒ 既有 scan_stale 受控停止、停止有效；b 在途預測給非有限 ⇒ compact_clear_invalid；c 位姿集合加入抓取時刻的實測底盤位姿（＝退回第一桌旁；邏輯注入，實際底盤沒有後退）⇒ compact_table_clear_lost。
  三者皆須：停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、瓶身＋夾爪對第一桌 > 0、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm。
    python3 evaluation/test_bottle_w1_compactcarry2.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
import bottle_w1_compactcarry_ctrl as CC                # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_frontcarry as TF                  # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []
POSE = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))
TIMING = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_timing.json')))


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def ref_velocity(T_fn, t0, t1, dt=1e-3):
    ts = np.arange(t0, t1, dt)
    P = np.array([T_fn(t)[:3, 3] for t in ts])
    V = np.diff(P, axis=0) / dt
    return ts[1:], V


def case_velocity():
    T_side = np.array(POSE['reference']['T_side'])
    c2 = object.__new__(C2.W1PlaceController)
    c2.cc = dict(stage='to_compact', t0=1.0, t_ret=None, T_side=T_side)
    ts, V = ref_velocity(c2._cc_T, 0.5, 1.0 + C2.T_TO + 0.5)
    sp = np.linalg.norm(V, axis=1)
    out = dict(to_compact=dict(peak=float(sp.max()), jump=float(np.linalg.norm(np.diff(V, axis=0), axis=1).max()), v_end=float(sp[np.argmin(np.abs(ts - (1.0 + C2.T_TO)))])))
    c2.cc = dict(stage='return', t0=1.0, t_ret=1.0, T_side=T_side)
    ts, V = ref_velocity(c2._cc_T, 0.5, 1.0 + C2.T_UP + C2.T_OUT + 0.5)
    sp = np.linalg.norm(V, axis=1)
    k = np.argmin(np.abs(ts - (1.0 + C2.T_UP)))
    out['return'] = dict(peak_up=float(sp[ts <= 1.0 + C2.T_UP].max()), peak_out=float(sp[ts > 1.0 + C2.T_UP].max()), v_mid=float(sp[k]),
                         jump=float(np.linalg.norm(np.diff(V, axis=0), axis=1).max()), v_end=float(sp[np.argmin(np.abs(ts - (1.0 + C2.T_UP + C2.T_OUT)))]))
    c1 = object.__new__(CC.W1PlaceController)
    T_c = T_side.copy()
    T_c[:3, 3] = CC.TCP_C
    c1.cc = dict(t0=1.0, T_tr=CC.T_TR, T_a=T_c, T_b=T_side, path='raise_then_out')
    ts, V = ref_velocity(c1._cc_T, 0.5, 1.0 + CC.T_TR + 0.5)
    k = np.argmin(np.abs(ts - (1.0 + CC.T_TR / 2)))
    out['v1_return'] = dict(jump=float(np.linalg.norm(V[k + 1] - V[k - 1])), v_before=V[k - 1].tolist(), v_after=V[k + 1].tolist())
    print(f"   v2 收近：峰值 {out['to_compact']['peak']:.3f} m/s、相鄰跳變 {out['to_compact']['jump'] * 1e3:.3f} mm/s、終點速度 {out['to_compact']['v_end'] * 1e3:.3f} mm/s")
    print(f"   v2 轉回：升高峰值 {out['return']['peak_up']:.3f}、外伸峰值 {out['return']['peak_out']:.3f} m/s、交界速度 {out['return']['v_mid'] * 1e3:.3f} mm/s、相鄰跳變 {out['return']['jump'] * 1e3:.3f} mm/s、終點 {out['return']['v_end'] * 1e3:.3f} mm/s")
    print(f"   v1 轉回中點：速度 {np.round(out['v1_return']['v_before'], 3).tolist()} → {np.round(out['v1_return']['v_after'], 3).tolist()} m/s，向量跳變 {out['v1_return']['jump']:.3f} m/s")
    chk('0 v2 各段起訖速度 ≈ 0（< 1 mm/s）、相鄰 1 ms 取樣速度跳變 < 1 mm/s',
        max(out['to_compact']['v_end'], out['return']['v_mid'], out['return']['v_end']) < 1e-3 and max(out['to_compact']['jump'], out['return']['jump']) < 1e-3, out)
    lim = {}
    for nm, T_ in (('to_compact', C2.T_TO), ('return_raise', C2.T_UP), ('return_out', C2.T_OUT)):
        sg = TIMING[nm]
        v = sg['max_dq_ds'] * 1.875 / T_
        a = sg['max_dq_ds'] * 5.7735 / T_ ** 2 + sg['max_d2q_ds2'] * (1.875 / T_) ** 2
        lim[nm] = dict(T=T_, v_peak=v, a_peak=a)
    v1v = TIMING['to_compact']['max_dq_ds'] * 1.875 / CC.T_TR
    print('   沿路徑峰值關節速度／加速度：' + '；'.join(f"{k} T {v['T']:.2f} s → {v['v_peak']:.3f} rad/s、{v['a_peak']:.2f} rad/s²" for k, v in lim.items()) + f"（v_arm {B.W_ARM['v_arm']}、a_arm {B.W_ARM['a_arm']}；v1 收近 2.22 s → {v1v:.3f} rad/s）")
    chk('0 v2 各段沿路徑峰值關節速度 ≤ v_arm、加速度 ≤ a_arm', all(v['v_peak'] <= B.W_ARM['v_arm'] and v['a_peak'] <= B.W_ARM['a_arm'] for v in lim.values()), lim)
    out.update(joint_limits=lim, v1_to_compact_joint_peak=v1v)
    return out


def table1_clear(c, f, t0, t1):
    """瓶身＋夾爪格點（實測構型）對第一桌桌面與桌腳的三維最小距離，t0 ≤ t ≤ t1 且持瓶。"""
    Pt = c._pts_tcp()
    boxes = [(n, c_, sz) for n, c_, sz in G.table_boxes()]
    best = (9.9, None)
    for h in f.hist[::2]:
        if h['phase'] not in ('LIFT', 'CARRY') or h['t'] < t0 - 1e-9 or h['t'] > t1 + 1e-9:
            continue
        Tw = T.T.K.fk(h['q'], 'link_tcp')
        Pw = (Tw[:3, :3] @ Pt.T).T + Tw[:3, 3]
        d = min(float(G.box_dist(Pw, c_, sz).min()) for _, c_, sz in boxes)
        if d < best[0]:
            best = (d, round(float(h['t']), 2))
    return best


def run2(cut=None, inject=None):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = C2.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    st = dict(t_cut=None, last=None)

    def t_cut():
        if st['t_cut'] is None and c.cc is not None:
            st['t_cut'] = c.cc['t0'] + cut * C2.T_TO
        return st['t_cut']
    if inject == 'scan':
        fn0 = c.scan_fn

        def scan_cut(t, q):
            tc = t_cut()
            if tc is not None and t >= tc - 1e-9:
                return None
            st['last'] = float(t)
            return fn0(t, q)
        c.scan_fn = scan_cut
    elif inject == 'nonfinite':
        inf0 = c._inflight

        def inflight(q):
            qp, sp = inf0(q)
            tc = t_cut()
            if tc is not None and c.t >= tc - 1e-9 and c.cc is not None and c.cc['stage'] == 'to_compact':
                qp = qp.copy()
                qp[0] = np.nan
            return qp, sp
        c._inflight = inflight
    elif inject == 'backpose':
        po0 = c._poses

        def poses(q, t_rem):
            P = po0(q, t_rem)
            tc = t_cut()
            if tc is not None and c.t >= tc - 1e-9:
                h = min(f.hist, key=lambda h_: abs(h_['t'] - c.judge.t_grasp))
                P = np.vstack([P, np.asarray(h['q'], float)[:3]])
            return P
        c._poses = poses
    r, _ = T.run(c, f)
    return c, f, r, st


def gate_state_v1(c1, f1, r1, c2):
    """v1 名目收近起點的實測狀態代入 v2 閘幾何（底盤就地停住＝只用實測位姿）。"""
    ev = {e['what']: e['t'] for e in r1['events'] if e['what'] == 'compact_start'}
    t1 = ev.get('compact_start')
    h = min(f1.hist, key=lambda h_: abs(h_['t'] - t1))
    q = np.asarray(h['q'], float)
    Tw = T.T.K.fk(q, 'link_tcp')
    Pt = c2._pts_tcp()
    Pw = (Tw[:3, :3] @ Pt.T).T + Tw[:3, 3]
    m = C2.clear_margin(c2._sweep_base(np.asarray(c2.st.T_carry_rel, float)), q[None, :3], Pw)
    return dict(t=t1, base_x=float(q[0]), margin_stopped=m)


def main():
    res = dict(velocity=case_velocity())
    base = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compactcarry_offline.json')))['baseline']
    c1, f1, r1, _ = TC.run(CC.W1PlaceController)
    o1 = TC.summarize('v1（對照）', c1, f1, r1)
    c, f, r, _ = run2()
    o = TC.summarize('v2 名目', c, f, r)
    # 1 閘單元
    q_ok = np.asarray(f.hist[-1]['q'], float)
    P_ok = c._sweep_base(np.asarray(c.st.T_carry_rel, float))
    u = dict(nan_q=C2.clear_margin(P_ok, np.array([[np.nan, 0.0, 0.0]])), nan_pts=C2.clear_margin(P_ok * np.nan, q_ok[None, :3]))
    Tr0 = c.judge.T_rel                                   # 放置完成後 judge.T_rel 已清除；單元核對用設計參考的 T_rel
    c.judge.T_rel = np.array(POSE['reference']['T_rel']) * np.nan
    u['nan_T_rel'] = c._gate(q_ok, 1.0)[0]
    c.judge.T_rel = Tr0
    u['nan_q_gate'] = c._gate(q_ok * np.nan, 1.0)[0]
    chk('1 非有限 位姿／點／T_rel／q ⇒ 閘回 None（不放行）', all(v is None for v in u.values()), u)
    gv1 = gate_state_v1(c1, f1, r1, c)
    print(f"   v1 收近起點 t {gv1['t']}、底盤 x {gv1['base_x']:.3f}：代入 v2 閘（底盤就地停住）餘裕 {gv1['margin_stopped']:+.4f} m")
    res['gate_unit'] = dict(nonfinite=u, v1_start_state=gv1)
    # 2 名目
    g = r['compact_carry']['gate']
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    h_start = min(f.hist, key=lambda h_: abs(h_['t'] - ev['compact_start'])) if 'compact_start' in ev else None
    t1c = table1_clear(c, f, ev.get('compact_start', 1e9), pe.get('LOWER', 1e9))
    rate1 = o1['rate']
    print(f"   v2 收近起點底盤 x {h_start and round(float(h_start['q'][0]), 3)}（v1 {gv1['base_x']:.3f}）；閘：起點餘裕 {g['trigger'] and round(g['trigger']['margin'], 4)}、位姿數 {g['trigger'] and g['trigger']['n_poses']}、"
          f"收近中最小 {g['monitor_min'] and round(g['monitor_min'], 4)}；單拍耗時 平均 {g['ms_mean']:.2f}／最大 {g['ms_max']:.2f} ms（{g['n']} 拍）")
    print(f"   收近起點後到 LOWER：瓶身＋夾爪對第一桌三維最小 {t1c[0] * 1e3:.1f} mm @ {t1c[1]}；設定點速率峰 v2 {o['rate']:.3f}（v1 {rate1:.3f}）；轉回升高完成 {ev.get('compact_raised')}")
    chk('2 v2 完成且放置判準通過', o['done'] and bool((o['place'] or {}).get('ok')), (o['stop'], o['place']))
    chk('2 收近不早於抓持與離桌判定、閘起點餘裕 ≥ 0、收近中閘餘裕 ≥ 0；轉回完成早於 PARKED',
        o['cc'][0] is not None and o['cc'][0] >= max(o['t_grasp'], o['t_lift_ok']) - 1e-9 and g['trigger']['margin'] >= 0 and (g['monitor_min'] or 0.0) >= 0
        and o['cc'][3] is not None and o['parked'] is not None and o['cc'][3] <= o['parked'] + 1e-9, (o['cc'], o['parked'], g['trigger'], g['monitor_min']))
    chk('2 持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限', all(v[0] >= TL.REJ * 1e3 for v in o['held'].values()) and o['self_min_mm'] > 0 and o['forb'] == 0 and o['finite'], (o['held'], o['self_min_mm'], o['forb']))
    chk('2 收近起點後瓶身＋夾爪對第一桌 > 0', t1c[0] > 0, t1c)
    chk('2 設定點速率峰 ≤ v_arm', o['rate'] <= B.W_ARM['v_arm'] + 1e-6, o['rate'])
    cs = o['env'].get('compact', {})
    chk('2 緊湊穩態包絡（瓶身＋夾爪）水平半徑小於側持基線', cs.get('rad', 9.9) < base['env'].get('other_held', {}).get('rad', 0.0), (cs, base['env']))
    res.update(v1=o1, v2=o, v2_gate=dict({k: v for k, v in g.items() if k != 'log'}), v2_table1_clear=t1c, v2_start_base_x=h_start and float(h_start['q'][0]))
    # 3 停止路徑
    want = dict(scan=('scan_stale', 'shield_intervened'), nonfinite=('compact_clear_invalid',), backpose=('compact_table_clear_lost',))
    for inj, ok_why in want.items():
        c3, f3, r3, st = run2(cut=0.5, inject=inj)
        s3 = TC.summarize(f'收近中注入 {inj}', c3, f3, r3)
        t_low3 = next((e['t'] for e in r3['events'] if e['what'] == 'place_phase' and e.get('to') == 'LOWER'), 1e9)
        held3, tilt3, _, _ = TF.held_tables(f3, st['t_cut'] or 0.0, t_low3)
        t13 = table1_clear(c3, f3, st['t_cut'] or 0.0, t_low3)
        se = s3['stop_effect'] or {}
        print(f"      注入起 {st['t_cut']}；停止 {s3['stop'] and (round(s3['stop']['t'], 2), s3['stop']['why'])}；停止有效 {se.get('ok')} @ {se.get('t')}；注入後持瓶對桌 {held3}；對第一桌 {t13[0] * 1e3:.1f} mm；傾角 {tilt3:.2f}°")
        chk(f'3 {inj}：受控停止（{"/".join(ok_why)}）且停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、瓶身＋夾爪對第一桌 > 0、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm',
            (not s3['done']) and (s3['stop'] or {}).get('why') in ok_why and se.get('ok') is True and all(v[0] >= TL.REJ * 1e3 for v in held3.values()) and t13[0] > 0
            and s3['self_min_mm'] > 0 and s3['margin'] > 0 and s3['forb'] == 0 and s3['slip'][0] < 5.0, (s3['stop'], se, held3, t13))
        s3.update(t_cut=st['t_cut'], last_scan=st['last'], held_after=held3, table1_after=t13, tilt_after=tilt3)
        res[f'stop_{inj}'] = s3
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_compactcarry2_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
