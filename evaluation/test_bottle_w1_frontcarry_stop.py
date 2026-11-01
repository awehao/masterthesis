#!/usr/bin/env python3
"""前持過渡中掃描中斷 ⇒ 受控停止的最小離線核對（Codex 20261101_193038；假受控對象，非物理）。
注入規則（事前固定）：前持開始（front_carry_start）後 0.5×T_tr 起，掃描函式回傳 None（該幀掃描沒發布）；不關 Shield、不改狀態、不用手腕期限。
預期：既有 scan_stale（掃描年齡 > OBS_MAX_AGE 0.5 s）或 Shield 過期限速觸發受控停止；記錄最後掃描、注入、停止、停穩時刻。
核對：未完成任務、受控停止且停止有效；注入後到結束，持瓶（LIFT 起剛性附著）對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、命令有限；
瓶軸傾角最大、滑移、抓持保持（停止後瓶相對 TCP 不脫落：滑移 < 5 mm）。
    python3 evaluation/test_bottle_w1_frontcarry_stop.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_w1_frontcarry_ctrl as FC                  # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_frontcarry as TF                  # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []
FRAC = 0.5


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def main():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = FC.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    st = dict(t_cut=None, last_scan=None, n_drop=0)
    fn0 = c.scan_fn

    def scan_cut(t, q):
        if st['t_cut'] is None and c.fc is not None:
            st['t_cut'] = c.fc['t0'] + FRAC * c.fc['T_tr']
        if st['t_cut'] is not None and t >= st['t_cut'] - 1e-9:
            st['n_drop'] += 1
            return None
        st['last_scan'] = float(t)
        return fn0(t, q)
    c.scan_fn = scan_cut
    r, _ = T.run(c, f)
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    mins, sm, ns, rate, marg = PCK.check(f)
    held, tilt, fwd, side = TF.held_tables(f, st['t_cut'] or 0.0)
    stp = r['stop'] or {}
    se = r.get('stop_effect') or {}
    o = dict(t_cut=st['t_cut'], last_scan=st['last_scan'], n_drop=st['n_drop'], fc_start=ev.get('front_carry_start'), T_tr=(c.fc or {}).get('T_tr'), stop=stp, stop_effect=se,
             done=bool(r['mission_done']), held_after_cut=held, tilt_max=tilt, self_min_mm=round(sm[0] * 1e3, 2), joint_margin=marg, forb=max(v['peak'] for v in r['forbidden_pulses'].values()),
             finite=bool(T.finite_cmds(f)), slip=r['slip_max_mm_deg'], shield=r['nav_chain']['shield'], phi_at_stop_deg=None)
    if c.fc is not None and stp.get('t') is not None:
        o['phi_at_stop_deg'] = round(math.degrees(c._fc_phi(stp['t'])), 2)
    print(f"   前持開始 {o['fc_start']}、T_tr {o['T_tr']:.3f} s ⇒ 中斷起 {o['t_cut']:.3f}（最後掃描 {o['last_scan']}，丟 {o['n_drop']} 幀）；停止 {stp.get('t')} {stp.get('why')}（當時目標轉角 {o['phi_at_stop_deg']}°）；"
          f"停止有效 {se.get('ok')} @ {se.get('t')}；Shield {o['shield']}")
    print(f"   注入後持瓶對桌 {held}；傾角最大 {tilt:.2f}°；自碰 {o['self_min_mm']} mm；關節餘裕 {marg:.3f} rad；禁止接觸 {o['forb']}；滑移 {np.round(o['slip'], 2).tolist()}")
    chk('受控停止（未完成任務）、停止有效、命令有限', (not o['done']) and stp.get('why') is not None and se.get('ok') is True and o['finite'], (stp, se))
    chk('停止原因來自掃描過期（scan_stale 或 Shield 過期限速造成的改寫）', stp.get('why') in ('scan_stale', 'shield_intervened'), stp.get('why'))
    chk('注入後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm', all(v[0] >= TL.REJ * 1e3 for v in held.values()) and o['self_min_mm'] > 0 and marg > 0 and o['forb'] == 0
        and o['slip'][0] < 5.0, (held, o['self_min_mm'], marg, o['forb'], o['slip']))
    json.dump(o, open(os.path.join(HERE, 'results', 'bottle', 'w1_frontcarry_stop_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
