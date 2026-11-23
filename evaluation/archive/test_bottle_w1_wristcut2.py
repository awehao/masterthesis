#!/usr/bin/env python3
"""手腕估計轉送中斷（形成量測後）的離線核對（test_bottle_w1_wristcut.py 的版本化複本，Codex 20261101_135333；假受控對象，非物理）。凍結 SHROW 名目場景，控制器與期限不改。
注入規則（事前固定）：本趟首次形成有效手腕量測組（ctrl.w_groups 第一筆）後的下一個控制週期（0.05 s 邊界）起，手腕估計在轉送層一律丟棄；
不清緩衝、不改狀態。底盤相機、掃描、Shield、控制照常。預期依既有規則 wrist_stale（鎖定前最近有效影格逾 0.5 s）、未閉爪。
核對：
1 只中斷手腕一路：CUT 後轉送給 wrist_intake 的筆數 0；底盤相機凍結照常（CUT 前已凍結）；掃描鏈與 Shield 照常（掃描未過期）。
2 沒有舊訊息刷新期限：控制器最後有效手腕影格的擷取時刻 ≤ CUT。
3 結果：未閉爪（無 close_request）；受控停止（原因與時刻照記：預期依既有規則 wrist_no_fix／wrist_stale）；停止有效；無禁止接觸；命令有限。
4 對照：同設定不注入時完成（確認注入以外相同）。
    python3 evaluation/test_bottle_w1_wristcut2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_v2_link as VL                             # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import test_bottle_s0b as T0                            # noqa: E402
import test_bottle_v2 as TV2                            # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

CUT_DT = 0.05
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(cut):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = SR.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    capb, capw = TV2.FakeCap(), TV2.FakeCap()
    lb, lw = VL.V2Link(capb, c), VL.V2Link(capw, c, sink=c.wrist_intake)
    st = dict(nb=0.0, nw=0.0, pend=[], n_after=0, n_cut=0)
    sink0 = c.wrist_intake

    st.update(t_fix=None, t_cut=None)

    def sink(t_cap, t_recv, e):
        if st['t_cut'] is not None and t_recv >= st['t_cut'] - 1e-9:
            st['n_after'] += 1
        return sink0(t_cap, t_recv, e)
    lw.sink = sink

    def hook(t, s):
        for p_ in [p_ for p_ in st['pend'] if p_[0] <= t + 1e-9]:
            p_[1].ests[p_[2]] = (p_[0], p_[3])
            st['pend'].remove(p_)
        lb.tick(t)
        if cut and st['t_fix'] is None and c.w_groups:                # 首次形成量測組 ⇒ 下一個控制週期邊界起中斷
            st['t_fix'] = float(c.w_groups[0]['t'])
            st['t_cut'] = (np.floor(t / CUT_DT + 1e-9) + 1) * CUT_DT
        if cut and st['t_cut'] is not None and t >= st['t_cut'] - 1e-9:   # 注入：此後到達的手腕估計不轉送
            for k in sorted(k for k in capw.ests if k not in lw.seen):
                lw.seen.add(k)
                lw._rej(k, 'injected_cut')
                st['n_cut'] += 1
        else:
            lw.tick(t)
        ph = s.c.st.phase
        if t >= st['nb'] - 1e-9 and ph == 'CRUISE':
            st['nb'] = t + 0.2
            ok, cam = TV2.visible(s.q)
            if ok:
                xy = np.array([T0.BX, T0.BY]) + np.array(TV2.BIAS_R2) + rng.normal(0, TV2.NOISE_R2)
                k = round(t, 6)
                capb.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                st['pend'].append((t + 0.02, capb, k, T.msg(t, xy, cam)))
        if t >= st['nw'] - 1e-9 and ph in ('CRUISE', 'PREGRASP', 'INSERT'):
            st['nw'] = t + 0.2
            cam = (G.C.K.fk(s.q, 'link_eef') @ T.T_WR)[:3, 3]
            if float(np.linalg.norm(np.array([T0.BX, T0.BY, T0.TOP + 0.1]) - cam)) <= 0.35:
                xy = np.array([T0.BX, T0.BY]) + np.array(T.W_BIAS) + rng.normal(0, T.W_NOISE)
                k = round(t, 6)
                capw.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                st['pend'].append((t + 0.05, capw, k, T.msg(t, xy, cam)))
        if t >= 72.0 and not s.c.mission_done:
            s.c._trigger_stop('sim_limit')
        return {}
    f.run(75.0, hook, tail=2.6)
    return c, f, c.report(), st, lw


def main():
    res = {}
    for cut in (False, True):
        c, f, r, st, lw = run(cut)
        ev = {e['what']: e['t'] for e in r['events']}
        w = r['w1']
        t_stop = (r['stop'] or {}).get('t')
        sh = r['nav_chain']['shield']
        o = dict(cut=cut, done=bool(r['mission_done']), stop=r['stop'], close_request=ev.get('close_request'), vision_freeze=ev.get('vision_freeze'), wrist_fix=ev.get('wrist_fix'),
                 wrist_lock=ev.get('wrist_lock'), w_last_valid=c.w_last_valid, t_fix=st['t_fix'], t_cut=st['t_cut'], n_buf=len(c.wbuf), n_fwd_after_cut=st['n_after'], n_cut=st['n_cut'], link=lw.report(), stop_effect=r['stop_effect'],
                 forb=max(v['peak'] for v in r['forbidden_pulses'].values()), finite=bool(T.finite_cmds(f)), shield=sh, scan_age_max=max((f_ for f_ in []), default=None), w1=dict(locked=w.get('locked'), n_groups=w.get('n_groups'), rejected=w.get('rejected')))
        res['cut' if cut else 'nocut'] = o
        print(f"   [{'注入' if cut else '不注入'}] 完成 {o['done']} 阻點 {o['stop']} 閉爪請求 {o['close_request']} 底盤凍結 {o['vision_freeze']} 手腕修正 {o['wrist_fix']} 鎖定 {o['wrist_lock']} "
              f"最後有效手腕擷取 {o['w_last_valid']}；CUT 後轉送 {o['n_fwd_after_cut']}、丟棄 {o['n_cut']}；停止有效 {o['stop_effect'] and o['stop_effect'].get('ok')}；Shield {sh['active']}／過期 {sh['stale']}")
        if not cut:
            chk('4 不注入：完成', o['done'], o['stop'])
        else:
            print(f"   形成量測 {o['t_fix']}、中斷起 {o['t_cut']}、緩衝 {o['n_buf']} 筆（保留）")
            chk('1 只中斷手腕：中斷後轉送 0、底盤凍結在中斷前、掃描未過期', o['t_cut'] is not None and o['n_fwd_after_cut'] == 0 and o['vision_freeze'] is not None and o['vision_freeze'] < o['t_cut'] and sh['stale'] == 0, o)
            chk('2 無舊訊息刷新：最後有效手腕擷取 ≤ 中斷起點', o['w_last_valid'] is not None and o['w_last_valid'] <= o['t_cut'] + 1e-9, o['w_last_valid'])
            chk('3 經 wrist_stale 受控停止、未閉爪、停止有效、無禁止接觸、命令有限', o['close_request'] is None and (o['stop'] or {}).get('why') == 'wrist_stale' and (o['stop_effect'] or {}).get('ok') is True and o['forb'] == 0 and o['finite'],
                (o['close_request'], o['stop'], o['stop_effect']))
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_wristcut2_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
