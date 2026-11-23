#!/usr/bin/env python3
"""近距手腕視覺中斷的離線核對（Codex 20261101_134541；假受控對象，非物理）。凍結 SHROW（bottle_w1_shrow_ctrl）名目場景，控制器與期限不改。
注入：任務時間 CUT＝15.0 s 起，手腕估計在轉送層一律丟棄（記為 injected_cut，不呼叫 ctrl.wrist_intake）；底盤相機、掃描、Shield、控制照常。
核對：
1 只中斷手腕一路：CUT 後轉送給 wrist_intake 的筆數 0；底盤相機凍結照常（CUT 前已凍結）；掃描鏈與 Shield 照常（掃描未過期）。
2 沒有舊訊息刷新期限：控制器最後有效手腕影格的擷取時刻 ≤ CUT。
3 結果：未閉爪（無 close_request）；受控停止（原因與時刻照記：預期依既有規則 wrist_no_fix／wrist_stale）；停止有效；無禁止接觸；命令有限。
4 對照：同設定不注入時完成（確認注入以外相同）。
    python3 evaluation/test_bottle_w1_wristcut.py
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

CUT = 15.0
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

    def sink(t_cap, t_recv, e):
        if t_recv >= CUT - 1e-9:
            st['n_after'] += 1
        return sink0(t_cap, t_recv, e)
    lw.sink = sink

    def hook(t, s):
        for p_ in [p_ for p_ in st['pend'] if p_[0] <= t + 1e-9]:
            p_[1].ests[p_[2]] = (p_[0], p_[3])
            st['pend'].remove(p_)
        lb.tick(t)
        if cut and t >= CUT - 1e-9:                                  # 注入：CUT 起到達的手腕估計不轉送
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
                 wrist_lock=ev.get('wrist_lock'), w_last_valid=c.w_last_valid, n_fwd_after_cut=st['n_after'], n_cut=st['n_cut'], link=lw.report(), stop_effect=r['stop_effect'],
                 forb=max(v['peak'] for v in r['forbidden_pulses'].values()), finite=bool(T.finite_cmds(f)), shield=sh, scan_age_max=max((f_ for f_ in []), default=None), w1=dict(locked=w.get('locked'), n_groups=w.get('n_groups'), rejected=w.get('rejected')))
        res['cut' if cut else 'nocut'] = o
        print(f"   [{'注入' if cut else '不注入'}] 完成 {o['done']} 阻點 {o['stop']} 閉爪請求 {o['close_request']} 底盤凍結 {o['vision_freeze']} 手腕修正 {o['wrist_fix']} 鎖定 {o['wrist_lock']} "
              f"最後有效手腕擷取 {o['w_last_valid']}；CUT 後轉送 {o['n_fwd_after_cut']}、丟棄 {o['n_cut']}；停止有效 {o['stop_effect'] and o['stop_effect'].get('ok')}；Shield {sh['active']}／過期 {sh['stale']}")
        if not cut:
            chk('4 不注入：完成', o['done'], o['stop'])
        else:
            chk('1 只中斷手腕：CUT 後轉送 0、底盤凍結在 CUT 前、掃描未過期', o['n_fwd_after_cut'] == 0 and o['vision_freeze'] is not None and o['vision_freeze'] < CUT and sh['stale'] == 0, o)
            chk('2 無舊訊息刷新：最後有效手腕擷取 ≤ CUT', o['w_last_valid'] is None or o['w_last_valid'] <= CUT + 1e-9, o['w_last_valid'])
            chk('3 未閉爪、受控停止、停止有效、無禁止接觸、命令有限', o['close_request'] is None and o['stop'] is not None and (o['stop_effect'] or {}).get('ok') is True and o['forb'] == 0 and o['finite'],
                (o['close_request'], o['stop'], o['stop_effect']))
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_wristcut_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
