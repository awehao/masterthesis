#!/usr/bin/env python3
"""第一拍 Shield 一致列消融的最小核對（Codex 20261101_185350；只讀紀錄、非物理）。來源＝開啟組物理趟 bottle_w1_dyn1sr_trk1（有 scan_log）。
1 開啟時能重現：貼合拍與非貼合拍各取紀錄（s1_cbf），以該拍之前最近的實際掃描重建列，同輸入重解 ⇒ 回傳命令與紀錄命令差 ≤ 1e-6。
2 關閉時只少該組列：同輸入 u0_rows=None 與完全不給 u0_rows 參數逐位元相同；非貼合拍開／關命令差 ≤ 1e-9；貼合拍關閉後的命令違反影子列（> 1e-6）。
3 消融子類：_u0_rows 回 None（求解器不收列）、影子列照算。
    python3 evaluation/test_bottle_w1_shrow_off.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import bottle_w1_shrow_off_ctrl as OFF                  # noqa: E402
import test_bottle_w1_box2 as TB2                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402

RUN = 'bottle_w1_dyn1sr_trk1'
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def rows_at(t):
    S = np.load(os.path.join(HERE, 'runs', RUN, 'scan_log.npz'))
    j = np.flatnonzero(S['t'] <= t + 1e-9)[-1]
    sh = SR.make_shield()
    A, b, info = SR.shield_rows(sh, SR.scan_points(sh, S['ranges'][j]), t - float(S['t'][j]) + SR.AGE_PAD)
    return A, b, info


def main():
    TB2.RUN = RUN
    rep = json.load(open(os.path.join(HERE, 'runs', RUN, 's1_report.json')))
    u0 = rep['paired_strategy']['u0_rows']
    t_bind = u0['bind_first'][0][0]
    Zc = np.load(os.path.join(HERE, 'runs', RUN, 's1_cbf.npz'), allow_pickle=True)
    cfg = TB2.nav(None, 0.0)
    res = {}
    for nm, t in (('bind', t_bind), ('free', 20.0)):
        kw, _, tt, ph = TB2.record(t)
        i = int(np.argmin(np.abs(Zc['t'] - t)))
        A, b, info = rows_at(tt)
        on = GW.GMPC(cfg).solve(**kw, u0_rows=(A, b))
        off = GW.GMPC(cfg).solve(**kw, u0_rows=None)
        bare = GW.GMPC(cfg).solve(**kw)
        u_rec = np.asarray(Zc['u'][i][:3], float)
        e_on = float(np.abs(np.asarray(on.u_opt) - u_rec).max())
        d_onoff = float(np.abs(np.asarray(on.u_opt) - np.asarray(off.u_opt)).max())
        v_off = float(np.max(A @ np.asarray(off.u_opt)[:3] - b))
        same_bare = np.array_equal(np.asarray(off.u_opt), np.asarray(bare.u_opt))
        res[nm] = dict(t=tt, phase=ph, n_rows=info['n_rows'], err_on_vs_record=e_on, diff_on_off=d_onoff, off_violation=v_off, off_equals_bare=bool(same_bare),
                       u_on=np.asarray(on.u_opt).tolist(), u_off=np.asarray(off.u_opt).tolist())
        print(f"   {nm} {tt:.2f} s {ph}：列 {info['n_rows']}；開啟重解對紀錄差 {e_on:.1e}；開／關命令差 {d_onoff:.2e}；關閉命令違反影子列 {v_off:+.2e}；None 與不給參數逐位元相同 {same_bare}")
        chk(f'1 {nm}：開啟時重現紀錄命令（≤ 1e-6）', e_on <= 1e-6, e_on)
        chk(f'2 {nm}：u0_rows=None 與不給參數逐位元相同', same_bare)
        if nm == 'free':
            chk('2 非貼合拍：開／關命令差 ≤ 1e-9', d_onoff <= 1e-9, d_onoff)
        else:
            chk('2 貼合拍：關閉後命令違反影子列（> 1e-6），即列確實改變了命令', v_off > SR.TOL, v_off)
    c = OFF.W1PlaceController.__mro__
    chk('3 消融子類 _u0_rows 回 None（求解器不收列）、影子列照算', OFF.W1PlaceController._u0_rows is not SR.W1PlaceController._u0_rows and OFF.W1PlaceController.__bases__[0] is SR.W1PlaceController, c[:2])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_shrow_off_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
