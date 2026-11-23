#!/usr/bin/env python3
"""60° 版換抓參考包（Howard 10/30「門開大一點」；Codex 20261030_001949：後退開門約 60°，換抓／繞行／內側接近／拉關起點不能沿用 48°）。
流程＝door_regrasp_pb1f 的 60° 版（演算法、判準、STOW_F 全同；只換門角與輸出檔名）：
  P1–P4（door_regrasp_pb1）：起點＝door_wide_w1f_60_3 門角最接近 Q0 的實測物理步；Q0＝59.888663°（＝60 − 0.111337，比照 48° 版目標 48 與 q_ref 47.888663 的差）；
     重規劃 replan_v8.run_variant(φ＋30°) 的 B1＝48° 版 B1 (0.65, −0.05, 270°) **繞鉸鏈（原點）剛體轉 −ΔQ**（ΔQ＝Q0 − 47.888663°；與執行期 mode 5 門角修正同一原則：底盤相對內側把手不變）；
  P5 關門規劃：door_regrasp_close_plan_v1 的演算法，起點＝本包 v4 mode 5 終點、門角＝本包 q_ref；P6 關門時間化：door_regrasp_close_retime_v1；
  P7 建包 v5（door_regrasp_pb1_finish 同一處置：關門列偏航 2π 換算、銜接容差 1e-3）；P8 pb1e（mode 3／4 交界以退離終點切段）。
輸出（皆不覆寫）：results/door/door_regrasp_pb60_{release,replan,retime,v4,close_plan,close_retime,close_retime_yaw,v5}.json、door_regrasp_ref_pb60.json。
界線：離線運動學；60° 的整條參考未經實跑。
    python3 -u evaluation/door_regrasp_pb60.py
"""
import inspect
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_v1 as TJ                       # noqa: E402

RD = os.path.join(HERE, 'results', 'door')
Q48 = 47.888663359359974
Q0 = 60.0 - (48.0 - Q48)
DQ = math.radians(Q0 - Q48)
STOW_F = np.array([0.0, 0.0, 0.15, 0.0, -1.10, 0.0])
B1_48 = np.array([0.65, -0.05, math.radians(270.0)])
c, s = math.cos(-DQ), math.sin(-DQ)
B1_60 = np.array([c * B1_48[0] - s * B1_48[1], s * B1_48[0] + c * B1_48[1], B1_48[2] - DQ])
F = {k: os.path.join(RD, f'door_regrasp_pb60_{k}.json') for k in ('release', 'replan', 'retime', 'v4', 'v5unused', 'close_plan', 'close_retime',
                                                                  'close_retime_yaw', 'v5')}
OUT = os.path.join(RD, 'door_regrasp_ref_pb60.json')


def main():
    for p in list(F.values()) + [OUT]:
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    t0 = time.time()
    TJ.STOW[:] = STOW_F
    print(f'Q0 {Q0:.6f}°、ΔQ {math.degrees(DQ):.4f}°、B1_60 ({B1_60[0]:.4f}, {B1_60[1]:.4f}, {math.degrees(B1_60[2]):.2f}°)', flush=True)
    import door_regrasp_pb1 as P
    import door_regrasp_replan_v8 as RV8
    RV8.B1 = B1_60.copy()
    P.Q0 = Q0
    P.F = {'rel': F['release'], 'replan': F['replan'], 'retime': F['retime'], 'v4': F['v4'], 'v5': F['v5unused']}
    try:
        P.main()
    except SystemExit as e:
        print(f'P1–P4 停於：{e}（預期為舊關門計畫的銜接核）', flush=True)
    if not os.path.exists(F['v4']):
        raise SystemExit('P1–P4 未產生 v4 包 ⇒ 停止（見上方阻擋）')
    # P5 關門規劃（起點＝本包 v4 mode 5 終點）
    import door_regrasp_close_plan_v1 as CPL
    CPL.PKG = json.load(open(F['v4']))
    CPL.THG = np.asarray(CPL.PKG['inner_grasp']['T_HG'], float).reshape(4, 4)
    CPL.OUT = F['close_plan']
    CPL.main()
    cp = json.load(open(F['close_plan']))
    if not cp.get('ok', True) or cp.get('block'):
        raise SystemExit(f"P5 關門規劃不成立：{cp.get('block')}")
    # P6 關門時間化
    import door_regrasp_close_retime_v1 as CRT
    CRT.SRC, CRT.OUT = F['close_plan'], F['close_retime']
    CRT.main()
    # P7 建包 v5（偏航 2π 換算＋銜接容差 1e-3，同 pb1_finish）
    import door_regrasp_ref_build_v5 as RB5
    pkg = json.load(open(F['v4']))
    y5 = float(pkg['segments']['5']['base'][-1][2])
    cr = json.load(open(F['close_retime']))
    y7 = float(cr['reference_20hz']['rows'][0][1][2])
    k = round((y7 - y5) / (2 * math.pi))
    if abs((y7 - y5) - 2 * math.pi * k) > 1e-6:
        raise SystemExit(f'偏航差不是 2π 整數倍：{y7 - y5}')
    for r in cr['reference_20hz']['rows']:
        r[1][2] = float(r[1][2]) - 2 * math.pi * k
    cr['pb60_yaw_shift'] = {'k_2pi': -k, 'note': '同一姿態的偏航表示換算；幾何不變'}
    json.dump(cr, open(F['close_retime_yaw'], 'w'), ensure_ascii=False, indent=1, default=float)
    RB5.V4, RB5.OUT, RB5.CR, RB5.CP = F['v4'], F['v5'], F['close_retime_yaw'], F['close_plan']
    src = inspect.getsource(RB5.main).replace('if d_join > 1e-9:', 'if d_join > JOIN_TOL:')
    assert 'JOIN_TOL' in src
    ns = dict(vars(RB5), JOIN_TOL=1e-3)
    exec(compile(src, RB5.__file__, 'exec'), ns)
    ns['main']()
    # P8 pb1e（mode 3／4 交界以退離終點切段）
    import door_regrasp_ref_pb1e as E
    E.SRC, E.REL, E.OUT = F['v5'], F['release'], OUT
    E.main()
    pk = json.load(open(OUT))
    pk['pb60'] = {'q0_deg': Q0, 'dq_from_48_deg': math.degrees(DQ), 'B1_48': B1_48.tolist(), 'B1_60': B1_60.tolist(), 'stow': STOW_F.tolist(),
                  'files': {k: os.path.relpath(v, HERE) for k, v in F.items()}, 'wall_s': time.time() - t0,
                  'rule': 'B1 繞鉸鏈剛體轉 −ΔQ；其餘演算法、判準同 pb1f',
                  'note_replan_B1_deg': 'replan.B1_deg 欄位為 replan_v8 寫死的 48° 紀錄值；實際 B1＝包內 B1（mode 5 起點）＝B1_60'}
    json.dump(pk, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    import door_regrasp_ref_registry as REGY
    print('pb60 完成', {k_: round(v['t'][-1], 2) for k_, v in pk['segments'].items()}, 'validate', REGY.validate(pk), f'牆鐘 {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
