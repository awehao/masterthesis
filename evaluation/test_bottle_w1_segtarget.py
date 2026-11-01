#!/usr/bin/env python3
"""學習模型供目標（bottle_w1_segtarget_ctrl）的離線閉迴路核對（Codex 20261102_002139；假世界、假候選，非物理）。
假候選：每個控制拍以 5 Hz 產生 known_type 欄位（快照每 1 s 換序號），中心＝真值 ± 0.5 mm 雜訊、z_support＝桌面（或測試給定值），只在給定時間窗內 ready（模擬手腕相機只在接近途中某段看得到支撐）。
舊估計器的假影格照常送（V2／W1），用來驗證控制器確實忽略。
  A 名目（ready 窗 9–11 s）：凍結、完成放置、T_grasp z＝z_support＋0.10、V2／W1 舊估計被忽略（計數 > 0、不影響）；持瓶對兩桌、自碰、禁止接觸、命令有限照既有核對；
  B 從未 ready ⇒ segtarget_no_fix（漸變開始時）；C 觀測年齡 0.8 s ⇒ 拒收 stale ⇒ no_fix；D 凍結後下一快照偏 7 mm ⇒ segtarget_inconsistent；
  E 快照檢查一直 pending（snap_ik＝1）⇒ 不採用 ⇒ no_fix；F 凍結後訊息中斷 ⇒ 依凍結語義照常完成；G z_support＋0.05 ⇒ 凍結的 T_grasp z 跟著＋0.05（只核目標，不要求抓取成功）；\n  H 目標 y＋0.5 m（window_branch 無解）⇒ segtarget_unreachable 受控停止、不閉爪、命令有限、保留未採用。
    python3 evaluation/test_bottle_w1_segtarget.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_w1_segtarget_ctrl as SG                   # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(win=(9.0, 11.0), age=0.05, ik=2, dz=0.0, jump_after=None, cut_after_freeze=False, seed=0, dy=0.0):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = SG.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(seed)
    st = dict(next_t=0.0, seq=0, t_seq=-9.0, z_at_freeze=None)
    step0 = c.step

    def step(t, q, **kw):
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            in_win = win is not None and win[0] <= t <= win[1]
            frozen = c.vfix is not None
            if in_win and not (cut_after_freeze and frozen):
                if t - st['t_seq'] >= 1.0:
                    st['seq'] += 1
                    st['t_seq'] = t
                off = np.zeros(2)
                if jump_after is not None and frozen and st['seq'] >= jump_after:
                    off = np.array([0.007, 0.0])
                xy = np.array([T.T.BX, T.T.BY + dy]) + rng.normal(0, 0.0005, 2) + off
                cand = dict(ready=1.0, age_s=age, snap_seq=float(st['seq']), snap_t_last=t - age, snap_cx=float(xy[0]), snap_cy=float(xy[1]),
                            snap_z_support=T.T.TOP + dz, snap_ik=float(ik), snap_local=2.0, snap_dev_mm=0.1, snap_consistent=1.0)
                c.candidate_intake(t, cand)
        out = step0(t, q, **kw)
        if c.vfix is not None and st['z_at_freeze'] is None:
            st['z_at_freeze'] = float(c.task.T_grasp[2, 3])
        return out
    c.step = step
    r, _ = T.run(c, f)
    return c, f, r, st


def main():
    import bottle_s0_ctrl as C0
    c, f, r, st = run()
    o = TC.summarize('A 名目', c, f, r)
    sg = r['segtarget']
    zg = C0.grasp_target(T.T.BX, T.T.BY, T.T.TOP + 0.10)[2, 3]
    print(f"   凍結 {sg['fix'] and {k: sg['fix'][k] for k in ('t', 'xy', 'z_support', 'seq')}}；採用 {len(sg['adopt'])}；拒收 {sg['rejected']}；忽略舊估計 V2 {sg['ignored_v2']}、W1 {sg['ignored_w1']}；"
          f"T_grasp z {st['z_at_freeze']:.4f}（期望 {zg:.4f}）")
    chk('A 名目完成且放置判準通過', o['done'] and bool((o['place'] or {}).get('ok')), (o['stop'], o['place']))
    chk('A 目標只來自候選：凍結 z＝z_support＋0.10、舊估計被忽略（計數 > 0）', abs(st['z_at_freeze'] - zg) < 1e-9 and sg['ignored_v2'] > 0 and sg['ignored_w1'] > 0, (st['z_at_freeze'], sg['ignored_v2'], sg['ignored_w1']))
    chk('A 持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限', all(v[0] >= TL.REJ * 1e3 for v in o['held'].values()) and o['self_min_mm'] > 0 and o['forb'] == 0 and o['finite'], (o['held'], o['self_min_mm'], o['forb']))
    for nm, kw, want in (('B 從未 ready', dict(win=None), 'segtarget_no_fix'), ('C 觀測年齡 0.8 s', dict(age=0.8), 'segtarget_no_fix'),
                         ('D 凍結後快照偏 7 mm', dict(jump_after=2), 'segtarget_inconsistent'), ('E 檢查一直 pending', dict(ik=1), 'segtarget_no_fix')):
        c2, f2, r2, st2 = run(**kw)
        why = (r2['stop'] or {}).get('why')
        print(f"   {nm}：停止 {why} @ {(r2['stop'] or {}).get('t')}；拒收 {r2['segtarget']['rejected']}")
        chk(f'{nm} ⇒ {want}（受控停止，不退回舊估計或桌高）', why == want and not r2['mission_done'], r2['stop'])
    c3, f3, r3, st3 = run(cut_after_freeze=True)
    o3 = TC.summarize('F 凍結後訊息中斷', c3, f3, r3)
    chk('F 凍結後訊息中斷 ⇒ 依凍結語義完成（期限兜底）', o3['done'] and bool((o3['place'] or {}).get('ok')), o3['stop'])
    c4, f4, r4, st4 = run(dz=0.05)
    print(f"   G z_support＋0.05：凍結 T_grasp z {st4['z_at_freeze']}；結果 {(r4['stop'] or {}).get('why')}")
    chk('G 支撐高度改變 ⇒ 凍結目標 z 跟著改變（＋0.05 m）', st4['z_at_freeze'] is not None and abs(st4['z_at_freeze'] - (zg + 0.05)) < 1e-9, st4['z_at_freeze'])
    c5, f5, r5, st5 = run(dy=0.5)
    why5 = (r5['stop'] or {}).get('why')
    print(f"   H 目標 y＋0.5 m（分支無解）：停止 {why5} @ {(r5['stop'] or {}).get('t')}；閉爪請求 {c5.t_close_req}；凍結 {c5.vfix is not None}")
    chk('H 分支無解 ⇒ segtarget_unreachable 受控停止、不閉爪、命令有限、保留未採用', why5 == 'segtarget_unreachable' and c5.t_close_req is None and T.finite_cmds(f5) and c5.vfix is None,
        (r5['stop'], c5.t_close_req))
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
