#!/usr/bin/env python3
"""右側：抓取計畫回覆的目標核對走**實際 MRO 解析到的 _gp_poll**（bottle_motm_right_nom_3 於 7.2 s 以 grasp_plan_target_mismatch 停止：
online2_ctrl._gp_poll 寫死 C0.grasp_target（左側），先前只改了被它覆寫掉的 online_ctrl 版本）。真 _gp_poll＋真 _ev／S0Judge._stop；工作者回覆以真 bottle_side.grasp_target 產生。
案例：右側控制＋右側回覆 ⇒ 不停；右側控制＋左側回覆 ⇒ grasp_plan_target_mismatch；左側控制＋左側回覆 ⇒ 不停，且 _grasp_T 與 C0.grasp_target 逐元素相同。
另列出實際 MRO 各檔中仍直接呼叫 C0.grasp_target 的位置與是否被解析到（靜態掃描）。判定不過 ⇒ 離開碼 1。"""
import inspect
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_side as BSD                                        # noqa: E402
import right_side_integration_check as RC                        # noqa: E402

C = RC.C


def case(ctrl_side, reply_side):
    o = RC.mk(ctrl_side)
    o.grasp_side = ctrl_side
    xy, zs = (0.02, BSD.sgn(ctrl_side) * 0.41), 0.58
    o.gp = dict(state='pending', id=9, seq=1, xy=np.array(xy), zs=zs, x_latest=5.0, t_req=1.0)
    o.obs_last, o.seg = None, dict(dev_tol_mm=5.0, max_age=1.0)       # 無新觀測 ⇒ 目標核對過後停在等待觀測（不進後續採用）
    Tg = BSD.grasp_target(xy[0], xy[1], zs + 0.10, reply_side)
    o.pc.poll = lambda rid: dict(ok=True, error=None, result=dict(T_g=np.asarray(Tg).tolist()), t_wall_recv=0.0, wall=dict(t0=0.0, t1=0.1))
    impl = inspect.getsourcefile(C._gp_poll)
    C._gp_poll(o, np.array([-2.8, 0.0, 0.0]))
    return dict(ctrl=ctrl_side, reply=reply_side, impl=os.path.basename(impl), state=o.gp['state'] if o.gp else None,
                stop=_why(o.judge.stop))


def _why(st):
    if st is None:
        return None
    return st.get('why') if isinstance(st, dict) else str(st)


def scan():
    out = []
    for c in C.__mro__:
        try:
            f = inspect.getsourcefile(c)
        except TypeError:
            continue
        if not f or not os.path.basename(f).startswith('bottle_'):
            continue
        src, _ = inspect.getsourcelines(c)
        cur = None
        for ln in src:
            m = re.match(r'\s+def (\w+)\(', ln)
            if m:
                cur = m.group(1)
            if 'C0.grasp_target(' in ln and not ln.strip().startswith('#'):
                res = getattr(C, cur, None)
                out.append(dict(file=os.path.basename(f), method=cur, resolved_here=bool(res is not None and inspect.getsourcefile(res) == f and res.__qualname__.split('.')[0] == c.__name__),
                                is_default_hook=cur == '_grasp_T'))
    return out


def main():
    import bottle_s0_ctrl  # noqa
    import bottle_w1_segtarget_ctrl as SG
    r = dict(cases=[case('right', 'right'), case('right', 'left'), case('left', 'left')], scan=scan())
    L_ = SG.C0.grasp_target(0.02, 0.41, 0.68)
    o = RC.mk('left')
    o.grasp_side = 'left'
    r['left_identity'] = bool(np.array_equal(np.asarray(o._grasp_T(0.02, 0.41, 0.68)), np.asarray(L_)))
    c = r['cases']
    live = [s for s in r['scan'] if s['resolved_here'] and not s['is_default_hook']]
    r['live_hardcoded'] = live
    fails = []
    if c[0]['stop'] is not None or c[0]['state'] != 'ready':
        fails.append('right/right 應通過')
    if c[1]['stop'] != 'grasp_plan_target_mismatch':
        fails.append('right/left 應停止')
    if c[2]['stop'] is not None or c[2]['state'] != 'ready':
        fails.append('left/left 應通過')
    if not r['left_identity']:
        fails.append('左側 _grasp_T 與 C0.grasp_target 不同')
    r['verdict'] = dict(ok=not fails, fails=fails, note='live_hardcoded＝仍在實際解析路徑上直接用左側目標的方法（列出供判讀，不自動判失敗）')
    json.dump(r, open(os.path.join(HERE, 'results/bottle/right_gp_poll_check_r1.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
