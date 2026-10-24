#!/usr/bin/env python3
"""W1b 判定器（Codex 20261024_231908 B5：判定器須明確支援目標角，不得暗用 8° 常數）。door_k3m_check_p4b2.py 以上不改。

一、登錄核對（任一不符 ⇒ 理由，且不以登錄值改寫判準）：
  task.json w1b.params_path 的 sha256＝w1b.params_sha256；task config θ_goal／over_goal＝登錄值；OPEN 逾時＝登錄值；
  執行端 room.k3m.config：走廊表 sha256＝登錄值、內側帶＝登錄值、envelope＝登錄包絡（≤ 1e-4）；
  獨立觀察器 lw_observer.json config.door_max_deg＝登錄 θ＋4°。
二、判準：以登錄 θ 建 K3AConfig，暫時替換 door_k3a_check 的 CFG／TCFG／THETA 與 door_k3m_check 的 ENVELOPE，
  呼叫 door_k3m_check_p4b2.judge（K3-M、SETTLE p2、主動保持停止 P1–P4、手臂 0.5 s 窗口、零長度 END_HOLD 事件修正），**判完一定還原**。
三、引導相位（控制成本項的使用範圍）：trace 每列 guide.w＞0 只能出現在 phase ∈ {OPEN, SETTLE} 且 k3m_mode＝free；
  PAUSE 及其後各列 guide.w＝0；曾進入 OPEN 且切 free 卻從未開啟引導 ⇒ guide_never_on。

r2（Codex 20261024_234630 第 3 項）：缺證據不得誤收——登錄核對先驗證必要欄位存在、型別正確且有限
  （task config θ／over_goal／OPEN 逾時；room 走廊表 sha256（64 位十六進位）、內帶、包絡 5 元素；觀察器上限），缺漏／NaN／非法 ⇒
  w1b_<欄位>_missing_or_invalid；引導相位：每列 guide.w 必須有限且等於登錄的開（w_vref）或關（0）值；開啟列的 v_body 必須 3 個有限值；
  開啟只能在 OPEN／SETTLE 且 free、PAUSE 之後不得開啟。

    python3 evaluation/door_k3m_check_w1b.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1b.json；不覆寫
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3a_check as KC3                           # noqa: E402
import door_k3a_policy as KP                           # noqa: E402
import door_k3m_check as KM                            # noqa: E402
import door_k3m_check_p4b2 as C4B2                     # noqa: E402


@contextlib.contextmanager
def target_constants(theta_deg, over_goal_deg, envelope):
    saved = (KC3.CFG, KC3.TCFG, KC3.THETA, KM.ENVELOPE)
    try:
        cfg = KP.K3AConfig(theta_goal_deg=float(theta_deg), over_goal_deg=float(over_goal_deg))
        KC3.CFG, KC3.TCFG = cfg, cfg.target_cfg()
        KC3.THETA = KC3.TCFG.theta_goal_rad
        KM.ENVELOPE = tuple(float(v) for v in envelope)
        yield
    finally:
        KC3.CFG, KC3.TCFG, KC3.THETA, KM.ENVELOPE = saved


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def check_registration(task, room, observer, params, why, st):
    tp, ps = params['task_param_extension'], params['protection_scope_change']
    cfg = task.get('config') or {}
    e = ps['envelope']
    reg = [e['fwd_lo'], e['fwd_hi'], e['lat_lo'], e['lat_hi'], e['yaw_abs_deg']]

    def same(name, got, want, tol=1e-9):
        if not _num(got):
            why.append(f'w1b_{name}_missing_or_invalid')
        elif abs(float(got) - float(want)) > tol:
            why.append(f'w1b_{name}_mismatch')
    same('theta', cfg.get('theta_goal_deg'), tp['theta_goal_deg'])
    same('over_goal', cfg.get('over_goal_deg'), tp['over_goal_deg'])
    same('open_timeout', (cfg.get('timeouts') or {}).get('OPEN'), tp['open_timeout_s'])
    kc = (room.get('k3m') or {}).get('config') or {}
    sha = kc.get('corridor_table_sha256')
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)):
        why.append('w1b_corridor_table_missing_or_invalid')
    elif sha != ps['corridor_table_sha256']:
        why.append('w1b_corridor_table_mismatch')
    same('inner_mm', kc.get('inner_mm'), ps['corridor_inner_mm'])
    same('inner_deg', kc.get('inner_deg'), ps['corridor_inner_deg'])
    env = kc.get('envelope')
    if not (isinstance(env, (list, tuple)) and len(env) == 5 and all(_num(v) for v in env)):
        why.append('w1b_envelope_missing_or_invalid')
    elif any(abs(float(a_) - float(b_)) > 1e-4 for a_, b_ in zip(env, reg)):
        why.append('w1b_envelope_mismatch')
    same('observer_limit', ((observer or {}).get('config') or {}).get('door_max_deg'), ps['observer_door_max_deg'])
    st['w1b_registration'] = {'theta': cfg.get('theta_goal_deg'), 'envelope_room': env, 'envelope_registered': reg,
                              'observer_door_max_deg': ((observer or {}).get('config') or {}).get('door_max_deg')}
    return reg


def check_guidance(task, why, st, w_on):
    tr = task.get('trace') or []
    bad, n_on, after_pause, seen_free_open = [], 0, False, False
    for x in tr:
        g = x.get('guide')
        if x.get('phase') == 'PAUSE':
            after_pause = True
        if x.get('phase') == 'OPEN' and x.get('k3m_mode') == 'free':
            seen_free_open = True
        if not isinstance(g, dict) or not _num(g.get('w')):
            bad.append((x.get('t'), 'missing_or_nonfinite'))
            continue
        w = float(g['w'])
        if w == 0.0:
            continue
        if w != float(w_on):
            bad.append((x.get('t'), f'weight_not_registered:{w}'))
            continue
        n_on += 1
        vb = g.get('v_body')
        if not (isinstance(vb, (list, tuple)) and len(vb) == 3 and all(_num(v) for v in vb)):
            bad.append((x.get('t'), 'v_body_invalid'))
        if x.get('phase') not in ('OPEN', 'SETTLE') or x.get('k3m_mode') != 'free' or after_pause:
            bad.append((x.get('t'), x.get('phase')))
    st['w1b_guidance'] = {'n_rows': len(tr), 'n_on': n_on, 'n_bad': len(bad), 'first_bad': bad[:5], 'w_on_registered': w_on}
    if bad:
        why.append(f'guide_phase_violation_{len(bad)}')
    if seen_free_open and n_on == 0:
        why.append('guide_never_on')


def judge_with(task, room, gate, solver, observer, params):
    why_r, st_r = [], {}
    reg = check_registration(task, room, observer, params, why_r, st_r)
    tp = params['task_param_extension']
    with target_constants(tp['theta_goal_deg'], tp['over_goal_deg'], reg):
        res = C4B2.judge(task, room, gate, solver)
    st = dict(res['stats'])
    st.update(st_r)
    why_g = []
    check_guidance(task, why_g, st, params['control_cost_change']['guidance']['w_vref'])
    why = why_r + list(res['why']) + why_g
    return dict(res, why=why, stats=st, **{'pass': not why, 'w1b_registration_ok': not why_r, 'w1b_guidance_ok': not why_g,
                                           'theta_goal_deg': tp['theta_goal_deg']})


def judge(task, room, gate, solver, observer):
    w = task.get('w1b') or {}
    p = w.get('params_path')
    if not p:
        return {'pass': False, 'why': ['w1b_params_missing'], 'stats': {}}
    pp = p if os.path.isabs(p) else os.path.join(WS, p)
    if hashlib.sha256(open(pp, 'rb').read()).hexdigest() != w.get('params_sha256'):
        return {'pass': False, 'why': ['w1b_params_sha_mismatch'], 'stats': {}}
    return judge_with(task, room, gate, solver, observer, json.load(open(pp)))


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1b.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'w1b_registration_ok', 'w1b_guidance_ok', 'why')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
