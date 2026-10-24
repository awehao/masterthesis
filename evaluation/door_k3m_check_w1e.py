#!/usr/bin/env python3
"""W1e 判定器（door_k3m_check_w1c.py r2 的版本化擴充；Codex 20261025_023608）：w1c 全部判準照用（w1b 全部＋偏移估計保持 H1–H6），
另加**暫停期手指驅動輸出上限**生效證據（登錄 executor_change.pause_finger_max_effort > 0 時必查；缺證據一律不通過；未確認生效不宣稱介入成立）。

  F1 room_run finger_effort 存在；arg_pause_max_effort＝登錄值；init.ok 且兩指啟動讀回＝登錄 original_effort（≤ tol）；joints＝登錄。
  F2 pause 紀錄存在：step＝room k3m.pause_switch.switch_step；set＝登錄值；兩指 readback＝登錄值（≤ tol）且 ok；old＝啟動讀回。
  F3 定期讀回：自切換起每 1 s 一筆，筆數 ≥ ⌊(最後物理步 t − 切換 t)／1 s⌋ − 1，且每筆兩指＝登錄值、ok。
  原停止與中止條件全部保留（w1c／w1b 判準不變、不因加強夾持而放寬）。
r2（Codex 20261025_024339 必修②③）：只數筆數會誤收（重複 periodic、切換時間 99999、索引 [0,1]）——
  F1' 啟動：兩索引為不同整數、在名稱表內、名稱映射＝登錄 joints；完整上限向量長度＝名稱數；arg_original＝登錄；
  F2' 切換 step／t＝pause_switch 的 switch_step／switch_t，且對上 room 真實物理步（step→sim_t）；索引＝啟動索引；
      ok 必須明確為 True、兩讀值有限＝設定；切換前後完整向量：非手指不變、啟動向量＝切換前（非手指）、切換後兩指＝設定；
  F3' periodic：步號與時間嚴格前進、間隔 ≥ 1 s、每筆對上 room 真實步、ok 明確 True、others_same True；重複或偽造不算覆蓋，
      覆蓋筆數以 room 最後物理步時間計；任何不合格筆即不通過。切換失敗紀錄存在 ⇒ 不通過。

    python3 evaluation/door_k3m_check_w1e.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1e.json；不覆寫
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3m_check_w1b as W1B                       # noqa: E402
import door_k3m_check_w1c as W1C                       # noqa: E402

TOL = 1e-3


def _all_eq(v, x, n=2):
    return isinstance(v, list) and len(v) == n and all(W1B._num(u) and abs(float(u) - x) <= TOL for u in v)


def _room_time(room):
    try:
        return {int(r[0]): float(r[1]) for r in room['steps']}
    except (KeyError, TypeError, ValueError, IndexError):
        return {}


def _at_room(st_t, step, t):
    return isinstance(step, int) and W1B._num(t) and step in st_t and abs(st_t[step] - float(t)) <= 1e-6


def _others_same(a, b, idx):
    if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
        return False
    return all((k in idx) or (W1B._num(a[k]) and W1B._num(b[k]) and a[k] == b[k]) for k in range(len(a)))


def check_finger_effort(room, ec, why, st):
    V, V0, J = ec.get('pause_finger_max_effort'), ec.get('original_effort'), ec.get('joints')
    fe = (room or {}).get('finger_effort')
    if not isinstance(fe, dict):
        why.append('finger_effort_missing')
        st['finger_effort_check'] = None
        return False
    ok = True

    def bad(tag):
        nonlocal ok
        why.append(tag)
        ok = False
    ini, pz = fe.get('init') or {}, fe.get('pause')
    st_t = _room_time(room)
    names, idx = ini.get('names'), ini.get('indices')
    idx_ok = (isinstance(names, list) and isinstance(idx, list) and len(idx) == 2 and all(isinstance(k, int) for k in idx)
              and len(set(idx)) == 2 and all(0 <= k < len(names) for k in idx) and [names[k] for k in idx] == J)
    if fe.get('arg_pause_max_effort') != V or fe.get('arg_original_effort') != V0:
        bad('finger_effort_arg_mismatch')
    if ini.get('ok') is not True or not _all_eq(ini.get('readback'), V0) or ini.get('joints') != J or not idx_ok \
            or not isinstance(ini.get('full'), list) or len(ini['full']) != len(names or []) \
            or not all(W1B._num(ini['full'][k]) and abs(ini['full'][k] - V0) <= TOL for k in (idx if idx_ok else [])):
        bad('finger_effort_init_invalid')
    sw = ((room or {}).get('k3m') or {}).get('pause_switch') or {}
    per = []
    if not isinstance(pz, dict):
        bad('finger_effort_pause_missing')
    else:
        if pz.get('step') != sw.get('switch_step') or not W1B._num(pz.get('t')) or not W1B._num(sw.get('switch_t')) \
                or abs(float(pz['t']) - float(sw['switch_t'])) > 1e-6 or not _at_room(st_t, pz.get('step'), pz.get('t')):
            bad('finger_effort_not_at_pause_switch')
        if pz.get('indices') != idx or pz.get('joints') != J:
            bad('finger_effort_index_mismatch')
        if pz.get('set') != V or pz.get('ok') is not True or not _all_eq(pz.get('readback'), V):
            bad('finger_effort_not_effective')
        if not _all_eq(pz.get('old'), V0):
            bad('finger_effort_old_mismatch')
        fb, fa = pz.get('full_before'), pz.get('full_after')
        if not idx_ok or not _others_same(fb, fa, idx) or not _others_same(ini.get('full'), fb, idx) \
                or not all(W1B._num(fa[k]) and abs(fa[k] - V) <= TOL for k in idx):
            bad('finger_effort_non_finger_changed_or_missing')
        per = pz.get('periodic') or []
        prev_s, prev_t, n_bad, n_cov = pz.get('step'), pz.get('t'), 0, 0
        for e in per:
            s_, t_ = e.get('step'), e.get('t')
            fine = (isinstance(s_, int) and isinstance(prev_s, int) and s_ > prev_s and W1B._num(t_) and W1B._num(prev_t)
                    and float(t_) >= float(prev_t) + 1.0 - 1e-6 and _at_room(st_t, s_, t_)
                    and e.get('ok') is True and _all_eq(e.get('readback'), V) and e.get('others_same') is True)
            if fine:
                n_cov += 1
                prev_s, prev_t = s_, t_
            else:
                n_bad += 1
        t_last = max(st_t.values()) if st_t else float('nan')
        need = (int(math.floor((t_last - float(pz['t'])) / 1.0)) - 1) if W1B._num(pz.get('t')) and math.isfinite(t_last) else None
        if n_bad:
            bad(f'finger_effort_periodic_invalid_{n_bad}')
        if need is None or n_cov < max(need, 0):
            bad(f'finger_effort_periodic_short_{n_cov}')
    if fe.get('pause_switch_failed'):
        bad('finger_effort_switch_failed')
    st['finger_effort_check'] = {'set': V, 'original': V0, 'init_readback': ini.get('readback'), 'indices': idx,
                                 'pause_step': None if not isinstance(pz, dict) else pz.get('step'), 'switch_step': sw.get('switch_step'),
                                 'readback': None if not isinstance(pz, dict) else pz.get('readback'), 'n_periodic': len(per), 'ok': ok}
    return ok


def judge_with(task, room, gate, solver, observer, params):
    res = W1C.judge_with(task, room, gate, solver, observer, params)
    why_e, st = [], dict(res['stats'])
    ec = params.get('executor_change') or {}
    if W1B._num(ec.get('pause_finger_max_effort')) and ec['pause_finger_max_effort'] > 0:
        fe_ok = check_finger_effort(room, ec, why_e, st)
    else:
        why_e.append('w1e_executor_change_not_registered')
        fe_ok = False
    why = list(res['why']) + why_e
    return dict(res, why=why, stats=st, **{'pass': not why, 'finger_effort_ok': fe_ok})


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
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1e.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'w1b_registration_ok', 'w1b_guidance_ok', 'offset_hold_ok', 'finger_effort_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
