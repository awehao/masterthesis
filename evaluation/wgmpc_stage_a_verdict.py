#!/usr/bin/env python3
"""一趟階段 A 的成功判定 ＋ 指標。

成功判準（**全部**成立）
  1 前置調姿完成（有完成宣告）
  2 交棒通過（handover.json verdict == pass）
  3 到達 5 mm／0.02 rad 並**連續保持 2 s**
  4 沒有失效閂鎖
  5 沒有設定點有效限位違反
  6 封存完整
實測關節角是否越過有效限位**另行核實**（不列入成功判準 —— 執行端那一層
只保護設定點，一階遲滯下實測角仍可能落在界外）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

ARM = [f'joint{i}' for i in range(1, 7)]


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _reach_from_log(sim, tgt, hold_s=2.0, tol_p=0.005, tol_r=0.02):
    """由實錄重算到達與保持。

    位置誤差取 log 的 tcp_x/y/z 對目標；姿態誤差取 tcp_r** 對 R_DES 的
    測地角（acos((tr(R_des^T R)-1)/2)）。保持要求**連續**落在容差內 hold_s。
    """
    if not sim or not sim.get('log_cols') or not sim.get('log'):
        return {'error': '無 log'}
    if not tgt or not tgt.get('tcp_world') or not tgt.get('R_world'):
        return {'error': '無 target.json'}
    c = {n: i for i, n in enumerate(sim['log_cols'])}
    need = ['t', 'tcp_x', 'tcp_y', 'tcp_z'] + [
        f'tcp_r{i}{j}' for i in range(3) for j in range(3)]
    if any(k not in c for k in need):
        return {'error': f'log 缺欄位 {[k for k in need if k not in c][:4]}'}
    L = sim['log']
    t = np.array([r[c['t']] for r in L], float)
    P = np.array([[r[c[k]] for k in ('tcp_x', 'tcp_y', 'tcp_z')] for r in L],
                 float)
    R = np.array([[[r[c[f'tcp_r{i}{j}']] for j in range(3)]
                   for i in range(3)] for r in L], float)
    pd_ = np.asarray(tgt['tcp_world'], float)
    Rd = np.asarray(tgt['R_world'], float)
    ep = np.linalg.norm(P - pd_, axis=1)
    tr = np.einsum('ij,kij->k', Rd, R)          # tr(Rd^T R)
    er = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    ok = np.isfinite(ep) & np.isfinite(er)
    if not ok.any():
        return {'error': 'TCP 欄位全為非有限值'}
    inside = ok & (ep <= tol_p) & (er <= tol_r)
    # 最長連續在容差內的時間窗
    best, best_i0 = 0.0, None
    i = 0
    while i < len(inside):
        if not inside[i]:
            i += 1; continue
        j = i
        while j + 1 < len(inside) and inside[j + 1]:
            j += 1
        span = float(t[j] - t[i])
        if span > best:
            best, best_i0 = span, i
        i = j + 1
    held = best >= hold_s
    return {
        'reached_held_measured': bool(held),
        'longest_in_tol_s': best,
        't_first_in_tol_s': (None if best_i0 is None else float(t[best_i0])),
        't_reach_s': (float(t[best_i0]) if held else None),
        'err_p_min_mm': float(ep[ok].min() * 1e3),
        'err_r_min_rad': float(er[ok].min()),
        'err_p_final_mm': float(ep[ok][-1] * 1e3),
        'err_r_final_rad': float(er[ok][-1]),
        'tol': {'pos_m': tol_p, 'rot_rad': tol_r, 'hold_s': hold_s},
        'n_samples_in_tol': int(inside.sum()),
        'basis': ('由 log 的 tcp_x/y/z 與 tcp_r** 對 target.json 重算；'
                  '姿態用測地角。**不照抄節點回報的 reached_held**'),
    }


def _base_motion(sim, hand_t):
    """交棒後底盤的**實測**位移與路徑長。"""
    if not sim or not sim.get('log_cols'):
        return {'error': '無 log'}
    c = {n: i for i, n in enumerate(sim['log_cols'])}
    if any(k not in c for k in ('t', 'base_x', 'base_y', 'base_yaw')):
        return {'error': 'log 缺 base 欄位'}
    L = sim['log']
    t = np.array([r[c['t']] for r in L], float)
    bx = np.array([r[c['base_x']] for r in L], float)
    by = np.array([r[c['base_y']] for r in L], float)
    yw = np.array([r[c['base_yaw']] for r in L], float)
    m = np.ones(len(t), bool) if hand_t is None else (t >= float(hand_t))
    if m.sum() < 3:
        return {'error': '交棒後樣本不足'}
    bx_, by_, yw_ = bx[m], by[m], yw[m]
    return {
        'window': ('交棒後' if hand_t is not None else '全趟'),
        'disp_m': float(np.hypot(bx_[-1] - bx_[0], by_[-1] - by_[0])),
        'path_m': float(np.hypot(np.diff(bx_), np.diff(by_)).sum()),
        'max_disp_from_start_m': float(
            np.hypot(bx_ - bx_[0], by_ - by_[0]).max()),
        'yaw_change_rad': float(yw_[-1] - yw_[0]),
        'basis': '實測位姿差分（**不用**回報的 twist）',
    }


def _hold_stats(sim, tgt, reach):
    """保持窗內的誤差統計。"""
    if not sim or not tgt or reach.get('t_reach_s') is None:
        return {'error': '未到達保持或缺資料'}
    c = {n: i for i, n in enumerate(sim['log_cols'])}
    need = ['t', 'tcp_x', 'tcp_y', 'tcp_z'] + [
        f'tcp_r{i}{j}' for i in range(3) for j in range(3)]
    if any(k not in c for k in need):
        return {'error': 'log 缺 tcp 欄位'}
    L = sim['log']
    t = np.array([r[c['t']] for r in L], float)
    P = np.array([[r[c[k]] for k in ('tcp_x', 'tcp_y', 'tcp_z')] for r in L],
                 float)
    R = np.array([[[r[c[f'tcp_r{i}{j}']] for j in range(3)]
                   for i in range(3)] for r in L], float)
    pd_ = np.asarray(tgt['tcp_world'], float)
    Rd = np.asarray(tgt['R_world'], float)
    ep = np.linalg.norm(P - pd_, axis=1)
    er = np.arccos(np.clip(
        (np.einsum('ij,kij->k', Rd, R) - 1.0) / 2.0, -1.0, 1.0))
    m = (t >= float(reach['t_reach_s'])) & np.isfinite(ep) & np.isfinite(er)
    if m.sum() < 3:
        return {'error': '保持窗樣本不足'}
    return {
        'window_s': [float(t[m][0]), float(t[m][-1])],
        'n': int(m.sum()),
        'err_p_p50_mm': float(np.median(ep[m]) * 1e3),
        'err_p_p95_mm': float(np.percentile(ep[m], 95) * 1e3),
        'err_p_max_mm': float(ep[m].max() * 1e3),
        'err_r_p50_rad': float(np.median(er[m])),
        'err_r_max_rad': float(er[m].max()),
    }


def assess(d):
    r = {'dir': os.path.relpath(d, WS)}
    bad, missing = [], []
    prep = _load(os.path.join(d, 'prepos.json'))
    ho = _load(os.path.join(d, 'handover.json'))
    wg = _load(os.path.join(d, 'wg2_out.json'))
    sim = _load(os.path.join(d, 'sim/wb_run.json'))
    arch = _load(os.path.join(d, 'archive_check.json'))
    drw = _load(os.path.join(d, 'drawer_check.json'))
    tgt = _load(os.path.join(d, 'target.json'))
    rcfg = _load(os.path.join(d, 'run_config.json'))

    # **起始路徑**決定前兩項判準。spawn（生成式起始構型）沒有前置調姿，
    # 起始閘門也**不是交棒判定**（判準較弱：無設定點、無套用回報可核），
    # 所以不能套用 move 路徑的那兩條，也不能把它報成交棒通過。
    # start_mode 寫在**運行器的** run_config.json；wb_run.json 是 Isaac 寫的，
    # 沒有這個鍵。退路是看起始閘門自己蓋的 gate 標記。
    start_mode = ((rcfg or {}).get('start_mode')
                  or (sim or {}).get('start_mode')
                  or ('spawn' if (ho or {}).get('gate') == 'start_posture'
                      else 'move'))
    r['start_mode'] = start_mode
    r['start_mode_source'] = ('run_config' if (rcfg or {}).get('start_mode')
                              else ('wb_run' if (sim or {}).get('start_mode')
                                    else 'gate_marker'))
    if start_mode == 'spawn':
        # **交叉核對**：spawn 要靠 Isaac 的 --init-arm-q 生效，所以執行端的
        # wb_run.json 必須記下它。沒記 ⇒ 機器人未必生成在起始構型。
        _iq = (sim or {}).get('init_arm_q')
        _want = (rcfg or {}).get('init_arm_q')
        if not _iq:
            bad.append('標為 spawn 但 wb_run.json 沒有 init_arm_q ⇒ '
                       '無法確認機器人生成在起始構型')
        elif _want:
            try:
                _w = [float(v) for v in str(_want).split(',')]
                if len(_w) != len(_iq) or max(
                        abs(float(a) - float(b))
                        for a, b in zip(_iq, _w)) > 1e-9:
                    bad.append(f'運行器要求的 init_arm_q 與執行端記錄不符：'
                               f'{_want} vs {_iq}')
            except (TypeError, ValueError):
                bad.append(f'運行器的 init_arm_q 無法解析：{_want!r}')

    # ---- 1 起始構型 ----
    if start_mode == 'spawn':
        # 生成式起步：**沒有前置調姿可判**。這一項不是「通過」，是
        # **不適用**；要判的是生成參數真的生效（實測角 == 要求的構型）。
        r['prepos'] = {
            'ok': None, 'not_applicable': '生成式起始構型，未做前置調姿',
            'init_arm_q': (sim or {}).get('init_arm_q')}
        if prep:
            # 檔案存在表示這趟其實跑過前置調姿 ⇒ 配置與紀錄不一致，
            # 不能默默當成 spawn。
            r['prepos']['ok'] = False
            bad.append('標為 spawn 卻有 prepos.json ⇒ 起始路徑與紀錄不一致')
    elif prep and prep.get('done'):
        r['prepos'] = {'ok': True, 'n_commands': prep['done']['n_commands'],
                       'q_goal': prep['done']['q_goal'],
                       'last_cmd_sim_t': prep['done']['last_cmd_sim_t'],
                       'duration_s': next((e['duration_s'] for e in
                                           prep.get('events', [])
                                           if e.get('ev') == 'built'), None)}
    else:
        r['prepos'] = {'ok': False}
        bad.append('前置調姿未完成（無 prepos.json 的完成宣告）')

    # ---- 2 起始閘門 ----
    if start_mode == 'spawn':
        # **不是交棒判定**，不得報成交棒通過（判準不同，不可混用）。
        if ho and ho.get('verdict') == 'pass' and ho.get('gate') == 'start_posture':
            _dt = ho.get('detail') or {}
            r['handover'] = {
                'ok': None,
                'not_applicable': '生成式起始構型，無前置來源可交棒',
                'gate': 'start_posture',
                'differs_from_handover':
                    '無前置來源的完成宣告與安靜時間可核（運行器在此模式下'
                    '不啟動前置節點，屬結構性質）；但多核「起始構型真的'
                    '生效」與「手臂確實靜止」',
                'start_max_dev_rad': _dt.get('start_max_dev_rad'),
                'start_tol_rad': _dt.get('start_tol_rad'),
                'setpoint_meas_max_dev_rad':
                    _dt.get('setpoint_meas_max_dev_rad'),
                'arm_rest_max_abs_u': _dt.get('arm_rest_max_abs_u'),
                'rest_tol': _dt.get('rest_tol'),
                'n_recv': _dt.get('n_recv'),
                'n_rejected': _dt.get('n_rejected'),
                'min_effective_slack_rad': _dt.get('min_effective_slack_rad'),
                'u_prev': ho.get('u_prev'),
                'u_prev_source': _dt.get('u_prev_source')}
            if _dt.get('u_prev_source') != 'applied':
                r['handover']['ok'] = False
                bad.append(f'起始核對的 u_prev 來源是 '
                           f'{_dt.get("u_prev_source")!r}，不是實際套用回報')
        else:
            r['handover'] = {
                'ok': False, 'gate': 'start_posture',
                'why': (ho or {}).get('why', '無 handover.json')}
            bad.append(f'起始構型核對未通過：{r["handover"]["why"]}')
    elif ho and ho.get('verdict') == 'pass':
        r['handover'] = {'ok': True, 'quiet_s': ho['detail'].get('quiet_s'),
                         'setpoint_meas_max_dev_rad':
                             ho['detail'].get('setpoint_meas_max_dev_rad'),
                         'u_prev': ho.get('u_prev'),
                         'u_prev_source': ho['detail'].get('u_prev_source'),
                         'exec_mode': ho.get('exec_mode_at_handover'),
                         'q_meas': ho.get('q_meas_at_handover'),
                         'setpoint': ho.get('setpoint_at_handover')}
    else:
        r['handover'] = {'ok': False,
                         'why': (ho or {}).get('why', '無 handover.json')}
        bad.append(f'交棒未通過：{r["handover"]["why"]}')

    # ---- 3 到達與保持：**自己從實錄算**，不照抄節點的宣稱 ----
    # stats 裡沒有 t_reach 或誤差欄位，所以由 log 的 TCP 位姿對 target.json
    # 的目標重算。這同時獨立驗證節點回報的 reached_held。
    st = (wg or {}).get('stats') or {}
    r['reach'] = {'node_reported_reached_held': st.get('reached_held')}
    if not st:
        bad.append('無求解器統計（wg2_out.json 缺 stats）'); missing.append('stats')
    rr = _reach_from_log(sim, tgt, hold_s=2.0,
                         tol_p=0.005, tol_r=0.02)
    r['reach'].update(rr)
    if rr.get('error'):
        missing.append('reach_from_log')
        if not st.get('reached_held'):
            bad.append(f'未到達並保持（節點回報 '
                       f'{st.get("reached_held")}；實錄重算失敗：{rr["error"]}）')
    elif not rr['reached_held_measured']:
        bad.append(f'**實錄重算未達到達保持**：最佳連續在容差內 '
                   f'{rr["longest_in_tol_s"]:.3f} s < 2.0 s'
                   f'（最小位置誤差 {rr["err_p_min_mm"]:.2f} mm、'
                   f'最小姿態誤差 {rr["err_r_min_rad"]:.4f} rad）')
    elif st.get('reached_held') is not True:
        bad.append(f'實錄顯示到達保持，但節點回報 {st.get("reached_held")}'
                   f' ⇒ 兩者不一致，先查清再採信')

    # ---- 4 失效閂鎖 ----
    cc = (sim or {}).get('cmd_chain') or {}
    r['chain'] = {'fail': cc.get('fail'), 'received': cc.get('received'),
                  'rejected': cc.get('rejected'),
                  'frozen_steps': cc.get('frozen_steps'),
                  'stop_reason': (sim or {}).get('stop_reason')}
    if cc.get('fail'):
        bad.append(f'執行端失效閂鎖：{cc["fail"]}')
    if sim is None:
        bad.append('無 sim/wb_run.json'); missing.append('wb_run.json')

    # ---- 5 設定點有效限位違反 ----
    # 執行端的 _fail 會把原因寫進 cmd_chain.fail；另核 log 的設定點欄位
    sp_viol = None
    if sim and sim.get('log_cols') and sim.get('log'):
        try:
            from ammr_wholebody_mpc.wholebody_kinematics import (
                WholeBodyKinematics)
            K = WholeBodyKinematics.from_urdf_file(
                os.path.join(WS,
                             'evaluation/models/omni_bot_wholebody_expanded.urdf'))
            lim = np.array(K.joint_limits())
            m = 0.05
            elo, ehi = lim[0, 3:] + m, lim[1, 3:] - m
            c = {n: i for i, n in enumerate(sim['log_cols'])}
            # **欄名是 joint*_sp 與 joint*_act**（先前寫 c[j] ⇒ KeyError('joint1')）
            miss = [k for j in ARM for k in (f'{j}_sp', f'{j}_act')
                    if k not in c]
            if miss:
                raise KeyError(f'log 缺欄位 {miss[:4]}')
            SP = np.array([[row[c[f'{j}_sp']] for j in ARM]
                           for row in sim['log']], float)
            MS = np.array([[row[c[f'{j}_act']] for j in ARM]
                           for row in sim['log']], float)
            spf = np.isfinite(SP).all(axis=1)
            msf = np.isfinite(MS).all(axis=1)
            sl_sp = np.minimum(SP[spf] - elo, ehi - SP[spf])
            sl_ms = np.minimum(MS[msf] - elo, ehi - MS[msf])
            r['margins'] = {
                'setpoint_min_effective_slack_rad':
                    (float(sl_sp.min()) if len(sl_sp) else None),
                'setpoint_min_per_joint':
                    (np.round(sl_sp.min(0), 6).tolist() if len(sl_sp) else None),
                'measured_min_effective_slack_rad':
                    (float(sl_ms.min()) if len(sl_ms) else None),
                'measured_min_per_joint':
                    (np.round(sl_ms.min(0), 6).tolist() if len(sl_ms) else None),
                'n_setpoint_samples': int(spf.sum()),
                'n_measured_samples': int(msf.sum()),
                'note': ('執行端那一層只保護**設定點**；實測角在一階遲滯下'
                         '仍可能落在界外，故實測角另行核實、不列入成功判準'),
            }
            sp_viol = bool(len(sl_sp) and sl_sp.min() < 0.0)
            if sp_viol:
                bad.append(f'**設定點越過有效限位**：最小餘裕 '
                           f'{sl_sp.min():+.6f} rad')
        except Exception as e:
            r['margins'] = {'error': repr(e)}
            missing.append('margins')
    else:
        missing.append('log')

    # ---- 6 封存完整 ----
    # **鍵名是 archive_complete**（先前寫 ok／complete ⇒ 永遠讀成 False，
    # 而日誌明寫「封存完整」）。exit_code 一併記。
    r['archive'] = {'ok': bool(arch and arch.get('archive_complete')),
                    'exit_code': (arch or {}).get('exit_code'),
                    'failed_checks': [c['name'] for c in (arch or {}).get(
                        'checks', []) if not c.get('ok')]}
    if arch is None:
        bad.append('無 archive_check.json'); missing.append('archive_check')
    elif not r['archive']['ok']:
        bad.append(f'封存不完整（exit {r["archive"]["exit_code"]}）：'
                   f'未通過項 {r["archive"]["failed_checks"]}')

    # ---- 指標（不影響判定）----
    # 實測底盤位移：交棒後的**位姿**差分（不用回報的 twist —— 靜止基線已
    # 定位它在靜止後會停更）。協同接近的成果之一就是底盤真的參與了。
    r['base_motion'] = _base_motion(sim, (ho or {}).get('handover_sim_t'))
    # 保持窗誤差統計（由實錄重算的誤差序列取）
    r['hold_window'] = _hold_stats(sim, tgt, r['reach'])
    r['target'] = (None if not tgt else
                   {'tcp_world': tgt.get('tcp_world'),
                    'source': tgt.get('target_source'),
                    'stamp_sim_t': tgt.get('stamp_sim_t')})
    r['drawer'] = (None if not drw else
                   {'verdict': drw.get('verdict'),
                    'opening_max_abs_m':
                        drw['displacement']['opening_max_abs_m'],
                    'opening_tol_m': drw['displacement']['tol_m'],
                    'over_baseline': drw['displacement']['over_baseline'],
                    'contact_verdict': drw['contact']['verdict'],
                    'contact_fmag_max_n':
                        (drw['contact']['force'] or {}).get('fmag_max_n')})
    r['missing'] = missing
    r['violations'] = bad
    r['verdict'] = 'pass' if not bad else 'fail'
    r['first_failure'] = bad[0] if bad else None
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    r = assess(a.run_dir)
    out = a.out or os.path.join(a.run_dir, 'verdict.json')
    try:
        json.dump(r, open(out, 'w'), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass
    if a.quiet:
        print(r['verdict'])
    else:
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    return 0 if r['verdict'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
