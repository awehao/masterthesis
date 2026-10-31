#!/usr/bin/env python3
"""K1 步驟 2（Codex 20261031_153959）：以 rt_43 現用包為底局部替換，只提速 R3、S8（×1.5），其他段逐位元保留。
  門段 pb60o → pb60p：段 41 換成 results/k1/r3_x1.5.json 的段 41（door_ref_r3_retime2 流程，v_max 0.045、w_pk 0.225）；
      m4r.r3_w_cap 0.19→0.285、新增 m4r.r3_v_cap_body 0.0525（R3 引導改當下本體座標逐軸限幅）；r3_retime2 記錄換成 ×1.5 版。
  返程 v3 → v4：A 段 S8 前綴（原 579 列）換成 door_ref_overlap_v1.retime 以 V_MAX 0.045 重排的列；平面路徑保留、
      平移與偏航各自重排時程（**不是純時間縮放**，朝向沿路配置隨倍率改變）；其後各列（PRE_NAV 之後）逐位元保留；s8_overlap.end_row 更新。
核對：未替換的段／列與來源相同；接點（段 41 首末＝原首末；S8 首列＝原首列、末列→下一原列的跳差）；新段完整幾何監看（真最小以公尺比較）stop／warn 皆 0 才寫檔。
    python3 evaluation/door_ref_k1_splice.py
"""
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_ref_overlap_v1 as OV                       # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
from geom_min_track import MinTracker                  # noqa: E402

DOOR = os.path.join(HERE, 'results', 'door')
K_SPEED = 1.5
V_CAP_BODY, W_CAP_R3 = round(0.035 * K_SPEED, 6), round(0.19 * K_SPEED, 6)
BOX_LIN, BOX_ANG = round(0.035255 * 2.0, 6), round(0.1999 * 2.0, 6)          # 介面框（×2）


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def monitor(rows, q_rad, qf):
    g = json.load(open(os.path.join(DOOR, 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    t, st, wn, tr = 0.0, 0, 0, MinTracker()
    for i, x in enumerate(rows):
        t += 0.05
        dec, det = mon.step(t, np.asarray(x, float), q_rad, qf, 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        tr.update(det, i)
    return {'n': len(rows), 'stop': st, 'warn': wn, 'true_min_mm': tr.as_mm()}


def body_peaks(B, dt=0.05):
    d = np.diff(B, axis=0) / dt
    d[:, 2] = np.diff(np.unwrap(B[:, 2])) / dt
    c, s = np.cos(B[:-1, 2]), np.sin(B[:-1, 2])
    vb = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]]
    return np.abs(vb).max(0)


def door_pack():
    src = os.path.join(DOOR, 'door_regrasp_ref_pb60o.json')
    out = os.path.join(DOOR, 'door_regrasp_ref_pb60p.json')
    P = json.load(open(src))
    X = json.load(open(os.path.join(HERE, 'results', 'k1', 'r3_x1.5.json')))
    X1 = json.load(open(os.path.join(HERE, 'results', 'k1', 'r3_x1.json')))
    if X1['segments']['41'] != P['segments']['41']:
        raise SystemExit('pb60o 段 41 與 r3_x1（＝pb60n）不同，拼接基準不符')
    s0, s1 = P['segments']['41'], X['segments']['41']
    for k in ('base', 'arm'):
        if not (np.allclose(s0[k][0], s1[k][0], atol=0, rtol=0) and np.allclose(s0[k][-1], s1[k][-1], atol=1e-12)):
            raise SystemExit(f'段 41 首末 {k} 與原包不同')
    i0 = int(X['r3_retime2']['i0'])
    if any(s0[k][:i0 + 1] != s1[k][:i0 + 1] for k in s0 if isinstance(s0[k], list)):
        raise SystemExit('段 41 前導（列 0..i0）不同')
    Q = copy.deepcopy(P)
    Q['segments']['41'] = s1
    Q['m4r']['r3_w_cap'] = W_CAP_R3
    Q['m4r']['r3_v_cap_body'] = V_CAP_BODY
    Q['r3_retime2'] = X['r3_retime2']
    Q['k1_splice'] = {'src': os.path.basename(src), 'src_sha256': sha(src), 'seg41_from': 'results/k1/r3_x1.5.json',
                      'seg41_from_sha256': sha(os.path.join(HERE, 'results', 'k1', 'r3_x1.5.json')), 'k_speed': K_SPEED,
                      'r3_w_cap': W_CAP_R3, 'r3_v_cap_body': V_CAP_BODY, 'box_lin_ang': [BOX_LIN, BOX_ANG]}
    same = [k for k in P['segments'] if k != '41' and P['segments'][k] == Q['segments'][k]]
    B = np.asarray(s1['base'], float)
    pk = body_peaks(B)
    ff = np.abs(np.asarray(s1['base_v'], float)).max(0)
    mon = monitor(np.c_[B, np.asarray(s1['arm'], float)], math.radians(float(P['q_ref_deg'])), 0.0089)
    rep = {'T_seg41_s': [len(s0['t']) * 0.05, len(s1['t']) * 0.05], 'other_segments_identical': sorted(same), 'n_other': len(P['segments']) - 1,
           'body_vel_peak': pk.round(4).tolist(), 'ff_world_peak': ff.round(4).tolist(), 'monitor': mon}
    print(f"pb60p：段 41 {rep['T_seg41_s'][0]:.2f}→{rep['T_seg41_s'][1]:.2f} s；其他段相同 {len(same)}/{rep['n_other']}；本體差分速度峰 {rep['body_vel_peak']}（引導上限 {V_CAP_BODY}/{W_CAP_R3}、框 {BOX_LIN:.6f}/{BOX_ANG}）；"
          f"監看 stop {mon['stop']} warn {mon['warn']} 真最小 {mon['true_min_mm']}")
    if len(same) != rep['n_other'] or mon['stop'] or mon['warn'] or pk[0] > V_CAP_BODY or pk[1] > V_CAP_BODY or pk[2] > W_CAP_R3:
        raise SystemExit('pb60p 核對未過，不寫檔')
    Q['k1_splice']['offline'] = rep
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    json.dump(Q, open(out, 'w'), ensure_ascii=False)
    print('寫入', out, sha(out)[:16])


def return_pack():
    src = os.path.join(DOOR, 'door_return_ref_v3.json')
    out = os.path.join(DOOR, 'door_return_ref_v4.json')
    R = json.load(open(src))
    R1 = json.load(open(os.path.join(DOOR, 'door_return_ref_v1.json')))
    A1, A3 = R1['A'], R['A']
    i8 = [i for i, r in enumerate(A1) if r['stage'].startswith('S8')]
    ie = i8[0] + 800
    j8 = [i for i, r in enumerate(A3) if r['stage'].startswith('S8')][0]
    n_old = int(R['s8_overlap']['end_row']) + 1
    if A3[j8 + n_old:] != A1[ie + 1:]:
        raise SystemExit('v3 S8 前綴之後的列與 v1 原列不同，拼接基準不符')
    X = np.array([r['x'] for r in A1])
    v_e = float(np.hypot(*(X[ie + 1, :2] - X[ie, :2]) / OV.DT))
    w_e = float((X[ie + 1, 2] - X[ie, 2]) / OV.DT)
    V0 = OV.V_MAX
    OV.V_MAX = V0 * K_SPEED
    try:
        new, T, vc, wc = OV.retime(X[i8[0]:ie + 1, :2], (float(X[i8[0], 2]), float(X[ie, 2])), 0.0, v_e, 0.0, w_e)
    finally:
        OV.V_MAX = V0
    armA = X[i8[0], 3:]
    tpl = A3[j8]
    rows = [dict(tpl, x=[float(v) for v in np.r_[x, armA]]) for x in new]
    Q = copy.deepcopy(R)
    Q['A'] = A3[:j8] + rows + A3[j8 + n_old:]
    Q['s8_overlap'] = {'T': T, 'vc': vc, 'wc': wc, 'end_row': len(rows) - 1, 'k_speed': K_SPEED, 'V_MAX': round(V0 * K_SPEED, 6),
                       'note': '平面路徑保留；平移與偏航各自重排時程（非純時間縮放）'}
    Q['k1_splice'] = {'src': os.path.basename(src), 'src_sha256': sha(src), 'k_speed': K_SPEED}
    Bn = np.array([r['x'][:3] for r in rows])
    jump_in = np.abs(np.asarray(A3[j8 - 1]['x'][:3]) - Bn[0]).max()
    nxt = np.asarray(A3[j8 + n_old]['x'][:3], float)
    jump_out = float(np.hypot(*(nxt[:2] - Bn[-1, :2])))
    step_typ = float(np.hypot(*(Bn[-1, :2] - Bn[-2, :2])))
    pk = body_peaks(np.vstack([Bn, nxt]))
    mon = monitor(np.c_[Bn, np.tile(armA, (len(Bn), 1))], math.radians(59.9), 0.0089)
    rep = {'T_s8_s': [n_old * 0.05, len(rows) * 0.05], 'end_row': [n_old - 1, len(rows) - 1], 'prefix_identical': A3[:j8] == Q['A'][:j8],
           'suffix_identical': A3[j8 + n_old:] == Q['A'][j8 + len(rows):], 'jump_in_max': float(jump_in), 'jump_out_m': jump_out,
           'step_last_m': step_typ, 'v_e': v_e, 'w_e': w_e, 'body_vel_peak': pk.round(4).tolist(), 'monitor': mon}
    print(f"v4：S8 前綴 {rep['T_s8_s'][0]:.2f}→{rep['T_s8_s'][1]:.2f} s，end_row {rep['end_row'][0]}→{rep['end_row'][1]}；前後列相同 {rep['prefix_identical']}／{rep['suffix_identical']}；"
          f"首接點差 {jump_in:.2e}、末列→下一原列 {jump_out * 1e3:.2f} mm（末步 {step_typ * 1e3:.2f} mm）；本體速度峰 {rep['body_vel_peak']}；監看 stop {mon['stop']} warn {mon['warn']} 真最小 {mon['true_min_mm']}")
    if not (rep['prefix_identical'] and rep['suffix_identical']) or mon['stop'] or mon['warn'] or pk[0] > BOX_LIN or pk[1] > BOX_LIN or pk[2] > BOX_ANG:
        raise SystemExit('v4 核對未過，不寫檔')
    Q['k1_splice']['offline'] = rep
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    json.dump(Q, open(out, 'w'), ensure_ascii=False)
    print('寫入', out, sha(out)[:16])


if __name__ == '__main__':
    door_pack()
    return_pack()
