#!/usr/bin/env python3
"""DL2 評估端：兩開發群組 186 格 × 四條路徑（G0／L1 × N-obs／N-prior）（規格 DL2_obs_to_target_spec.md draft-2）。

真值只在此檔：T_WH* = [I | c*]，c*＝逐格 handle_center_world_at_capture（資產 yaw 0）。偵測結果不重算，取自凍結的
DL0_dev_eval_v3.json（核對 freeze_dl0_model_v2.sha256 內的雜湊）。描述性結果、不設門檻。

    python3 evaluation/dl2_eval.py            # 輸出 results/vision/DL2_dev_eval_r3.json（已存在則拒絕覆寫）

r2（審查必修 1）：逐格保存實際候選（T_WH、T_WE_s0、T_WE_s、normal_W、候選索引與選擇距離）、
時間（重播擷取 rendering_time 作 t_obs、query_t＝同一時刻、age_s；read_time、pose_time 另存；來源實錄時間 src_record_t 分欄），
以及共用配置（T_HG、a_H、工具參考旋轉）與程式／輸入版本。r1 檔（DL2_dev_eval.json）保留不改。
r3（封存前小修）：拒絕列也保存 t_obs／query_t／age_s（時間檢查後即寫入）與原始時間 time_raw。r2 檔保留不改。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import dl0_autolabel as AL                           # noqa: E402
import dl2_obs_to_target as E                        # noqa: E402
import object_target_geometry as OT                  # noqa: E402

VIS = os.path.join(HERE, 'results', 'vision')
EVAL = os.path.join(VIS, 'DL0_dev_eval_v3.json')
FREEZE = os.path.join(VIS, 'freeze_dl0_model_v2.sha256')
SRC = {'traj_wg4b_f02_P': 'd1_dev_f02P/wrist_v0', 'traj_mt_b1_02_P': 'd1_hold_mt02P/wrist_v0'}
BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0)]
ROUTES = [('G0', 'N-obs'), ('G0', 'N-prior'), ('L1', 'N-obs'), ('L1', 'N-prior')]
PARK = [-0.136412, 0.560, 1.297349]
Q_GRASP = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]


def R_grasp():
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    return K.fk(np.array(PARK + Q_GRASP, float), 'link_tcp')[:3, :3].copy()


def verify_eval_hash():
    h = hashlib.sha256(open(EVAL, 'rb').read()).hexdigest()
    rel = os.path.relpath(EVAL, WS)
    for line in open(FREEZE):
        if line.strip().endswith(rel) or line.strip().endswith('DL0_dev_eval_v3.json'):
            assert line.split()[0] == h, 'DL0_dev_eval_v3.json 與凍結清單不符'
            return h
    raise AssertionError('DL0_dev_eval_v3.json 不在凍結清單')


def errors(est, c_true, T_HG, a_H):
    """誤差定義（draft-2 §5）。真值 R* = I（x* 沿桿、y* 指向內側）。"""
    T_true = np.eye(4)
    T_true[:3, 3] = c_true
    xs, ys = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    Rh = est['T_WH'][:3, :3]
    xh, yh = Rh[:, 0], Rh[:, 1]
    yp = yh - (yh @ xs) * xs
    yp /= np.linalg.norm(yp)
    t0 = OT.object_target(T_true, T_HG, a_H, 0.0)['T_WE']
    ts = OT.object_target(T_true, T_HG, a_H, E.S_RETREAT)['T_WE']
    deg = math.degrees
    return {
        'center_mm': float(np.linalg.norm(est['T_WH'][:3, 3] - c_true) * 1e3),
        'rot_geodesic_deg': deg(OT.geodesic(Rh, np.eye(3))),
        'axial_deg': deg(math.acos(min(1.0, abs(float(xh @ xs))))),
        'roll_deg': deg(math.atan2(float(np.linalg.norm(np.cross(yp, ys))), float(yp @ ys))),
        'axis_sign_agrees': bool(xh @ xs > 0),
        'tcp_s0_mm': float(np.linalg.norm(est['T_WE_s0'][:3, 3] - t0[:3, 3]) * 1e3),
        'tcp_s0_deg': deg(OT.geodesic(est['T_WE_s0'][:3, :3], t0[:3, :3])),
        'tcp_s_mm': float(np.linalg.norm(est['T_WE_s'][:3, 3] - ts[:3, 3]) * 1e3),
        'tcp_s_deg': deg(OT.geodesic(est['T_WE_s'][:3, :3], ts[:3, :3])),
    }


def q(v, p):
    return round(float(np.percentile(v, p)), 3) if len(v) else None


def summarize(rows, route):
    out = {}
    for lo, hi in BINS + [(3.0, 99.0)]:
        rr = [r for r in rows if r['kind'] == 'positive' and lo <= r['dist'] < hi]
        ok = [r['routes'][route] for r in rr if r['routes'][route]['ok']]
        rej = {}
        for r in rr:
            x = r['routes'][route]
            if not x['ok']:
                k = x['why'].split(':')[0] + (':' + x['why'].split(':')[1] if x['why'].startswith('detector') else '')
                rej[k] = rej.get(k, 0) + 1
        s = {'n_positive': len(rr), 'n_candidate': len(ok), 'reject_reasons': rej}
        for key in ('center_mm', 'rot_geodesic_deg', 'axial_deg', 'roll_deg', 'tcp_s0_mm', 'tcp_s0_deg', 'tcp_s_mm', 'tcp_s_deg'):
            v = [o['err'][key] for o in ok]
            s[key] = {'median': q(v, 50), 'p95': q(v, 95), 'max': (round(max(v), 3) if v else None)}
        s['axis_sign_agrees'] = sum(o['err']['axis_sign_agrees'] for o in ok)
        out[f'{lo}-{hi}'] = s
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(VIS, 'DL2_dev_eval_r3.json'))
    out_p = ap.parse_args().out
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    h = verify_eval_hash()
    ev = json.load(open(EVAL))
    Rg = R_grasp()
    T_HG, a_H = E.baseline_grasp(Rg)
    res = {'schema': 'dl2_dev_eval/1', 'source_eval': os.path.relpath(EVAL, WS), 'source_eval_sha256': h,
           'note': '描述性；真值只用於評估；geometry_contract: unchecked；新鮮度未驗收（同一擷取時刻，年齡 0）', 'groups': {},
           'accounting': {},
           'config': {'T_HG': T_HG.tolist(), 'a_H': a_H.tolist(), 'R_ref_tool': Rg.tolist(),
                      'R_ref_source': 'phf_01_M 停車位姿＋q_grasp 的 link_tcp FK', 's_retreat_m': E.S_RETREAT,
                      'time_semantics': 't_obs＝重播擷取 rendering_time；query_t＝同一時刻（離線同時刻生成，age 0）；src_record_t＝來源實錄時間（分欄，不混用）'},
           'versions': {rel: hashlib.sha256(open(os.path.join(WS, rel), 'rb').read()).hexdigest() for rel in (
               'evaluation/dl2_obs_to_target.py', 'evaluation/dl2_eval.py', 'evaluation/object_target_geometry.py',
               'evaluation/d1_handle_detect.py', 'evaluation/dl0_autolabel.py', 'evaluation/models/omni_bot_wholebody_expanded.urdf',
               'evaluation/results/vision/DL0_dev_eval_v3.json')}}
    for grp, cap in SRC.items():
        G = ev['groups'][grp]
        frs = {f['n']: f for f in AL.frames_of(cap)}
        mt = {f['n']: f for f in json.load(open(os.path.join(AL.RUNS, cap, 'meta.json')))['frames']}
        rows = []
        for r in G['rows']:
            fr = frs.get(r['n'])
            m = mt.get(r['n'], {})
            rec = {'n': r['n'], 'kind': r['kind'], 'flags': r['flags'], 'dist': r['dist'], 'routes': {},
                   'time': {'replay_rendering_t': m.get('rendering_time'), 'replay_read_t': m.get('read_time'),
                            'replay_pose_t': m.get('pose_time'), 'src_record_t': m.get('src_record_t')}}
            if fr is None or fr['c'] is None or fr['c_source'] != 'per_frame':
                rec['evidence'] = 'evidence_insufficient'
                rows.append(rec)
                continue
            dep = AL.load_depth(fr['depth_path'])
            c_true = np.asarray(fr['c'], float)
            for path, nm in ROUTES:
                d = r[path]
                obs = {'path': path, 'L1': d['L1'], 'L2': d['L2'], 'reject': d['reject'], 'reject_L2': d['reject_L2'],
                       'depth': dep, 'K': fr['K'], 'T_cam': fr['T'], 't_obs': m.get('rendering_time')}
                est = E.estimate(obs, nm, Rg, T_HG, a_H, query_t=m.get('rendering_time'))
                x = {'ok': est['ok'], 'why': est.get('why'), 'prior_used': est['prior_used'], 'freshness': est.get('freshness'),
                     'normal_diag': est.get('normal_diag'), 'geometry_contract': est['geometry_contract'],
                     't_obs': est.get('t_obs'), 'query_t': est.get('query_t'), 'age_s': est.get('age_s'), 'time_raw': est.get('time_raw')}
                if est['ok']:
                    x['err'] = errors(est, c_true, T_HG, a_H)
                    x['candidate_index'] = est['candidate_index']
                    x['candidate_distances_rad'] = est['candidate_distances']
                    x['normal_W'] = est['normal_W']
                    for k in ('T_WH', 'T_WE_s0', 'T_WE_s'):
                        x[k] = np.asarray(est[k]).tolist()
                rec['routes'][f'{path}/{nm}'] = x
            rows.append(rec)
        acc = {'n_source_rows': len(G['rows']), 'n_rows': len(rows),
               'positive': sum(r['kind'] == 'positive' for r in rows), 'negative': sum(r['kind'] == 'negative' for r in rows),
               'excluded': sum(r['kind'] == 'excluded' for r in rows),
               'evidence_insufficient': sum(r.get('evidence') == 'evidence_insufficient' for r in rows)}
        acc['negative_with_candidate'] = {f'{p}/{m}': sum(1 for r in rows if r['kind'] == 'negative' and r['routes'].get(f'{p}/{m}', {}).get('ok'))
                                          for p, m in ROUTES}
        res['accounting'][grp] = acc
        res['groups'][grp] = {f'{p}/{m}': summarize(rows, f'{p}/{m}') for p, m in ROUTES}
        res['groups'][grp]['rows'] = rows
        print(grp, json.dumps(acc, ensure_ascii=False))
        for p, m in ROUTES:
            S = res['groups'][grp][f'{p}/{m}']
            print('  ', f'{p}/{m}', {b: (v['n_candidate'], v['n_positive'], v['center_mm']['median'], v['roll_deg']['median'])
                                     for b, v in S.items() if v['n_positive']})
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('寫出', out_p)


if __name__ == '__main__':
    main()
