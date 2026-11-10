#!/usr/bin/env python3
"""門把手：視覺估計 → 預抓目標 → 幾何可行性（離線；Codex 20261110_225452 第 3 項；不閉爪、不開門、不跑物理）。
輸入：m8_door_detect.py 的開發集偵測（door_detect_dev.json）、擷取 meta（cam_T_fk、深度、內參）、視點登錄（底盤、手臂）。
目標：dl2_obs_to_target_v2.estimate（N-obs：法向由深度；預抓退讓 s＝0.03 m 沿 a_H 向外）。
  抓取模型 T_HG、a_H 與候選選擇用的工具參考旋轉＝dl3_vision_target_node.baseline_reference()（bar26 型把手的設計抓取；
  door_local_v3.yaml grasp 欄同一依據）——把手**類型**的抓取先驗，照實揭露；不讀把手位置，失敗也不回填任何舊停位或離線參考包。
環境模型（偵測端可得）：門扇與兩側牆＝地圖／門洞先驗（q＝0、牆面 y≥0）；立柱與橫桿＝類型先驗（尺寸），放在**估計**的把手座標上。
可行判準（沿用 door_offline_check.py 既有數字，不另訂）：IK 位置 < 1 mm、姿態 < 0.01 rad；六軸離 LITE6_SAFE ≥ 0.05 rad；
  手臂連桿點雲（手指、夾爪殼除外）對門扇／牆／立柱與橫桿 ≥ 0.005 m；夾爪殼對橫桿另報；底盤圓盤（0.30）對門扇／牆 ≥ 0.02 m。
流程（每個開發集有效視點，取第一格有效估計；計數單位＝視點）：
  A 當下構型：底盤不動，手臂 IK 到預抓與抓取（先以當下手臂為種子），判可行。
  B 不可達時：底盤沿當下車頭向前 0–1.0 m（0.05 步）× 側向 {0, ±0.1} m，依位移由小到大找第一個可行底盤；
    另核底盤自當下直線移到該位姿的圓盤掃掠（只對門扇／牆；未知箱體不在此模型）。
  C 接近掃掠：觀測手臂 → 預抓手臂的關節直線內插（11 點）對環境餘裕；預抓 → 抓取沿 a_H（s 0.03→0，5 點）IK 延續與餘裕（手指—橫桿為設計接觸）。
  D 事後真值核對（偵測、目標、IK 都完成後才讀 truth.json）：同一組關節對真值把手幾何的餘裕；抓取 TCP 對真值目標的誤差（沿工具 z／側向）。
不代表整機碰撞安全、動態可達或物理夾持成功；瞬移擷取不證明機器人能安全到達觀察姿態。
    python3 evaluation/m8_door_pregrasp_check.py m8doorcap_1 [--group dev|holdout]
分組輸入（Codex 232017 資料接線）：讀 door_detect_<group>.json、寫 door_pregrasp_<group>.json；保留集須凍結雜湊全部相符。
"""
import json
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_offline_check as DOC  # noqa: E402
import dl2_obs_to_target_v2 as E  # noqa: E402
from dl3_vision_target_node import baseline_reference  # noqa: E402

V3 = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/door_local_v3.yaml')))
K = DOC.K
IDX = DOC.IDX
LIM_M, ENV_M, BASE_M, RB = DOC.LIM_M, DOC.ENV_M, DOC.BASE_M, DOC.RB
S_PRE = E.S_RETREAT


def static_shapes():
    """門扇（q＝0）與兩側牆：地圖／門洞先驗。"""
    L, t = V3['leaf']['length'], V3['leaf']['thickness']
    z0, z1 = V3['leaf']['z']
    I3 = np.eye(3)
    W, T, Ht = V3['frame_clear_width'], V3['wall']['thickness'], V3['wall']['height']
    x0, x1 = V3['wall']['extent_x']
    rec = V3['wall'].get('hinge_side_recess', 0.0)
    return [('leaf', np.array([L / 2, t / 2, (z0 + z1) / 2]), I3, np.array([L / 2, t / 2, (z1 - z0) / 2])),
            ('wall_hinge_side', np.array([(x0 - rec) / 2, T / 2, Ht / 2]), I3, np.array([(-rec - x0) / 2, T / 2, Ht / 2])),
            ('wall_latch_side', np.array([(W + x1) / 2, T / 2, Ht / 2]), I3, np.array([(x1 - W) / 2, T / 2, Ht / 2]))]


def handle_shapes(T_WH):
    """立柱（類型先驗尺寸）放在把手座標 H：x 沿桿、y 指向門（內側）。回傳立柱方盒與橫桿（中心、軸、半長、半徑）。"""
    H = V3['handle']
    R, c = T_WH[:3, :3], T_WH[:3, 3]
    su, sn, sz = H['posts']['size']
    posts = [(f'post{k}', c + R @ np.array([du, H['standoff'] - sn / 2, 0.0]), R, np.array([su / 2, sn / 2, sz / 2])) for k, du in enumerate(H['posts']['along_u'])]
    return posts, (c, R[:, 0], H['bar']['length'] / 2, H['bar']['radius'])


def bar_dist(P, bar):
    c, u, hl, r = bar
    s = np.clip((P - c) @ u, -hl, hl)
    return np.linalg.norm(P - (c + s[:, None] * u), axis=1) - r


def env_eval(qf, T_WH):
    W = DOC.world_points(qf, DOC.ARM_LINKS)
    posts, bar = handle_shapes(T_WH)
    sh = static_shapes() + posts
    best, who = np.inf, None
    for n, P in W.items():
        for name, c, R, hs in sh:
            d = float(DOC.box_dist(P, c, R, hs).min())
            if d < best:
                best, who = d, f'{n}|{name}'
        if n not in DOC.FINGERS and n != 'uflite_gripper_link':
            d = float(bar_dist(P, bar).min())
            if d < best:
                best, who = d, f'{n}|bar'
    shell = float(bar_dist(W['uflite_gripper_link'], bar).min()) if 'uflite_gripper_link' in W else None
    base = min((float(DOC.box_dist(np.array([[qf[0], qf[1], c[2]]]), c, R, hs)[0]) - RB, name) for name, c, R, hs in static_shapes())
    return best, who, shell, base


def feasible(qf, T_WH):
    if qf is None:
        return False, dict(why='ik')
    m, j = DOC.jmargin(qf[IDX])
    env, who, shell, (bc, bwho) = env_eval(qf, T_WH)
    ok = m >= LIM_M and env >= ENV_M and bc >= BASE_M
    return ok, dict(joint_margin=round(m, 4), joint=j, env=round(env, 4), env_who=who, shell_bar=None if shell is None else round(shell, 4),
                    base_clear=round(bc, 4), base_who=bwho, why=None if ok else ('joint' if m < LIM_M else 'env' if env < ENV_M else 'base'))


def ik(base, q_arm_seed, T, extra_seeds=True):
    seeds = [np.asarray(q_arm_seed, float)] + (DOC.SEEDS[:3] if extra_seeds else [])
    best = None
    for sd in seeds:
        qf = np.zeros(9)
        qf[:3] = base
        qf[IDX] = sd
        r = DOC.solve_ik(K, qf, T)
        if not (r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2):
            continue
        m, _ = DOC.jmargin(r.q[IDX])
        if best is None or m > best[0]:
            best = (m, r.q)
        if m >= LIM_M:
            break
    return None if best is None else best[1]


def plan_at(base, q_arm, T_WH, T_pre, T_gr):
    q_pre = ik(base, q_arm, T_pre)
    okp, ip = feasible(q_pre, T_WH)
    if not okp:
        return None, dict(pre=ip)
    q_gr = ik(base, q_pre[IDX], T_gr, extra_seeds=False)
    okg, ig = feasible(q_gr, T_WH)
    if not okg:
        return None, dict(pre=ip, grasp=ig)
    return (q_pre, q_gr), dict(pre=ip, grasp=ig)


def sweeps(base, q_obs, q_pre, T_WH, T_WE_at):
    qf0 = np.r_[base, q_obs]
    j_env = []
    for a in np.linspace(0.0, 1.0, 11):
        qf = (1 - a) * qf0 + a * q_pre
        j_env.append(env_eval(qf, T_WH)[0])
    a_env, a_ok, prev = [], True, q_pre
    for s in np.linspace(S_PRE, 0.0, 5):
        q = ik(base, prev[IDX], T_WE_at(s), extra_seeds=False)
        ok, info = feasible(q, T_WH)
        a_env.append(info.get('env'))
        a_ok = a_ok and ok
        prev = q if q is not None else prev
    return dict(joint_interp_env_min=round(float(min(j_env)), 4), joint_interp_ok=bool(min(j_env) >= ENV_M),
                approach_env=a_env, approach_ok=bool(a_ok))


def base_sweep(b0, b1):
    out = np.inf
    for a in np.linspace(0.0, 1.0, 21):
        p = (1 - a) * np.asarray(b0[:2]) + a * np.asarray(b1[:2])
        for name, c, R, hs in static_shapes():
            out = min(out, float(DOC.box_dist(np.array([[p[0], p[1], c[2]]]), c, R, hs)[0]) - RB)
    return round(out, 4)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--group', choices=('dev', 'holdout'), default='dev')
    aa = ap.parse_args()
    run, group = aa.run, aa.group
    if group == 'holdout':
        import m8_door_detect as MD
        ok, bad = MD.verify_freeze()
        if not ok:
            print(f'保留集須凍結雜湊全部相符 ⇒ 拒跑（不符：{bad}）')
            return 2
    D = os.path.join(HERE, 'runs', run, 'wrist_door')
    det = json.load(open(os.path.join(D, f'door_detect_{group}.json')))
    meta = json.load(open(os.path.join(D, 'meta.json')))
    reg = json.load(open(os.path.join(WS, meta['registration'])))
    VP = {v['id']: v for v in reg['viewpoints']}
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = baseline_reference()
    rows = []
    t0 = time.perf_counter()
    for v in det['per_viewpoint']:
        rr = [r for r in det['rows'] if r['viewpoint'] == v['viewpoint'] and r['cls'] == 'center_valid']
        if v['n_center_valid'] < 2 or not rr:
            continue
        r = rr[0]
        f = FR[r['stem']]
        dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        obs = dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time'])
        est = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        row = dict(viewpoint=v['viewpoint'], stem=r['stem'], est_ok=est['ok'], est_why=est.get('why'), prior_used=est.get('prior_used'))
        if not est['ok']:
            rows.append(row)
            continue
        T_WH = np.asarray(est['T_WH'], float)

        def T_WE_at(s, T_WH=T_WH):
            return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
        T_pre, T_gr = T_WE_at(S_PRE), T_WE_at(0.0)
        vp = VP[v['viewpoint']]
        base0, q_obs = np.array(vp['base'], float), np.array(vp['q_arm'], float)
        sol, info = plan_at(base0, q_obs, T_WH, T_pre, T_gr)
        row['A_current_base'] = dict(ok=sol is not None, **info)
        base_used = base0
        if sol is None:
            h = base0[2]
            fwd, lat = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
            cands = sorted(((math.hypot(df, dl), df, dl) for df in np.arange(0.0, 1.0001, 0.05) for dl in (0.0, -0.1, 0.1)), key=lambda z: z[0])
            tried = 0
            for disp, df, dl in cands:
                b = np.r_[base0[:2] + df * fwd + dl * lat, h]
                tried += 1
                s2, i2 = plan_at(b, q_obs, T_WH, T_pre, T_gr)
                if s2 is not None:
                    sol, base_used = s2, b
                    row['B_moved_base'] = dict(ok=True, base=b.round(4).tolist(), displacement_m=round(disp, 3), forward_m=round(df, 3), lateral_m=dl,
                                               base_sweep_clear_min=base_sweep(base0, b), n_tried=tried, **i2)
                    break
            if sol is None:
                row['B_moved_base'] = dict(ok=False, n_tried=tried)
        if sol is not None:
            q_pre, q_gr = sol
            row['C_sweeps'] = sweeps(base_used, q_obs, q_pre, T_WH, T_WE_at)
            row['_q'] = dict(base=base_used.tolist(), q_pre=q_pre.tolist(), q_gr=q_gr.tolist())
        row['_T_WH_est'] = T_WH.tolist()
        rows.append(row)
        print(f"{v['viewpoint']}: A {row['A_current_base']['ok']}  B {row.get('B_moved_base', {}).get('ok')} {row.get('B_moved_base', {}).get('displacement_m')}  "
              f"C {row.get('C_sweeps', {}).get('joint_interp_ok')}/{row.get('C_sweeps', {}).get('approach_ok')}", flush=True)
    truth = json.load(open(os.path.join(D, 'truth.json')))                # ---- D 事後真值核對
    for row in rows:
        if '_q' not in row:
            continue
        Tt = np.array(truth['per_viewpoint'][row['viewpoint']]['M_world_handle_prim'])
        qp, qg = np.array(row['_q']['q_pre']), np.array(row['_q']['q_gr'])
        okp, ip = feasible(qp, Tt)
        okg, ig = feasible(qg, Tt)
        T_tool = K.fk(qg, 'link_tcp')
        T_true = np.asarray(E.OT.object_target(Tt, T_HG, a_H, 0.0)['T_WE'], float)
        e = T_tool[:3, 3] - T_true[:3, 3]
        z = T_true[:3, 2]
        row['D_truth'] = dict(pre_ok=okp, pre=ip, grasp_ok=okg, grasp=ig, tcp_err_mm=round(float(np.linalg.norm(e)) * 1e3, 2),
                              tcp_err_along_tool_z_mm=round(float(e @ z) * 1e3, 2), tcp_err_lateral_mm=round(float(np.linalg.norm(e - (e @ z) * z)) * 1e3, 2),
                              tcp_rot_err_deg=round(math.degrees(E.OT.geodesic(T_tool[:3, :3], T_true[:3, :3])), 3))
    n = len(rows)
    S = dict(run=run, group=group, n_viewpoints_valid_center=n, est_ok=sum(1 for r in rows if r['est_ok']),
             A_feasible_at_current_base=sum(1 for r in rows if r.get('A_current_base', {}).get('ok')),
             B_feasible_after_base_move=sum(1 for r in rows if r.get('B_moved_base', {}).get('ok')),
             B_displacement_m=sorted(r['B_moved_base']['displacement_m'] for r in rows if r.get('B_moved_base', {}).get('ok')),
             C_joint_interp_ok=sum(1 for r in rows if r.get('C_sweeps', {}).get('joint_interp_ok')), C_approach_ok=sum(1 for r in rows if r.get('C_sweeps', {}).get('approach_ok')),
             D_truth_grasp_ok=sum(1 for r in rows if r.get('D_truth', {}).get('grasp_ok')), D_truth_pre_ok=sum(1 for r in rows if r.get('D_truth', {}).get('pre_ok')),
             D_tcp_err_mm=[r['D_truth']['tcp_err_mm'] for r in rows if 'D_truth' in r], D_tcp_err_along_z_mm=[r['D_truth']['tcp_err_along_tool_z_mm'] for r in rows if 'D_truth' in r],
             grasp_prior='bar26 型設計抓取（baseline_reference：phf_01_M 的 link_tcp FK；door_local_v3 grasp 同依據）', s_pre_m=S_PRE,
             wall_s=round(time.perf_counter() - t0, 1),
             note='離線幾何；門扇／牆＝地圖先驗、立柱／橫桿＝類型先驗放在估計把手座標；底盤掃掠只對門扇／牆；不代表整機碰撞安全、動態可達或物理夾持成功')
    json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_pregrasp_{group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.exit(main())
