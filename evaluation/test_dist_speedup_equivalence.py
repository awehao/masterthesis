"""距離節點等價加速的核對：**同一批輸入必須產出完全相同的列**。

加速只做兩件事，都不改約束內容：
  1. 去重鍵集合改為**增量維護**（原本每個障礙物迴圈都對全部列重建 ⇒ O(n²)）
  2. 障礙物表面速度改為**逐配對批次**（原本逐列呼叫）

本檔用**樸素參考實作**（原本的寫法）與現行實作跑同一批輸入，
逐列逐欄位比對距離、下界、配對索引、速度與有效性。
"""
from __future__ import annotations
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import (                        # noqa
    STATUS_OK, VOBS_OK, VOBS_STATIC, expand_pair_rows, forced_pair_rows,
    model_twist, parse_obstacles, surface_point_velocity)
from ammr_wholebody_mpc.arm_link_geometry import (                        # noqa
    arm_link_names, obstacle_distance_matrix, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa
from coman_zero_cmd_feasible import hold_state                             # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
PAIR_ROWS = ['link4:*', 'link5:*', 'link6:*', 'uflite_finger1:*',
             'uflite_finger2:*', 'uflite_gripper_link:*']
EXEMPT = ['uflite_finger1:handle_bar', 'uflite_finger2:handle_bar']


def naive_vobs_per_row(o, twist):
    """**舊寫法**：逐列各算一次。"""
    def f(obstacle, p_surf):
        if not obstacle.model:
            return np.zeros(3), VOBS_STATIC
        V = surface_point_velocity(twist, obstacle.T_world_link[:3, 3],
                                   np.asarray(p_surf, float)[None, :])
        if V is None or not np.isfinite(V).all():
            return np.zeros(3), 0
        return V[0], VOBS_OK
    return f


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    st = hold_state()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    xml = open(URDF, encoding='utf-8').read()
    q = np.zeros(len(K.dof_names))
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']
    sp = subprocess.run([sys.executable, os.path.join(HERE,
                         'coman_obstacle_specs.py'), str(st['unit'][0]),
                         str(st['unit'][1])], capture_output=True, text=True,
                        check=True)
    obs = parse_obstacles([x for x in sp.stdout.split() if x.strip()])
    T0 = np.eye(4)
    T0[0, 3] = st['unit'][0]
    T0[1, 3] = st['drawer_y0'] - st['opening']
    T1 = T0.copy()
    T1[1, 3] -= 0.0005                      # 抽屜在動 ⇒ 速度非零，才測得到
    for o in obs:
        if o.model == 'drawer_body':
            o.T_world_link = T1
    tw = model_twist(T0, 0.0, T1, 0.05)
    check('前提：抽屜 twist 非零（否則測不到速度路徑）',
          tw is not None and abs(tw[0][1]) > 1e-6, f'  v={np.round(tw[0],4)}')

    names = [o.name for o in obs]
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    links = arm_link_names(xml)
    pairs, _ = expand_pair_rows(PAIR_ROWS, EXEMPT, names)

    def build(incremental, batched):
        rows, dup = [], 0
        for li, nm in enumerate(links):
            Tw = K.fk(q, nm)
            Sx = S[nm]
            W = (Sx.points @ Tw[:3, :3].T) + Tw[:3, 3]
            D, V = obstacle_distance_matrix(W, obs)
            jm = np.argmin(D, axis=1)
            ixr = np.arange(len(W))
            dd, vv = D[ixr, jm], V[ixr, jm]
            wh = [names[k] for k in jm]
            dmin = float(dd.min())
            sel = np.nonzero(dd <= dmin + Sx.rho)[0]
            sel = sel[np.argsort(dd[sel])]
            seen = set()
            for k in sel:
                nh = vv[k] / max(abs(float(dd[k])), 1e-9)
                rows.append(list(W[k]) + list(nh)
                            + [float(dd[k]), float(STATUS_OK), 0.0, 0.0,
                               float(li)] + list(Sx.points[k])
                            + [float(Sx.rho), float(names.index(wh[k])),
                               0.0, 0.0, 0.0, float(VOBS_STATIC), 0.0, 0.0])
                seen.add((li, names.index(wh[k]),
                          round(float(Sx.points[k][0]), 12),
                          round(float(Sx.points[k][1]), 12),
                          round(float(Sx.points[k][2]), 12)))
            for obn in pairs.get(nm, []):
                oi = names.index(obn)
                if batched:
                    def vb(o_, pts):
                        if not o_.model:
                            return np.zeros((len(pts), 3)), VOBS_STATIC
                        Vv = surface_point_velocity(tw, o_.T_world_link[:3, 3],
                                                    np.atleast_2d(pts))
                        return ((Vv, VOBS_OK) if Vv is not None
                                and np.isfinite(Vv).all()
                                else (np.zeros((len(pts), 3)), 0))
                    ex = forced_pair_rows(W, Sx.points, obs[oi], oi, li, Sx.rho,
                                          STATUS_OK, 0.0, lambda p, v: 0.0, 3.0,
                                          0, vobs_batch=vb, d_col=D[:, oi],
                                          v_col=V[:, oi])
                else:
                    # 樸素版：逐列取速度後自行組列（與 forced_pair_rows 同式）
                    dcol, vcol = D[:, oi], V[:, oi]
                    dm = float(dcol.min())
                    s2 = np.nonzero(dcol <= dm + Sx.rho)[0]
                    s2 = s2[np.argsort(dcol[s2])]
                    f = naive_vobs_per_row(obs[oi], tw)
                    ex = []
                    for k in s2:
                        nh = vcol[k] / max(abs(float(dcol[k])), 1e-9)
                        vo, vst = f(obs[oi], W[k] + vcol[k])
                        ex.append(list(W[k]) + list(nh)
                                  + [float(dcol[k]), float(STATUS_OK), 0.0, 0.0,
                                     float(li)] + list(Sx.points[k])
                                  + [float(Sx.rho), float(oi)]
                                  + [float(vo[0]), float(vo[1]), float(vo[2]),
                                     float(vst)] + [0.0, 0.0])
                if incremental:
                    kept = []
                    for r in ex:
                        key = (int(r[10]), int(r[15]), round(float(r[11]), 12),
                               round(float(r[12]), 12), round(float(r[13]), 12))
                        if key in seen:
                            dup += 1
                            continue
                        seen.add(key)
                        kept.append(r)
                else:
                    sn = {(int(r[10]), int(r[15]), round(float(r[11]), 12),
                           round(float(r[12]), 12), round(float(r[13]), 12))
                          for r in rows}
                    kept = [r for r in ex
                            if (int(r[10]), int(r[15]), round(float(r[11]), 12),
                                round(float(r[12]), 12),
                                round(float(r[13]), 12)) not in sn]
                    dup += len(ex) - len(kept)
                rows.extend(kept)
        return np.asarray(rows, float), dup

    A, dupA = build(incremental=False, batched=False)   # 舊：全重建 ＋ 逐列
    B, dupB = build(incremental=True, batched=True)     # 新：增量 ＋ 批次
    check('列數相同', A.shape == B.shape, f'  {A.shape} vs {B.shape}')
    if A.shape == B.shape:
        check('**所有欄位逐位元相同**', np.array_equal(A, B),
              f'  最大差 {np.abs(A-B).max():.3e}')
        for nm_, c in (('距離 d', 6), ('rho', 14), ('障礙物索引', 15),
                       ('速度 vox', 16), ('速度 voy', 17),
                       ('有效性 vobs', 19), ('下界 dlb', 20),
                       ('下界有效 dlbv', 21)):
            check(f'{nm_} 相同', np.array_equal(A[:, c], B[:, c]))
    check('去重丟棄數相同', dupA == dupB, f'  {dupA} vs {dupB}')
    check('確有速度非零的列（真的走到速度路徑）',
          bool((np.abs(B[:, 16:19]).sum(axis=1) > 0).any()))

    print('距離節點加速等價性：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
