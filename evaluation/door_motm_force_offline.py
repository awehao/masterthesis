#!/usr/bin/env python3
"""MotM 開門第 3、4 趟的手腕反力與底盤漂移離線分析（只讀紀錄；Codex 20261115_135017）。
  1. 力訊號座標與正負號：用接觸前（前導段）的列核對——假設「joint6 量到的 6 維量在 link6 座標、作用點＝joint6 原點」，
     轉到世界座標後應等於下游剛體（link6＋夾爪＋手指）重量的支撐力／力矩；兩種正負號都算，取相符者，不符就照實列出。
  2. 外力旋量：W_ext＝−(量測)−(下游重力)，先在 joint6 原點，再換到 TCP（力矩含力臂項 M_tcp＝M_o−r×F）。
  3. 門機構座標分量：鉸鏈在世界原點、軸 z；對 TCP 位置分成徑向（鉸鏈→把手）、切向（開門方向）、垂直。
  4. 關節力矩對照（開門段才有設定點）：G(q)＝τ_PD＋JᵀW_ext（靜止）；τ_PD＝kp(s−q)（−kd·q̇ 另列）。列 G、τ_PD、JᵀW_ext、殘差。
  5. 底盤：停止命令歸零後的實測位置／偏航變化率（各趟末 2 s 線性擬合）。
    python3 evaluation/door_motm_force_offline.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_body_model as BM
import door_open_exec_ctrl as XO
K = XO.K
DOWN = ['link6', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2']
G_ = 9.81


def down_gravity(q9):
    """下游剛體重力對 joint6 原點的旋量（世界座標）：F＝(0,0,−Mg)，M＝Σ r_i×(0,0,−m_i g)。"""
    T6 = K.fk(q9, 'link6')
    F, M = np.zeros(3), np.zeros(3)
    for b in DOWN:
        m, com, _ = BM.INERT[b]
        Tb = K.fk(q9, b)
        pc = Tb[:3, :3] @ np.asarray(com, float) + Tb[:3, 3]
        w = np.array([0.0, 0.0, -m * G_])
        F += w
        M += np.cross(pc - T6[:3, 3], w)
    return F, M, T6


def analyse(run):
    D = os.path.join(HERE, 'runs', run)
    L = np.load(os.path.join(D, 'log.npz'), allow_pickle=True)
    cols = [str(c) for c in L['cols']]
    R = np.asarray(L['rows'], float)
    i = cols.index
    res = json.load(open(os.path.join(D, 'result.json')))
    t = R[:, i('t')]
    q9 = lambda r: np.r_[r[i('bx')], r[i('by')], r[i('byaw')], r[i('q1'):i('q6') + 1]]
    Wm = lambda r: (r[i('Fx'):i('Fz') + 1].copy(), r[i('Tx'):i('Tz') + 1].copy())
    out = dict(run=run)
    # 1. 座標與正負號（接觸前：t 0.1–0.4 s，前導段）
    pre = [r for r in R if 0.1 <= r[i('t')] <= 0.4]
    chk = []
    for r in pre[::3]:
        Fg, Mg, T6 = down_gravity(q9(r))
        F, M = Wm(r)
        Fw, Mw = T6[:3, :3] @ F, T6[:3, :3] @ M
        chk.append(np.r_[Fw + Fg, Mw + Mg, Fw - Fg, Mw - Mg, Fw, Fg, Mw, Mg])
    c = np.mean(chk, axis=0)
    out['frame_check'] = dict(n=len(chk), down_mass_kg=round(sum(BM.INERT[b][0] for b in DOWN), 4),
                              meas_world_F=np.round(c[12:15], 3).tolist(), gravity_F=np.round(c[15:18], 3).tolist(), meas_world_M=np.round(c[18:21], 4).tolist(), gravity_M=np.round(c[21:24], 4).tolist(),
                              resid_if_meas_is_support=dict(F=np.round(c[0:3], 3).tolist(), M=np.round(c[3:6], 4).tolist()),
                              resid_if_meas_is_load=dict(F=np.round(c[6:9], 3).tolist(), M=np.round(c[9:12], 4).tolist()))
    sup = np.linalg.norm(c[0:3]) < np.linalg.norm(c[6:9])
    out['frame_check']['sign'] = 'meas＝母連桿對子連桿（支撐）' if sup else 'meas＝子連桿對母連桿（負載）'
    sgn = 1.0 if sup else -1.0

    def ext(r):
        """外界（把手／門）作用在下游剛體的旋量：世界座標，分別對 joint6 原點與 TCP。"""
        q = q9(r)
        Fg, Mg, T6 = down_gravity(q)
        F, M = Wm(r)
        Fw, Mw = sgn * (T6[:3, :3] @ F), sgn * (T6[:3, :3] @ M)
        Fe, Mo = -(Fw + Fg), -(Mw + Mg)
        Tt = K.fk(q, 'link_tcp')
        rr = Tt[:3, 3] - T6[:3, 3]
        Mt = Mo - np.cross(rr, Fe)
        p = Tt[:3, 3]
        er = np.array([p[0], p[1], 0.0]); er /= np.linalg.norm(er)
        et = np.array([er[1], -er[0], 0.0])                        # 開門方向（門角定義：φ 增加時把手沿 −θ̂）
        return dict(q=q, Fe=Fe, Mo=Mo, Mt=Mt, p=p, radial=float(Fe @ er), tangential_open=float(Fe @ et), vertical=float(Fe[2]), M_tcp_z=float(Mt[2]),
                    M_tcp_horiz=float(np.hypot(Mt[0], Mt[1])), tcp_lever_m=float(np.linalg.norm(rr)))

    def summ(sel, tag):
        E = [ext(r) for r in sel]
        if not E:
            return None
        f = lambda k: round(float(np.mean([e[k] for e in E])), 3)
        return dict(tag=tag, n=len(E), t=[round(float(sel[0][i('t')]), 2), round(float(sel[-1][i('t')]), 2)], door_deg=round(math.degrees(float(np.mean([r[i('door_q')] for r in sel]))), 3),
                    F_ext_world=np.round(np.mean([e['Fe'] for e in E], axis=0), 2).tolist(), F_norm=round(float(np.mean([np.linalg.norm(e['Fe']) for e in E])), 2),
                    radial_N=f('radial'), tangential_open_N=f('tangential_open'), vertical_N=f('vertical'),
                    M_at_tcp_world=np.round(np.mean([e['Mt'] for e in E], axis=0), 3).tolist(), M_at_joint6_world=np.round(np.mean([e['Mo'] for e in E], axis=0), 3).tolist(),
                    raw_meas_norm=round(float(np.mean([np.linalg.norm(r[i('Fx'):i('Fz') + 1]) for r in sel])), 2))
    tb = None
    for e in res.get('events', []):
        if e['what'] == 'bind_and_handover':
            tb = float(e['t'])
    out['t_bind'] = tb
    segs = []
    if tb is not None:
        for (a, b, tag) in ((tb - 0.10, tb - 0.01, '綁定前 0.1 s（已閉爪）'), (tb, tb + 0.05, '綁定後第一拍'), (tb + 0.10, tb + 0.25, '門起動'), (tb + 0.3, tb + 0.6, '減速中'), (t[-1] - 2.0, t[-1], '末 2 s（靜止）')):
            s_ = summ([r for r in R if a <= r[i('t')] <= b], tag)
            if s_:
                segs.append(s_)
    out['segments'] = segs
    # 4. 關節力矩對照（開門段）
    OS = os.path.join(D, 'open_steps.npz')
    if tb is not None and os.path.exists(OS):
        Z = np.load(OS, allow_pickle=True)
        C, M = Z['ctrl'], Z['meas']
        kp = float(res['gains']['kp'])
        bodies = BM.arm_bodies(K)
        jt = []
        for k in range(len(M) - 200, len(M), 20):
            tm = float(M[k, 0])
            r = R[int(np.argmin(np.abs(t - (tb + tm))))]
            e = ext(r)
            q = e['q']
            s = C[min(k + 1, len(C) - 1), -6:]
            Gq = BM.arm_gravity(K, q, bodies)                         # 廣義重力（τ_drive＝−G 才靜止）
            tau_pd = kp * (s - q[3:])
            J = K.jacobian(q, 'link_tcp')                              # 6×9（線、角）
            # 外界對下游的旋量經 TCP 作用在手臂：τ_ext＝Jᵀ[F; M_tcp]
            tau_ext = J[:, 3:9].T @ np.r_[e['Fe'], e['Mt']]
            jt.append(np.r_[tm, tau_pd, Gq, tau_ext, tau_pd + Gq + tau_ext])
        jt = np.mean(jt, axis=0)
        out['joint_torque_static_tail'] = dict(note='靜止平衡式：τ_PD＋G(廣義重力)＋JᵀW_ext＝0；resid＝三者之和（未建模項：夾爪夾持內力經 FixedJoint、接觸、模型慣量誤差）',
                                               tau_pd=np.round(jt[1:7], 2).tolist(), G=np.round(jt[7:13], 2).tolist(), JT_W_ext=np.round(jt[13:19], 2).tolist(), resid=np.round(jt[19:25], 2).tolist())
    # 5. 底盤漂移（末 2 s）
    sel = R[t >= t[-1] - 2.0]
    A = np.c_[sel[:, i('t')], np.ones(len(sel))]
    drift = {k: float(np.linalg.lstsq(A, sel[:, i(k)], rcond=None)[0][0]) for k in ('bx', 'by', 'byaw', 'door_q')}
    out['drift_last_2s'] = dict(bx_mps=round(drift['bx'], 6), by_mps=round(drift['by'], 6), byaw_rps=round(drift['byaw'], 6), door_rps=round(drift['door_q'], 6),
                                arm_rate_max=round(float(np.abs((sel[-1, i('q1'):i('q6') + 1] - sel[0, i('q1'):i('q6') + 1]) / 2.0).max()), 6))
    return out


if __name__ == '__main__':
    O = {r: analyse(r) for r in ('m8house_door_motm_3', 'm8house_door_motm_4')}
    json.dump(O, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    for r, o in O.items():
        print(r, json.dumps(o['frame_check'], ensure_ascii=False))
        for s in o['segments']:
            print('  ', json.dumps(s, ensure_ascii=False))
        print('  joint', json.dumps(o.get('joint_torque_static_tail'), ensure_ascii=False))
        print('  drift', o['drift_last_2s'])
