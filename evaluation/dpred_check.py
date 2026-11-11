#!/usr/bin/env python3
"""偏差補償速度一致性原型的離線核對（真函式；不啟動模擬）。失敗輸入＝lateslow_fake_late_swdiag_r1 的切換當拍（τ>T，rel_ref 只依 base_end／v_c／T_pre）。
  1 同一輸入：常數 D 的速度差重現 0.037（拒絕）；前推 D 後速度差與分類。
  2 參考連續：_rel_target 模型下，相鄰預覽步的手臂相對參考（TCP 相對底盤）速度＝相位項＋偏差項；對照常數 D。
  3 手臂能力：補償底盤橫移所需的 TCP 相對速度換算關節速度（該構型的 Jacobian 最小奇異值粗估），對 v_arm。
  4 有界：橫偏朝 0 收斂不越過 0；前推時間夾在 D_HORIZON；變化率夾限。
  5 預設 hook：既有控制器的 _switch_check 解析到 tpas 檔。
判定不過 ⇒ 離開碼 1。    python3 evaluation/dpred_check.py OUT.json"""
import inspect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_r0_approach as R0                                   # noqa: E402
import bottle_s0b_ctrl as B                                       # noqa: E402
import bottle_w1_dpred_ctrl as DP                                 # noqa: E402
import bottle_w1_side_ctrl as SC                                  # noqa: E402
import bottle_w1_tpas_ctrl as TS                                  # noqa: E402

FAIL, res = [], {}


def chk(n, c, info=None):
    res[n] = dict(ok=bool(c), info=info)
    print(('PASS ' if c else 'FAIL ') + n, '' if info is None else json.dumps(info, ensure_ascii=False, default=float)[:320], flush=True)
    if not c:
        FAIL.append(n)


def main():
    d = json.load(open(os.path.join(HERE, 'results/bottle/lateslow_fake_late_swdiag_r1.json')))
    x = d['swdiag'][-1]
    qb, ub = np.array(x['qp_base']), np.array(x['u_last_body'])
    D = TS._tb(np.r_[0.0, x['D']['dy'], x['D']['dth']])
    T_pre = np.eye(4)
    T_pre[:3, 3] = [0.0042 + 0.0, 0.4564, 0.70]                         # 位置只影響 dp（不比較）；速度與之無關
    y_lane = 0.0287
    P = dict(T=x['T'], base_end=np.array([qb[0] - 0.1 * x['tau_over_T'], y_lane, 0.0]), v_c_eff=x['v_c_eff'], T_pre=T_pre)
    r_old = TS.switch_check(R0.K, qb, ub, P, x['tau_eff'], x['rate'], T_pre, D=D)
    rates = DP.d_rate(ub, x['D']['dth'], 0.0)
    r_new = DP.switch_check(R0.K, qb, ub, P, x['tau_eff'], x['rate'], T_pre, D, rates)
    chk('1a 常數 D：重現速度差 0.037（紀錄 %.4f）' % x['dv'], abs(r_old['dv'] - x['dv']) < 2e-3 and r_old['dv'] > TS.SW_V, dict(dv=r_old['dv'], sw_v=TS.SW_V))
    chk('1b 前推 D：速度差在門檻內', r_new['dv'] <= TS.SW_V and abs(r_new['dv_const_D'] - r_old['dv']) < 1e-9, dict(dv=r_new['dv'], rates=[round(v, 4) for v in rates]))
    # 2 參考速度：相對參考在 dt 內的變化
    dt = B.DT

    def rel(delta, pred):
        Dk = DP.d_at(x['D']['dy'], x['D']['dth'], rates[0], rates[1], delta)[0] if pred else D
        return (np.linalg.inv(Dk) @ TS.rel_ref(R0.K, P, x['tau_eff'] + x['rate'] * delta))[:3, 3]
    v_c = (rel(dt, False) - rel(0.0, False)) / dt
    v_p = (rel(dt, True) - rel(0.0, True)) / dt
    seq = np.array([rel(k * dt, True) for k in range(B.N + 1)])
    d2 = np.abs(np.diff(seq, 2, axis=0)).max() / dt ** 2
    chk('2 相對參考速度：前推版多出橫向補償項、預覽二階差分有界', abs((v_p - v_c)[1] + rates[0]) < 5e-3 and d2 < 0.5, dict(v_const=v_c.round(4).tolist(), v_pred=v_p.round(4).tolist(), acc_max=round(float(d2), 4)))
    # 3 手臂能力（粗估）：|Δv_tcp|／σ_min(J_pos)
    q_pre = np.asarray(R0.Q_VHIGH, float)
    J = np.zeros((3, 6))
    f0 = R0.K.fk(np.r_[0, 0, 0, q_pre], 'link_tcp')[:3, 3]
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1e-5
        J[:, j] = (R0.K.fk(np.r_[0, 0, 0, q_pre + e], 'link_tcp')[:3, 3] - f0) / 1e-5
    smin = float(np.linalg.svd(J, compute_uv=False).min())
    need = float(abs(rates[0])) / max(smin, 1e-6)
    chk('3 補償橫移所需關節速度（粗估）≤ 0.5·v_arm', need <= 0.5 * B.W_ARM['v_arm'], dict(dy_rate=round(rates[0], 4), sigma_min=round(smin, 3), qd_need=round(need, 3), v_arm=B.W_ARM['v_arm'],
                                                                                 note='以 V_HIGH 構型的位置 Jacobian 最小奇異值粗估，不是該拍實際構型'))
    # 4 過零連續、預測邊界、速率超界、相鄰控制拍連續
    ys = [DP.d_at(-0.001, 0.0, 0.04, 0.0, k * dt)[1] for k in range(B.N + 1)]
    vy = np.diff(ys) / dt
    chk('4a 過零不截斷：預測速度全程 0.04（橫偏 −1 mm 起）', np.allclose(vy, 0.04, atol=1e-9) and ys[1] > 0, dict(y=[round(v, 4) for v in ys[:4]], v_min=float(vy.min()), v_max=float(vy.max())))
    yb = [DP.d_at(-0.04, 0.0, 0.04, 0.0, t_)[1] for t_ in (DP.D_HORIZON - 0.01, DP.D_HORIZON, DP.D_HORIZON + 0.5, -0.3)]
    chk('4b 預測邊界：超過上限保持上限值、負時間取當拍值', abs(yb[1] - yb[2]) < 1e-12 and abs(yb[3] + 0.04) < 1e-12 and yb[0] < yb[1], [round(v, 5) for v in yb])

    class Stub:
        u_last = np.r_[0.10, 0.15, 0.0, np.zeros(6)]
        hd, s_ref = None, 0.0
        dp_stat = dict(n=0, n_clip=0, dy_rate_max=0.0, dth_rate_max=0.0)
        t = 1.0
    import bottle_s0_logic as L0
    Stub.judge = L0.S0Judge.__new__(L0.S0Judge)
    Stub.judge.stop, Stub.judge.log = None, []
    r_over = DP.W1PlaceController._d_rates(Stub(), dict(D=D))
    chk('4c 速率超界 ⇒ 受控停止 dpred_rate_out_of_domain', r_over == (0.0, 0.0) and Stub.dp_stat['n_clip'] == 1 and (Stub.judge.stop or {}).get('why') == 'dpred_rate_out_of_domain', dict(rates=r_over, stat=Stub.dp_stat))
    # 相鄰控制拍：第 k 拍預覽的第 1 步 vs 第 k+1 拍的當拍值（底盤照命令走時應一致）；底盤橫向速度在兩拍間變化 a_y·DT 時的參考跳動
    y0, vy0 = -0.04, 0.04
    pred_next = DP.d_at(y0, 0.0, vy0, 0.0, dt)[1]
    act_next = y0 + vy0 * dt
    jump_same = abs(pred_next - act_next)
    a_y = 0.4                                                           # 底盤橫向加速度上界（A_SLEW 同級）
    act_next2 = y0 + vy0 * dt + 0.5 * a_y * dt ** 2
    jump_acc = abs(pred_next - act_next2)
    chk('4d 相鄰控制拍參考連續：等速時 0；底盤橫向加速 0.4 m/s² 時跳動 ≤ 1 mm', jump_same < 1e-12 and jump_acc <= 1e-3, dict(jump_same=jump_same, jump_acc_mm=round(jump_acc * 1e3, 3), dt=dt))
    # 5 MRO
    a = os.path.basename(inspect.getsourcefile(SC.W1PlaceController._switch_check))
    b = os.path.basename(inspect.getsourcefile(DP.W1PlaceController._switch_check))
    c = os.path.basename(inspect.getsourcefile(DP.W1PlaceController._rel_target))
    chk('5 hook 解析', a == 'bottle_w1_tpas_ctrl.py' and b == 'bottle_w1_dpred_ctrl.py' and c == 'bottle_w1_dpred_ctrl.py', dict(base=a, dpred=b, rel=c))
    res['verdict'] = dict(ok=not FAIL, fails=FAIL)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
    print('判定', 'PASS' if not FAIL else 'FAIL', FAIL)
    sys.exit(0 if not FAIL else 1)


if __name__ == '__main__':
    main()
