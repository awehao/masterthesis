#!/usr/bin/env python3
"""S0a 停止負測與流程測試：假受控對象（設定點模型＋重力偏移＋合成接觸）驅動**同一份** S0aController（Isaac 腳本也用它）。
無接觸物理：這裡只驗判定、相位、停止與命令邏輯，不驗抓持。Codex 要求的資料失效／求解失敗／失抓負測在此。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_offline as BO                                           # noqa: E402
import bottle_s0_ctrl as C                                            # noqa: E402

K = BO.K
ID = BO.IDENT
ALPHA = np.asarray(ID['alpha'], float)
TOP, R_B, H_B, MASS = 0.60, 0.015, 0.20, 0.2
BX, BY, HG = 0.0, 0.42, 0.70
BASE = np.array([-0.10, 0.0, 0.0])
_Tg = C.grasp_target(BX, BY, HG)
_Tp = _Tg.copy()
_Tp[:3, 3] -= 0.06 * _Tg[:3, 2]
Q_READY, T_READY = C.ready_from_pre(BASE, _Tp)
D_TRUE = np.array([0.0, 0.018, -0.008, 0.0, 0.0, 0.0])                # 重力下垂式偏移（驗 d̂）


def bottle_T(z=TOP, x=BX, y=BY):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


class Fake:
    """手臂：q⁺＝q＋α(s−q＋d)；瓶子：抓住後隨 TCP；手指合成力。"""

    def __init__(self, ctrl, f_hold=2.5):
        self.c, self.q = ctrl, np.r_[BASE, Q_READY]
        self.t, self.Tb, self.grasped, self.Trel = 0.0, bottle_T(), False, None
        self.f_hold = f_hold
        self.hist = []

    def meas(self, **ov):
        T_tcp = K.fk(self.q, 'link_tcp')
        closed = self.c.f_cmd <= 0.0040                                 # 指間距到 30 mm
        near = np.linalg.norm(T_tcp[:3, 3] - np.array([BX, BY, HG])) < 0.03 or self.grasped
        f = (self.f_hold, self.f_hold) if (closed and near) else (0.0, 0.0)
        if closed and near and not self.grasped:
            self.grasped, self.Trel = True, np.linalg.inv(T_tcp) @ self.Tb
        if self.grasped and self.c.st.phase in ('LIFT', 'CARRY'):
            self.Tb = T_tcp @ self.Trel
        gap = self.Tb[2, 3] - TOP
        m = dict(f_fingers=f, t_contact=self.t, T_bottle=self.Tb.copy(), t_bottle=self.t,
                 f_table=(MASS * 9.81 if gap <= 1e-6 else 0.0), forbidden={'nonfinger_bottle': 0.0, 'robot_table': 0.0}, penetration=0.0)
        m.update(ov)
        return m

    def run(self, t_end, fault=None):
        while self.t < t_end and not self.c.done:
            ov = fault(self.t, self) if fault else {}
            q_meas = ov.pop('q', self.q)
            out = self.c.step(self.t, q_meas, **self.meas(**ov))
            self.q[3:] = self.q[3:] + ALPHA * (out['arm_target'] - self.q[3:] + D_TRUE)
            self.hist.append((self.t, out['phase'], float(K.fk(self.q, 'link_tcp')[2, 3]), out['stopped'], self.c.s.copy(), self.q.copy()))
            self.t += 0.01
        return self


def ctrl():
    return C.S0aController(K, ID, C.grasp_target(BX, BY, HG), T_READY, Q_READY, TOP, R_B, t_reach=4.0)


BAD, NCHK, LOG = [], 0, []


def chk(name, cond, info=''):
    global NCHK
    NCHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


def stop_of(f):
    return (f.c.judge.stop or {}).get('why')


def lifted_above(f, dz=0.002):
    """閉爪請求之後 TCP 是否高於抓取高度（備妥姿態本來就較高，不能看全程）。"""
    tc = f.c.judge.t_close
    return tc is not None and max(h[2] for h in f.hist if h[0] >= tc) > HG + dz


def main():
    # N1 正常流程
    f = Fake(ctrl()).run(30.0)
    r = f.c.report()
    LOG.append(('N1', r['result'], r['events'], r['d_hat'], r['solve_ms']))
    chk('N1 成功', r['success'], f"{r['result']} stop {r['stop']}")
    chk('N1 相位依序', [e['to'] for e in r['events'] if e['what'] == 'phase'] == ['PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY'], str(r['events']))
    chk('N1 d̂ 估到重力偏移', abs(r['d_hat'][1] - D_TRUE[1]) < 0.004, str(r['d_hat']))
    chk('N1 抓持證據 ≥ 1 s 才抬升', r['t_lift_start'] - r['t_close'] >= 1.0, f"{r['t_lift_start']} {r['t_close']}")
    # S1 資料失效：閉爪後瓶位姿停更
    t_c = r['t_close']
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(t_bottle=t_c + 0.2) if t > t_c + 0.2 else {})
    chk('S1 瓶位姿過期 ⇒ 停止', stop_of(f) == 'data_stale_bottle_pose', str(stop_of(f)))
    chk('S1 未抬升', not lifted_above(f))
    # S2 接觸 NaN：不當零、不當過，過期後停止
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_fingers=(float('nan'), float('nan'))) if t > t_c + 0.6 else {})
    chk('S2 接觸無效 ⇒ 停止', stop_of(f) == 'data_stale_contact', str(stop_of(f)))
    chk('S2 無證據', f.c.judge.t_grasp is None)
    # S3 求解失敗（抬升中）
    c = ctrl()
    t_l = r['t_lift_start']
    c.fail_inject = lambda t: t > t_l + 0.3
    f = Fake(c).run(30.0)
    chk('S3 求解失敗 ⇒ 停止', stop_of(f) == 'solver_fail', str(stop_of(f)))
    chk('S3 停止有效（1 s 內手臂 < 1e-3 rad/s）', (c.stop_ok or {}).get('ok') is True, str(c.stop_ok))
    chk('S3 手指維持閉合命令', c.f_cmd == C.FINGER_CLOSED, str(c.f_cmd))
    s_after = [h[4] for h in f.hist if h[0] > c.stopped_t + 0.2]
    chk('S3 設定點於在途命令後凍結', len(s_after) > 5 and np.abs(np.array(s_after) - s_after[0]).max() < 1e-12)
    # S4 失抓：任一指
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_fingers=(2.5, 0.05)) if t > t_l + 0.3 else {})
    chk('S4 單指失抓 ⇒ 停止', stop_of(f) == 'grasp_loss_finger2', str(stop_of(f)))
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_fingers=(2.5, 0.05)) if t_l + 0.3 < t < t_l + 0.45 else {})
    chk('S4b 0.15 s 短暫掉力不停', f.c.report()['success'], str(stop_of(f)))
    # S5 禁止接觸脈衝
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(forbidden={'nonfinger_bottle': 0.2, 'robot_table': 0.0}) if 5.0 <= t < 5.02 else {})
    p = f.c.report()['forbidden_pulses']['nonfinger_bottle']
    chk('S5 2 步脈衝不觸發但照錄', stop_of(f) is None and p['peak'] == 0.2 and p['steps'] == 2, str(p))
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(forbidden={'nonfinger_bottle': 0.2, 'robot_table': 0.0}) if 5.0 <= t < 5.035 else {})
    chk('S5b 連續 3 步 ⇒ 停止', stop_of(f) == 'forbidden_contact:nonfinger_bottle', str(stop_of(f)))
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(forbidden={'nonfinger_bottle': float('nan'), 'robot_table': 0.0}) if t < 2.0 else {})
    chk('S5c NaN 記無資料、不當零也不當碰撞', f.c.report()['forbidden_pulses']['nonfinger_bottle']['n_nodata'] > 100)
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(penetration=0.002) if t > 5.0 else {})
    chk('S5d 穿透 ≥ 1 mm ⇒ 停止', stop_of(f) == 'forbidden_penetration', str(stop_of(f)))
    # S6 閉爪前推瓶
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(T_bottle=bottle_T(x=BX + 0.006)) if t > 3.0 else {})
    chk('S6 推瓶 ⇒ 停止、不閉爪成功', (stop_of(f) or '').startswith('bottle_pushed') and f.c.judge.t_grasp is None, str(stop_of(f)))
    # S7 證據期限：力始終不到；期限不因時間前進或重送刷新；無證據不抬升
    c = ctrl()
    f = Fake(c, f_hold=0.3).run(30.0)
    chk('S7 期限 ⇒ 停止', stop_of(f) == 'grasp_evidence_deadline', str(stop_of(f)))
    chk('S7 約 5 s 觸發', abs(c.judge.stop['t'] - c.judge.t_close - 5.0) < 0.05, str(c.judge.stop))
    chk('S7 無證據不抬升', not lifted_above(f))
    t0_close = c.judge.t_close
    c.judge.on_close_request(99.0, None)
    chk('S7b 重送不刷新', c.judge.t_close == t0_close)
    # S8 瓶子卡住（持續有桌面支撐）⇒ 抬離期限
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_table=1.9) if t > t_l else {})
    chk('S8 判據矛盾 ⇒ 未通過', stop_of(f) == 'lift_deadline', str(stop_of(f)))
    # S9 缺桌面資料不當零
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_table=None) if t > t_l else {})
    chk('S9 缺支撐資料 ⇒ 不算離桌', stop_of(f) == 'lift_deadline' and f.c.judge.res['lifted'] is False, str(stop_of(f)))
    # S10 證據連續性：中途掉一步要重算
    t_e = t_c + 0.9
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(f_fingers=(2.5, 0.4)) if t_e <= t < t_e + 0.015 else {})
    chk('S10 掉力重算連續', f.c.judge.t_grasp is not None and f.c.judge.t_grasp - (t_e + 0.015) >= 0.99, str(f.c.judge.t_grasp))
    # S11 延遲佇列＋非零命令下停止：逐 tick 減速不超上限、最終設定點靜止
    c = ctrl()
    c.fail_inject = lambda t: t > 2.0                                  # 展臂移動中失敗（命令非零、設定點未追上）
    f = Fake(c).run(6.0)
    chk('S11 移動中求解失敗 ⇒ 停止', stop_of(f) == 'solver_fail', str(stop_of(f)))
    ev = [e for e in c.events if e['what'] == 'stop'][0]
    chk('S11 停止當下手臂命令非零（負測情境成立）', ev['u_arm_max'] > 0.01, str(ev))
    chk('S11 佇列清空、命令歸零', np.abs(f.c.pipe[0]).max() == 0.0 and np.abs(f.c.u_last).max() == 0.0)
    ds = [np.abs(h[4] - f.hist[i][4]).max() / 0.01 for i, h in enumerate(f.hist[1:]) if h[0] > c.stopped_t]
    chk('S11 設定點速率單調受限（≤ v_arm）', max(ds) <= 1.0, f'{max(ds):.3f}')
    chk('S11 停止有效', (c.stop_ok or {}).get('ok') is True, str(c.stop_ok))
    # S12 關節量測非有限
    f = Fake(ctrl()).run(30.0, lambda t, s: dict(q=np.full(9, np.nan)) if t > 3.0 else {})
    chk('S12 關節 NaN ⇒ 停止', stop_of(f) == 'joint_state_nonfinite', str(stop_of(f)))
    print(f'S0a 負測 {NCHK} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD)))
    print('N1', json.dumps(LOG[0][1], ensure_ascii=False), '事件', [(e['t'], e.get('to', e['what'])) for e in LOG[0][2]], 'd̂', np.round(LOG[0][3], 4).tolist(), '求解 ms', LOG[0][4])
    return 0 if not BAD else 1


if __name__ == '__main__':
    sys.exit(main())
