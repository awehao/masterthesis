#!/usr/bin/env python3
"""door_w2_mode6 HELD 監督器 r6（mode 2 釋放保持交接）純邏輯反例（Codex 20261026_021025 §2）。
先以 test_door_w2_mode.run_held 同一驅動進入 HELD，之後以腳本化的門角、手臂、底盤軌跡續跑；同步狀態 50 Hz、週期 20 Hz、套用回報 100 Hz。
  R1 未受理 mode 2 請求，門以 1°/s 轉 ⇒ 立即 FAIL moved_in_held（不暫緩，同 r4/r5）。
  R2 已受理 mode 2，提交 t_c、門自 t_c 起 1°/s（> 舊門檻 0.002 rad/s、< 保護 0.05 rad/s），權威狀態延遲 0.1 s 才得知
     ⇒ 暫緩→RELEASE_OPEN，持續發布；之後 3 s 門持續轉也不 FAIL；mode 3 交接成功。
  R3 已受理但權威狀態一直未到（ack／狀態遺失）⇒ 暫緩 0.2 s 逾時 FAIL moved_in_held。
  R4 RELEASE_OPEN 中門速 0.06 rad/s ⇒ FAIL door_rate_in_release（真正越界仍中止）。
  R5 RELEASE_OPEN 中手臂 0.01 rad/s ⇒ FAIL moved_in_release_open（手臂仍須靜止，門檻不變）。
  R6 RELEASE_OPEN 中底盤 2 mm/s ⇒ FAIL moved_in_release_open。
  R7 門在提交前已轉（t_c 晚於首次超限）⇒ FAIL moved_before_release。
  R8 門在提交前已轉、且無暫緩（未受理）時先轉後得知 ⇒ 依 R1 立即 FAIL；另：門靜止時得知提交但提交前窗口門速超限 ⇒ moved_before_release。
  R9 RELEASE_OPEN 中量測停更 0.15 s ⇒ FAIL measurement_stale。
  R10 RELEASE_OPEN 中套用非保持命令 ⇒ FAIL old_command_in_held（保持形核對沿用）。
  R11 暫緩中門又靜止 ⇒ 暫緩解除、仍在 HELD（sub held）。
另：REPLAY 以 door_wide_k3b_30_1 實錄（room 逐步底盤、門、手臂；同步 50 Hz 取樣）重放 r6，記錄結果（**不是設計通過的證據**）。

    python3 evaluation/test_door_w2_mode6.py
"""
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_sup_p4 as P4                     # noqa: E402
import door_w2_mode6 as M6                             # noqa: E402
import test_door_w2_mode as TM                         # noqa: E402

DEG = math.pi / 180.0


def to_held():
    sup = M6.HeldSupervisorP4b6(arm_win_s=0.5, held_max_s=60.0)
    o, th = TM.run_held(sup, t_end=30.0)
    assert o['state'] == 'HELD', o
    return sup


def drive(sup, t0, dur, door=lambda t: 0.14, arm=lambda t: 0.0, bx=lambda t: 0.6, pending2_at=None, commit_t=None, known_at=None,
          stale=None, bad_cmd_at=None, handover_at=None):
    """t0 起續跑 dur 秒。door/arm/bx：時間→量測。pending2_at：受理 mode 2 時刻；commit_t：提交時刻；known_at：求解端得知提交時刻。"""
    seq = 10_000
    k0 = int(round(t0 / 0.01))                    # run_held 止於 t0 − 0.01（最後同步 t0 − 0.02）
    res = {'n_pub': 0, 'n_nopub': 0, 'first_nopub': None, 'released': None, 'handed': None}
    for k in range(k0, k0 + int(round(dur / 0.01))):
        t = round(k * 0.01, 4)
        if pending2_at is not None and t >= pending2_at and not sup.pending2:
            sup.set_pending2(True, t)
        if known_at is not None and commit_t is not None and t >= known_at and res['released'] is None and sup.state == 'HELD':
            res['released'] = sup.release_open(commit_t, t)
        if handover_at is not None and t >= handover_at and res['handed'] is None:
            res['handed'] = sup.handover(t)
        if k % 2 == 0 and not (stale is not None and stale[0] <= t < stale[1]):
            sup.feed_state(k, t, bx(t), -0.53, 1.57, door(t), [arm(t)] + [0.0] * 5, wall=t)
        if k % 5 == 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(TM.HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                res['n_pub'] += 1
                sup.note_published(seq, a['zero'])
            else:
                res['n_nopub'] += 1
                res['first_nopub'] = res['first_nopub'] or t
        if sup.state == 'HELD':
            bad = bad_cmd_at is not None and t >= bad_cmd_at
            sup.applied(k, t, [0.01, 0.0, 0.0] if bad else list(TM.HOLD), [0.0] * 6, seq)
        if sup.state in ('DONE', 'FAIL'):
            break
    res['outcome'] = sup.outcome(t_end=t)
    return res


def ramp(t_c, rate, q0=0.14):
    return lambda t: q0 + max(0.0, t - t_c) * rate


class TR(unittest.TestCase):
    T0 = 30.0

    def test_r1_no_pending_immediate_fail(self):
        sup = to_held()
        r = drive(sup, self.T0, 2.0, door=ramp(30.5, 1.0 * DEG))
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_in_held'))
        self.assertLess(r['first_nopub'], 30.5 + 0.15)

    def test_r2_release_open_delayed_state(self):
        sup = to_held()
        tc = 30.505
        r = drive(sup, self.T0, 4.0, door=ramp(tc, 1.0 * DEG), pending2_at=30.3, commit_t=tc, known_at=tc + 0.1, handover_at=33.8)
        o = r['outcome']
        self.assertTrue(r['released'], o.get('release_checks'))
        self.assertEqual(r['n_nopub'], 0)
        self.assertTrue(r['handed'])
        self.assertEqual(o['state'], 'HANDED_OVER')
        self.assertEqual(o['release_commit_t'], tc)
        self.assertTrue(any(d.get('defer_start') for d in o['defer_log']), o['defer_log'])

    def test_r3_state_never_arrives(self):
        sup = to_held()
        r = drive(sup, self.T0, 2.0, door=ramp(30.505, 1.0 * DEG), pending2_at=30.3)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_in_held'))
        o = r['outcome']
        d0 = next(d['t'] for d in o['defer_log'] if d.get('defer_start'))
        self.assertAlmostEqual(r['first_nopub'] - d0, 0.25, delta=0.051)

    def test_r4_door_rate_over_protect(self):
        sup = to_held()
        tc = 30.505
        drive(sup, self.T0, 2.0, door=ramp(tc, 0.06), pending2_at=30.3, commit_t=tc, known_at=tc + 0.02)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'door_rate_in_release'))

    def test_r5_arm_moves(self):
        sup = to_held()
        tc = 30.505
        drive(sup, self.T0, 3.0, door=ramp(tc, 1.0 * DEG), arm=lambda t: max(0.0, t - 31.0) * 0.01, pending2_at=30.3, commit_t=tc, known_at=tc + 0.02)
        self.assertEqual((sup.state, sup.seq.why, sup.sub), ('FAIL', 'moved_in_release_open', 'release_open'))

    def test_r6_base_moves(self):
        sup = to_held()
        tc = 30.505
        drive(sup, self.T0, 3.0, door=ramp(tc, 1.0 * DEG), bx=lambda t: 0.6 + max(0.0, t - 31.0) * 0.002, pending2_at=30.3, commit_t=tc,
              known_at=tc + 0.02)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_in_release_open'))

    def test_r7_door_moves_before_commit(self):
        sup = to_held()
        drive(sup, self.T0, 2.0, door=ramp(30.4, 1.0 * DEG), pending2_at=30.3, commit_t=30.505, known_at=30.55)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_before_release'))

    def test_r8_pre_commit_window(self):
        sup = to_held()
        # 門在 30.40–30.45 短暫轉動後停住，暫緩於 30.60 前解除；30.62 才得知 30.50 的提交：提交前窗口（30.40–30.50）門速超限 ⇒ moved_before_release
        door = lambda t: 0.14 + (min(max(t - 30.40, 0.0), 0.05)) * 3.0 * DEG
        r = drive(sup, self.T0, 1.0, door=door, pending2_at=30.3, commit_t=30.50, known_at=30.62)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_before_release'))
        rc = r['outcome']['release_checks']
        self.assertIsNone(rc['t_defer0'])                                          # 由窗口規則拒收，不是暫緩起點規則
        self.assertGreater(rc['door_rate_pre'], M6.PSUP.StopTh().door)

    def test_r9_stale_in_release(self):
        sup = to_held()
        tc = 30.505
        drive(sup, self.T0, 3.0, door=ramp(tc, 1.0 * DEG), pending2_at=30.3, commit_t=tc, known_at=tc + 0.02, stale=(31.5, 31.65))
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'measurement_stale'))

    def test_r10_bad_cmd_in_release(self):
        sup = to_held()
        tc = 30.505
        drive(sup, self.T0, 3.0, door=ramp(tc, 1.0 * DEG), pending2_at=30.3, commit_t=tc, known_at=tc + 0.02, bad_cmd_at=31.5)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'old_command_in_held'))

    def test_r11_defer_clears(self):
        sup = to_held()
        door = lambda t: 0.14 + (min(max(t - 30.40, 0.0), 0.05)) * 3.0 * DEG
        r = drive(sup, self.T0, 1.5, door=door, pending2_at=30.3)
        self.assertEqual((sup.state, sup.sub), ('HELD', 'held'))
        self.assertTrue(any(d.get('defer_cleared') for d in r['outcome']['defer_log']))


def replay_k3b_30_1():
    """door_wide_k3b_30_1 實錄重放：自 HELD（實錄 66.47 s）前 12 s 起，以 room 逐物理步取每 2 步（50 Hz）餵入；
    受理 mode 2＝求解端收到請求（以任務端請求時刻 70.80 近似）；提交 70.91；得知＝提交＋0.04 s（權威狀態每 2 步發布＋傳遞，近似）。"""
    rd = lambda f: json.load(open(os.path.join(HERE, 'runs', 'door_wide_k3b_30_1', f)))
    room = rd('room_run.json')
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    sup = M6.HeldSupervisorP4b6(arm_win_s=0.5, held_max_s=60.0)
    TM.run_held(sup, t_end=30.0)                     # 只用於把監督器帶到 HELD；之後時間平移到實錄
    assert sup.state == 'HELD'
    shift = 66.0 - 30.0
    sup.t_held += shift
    sup.ws.buf.clear()
    sup.ws.last_ok = None
    sup._held_last = None
    out = {'n_pub': 0, 'first_nopub': None}
    released = None
    for s in room['steps']:
        k, t = s[0], s[1]
        if t < 65.0:
            continue                                     # 65–66 s 只餵量測（補滿 0.5 s 手臂窗），66 s 起才跑週期
        if t >= 70.80 and not sup.pending2:
            sup.set_pending2(True, t)
        if t >= 70.95 and released is None and sup.state == 'HELD':
            released = sup.release_open(70.91, t)
        if k % 2 == 0:
            b = s[I['base_xyth']]
            sup.feed_state(k, t, b[0], b[1], b[2], s[I['door_q_rad']], s[I['q_arm_meas']], wall=t)
        if k % 5 == 0 and t >= 66.0:
            a = sup.cycle(t, True, True, wall=t)
            if a['publish']:
                out['n_pub'] += 1
            else:
                out['first_nopub'] = out['first_nopub'] or t
        if sup.state in ('DONE', 'FAIL') or t > 72.0:
            break
    o = sup.outcome(t_end=t)
    return {'state': o['state'], 'fail_why': o['fail_why'], 'sub': o['sub'], 'released': released, 'release_checks': o['release_checks'],
            'last_speeds': sup.last_speeds, 'first_nopub': out['first_nopub'], 'thresholds': o['thresholds'], 'defer_log': o['defer_log'][:6]}


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'REPLAY':
        print(json.dumps(replay_k3b_30_1(), ensure_ascii=False, default=float, indent=1))
    else:
        unittest.main(verbosity=1)
