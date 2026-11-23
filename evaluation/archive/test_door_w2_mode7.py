#!/usr/bin/env python3
"""door_w2_mode7 反例（Codex 20261026_023226 §1 三項誤收＋閉爪卸載候選的求解端子狀態與 mode 2 准入；**合成資料**）。
驅動沿用 test_door_w2_mode6（run_held 進 HELD 後腳本化續跑）；權威測試沿用 test_door_w2_mode5_chain 的真 E2 命令鏈世界。
Codex 三項：
  C1 暫緩 30.55 起、提交通知 30.76 到達（> 0.2 s）且回呼先於下一週期 ⇒ on_commit 當下拒收 moved_in_held（r6 會接受，對照）。
  C2 清空速度窗口後提交 ⇒ pre_commit_evidence_missing（r6 door_rate_pre None 仍放行，對照）；
     暫緩中超限後恢復低速，再得知較晚的提交 ⇒ moved_before_commit（超限不抹除）。
  C3 validated_commit：他趟訊息、缺 commits、applied_confirmed 非 True、req_seq 不符、提交時刻早於受理或晚於現在、已中止 ⇒ 全拒；
     Codex 重現序列（本趟 mode 2 無 commits → 他趟帶提交時刻）⇒ 皆不產生提交時刻。
交接（無卸載）：
  V1 門 1°/s、狀態延遲 0.1 s ⇒ release_open、不停發、可交接；V2 門 0.06 ⇒ door_rate_in_release；V3 手臂動 ⇒ moved_in_release_open。
卸載候選：
  W1 commit1 → unload（仿 W1e 瞬態：門 +0.24°、底盤 3 mm、手臂 0.004 rad，τ 0.5 s）→ unloaded → commit2（門 1°/s）→ release_open → 交接；
  W2 mode 1 通知延遲 0.1 s（瞬態已開始）⇒ 暫緩（k＝1 容許）後 unload；
  W3 卸載中底盤偏離 25 mm ⇒ base_dev_in_unload；W4 不收斂（門 0.003 rad/s）⇒ unload_timeout；W5 卸載中門 0.06 ⇒ door_rate_in_unload；
  W6 提交次序：卸載啟用卻先得知 mode 2 ⇒ commit_order；W7 mode 2 提交時刻早於 unloaded ⇒ commit_before_reconverge。
ModeAuthority7（真 E2 命令鏈）：
  A1 gate2 假 ⇒ mode 2 請求 nack unload_evidence；A2 請求時真、提交當下假 ⇒ unload_evidence_at_commit、仍 mode 1；A3 gate2 真 ⇒ 1→2→3 正常。

    python3 evaluation/test_door_w2_mode7.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_sup_p4 as P4                     # noqa: E402
import door_w2_mode6 as M6                             # noqa: E402
import door_w2_mode7 as M7                             # noqa: E402
import test_door_w2_mode as TM                         # noqa: E402
import test_door_w2_mode5_chain as T5                  # noqa: E402

DEG = math.pi / 180.0
TAU = 0.5


def to_held(cls=M7.HeldSupervisorP4b7, **kw):
    sup = cls(arm_win_s=0.5, held_max_s=60.0, **kw)
    o, _ = TM.run_held(sup, t_end=30.0)
    assert o['state'] == 'HELD', o
    return sup


def tr(t0, amp, t):
    return amp * (1 - math.exp(-max(0.0, t - t0) / TAU)) if t0 is not None else 0.0


def drive(sup, t0, dur, door=lambda t: 0.14, arm=lambda t: 0.0, bx=lambda t: 0.6, pend=(), commits=(), handover_at=None, clear_at=None,
          r6=False):
    """pend：[(t, k)] 受理請求；commits：[(t_known, k, t_c)]。r6：用 r6 介面（release_open／set_pending2）作對照。"""
    seq = 10_000
    k0 = int(round(t0 / 0.01))
    res = {'n_nopub': 0, 'first_nopub': None, 'ok': {}, 'handed': None}
    done = set()
    for kk in range(k0, k0 + int(round(dur / 0.01))):
        t = round(kk * 0.01, 4)
        for tp, k in pend:
            if t >= tp and ('p', k) not in done:
                done.add(('p', k))
                sup.set_pending2(True, t) if r6 else sup.set_pending(k, True, t)
        if clear_at is not None and t >= clear_at and 'clr' not in done:
            done.add('clr')
            sup.ws.buf.clear()
        for tk, k, tc in commits:
            if t >= tk and ('c', k) not in done and sup.state == 'HELD':
                done.add(('c', k))
                res['ok'][k] = sup.release_open(tc, t) if r6 else sup.on_commit(k, tc, t)
        if handover_at is not None and t >= handover_at and res['handed'] is None:
            res['handed'] = sup.handover(t)
        if kk % 2 == 0:
            sup.feed_state(kk, t, bx(t), -0.53, 1.57, door(t), [arm(t)] + [0.0] * 5, wall=t)
        if kk % 5 == 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(TM.HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
            else:
                res['n_nopub'] += 1
                res['first_nopub'] = res['first_nopub'] or t
        if sup.state == 'HELD':
            sup.applied(kk, t, list(TM.HOLD), [0.0] * 6, seq)
        if sup.state in ('DONE', 'FAIL'):
            break
    res['outcome'] = sup.outcome(t_end=t)
    return res


def ramp(t_c, rate, q0=0.14):
    return lambda t: q0 + max(0.0, t - t_c) * rate


class TC3(unittest.TestCase):
    def test_c1_notice_after_deadline(self):
        door = ramp(30.505, 1.0 * DEG)                                     # 首次超限於 30.55 週期
        sup = to_held()
        r = drive(sup, 30.0, 1.0, door=door, pend=[(30.3, 2)], commits=[(30.76, 2, 30.505)])
        self.assertEqual(r['ok'].get(2), False)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_in_held'))
        self.assertEqual(r['outcome']['checks']['2'].get('why'), 'notice_after_defer_deadline')
        self.assertAlmostEqual(r['outcome']['checks']['2']['t_exc_first'], 30.55, places=6)
        sup6 = to_held(M6.HeldSupervisorP4b6)                                # 對照：r6 接受
        r6 = drive(sup6, 30.0, 1.0, door=door, pend=[(30.3, 2)], commits=[(30.76, 2, 30.505)], r6=True)
        self.assertTrue(r6['ok'].get(2))

    def test_c2_evidence(self):
        sup = to_held()
        drive(sup, 30.0, 1.0, pend=[(30.3, 2)], commits=[(30.52, 2, 30.505)], clear_at=30.51)       # 清空後、下一週期前得知
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'pre_commit_evidence_missing'))
        sup6 = to_held(M6.HeldSupervisorP4b6)
        r6 = drive(sup6, 30.0, 1.0, pend=[(30.3, 2)], commits=[(30.52, 2, 30.505)], clear_at=30.51, r6=True)
        self.assertTrue(r6['ok'].get(2))
        self.assertIsNone(r6['outcome']['release_checks']['door_rate_pre'])
        # 暫緩中超限（30.40–30.45 轉動）後恢復低速，0.15 s 內得知 30.50 的提交 ⇒ moved_before_commit（不抹除）
        door = lambda t: 0.14 + min(max(t - 30.40, 0.0), 0.05) * 3.0 * DEG
        sup = to_held()
        r = drive(sup, 30.0, 1.0, door=door, pend=[(30.3, 2)], commits=[(30.56, 2, 30.50)])
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'moved_before_commit'))
        self.assertEqual(r['outcome']['checks']['2'].get('why'), 'exceed_before_commit')

    def test_c3_validated_commit(self):
        good = {'run_id': 'r', 'mode_seq': 2, 'aborted': None, 'commits': [{'mode_seq': 1, 't': 20.0, 'applied_confirmed': True},
                                                                            {'mode_seq': 2, 't': 21.0, 'applied_confirmed': True}],
                'commit_req_seqs': [1, 2]}
        self.assertEqual(M7.validated_commit(good, 'r', 2, 2, 20.5, 21.1), (21.0, None))
        bad = [(dict(good, run_id='other'), 'wrong_run_or_malformed'), (dict(good, commits=None), 'commits_missing'),
               (dict(good, commits=[good['commits'][0], dict(good['commits'][1], applied_confirmed=None)]), 'not_applied_confirmed'),
               (dict(good, commit_req_seqs=[1, 7]), 'req_seq_mismatch'), (dict(good, aborted={'why': 'x'}), 'aborted'),
               (dict(good, mode_seq=1), 'not_committed')]
        for st, why in bad:
            self.assertEqual(M7.validated_commit(st, 'r', 2, 2, 20.5, 21.1), (None, why))
        self.assertEqual(M7.validated_commit(good, 'r', 2, 2, 21.5, 22.0)[1], 'commit_time_inconsistent')   # 早於受理
        self.assertEqual(M7.validated_commit(good, 'r', 2, 2, 20.5, 20.9)[1], 'commit_time_inconsistent')   # 晚於現在
        # Codex 重現：本趟 mode 2 但缺 commits，接著他趟帶提交時刻
        seq_msgs = [{'run_id': 'r', 'mode_seq': 2, 'aborted': None}, dict(good, run_id='other')]
        self.assertTrue(all(M7.validated_commit(m, 'r', 2, 2, 20.5, 21.1)[0] is None for m in seq_msgs))


class TV(unittest.TestCase):
    def test_v1_release_open(self):
        sup = to_held()
        tc = 30.505
        r = drive(sup, 30.0, 4.0, door=ramp(tc, 1.0 * DEG), pend=[(30.3, 2)], commits=[(tc + 0.1, 2, tc)], handover_at=33.8)
        self.assertTrue(r['ok'][2], r['outcome']['checks'])
        self.assertEqual(r['n_nopub'], 0)
        self.assertEqual((r['outcome']['state'], r['outcome']['sub']), ('HANDED_OVER', 'release_open'))

    def test_v2_v3(self):
        sup = to_held()
        drive(sup, 30.0, 2.0, door=ramp(30.505, 0.06), pend=[(30.3, 2)], commits=[(30.53, 2, 30.505)])
        self.assertEqual(sup.seq.why, 'door_rate_in_release')
        sup = to_held()
        drive(sup, 30.0, 3.0, door=ramp(30.505, 1.0 * DEG), arm=lambda t: max(0.0, t - 31.0) * 0.01, pend=[(30.3, 2)], commits=[(30.53, 2, 30.505)])
        self.assertEqual(sup.seq.why, 'moved_in_release_open')


class TW(unittest.TestCase):
    def unload_world(self, t1=30.505, known1=None, base_amp=0.003, door_amp=0.24 * DEG, arm_amp=0.004, extra_door=None, t2=None):
        known1 = t1 + 0.02 if known1 is None else known1
        door = lambda t: 0.14 + tr(t1, door_amp, t) + (extra_door(t) if extra_door else 0.0)
        bx = lambda t: 0.6 + tr(t1, base_amp, t)
        arm = lambda t: tr(t1, arm_amp, t)
        return door, bx, arm, known1

    def test_w1_unload_then_open(self):
        sup = to_held(unload=True)
        door, bx, arm, kn = self.unload_world()
        r = drive(sup, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        self.assertTrue(r['ok'][1], r['outcome']['checks'])
        self.assertEqual(sup.sub, 'unloaded', r['outcome'])
        tu = sup.t_unloaded
        self.assertGreaterEqual(tu, 30.505 + 1.0)
        t2 = 34.305
        door2 = lambda t: door(t) + max(0.0, t - t2) * 1.0 * DEG
        r = drive(sup, 34.0, 3.0, door=door2, bx=bx, arm=arm, pend=[(34.1, 2)], commits=[(t2 + 0.05, 2, t2)], handover_at=36.5)
        self.assertTrue(r['ok'][2], r['outcome']['checks'])
        self.assertEqual(r['n_nopub'], 0)
        self.assertEqual((r['outcome']['state'], r['outcome']['sub']), ('HANDED_OVER', 'release_open'))

    def test_w2_mode1_notice_delayed(self):
        sup = to_held(unload=True)
        door, bx, arm, kn = self.unload_world(known1=30.505 + 0.1)
        r = drive(sup, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        self.assertTrue(r['ok'][1], r['outcome']['checks'])
        self.assertTrue(any(x.get('defer_start') == 1 for x in r['outcome']['log7']), r['outcome']['log7'])
        self.assertEqual(sup.sub, 'unloaded')

    def test_w3_w4_w5(self):
        sup = to_held(unload=True)
        door, bx, arm, kn = self.unload_world(base_amp=0.025)
        drive(sup, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        self.assertEqual(sup.seq.why, 'base_dev_in_unload')
        sup = to_held(unload=True)
        door, bx, arm, kn = self.unload_world(extra_door=lambda t: 0.003 * max(0.0, t - 30.505))
        drive(sup, 30.0, 6.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        self.assertEqual(sup.seq.why, 'unload_timeout')
        sup = to_held(unload=True)
        door, bx, arm, kn = self.unload_world(extra_door=lambda t: 0.06 * max(0.0, t - 30.8))
        drive(sup, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        self.assertEqual(sup.seq.why, 'door_rate_in_unload')

    def test_w6_w7(self):
        sup = to_held(unload=True)
        drive(sup, 30.0, 1.0, pend=[(30.3, 1)], commits=[(30.52, 2, 30.505)])
        self.assertEqual(sup.seq.why, 'commit_order')
        door, bx, arm, kn = self.unload_world()
        ref = to_held(unload=True)
        drive(ref, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1)], commits=[(kn, 1, 30.505)])
        tu = ref.t_unloaded                                                   # 同一確定性輸入的 unloaded 時刻
        sup = to_held(unload=True)
        drive(sup, 30.0, 4.0, door=door, bx=bx, arm=arm, pend=[(30.3, 1), (31.0, 2)], commits=[(kn, 1, 30.505), (tu + 0.05, 2, tu - 0.03)])
        self.assertEqual(sup.seq.why, 'commit_before_reconverge')


class World7(T5.World5):
    def __init__(self, gate2):
        super().__init__()
        self.auth = M7.ModeAuthority7('r', run_id_num=T5.RUNN, dt=T5.DT, gate3=lambda step: True, gate2=gate2)
        self.wx = T5.WX2.W2Exec2(self.auth, self.intake)


class TA(unittest.TestCase):
    def test_a1_a2_a3(self):
        w = World7(lambda step: (False, 'unload_state:SETTLING'))
        T5.full(w)
        self.assertEqual(w.auth.mode_seq, 1)
        self.assertEqual(next(r['why'] for r in w.auth.rejections if r['req_seq'] == 2), 'unload_evidence:unload_state:SETTLING')
        st = {'n': 0}

        def g(step):
            st['n'] += 1
            return st['n'] <= 1
        w = World7(g)
        T5.full(w)
        self.assertEqual(w.auth.mode_seq, 1)
        self.assertTrue(str(next(r['why'] for r in w.auth.rejections if r['req_seq'] == 2)).startswith('unload_evidence_at_commit'))
        w = World7(lambda step: True)
        T5.full(w)
        self.assertEqual([c['mode_seq'] for c in w.auth.commits], [1, 2, 3])
        self.assertTrue(all(c['applied_confirmed'] is True for c in w.auth.commits))


if __name__ == '__main__':
    unittest.main(verbosity=1)
