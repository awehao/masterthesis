#!/usr/bin/env python3
"""door_w2_mode3（ModeAuthority3）純邏輯反例（Codex 20261025_214144 第 1、2 項）。
  R1 Codex 重現一：stamp＝0、接收＝0.19、選定＝0.30 ⇒ 不提交（候選於選定當下已老化 0.30 s）。
  R2 Codex 重現二：請求＝0、候選 1.01 s 到達 ⇒ candidate_timeout、不提交（先核請求期限）。
  R3 一般流同樣以 stamp 判新鮮：stamp＝0、接收＝0.19、選定＝0.25 ⇒ 本步不套用。
  R4 套用確認缺失：select 後未 confirm_step 即下一次 select ⇒ 中止 confirm_missing（提交前、提交後皆是；提交後不回切）。
  R5 提交步確認不符：來源序號、mode_seq、鏈模式（逾時）、限幅後非保持形、非有限 ⇒ applied_confirmed False、
     中止 commit_not_applied:<why>、mode_seq 仍 1、之後不套用。
  R6 提交後實際套用非保持形 ⇒ 中止 applied_not_mode_shaped。
  R7 提交前實際套用 mode 1 來源 ⇒ 中止 applied_mode_mismatch。
  R8 候選序號不大於已送鏈序號 ⇒ 丟棄該筆、不提交；下一筆較新候選才提交。
  R9 確認步號與待確認步不符 ⇒ 中止 confirm_order。R10 r2 的 confirm_applied（只有序號）停用。
  R11 提交後斷流以 stamp 計：t − 最新 stamp > 0.25 ⇒ 中止 post_commit_source_lost、不回切。
  R12 正常：提交步確認相符 ⇒ applied_confirmed True、紀錄列含九軸實際值。

    python3 evaluation/test_door_w2_mode3.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_w2_mode3 as M3                             # noqa: E402

RUNN = 77.0
U_HOLD = [0.0022, -0.0013, 0.0018] + [0.0] * 6
REQ = {'run_id': 'r', 'req_seq': 1, 'from_mode': M3.MODE_HELD, 'to_mode': M3.MODE_RELEASE_TEST, 'mode_seq': 1}


def env(ms, src, u=None, stamp=0.0):
    return {'mode_seq': ms, 'source_seq': src, 'u': list(u if u is not None else U_HOLD), 'stamp_sim_t': stamp,
            'stage_id': 2, 'kind': 0, 'derived': True, 'run_id': RUNN}


def auth():
    return M3.ModeAuthority3('r', run_id_num=RUNN, dt=0.01)


def rec_of(sel, u=None, cm=0):
    e = sel['apply']
    if e is None:
        return {'source_seq': -1, 'mode_seq': None, 'chain_mode': 1, 'u': [0.0] * 9}
    return {'source_seq': e['source_seq'], 'mode_seq': e['mode_seq'], 'chain_mode': cm, 'u': list(u if u is not None else e['u'])}


def step(a, k, t, u_prev=U_HOLD, rec=None):
    sel = a.select(k, t, u_prev)
    r = rec_of(sel) if rec is None else rec(sel)
    ok, why = a.confirm_step(k, r)
    return sel, ok, why


def to_commit(a, t_req=0.0):
    """mode 0 流 → 請求 → 候選，回傳提交步的 select 結果（尚未 confirm）。"""
    a.on_env(env(0, 10, stamp=t_req), t_req)
    step(a, 1, t_req)
    a.on_request(dict(REQ), 2, t_req)
    a.on_env(env(0, 11, stamp=t_req + 0.01), t_req + 0.01)
    a.on_env(env(1, 12, stamp=t_req + 0.01), t_req + 0.01)
    sel = a.select(2, t_req + 0.02, U_HOLD)
    return sel


class TR3(unittest.TestCase):
    def test_r1_codex_repro_stale_at_select(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.0)
        step(a, 0, 0.0)
        a.on_request(dict(REQ), 1, 0.0)
        self.assertEqual(a.on_env(env(1, 11, stamp=0.0), 0.19)[0], 'buffered_candidate')
        sel, _, _ = step(a, 30, 0.30)
        self.assertIsNone(a.commit)
        self.assertEqual(a.mode_seq, 0)
        self.assertNotEqual((sel['event'] or {}).get('kind'), 'commit')
        self.assertEqual(a.n_cand_stale_at_select, 1)
        self.assertIsNotNone(a.pending)

    def test_r2_codex_repro_deadline_first(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.0)
        step(a, 0, 0.0)
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(0, 11, stamp=1.0), 1.0)
        step(a, 100, 1.0)
        self.assertEqual(a.on_env(env(1, 12, stamp=1.01), 1.01)[0], 'buffered_candidate')
        sel, _, _ = step(a, 101, 1.01)
        self.assertIsNone(a.commit)
        self.assertEqual(sel['event'], {'kind': 'candidate_timeout'})
        self.assertEqual(a.last_reject['why'], 'candidate_timeout')
        self.assertEqual(a.mode_seq, 0)

    def test_r3_normal_stream_by_stamp(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.19)
        sel, _, _ = step(a, 25, 0.25)
        self.assertIsNone(sel['apply'])
        self.assertEqual(a.n_cur_stale_at_select, 1)
        a.on_env(env(0, 11, stamp=0.24), 0.25)
        sel, _, _ = step(a, 26, 0.26)
        self.assertEqual(sel['apply']['source_seq'], 11)

    def test_r4_confirm_missing(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.0)
        a.select(0, 0.0, U_HOLD)
        sel = a.select(1, 0.01, U_HOLD)
        self.assertEqual(a.aborted['why'], 'confirm_missing')
        self.assertIsNone(sel['apply'])
        a = auth()
        sel = to_commit(a)
        self.assertEqual((sel['event'] or {}).get('kind'), 'commit')
        sel2 = a.select(3, 0.03, U_HOLD)                                  # 提交步缺確認
        self.assertEqual(a.aborted['why'], 'confirm_missing')
        self.assertIsNone(sel2['apply'])
        self.assertEqual(a.mode_seq, 1)
        self.assertIsNone(a.commit['applied_confirmed'])

    def test_r5_commit_step_mismatch(self):
        cases = {'src_ne_first_source_seq': lambda s: dict(rec_of(s), source_seq=11),
                 'mode_seq_ne_commit': lambda s: dict(rec_of(s), mode_seq=0),
                 'chain_not_normal': lambda s: dict(rec_of(s), chain_mode=1),
                 'base_not_hold': lambda s: rec_of(s, u=[0.02, 0.0, 0.0] + [0.0] * 6),
                 'arm_not_zero': lambda s: rec_of(s, u=U_HOLD[:3] + [1e-3] + [0.0] * 5),
                 'malformed_applied_record': lambda s: rec_of(s, u=[float('nan')] * 9)}
        for why, f in cases.items():
            a = auth()
            sel = to_commit(a)
            ok, w = a.confirm_step(2, f(sel))
            self.assertFalse(ok, why)
            self.assertEqual(w, why)
            self.assertEqual(a.aborted['why'], f'commit_not_applied:{why}')
            self.assertIs(a.commit['applied_confirmed'], False)
            self.assertEqual(a.mode_seq, 1)
            a.on_env(env(1, 13, stamp=0.03), 0.03)
            self.assertIsNone(a.select(3, 0.03, U_HOLD)['apply'])

    def test_r6_post_commit_not_shaped(self):
        a = auth()
        sel = to_commit(a)
        self.assertTrue(a.confirm_step(2, rec_of(sel))[0])
        a.on_env(env(1, 13, stamp=0.02), 0.03)
        sel = a.select(3, 0.03, U_HOLD)
        ok, w = a.confirm_step(3, rec_of(sel, u=[0.0, 0.0, 0.05] + [0.0] * 6))
        self.assertFalse(ok)
        self.assertEqual(a.aborted['why'], 'applied_not_mode_shaped:base_not_hold')
        self.assertEqual(a.mode_seq, 1)

    def test_r7_pre_commit_mode_mismatch(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.0)
        sel = a.select(0, 0.0, U_HOLD)
        ok, w = a.confirm_step(0, dict(rec_of(sel), mode_seq=1))
        self.assertFalse(ok)
        self.assertEqual(a.aborted['why'], 'applied_mode_mismatch')

    def test_r8_candidate_not_newer_than_fed(self):
        a = auth()
        a.on_request(dict(REQ), 0, 0.0)
        a.on_env(env(1, 11, stamp=0.0), 0.0)
        a.on_env(env(0, 12, stamp=0.0), 0.0)       # 先送鏈 12（mode 0 流在候選之前已被選定）
        a.cand = None                              # 本步候選尚不可見（另一路徑晚到）
        step(a, 0, 0.0)
        self.assertEqual(a.last_fed_seq, 12)
        a.cand = (env(1, 11, stamp=0.0), 0.0)       # 較舊候選此時才可見
        sel, _, _ = step(a, 1, 0.01)
        self.assertIsNone(a.commit)
        self.assertEqual(a.n_cand_not_newer, 1)
        a.on_env(env(1, 13, stamp=0.01), 0.02)
        sel = a.select(2, 0.02, U_HOLD)
        self.assertEqual(a.commit['first_source_seq'], 13)
        self.assertTrue(a.confirm_step(2, rec_of(sel))[0])

    def test_r9_r10_confirm_order_and_r2_interface(self):
        a = auth()
        a.on_env(env(0, 10, stamp=0.0), 0.0)
        sel = a.select(0, 0.0, U_HOLD)
        ok, w = a.confirm_step(1, rec_of(sel))
        self.assertEqual((ok, w, a.aborted['why']), (False, 'confirm_order', 'confirm_order'))
        with self.assertRaises(RuntimeError):
            auth().confirm_applied(0, 10)

    def test_r11_post_commit_loss_by_stamp(self):
        a = auth()
        sel = to_commit(a)
        a.confirm_step(2, rec_of(sel))
        # 提交筆 stamp＝0.01；之後無新命令（接收再晚也不重新起算）
        for k in range(3, 40):
            t = k * 0.01
            sel = a.select(k, t, U_HOLD)
            if a.aborted is not None:
                break
            a.confirm_step(k, rec_of(sel) if sel['apply'] is not None else
                           {'source_seq': -1, 'mode_seq': None, 'chain_mode': 1, 'u': [0.0] * 9})
        self.assertEqual(a.aborted['why'], 'post_commit_source_lost')
        self.assertAlmostEqual(a.aborted['t'], 0.27, places=6)
        self.assertEqual(a.mode_seq, 1)

    def test_r12_normal(self):
        a = auth()
        sel = to_commit(a)
        self.assertEqual(sel['apply']['source_seq'], 12)
        self.assertEqual(sel['profile'], 'release_prep_test')
        self.assertEqual(a.confirm_step(2, rec_of(sel)), (True, None))
        self.assertIs(a.commit['applied_confirmed'], True)
        self.assertEqual(a.applied_rows[-1]['u'], U_HOLD)
        self.assertEqual(a.stats()['commit']['first_source_seq'], 12)


if __name__ == '__main__':
    unittest.main(verbosity=1)
