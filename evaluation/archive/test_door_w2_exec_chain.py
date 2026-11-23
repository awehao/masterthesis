#!/usr/bin/env python3
"""W2 執行端接線 × 真正 E2 命令鏈（CmdChainE2＋EnvIntake＋applied_source；Codex 20261025_214144 第 3 項）。
測試端只產生封裝與 K2 過濾行為；**套用來源一律由 applied_source 讀鏈**，不由測試指定。
  C1 正常：mode 0 流 → 請求 → 候選旁路 ⇒ 提交步實際首筆來源＝first_source_seq、profile 同步切換、鏈正常；之後每步 mode 1。
  C2 提交步 K2 過濾擋下候選（鏈仍套用舊 mode 0 快照）⇒ commit_not_applied:src_ne_first_source_seq、不回切。
  C3 提交步鏈拒收（輪級檢查失敗 ⇒ 鏈失效）⇒ commit_not_applied（鏈模式 3）。
  C4 限幅：權威加速度框以 0.05 s 寬鬆設定，候選通過；鏈的底盤變化率上限把套用值留在上一步大命令附近 ⇒
     實際套用非保持形 ⇒ commit_not_applied:base_not_hold。同一情境權威以物理步 0.01 s ⇒ 候選先被拒（accel_box），不提交。
  C5 提交後新模式斷流：鏈在來源 stamp＋0.2 s 進逾時減速（從來源時間起算，不在進鏈時重新起算），
     權威在 stamp＋0.25 s 中止 post_commit_source_lost、不回切。
  C6 Codex 重現（真鏈）：stamp＝0 接收 0.19 選定 0.30 ⇒ 不提交；請求 0、候選 1.01 到 ⇒ 逾時不提交；兩者期間鏈持續套用 mode 0。
  C7 缺確認：提交步後未呼叫 post_step 即下一步 ⇒ confirm_missing 中止。
  C8 一般流老化：來源停 0.3 s ⇒ 權威 0.2 s 後不再選定、鏈同時以來源時間逾時（兩者一致），無舊筆被當新筆套用。

    python3 evaluation/test_door_w2_exec_chain.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_applied_src as K3MSRC                  # noqa: E402
import door_w2_check_mode3 as CK3                      # noqa: E402
import door_w2_exec as WX                              # noqa: E402
import door_w2_mode3 as M3                             # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                 # noqa: E402
from wb_wheel_limit import WheelLimitConfig            # noqa: E402

RUNN = 77.0
DT = 0.01
U_HOLD = [0.0022, -0.0013, 0.0018] + [0.0] * 6
REQ = {'run_id': 'r', 'req_seq': 1, 'from_mode': M3.MODE_HELD, 'to_mode': M3.MODE_RELEASE_TEST, 'mode_seq': 1}


def env(ms, src, u=None, stamp=0.0):
    return {'mode_seq': ms, 'source_seq': src, 'u': list(u if u is not None else U_HOLD), 'stamp_sim_t': round(stamp, 6),
            'stage_id': 2, 'kind': 0, 'derived': True, 'run_id': RUNN}


def mk_chain(wheel_ok=None):
    def low(vx, vy, wz):
        return (True, None) if (vx * vx + vy * vy) ** 0.5 <= 0.05 and abs(wz) <= 0.2 else (False, 'over')
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=wheel_ok or low,
                      joint_lower=(-3.0,) * 6, joint_upper=(3.0,) * 6, joint_margin=0.05, mode='solver_drawer',
                      wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05),
                      base_accel_max_lin=0.5, base_accel_max_ang=2.0)


class World:
    """10 ms 物理步；求解端 50 ms 週期：一般流（committed）＋候選旁路（prep），全域遞增 source_seq。"""

    def __init__(self, auth_dt=DT, wheel_ok=None, filt=None):
        self.auth = M3.ModeAuthority3('r', run_id_num=RUNN, dt=auth_dt)
        self.chain = mk_chain(wheel_ok)
        self.intake = K3MSRC.EnvIntake(run_id_num=RUNN, stage_id=2, kinds=(0,))
        self.wx = WX.W2Exec(self.auth, self.intake, filt=filt or (lambda u: u))
        self.seq, self.committed, self.prep = 100, 0, None
        self.u_solver = list(U_HOLD)
        self.cand_u = None
        self.k = 0
        self.t_req = None
        self.inbox = []          # (deliver_t, env)

    def solver_cycle(self, t, delay=0.0, stop_mode=None, cand_delay=0.0):
        if stop_mode is not None and self.committed >= stop_mode:
            return
        self.seq += 1
        self.inbox.append((t + delay, env(self.committed, self.seq, self.u_solver, stamp=t)))
        if self.prep is not None:
            self.seq += 1
            self.inbox.append((t + delay + cand_delay, env(self.prep, self.seq, self.cand_u or self.u_solver, stamp=t)))

    def run(self, t_end, req_at=None, solver_until=None, delay=0.0, cand_delay=0.0, stop_mode=None, skip_post_at_commit=False,
            filt_after_req=False):
        rows = []
        while self.k * DT <= t_end + 1e-9:
            t = round(self.k * DT, 6)
            if self.k % 5 == 0 and (solver_until is None or t < solver_until):
                self.solver_cycle(t, delay, stop_mode, cand_delay)
            if req_at is not None and self.t_req is None and t >= req_at:
                kind, _ = self.auth.on_request(dict(REQ), self.k, t)
                self.t_req = t
                if kind == 'pending':                     # 執行端核可後才轉給求解端（ev6 的轉送次序）
                    self.prep = 1
            for item in [x for x in self.inbox if x[0] <= t + 1e-9]:
                self.inbox.remove(item)
                self.wx.on_env3(item[1], t, t_wall=1000.0 + t)
            sel = self.wx.pre_step(self.k, t, self.chain)
            self.chain.step(t, DT, [0.0] * 6)
            u9 = list(self.chain.u_prev) if self.chain.u_prev is not None else [0.0] * 9
            ev = (sel['event'] or {}).get('kind')
            if not (skip_post_at_commit and ev == 'commit'):
                rows.append(self.wx.post_step(self.k, t, self.chain, True, u9, sel))
            st = self.auth.state_msg(self.k, t)
            if self.prep is not None and st['mode_seq'] >= self.prep:
                self.committed, self.prep = self.prep, None
            elif self.prep is not None and (st['aborted'] or (st['last_reject'] or {}).get('req_seq') == 1):
                self.prep = None
            self.k += 1
        return rows


class TChain(unittest.TestCase):
    def test_c1_normal(self):
        w = World()
        rows = w.run(3.0, req_at=1.0)
        c = w.auth.commit
        self.assertIsNotNone(c)
        self.assertIs(c['applied_confirmed'], True)
        rk = next(r for r in rows if r['step'] == c['step'])
        self.assertEqual((rk['src'], rk['mode_seq'], rk['chain_mode'], rk['profile']), (c['first_source_seq'], 1, 0, 'release_prep_test'))
        self.assertTrue(all(r['profile'] == 'pause_hold' for r in rows if r['step'] < c['step']))
        self.assertTrue(all(r['mode_seq'] == 1 and r['chain_mode'] == 0 for r in rows if r['step'] >= c['step']))
        self.assertIsNone(w.auth.aborted)
        self.assertEqual(w.intake.violations, [])
        chk = CK3.check(rows, w.auth.stats(), w.auth.applied_rows, {'committed': 1, 'first_src_by_mode': {'1': c['first_source_seq']}}, t_req=w.t_req,
                        t_obs_passed=0.5, window=50)
        self.assertTrue(chk['ok'], chk)

    def test_c2_filtered_at_commit(self):
        st = {'n': 0}
        w = None

        def filt(u):
            if w.auth.pending is None and w.auth.commit is not None and st['n'] == 0:
                st['n'] += 1
                return None                               # 提交步：K2 過濾擋下候選
            return u
        w = World(filt=filt)
        w.run(2.0, req_at=1.0)
        self.assertEqual(w.auth.aborted['why'], 'commit_not_applied:src_ne_first_source_seq')
        self.assertIs(w.auth.commit['applied_confirmed'], False)
        self.assertEqual(w.auth.mode_seq, 1)

    def test_c3_chain_reject_at_commit(self):
        st = {'armed': False}

        def wheel(vx, vy, wz):
            return (False, 'test_reject') if st['armed'] else (True, None)
        w = World(wheel_ok=wheel)
        orig = w.auth.select

        def sel(step, t, u):
            r = orig(step, t, u)
            if (r['event'] or {}).get('kind') == 'commit':
                st['armed'] = True
            return r
        w.auth.select = sel
        w.run(2.0, req_at=1.0)
        self.assertTrue(w.auth.aborted['why'].startswith('commit_not_applied:'), w.auth.aborted)
        row = next(r for r in w.wx.rows if r['step'] == w.auth.commit['step'])
        self.assertEqual((row['src'], row['chain_mode']), (-1, 3))
        self.assertEqual(w.auth.mode_seq, 1)

    def _limit_world(self, auth_dt):
        w = World(auth_dt=auth_dt)
        w.u_solver = [0.0145, 0.0, 0.0] + [0.0] * 6        # mode 0 流（合成）：鏈實際套用 0.0145
        w.run(1.19)
        w.cand_u = [-0.0045, 0.0, 0.0] + [0.0] * 6         # 候選：保持形；與上一步實際套用差 0.019（≤ 0.5×0.05、> 0.5×0.01）
        w.run(1.6, req_at=1.2)
        return w

    def test_c4_limiting(self):
        w = self._limit_world(auth_dt=0.05)              # 寬鬆框：候選過；鏈限幅後實際值 ≈ 上一步 − 0.005 ⇒ 非保持形
        self.assertIsNotNone(w.auth.commit)
        self.assertEqual(w.auth.aborted['why'], 'commit_not_applied:base_not_hold')
        self.assertEqual(w.auth.mode_seq, 1)
        w = self._limit_world(auth_dt=DT)                # 物理步框：候選先被拒
        self.assertIsNone(w.auth.commit)
        self.assertEqual(w.auth.last_reject['why'], 'candidate_invalid:accel_box')
        self.assertEqual(w.auth.mode_seq, 0)

    def test_c5_post_commit_source_lost(self):
        w = World()
        rows = w.run(3.0, req_at=1.0, stop_mode=1)
        c = w.auth.commit
        self.assertIs(c['applied_confirmed'], True)
        last_stamp = c['stamp_sim_t']
        to = [r for r in rows if r['chain_mode'] == 1]
        self.assertTrue(to)
        self.assertAlmostEqual(to[0]['t'], round(last_stamp + 0.21, 6), places=6)   # 鏈：來源 stamp＋0.2 後首步逾時
        self.assertEqual(w.auth.aborted['why'], 'post_commit_source_lost')
        self.assertAlmostEqual(w.auth.aborted['t'], round(last_stamp + 0.26, 6), places=6)
        self.assertEqual(w.auth.mode_seq, 1)

    def test_c6_codex_repro_real_chain(self):
        w = World()
        w.run(1.18)
        w.auth.on_request(dict(REQ), w.k, 1.0)            # 只由權威登記（求解端不產生候選；本情境只送一筆）
        w.wx.on_env3(env(1, 900, stamp=1.0), 1.19)        # stamp 1.0、接收 1.19（入口當下仍新鮮）
        w.k = 130                                          # 執行端在 1.30 才下一次選定（物理步停滯）
        w.run(1.30)
        self.assertIsNone(w.auth.commit)
        self.assertEqual(w.auth.mode_seq, 0)
        self.assertEqual(w.auth.n_cand_stale_at_select, 1)
        self.assertEqual(w.wx.rows[-1]['mode_seq'], 0)
        w2 = World()
        w2.run(0.99)
        w2.auth.on_request(dict(REQ), w2.k, 1.0)
        w2.inbox.append((2.01, env(1, 9000, stamp=2.01)))  # 候選 1.01 s 後才到
        rows = w2.run(2.05)
        self.assertIsNone(w2.auth.commit)
        self.assertEqual(w2.auth.last_reject['why'], 'candidate_timeout')
        self.assertEqual(w2.auth.last_reject['t'], 2.01)
        self.assertTrue(all(r['src'] >= 0 and r['mode_seq'] == 0 and r['chain_mode'] == 0 for r in rows))

    def test_c7_confirm_missing(self):
        w = World()
        w.run(2.0, req_at=1.0, skip_post_at_commit=True)
        self.assertEqual(w.auth.aborted['why'], 'confirm_missing')
        self.assertEqual(w.auth.aborted['step'], w.auth.commit['step'] + 1)
        self.assertEqual(w.auth.mode_seq, 1)

    def test_c8_normal_stream_aging_consistent(self):
        w = World()
        rows = w.run(1.6, solver_until=1.0)
        last = max(r['t'] for r in rows if r['src'] >= 0)
        self.assertAlmostEqual(last, 1.15, places=6)        # 最後一筆 stamp 0.95，0.95＋0.2
        tos = [r for r in rows if r['t'] > 1.15]
        self.assertTrue(tos and all(r['src'] == -1 and r['chain_mode'] == 1 for r in tos))
        self.assertEqual(w.auth.n_cur_stale_at_select, len(tos))


if __name__ == '__main__':
    unittest.main(verbosity=1)
