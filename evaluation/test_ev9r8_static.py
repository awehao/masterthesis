#!/usr/bin/env python3
"""isaac_door_k2_sim_ev9r8 靜態接線核對（Isaac 不在本機離線執行；Codex r63／r64）。
  E1 results/door/mk_ev9r8_patch.py --check：由 ev9r7、ev5、ev8 三方合併＋RG8 替換重建，與現檔逐位元相同；
  E2 相對純合併版（無 RG8）：刪除的原文恰為 3 個敘述（收件匣 while、走廊監看 if、台架 release_intake 呼叫），其餘全為新增，且每段新增含 RG8 標記；
  E3 每段新增的第一個程式行是 --regrasp 守衛、定義或匯入（--regrasp 未給 ⇒ 不改行為）；
  E4 每物理步次序：step_begin → RG8 on_commit → … → _snap.take → _c3_r 重設 → v3 step_record → RG8 core.step → world.step；
     state 發布處同時發 v3 摘要；台架凍結期 RG8 路徑在原 release_intake 之前判斷；
  E5 啟動核：互斥與必要旗標、registry.validate、門規格 sha、門根單位、拒跑碼 66；換抓中止走 K2 閂鎖（expire_chain、k3m_abort）；
  E6 room_run 記 regrasp（config、core.report、grip_gate、k3m_regrasp_end）；
  E7 gate 2 接線（Codex r66 §3）：權威 RegraspAuthority2；--regrasp 時訂閱 /wgmpc/pause_status（latched）；core.step 前以 ZERO_PUB set_zero；
     core.step 傳入本步實際套用的底盤／手臂命令（res.base_cmd、res.u_applied[3:]，與 room 紀錄同源）。

    python3 evaluation/test_ev9r8_static.py
"""
import difflib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r8.py')).read()


def merged_base():
    m = subprocess.run(['git', 'merge-file', '-p', 'isaac_door_k2_sim_ev9r7.py', 'isaac_door_k2_sim_ev5.py', 'isaac_door_k2_sim_ev8.py'],
                       capture_output=True, text=True, cwd=HERE)
    return m.stdout


def hunks():
    base = merged_base()
    a0, b0, c0 = base.index('<<<<<<< '), base.index('\n=======\n'), base.index('>>>>>>> ')
    body_base = base[base.index('>>>>>>> '):].split('\n', 1)[1]
    body_cur = SRC[SRC.index('# ---- 以下為 ev8 線說明 ----'):]
    body_cur = body_cur[body_cur.index(base[b0 + 9:c0][:60]):]
    body_cur = body_cur[len(base[b0 + 9:c0]):]
    A, B = body_base.splitlines(keepends=True), body_cur.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    return [(tag, A[i1:i2], B[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal'], (a0, b0, c0)


GUARDS = ('def _rg8_ps', 'rg8_core is not None', "getattr(a, 'regrasp', False)", 'import ', 'ap.add_argument', 'rg8_core, rg8_rt', 'def _rg8_tcp',
          '_c3_r = None', 'while node.w2_inbox and not (rg8_core', "'regrasp_end' not in k3m", 'if rg8_core is not None and bench_first_pending')


class TE(unittest.TestCase):
    def test_e1_regen(self):
        r = subprocess.run([sys.executable, 'results/door/mk_ev9r8_patch.py', '--check'], capture_output=True, text=True, cwd=HERE)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, 'identical'), r.stderr[-500:])

    def test_e2_e3_insert_only(self):
        hs, _ = hunks()
        removed = [ln.strip() for _, old, _ in hs for ln in old]
        self.assertEqual(sorted(removed), sorted([
            "_ok, _why, _irec = BENCH.release_intake(env_in, chain, node.wb_env_latest, (bench_rel.announced or {}).get('source_seq'),",
            't_b0, k2_latch.filter_wb, _ab)', 'while node.w2_inbox:', "if k3m['active'] and k3m['violation'] is None:",
            '_w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)']))
        for tag, old, new in hs:
            txt = ''.join(new)
            self.assertIn('RG8', txt, txt[:200])
            first = next(ln for ln in new if ln.strip() and not ln.strip().startswith('#'))
            self.assertTrue(any(g in first for g in GUARDS), first)

    def test_e4_order(self):
        i = [SRC.index(k) for k in ('_w2sel, _relay6, _nv6 = w2x.step_begin(', 'rg8_core.on_commit(step_id, t, w2_auth.mode_seq, _rg8_tcp())',
                                    '_snap.take(step_id,', "_c3_r = None                                                                # RG8",
                                    '_c3_r = DCA.step_record(', '_why8s = rg8_core.step(step_id, t,', "_xt.mark('before_world_step')",
                                    'world.step(render=False)')]
        self.assertEqual(i, sorted(i))
        j = SRC.index("node.w2_state_pub.publish(String(data=json.dumps(w2_auth.state_msg(step_id, t), default=float)))\n                if rg8_core is not None:")
        self.assertGreater(SRC.index('node.rg8_c3_pub.publish(', j), j)
        k1 = SRC.index("_ok, _why, _irec, rg8_rt['rel_sel'] = RGX.bench_release_w2(w2x, chain, _e8, _as8, 0, t_b0, _ab)")
        k2 = SRC.index('_ok, _why, _irec = BENCH.release_intake(env_in, chain, node.wb_env_latest')
        self.assertLess(k1, k2)

    def test_e5_startup_and_abort(self):
        for k in ("--regrasp 需 --bench-start、--contact-v3-out、--regrasp-ref", '--regrasp 與 --w2-release／--w2-unload 互斥',
                  'RGREG.validate(rg8_pkg)', '門規格與登錄核驗所用 door_local_v3.yaml 不同', 'np.max(np.abs(_door_T() - np.eye(4)))) > 1e-6',
                  'sys.exit(66)', "k2_latch.on_msg(f'regrasp：{_why8s}', step_id, t, _g8)", 'k2_latch.expire_chain(chain)',
                  "'source': 'executor_rg8'", "w2_auth.extra_state = lambda: {'gates': rg8_core.gates_state(rg8_rt['step'])"):
            self.assertIn(k, SRC)
        blk = SRC[SRC.index("if getattr(a, 'regrasp', False):                                            # RG8"):SRC.index('        w2_rt = {\'abort_handled\': False}')]
        self.assertLess(blk.index('sys.exit(66)'), blk.index('RGM2.RegraspAuthority2('))

    def test_e7_gate2(self):
        for k in ('RGM2.RegraspAuthority2(a.w2_mode_run_id,', "self.create_subscription(String, '/wgmpc/pause_status', self._counted(self._rg8_ps), _lat_w2)",
                  "rg8_core.g[2].set_zero(_z8)", 'base_cmd=[float(x) for x in res.base_cmd], arm_cmd=[float(x) for x in res.u_applied[3:]])'):
            self.assertIn(k, SRC)
        self.assertLess(SRC.index('rg8_core.g[2].set_zero(_z8)'), SRC.index('_why8s = rg8_core.step(step_id, t,'))
        self.assertNotIn('RGM.RegraspAuthority(', SRC)

    def test_e6_record(self):
        self.assertIn("rec['regrasp'] = {'config': rg8_cfg, 'core': rg8_core.report(), 'runtime': dict(rg8_rt), 'grip_gate': w2_grip.report(),", SRC)


if __name__ == '__main__':
    unittest.main(verbosity=1)
