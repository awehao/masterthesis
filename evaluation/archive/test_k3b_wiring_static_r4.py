#!/usr/bin/env python3
"""K3-B r4（閉爪卸載候選整合）控制接線靜態核對（Isaac 與整趟無法離線執行）。
  W1 mk_ev8／mk_task_k3b4／mk_adm15 重跑輸出與現檔逐位元相同，且皆 insert-only（ev8 對 ev7r2；task_k3b4 對 k3b3 與 w1b2；ADM15 對 ADM12）。
  W2 ev8 每物理步次序：
     step_begin → mode 2 提交步開解除監看 → mode 1／2 提交步卸載 start／handoff → … → 卸載中止優先 → 斜坡 set_ramp
     → W1e 定期讀回 → 手指命令閘門 → 解除監看 step → 卸載監看 step（同步驟資料）→ world.step。
  W3 ev8 語意：權威 r8 unload＝True 且帶 gate2；已知中止含 unload_monitor；卸載模式 5 N 請求改為無效（不受理）；
     權威狀態附 unload（state_brief）與 finger_effort；room 紀錄 w2.unload／unload_ctrl；--w2-unload 需 --w2-release。
  W4 任務節點 k3b4：只在 --w2-unload（且在 --w2-release 區塊內）建 K3MReleasePolicy4(unload=True)；--w2-unload 需 --w2-release。
  W5 ADM15：監督器 r8＋tcp_fn（本節點 K.fk）；提交只經 validated_commit；t_settling 掛鉤在本趟 run_id 過濾之後；不含 ADM13／ADM14 痕跡。
  W6 跑法 r4：只換 ev8／k3b4／ADM15（含旗標）、判定器 k3b4、凍結 r4；說明行不再寫「不張爪、不退離」。

    python3 evaluation/test_k3b_wiring_static_r4.py
"""
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p):
    with open(os.path.join(HERE, p), encoding='utf-8') as fh:
        return fh.read()


def order(s, keys):
    pos = []
    for k in keys:
        assert s.count(k) == 1, (k, s.count(k))
        pos.append(s.index(k))
    return pos


class TW(unittest.TestCase):
    def test_w1_regen(self):
        for mk, outp in (('results/door/mk_ev8_patch.py', 'isaac_door_k2_sim_ev8.py'), ('results/door/mk_task_k3b4_patch.py', 'door_k3m_task_node_k3b4.py'),
                         ('results/door/mk_adm15_patch.py', 'dl3_wgmpc_wg2_adm15_node.py')):
            h0 = hashlib.sha256(rd(outp).encode()).hexdigest()
            r = subprocess.run([sys.executable, mk], cwd=HERE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('insert-only ok', r.stdout)
            self.assertEqual(hashlib.sha256(rd(outp).encode()).hexdigest(), h0, outp)

    def test_w2_ev8_order(self):
        s = rd('isaac_door_k2_sim_ev8.py')
        keys = ['_w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)',
                '                w2_xm.open(step_id, t)',
                '                    w2_xu.start(step_id, t)',
                '                    w2_xu.handoff(step_id, t)',
                '            _ka8 = _w2_known_abort()',
                "                    _ru8 = K3BUC.set_ramp(fin_eff, robot.get_articulation_controller(), _fu8, step_id, t)",
                '            _pe5 = fin_eff.periodic(robot.get_articulation_controller(), step_id, t)',
                "            _fcmd = w2_grip.command(step_id, t, k2_latch, w2_auth.mode_seq, w2_auth.finger_target(), _fcmd)",
                '            w2_xm.step(step_id, t, _fq7, _pf7, _lc is not None, door_q=float(_qd_rec))',
                "            w2_xu.step(step_id, t, w2_xu_step['cur'][1], w2_xu_step['cur'][0], _pfu8, _lc is not None, float(_qd_rec),",
                '        world.step(render=False)']
        pos = order(s, keys)
        self.assertEqual(pos, sorted(pos))
        i = s.index('            _ka8 = _w2_known_abort()')
        blk = s[i:i + 700]
        self.assertLess(blk.index('w2_xu.abort('), blk.index('w2_xu.command(t)'))

    def test_w3_ev8_semantics(self):
        s = rd('isaac_door_k2_sim_ev8.py')
        i = s.index('            w2_auth = W2M8.ModeAuthority8(')
        self.assertIn('gate2=lambda _s8: w2_xu.confirmed_fresh(_s8), unload=True)', s[i:i + 900])
        j = s.index('    def _w2_known_abort():')
        self.assertIn("if w2_xu is not None and w2_xu.state == 'FAILED':", s[j:j + 1200])
        self.assertIn("                        _r7 = {'run_id': None, 'unload_mode_rejected': _r7}", s)
        self.assertIn("'unload': w2_xu.state_brief(), 'finger_effort': _fe_last8()}", s)
        self.assertIn("rec['w2']['unload'] = w2_xu.report()", s)
        self.assertIn("rec['w2']['unload_ctrl'] = w2_xu_ctrl", s)
        self.assertIn("print('[door] **--w2-unload 需要 --w2-release** ⇒ 中止', flush=True)", s)

    def test_w4_task_node(self):
        s = rd('door_k3m_task_node_k3b4.py')
        i = s.index('    if a.w2_release:                                                       # K3-B：同參數另建受控釋放策略')
        j = s.index('        if a.w2_unload:                                                    # K3-B r4')
        self.assertGreater(j, i)
        self.assertIn('PK3B4.K3MReleasePolicy4(', s[j:j + 400])
        self.assertIn('unload=True)', s[j:j + 600])
        self.assertEqual(s.count('PK3B4.K3MReleasePolicy4('), 1)
        self.assertIn("raise SystemExit('--w2-unload 需要 --w2-release')", s)

    def test_w5_adm15(self):
        s = rd('dl3_wgmpc_wg2_adm15_node.py')
        self.assertEqual(s.count('W2M8.HeldSupervisorP4b8('), 1)
        self.assertIn('tcp_fn=lambda _b, _q: _K15.fk(np.asarray(list(_b) + list(_q), float), _tcp15))', s)
        k = s.index('    def _w2r7_state(self, st):')
        blk = s[k:k + 2200]
        a = blk.index("st.get('run_id') != self._w2src.run_id")
        self.assertLess(a, blk.index('W2M8.exec_unload_of(st)'))
        self.assertLess(blk.index('W2M8.exec_unload_of(st)'), blk.index('W2M7.validated_commit('))
        self.assertLess(blk.index('W2M7.validated_commit('), blk.index('self._pause_sup.on_commit('))
        for bad in ('ADM13', 'ADM14', 'release_open(', 'commit_t_of', 'HeldSupervisorP4b7('):
            self.assertNotIn(bad, s)

    def test_w6_run_script(self):
        s = rd('run_door_wide_k3b_r4.sh')
        for k in ('evaluation/isaac_door_k2_sim_ev8.py --w2-release --w2-unload ', 'evaluation/door_k3m_task_node_k3b4.py --w2-release --w2-unload ',
                  'evaluation/dl3_wgmpc_wg2_adm15_node.py --w2-retreat --w2-hold-r8 --w2-unload ',
                  'FRZ=evaluation/results/door/freeze_door_wide_k3b_dev_r4.sha256', '通過與否由 door_k3m_check_k3b4.py 判定'):
            self.assertEqual(s.count(k), 1, k)
        self.assertNotIn('保持（不張爪、不退離）==="', s)


if __name__ == '__main__':
    unittest.main(verbosity=1)
