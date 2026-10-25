#!/usr/bin/env python3
"""K3-B 控制接線靜態核對 r2（test_k3b_wiring_static.py 的版本化複本：ev7r2；手指閘門）。以下為 r1 說明：
K3-B 控制接線靜態核對（Isaac 與整趟無法離線執行）。
  W1 mk_adm12／mk_ev7／mk_task_k3b 重跑輸出與現檔逐位元相同，且皆 insert-only（腳本內 assert；ev7 對 ev5、task_k3b 對 w1b2 亦然）。
  W2 ev7 每物理步次序：
     step_begin → mode 2 提交步開解除監看 → ex.step → post_step → sync_after → 5 N 權威 → 狀態發布
     → 手指命令閘門 → 逐步解除監看（本步接觸）→ 迴圈末 sync_after → step_id += 1。
  W3 ev7：已知中止含 release_monitor；5 N 權威中止優先於請求；手指閘門只在 K2 未閂鎖且 mode ≥ 1；mode 3 准入接 confirmed_fresh。
  W4 ADM12：
     - mode 3 候選在一般發布之後另解，且候選段內不寫 _cmd_hist；
     - 候選經 /door/applied_src 回報實際套用才插入歷史，並在同一處交接；
     - 交接後目標＝話題目標；手臂不歸零在 pause_u_solve 紀錄之前。
  W5 task_node_k3b：
     - 只在 --w2-release 才建 K3MReleasePolicy；
     - 5 N 請求話題 latched、只在 w2_release 區塊；k3b 紀錄只在 w2_release。

    python3 evaluation/test_k3b_wiring_static.py
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
        for mk, outp in (('results/door/mk_adm12_patch.py', 'dl3_wgmpc_wg2_adm12_node.py'), ('results/door/mk_ev7r2_patch.py', 'isaac_door_k2_sim_ev7r2.py'),
                         ('results/door/mk_task_k3b_patch.py', 'door_k3m_task_node_k3b.py'),
                         ('results/door/mk_task_k3b2_patch.py', 'door_k3m_task_node_k3b2.py')):
            h0 = hashlib.sha256(rd(outp).encode()).hexdigest()
            r = subprocess.run([sys.executable, mk], cwd=HERE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('insert-only ok', r.stdout)
            self.assertEqual(hashlib.sha256(rd(outp).encode()).hexdigest(), h0, outp)

    def test_w2_ev7_order(self):
        s = rd('isaac_door_k2_sim_ev7r2.py')
        keys = ['_w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)',
                '                w2_xm.open(step_id, t)',
                'res = ex.step(step_id, t, a.physics_dt',
                'w2x.post_step(step_id, t, chain, res.owner == AUTH_WHOLEBODY,',
                'w2x.sync_after(step_id, t, known_abort=_w2_known_abort())                # 本步 select 之後',
                '_o7 = w2_effa.tick(step_id, t, w2_xm, w2_auth.mode_seq,',
                'node.w2_state_pub.publish(',
                "            _fcmd = w2_grip.command(step_id, t, k2_latch, w2_auth.mode_seq, w2_auth.finger_target(), _fcmd)",
                '            w2_xm.step(step_id, t, _fq7, _pf7, _lc is not None, door_q=float(_qd_rec))',
                '            w2x.sync_after(step_id, t, known_abort=_w2_known_abort())\n        step_id += 1']
        pos = order(s, keys)
        self.assertEqual(pos, sorted(pos))

    def test_w3_ev7_semantics(self):
        s = rd('isaac_door_k2_sim_ev7r2.py')
        i = s.index('    def _w2_known_abort():')
        self.assertIn("if w2_xm is not None and w2_xm.state == 'FAILED':", s[i:i + 900])
        j = s.index('            if w2_effa is not None:')
        blk = s[j:j + 600]
        self.assertLess(blk.index('w2_effa.abort('), blk.index('w2_effa.request('))
        self.assertIn('w2_grip = K3BG.GripGate(a.finger_open, a.finger_closed)', s)
        self.assertNotIn('w2_auth.finger_target() == \'open\' else a.finger_closed', s)
        self.assertIn('gate3=lambda _s7: w2_xm.confirmed_fresh(_s7)', s)

    def test_w4_adm12(self):
        s = rd('dl3_wgmpc_wg2_adm12_node.py')
        i = s.index("            if self._w2r is not None and self._w2src.tag_candidate3() == 3:")
        j = s.index("                        self._w2r['cand_fail_why'].append({'slot': slot, 'why': _why12})", i)
        blk = s[i:j]
        self.assertGreater(i, s.index("            rec['publish_sim_t_exact'] = self._last_pub_t"))
        self.assertNotIn('_cmd_hist', blk)
        k = s.index("    def _on_applied_src(self, m):")
        blk2 = s[k:k + 1800]
        self.assertLess(blk2.index("self._cmd_hist.insert("), blk2.index("if not self._pause_active():"))
        self.assertIn("'by': 'applied_candidate'", blk2)
        self.assertIn("T_cyc, _tgt_src = np.array(_T12, float), f'w2_retreat:{_src12}'", s)
        self.assertLess(s.index("rec['w2_retreat_out'] = True"), s.index("rec['pause_u_solve'] = [float(x) for x in np.asarray(r.u0, float)]"))

    def test_w6_r2_files(self):
        s = rd('isaac_door_k2_sim_ev7r2.py')
        self.assertIn('import door_k3b_release3 as K3BR', s)
        self.assertNotIn('import door_k3b_release2 as K3BR', s)
        self.assertIn("rec['w2']['grip_gate'] = w2_grip.report()", s)
        n = rd('door_k3m_task_node_k3b2.py')
        i = n.index('        pol = PK3B2.K3MReleasePolicy2(')
        self.assertGreater(i, n.index('        pol = PK3B.K3MReleasePolicy('))   # r2 物件取代 r1 物件（之後才建）
        self.assertIn('import door_k3b_release3 as KR', rd('door_k3m_policy_k3b2.py'))
        self.assertIn('import door_k3b_release3 as KR', rd('door_k3m_check_k3b2.py'))

    def test_w5_task_node(self):
        s = rd('door_k3m_task_node_k3b.py')
        i = s.index('    if a.w2_release:                                                       # K3-B：同參數另建受控釋放策略')
        self.assertIn('PK3B.K3MReleasePolicy(', s[i:i + 500])
        self.assertEqual(s.count("'/door/finger_effort_req'"), 1)
        self.assertIn("nd.eff_req_pub = nd.create_publisher(String, '/door/finger_effort_req', _lat_w2)", s)
        self.assertIn("rep['k3b'] = dict(pol.k3b_report()", s)


if __name__ == '__main__':
    unittest.main(verbosity=1)
