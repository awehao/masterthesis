#!/usr/bin/env python3
"""判定器 r11 測試（Codex r22）。完整正例與反例都以 door_coopen_bench_positive_synth 生成的**合成資料**（以真實通過鏈判定的 w1f_60_3 為底）。
  Z1 最外層 judge() 完整正例（不 mock 子判定器）⇒ why＝[]、chain_exempt＝[]、錨點交叉核對一致、iso_replaced ⊆ {iso_phase_sequence, iso_setpoint_mismatch}。
  Z2 Codex 重現：保留 first_step 的 ok／序號／模式碼／自報 setpoint_dev，改成 owner＝nav、kind 錯、九軸 NaN、設定點全 100 ⇒ 各自 bench_first_step_raw。
  Z3 與 room 列交叉核對：applied_src seq 不符、base_cmd_body 改 1e-5、sim_t 偏 1e-3、owner 碼改 0、對應列缺 ⇒ bench_first_step_room。
  Z4 與證據 arm_sp：改 1e-6、該步證據缺 ⇒ bench_first_step_evidence。
  Z5 缺欄／型別錯：u_applied 缺、step 為字串、room config.physics_dt 缺 ⇒ bench_first_step_raw。
  Z6 最外層 judge()：room bench 錨點偏 1e-6（與事件／求解端不一致）⇒ bench_anchor，且鏈以原 gate（anchor None）判 ⇒ k3m_room_evidence_invalid。

    python3 evaluation/test_door_coopen_check_core11.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_bench_positive_synth as SY          # noqa: E402
import door_coopen_check_core11 as J11                 # noqa: E402

OUT = '/tmp/claude-1000/coopen_full_positive_test'
_D = {}


def data():
    if not _D:
        if os.path.exists(OUT):
            import shutil
            shutil.rmtree(OUT)
        SY.build(OUT)
        ld = (lambda f: json.load(open(os.path.join(OUT, f))))
        _D.update(task=ld('task.json'), room=ld('room_run.json'), gate=ld('park_gate.json'), solver=ld('align_solver.json'), obs=ld('lw_observer.json'))
    return copy.deepcopy(_D['task']), copy.deepcopy(_D['room']), copy.deepcopy(_D['gate']), copy.deepcopy(_D['solver']), _D['obs']


EVP, TRP = os.path.join(OUT, 'evidence.jsonl'), os.path.join(OUT, 'task_trace.jsonl')


def bench_why(t, r, s, evp=None):
    why = []
    J11.check_bench(t, r, s, why, evp or EVP)
    return why


class Z(unittest.TestCase):
    def test_z1_full_positive(self):
        t, r, g, s, o = data()
        res = J11.judge(t, r, g, s, o, EVP, TRP)
        self.assertEqual(res['why'], [], res['why'])
        self.assertEqual(res['chain_exempt'], [])
        self.assertTrue(res['bench']['anchor_crosscheck']['consistent'])
        self.assertTrue(set(res['iso_replaced']) <= {'iso_phase_sequence', 'iso_setpoint_mismatch'})
        self.assertTrue(res['chain_pass'])

    def test_z2_codex_repro(self):
        for f in (lambda fs: fs.update(owner='nav'), lambda fs: fs.update(kind='wholebody_no_command'),
                  lambda fs: fs.update(u_applied=[float('nan')] * 9), lambda fs: fs.update(arm_setpoint=[100.0] * 6)):
            t, r, g, s, o = data()
            f(r['bench']['first_step'])
            self.assertIn('bench_first_step_raw', bench_why(t, r, s))

    def test_z3_room(self):
        c = None
        for kind in ('src', 'cmd', 'simt', 'owner', 'missing'):
            t, r, g, s, o = data()
            c = {k: i for i, k in enumerate(r['steps_cols'])}
            k = r['bench']['first_step']['step']
            row = next(x for x in r['steps'] if x[c['step']] == k)
            if kind == 'src':
                row[c['applied_src']][2] += 1
            elif kind == 'cmd':
                row[c['base_cmd_body']][0] += 1e-5
            elif kind == 'simt':
                row[c['sim_t']] += 1e-3
            elif kind == 'owner':
                row[c['owner(0=nav,1=wb,2=glide)']] = 0
            else:
                r['steps'] = [x for x in r['steps'] if x[c['step']] != k]
            self.assertIn('bench_first_step_room', bench_why(t, r, s), kind)

    def test_z4_evidence(self):
        t, r, g, s, o = data()
        k = r['bench']['first_step']['step']
        lines = [json.loads(ln) for ln in open(EVP)]
        for mode in ('shift', 'missing'):
            p = os.path.join(OUT, f'ev_{mode}.jsonl')
            with open(p, 'w') as fh:
                for e in lines:
                    if e.get('step') == k:
                        if mode == 'missing':
                            continue
                        e = dict(e, arm_sp=[e['arm_sp'][0] + 1e-6] + e['arm_sp'][1:])
                    fh.write(json.dumps(e) + '\n')
            self.assertIn('bench_first_step_evidence', bench_why(t, r, s, p), mode)

    def test_z5_fields(self):
        for f in (lambda r: r['bench']['first_step'].pop('u_applied'), lambda r: r['bench']['first_step'].update(step='1'),
                  lambda r: r['config'].pop('physics_dt')):
            t, r, g, s, o = data()
            f(r)
            self.assertIn('bench_first_step_raw', bench_why(t, r, s))

    def test_z6_anchor_tamper_outer(self):
        t, r, g, s, o = data()
        r['bench']['anchor'] = [r['bench']['anchor'][0] + 1e-6] + r['bench']['anchor'][1:]
        res = J11.judge(t, r, g, s, o, EVP, TRP)
        self.assertIn('bench_anchor', res['why'])
        self.assertIn('k3m_room_evidence_invalid', res['why'])
        self.assertFalse(res['chain_pass'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
