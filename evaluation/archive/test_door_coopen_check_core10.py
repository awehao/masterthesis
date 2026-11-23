#!/usr/bin/env python3
"""判定器 r10 測試（Codex r21 §1、§3）。
  V1 check_bench 正常（合成）⇒ 無理由；r9 的 19 個反例照用（以 r10 重跑）＋ first_intake（ok False、seq 不符）、first_step（ok False、seq 不符、
     setpoint_dev 1e-6、mode_code 3）⇒ bench_first_intake／bench_first_step。
  V2 anchor_view（合成）：三錨點一致且近 start_pose ⇒ gate.anchor＝bench_anchor、passed 不變、anchor_source 記錄；求解端錨點差 1e-6、
     hold_start 事件缺、start_pose 差 1 cm ⇒ 不一致（gate 不替換）。
  V3 **真鏈串接**（door_wide_w1f_60_3 實錄 ＋ door_k3m_check_w1f2.judge，該趟原判通過）：
     a 原 gate ⇒ 通過；b gate.anchor＝None ⇒ k3m_room_evidence_invalid（Codex 重現）；
     c gate.anchor＝None 但 room／solver 補上與原錨點相同的 bench 錨點紀錄 ⇒ anchor_view 一致 ⇒ 鏈判結果與 a 完全相同（通過、理由空）；
     d 同 c 但 bench 錨點整體平移 50 mm（三處一致、start_pose 同步改）⇒ anchor_view 一致但鏈的底盤／走廊核對**照算並不通過**（不豁免）。
  V4 最外層 judge()（r10）以 V3-c 的資料執行：不拋例外、chain_exempt 空、無 k3m_room_evidence_invalid／park_anchor_missing；
     其餘理由為該趟非 coopen 台架流程的必然差異（登錄、相序等），列出不作通過宣稱。**本專案目前沒有任何 coopen 台架實錄可作完整通過例（揭露）。**

    python3 evaluation/test_door_coopen_check_core10.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core10 as J10                 # noqa: E402
import door_k3m_check_w1f2 as W1F2                     # noqa: E402
import test_door_coopen_check_core9 as T9              # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')
_MK9 = T9.mk


def mk():
    t, r, s = _MK9()
    r['bench']['first_intake'] = {'ok': True, 'source_seq': 3}
    r['bench']['first_step'] = {'ok': True, 'applied_source_seq': 3, 'mode_code': 0, 'setpoint_dev': 0.0}
    r['events'] = [{'park': 'hold_start', 'bench': True, 'anchor': [1.02, -0.5, 3.03]}]
    r['bench']['anchor'] = [1.02, -0.5, 3.03]
    r['config']['start_pose'] = '1.02,-0.5,3.03'
    s['bench']['anchor'] = [1.02, -0.5, 3.03]
    return t, r, s


def run(t, r, s):
    why = []
    J10.check_bench(t, r, s, why)
    return why


def load_real():
    ld = (lambda f: json.load(open(os.path.join(RUN, f))))
    return ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json')


def bench_like(room, solver, gate, shift=0.0):
    a = [float(v) for v in gate['gate']['anchor']]
    a[0] += shift
    r, s, g = copy.deepcopy(room), copy.deepcopy(solver), copy.deepcopy(gate)
    g['gate']['anchor'] = None
    r['bench'] = {'anchor': list(a)}
    r.setdefault('events', []).append({'park': 'hold_start', 'bench': True, 'anchor': list(a)})
    r['config']['start_pose'] = ','.join(str(v) for v in a)
    s['bench'] = {'anchor': list(a)}
    return r, s, g


class V(unittest.TestCase):
    def test_v1_bench(self):
        self.assertEqual(run(*mk()), [])
        for f, want in ((lambda t, r, s: r['bench']['first_intake'].update(ok=False), 'bench_first_intake'),
                        (lambda t, r, s: r['bench']['first_intake'].update(source_seq=4), 'bench_first_intake'),
                        (lambda t, r, s: r['bench']['first_step'].update(ok=False), 'bench_first_step'),
                        (lambda t, r, s: r['bench']['first_step'].update(applied_source_seq=4), 'bench_first_step'),
                        (lambda t, r, s: r['bench']['first_step'].update(setpoint_dev=1e-6), 'bench_first_step'),
                        (lambda t, r, s: r['bench']['first_step'].update(mode_code=3), 'bench_first_step')):
            t, r, s = mk()
            f(t, r, s)
            self.assertIn(want, run(t, r, s), want)

    def test_v1b_r9_cases(self):
        orig = T9.mk
        T9.mk = mk
        try:
            q = T9.Q('test_q2_reasons')
            T9.J9 = J10
            q.test_q2_reasons()
        finally:
            T9.mk = orig
            import door_coopen_check_core9 as J9
            T9.J9 = J9

    def test_v2_anchor_view(self):
        t, r, s = mk()
        g = {'gate': {'passed': False, 'anchor': None}}
        gv, rec = J10.anchor_view(g, r, s)
        self.assertTrue(rec['consistent'])
        self.assertEqual(gv['gate']['anchor'], [1.02, -0.5, 3.03])
        self.assertIs(gv['gate']['passed'], False)
        self.assertIsNone(g['gate']['anchor'])
        for f in (lambda t, r, s: s['bench'].update(anchor=[1.02 + 1e-6, -0.5, 3.03]), lambda t, r, s: r.__setitem__('events', []),
                  lambda t, r, s: r['config'].__setitem__('start_pose', '1.03,-0.5,3.03')):
            t, r, s = mk()
            f(t, r, s)
            gv, rec = J10.anchor_view(g, r, s)
            self.assertFalse(rec['consistent'])
            self.assertIs(gv, g)

    def test_v3_real_chain(self):
        task, room, gate, solver, obs = load_real()
        ra = W1F2.judge(task, room, gate, solver, obs)
        self.assertTrue(ra['pass'], ra.get('why'))
        g0 = copy.deepcopy(gate)
        g0['gate']['anchor'] = None
        rb = W1F2.judge(task, room, g0, solver, obs)
        self.assertIn('k3m_room_evidence_invalid', rb['why'])
        r2, s2, g2 = bench_like(room, solver, gate)
        gv, rec = J10.anchor_view(g2, r2, s2)
        self.assertTrue(rec['consistent'], rec)
        rc = W1F2.judge(task, r2, gv, s2, obs)
        self.assertEqual((rc['pass'], rc['why']), (ra['pass'], ra['why']))
        r3, s3, g3 = bench_like(room, solver, gate, shift=0.05)
        gv3, rec3 = J10.anchor_view(g3, r3, s3)
        self.assertTrue(rec3['consistent'], rec3)
        rd = W1F2.judge(task, r3, gv3, s3, obs)
        self.assertFalse(rd['pass'])
        print('\n  V3-d 平移 50 mm 的鏈理由：', rd['why'][:6])

    def test_v4_outer_judge(self):
        task, room, gate, solver, obs = load_real()
        r2, s2, g2 = bench_like(room, solver, gate)
        res = J10.judge(task, r2, g2, s2, obs, os.path.join(RUN, 'evidence.jsonl'), os.path.join(RUN, 'task_trace.jsonl'))
        self.assertEqual(res['chain_exempt'], [])
        self.assertNotIn('k3m_room_evidence_invalid', res['why'])
        self.assertNotIn('park_anchor_missing', res['why'])
        self.assertTrue(res['bench']['anchor_crosscheck']['consistent'])
        print('\n  V4 外層 judge() 其餘理由（非台架趟的必然差異，不作通過宣稱）：', res['why'][:12])


if __name__ == '__main__':
    unittest.main(verbosity=2)
