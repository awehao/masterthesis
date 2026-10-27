#!/usr/bin/env python3
"""台架啟動 r2 的版本化差異靜態測試（Codex r21）。
  W1 ev9r2 對 ev9：被刪改的非註解行限於：純邏輯匯入、釋放當輪 break 區段（加真入口）、released 狀態發布（改為第 1 步核對後）、bench_go 後、封裝入口條件
     （第 1 步首筆固定）；新增含 BENCH.release_intake、BENCH.verify_first_step、首步失敗閂鎖。
  W2 任務節點 v8 對 v7：刪改限於兩個匯入、_bench_st 三行、策略建構一行；新增 StateFilter 建立。
  W3 跑法 bench2 對 bench：只差 FRZ、執行端、任務節點、結束字樣。

    python3 evaluation/test_door_coopen_core_bench2_static.py
"""
import difflib
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def diff(a, b):
    A, B = open(os.path.join(HERE, a)).read().splitlines(), open(os.path.join(HERE, b)).read().splitlines()
    d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
    rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
    return rm, open(os.path.join(HERE, b)).read()


class W(unittest.TestCase):
    def test_w1_ev9r2(self):
        rm, src = diff('isaac_door_k2_sim_ev9.py', 'isaac_door_k2_sim_ev9r2.py')
        ok = ('import door_bench_start as BENCH', '                break', "        node.bench_state_pub.publish(String(data=json.dumps({'state': bench_rel.state",
              '                                                            default=float)))', '        if a.cmd_env and node.wb_env_latest is not None:')
        for ln in rm:
            self.assertTrue(ln.startswith(ok), ln)
        for k in ('BENCH.release_intake(', 'BENCH.verify_first_step(', "k2_latch.on_msg(f'bench_first_step：{_whyf}'", 'and not bench_first_pending'):
            self.assertIn(k, src, k)

    def test_w2_node8(self):
        rm, src = diff('door_coopen_core_task_node7.py', 'door_coopen_core_task_node8.py')
        ok = ('import door_coopen_bench_grasp as BG', 'import door_bench_start as BENCH', '    def _bench_st(self, m):', '            self.bench = json.loads(m.data)',
              '            pass', '    pol = BG.bench_grasp(', '        try:', '        except Exception:')
        for ln in rm:
            self.assertTrue(ln.startswith(ok), ln)
        self.assertIn('nd.bench_filter = BENCH.StateFilter(a.run_id)', src)

    def test_w3_runner(self):
        rm, src = diff('run_door_coopen_core_bench.sh', 'run_door_coopen_core_bench2.sh')
        for ln in rm:
            self.assertTrue(ln.startswith(('FRZ=', 'spawn sim ', 'spawn task ', 'say "=== 結束')), ln)
        self.assertIn('isaac_door_k2_sim_ev9r2.py --bench-start', src)
        self.assertIn('door_coopen_core_task_node8.py', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
