#!/usr/bin/env python3
"""台架初態啟動的版本化差異靜態測試（Isaac 執行端無法離線執行；以差異限定＋旗標關閉回退表達式核對「預設關閉時原流程不變」）。
  S1 ev9 對 ev5：被刪改的非註解行恰為 4 行（初始 apply_action、DualSourceExecutor 初始控制者、主迴圈條件、/park/gate 字典尾），
     且對應的新行含旗標關閉回退：_tgt0＝q0（預設）、initial=(AUTH_WHOLEBODY if a.bench_start else AUTH_NAV)、bench_go 預設 True、
     /park/gate 額外欄位只在 a.bench_start；所有新增的非註解行都在 BENCH 插入區段內（以檔頭說明的標記 B1–B5 或 bench 字樣定位）。
  S2 ADM17 對 ADM9：刪改恰 1 行（觀測器建構，改為帶 d0；d0 在無 bench 時為 None ⇒ 原行為）。
  S3 任務節點 v7 對 v6：刪改恰 1 行（策略建構）。
  S4 跑法 bench 對 iso2：刪改限於 FRZ、PRM、Q_INIT、起始字樣、執行端、展開節點兩行、任務節點、求解節點、等待旗標行、程序索引、展開節點等待、成功條件、結束字樣；
     新跑法不含 door_k2_wholebody_node 與 --wait-for-drawer-handover。

    python3 evaluation/test_door_coopen_core_bench_static.py
"""
import difflib
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def diff(a, b):
    A, B = open(os.path.join(HERE, a)).read().splitlines(), open(os.path.join(HERE, b)).read().splitlines()
    d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
    rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
    add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
    return rm, add, open(os.path.join(HERE, b)).read()


class S(unittest.TestCase):
    def test_s1_ev9(self):
        rm, add, src = diff('isaac_door_k2_sim_ev5.py', 'isaac_door_k2_sim_ev9.py')
        self.assertEqual(len(rm), 4, rm)
        for k in ('ArticulationAction(joint_positions=q0))', 'initial=AUTH_NAV', 'while rclpy.ok():', "'violation': park_state['violation'] is not None})))"):
            self.assertTrue(any(k in ln for ln in rm), k)
        for k in ('_tgt0, bench_cfg, bench_sp = q0, None, None', 'initial=(AUTH_WHOLEBODY if a.bench_start else AUTH_NAV)',
                  'bench_rel, bench_go = None, True', 'while rclpy.ok() and bench_go:',
                  "**({'bench': True, 'bench_anchor': park_state['bench_anchor']} if a.bench_start else {})"):
            self.assertIn(k, src, k)
        self.assertIn("ap.add_argument('--bench-start', action='store_true'", src)

    def test_s2_adm17(self):
        rm, add, src = diff('dl3_wgmpc_wg2_adm9_node.py', 'dl3_wgmpc_wg2_adm17_node.py')
        self.assertEqual(len(rm), 1, rm)
        self.assertIn('OINT.IntervalOffsetObserver(', rm[0])
        self.assertIn("d0=(None if self._bench is None else list(self._bench['cfg'].d_init))", src)
        self.assertIn("if getattr(a, 'bench_start', False):", src)

    def test_s3_node7(self):
        rm, add, src = diff('door_coopen_core_task_node6.py', 'door_coopen_core_task_node7.py')
        self.assertEqual(len(rm), 1, rm)
        self.assertIn('pol = IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy))(cfg,', rm[0])
        self.assertIn('pol = BG.bench_grasp(IG2.init_grasp2(AG2.gated2(PP2.K3MP2Policy)))(cfg,', src)

    def test_s4_runner(self):
        rm, add, src = diff('run_door_coopen_core_iso2.sh', 'run_door_coopen_core_bench.sh')
        ok = ('FRZ=', 'PRM=', 'Q_INIT=', 'say "=== 核心段', 'spawn sim ', 'spawn wholebody ', '  --stow-q "$Q_INIT" --park', 'spawn task ',
              'spawn solver ', '  --continuous --wait-for-drawer-handover', 'SIM_PID=', 'wait_or_stop "$WB_PID"', '[ "$HOW_TASK" = exited ]', 'say "=== 結束')
        for ln in rm:
            self.assertTrue(ln.startswith(ok), ln)
        code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
        self.assertNotIn('door_k2_wholebody_node', code)
        self.assertNotIn('--wait-for-drawer-handover', code)
        self.assertIn('isaac_door_k2_sim_ev9.py --bench-start', src)
        self.assertIn('dl3_wgmpc_wg2_adm17_node.py --bench-start', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
