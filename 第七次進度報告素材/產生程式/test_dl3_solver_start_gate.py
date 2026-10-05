#!/usr/bin/env python3
"""DL3 求解節點複本（dl3_wgmpc_wg2_node.py）待命條件：--require-topic-target 時，未收到合格話題目標不得離開待命；
不帶旗標時與原節點相同（RELEASED＋solver_start 即離開）。另核原節點檔未被改動、複本只多這兩處。

    python3 evaluation/test_dl3_solver_start_gate.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


import dl3_wgmpc_wg2_node as S                                         # noqa: E402
import rclpy                                                            # noqa: E402

rclpy.init()


class Pub:
    def publish(self, m):
        pass


class Ex:
    def __init__(self, nd, rx_after=None):
        self.nd, self.k, self.rx_after = nd, 0, rx_after

    def spin_once(self, timeout_sec=0.0):
        self.k += 1
        if self.rx_after is not None and self.k == self.rx_after:
            self.nd._T_rx = 'T'


def nd_(require):
    return types.SimpleNamespace(a=types.SimpleNamespace(require_topic_target=require), drawer_ready_pub=Pub(),
                                 _stop_req=None, _drawer_released=True, _drawer_start=True, _T_rx=None)


nd = nd_(True)
why = S.wait_for_drawer_handover(nd, Ex(nd), timeout_s=0.3)
check('require_flag_waits_without_target', why is not None and '逾時' in why, why)
nd = nd_(True)
ex = Ex(nd, rx_after=5)
why = S.wait_for_drawer_handover(nd, ex, timeout_s=5.0)
check('require_flag_starts_after_target_rx', why is None and ex.k >= 5, (why, ex.k))
nd = nd_(False)
check('without_flag_same_as_original', S.wait_for_drawer_handover(nd, Ex(nd), timeout_s=0.3) is None)
import subprocess                                                        # noqa: E402
d = subprocess.run(['diff', os.path.join(HERE, 'wgmpc_wg2_node.py'), os.path.join(HERE, 'dl3_wgmpc_wg2_node.py')],
                   capture_output=True, text=True).stdout
removed = [l for l in d.splitlines() if l.startswith('< ')]
check('copy_only_changes_wait_condition', len(removed) == 1 and 'nd._drawer_released and nd._drawer_start' in removed[0], removed)
rclpy.shutdown()
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
