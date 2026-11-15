#!/usr/bin/env python3
"""閉鏈一致停法的共用定義（執行層的停穩判定與 door_cstop 的停止預測用同一個函式）。"""
D_STEPS = 8                                                          # 命令管線長度（物理步；同 door_open_exec_ctrl.D_STEPS，起始時核對）


def motion_cleared(steps_since_anchor):
    """底盤保持錨定後，錨定前發出的運動命令是否已全部排出管線：錨定那拍起算滿 D_STEPS 個物理步。
    之後管線內只有保持模式下發出的命令（底盤保持＋隨之的手臂閉鏈跟隨）。"""
    return steps_since_anchor >= D_STEPS
