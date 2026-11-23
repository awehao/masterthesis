#!/usr/bin/env python3
"""晚降速候選（版本化子類，基線＝bottle_w1_fprof_ctrl；Codex 20261111 第 9 輪核可離線接線與假世界短跑，**未經 Isaac**）。
終端低速區由 x_pre 前 0.6 m 縮為 SLOW_BACK＝0.05 m；由巡航降到 v_c 的減速位置由同源剖面後向掃描（A_SLEW）決定（約再提早 0.08 m）。
參考速度、MPC 預覽、站位時間成本、TPA 時長、TPAS 進度表仍共用同一份剖面；x_pre 處速度、INSERT、閉爪、切換閘門、安全判準不變。0.05 是單一候選值，不是最適值。"""
import bottle_w1_fprof_ctrl as FC

SLOW_BACK = 0.05


class W1PlaceController(FC.W1PlaceController):
    group = FC.W1PlaceController.group + '_LATESLOW'
    slow_back = SLOW_BACK
