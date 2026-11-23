#!/usr/bin/env python3
"""is_ros_shutdown_exc：只有 ROS 外部關閉才算，程式錯誤不得被吞成 external_shutdown。"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rclpy.executors import ExternalShutdownException             # noqa: E402
from wgmpc_wg2_node import is_ros_shutdown_exc                   # noqa: E402


class RCLError(RuntimeError):
    """與 rclpy 的 RCLError 同名的替身（只比對型別名稱）。"""


@pytest.mark.parametrize('exc,ros_ok,expect', [
    (ExternalShutdownException(), True, True),
    (ExternalShutdownException(), False, True),
    (RCLError('context is not valid'), False, True),
    (RCLError('some other ros failure'), True, False),   # ROS 仍在 ⇒ 是真錯誤
    (ValueError('bug'), False, False),                    # 程式錯誤，即使 ROS 已關閉
    (KeyError('x'), False, False),
    (RuntimeError('context is not valid'), False, False),  # 型別不對就不算
])
def test_classification(exc, ros_ok, expect):
    assert is_ros_shutdown_exc(exc, ros_ok) is expect
