#!/usr/bin/env python3
"""door_regrasp_replan_v4.via_candidate_ok 反例（Codex r56 §3）：前段通過、後段失敗 ⇒ 不得成為候選；兩段都通過才回最小餘裕。

    python3 evaluation/test_door_regrasp_replan_v4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_regrasp_replan_v4 as R4                    # noqa: E402

assert R4.via_candidate_ok((True, 0.3), (False, {'stage': 'A'})) == (False, None)
assert R4.via_candidate_ok((False, {'stage': 'A'}), (True, 0.2)) == (False, None)
assert R4.via_candidate_ok((True, 0.3), (True, 0.2)) == (True, 0.2)
print('全部通過')
