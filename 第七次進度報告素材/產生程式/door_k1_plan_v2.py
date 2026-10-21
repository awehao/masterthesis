#!/usr/bin/env python3
"""K1 v2 登錄計畫：與 door_k1_plan.py（r2 凍結，不改）**完全相同的短測、力值、時窗**，只把規格改指 door_local_v2。"""
from door_k1_plan import A_OPEN, DT, WALL_TEST_S, WALL_TOTAL_S, WALLS, handle_radius, n_steps, plan   # noqa: F401

SPEC_REL = 'src/my_omnibot_description/config/door_local_v2.yaml'
