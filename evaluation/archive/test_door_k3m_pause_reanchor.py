#!/usr/bin/env python3
"""door_k3m_pause_reanchor.HoldReanchor 純邏輯測試（W1d；Codex 20261025_010414 要求的測試面）。

    python3 evaluation/test_door_k3m_pause_reanchor.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_pause_reanchor as PRA      # noqa: E402

FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:56s} {"ok" if ok else "**錯**"}  {d}')
    if not ok:
        FAIL.append(n)


OLD = (1.0, 2.0, 0.5)
POSE = (1.0016, 1.9990, 0.5003)


def main():
    r = PRA.HoldReanchor(False)
    ck('未啟用：回傳 None、record 為 None', r.on_zero_edge(10.0, 1, 9.99, POSE, OLD, False) is None and r.record is None)
    r = PRA.HoldReanchor(True)
    na = r.on_zero_edge(10.0, {'slot': 7}, 9.99, POSE, OLD, False)
    ck('非零錨點誤差：applied、new＝實測', na == POSE and r.applied and r.record['new_anchor'] == list(POSE))
    ck('anchor_shift＝new−old（yaw 包角）', all(abs(a - b) < 1e-12 for a, b in zip(r.record['anchor_shift'],
                                                                                   [POSE[0] - OLD[0], POSE[1] - OLD[1], POSE[2] - OLD[2]])))
    ck('age 記錄', abs(r.record['age_s'] - 0.01) < 1e-9)
    ck('重送：第二次呼叫不再重設', r.on_zero_edge(11.0, 2, 10.99, (5.0, 5.0, 0.0), r.record['new_anchor'], False) is None
       and r.record['new_anchor'] == list(POSE) and r.n_calls == 2)
    r.note_published(3, 10, 10.05)
    ck('未 effective 前不登錄 first_published', r.record['first_published'] is None)
    r.note_effective(8, 10.05, [0.0, 0.0, 0.0])
    r.note_effective(9, 10.10, [1.0, 0.0, 0.0])
    ck('effective 只記第一次', r.record['effective']['slot'] == 8)
    r.note_published(8, 11, 10.06)
    r.note_published(9, 12, 10.11)
    ck('first_published 只記第一次', r.record['first_published'] == {'slot': 8, 'source_seq': 11, 'publish_sim_t_exact': 10.06})
    r = PRA.HoldReanchor(True)
    ck('中止優先：aborted ⇒ 不重設', r.on_zero_edge(10.0, 1, 9.99, POSE, OLD, True) is None and r.record['status'] == 'skipped'
       and r.record['why'] == 'aborted')
    ck('中止後不再嘗試', r.on_zero_edge(10.1, 2, 10.09, POSE, OLD, False) is None and not r.applied)
    for tag, pose in (('缺量測 None', None), ('NaN', (1.0, math.nan, 0.0)), ('inf', (math.inf, 0.0, 0.0)), ('長度 2', (1.0, 2.0)),
                      ('非數字', ('a', 0.0, 0.0))):
        r = PRA.HoldReanchor(True)
        ck(f'量測不合法拒收（{tag}）', r.on_zero_edge(10.0, 1, 9.99, pose, OLD, False) is None and r.record['status'] == 'rejected'
           and r.record['why'] == 'pose_invalid' and not r.applied)
    r = PRA.HoldReanchor(True, max_age_s=0.2)
    ck('量測過期拒收（0.25 s）', r.on_zero_edge(10.0, 1, 9.75, POSE, OLD, False) is None and r.record['why'] == 'pose_stale')
    r = PRA.HoldReanchor(True, max_age_s=0.2)
    ck('量測時間在未來拒收', r.on_zero_edge(10.0, 1, 10.05, POSE, OLD, False) is None and r.record['why'] == 'pose_stale')
    r = PRA.HoldReanchor(True, max_age_s=0.2)
    ck('量測時間 NaN 拒收', r.on_zero_edge(10.0, 1, math.nan, POSE, OLD, False) is None and r.record['why'] == 'pose_stale')
    r = PRA.HoldReanchor(True, max_age_s=0.2)
    ck('邊界 age＝0.2 接受', r.on_zero_edge(10.0, 1, 9.8, POSE, OLD, False) == POSE)
    r = PRA.HoldReanchor(True)
    ck('舊錨點缺 ⇒ 拒收', r.on_zero_edge(10.0, 1, 9.99, POSE, None, False) is None and r.record['why'] == 'old_anchor_missing')
    r = PRA.HoldReanchor(True)
    r.on_zero_edge(10.0, 1, 9.99, (1.0, 2.0, math.pi - 0.01), (1.0, 2.0, -math.pi + 0.01), False)
    ck('yaw 包角：shift≈−0.02', abs(r.record['anchor_shift'][2] + 0.02) < 1e-9, str(r.record['anchor_shift'][2]))
    r = PRA.HoldReanchor(True)
    r.on_zero_edge(10.0, 1, 9.99, None, OLD, False)
    r.note_effective(1, 10.0, [0, 0, 0])
    ck('拒收後 note_effective 不登錄', r.record['effective'] is None)
    print('全部通過' if not FAIL else '失敗：' + '、'.join(FAIL))
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
