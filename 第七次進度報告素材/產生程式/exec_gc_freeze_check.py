#!/usr/bin/env python3
"""一次性 gc.freeze 啟用後的開跑條件（Codex r31 §1；純邏輯）。旗標啟用時任一不符 ⇒ 執行端須在物理釋放前拒跑並封存（不得默默繼續、不得算成 GC 候選）。
verify(rec)（rec＝exec_gc_freeze.freeze_once() 的回傳）回傳 None（通過）或理由字串：
  freeze_not_ok        ok 不為 True（含例外）；
  record_incomplete    before／after 缺、或缺 enabled／threshold／freeze_count；
  gc_disabled_before   freeze 前自動 GC 未啟用；
  enabled_changed      freeze 前後 isenabled 不一致；
  threshold_changed    freeze 前後 get_threshold 不一致；
  freeze_count_not_increased  freeze 後永久世代數未增加（freeze 未生效）。

    python3 evaluation/test_exec_gc_freeze_check.py
"""
from __future__ import annotations


def verify(rec):
    if not isinstance(rec, dict) or rec.get('ok') is not True:
        return 'freeze_not_ok' + ('' if not isinstance(rec, dict) or not rec.get('error') else f"：{rec['error']}")
    b, a = rec.get('before'), rec.get('after')
    keys = ('enabled', 'threshold', 'freeze_count')
    if not isinstance(b, dict) or not isinstance(a, dict) or any(k not in b or k not in a for k in keys):
        return 'record_incomplete'
    if b['enabled'] is not True:
        return 'gc_disabled_before'
    if a['enabled'] != b['enabled']:
        return 'enabled_changed'
    if list(a['threshold']) != list(b['threshold']):
        return 'threshold_changed'
    if not (isinstance(a['freeze_count'], int) and isinstance(b['freeze_count'], int) and a['freeze_count'] > b['freeze_count']):
        return 'freeze_count_not_increased'
    return None
