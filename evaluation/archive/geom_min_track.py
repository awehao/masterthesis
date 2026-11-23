#!/usr/bin/env python3
"""幾何監看各類 d_pred 的真最小值追蹤（統一以公尺比較，輸出時才轉 mm）。
修正 10/31 發現的錯誤寫法：`if v['d_pred'] < mn.get(c, (9,))[0]: mn[c] = (round(v['d_pred'] * 1e3, 1), ...)`
——以公尺值比較已存的 mm 值，第一筆之後每列都會覆寫，結果是末列值而非最小值。
"""


class MinTracker:
    def __init__(self):
        self.m = {}                                  # 類別 → (d_pred[m], who, 列, 其他標記)

    def update(self, det, i, tag=None):
        for c, v in ((det or {}).get('classes') or {}).items():
            d = float(v['d_pred'])
            if c not in self.m or d < self.m[c][0]:
                self.m[c] = (d, v.get('who'), i, tag)

    def as_mm(self, nd=2):
        return {c: [round(d * 1e3, nd), w, i] + ([t] if t is not None else []) for c, (d, w, i, t) in self.m.items()}


def buggy_min(dets):
    """重現舊工具寫法（只供測試對照）。"""
    mn = {}
    for i, det in enumerate(dets):
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v.get('who'), i)
    return mn
