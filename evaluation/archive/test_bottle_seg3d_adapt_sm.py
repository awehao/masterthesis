#!/usr/bin/env python3
"""自適應狀態機的最小核對（Codex 20261101_232233；無 Isaac）。
1 假 articulation：關節索引 ACT（整數陣列），每步把狀態機的設定點寫進 tgt 再以 tgt[ACT] 下指令（同 isaac_bottle_seg3d_collect2.apply）；
  走「V0 缺底 → V_LOW 完整 → 接受 → 經 V0 回 K2_READY → 回收」整段，核對：每步 ACT 仍為整數陣列、設定點有限、決策軌跡與期望一致、終點為 q_tuck；
2 決策紀錄：每次決策後 decisions.jsonl 立刻可讀到該行（flush／fsync）；
3 例外存檔：主迴圈在第一次決策後丟例外 ⇒ save_fn 被呼叫且收到原例外、原例外照樣拋出；save_fn 自己也丟例外時，拋出的仍是原例外。
    python3 evaluation/test_bottle_seg3d_adapt_sm.py
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_seg3d_adapt_sm as SM                      # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


class FakeArt:
    def __init__(self):
        self.dof = 9
        self.idx = [0, 1, 2, 3, 4, 5]
        self.ACT = np.array(sorted(self.idx + [6]), np.int32)
        self.tgt = np.zeros(self.dof)
        self.n = 0

    def apply(self, q_arm):
        self.tgt[self.idx] = q_arm
        cmd = np.asarray(self.tgt, np.float32)[self.ACT]              # 同腳本：索引必須是整數陣列
        assert self.ACT.dtype.kind == 'i' and np.isfinite(cmd).all()
        self.n += 1


def mk(st, **kw):
    return dict(dict(stamp=st, recv_age_s=0.06, valid=1.0, xy_valid=1.0, top_observed=1.0, support_observed=1.0, height_valid=1.0, bottom_cut=0.0, top_cut=0.0, edge_top_unknown=0.0), **kw)


def build(log_path):
    views = dict(V0=np.full(6, 0.1), V_LOW=np.full(6, 0.3), V_HIGH=np.full(6, 0.5))
    return SM.AdaptiveSM(np.zeros(6), np.full(6, -0.2), views, 1.0, 0.8, lambda a, b: 1.5, 2.0, 4.0, log_path)


def main():
    d = tempfile.mkdtemp()
    log = os.path.join(d, 'decisions.jsonl')
    sm, art = build(log), FakeArt()
    rows = []
    lines_after_first = None
    t, dt = 0.0, 0.01
    while t < 20.0 and sm.state != 'end':
        if sm.state == 'observe':                                    # 依目前視點產生 5 Hz 旁路列：V0 缺底，V_LOW 完整
            if abs((t * 5) - round(t * 5)) < 1e-6:
                rows.append(mk(round(t, 4)) if sm.view == 'V_LOW' else mk(round(t, 4), support_observed=0.0, height_valid=0.0, bottom_cut=1.0))
        n_tr = len(sm.trace)
        q = sm.step(t, lambda: rows)
        art.apply(q if q is not None else np.zeros(6))
        if n_tr == 0 and len(sm.trace) == 1:
            lines_after_first = open(log).read().strip().splitlines()
        t = round(t + dt, 6)
    acts = [(x['view'], x['action'], x['arg']) for x in sm.trace]
    chk('1 假 articulation：每步索引為整數陣列、設定點有限（含決策後的過渡與回程）', art.n > 100, art.n)
    chk('1 決策軌跡 V0 缺底→V_LOW、V_LOW 接受', acts == [('V0', 'move', 'V_LOW'), ('V_LOW', 'accept', None)], acts)
    chk('1 結束於 end、最後設定點＝q_tuck', sm.state == 'end' and np.allclose(art.tgt[:6], np.full(6, -0.2)), (sm.state, art.tgt[:6]))
    chk('2 第一次決策後 decisions.jsonl 立即有該行', lines_after_first is not None and len(lines_after_first) == 1 and json.loads(lines_after_first[0])['arg'] == 'V_LOW', lines_after_first)
    # 3 例外存檔
    got = {}

    def loop_boom():
        sm2 = build(os.path.join(d, 'd2.jsonl'))
        rows2, tt = [], 0.0
        while True:
            if sm2.state == 'observe' and abs((tt * 5) - round(tt * 5)) < 1e-6:
                rows2.append(mk(round(tt, 4), support_observed=0.0, height_valid=0.0, bottom_cut=1.0))
            sm2.step(tt, lambda: rows2)
            if sm2.trace:
                raise IndexError('模擬：決策後第一步出錯')
            tt = round(tt + 0.01, 6)

    def save_ok(err):
        got['saved'] = repr(err)
    try:
        SM.run_with_partial_save(loop_boom, save_ok)
        raised = None
    except IndexError as e:
        raised = e
    chk('3 例外 ⇒ 部分存檔被呼叫且收到原例外、原例外照樣拋出', raised is not None and 'IndexError' in got.get('saved', ''), (raised, got))

    def save_boom(err):
        raise OSError('模擬：存檔失敗')
    try:
        SM.run_with_partial_save(loop_boom, save_boom)
        raised2 = None
    except BaseException as e:                                       # noqa: BLE001
        raised2 = e
    chk('3 存檔自己也失敗 ⇒ 拋出的仍是原例外（IndexError）', isinstance(raised2, IndexError), type(raised2))
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
