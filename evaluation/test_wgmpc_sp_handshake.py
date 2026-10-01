#!/usr/bin/env python3
"""設定點握手閘門的離線核對。不開模擬器、不用 ROS。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wgmpc_sp_handshake import (ARMED, EXPECT_G, EXPECT_INSTANT, HOLD,  # noqa
                                INIT, SetpointGate, SpSample)

FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:58s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


GOOD_META = {'sampling_instant': EXPECT_INSTANT,
             'composed_G_for_dt_0p05': EXPECT_G}


def samp(step, t, ready, sp=None, mode=0, api=True):
    return SpSample.from_data(
        [step, t, 1.0 if ready else 0.0]
        + list(sp if sp is not None else [float('nan')] * 6)
        + [mode, 1.0 if api else 0.0])


def main() -> int:
    SP = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]

    print('H1  尚未收到 meta ⇒ INIT，且**不給設定點**')
    g = SetpointGate()
    st, s, why = g.decide(1.0)
    ck('狀態為 INIT 且 s 為 None', st == INIT and s is None, why)

    print('\nH2  meta 契約不吻合 ⇒ 拒絕進入求解')
    for bad, tag in (({'sampling_instant': '本步寫入前',
                       'composed_G_for_dt_0p05': EXPECT_G}, '取樣時刻'),
                     ({'sampling_instant': EXPECT_INSTANT,
                       'composed_G_for_dt_0p05': 0.012570}, '合成增益')):
        g = SetpointGate(); g.feed_meta(bad)
        g.feed(samp(10, 1.0, True, SP))
        st, s, why = g.decide(1.0)
        ck(f'{tag}不吻合 ⇒ INIT、s 為 None', st == INIT and s is None, why)

    print('\nH3  ready = 0 ⇒ 只送初始化命令，**不以 s = q 代替**')
    g = SetpointGate(); g.feed_meta(GOOD_META)
    g.feed(samp(1, 0.10, False))
    st, s, why = g.decide(0.10)
    ck('狀態 INIT 且 s 為 None', st == INIT and s is None, why)
    u = g.init_command()
    ck('初始化命令為全零九維', len(u) == 9 and all(v == 0.0 for v in u),
       f'n_init_cmd={g.n_init_cmd}')

    print('\nH4  ready = 1 且新鮮 ⇒ ARMED，回傳**執行端回報的**設定點')
    g.feed(samp(2, 0.20, True, SP))
    st, s, why = g.decide(0.21)
    ck('狀態 ARMED', st == ARMED, why)
    ck('回傳的設定點逐項等於回報值', s is not None and list(s) == SP,
       str(s))

    print('\nH5  曾就緒後樣本過期 ⇒ HOLD，**不沿用舊設定點**')
    st, s, why = g.decide(0.20 + 0.25)
    ck('狀態 HOLD 且 s 為 None', st == HOLD and s is None, why)

    print('\nH6  ready 轉回 0（例如鏈失效）⇒ 回到 INIT、不給設定點')
    g.feed(samp(3, 0.50, False))
    st, s, why = g.decide(0.50)
    ck('狀態 INIT 且 s 為 None', st == INIT and s is None, why)

    print('\nH7  含非有限值的「就緒」樣本要被拒絕')
    g2 = SetpointGate(); g2.feed_meta(GOOD_META)
    g2.feed(samp(4, 1.00, True, [0.1] * 5 + [float('nan')]))
    st, s, why = g2.decide(1.00)
    ck('狀態 INIT 且 s 為 None', st == INIT and s is None, why)

    print('\nH8  未來時間戳（負年齡）要被拒絕，不當成新鮮')
    g3 = SetpointGate(); g3.feed_meta(GOOD_META)
    g3.feed(samp(5, 2.00, True, SP))
    st, s, why = g3.decide(1.50)
    ck('負年齡 ⇒ 不是 ARMED', st != ARMED, why)

    print('\nH9  欄數不符要拋錯（不靜默補零）')
    try:
        SpSample.from_data([0.0] * 9)
        ck('欄數不符拋錯', False, '**未拋錯**')
    except ValueError as e:
        ck('欄數不符拋錯', '11 欄' in str(e), str(e))

    print('\nH10 執行端 meta 的實際內容必須通得過本閘門')
    # 從執行端原始碼取出 meta 的字面值，避免兩邊各寫一份而漂移
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'isaac_wholebody_sim_e2.py'),
               encoding='utf-8').read()
    has_inst = "'sampling_instant': '**本步寫入後**（apply_action 之後）'" in src
    has_g = "'composed_G_for_dt_0p05': 0.008640" in src
    ck('執行端 meta 的 sampling_instant 與閘門期望一致', has_inst,
       EXPECT_INSTANT if has_inst else '**對不上**')
    ck('執行端 meta 的 composed_G 與閘門期望一致', has_g,
       f'{EXPECT_G}' if has_g else '**對不上**')
    g4 = SetpointGate()
    g4.feed_meta(GOOD_META)
    g4.feed(samp(9, 3.00, True, SP))
    ck('以該 meta 可進入 ARMED', g4.decide(3.0)[0] == ARMED,
       json.dumps(g4.report()['contract'], ensure_ascii=False))

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**這是握手協定的離線核對**，'
              '不含與執行端的實際連線 —— 那要等物理對照。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
