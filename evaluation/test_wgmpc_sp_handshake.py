#!/usr/bin/env python3
"""設定點握手閘門的離線核對。不開模擬器、不用 ROS。

H6 的預期**已更正**：曾就緒之後 `ready = 0` 應進 HOLD，不是回 INIT。
原預期（回 INIT）會讓「運行中資料失效」被誤當成首次啟動而重送初始化命令。
"""
from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wgmpc_sp_handshake import (ARMED, EXPECT_INSTANT, FAILED, HOLD,  # noqa
                                INIT, SP_COLS, SetpointGate, SpSample)

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def ck(n, ok, d=''):
    print(f'  {n:56s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


PHYS_DT = 0.01
GOOD_META = {'sampling_instant': EXPECT_INSTANT, 'physics_dt_s': PHYS_DT,
             'cols': list(SP_COLS),
             'joint_order': [f'joint{i}' for i in range(1, 7)]}
SP = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]


def samp(step, t, ready, sp=None, mode=0, api=True):
    return SpSample.from_data(
        [step, t, 1.0 if ready else 0.0]
        + list(sp if sp is not None else ([float('nan')] * 6 if not ready
                                          else SP))
        + [mode, 1.0 if api else 0.0])


def armed_gate():
    g = SetpointGate(phys_dt_s=PHYS_DT)
    g.feed_meta(GOOD_META)
    g.feed(samp(1, 1.00, True))
    assert g.decide(1.00)[0] == ARMED
    return g


def main() -> int:
    print('H1  尚未收到 meta ⇒ INIT，且**不給設定點**')
    g = SetpointGate(phys_dt_s=PHYS_DT)
    st, s, why = g.decide(1.0)
    ck('INIT 且 s 為 None', st == INIT and s is None, why)

    print('\nH2  介面契約核對：取樣時刻／物理步長／欄位／關節順序')
    for bad, tag in (
            (dict(GOOD_META, sampling_instant='本步寫入前'), '取樣時刻'),
            (dict(GOOD_META, physics_dt_s=0.005), '物理步長'),
            (dict(GOOD_META, physics_dt_s=float('nan')), '物理步長為 NaN'),
            (dict(GOOD_META, cols=['x', 'y']), '欄位'),
            (dict(GOOD_META, joint_order=['j1'] * 6), '關節順序'),
            (dict(GOOD_META, some_coeff=float('nan')), '任一數值欄位為 NaN')):
        gg = SetpointGate(phys_dt_s=PHYS_DT)
        gg.feed_meta(bad)
        gg.feed(samp(10, 1.0, True))
        st, s, why = gg.decide(1.0)
        ck(f'{tag}不吻合 ⇒ 不是 ARMED、s 為 None',
           st != ARMED and s is None, why[:76])
    ck('契約**不核對 G**（模型係數，非執行端量測）',
       'G' not in str(SetpointGate(phys_dt_s=PHYS_DT).report()
                      ['contract_checked']),
       str(SetpointGate(phys_dt_s=PHYS_DT).report()['contract_checked']))

    print('\nH3  首次 ready = 0 ⇒ INIT，只送全零初始化命令')
    g = SetpointGate(phys_dt_s=PHYS_DT)
    g.feed_meta(GOOD_META)
    g.feed(samp(1, 0.10, False))
    st, s, why = g.decide(0.10)
    ck('INIT 且 s 為 None', st == INIT and s is None, why)
    u = g.init_command()
    ck('初始化命令為全零九維', len(u) == 9 and all(v == 0.0 for v in u),
       f'n_init_cmd={g.n_init_cmd}')

    print('\nH4  ready = 1 且新鮮 ⇒ ARMED，回傳**執行端回報的**設定點')
    g.feed(samp(2, 0.20, True))
    st, s, why = g.decide(0.21)
    ck('ARMED', st == ARMED, why)
    ck('設定點逐項等於回報值', s is not None and list(s) == SP, str(s))

    print('\nH5  曾就緒後樣本過期 ⇒ HOLD，不沿用舊設定點')
    st, s, why = g.decide(0.20 + 0.25)
    ck('HOLD 且 s 為 None', st == HOLD and s is None, why[:70])

    print('\nH6  **已更正**：曾就緒後 ready = 0 ⇒ HOLD（不是 INIT）')
    g.feed(samp(3, 0.50, False))
    st, s, why = g.decide(0.50)
    ck('HOLD 且 s 為 None', st == HOLD and s is None, why[:70])
    try:
        g.init_command()
        ck('曾就緒後**不得**再送初始化命令', False, '**未拋錯**')
    except RuntimeError as e:
        ck('曾就緒後**不得**再送初始化命令', '不重啟握手' in str(e), 'RuntimeError')

    print('\nH7  含非有限值的「就緒」樣本要被拒絕')
    g2 = SetpointGate(phys_dt_s=PHYS_DT)
    g2.feed_meta(GOOD_META)
    g2.feed(samp(4, 1.00, True, [0.1] * 5 + [float('nan')]))
    st, s, why = g2.decide(1.00)
    ck('不是 ARMED 且 s 為 None', st != ARMED and s is None, why)

    print('\nH8  未來時間戳（負年齡）要被拒絕')
    g3 = SetpointGate(phys_dt_s=PHYS_DT)
    g3.feed_meta(GOOD_META)
    g3.feed(samp(5, 2.00, True))
    st, s, why = g3.decide(1.50)
    ck('不是 ARMED', st != ARMED, why[:70])

    print('\nH9  欄數不符要拋錯（不靜默補零）')
    try:
        SpSample.from_data([0.0] * 9)
        ck('欄數不符拋錯', False, '**未拋錯**')
    except ValueError as e:
        ck('欄數不符拋錯', f'{len(SP_COLS)} 欄' in str(e), str(e))

    print('\nH10 **新增反例** exec_mode = 3（失效閂鎖）⇒ FAILED，不得求解')
    g4 = SetpointGate(phys_dt_s=PHYS_DT)
    g4.feed_meta(GOOD_META)
    g4.feed(samp(6, 1.00, True, mode=3))
    st, s, why = g4.decide(1.00)
    ck('FAILED 且 s 為 None', st == FAILED and s is None, why[:70])
    ck('失效閂鎖優先於契約問題', (lambda gg: (gg.feed_meta({}),
                                     gg.feed(samp(6, 1.0, True, mode=3)),
                                     gg.decide(1.0)[0])[-1] == FAILED)(
        SetpointGate(phys_dt_s=PHYS_DT)), '未收 meta 也先報 FAILED')

    print('\nH11 **新增反例** api_applied = False ⇒ 不得當成已套用放行')
    g5 = SetpointGate(phys_dt_s=PHYS_DT)
    g5.feed_meta(GOOD_META)
    g5.feed(samp(7, 1.00, True, api=False))
    st, s, why = g5.decide(1.00)
    ck('首次出現 ⇒ INIT、s 為 None', st == INIT and s is None, why[:70])
    g6 = armed_gate()
    g6.feed(samp(8, 1.05, True, api=False))
    st, s, why = g6.decide(1.05)
    ck('曾就緒後出現 ⇒ HOLD、s 為 None', st == HOLD and s is None, why[:60])

    print('\nH12 執行端 meta 的實際內容必須通得過本閘門（由原始碼取值）')
    src = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
               encoding='utf-8').read()
    i = src.index('node.sp_meta_pub.publish')
    blk = src[src.index('json.dumps({', i):]
    blk = blk[:blk.index('}, ensure_ascii=False)') + 1]
    lit = {}
    node = ast.parse('d = ' + blk[len('json.dumps('):]).body[0].value
    for k, v in zip(node.keys, node.values):
        try:
            lit[ast.literal_eval(k)] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            lit[ast.literal_eval(k)] = '<非字面值>'
    ck('meta 的 sampling_instant 與閘門期望一致',
       lit.get('sampling_instant') == EXPECT_INSTANT,
       str(lit.get('sampling_instant'))[:40])
    ck('meta 有回報 physics_dt_s（執行端量測事實）',
       'physics_dt_s' in lit, str(lit.get('physics_dt_s')))
    ck('meta **不再**硬編碼合成增益 G',
       not any('composed_G' in k for k in lit), ','.join(sorted(lit))[:90])
    g7 = SetpointGate(phys_dt_s=PHYS_DT)
    g7.feed_meta(GOOD_META)
    g7.feed(samp(9, 3.00, True))
    ck('以該契約可進入 ARMED', g7.decide(3.0)[0] == ARMED,
       json.dumps(g7.report()['contract_checked'], ensure_ascii=False))

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**這是握手協定的離線核對**，'
              '不含與執行端的實際連線 —— 那要等物理對照。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
