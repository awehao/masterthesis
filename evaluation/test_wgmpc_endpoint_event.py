#!/usr/bin/env python3
"""執行端 stage 3 回報判定的反例核對（**測的是執行端實際呼叫的那支函式**）。

`ENV.endpoint_event` 是執行端主迴圈裡實際使用的判定；本檔直接驅動它，
不另寫一份模仿邏輯。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402

FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:56s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def ident(seq, adapter_seq=900, recv=1.0):
    return {'run_id': ENV.run_id_num('ep_test'), 'source_seq': seq,
            'adapter_out_seq': adapter_seq, 'derived': True,
            'recv_sim_t': recv}


def call(idn, **kw):
    base = dict(failed=False, fail_reported=False, seen_has=False,
                modified=False, exec_mode=0, sim_t=2.0,
                applied9=[0.0] * 9, lam=1.0, chain_recv_seq=7, step_id=11)
    base.update(kw)
    return ENV.endpoint_event(idn, **base)


def main() -> int:
    print('P1  **正常套用 A → A 已記入 seen → 再失效，仍必須留下停止事件**')
    seen = set()
    fail_reported = False
    a = ident(501)
    e1 = call(a, chain_recv_seq=7)
    ck('A 的首次套用有事件', e1 is not None and e1['kind'] == ENV.K_NORMAL,
       e1['kind_name'] if e1 else 'None')
    if e1:
        seen.add(7)
    e2 = call(a, chain_recv_seq=7, seen_has=(7 in seen))
    ck('A 不會重複回報首次套用', e2 is None, str(e2))
    # 之後才失效：**同一個 chain_recv_seq，已在 seen 裡**
    e3 = call(a, chain_recv_seq=7, seen_has=(7 in seen),
              failed=True, fail_reported=fail_reported, exec_mode=3, sim_t=5.0)
    ck('**失效停止仍留下事件**（不被 seen 擋掉）',
       e3 is not None and e3['kind'] == ENV.K_FAIL_LATCHED,
       e3['kind_name'] if e3 else '**None —— 被擋掉了**')
    if e3:
        fail_reported = True
        ck('失效事件 source_seq = −1（不冒稱原命令成功套用）',
           e3['source_seq'] == -1 and e3['derived'] is False,
           f"source_seq={e3['source_seq']} derived={e3['derived']}")
        ck('原命令序號只在**診斷欄位**',
           e3['diag_last_source_seq_before_fail'] == 501,
           str(e3['diag_last_source_seq_before_fail']))
        ck('上游來歷（src_seq）仍保留', e3['src_seq'] == 900,
           str(e3['src_seq']))
    e4 = call(a, chain_recv_seq=7, seen_has=True, failed=True,
              fail_reported=fail_reported, exec_mode=3)
    ck('失效事件只報一次', e4 is None, str(e4))

    print('\nP2  話題 payload 與檔案事件**同一份來源語意**')
    if e3:
        pl = ENV.event_to_payload(e3, 42)
        d = ENV.decode(pl)
        ck('payload 的 source_seq 與事件一致',
           d['source_seq'] == e3['source_seq'], f"{d['source_seq']}")
        ck('payload 的 kind 與事件一致', d['kind'] == e3['kind'],
           d['kind_name'])
        ck('payload 的 src_seq 與事件一致', d['src_seq'] == e3['src_seq'],
           str(d['src_seq']))
        ck('payload 的 derived 與事件一致', d['derived'] == e3['derived'], '')
    if e1:
        pl1 = ENV.event_to_payload(e1, 41)
        d1 = ENV.decode(pl1)
        ck('正常事件的 source_seq 一致',
           d1['source_seq'] == e1['source_seq'] == 501, str(d1['source_seq']))

    print('\nP3  其他分支')
    ck('無身分（_env_map 查不到）且未失效 ⇒ 不報',
       call(None) is None, '')
    ck('無身分但**已失效** ⇒ 仍報停止事件',
       (lambda e: e is not None and e['kind'] == ENV.K_FAIL_LATCHED)(
           call(None, failed=True)), '')
    em = call(ident(502, 901), modified=True, chain_recv_seq=8)
    ck('E2 修改標 modified_by_this_stage',
       em is not None and em['kind'] == ENV.K_MODIFIED,
       em['kind_name'] if em else 'None')
    ck('修改事件仍保留 source_seq（可歸屬）',
       em is not None and em['source_seq'] == 502, '')

    print('\nP4  執行端確實呼叫這支函式（非另寫一份）')
    src = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
               encoding='utf-8').read()
    ck('執行端呼叫 ENV.endpoint_event', 'ENV.endpoint_event(' in src, '')
    ck('執行端用 ENV.event_to_payload 組話題',
       'ENV.event_to_payload(_ev, node._env_out_seq)' in src, '')
    ck('執行端的失效判定不在 seen 條件內',
       '_rs not in node._env_seen' not in src,
       '舊的巢狀條件已移除')

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。測的是**執行端實際呼叫的判定函式**。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
