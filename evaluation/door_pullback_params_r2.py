#!/usr/bin/env python3
"""往後拉開 48° 登錄 r2：r1 只改 solver_change——開門段偏移估計改回 W1 語義（切入 free 即凍結，不在 OPEN 更新）。
依據：door_pullback_pb1_2 OPEN 期間 d̂ 由 d_init（j2 0.0171、j3 −0.0088）學到 0.0270／−0.0146（384 次更新），手相對把手偏移，
finger1 接觸力 5→0.15 N ⇒ finger_contact_lost @ 19.26 s（門 ~14°）。r1 的 offset_update_in_open 是自 coopen r11 抄入（r10 為繞緣版加，Codex 已註記可能把接觸負載學成偏差）；
w1f_params_60_r2（W1 成功配置）為 offset_hold_on_free、無 OPEN 更新。節點仍 ADM20（換抓需要）。其餘鍵值逐字同 r1。

    python3 evaluation/door_pullback_params_r2.py   # 寫 results/door/pullback_params_48_r2.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT, W1F = os.path.join(R, 'pullback_params_48_r1.json'), os.path.join(R, 'pullback_params_48_r2.json'), os.path.join(R, 'w1f_params_60_r2.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p, w = json.load(open(SRC)), json.load(open(W1F))
    if w['solver_change'].get('offset_hold_on_free') is not True or w['solver_change'].get('offset_update_in_open'):
        raise SystemExit('W1f solver_change 不是凍結語義 ⇒ 不建')
    p['solver_change'] = dict(w['solver_change'], node='dl3_wgmpc_wg2_adm20_node.py', offset_update_in_open=False,
                              rule_r2='W1 語義：底盤模式首次切入 free 的那一輪凍結 d̂，持續到 PAUSE／DONE 或中止；OPEN 期間不更新（不加 --offset-update-in-open）',
                              basis_r2='door_pullback_pb1_2：OPEN 期間 d̂ 學習（j2 0.0171→0.0270、j3 −0.0088→−0.0146）⇒ finger_contact_lost')
    p['derived_from_r1'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
