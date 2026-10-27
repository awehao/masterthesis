#!/usr/bin/env python3
"""核心段登錄檔 r10：r9＋solver_change.offset_update_in_open（Codex r27 唯一候選：開門段恢復既有 interval 觀測器的合格區間更新）。r9 不覆寫；其餘鍵值不變。
  solver_change.offset_hold_on_free 仍為 True（保持機制保留），新增 offset_update_in_open True：保持觸發改為 K3-M 暫停生效（不再於首次切入 free 凍結）；
  node 改為 dl3_wgmpc_wg2_adm17r4_node.py。依據：door_coopen_core_bench_3 的因果影子重播（開迴路；不代表閉迴路）。

    python3 evaluation/door_coopen_core_params_r10.py   # 寫 results/door/coopen_core_params_48_r10.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r9.json'), os.path.join(R, 'coopen_core_params_48_r10.json')
SHADOW = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3', 'analysis', 'door_coopen_core_bench_3_offset_shadow.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    sh = json.load(open(SHADOW))
    if p['solver_change'].get('offset_hold_on_free') is not True:
        raise SystemExit('r9 solver_change.offset_hold_on_free 不是 True ⇒ 不建')
    p['solver_change'] = dict(p['solver_change'], offset_update_in_open=True, node='dl3_wgmpc_wg2_adm17r4_node.py',
                              rule_r10='保持觸發改為 K3-M 暫停生效（_pause_active，本輪更新之前凍結）；OPEN（free）期間照既有 interval 合格區間規則更新；'
                                       '停止／失效／閘門非 ARMED 仍優先阻擋；暫停起不再學習；觀測器內部 d、_of_d、延遲預推與 QP bias＝α⊙d̂ 每輪同步',
                              rule_r9_replaced='首次切入 free 凍結（r9 rule）由 rule_r10 取代；offset_hold_on_free 保留為保持機制旗標',
                              basis_r10={'source': 'Codex r27：唯一候選＝開門段恢復偏移估計更新',
                                         'shadow': {'path': os.path.relpath(SHADOW, HERE), 'sha256': hashlib.sha256(open(SHADOW, 'rb').read()).hexdigest(),
                                                    'rms_vz_frozen_mm': sh['overall']['rms_vz_frozen_mm'], 'rms_vz_shadow_mm': sh['overall']['rms_vz_shadow_mm'],
                                                    'hit_clip': sh['shadow_d_traj']['hit_clip']},
                                         'limit': '開迴路因果重播；不代表閉迴路，無法排除把接觸負載學成偏差、加重夾持內力'})
    p['schema'] = 'coopen_core_params/7'
    p['derived_from_r9'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, json.dumps(p['solver_change']['basis_r10']['shadow'], ensure_ascii=False))


if __name__ == '__main__':
    main()
