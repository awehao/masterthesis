#!/usr/bin/env python3
"""核心段登錄檔 r11：r10＋executor_change.pause_finger_max_effort 20 → 5（Codex r31 §2：暫停期維持 5 N 的單一控制候選；只限目前這個抓法與配置）。r10 不覆寫。
  暫停上限登錄為 5 N＝原值（original_effort 5）：不做 20→5 的卸載流程；執行端照既有 W1e 流程在暫停切換步 set 5 並讀回、之後每秒讀回（不以 0 關閉驗證）。
  閉爪目標、增益、摩擦、偏移凍結時點、10 s 觀察與所有保護門檻不變。依據與限制見 basis_r11。

    python3 evaluation/door_coopen_core_params_r11.py   # 寫 results/door/coopen_core_params_48_r11.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r10.json'), os.path.join(R, 'coopen_core_params_48_r11.json')
EVID = [os.path.join(HERE, 'runs', 'door_coopen_core_bench_5', 'analysis', f) for f in
        ('door_coopen_core_bench_5_pause_creep.json', 'door_coopen_core_bench_5_pause_osc.json')]


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    ec = p['executor_change']
    if ec.get('pause_finger_max_effort') != 20.0 or ec.get('original_effort') != 5.0:
        raise SystemExit('r10 executor_change 不是 20／5 ⇒ 不建')
    p['executor_change'] = dict(ec, pause_finger_max_effort=5.0, pause_finger_max_effort_r10=20.0,
                                rule_r11='暫停上限登錄為 5 N（＝原值）；暫停切換步照既有流程 set＋讀回、每秒讀回；不做卸載流程；不以 0 關閉驗證',
                                basis_r11={'source': 'Codex r31 §2：暫停期維持 5 N 的單一控制候選（只限目前抓法與配置；不撤銷其他抓法的 W1e 結論）',
                                           'facts': '本抓法在 5 N 上限下完成 HOLD、長段 OPEN 與 SETTLE；bench_5 切 20 N 後承載與波動改變；STOPPED 後設定點固定但實測仍動',
                                           'not_proven': '尚未證明 20 N 是根因',
                                           'limit': '5→5 時讀回只能證明數值等於 5，無法由讀回區分 set 呼叫是否生效',
                                           'evidence': [{'path': os.path.relpath(f, HERE), 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest()} for f in EVID]})
    p['schema'] = 'coopen_core_params/8'
    p['derived_from_r10'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, 'pause_finger_max_effort', p['executor_change']['pause_finger_max_effort'])


if __name__ == '__main__':
    main()
