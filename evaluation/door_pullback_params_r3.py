#!/usr/bin/env python3
"""往後拉開 48° 登錄 r3：r2 只改 executor_change.pause_finger_max_effort 20 → 5 N（Codex 20261028 pb1_3 回覆選 (a)；＝coopen r11／bench1 鏈原值）。
依據：door_pullback_pb1_3 開門 48.21°、暫停成功；mode 2 張爪後門角增加、橫桿朝掌心接近（7.58→5.72 mm），預測淨空不足 ⇒ 中止。
接觸釋放造成門動是候選解釋，尚未證明衝量來源；5 N 是致動器輸出上限，不保證接觸力為 5 N。其餘鍵值逐字同 r2。

    python3 evaluation/door_pullback_params_r3.py   # 寫 results/door/pullback_params_48_r3.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'pullback_params_48_r2.json'), os.path.join(R, 'pullback_params_48_r3.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    p['executor_change'] = dict(p['executor_change'], pause_finger_max_effort=5.0,
                                rule_r3='暫停期手指驅動輸出上限 20 → 5 N（＝原上限；暫停切換時不再提高）；5 N 為致動器輸出上限，非保證接觸力',
                                basis_r3='door_pullback_pb1_3：張爪後門角增加、橫桿朝掌心接近，預測淨空不足中止（衝量來源未證）')
    p['derived_from_r2'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, p['executor_change']['pause_finger_max_effort'], p['executor_change']['original_effort'])


if __name__ == '__main__':
    main()
