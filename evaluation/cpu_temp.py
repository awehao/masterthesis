"""CPU 溫度讀取：hwmon 優先，`sensors` 只當備援。

為什麼另寫一支：既有腳本用 `subprocess.run(['sensors'])`，但這台機器沒有裝
lm-sensors，所以那個函式**一直回傳 None**，而中止判斷寫的是
    if c is not None and c >= limit
—— 也就是 92 °C 的熱中止線在先前所有手臂試驗中都沒有生效過。
這不是「沒有超溫」，是「沒有量到」，兩者不能混為一談。

回傳 (溫度 °C 或 None, 來源字串)。來源一定要跟著記錄走，這樣日後看紀錄就知道
那個數字是哪裡來的，或者為什麼是空的。
"""
from __future__ import annotations

import glob
import os

# 依序嘗試；同一個晶片可能有多個感測點，取最大值（Tctl 通常就是最大的那個）
_PREFER = ('k10temp', 'coretemp', 'zenpower', 'acpitz')


def read() -> tuple[float | None, str]:
    best, src = None, 'none'
    for name in _PREFER:
        vals = []
        for h in glob.glob('/sys/class/hwmon/hwmon*'):
            try:
                if open(os.path.join(h, 'name')).read().strip() != name:
                    continue
            except OSError:
                continue
            for f in sorted(glob.glob(os.path.join(h, 'temp*_input'))):
                try:
                    vals.append(float(open(f).read().strip()) / 1000.0)
                except (OSError, ValueError):
                    pass
        if vals:
            return max(vals), f'hwmon:{name}'
    try:
        import re
        import subprocess
        o = subprocess.run(['sensors'], capture_output=True, text=True,
                           timeout=3).stdout
        v = [float(x) for x in re.findall(r'\+(\d+\.\d)°C', o)]
        if v:
            return max(v), 'sensors'
    except Exception:
        pass
    return best, src


if __name__ == '__main__':
    t, s = read()
    print(f'{t if t is not None else "讀不到"}  來源 {s}')
