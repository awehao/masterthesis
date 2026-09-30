"""低速框讀回比對：**實際執行中的節點**印出什麼，而不是 runner 傳出什麼。

main4 的教訓：兩端各自載入自己的設定、誰都沒有核對，於是求解端產出
0.2775 m/s、執行端門檻 0.05 m/s，而整趟沒有一處擋下來。

本檔比對三件事：
  1. 安全節點讀回的 vmax_effective 與 runner 傳出的低速框**逐項相等**
  2. 逐軸框在**兩軸同時到頂**時的平面範數仍在 E2 的範數門檻內
     （這是逐軸框能產生的最大平面速度；只比單軸會漏掉）
  3. 角速度與關節速率各自在 E2 門檻內

失敗即具名退出 1（runner 轉為 exit 67），**不進任務**。
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coman_low_speed_box import (e2_compat_violations,             # noqa: E402
                                 worst_plane_norm)


def check(txt, bl, ba, ar, e2l, e2a, e2r):
    """回傳 (ok, 訊息列)。"""
    m = re.findall(r'wholebody_safety vmax_effective base_lin=([\d.]+) '
                   r'base_ang=([\d.]+) arm_max=([\d.]+)', txt)
    if not m:
        return False, ['**安全節點未印出 vmax_effective ⇒ '
                       '無法確認它是否套用了低速框**（缺少不等於已套用）']
    got = tuple(float(x) for x in m[-1])
    bad = []
    for name, g, want in (('base_lin', got[0], bl),
                          ('base_ang', got[1], ba),
                          ('arm_max', got[2], ar)):
        if abs(g - want) > 1e-9:
            bad.append(f'{name}：安全層讀回 {g:.6f} != runner 傳出 {want:.6f}')
    # **與求解端自檢同一段算術**（單一來源，避免兩處漂移）
    bad += e2_compat_violations(got[0], got[1], got[2], e2l, e2a, e2r)
    if bad:
        return False, ['**低速框讀回比對失敗**'] + ['  ' + b for b in bad]
    return True, [
        f'  安全層讀回 base_lin={got[0]:.6f} base_ang={got[1]:.6f} '
        f'arm_max={got[2]:.6f}，與 runner 逐項一致',
        f'  對 E2：平面範數上界 {worst_plane_norm(got[0]):.6f} <= {e2l}、'
        f'角速度 {got[1]:.6f} <= {e2a}、關節 {got[2]:.6f} <= {e2r}']


def main() -> int:
    log = sys.argv[1]
    bl, ba, ar, e2l, e2a, e2r = (float(x) for x in sys.argv[2:8])
    txt = open(log, encoding='utf-8', errors='replace').read()
    ok, msgs = check(txt, bl, ba, ar, e2l, e2a, e2r)
    for m in msgs:
        print(m, flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
