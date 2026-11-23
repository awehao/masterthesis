"""底盤模式守衛：兩種模式各有明確期望，相反配置必須被拒（離線，不開 Isaac）。

被修的缺陷：守衛寫死「恰好 1 個 world→根固定關節」，是為固定底座版寫的；
加上 `--free-base` 後沒跟著改 ⇒ **開放底盤模式每次都在啟動就中止**
（29 趟歷史紀錄全是 importer_fix_base，開放底盤路徑從未真正執行過）。

本檔用**與執行端同一段判定邏輯**（由原始碼抽出的期望值規則）核對四種組合，
不需要 USD 或 Isaac。
"""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'isaac_coman_drawer_sim.py')


def expected_from_source():
    """從執行端原始碼取出期望值規則，確認它**真的是模式相依**。"""
    b = open(SRC, encoding='utf-8').read()
    m = re.search(r'_n_expect\s*=\s*(\d+)\s*if\s*a\.free_base\s*else\s*(\d+)', b)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def guard(free_base, n_root_fixed, n_expect_rule):
    """重現守衛：不符期望即中止（回傳 None 表示通過，否則回傳 return code）。"""
    n_expect = n_expect_rule[0] if free_base else n_expect_rule[1]
    return None if n_root_fixed == n_expect else 10


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    rule = expected_from_source()
    check('執行端的期望值規則是模式相依（不是寫死 1）', rule is not None,
          f'  free_base→{rule[0]}、fix_base→{rule[1]}' if rule else '  **找不到規則**')
    if rule is None:
        print('底盤模式守衛測試：**無法取得規則，視為失敗**')
        return 1

    # 四種組合
    cases = [
        ('free_base 且 0 個固定關節', True, 0, None, '正確組合，應通過'),
        ('free_base 卻有 1 個固定關節', True, 1, 10,
         '**相反配置**：底盤其實被釘住，命令鏈與判定卻以為它自由 ⇒ 必須拒'),
        ('固定底座且 1 個固定關節', False, 1, None, '正確組合，應通過'),
        ('固定底座卻 0 個固定關節', False, 0, 10,
         '**相反配置**：沒有外部固定支撐 ⇒ 必須拒'),
        ('固定底座卻 2 個固定關節', False, 2, 10, '約束重複 ⇒ 必須拒'),
        ('free_base 卻有 2 個固定關節', True, 2, 10, '必須拒'),
    ]
    for name, fb, n, want, why in cases:
        got = guard(fb, n, rule)
        check(name, got == want,
              f'  {"通過" if got is None else f"中止 rc={got}"} —— {why}')

    # 舊守衛（寫死 1）會把正確的 free_base 組合擋掉 —— 漏洞重現
    old_guard = lambda n: None if n == 1 else 10
    check('**漏洞重現**：舊守衛把 free_base 的正確組合擋掉',
          old_guard(0) == 10, '  舊守衛 rc=10（這就是首趟中止的原因）')
    check('舊守衛也擋不住「free_base 卻被釘住」', old_guard(1) is None,
          '  舊守衛會**放行**這個危險組合')

    # 期望值不得被放寬成「不檢查」
    src = open(SRC, encoding='utf-8').read()
    check('守衛仍會 return 10（保護未移除）',
          re.search(r'_n_expect.*\n(?:.*\n){0,12}?\s*return 10', src) is not None)
    check('中止訊息帶出模式與實際數量（可追溯）',
          '實際' in src and '模式' in src)

    print('底盤模式守衛測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
