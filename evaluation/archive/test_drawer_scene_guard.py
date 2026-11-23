#!/usr/bin/env python3
"""抽屜場景守衛的反例核對（離線，不開 Isaac）。

要驗兩件事：
  1 `_under` 是**路徑分段**比對，相似名稱不得通過
    （既有守衛用 startswith，`/World/ground_decoy` 會矇混過關）
  2 solver_drawer 只額外放行 `/World/drawer_unit` 本身及其子樹
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:58s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def _load_under():
    """從 e2 原始碼取出 `_under`，**驗的是實際會跑的那一份**。"""
    src = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
               encoding='utf-8').read()
    lines = src.split('\n')
    i0 = next(i for i, L in enumerate(lines) if L.startswith('def _under('))
    i1 = i0 + 1
    while i1 < len(lines) and (lines[i1].startswith((' ', '\t'))
                               or not lines[i1].strip()):
        i1 += 1
    ns = {}
    exec('\n'.join(lines[i0:i1]), ns)
    return ns['_under'], src


def main() -> int:
    under, src = _load_under()
    OWN = ('/World/omni_bot', '/World/ground')
    DRW = OWN + ('/World/drawer_unit',)

    print('A  路徑分段比對（反例：startswith 會放行的相似名稱）')
    for p in ('/World/omni_bot', '/World/ground'):
        ck(f'放行子樹本身 {p}', under(p, OWN))
    for p in ('/World/omni_bot/base_link', '/World/ground/Plane'):
        ck(f'放行子樹的子節點 {p}', under(p, OWN))
    for p in ('/World/ground_decoy', '/World/groundX',
              '/World/omni_bot2', '/World/omni_botX/link'):
        ok = not under(p, OWN)
        ck(f'**相似名稱被擋** {p}', ok,
           '' if ok else 'startswith 會誤放')
    ck('**反例**：舊寫法 startswith 確實會放行 /World/ground_decoy',
       '/World/ground_decoy'.startswith(OWN),
       '所以這個修正不是美化，是補漏洞')

    print('\nB  solver_drawer 只額外放行 /World/drawer_unit 本身及其子樹')
    ck('抽屜子樹本身通過', under('/World/drawer_unit', DRW))
    for p in ('/World/drawer_unit/cabinet/side_left',
              '/World/drawer_unit/drawer/front_panel',
              '/World/drawer_unit/drawer/handle_bar'):
        ck(f'抽屜子節點通過 {p}', under(p, DRW))
    for p in ('/World/drawer_unit_decoy', '/World/drawer_unitX',
              '/World/drawer/handle', '/World/table'):
        ck(f'**非抽屜被擋** {p}', not under(p, DRW))
    ck('freespace 模式下抽屜**不**放行', not under('/World/drawer_unit', OWN),
       '兩個模式的放行集合不同')

    print('\nC  守衛本體確實改用 _under（不是殘留的 startswith）')
    ck('守衛呼叫 _under', 'not _under(pr.GetPath(), own)' in src)
    ck('守衛不再用 startswith 比對碰撞體路徑',
       'str(pr.GetPath()).startswith(own)' not in src)
    ck('solver_drawer 已在 --mode 的選項內', "'solver_drawer'" in src)
    ck('抽屜子樹是具名常數', "DRAWER_SUBTREE = '/World/drawer_unit'" in src)

    print(f'\n{"全部通過" if not FAIL else "**%d 項失敗**：%s" % (len(FAIL), "；".join(FAIL))}'
          '。這是**守衛判定規則**的離線核對；USD 場景實際建出來由實跑確認。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
