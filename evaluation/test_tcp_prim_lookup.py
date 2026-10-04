#!/usr/bin/env python3
"""`link_tcp` / `base_footprint` prim 取用的最小核對（離線，不開 Isaac）。

背景：原本 `TCP_PRIM` 以 `globals()` 夾帶到 `loop`，執行上通但相依關係藏起來，
靜態檢查只能報 undefined name。已改為顯式參數。
同一段還有真的缺陷：`startswith(ROBOT)` 會讓 `/World/omni_bot2/link_tcp`
這類相似名稱通過，已改用路徑分段比對。

本檔驗：
  A 取用規則（名稱 + 在機器人子樹內）在反例上正確
  B 原始碼確實改成顯式參數，且兩處 prefix 比對都換掉
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:58s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


class _Prim:
    def __init__(self, path):
        self._p = path

    def GetName(self):
        return self._p.rsplit('/', 1)[-1]

    def GetPath(self):
        return self._p


def _load(src):
    lines = src.split('\n')
    i0 = next(i for i, L in enumerate(lines) if L.startswith('def _under('))
    i1 = i0 + 1
    while i1 < len(lines) and (lines[i1].startswith((' ', '\t'))
                               or not lines[i1].strip()):
        i1 += 1
    ns = {}
    exec('\n'.join(lines[i0:i1]), ns)
    return ns['_under']


def main() -> int:
    src = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
               encoding='utf-8').read()
    under = _load(src)
    ROBOT = '/World/omni_bot'

    stage = [_Prim(p) for p in (
        '/World/omni_bot',
        '/World/omni_bot/base_footprint',
        '/World/omni_bot/link_base',
        '/World/omni_bot/link6/link_eef/link_tcp',
        '/World/omni_bot2/link_tcp',            # **相似名稱的機器人**
        '/World/omni_bot2/base_footprint',
        '/World/drawer/handle/bar',
        '/World/link_tcp',                       # 不在機器人子樹下
    )]

    def pick(name):
        return [str(pr.GetPath()) for pr in stage
                if pr.GetName() == name and under(pr.GetPath(), (ROBOT,))]

    print('A  取用規則（名稱 + 在機器人子樹內）')
    t = pick('link_tcp')
    ck('link_tcp 只取到一個', len(t) == 1, str(t))
    ck('取到的是機器人自己的那個',
       t == ['/World/omni_bot/link6/link_eef/link_tcp'], str(t))
    ck('**相似名稱機器人**的 link_tcp 被排除',
       '/World/omni_bot2/link_tcp' not in t)
    ck('機器人子樹外的 link_tcp 被排除', '/World/link_tcp' not in t)
    f = pick('base_footprint')
    ck('base_footprint 只取到一個', len(f) == 1, str(f))
    ck('base_footprint 取到機器人自己的',
       f == ['/World/omni_bot/base_footprint'], str(f))
    ck('**反例**：舊寫法 startswith 會多取到 omni_bot2 的',
       len([p for p in stage if p.GetName() == 'link_tcp'
            and str(p.GetPath()).startswith(ROBOT)]) == 2,
       '所以 prefix 修正不是美化')

    print('\nB  原始碼：顯式參數、無殘留 prefix 比對')
    # **結構性斷言**，不比對整行字面：loop 的簽名會因為其他顯式參數
    # （例如 drawer_v／DY0）而改變，字面比對會在那時誤報成「保護被移除」。
    # 要核的是「tcp_prim 是具名參數」與「呼叫端真的傳了它」。
    import ast as _ast
    _tree = _ast.parse(src)
    _loop = next((n for n in _tree.body
                  if isinstance(n, _ast.FunctionDef) and n.name == 'loop'), None)
    _params = ([] if _loop is None else
               [a.arg for a in _loop.args.args]
               + [a.arg for a in _loop.args.kwonlyargs])
    ck('loop 以參數接收 tcp_prim',
       'tcp_prim' in _params, f'參數 {_params}')
    _calls = [n for n in _ast.walk(_tree)
              if isinstance(n, _ast.Call)
              and getattr(n.func, 'id', None) == 'loop']
    _passed = any(
        any(getattr(x, 'id', None) == 'tcp_prim' for x in c.args)
        or any(k.arg == 'tcp_prim' for k in c.keywords)
        for c in _calls)
    ck('呼叫端傳入 tcp_prim', _passed, f'{len(_calls)} 處呼叫 loop')
    # **只看可執行碼**：註解裡提到舊寫法不該讓斷言失敗
    _code = '\n'.join(L.split('#')[0] for L in src.split('\n'))
    ck('不再用 globals() 夾帶 TCP_PRIM（只看可執行碼）',
       'TCP_PRIM' not in _code,
       '註解可提及舊寫法，程式碼不得殘留')
    ck('使用處改為區域參數', 'GetLocalToWorldTransform(tcp_prim)' in src)
    ck('兩處 prefix 比對都換成 _under',
       src.count('_under(pr.GetPath(), (ROBOT,))') == 2
       and 'startswith(ROBOT)' not in src)
    ck('找不到 link_tcp 仍中止（保護未被移除）',
       "**找不到 link_tcp prim**" in src and 'return 9' in src)

    print(f'\n{"全部通過" if not FAIL else "**%d 項失敗**：%s" % (len(FAIL), "；".join(FAIL))}'
          '。**這是取用規則的離線核對**；實際 USD stage 上取到哪個 prim 由實跑的 '
          '`[wb] link_tcp prim …` 那一行確認。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
