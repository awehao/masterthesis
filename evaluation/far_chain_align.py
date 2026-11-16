#!/usr/bin/env python3
"""抓持／開門段的底盤命令經安全鏈（選擇器→平滑器→Shield→GUARD）後的延遲對齊（Codex 20261116 A2(b)）。純計算，不依賴 ROS。
控制器每控制拍（NPH 物理步）在步 n_k 提出底盤命令 u_k；它的延遲模型＝D_STEPS 步後生效，並在 NPH 步內由上一拍命令斜坡到本拍命令。
安全鏈放行 r_k（/cmd_vel；GUARD 是唯一出口）在步 n_rx 收到。只要 n_rx ≤ n_k＋D_STEPS，放行值就照控制器的生效時刻套用：
    實際套用（步 n）＝控制器預期值（它自己的管線＋斜坡輸出）＋ramp(Δ_{k−1}, Δ_k)，Δ_k＝r_k−u_k
  ⇒ 放行值等於提案時，與控制器模型逐步相同；被安全鏈改動時，套用的是放行值（同樣的斜坡）。
晚到（n_rx > n_k＋D_STEPS）：該拍沿用 Δ_{k−1} 並計數。改動（|Δ_k| > tol）與晚到都回報，由呼叫端依政策停止。"""
import numpy as np


class ChainAlign:
    def __init__(self, d_steps=8, nph=5, tol=1e-3):
        self.D, self.NPH, self.tol = int(d_steps), int(nph), float(tol)
        self.props = []                                              # dict(k, n, u)
        self.n_late, self.n_modified, self.n_unmatched_release, self.lag_steps = 0, 0, 0, []
        self.consec_late, self.consec_mod = 0, 0
        self.releases, self.n_ok, self.delta_max, self.ptr = [], 0, 0.0, 0

    def propose(self, n, u):
        self.props.append(dict(k=len(self.props), n=int(n), u=np.asarray(u, float).copy(), r=None, n_rx=None, delta=None, late=False))

    def on_release(self, n_rx, v):
        self.releases.append((int(n_rx), np.asarray(v, float).copy()))

    def check(self, n):
        """保守版（Codex 20261117）：放行與已核命令逐步一致才執行。在第 i 筆提案的生效步（n＝n_i＋D）呼叫：
        須有放行在 (n_i, n_i＋D] 內收到、且依放行順序有一筆的值與提案一致（≤tol）。缺失、過期、不一致 ⇒ ok=False（呼叫端在它生效前拒絕）。
        值一致即表示實際執行的命令與控制器核過的管線逐步相同（重複命令彼此等值，不需區分）。非生效步回 ok=True。"""
        for p in self.props[-40:]:
            if p['n'] + self.D != n:
                continue
            rs = [(j, r) for j, r in enumerate(self.releases) if j >= self.ptr and p['n'] < r[0] <= n]
            if not rs:
                self.n_late += 1
                return dict(ok=False, why='release_missing', k=p['k'], n_prop=p['n'])
            hit = next(((j, r) for j, r in rs if float(np.abs(r[1] - p['u']).max()) <= self.tol), None)   # 依放行順序找與本提案一致的那一筆（放行流單調前進；等值的重複命令彼此可互換）
            if hit is None:
                self.n_modified += 1
                return dict(ok=False, why='release_differs', k=p['k'], n_prop=p['n'], proposed=[float(x) for x in p['u']], released_in_window=[[float(x) for x in r[1]] for _, r in rs][-3:])
            self.ptr = hit[0] + 1
            self.lag_steps.append(int(hit[1][0] - p['n']))
            self.delta_max = max(self.delta_max, float(np.abs(hit[1][1] - p['u']).max()))
            p['ok'] = True
            self.n_ok += 1
        return dict(ok=True)

    def report(self):
        lg = np.array(self.lag_steps, float)
        return dict(policy='放行與提案逐步一致才執行；缺失／不一致在生效前拒絕', n_proposed=len(self.props), n_checked_ok=self.n_ok, n_missing=self.n_late, n_differs=self.n_modified,
                    lag_steps=None if not len(lg) else dict(min=int(lg.min()), p50=float(np.median(lg)), max=int(lg.max())), delta_abs_max=float(self.delta_max), d_steps=self.D, tol=self.tol)

def best_phase(rx_steps, nph=5, back=4):
    """由最近的放行到達步推安全鏈的節拍相位 r（到達步 mod nph 的眾數），回建議的提案相位＝(r−(nph+1)) mod nph……
    使名目延遲約 nph＋1 步（落在 0～D 的中間，兩側各留約 2 步）。樣本不足回 None。"""
    xs = [int(x) % nph for x in rx_steps[-12:]]
    if len(xs) < 6:
        return None
    r = max(set(xs), key=xs.count)
    return (r - (nph + 1)) % nph


if __name__ == '__main__':                                           # 自我核對
    A = ChainAlign()
    for k in range(10):
        A.propose(5 * k, [0.05, 0, 0]); A.on_release(5 * k + 4, [0.05, 0, 0])
    assert all(A.check(n)['ok'] for n in range(60)) and A.report()['n_checked_ok'] == 10
    B = ChainAlign(); B.propose(0, [0.05, 0, 0]); B.on_release(4, [0.03, 0, 0])
    assert B.check(7)['ok'] and B.check(8)['why'] == 'release_differs'
    C = ChainAlign(); C.propose(0, [0.05, 0, 0]); C.on_release(9, [0.05, 0, 0])
    assert C.check(8)['why'] == 'release_missing'
    Dd = ChainAlign(); Dd.on_release(0, [0.05, 0, 0]); Dd.propose(0, [0.05, 0, 0])      # 提案前的舊放行不算
    assert Dd.check(8)['why'] == 'release_missing'
    E = ChainAlign()                                                 # 延遲 8 步且下一筆提前到：仍依序配到自己的那一筆
    E.propose(0, [0.05, 0, 0]); E.propose(5, [0.025, 0, 0]); E.on_release(7, [0.05, 0, 0]); E.on_release(8, [0.025, 0, 0])
    assert E.check(8)['ok'] and E.check(13)['ok']
    print('ChainAlign self-check ok', A.report())
