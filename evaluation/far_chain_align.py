#!/usr/bin/env python3
"""抓持／開門段的底盤命令經安全鏈（選擇器→平滑器→Shield→GUARD）的放行流核對（Codex 20261117；A2(b) 保守版）。純計算，不依賴 ROS。
控制器每控制拍在步 n_k 提出底盤命令 u_k（/cmd_vel_grasp）；它的延遲模型＝D 步後生效。安全鏈放行（/cmd_vel；GUARD 唯一出口）在步 n_rx 收到。
執行器套用的是控制器核過的管線輸出，不做任何事後修正；本類別核對「整條放行流」與提案序列一致，否則在生效前拒絕：
  1. 每一筆放行都必須等於「目前最新提案 u_j」或「前一筆 u_{j−1}」（安全鏈一個節拍的延遲；j＝n_j ≤ n_rx 的最新提案）；
     都不相符 ⇒ 違規（例如 Shield 煞車）——立即回報，不等到生效步。
     例外：接手後、第一筆提案被確認之前（最多 D 步）鏈內還有前擁有者的命令（乾跑 m34 實測：值逐拍不同），只計數不比對；第一筆提案仍須在生效步前被確認。
  2. 放行流單調：一旦出現等於 u_k 的放行，之後不得再回到只等於更舊提案的值。
  3. 每筆提案在其生效步（n_k＋D）之前必須已被某筆放行確認；否則 ⇒ 缺失（例如安全鏈扣住舊值不放）。
  等值的相鄰提案彼此可互換（執行的命令相同）。容許差 tol 只涵蓋訊息傳遞的浮點舍入（預設 1e-6）；超過即視為被修改。
限制照實：若安全鏈的修改恰好等於前一筆提案值，會被當成延遲（之後該拍提案仍須在生效前被確認，否則缺失）。"""
import numpy as np


class ChainAlign:
    def __init__(self, d_steps=8, nph=5, tol=1e-6, pre_release=None):
        self.D, self.NPH, self.tol = int(d_steps), int(nph), float(tol)
        self.props = []                                              # dict(k, n, u, confirmed, n_conf)
        self.pre = None if pre_release is None else np.asarray(pre_release, float).copy()
        self.releases, self.viol, self.last_idx = [], None, -1
        self.n_ok, self.n_missing, self.n_differs, self.lag_steps, self.delta_max, self.n_pre_match = 0, 0, 0, [], 0.0, 0

    def propose(self, n, u):
        self.props.append(dict(k=len(self.props), n=int(n), u=np.asarray(u, float).copy(), confirmed=False, n_conf=None))

    def on_release(self, n_rx, v):
        v = np.asarray(v, float)
        self.releases.append((int(n_rx), v.copy()))
        if not self.props or self.viol is not None:
            return
        j = max((p['k'] for p in self.props if p['n'] <= n_rx), default=None)
        if j is None:
            return
        cands = [i for i in (j, j - 1) if i >= max(self.last_idx, 0)]
        hits = [i for i in cands if float(np.abs(v - self.props[i]['u']).max()) <= self.tol]
        if hits:
            for i in hits:
                p = self.props[i]
                if not p['confirmed']:
                    p['confirmed'], p['n_conf'] = True, int(n_rx)
                    self.lag_steps.append(int(n_rx - p['n']))
                self.delta_max = max(self.delta_max, float(np.abs(v - p['u']).max()))
            self.last_idx = max(self.last_idx, min(hits))
            return
        if self.last_idx < 0 and n_rx <= self.props[0]['n'] + self.D:
            self.n_pre_match += 1                                    # 接手後、第一筆提案被確認前（最多 D 步）：鏈內還有前擁有者的命令，值不可由本端驗證 ⇒ 只計數。
            return                                                   # 這些放行不會被執行（執行的是控制器自己的管線）；第一筆提案仍須在生效步前被確認，否則缺失
        near = min(float(np.abs(v - self.props[i]['u']).max()) for i in (cands or [j]))
        self.n_differs += 1
        self.viol = dict(why='release_differs', n_rx=int(n_rx), released=[float(x) for x in v], latest_proposal=dict(k=j, n=self.props[j]['n'], u=[float(x) for x in self.props[j]['u']]), min_diff_to_candidates=near, last_confirmed_idx=self.last_idx)

    def check(self, n):
        """每個物理步呼叫（套用命令之前）。有違規 ⇒ 立刻不通過；本步有提案生效而尚未被放行確認 ⇒ 不通過。"""
        if self.viol is not None:
            return dict(ok=False, **self.viol)
        for p in self.props[-40:]:
            if p['n'] + self.D == n:
                if not (p['confirmed'] and p['n_conf'] <= n):
                    self.n_missing += 1
                    self.viol = dict(why='release_missing', k=p['k'], n_prop=p['n'], u=[float(x) for x in p['u']], last_releases=[[r[0], [float(x) for x in r[1]]] for r in self.releases[-3:]])
                    return dict(ok=False, **self.viol)
                self.n_ok += 1
        return dict(ok=True)

    def report(self):
        lg = np.array(self.lag_steps, float)
        return dict(policy='整條放行流須與提案序列一致（每筆放行＝最新或前一筆提案；單調；每筆提案生效前須被確認）；違規或缺失 ⇒ 生效前拒絕', n_proposed=len(self.props), n_effect_checked_ok=self.n_ok, n_missing=self.n_missing, n_differs=self.n_differs,
                    n_releases=len(self.releases), n_pre_owner_releases=self.n_pre_match, lag_steps=None if not len(lg) else dict(min=int(lg.min()), p50=float(np.median(lg)), max=int(lg.max())), delta_abs_max=float(self.delta_max), d_steps=self.D, tol=self.tol, violation=self.viol)


def best_phase(rx_steps, nph=5, back=4):
    """提案相位＝最近 /cmd_vel 放行到達步 mod nph 的眾數 r（＝GUARD 取樣那一步）：提案在 GUARD 剛取樣完送出，下一次取樣在 4–5 步後。
    依據 far_door_8 各層紀錄：平滑器在提案到達後 0–3 步才更新輸出（逐拍漂移），GUARD 以固定相位 20 Hz 取樣 Shield 輸出；
    舊規則 (r＋2) 使提案到 GUARD 下一次取樣只剩 2 步，平滑器慢到 3 步時 GUARD 取到舊值、該提案被跳過（7、8 兩趟）。
    這是排程修正，不是放行保證；真正放行仍由 ChainAlign 逐筆核對。樣本不足回 None。"""
    xs = [int(x) % nph for x in rx_steps[-12:]]
    if len(xs) < 6:
        return None
    return max(set(xs), key=xs.count)


if __name__ == '__main__':                                           # 自我核對（含 Codex 20261117 反例）
    A = ChainAlign(pre_release=[0.05, 0, 0])
    for k in range(10):
        A.propose(5 * k, [0.05, 0, 0]); A.on_release(5 * k + 4, [0.05, 0, 0])
    assert all(A.check(n)['ok'] for n in range(60)) and A.report()['n_effect_checked_ok'] == 10
    B = ChainAlign()                                                 # 反例：步 0 提案 0.05、步 4 放行 0.05、步 7 Shield 改成 0 ⇒ 步 7 就拒絕
    B.propose(0, [0.05, 0, 0]); B.on_release(4, [0.05, 0, 0]); assert B.check(6)['ok']
    B.on_release(7, [0.0, 0, 0]); assert not B.check(7)['ok'] and B.check(7)['why'] == 'release_differs'
    C = ChainAlign(); C.propose(0, [0.05, 0, 0]); C.on_release(9, [0.05, 0, 0])
    assert C.check(8)['why'] == 'release_missing'                     # 晚到
    Dd = ChainAlign(pre_release=[0.05, 0, 0]); Dd.propose(0, [0.04, 0, 0]); Dd.on_release(2, [0.05, 0, 0])   # 接手初期前擁有者的值：允許，但不算確認
    assert Dd.check(7)['ok'] and Dd.check(8)['why'] == 'release_missing'
    E = ChainAlign()                                                 # 減速：延遲 7 步、下一筆提前到 ⇒ 各自確認
    E.propose(0, [0.05, 0, 0]); E.propose(5, [0.025, 0, 0]); E.on_release(7, [0.05, 0, 0]); E.on_release(8, [0.025, 0, 0])
    assert E.check(8)['ok'] and E.check(13)['ok']
    F = ChainAlign()                                                 # 非單調：已放行新值後又回舊值 ⇒ 拒絕
    F.propose(0, [0.05, 0, 0]); F.propose(5, [0.025, 0, 0]); F.propose(10, [0.0, 0, 0]); F.on_release(4, [0.05, 0, 0]); F.on_release(9, [0.025, 0, 0]); F.on_release(12, [0.05, 0, 0])
    assert not F.check(12)['ok']
    H = ChainAlign()                                                 # 限速：放行 0.049 對提案 0.05（差 1e-3，超過舍入容許）⇒ 拒絕
    H.propose(0, [0.05, 0, 0]); H.on_release(4, [0.049, 0, 0]); assert H.check(7)['ok'] and H.check(8)['why'] == 'release_missing'   # 首筆確認前只計數；生效步仍未確認 ⇒ 拒絕
    H2 = ChainAlign(); H2.propose(0, [0.05, 0, 0]); H2.on_release(4, [0.05, 0, 0]); H2.propose(5, [0.05, 0, 0]); H2.on_release(9, [0.049, 0, 0]); assert not H2.check(9)['ok']   # 已確認後的限速 ⇒ 立即拒絕
    assert best_phase([1, 6, 11, 16, 21, 27, 31, 36]) == 1 and best_phase([2, 7, 12, 17, 22, 27]) == 2
    print('ChainAlign self-check ok', A.report()['n_effect_checked_ok'])
