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
        self.props = []                                              # dict(k, n, u, r, n_rx, delta, late)
        self.n_late, self.n_modified, self.n_unmatched_release, self.lag_steps = 0, 0, 0, []
        self.consec_late, self.consec_mod = 0, 0

    def propose(self, n, u):
        self.props.append(dict(k=len(self.props), n=int(n), u=np.asarray(u, float).copy(), r=None, n_rx=None, delta=None, late=False))

    def on_release(self, n_rx, v):
        v = np.asarray(v, float)
        cand = [p for p in self.props[-6:] if p['r'] is None and p['n'] <= n_rx]
        if not cand:
            self.n_unmatched_release += 1
            return None
        d = [float(np.abs(v - p['u']).max()) for p in cand]
        j = int(np.argmin(d))
        p = cand[j] if d[j] <= self.tol else cand[0]                 # 值相符者優先；都不符 ⇒ 依序配最舊的（視為被安全鏈改動）
        p['r'], p['n_rx'], p['delta'] = v.copy(), int(n_rx), v - p['u']
        p['late'] = bool(n_rx > p['n'] + self.D)
        self.lag_steps.append(int(n_rx - p['n']))
        return p

    def _delta_k(self, i, n):
        """第 i 拍在步 n 可用的 Δ：已放行且不晚於生效時刻 ⇒ 它自己的；否則沿用前一拍的（晚到）。"""
        if i < 0:
            return np.zeros(3)
        p = self.props[i]
        if p['delta'] is not None and p['n_rx'] <= max(n, p['n'] + self.D):
            return p['delta']
        return self._delta_k(i - 1, n)

    def correction(self, n):
        """步 n 要加在控制器預期底盤命令上的修正量（本體座標 3 維）。"""
        i = None
        for j in range(len(self.props) - 1, -1, -1):
            if self.props[j]['n'] + self.D <= n:
                i = j
                break
        if i is None:
            return np.zeros(3)
        jr = n - (self.props[i]['n'] + self.D)
        a, b = self._delta_k(i - 1, n), self._delta_k(i, n)
        w = min((jr + 1) / self.NPH, 1.0)
        return a + w * (b - a)

    def tick_status(self, n):
        """在控制拍邊界呼叫：統計剛到生效時刻的那一拍。回 dict(late, modified, consec_late, consec_mod)。"""
        out = dict(late=False, modified=False)
        for p in self.props:
            if p.get('counted') or p['n'] + self.D > n:
                continue
            p['counted'] = True
            late = p['r'] is None or p['late']
            mod = p['delta'] is not None and float(np.abs(p['delta']).max()) > self.tol
            self.n_late += int(late)
            self.n_modified += int(mod)
            self.consec_late = self.consec_late + 1 if late else 0
            self.consec_mod = self.consec_mod + 1 if mod else 0
            out = dict(late=late, modified=mod, delta=None if p['delta'] is None else [float(x) for x in p['delta']])
        out.update(consec_late=self.consec_late, consec_mod=self.consec_mod)
        return out

    def report(self):
        lg = np.array(self.lag_steps, float)
        dl = np.array([np.abs(p['delta']).max() for p in self.props if p['delta'] is not None], float)
        return dict(n_proposed=len(self.props), n_released_matched=int(len(lg)), n_late=self.n_late, n_modified=self.n_modified, n_unmatched_release=self.n_unmatched_release,
                    lag_steps=None if not len(lg) else dict(min=int(lg.min()), p50=float(np.median(lg)), max=int(lg.max())), delta_abs_max=None if not len(dl) else float(dl.max()), d_steps=self.D, tol=self.tol)


if __name__ == '__main__':                                           # 自我核對
    A = ChainAlign()
    for k in range(10):
        A.propose(5 * k, [0.05, 0, 0]); A.on_release(5 * k + 5, [0.05, 0, 0]); A.tick_status(5 * k)
    assert np.abs(A.correction(40)).max() == 0 and A.report()['n_late'] == 0
    B = ChainAlign()
    B.propose(0, [0.05, 0, 0]); B.on_release(5, [0.03, 0, 0])
    assert abs(B.correction(8)[0] + 0.004) < 1e-12 and abs(B.correction(12)[0] + 0.02) < 1e-12
    B.propose(5, [0.05, 0, 0]); B.on_release(15, [0.05, 0, 0])      # 晚到（15 > 5＋8）⇒ 步 13 沿用前一拍 Δ
    assert abs(B.correction(13)[0] + 0.02) < 1e-12 and B.props[1]['late']
    B.tick_status(20)
    assert B.n_late == 1 and B.n_modified == 1
    print('ChainAlign self-check ok', A.report(), B.report())
