"""低速框與執行端 E2 的相容性判斷：**單一來源**。

求解端的啟動自檢、runner 的讀回比對與離線核對全部呼叫這裡，
避免三處各寫一份而彼此漂移（main4 的成因正是兩端各看自己的設定）。

形式差異是重點：上游的框是**逐軸**（|vx|, |vy| 各自），
執行端 E2 檢查的是**平面範數** hypot(vx, vy)。
兩軸同時到頂時範數為 sqrt(2) x 逐軸值，**只比單軸會漏掉這個情形**。

規格：evaluation/results/specs/coman_low_speed_box_l1.yaml
"""
from __future__ import annotations
import math

SQRT2 = math.sqrt(2.0)


def worst_plane_norm(vmax_base_lin_per_axis: float) -> float:
    """逐軸框能產生的**最大**平面速度（兩軸同時到頂）。"""
    return SQRT2 * float(vmax_base_lin_per_axis)


def e2_compat_violations(base_lin_per_axis, base_ang, arm_max,
                         e2_base_lin, e2_base_ang, e2_arm_rate):
    """回傳違反項的說明串列；空串列 = 相容。

    門檻為非正值者視為「不自檢」而跳過（與各節點參數的 -1.0 約定一致）。
    """
    out = []
    checks = ((e2_base_lin, worst_plane_norm(base_lin_per_axis),
               'base_lin（平面範數；逐軸 x sqrt(2)）'),
              (e2_base_ang, float(base_ang), 'base_ang'),
              (e2_arm_rate, float(arm_max), 'arm_rate'))
    for lim, worst, name in checks:
        lim = float(lim)
        if lim > 0.0 and worst > lim:
            out.append(f'{name}：上游可產生 {worst:.6f} > E2 門檻 {lim:.6f}')
    return out


def tighten_vmax(vmax, per_axis_lin, base_ang, arm, np_mod):
    """把低速框套進 vmax，**只縮不放**。

    用 minimum 而非直接指派：覆寫值若比硬體框寬，硬體框仍然生效
    （速度與加速度是硬體絕對值，不得被放寬）。
    回傳 (新 vmax, 實際生效的覆寫 dict)。
    """
    vm = np_mod.asarray(vmax, dtype=float).copy()
    ov = {}
    for key, val, sl in (('vmax_base_lin', per_axis_lin, slice(0, 2)),
                         ('vmax_base_ang', base_ang, slice(2, 3)),
                         ('vmax_arm', arm, slice(3, None))):
        v = float(val)
        if v > 0.0:
            vm[sl] = np_mod.minimum(vm[sl], v)
            ov[key] = v
    return vm, ov


ADVICE = ('請以 --vmax-base-lin/--vmax-base-ang/--vmax-arm 收緊上游框；'
          '**不要提高 E2 門檻**（政策 wb_wheel_limit_policy_v2.md §7：'
          '現行低速介面界限 lin 0.05／ang 0.2 保持不變）。')
