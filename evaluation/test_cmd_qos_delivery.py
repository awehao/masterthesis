"""F14：命令路徑的 QoS 送達驗證（**不開 Isaac**，純通訊診斷）。

**這不是操作驗收** —— 只回答「以兩端各自實際使用的 QoS，九維命令是否送達、
內容是否一致、回呼是否被呼叫」。

被修的缺陷：執行端以 TRANSIENT_LOCAL 訂閱 `/wb_vel_cmd`，而 arm_vel_adapter
以預設（RELIABLE / VOLATILE）發布 ⇒ DURABILITY 不相容 ⇒ **一筆都收不到**
（實測 main4：求解端發 9 筆，chain9.received = 0）。
"""
from __future__ import annotations
import os, re, sys, time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = '/wb_vel_cmd'
CMDS = [[0.01 * i, -0.02 * i, 0.003 * i] + [0.1 * i + 0.01 * j
                                            for j in range(6)]
        for i in range(1, 10)]          # 9 筆，與 main4 的筆數相同


def endpoint_qos_from_source():
    """從執行端原始碼取它**實際**用來訂閱 /wb_vel_cmd 的 QoS。"""
    s = open(os.path.join(HERE, 'isaac_coman_drawer_sim.py'),
             encoding='utf-8').read()
    i = s.index("'/wb_vel_cmd'")
    seg = s[max(0, i - 900):i + 200]
    name = re.search(r"self\.create_subscription\(Float64MultiArray, '/wb_vel_cmd',\s*\n\s*self\._wb9, (\w+)\)", s)
    qname = name.group(1) if name else None
    dur = ('TRANSIENT_LOCAL' if re.search(
        rf'{qname}\s*=\s*QoSProfile\([^)]*TRANSIENT_LOCAL', seg, re.S)
        else 'VOLATILE')
    return qname, dur


class Sub(Node):
    def __init__(self, qos):
        super().__init__('f14_receiver')
        self.got = []
        self.create_subscription(Float64MultiArray, TOPIC, self._cb, qos)

    def _cb(self, m):
        self.got.append([float(x) for x in m.data])


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:54s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    qname, dur = endpoint_qos_from_source()
    print(f'  執行端訂閱 /wb_vel_cmd 用的 QoS 變數：{qname}（durability {dur}）')
    check('執行端的訂閱 durability 是 VOLATILE（與 adapter 相容）',
          dur == 'VOLATILE',
          '' if dur == 'VOLATILE' else '  **TRANSIENT_LOCAL ⇒ 收不到**')

    rclpy.init()
    try:
        # adapter 的**實際**發布設定：create_publisher(..., OUT, 10) ⇒ 預設 QoS
        pub_node = Node('f14_adapter_like')
        pub = pub_node.create_publisher(Float64MultiArray, TOPIC, 10)

        # 執行端**修好後**的訂閱設定
        good = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                          history=HistoryPolicy.KEEP_LAST)
        sub_ok = Sub(good)
        # **修好前**的設定，作為反例
        bad_q = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_ALL,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sub_old = Sub(bad_q)
        sub_old.get_logger().set_level(50)     # 壓掉預期中的不相容警告

        t0 = time.time()
        while time.time() - t0 < 2.0:          # 等配對
            rclpy.spin_once(sub_ok, timeout_sec=0.05)
            rclpy.spin_once(sub_old, timeout_sec=0.05)

        for c in CMDS:
            m = Float64MultiArray()
            m.data = [float(x) for x in c]
            pub.publish(m)
            time.sleep(0.02)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            rclpy.spin_once(sub_ok, timeout_sec=0.05)
            rclpy.spin_once(sub_old, timeout_sec=0.05)

        check(f'修好後的 QoS：收到 {len(sub_ok.got)}/9 筆',
              len(sub_ok.got) == len(CMDS))
        check('內容逐筆逐分量相同',
              all(len(g) == 9 and all(abs(a - b) < 1e-12 for a, b in zip(g, c))
                  for g, c in zip(sub_ok.got, CMDS)))
        check('**漏洞重現**：舊 QoS（TRANSIENT_LOCAL）收到 0 筆',
              len(sub_old.got) == 0, f'  實收 {len(sub_old.got)}')
        check('回呼確實被呼叫（不是只有計數）',
              bool(sub_ok.got) and isinstance(sub_ok.got[0], list))

        # 命令鏈能接受這些值（結構檢查，不是操作驗收）
        sys.path.insert(0, HERE)
        from wb_cmd_chain_e2 import CmdChainE2
        sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                        'src/ammr_wholebody_mpc'))
        from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as _L
        ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=True,
                        joint_lower=list(_L.lower), joint_upper=list(_L.upper),
                        mode='sync', expect_dof=9)
        n_ok = sum(1 for c in sub_ok.got if ch.receive(c, 0.01))
        check(f'命令鏈接受 {n_ok}/9 筆（結構與模式檢查）', n_ok == len(CMDS),
              f'  最後拒收原因 {ch.last_reject}')
    finally:
        rclpy.try_shutdown()

    print('F14 命令送達測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    print('  **這是通訊診斷，不是操作驗收** —— 只證明 QoS 相容且內容一致。')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
