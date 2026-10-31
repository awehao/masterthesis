#!/usr/bin/env python3
"""極薄的 ROS 替身：讓既有節點（scan_obstacle_tracker、scan_safety_shield）的**原始程式**在同一程序內、由模擬時鐘同步執行，不改也不抄演算法。
只提供這兩個節點用到的介面：參數、發布（記下最後一筆）、時鐘、TF 查詢、訊息容器。訂閱不接線，由 bottle_nav_chain 直接呼叫回呼。
必須在 import 節點模組之前呼叫 install()。S1 同程序沒有 rclpy，不與真 ROS 混用。
"""
import sys
import types
from types import SimpleNamespace as NS

OVERRIDES = {}
_NOW = [0.0]


def set_now(t):
    _NOW[0] = float(t)


class _Clock:
    def now(self):
        return NS(nanoseconds=int(round(_NOW[0] * 1e9)))


class Pub:
    def __init__(self, topic):
        self.topic, self.last = topic, None

    def publish(self, m):
        self.last = m


class _Log:
    def info(self, *a, **k):
        pass

    warn = error = debug = warning = info


class Node:
    def __init__(self, name, **k):
        self._p = {}

    def declare_parameter(self, k, v=None):
        self._p[k] = OVERRIDES.get(k, v)
        return NS(value=self._p[k])

    def get_parameter(self, k):
        return NS(value=self._p[k])

    def create_subscription(self, *a, **k):
        return None

    def create_publisher(self, m, topic, q=None, **k):
        return Pub(topic)

    def create_timer(self, *a, **k):
        return None

    def get_logger(self):
        return _Log()

    def get_clock(self):
        return _Clock()


class Msg:
    def __init__(self, **k):
        self.__dict__.update(k)


class Twist(Msg):
    def __init__(self):
        super().__init__(linear=NS(x=0.0, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=0.0))


class Float32MultiArray(Msg):
    def __init__(self):
        super().__init__(data=[])


class LaserScan(Msg):
    def __init__(self):
        super().__init__(header=NS(frame_id='', stamp=NS(sec=0, nanosec=0)), ranges=[], angle_min=0.0, angle_max=0.0, angle_increment=0.0, range_min=0.0, range_max=0.0, time_increment=0.0, scan_time=0.0, intensities=[])


class _Buffer:
    def __init__(self):
        self.tf = None

    def lookup_transform(self, *a, **k):
        if self.tf is None:
            raise RuntimeError('no tf')
        return self.tf


def transform(x, y, yaw):
    import math
    return NS(transform=NS(translation=NS(x=float(x), y=float(y), z=0.0), rotation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))))


def install():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    enum = type('E', (), dict(BEST_EFFORT=1, RELIABLE=2, VOLATILE=1, TRANSIENT_LOCAL=2, KEEP_LAST=1))
    rq = mod('rclpy.qos', QoSProfile=lambda **k: None, ReliabilityPolicy=enum, DurabilityPolicy=enum, HistoryPolicy=enum)
    rn = mod('rclpy.node', Node=Node)
    rd = mod('rclpy.duration', Duration=lambda **k: None)
    rt = mod('rclpy.time', Time=lambda **k: None)
    mod('rclpy', node=rn, qos=rq, duration=rd, time=rt, init=lambda **k: None, spin=lambda *a: None, shutdown=lambda: None)
    mod('tf2_ros', Buffer=_Buffer, TransformListener=lambda *a, **k: None)
    mod('geometry_msgs')
    mod('geometry_msgs.msg', Twist=Twist, TransformStamped=Msg)
    mod('sensor_msgs')
    mod('sensor_msgs.msg', LaserScan=LaserScan)
    mod('std_msgs')
    mod('std_msgs.msg', Float32MultiArray=Float32MultiArray)
    mod('nav_msgs')
    mod('nav_msgs.msg', OccupancyGrid=Msg)
    mod('visualization_msgs')
    mod('visualization_msgs.msg', Marker=Msg, MarkerArray=Msg)
