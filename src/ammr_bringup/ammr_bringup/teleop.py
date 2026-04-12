#!/usr/bin/env python3
"""
按住才動的 teleop node。
  W/↑ 前進   S/↓ 後退
  A/← 左轉   D/→ 右轉
  放開立刻停止
"""
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard

LINEAR_SPEED  = 0.3   # m/s
ANGULAR_SPEED = 1.0   # rad/s

KEYS_FORWARD  = {keyboard.Key.up,    keyboard.KeyCode.from_char('w')}
KEYS_BACKWARD = {keyboard.Key.down,  keyboard.KeyCode.from_char('s')}
KEYS_LEFT     = {keyboard.Key.left,  keyboard.KeyCode.from_char('a')}
KEYS_RIGHT    = {keyboard.Key.right, keyboard.KeyCode.from_char('d')}


class Teleop(Node):
    def __init__(self):
        super().__init__('ammr_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pressed = set()
        self.timer = self.create_timer(0.05, self.publish_cmd)  # 20Hz

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        listener.daemon = True
        listener.start()

        self.get_logger().info(
            '\n=== AMMR Teleop ===\n'
            '  W/↑  前進    S/↓  後退\n'
            '  A/←  左轉    D/→  右轉\n'
            '  Ctrl+C 離開\n'
        )

    def on_press(self, key):
        self.pressed.add(key)

    def on_release(self, key):
        self.pressed.discard(key)

    def publish_cmd(self):
        msg = Twist()
        if self.pressed & KEYS_FORWARD:
            msg.linear.x = LINEAR_SPEED
        elif self.pressed & KEYS_BACKWARD:
            msg.linear.x = -LINEAR_SPEED

        if self.pressed & KEYS_LEFT:
            msg.angular.z = ANGULAR_SPEED
        elif self.pressed & KEYS_RIGHT:
            msg.angular.z = -ANGULAR_SPEED

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Teleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())  # 停止
        rclpy.shutdown()
