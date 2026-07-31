"""Bring a trial to the starting line, waiting on conditions instead of clocks.

The shell version cost about 87 s per trial against a ~155 s trial: a flat
`sleep 32` sized for the worst case, four separate `ros2 topic` invocations that
each pay ~2-3 s of rclpy startup, a subscriber-count poll that spawned a fresh
process every second, and a goal published five times at 1 Hz whether or not the
first one landed.

All of that is one rclpy process here, and every wait ends when the thing it is
waiting for actually happens:

  1. /clock advances                  -> gz is stepping
  2. /amcl_pose arrives               -> localisation is publishing
  3. publish /initialpose, wait for the next /amcl_pose near the origin
  4. publish /goal_pose, and REPUBLISH until /plan arrives

Step 4 is the one that mattered for correctness as well as speed: a single
VOLATILE publish is easily missed by goal_to_plan_relay before its subscription
matches, which used to show up as plan_requests=0 and a robot that never moved.
The old fix was to publish five times and hope; this waits for the plan and
stops as soon as it has one.

The old script also gated on `/goal_pose` having >= 2 subscribers, polled once a
second for 30 s. That gate never actually passed -- it burned its full 30 s
every trial and then published anyway. Waiting for the PLAN is the real
handshake, and it is self-validating, so the subscriber count is gone.

Exits non-zero if any stage times out, so the caller can skip the trial rather
than record a run that was never going to move.

    python3 evaluation/trial_start.py --goal-x 17 --goal-y 17
"""
import argparse
import random
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy)

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock


LATCH = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                   reliability=QoSReliabilityPolicy.RELIABLE,
                   durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class Starter(Node):
    def __init__(self, goal_x, goal_y):
        super().__init__('trial_start')
        self.goal = (float(goal_x), float(goal_y))
        self.clock_t = None
        self.amcl = None
        self.plan = False
        self.create_subscription(Clock, '/clock', self._on_clock, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._on_amcl, 10)
        self.create_subscription(Path, '/plan', self._on_plan, 10)
        # TRANSIENT_LOCAL so a subscriber that matches late still receives the
        # goal; compatible with the VOLATILE subscribers already out there.
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', LATCH)
        self.init_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

    # NB: not `_clock`/`_logger`/etc -- rclpy.node.Node sets instance attributes
    # of those names in __init__, which shadow same-named methods and make the
    # subscription receive the node's Clock object instead of a callback.
    def _on_clock(self, m):
        self.clock_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _on_amcl(self, m):
        self.amcl = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _on_plan(self, m):
        if len(m.poses) > 1:
            self.plan = True

    def wait(self, cond, timeout, label):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cond():
                print(f'  {label}: {time.time() - t0:.1f} s', flush=True)
                return True
        print(f'  {label}: TIMEOUT after {timeout:.0f} s', flush=True)
        return False

    def goal_msg(self):
        g = PoseStamped()
        g.header.frame_id = 'map'
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = self.goal
        g.pose.orientation.w = 1.0
        return g


def wait_before_goal(n, delay_min, delay_max, seed):
    """Hold the goal back by a seeded random delay.

    The obstacles run on their own from the moment their driver starts, so this
    shifts WHERE they are when the robot sets off. Drawing it from the trial
    seed makes that offset reproducible and recorded, rather than being
    whatever bring-up happened to take that run.
    """
    d = random.Random(seed).uniform(delay_min, delay_max)
    print(f'HOLDING GOAL {d:.2f} s (seed {seed})', flush=True)
    end = time.time() + d
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=min(0.05, max(0.0, end - time.time())))
    return d


def publish_goal(n, left, t0):
    """Publish the goal, republishing until a plan comes back.

    One publish is usually enough; the retry exists because the relay's
    subscription can match after the first message goes out, which used to show
    up as plan_requests=0 and a robot that never moved.
    """
    for attempt in range(1, 11):
        n.goal_pub.publish(n.goal_msg())
        if n.wait(lambda: n.plan, min(2.0, left()),
                  f'/plan received (goal publish #{attempt})'):
            print(f'GOAL ACCEPTED in {time.time() - t0:.1f} s', flush=True)
            return 0
        if left() <= 5.0:
            break
    print('  no /plan after 10 goal publishes', flush=True)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-x', type=float, required=True)
    ap.add_argument('--goal-y', type=float, required=True)
    ap.add_argument('--start-x', type=float, default=0.0)
    ap.add_argument('--start-y', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=90.0,
                    help='cap for this phase')
    # Recording has to start after the stack is up but BEFORE the robot moves,
    # so the caller runs `prepare`, starts the recorder, then runs `goal`.
    ap.add_argument('--phase', choices=('prepare', 'goal', 'all'), default='all')
    # The goal is held back by a delay drawn from --seed, which shifts where
    # the obstacles are when the robot starts. Set --goal-delay-max 0 for none.
    ap.add_argument('--goal-delay-min', type=float, default=1.0)
    ap.add_argument('--goal-delay-max', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rclpy.init()
    n = Starter(a.goal_x, a.goal_y)
    t0 = time.time()
    left = lambda: max(5.0, a.timeout - (time.time() - t0))
    try:
        if a.phase == 'goal':
            wait_before_goal(n, a.goal_delay_min, a.goal_delay_max, a.seed)
            return publish_goal(n, left, t0)

        if not n.wait(lambda: n.clock_t is not None and n.clock_t > 1.0,
                      left(), 'gz clock stepping'):
            return 1
        if not n.wait(lambda: n.amcl is not None, left(), '/amcl_pose alive'):
            return 1

        # Reset localisation, then wait for AMCL to report the reset pose rather
        # than sleeping a fixed 3 s and hoping it converged.
        p = PoseWithCovarianceStamped()
        p.header.frame_id = 'map'
        p.pose.pose.position.x = a.start_x
        p.pose.pose.position.y = a.start_y
        p.pose.pose.orientation.w = 1.0
        n.amcl = None
        for _ in range(3):
            n.init_pub.publish(p)
            if n.wait(lambda: n.amcl is not None and
                      abs(n.amcl[0] - a.start_x) < 0.5 and
                      abs(n.amcl[1] - a.start_y) < 0.5,
                      min(6.0, left()), 'AMCL reset acknowledged'):
                break
        else:
            print('  AMCL never reported the reset pose', flush=True)
            return 1

        if a.phase == 'prepare':
            print(f'PREPARED in {time.time() - t0:.1f} s', flush=True)
            return 0
        wait_before_goal(n, a.goal_delay_min, a.goal_delay_max, a.seed)
        return publish_goal(n, left, t0)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
