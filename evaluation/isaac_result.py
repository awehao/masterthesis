"""Result fields for one Isaac trial, kept pure so they can be tested.

This lives outside isaac_bigarena_sim.py because that module cannot be imported
without Isaac Sim, which meant the arrival arithmetic -- the number every table
in the report is built from -- had no test at all.

The rule it enforces: a trial that did not arrive has no arrival time. The
previous version computed `arrival_time_s` from whatever timestamps existed, so
a run that timed out short of the goal still carried a plausible-looking
completion time, and any later mean over the column silently mixed the two.
"""

# The only stop reason that means the robot got there. Everything else is a
# trial that ended for a reason of its own -- and 'thermal_abort' is not even a
# navigation outcome (see project_dynamic_benchmark: the CPU limit stops runs).
ARRIVED = 'goal_reached_truth'
NOT_A_NAV_OUTCOME = ('thermal_abort', 'signal')


def arrival_fields(stop_reason, first_plan_t, stop_sim_t, motion_start_t,
                   dist_goal=None, arrive_tol=None):
    """Timing fields for one trial.

    `stop_sim_t` is the simulated time at the instant the run stopped.

    Returns a dict with:
      arrived              did this trial reach the goal
      arrival_time_s       time from first plan to arrival, or None
      motion_elapsed_s     time from first motion to arrival, or None
      elapsed_until_stop_s time from first plan to the stop, whatever the reason
      scoreable            whether the trial says anything about navigation
    """
    def _delta(a, b):
        return None if (a is None or b is None) else b - a

    arrived = (stop_reason == ARRIVED)
    # Cross-check the label against the geometry when both are available: a
    # stop_reason of 'goal_reached_truth' that sits outside the tolerance would
    # mean the two disagree, and that is worth failing loudly over rather than
    # averaging in.
    if arrived and dist_goal is not None and arrive_tol is not None:
        if not (dist_goal <= arrive_tol):
            raise ValueError(
                f'stop_reason={stop_reason!r} but the truth pose is '
                f'{dist_goal:.3f} m from the goal, outside tol={arrive_tol:.3f}')
    elapsed = _delta(first_plan_t, stop_sim_t)
    return dict(
        arrived=arrived,
        arrival_time_s=(_delta(first_plan_t, stop_sim_t) if arrived else None),
        motion_elapsed_s=(_delta(motion_start_t, stop_sim_t) if arrived else None),
        elapsed_until_stop_s=elapsed,
        scoreable=(stop_reason not in NOT_A_NAV_OUTCOME),
    )
