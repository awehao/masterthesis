#!/usr/bin/env python3
"""Read-only audit of actual base motion, independent of the MOTM mode label.

Uses physics-step pose differences, not the previously observed latched twist.
Outputs JSON to stdout. Thresholds are retrospective engineering screens for old
runs, not revisions to their registered task-success criteria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LIMITS = {
    "linear_speed_mps": 0.001,
    "angular_speed_radps": 0.01,
    "operation_excursion_m": 0.001,
    "operation_yaw_excursion_rad": float(np.deg2rad(0.5)),
    "command_linear_mps": 1e-6,
    "command_angular_radps": 1e-6,
}


def pose_rates(t, pose, lag=1):
    t, pose = np.asarray(t, float), np.asarray(pose, float)
    if len(t) < 2 or pose.shape != (len(t), 3):
        raise ValueError("insufficient or malformed pose samples")
    if not np.isfinite(t).all() or not np.isfinite(pose).all():
        raise ValueError("nonfinite timestamp or pose")
    if np.any(np.diff(t) <= 0) or lag < 1:
        raise ValueError("timestamps must strictly increase; lag must be positive")
    yaw = np.unwrap(pose[:, 2])
    linear = np.full(len(t), np.nan)
    angular = np.full(len(t), np.nan)
    elapsed = t[lag:] - t[:-lag]
    linear[lag:] = np.linalg.norm(pose[lag:, :2] - pose[:-lag, :2], axis=1) / elapsed
    angular[lag:] = np.abs(yaw[lag:] - yaw[:-lag]) / elapsed
    return linear, angular, yaw


def interval_metrics(t, pose, cmd, start, end, rates, limits=None):
    limits = LIMITS if limits is None else limits
    linear, angular, yaw = rates
    # Task timestamps are float32-derived; tolerate only the observed rounding
    # difference (< 1 us), not missing physical steps.
    ix = np.flatnonzero((t >= start - 1e-6) & (t < end - 1e-6))
    if len(ix) < 2:
        raise ValueError("interval has fewer than two physical samples")
    p, u = pose[ix], cmd[ix]
    if not np.isfinite(u).all() or not np.isfinite(linear[ix]).all() or not np.isfinite(angular[ix]).all():
        raise ValueError("missing command or velocity evidence")
    excursion = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
    yaw_excursion = np.abs(yaw[ix] - yaw[ix[0]])
    stationary = ((linear[ix] <= limits["linear_speed_mps"])
                  & (angular[ix] <= limits["angular_speed_radps"]))
    command_zero = ((np.linalg.norm(u[:, :2], axis=1) <= limits["command_linear_mps"])
                    & (np.abs(u[:, 2]) <= limits["command_angular_radps"]))
    coverage_ok = (abs(t[ix[0]] - start) <= 0.01001
                   and abs(t[ix[-1]] - end) <= 0.02001
                   and np.max(np.diff(t[ix])) <= 0.01001)
    return {
        "requested_span_sim_s": [float(start), float(end)],
        "sample_span_sim_s": [float(t[ix[0]]), float(t[ix[-1]])],
        "n_physics_samples": int(len(ix)),
        "coverage_ok": bool(coverage_ok),
        "net_displacement_mm": float(np.linalg.norm(p[-1, :2] - p[0, :2]) * 1000),
        "path_length_mm": float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() * 1000),
        "max_excursion_mm": float(excursion.max() * 1000),
        "max_yaw_excursion_deg": float(np.rad2deg(yaw_excursion.max())),
        "physics_speed_p50_p95_max_mmps": (np.percentile(linear[ix], [50, 95, 100]) * 1000).tolist(),
        "physics_angular_speed_p50_p95_max_radps": np.percentile(angular[ix], [50, 95, 100]).tolist(),
        "stationary_sample_fraction": float(stationary.mean()),
        "max_applied_linear_command_mmps": float(np.linalg.norm(u[:, :2], axis=1).max() * 1000),
        "max_applied_angular_command_radps": float(np.abs(u[:, 2]).max()),
        "zero_command_sample_fraction": float(command_zero.mean()),
        "retrospective_strict_stationary_screen": bool(
            coverage_ok and stationary.all() and command_zero.all()
            and excursion.max() <= limits["operation_excursion_m"]
            and yaw_excursion.max() <= limits["operation_yaw_excursion_rad"]),
    }


def audit_run(run):
    run = Path(run)
    paths = {name: run / name for name in ("room_run.json", "task.json", "wholebody.json")}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    room, task, wholebody = (data[name] for name in paths)
    cols = room["steps_cols"]
    rows = room["steps"]
    t = np.array([r[cols.index("sim_t")] for r in rows], float)
    pose = np.array([r[cols.index("base_xyth")] for r in rows], float)
    cmd = np.array([r[cols.index("base_cmd_body")] for r in rows], float)
    if cmd.shape != (len(t), 3) or not np.isfinite(cmd).all():
        raise ValueError("nonfinite or malformed base command")
    rates = pose_rates(t, pose)
    phases = {}
    for event in task["events"]:
        if "phase" in event:
            phases.setdefault(event["phase"], float(event["sim_t"]))
    unfolds = [float(e["sim_t"]) for e in wholebody["events"] if e.get("to") == "UNFOLD"]
    if not unfolds or "ALIGN" not in phases or "HANDBACK_WAIT" not in phases:
        raise ValueError("missing unfold, align or handback boundary")
    boundaries = sorted([(unfolds[0], "UNFOLD")]
                        + [(stamp, phase) for phase, stamp in phases.items()])
    per_phase = {}
    for (start, phase), (end, _) in zip(boundaries, boundaries[1:]):
        if unfolds[0] <= start < phases["HANDBACK_WAIT"]:
            per_phase[phase] = interval_metrics(t, pose, cmd, start, end, rates)
    operation = interval_metrics(t, pose, cmd, phases["ALIGN"], phases["HANDBACK_WAIT"], rates)
    grasp = next((float(e["sim_t"]) for e in task["events"] if e.get("grip") == "close"), None)
    attached = next((float(e["sim_t"]) for e in task["events"] if e.get("attached") is True), None)
    return {
        "run": run.name,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "recorded_motm": task["args"]["motm"],
        "completed": task.get("final_phase") == "DONE",
        "operation_scope": "ALIGN inclusive to HANDBACK_WAIT exclusive; UNFOLD reported separately",
        "per_phase": per_phase,
        "operation": operation,
        "grasp_close_to_attach": (interval_metrics(t, pose, cmd, grasp, attached, rates)
                                  if grasp is not None and attached is not None else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+")
    args = parser.parse_args()
    out = {
        "schema": "parked_operation_audit/1",
        "method": "each physical-step pose difference divided by recorded elapsed time; yaw unwrapped; no twist",
        "screen_limits": LIMITS,
        "status": "retrospective audit; does not revise historical S1-S6 or certify a new experiment",
        "runs": [audit_run(p) for p in args.run_dir],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
