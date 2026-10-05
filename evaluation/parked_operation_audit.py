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


def control_window_metrics(t, pose, cmd, owner, rates, limits=None):
    """PARK_FIXED 全窗：**實際轉給全身**（owner 首次 = 1）→ **實際交還導航**（之後首次 owner = 0）。

    與執行端 park_fixed.HoldMonitor 同一口徑（Codex 20261005_115350）：位姿與速率含切換當步（以切換前一步為起點）
    與交還當步（最後一個全身控制區間 k−1→k）；命令只核全身控制的步，不核導航首筆命令；偏離以切換前一步位姿為準。
    沒有交還 ⇒ 證據不足（不是通過）。
    """
    limits = LIMITS if limits is None else limits
    linear, angular, yaw = rates
    owner = np.asarray(owner)
    wb = np.flatnonzero(owner == 1)
    if len(wb) == 0:
        return {"status": "no_wholebody_control"}
    i0 = int(wb[0])
    later_nav = np.flatnonzero((np.arange(len(owner)) > i0) & (owner == 0))
    if i0 == 0:
        return {"status": "insufficient: no pre-switch sample"}
    if len(later_nav) == 0:
        return {"status": "insufficient: no handback to nav", "transfer_sim_t": float(t[i0])}
    i1 = int(later_nav[0])
    other = np.flatnonzero((np.arange(len(owner)) > i0) & (np.arange(len(owner)) < i1) & (owner != 1))
    ix = np.arange(i0, i1 + 1)                       # 位姿／速率：含交還當步（k−1→k）
    ic = np.arange(i0, i1)                           # 命令：只核全身控制的步
    if not (np.isfinite(linear[ix]).all() and np.isfinite(angular[ix]).all() and np.isfinite(cmd[ic]).all()):
        return {"status": "insufficient: nonfinite evidence"}
    ref = pose[i0 - 1]
    exc = np.linalg.norm(pose[ix, :2] - ref[:2], axis=1)
    yexc = np.abs(yaw[ix] - yaw[i0 - 1])
    stationary = (linear[ix] <= limits["linear_speed_mps"]) & (angular[ix] <= limits["angular_speed_radps"])
    czero = ((np.linalg.norm(cmd[ic, :2], axis=1) <= limits["command_linear_mps"])
             & (np.abs(cmd[ic, 2]) <= limits["command_angular_radps"]))
    gaps_ok = bool(np.max(np.diff(t[i0 - 1:i1 + 1])) <= 0.01001)
    ok = bool(gaps_ok and len(other) == 0 and stationary.all() and czero.all()
              and exc.max() <= limits["operation_excursion_m"]
              and yexc.max() <= limits["operation_yaw_excursion_rad"])
    return {
        "status": "evaluated",
        "transfer_sim_t": float(t[i0]), "handback_sim_t": float(t[i1]),
        "n_physics_samples": int(len(ix)), "gaps_ok": gaps_ok,
        "other_owner_steps_inside": int(len(other)),
        "max_excursion_mm": float(exc.max() * 1000),
        "max_yaw_excursion_deg": float(np.rad2deg(yexc.max())),
        "physics_speed_max_mmps": float(linear[ix].max() * 1000),
        "physics_angular_speed_max_radps": float(angular[ix].max()),
        "max_applied_linear_command_mmps": float(np.linalg.norm(cmd[ic, :2], axis=1).max() * 1000),
        "max_applied_angular_command_radps": float(np.abs(cmd[ic, 2]).max()),
        "strict_stationary_full_window": ok,
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
    owner = np.array([r[cols.index("owner(0=nav,1=wb,2=glide)")] for r in rows])
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
        # PARK_FIXED 全窗（與執行端 park_gate.json 同口徑；兩者不一致不得宣告模式通過）
        "control_window": control_window_metrics(t, pose, cmd, owner, rates),
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
