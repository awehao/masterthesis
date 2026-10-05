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


HOLD_V_MAX = 0.005      # PARK_HOLD 停車伺服：合線速度上限 m/s（與 park_fixed.HoldServo 同口徑）
HOLD_W_MAX = 0.02       # PARK_HOLD 停車伺服：角速度上限 rad/s


def control_window_metrics(t, pose, cmd, owner, rates, limits=None, mode='fixed', anchor=None):
    """PARK_FIXED 全窗：**實際轉給全身**（owner 首次 = 1）→ **實際交還導航**（之後首次 owner = 0）。

    與執行端 park_fixed.HoldMonitor 同一口徑（Codex 20261005_115350）：位姿與速率含切換當步（以切換前一步為起點）
    與交還當步（最後一個全身控制區間 k−1→k）；命令只核全身控制的步，不核導航首筆命令；偏離以切換前一步位姿為準。
    沒有交還 ⇒ 證據不足（不是通過）。
    mode = 'hold'（PARK_HOLD v2）：偏離以**閘門錨點**為準（須給 anchor）；套用命令只核伺服上界（合線速度 ≤ HOLD_V_MAX、
    |wz| ≤ HOLD_W_MAX）；實測速率與偏離門檻不變。mode = 'fixed'（v1 與舊趟次）照原口徑。
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
    handback = len(later_nav) > 0
    # 沒有交還 ⇒ 評估到紀錄末端：量到的違規照報（VIOLATION 不被「證據不足」蓋掉），否則 insufficient
    i1 = int(later_nav[0]) if handback else len(owner) - 1
    other = np.flatnonzero((np.arange(len(owner)) > i0) & (np.arange(len(owner)) < i1) & (owner != 1))
    ix = np.arange(i0, i1 + 1)                       # 位姿／速率：含交還當步（k−1→k）
    ic = np.arange(i0, i1) if handback else np.arange(i0, i1 + 1)   # 命令：只核全身控制的步
    if not (np.isfinite(linear[ix]).all() and np.isfinite(angular[ix]).all() and np.isfinite(cmd[ic]).all()):
        return {"status": "insufficient: nonfinite evidence"}
    if mode == 'hold':
        if anchor is None or len(anchor) != 3 or not np.isfinite(np.asarray(anchor, float)).all():
            return {"status": "insufficient: PARK_HOLD 需要閘門錨點"}
        ref = np.asarray(anchor, float)
        exc = np.linalg.norm(pose[ix, :2] - ref[:2], axis=1)
        yexc = np.abs((yaw[ix] - ref[2] + np.pi) % (2 * np.pi) - np.pi)
        czero = ((np.linalg.norm(cmd[ic, :2], axis=1) <= HOLD_V_MAX + 1e-9)
                 & (np.abs(cmd[ic, 2]) <= HOLD_W_MAX + 1e-9))
    else:
        ref = pose[i0 - 1]
        exc = np.linalg.norm(pose[ix, :2] - ref[:2], axis=1)
        yexc = np.abs(yaw[ix] - yaw[i0 - 1])
        czero = ((np.linalg.norm(cmd[ic, :2], axis=1) <= limits["command_linear_mps"])
                 & (np.abs(cmd[ic, 2]) <= limits["command_angular_radps"]))
    stationary = (linear[ix] <= limits["linear_speed_mps"]) & (angular[ix] <= limits["angular_speed_radps"])
    gaps_ok = bool(np.max(np.diff(t[i0 - 1:i1 + 1])) <= 0.01001)
    ok = bool(gaps_ok and len(other) == 0 and stationary.all() and czero.all()
              and exc.max() <= limits["operation_excursion_m"]
              and yexc.max() <= limits["operation_yaw_excursion_rad"])
    # 首次違規（逐物理步；速率／偏離以 ix、命令以 ic）
    bad = (~stationary) | (exc > limits["operation_excursion_m"]) | (yexc > limits["operation_yaw_excursion_rad"])
    first = None
    kb = np.flatnonzero(bad)
    kc = np.flatnonzero(~czero)
    cand = []
    if len(kb):
        j = int(ix[kb[0]])
        why = []
        if not stationary[kb[0]]:
            why.append(f"speed {linear[j] * 1000:.3f} mm/s or yaw rate {angular[j]:.4f}")
        if exc[kb[0]] > limits["operation_excursion_m"]:
            why.append(f"excursion {exc[kb[0]] * 1000:.3f} mm")
        if yexc[kb[0]] > limits["operation_yaw_excursion_rad"]:
            why.append(f"yaw excursion {np.rad2deg(yexc[kb[0]]):.3f} deg")
        cand.append((j, "; ".join(why)))
    if len(kc):
        cand.append((int(ic[kc[0]]), "applied command over PARK_HOLD servo bound" if mode == 'hold'
                     else "applied command nonzero"))
    if cand:
        j, why = min(cand)
        first = {"sim_t": float(t[j]), "why": why}
    status = ("violation" if first is not None
              else "evaluated" if handback else "insufficient: no handback to nav")
    return {
        "status": status,
        "mode": mode,
        "first_violation": first,
        "transfer_sim_t": float(t[i0]), "handback_sim_t": float(t[i1]) if handback else None,
        "n_physics_samples": int(len(ix)), "gaps_ok": gaps_ok,
        "other_owner_steps_inside": int(len(other)),
        "max_excursion_mm": float(exc.max() * 1000),
        "max_yaw_excursion_deg": float(np.rad2deg(yexc.max())),
        "physics_speed_max_mmps": float(linear[ix].max() * 1000),
        "physics_angular_speed_max_radps": float(angular[ix].max()),
        "max_applied_linear_command_mmps": float(np.linalg.norm(cmd[ic, :2], axis=1).max() * 1000),
        "max_applied_angular_command_radps": float(np.abs(cmd[ic, 2]).max()),
        "strict_stationary_full_window": bool(ok and handback),
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
    # PARK_HOLD 趟次：以執行端 park_gate.json 的模式與閘門錨點獨立重算；其餘照 v1 口徑
    _mode, _anchor = 'fixed', None
    _pg = run / "park_gate.json"
    if _pg.exists():
        _g = json.loads(_pg.read_text())
        if _g.get("mode") == "PARK_HOLD":
            _mode = 'hold'
            _anchor = (_g.get("gate") or {}).get("anchor")
    phases = {}
    for event in task["events"]:
        if "phase" in event:
            phases.setdefault(event["phase"], float(event["sim_t"]))
    unfolds = [float(e["sim_t"]) for e in wholebody["events"] if e.get("to") == "UNFOLD"]
    # 提早中止的趟次（例如未到 ALIGN）：缺的相位填 NA，控制窗照樣評估，不崩潰
    missing = [k for k, ok_ in (("UNFOLD", bool(unfolds)), ("ALIGN", "ALIGN" in phases),
                                ("HANDBACK_WAIT", "HANDBACK_WAIT" in phases)) if not ok_]
    per_phase = {}
    operation = None
    if not missing:
        boundaries = sorted([(unfolds[0], "UNFOLD")]
                            + [(stamp, phase) for phase, stamp in phases.items()])
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
        "missing_boundaries": missing or None,
        # PARK_FIXED 全窗（與執行端 park_gate.json 同口徑；兩者不一致不得宣告模式通過）
        "control_window": control_window_metrics(t, pose, cmd, owner, rates, mode=_mode, anchor=_anchor),
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
