# Safety model

This document describes software defenses, not a safety certification.

## Fail-closed checks

- The service binds to localhost by default.
- Only one process owns USB-CAN and one motion owner holds a lease.
- Feedback must be connected, fresh, complete, and free of reported arm errors.
- Every action is checked for finite dimensions and configured limits.
- Operator takeover revokes leases and preempts active writers.
- Expired leases trigger HOLD.
- Deadman release and stale teleoperation feedback stop velocity submission.
- Authority epochs reject queued or stale writes from an older controller state.
- Electronic emergency damping is disabled by default; use the physical power switch for a true emergency.

## Configuration boundaries

`config/runtime.json` includes a 1–5% speed range, conservative Cartesian workspace, flange-height floor, gripper width range, and force limit. These are examples for the verified setup, not universal NERO limits.

`config/teleop.json` defines control frequency, deadman and stale-data thresholds, Cartesian speed limits, solver parameters, and joint conventions.

## Required commissioning sequence

1. Inspect wiring, tool, payload, workspace, and physical stop.
2. Start with no person or fragile object in the reachable workspace.
3. Confirm fresh seven-joint, flange, arm-status, and gripper feedback.
4. Use Shadow for teleoperation input.
5. Enable hardware only at the lowest configured speed.
6. Test one small axis at a time before combined Cartesian motion.
7. Revalidate after every firmware, SDK, tool, URDF, or safety-config change.

Do not operate unattended.
