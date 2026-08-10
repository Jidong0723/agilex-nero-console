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

`config/runtime.json` includes a 1–5% speed range, gripper width range, and force limit. These are examples for the verified setup, not universal NERO limits.

`config/teleop.json` defines control frequency, deadman and stale-data thresholds, Cartesian speed and workspace limits, solver parameters, and joint conventions.

### Joint-motion envelope

The configured OSC envelope is `1.5 rad/s` per joint and `5.0 rad/s²` per
joint.  It is deliberately enforced at three points with the same values:

- Pink's `VelocityLimit` and `AccelerationLimit` constrain each IK proposal.
- The OSC dispatch loop rechecks acceleration using the actual elapsed dispatch
  interval after any stale-feedback scaling.
- The final safety gate rejects or clips a value that would violate the
  effective controller/URDF envelope.

These are defense-in-depth checks, not independent speed profiles.  In
hardware mode the OSC also writes the same CV/ACC/DCC profile through the
official NERO CPV API when the profile-sync action is invoked.  The fixed
NERO envelope reported by the backend is a higher physical capability bound;
it can only reduce an OSC request, never increase it.

`shadow_transport.max_joint_speed_rad_s` and
`shadow_transport.max_joint_acceleration_rad_s2` use the same values so the
shadow CPV plant follows the configured envelope.  Read-only CAN calibration
records observed CPV values but never overwrites this configured envelope.

## Required commissioning sequence

1. Inspect wiring, tool, payload, workspace, and physical stop.
2. Start with no person or fragile object in the reachable workspace.
3. Confirm fresh seven-joint, flange, arm-status, and gripper feedback.
4. Use Shadow for teleoperation input.
5. Enable hardware only at the lowest configured speed.
6. Test one small axis at a time before combined Cartesian motion.
7. Revalidate after every firmware, SDK, tool, URDF, or safety-config change.

Do not operate unattended.
