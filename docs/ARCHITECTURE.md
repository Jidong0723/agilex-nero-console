# Architecture

## Ownership model

Exactly one `RobotControlBroker` process owns the NERO USB-CAN adapter. Browser input, policy clients, and operator actions must pass through this broker; no other component may write CAN frames.

```text
Browser / local client
        |
        v
HTTP service (127.0.0.1:8765)
        |
        v
RobotControlBroker
  |-- lease manager
  |-- authority epoch / writer arbitration
  |-- teleoperation controller
  |-- safety validation
  |-- action lifecycle and logs
        |
        v
NeroRobot backend
        |
        v
pyAgxArm -> python-can -> CANDO -> NERO / AGX gripper
```

## Main modules

- `scripts/nero_control_server.py`: localhost HTTP/static-file server and process entry point.
- `supervisor/control.py`: broker, leases, preemption, action lifecycle, HOLD/FREEDRIVE handoff, and status snapshots.
- `supervisor/authority.py`: current hardware writer, servo mode, and monotonically increasing authority epoch.
- `nero_backend/robot.py`: NERO SDK access, feedback, mode transitions, CPV, joint/Cartesian motion, and gripper commands.
- `motion/safety.py`: finite-value, dimension, status, speed, workspace, and gripper validation.
- `motion/teleop.py`: deadman/staleness supervision, Pink solver client, Ruckig limiting, and CPV streaming.
- `motion/teleop_kinematics_server.py`: separate Python 3.11 kinematics process.

## Priorities

```text
physical power switch > operator takeover > leased action / teleoperation
```

A lease token is necessary but not sufficient to move. A request must also pass feedback-readiness, active-writer, authority-epoch, safety, and backend checks.
