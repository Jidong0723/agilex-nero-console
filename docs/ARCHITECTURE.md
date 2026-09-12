# Architecture

## Ownership model

Exactly one `OperationalSpaceController` process owns the NERO USB-CAN adapter. Browser input, policy clients, and operator actions must pass through OSC; no other component may write CAN frames.

```text
Browser / local client
        |
        v
HTTP service / AdapterRuntime (127.0.0.1:8765)
  |-- WebAdapter, pi0.5, PICO, cameras
  |-- narrow OSC client
        |
        v
OperationalSpaceController process
  |-- lease manager
  |-- authority epoch / writer arbitration
  |-- single Pink/Ruckig operational-space servo
  |-- safety validation
  |-- action lifecycle and logs
        |
        v
NeroRobot backend
        |
        v
pyAgxArm -> python-can -> CANDO -> NERO / AGX gripper
```

## Runtime and module boundaries

`src/nero_console/` is the public package boundary. It owns the `nero-console`
CLI, deterministic environment discovery, and future application/domain/
infrastructure/adapters namespaces. Existing modules below remain compatible
implementation modules while callers migrate to the package boundary.

- `application`: HTTP service lifecycle, AdapterRuntime, and OSC use cases.
- `domain`: OSC commands, state snapshots, safety and authority concepts.
- `infrastructure`: runtime discovery, process boundaries, CAN/SDK and solver transport.
- `adapters`: browser, pi0.5 and external PICO input adaptation. Adapters own
  anchors, device tracking, policy/camera lifecycle, and only emit absolute
  base-frame TCP targets to OSC.

The runtime is intentionally split: `.venv`/Python 3.12 owns the HTTP service,
watchdog, reset helper, SDK and CAN; `.conda/nero-kinematics`/Python 3.11 owns
only Pinocchio/Pink. A service started by global Python is rejected before it
can construct hardware transport.

## Compatible implementation modules

- `scripts/nero_control_server.py`: compatible localhost HTTP/static-file server entry point.
- `supervisor/control.py`: Operational Space Controller, CAN ownership, leases, preemption, action lifecycle, HOLD/FREEDRIVE handoff, and status snapshots.
- `supervisor/authority.py`: current hardware writer, servo mode, and monotonically increasing authority epoch.
- `nero_backend/robot.py`: NERO SDK access, feedback, mode transitions, CPV, joint/Cartesian motion, and gripper commands.
- `motion/safety.py`: finite-value, dimension, status, speed, workspace, and gripper validation.
- `motion/osc.py`: the OSC Pink/Ruckig/CPV servo for absolute TCP targets.
- `motion/osc_kinematics_server.py`: separate Python 3.11 kinematics process.

## Priorities

```text
physical power switch > operator takeover > leased action / teleoperation
```

A lease token is necessary but not sufficient to move. A request must also pass feedback-readiness, active-writer, authority-epoch, safety, and backend checks.
