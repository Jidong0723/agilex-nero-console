# AgileX NERO Console

A local, safety-oriented Operational Space Controller (OSC) console for the AgileX NERO 7-axis arm and AGX gripper. It provides a localhost Web UI, OSC HTTP API, WebAdapter, pi0.5 adapter, PICO server-side adapter protocol, Shadow validation, and guarded hardware control.

> **Safety warning:** This software can command physical machinery. Test with a clear workspace, low speed, accessible power switch, and an operator ready to intervene. No software stop replaces a physical emergency stop. Use at your own risk.

## Supported setup

- Windows 10/11 x64
- Python 3.12 for the control service (`.venv`)
- Conda with Python 3.11 for Pink/Pinocchio kinematics (`.conda/nero-kinematics`)
- AgileX NERO, verified with firmware v120
- AgileX CANDO USB-CAN adapter
- Optional AGX gripper

Other firmware and hardware combinations are unverified.

## Features

- Local Web console at `http://127.0.0.1:8765/`
- Single-process ownership of USB-CAN
- Motion leases and authority epochs to reject stale writers
- Operator takeover and fault preemption
- Conservative workspace, speed, gripper, and state checks
- HOLD and official NERO FREEDRIVE/Leader transitions
- Pink inverse kinematics and CPV joint-position output for Cartesian teleoperation
- Shadow teleoperation before hardware movement
- Structured runtime logs and watchdog-assisted service recovery
- Unit tests that use fake hardware and send no CAN frames

## Repository layout

`src/nero_console/` provides the stable CLI and runtime discovery boundary. Existing `motion/`, `supervisor/`, `nero_backend/`, and `shared/` modules remain supported implementation packages while OSC behavior is migrated behind that boundary.

See [Windows deployment](docs/DEPLOYMENT_WINDOWS.md), [Architecture](docs/ARCHITECTURE.md), [Safety](docs/SAFETY.md), and [HTTP API](docs/API.md) for details.

## Installation

Clone the repository and open PowerShell in its root:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd agilex-nero-console
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Replace `<YOUR_REPOSITORY_URL>` with the GitHub repository URL after publishing.

If Python 3.12 is not registered with the `py` launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -PythonExe C:\Path\To\python.exe
```

For Cartesian teleoperation, install Miniforge/Conda and create the local kinematics environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-kinematics.ps1
```

Do not copy `.venv` or `.conda` from another computer. Both contain machine-specific paths and native binaries and must be recreated locally. The control service, reset helper, watchdog, and hardware worker always use `.venv`; the isolated Pinocchio/Pink solver always uses `.conda/nero-kinematics`.

The project includes pinned local snapshots of the official AgileX [`pyAgxArm`](https://github.com/agilexrobotics/pyAgxArm) and [`python-can-agx-cando`](https://github.com/agilexrobotics/python-can-agx-cando) projects because the verified NERO v120 integration contains parser behavior that must remain reproducible. Their licenses are preserved; see [Third-party notices](THIRD_PARTY_NOTICES.md). See [repository portability](docs/REPOSITORY_PORTABILITY.md) for the complete GitHub handoff boundary.

## Start the console

Connect the CANDO adapter and NERO CAN cable, clear the workspace, then run:

```powershell
.\run_console.cmd
```

Open <http://127.0.0.1:8765/>. The service binds to localhost by default. Do not expose it directly to a public or untrusted network.

The first teleoperation session remains Shadow by default. Review live feedback and limits before explicitly enabling hardware control in the page.

## Preflight

```powershell
.\.venv\Scripts\python.exe -m nero_console doctor
```

The report shows the exact control and kinematics interpreter paths, core dependency origins, configuration paths, and CANDO enumeration. A missing CANDO device is reported diagnostically; the command does not send CAN frames.

## Tests

Tests use fake hardware and do not access USB-CAN:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GitHub Actions runs the hardware-free subset on Windows. Real CAN and physical-motion checks are deliberately excluded from CI.

## Verified mode handoff rules

These are control invariants measured on a NERO v120 arm:

- Never globally or individually enable a joint whose feedback already reports enabled. Re-enabling a live drive can produce a torque transient.
- CPV teleoperation already runs on the Follower side. After braking to zero, transition to software HOLD without sending `set_follower_mode()` again.
- Send `set_follower_mode()` only when returning from genuine FREEDRIVE/Leader or drag-teach mode. Enable only feedback-confirmed disabled joints, once per handoff.
- Normal HOLD and FREEDRIVE handoffs must not create a J-mode `move_j()` position target.

```text
Teleop CPV -> brake -> seven-axis CPV zero confirmed -> HOLD
FREEDRIVE/Leader -> set_follower_mode() -> conditional per-joint enable -> HOLD
```

Do not weaken these rules without controlled hardware validation.

## Configuration

- [`config/runtime.json`](config/runtime.json): CAN interface, motion timeouts, workspace, speed, gripper limits, logging, and service settings.
- [`config/teleop.json`](config/teleop.json): URDF, control frequency, deadman/staleness thresholds, input filtering, and Cartesian limits.

Defaults are intentionally conservative. Review every change with the physical workspace and tool geometry in mind.

## Contributing, security, and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [development guide](docs/DEVELOPMENT.md) before opening a pull request. For security or safety-sensitive reports, follow [SECURITY.md](SECURITY.md).

Original console code is licensed under the [MIT License](LICENSE). Bundled third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
