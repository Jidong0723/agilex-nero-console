# NERO Control Console

A local, safety-oriented Web control console for the AgileX NERO 7-axis robotic arm and AGX gripper.

The application owns the USB-CAN adapter, exposes a localhost HTTP API, and provides guarded HOLD, FREEDRIVE, Cartesian/joint motion, gripper control, and Pink/Ruckig Cartesian teleoperation. It intentionally contains no camera, dataset, perception, or policy-training code.

> **Safety warning:** This software can command physical machinery. Test with a clear workspace, low speed, accessible power switch, and an operator ready to intervene. No software stop replaces a physical emergency stop. Use at your own risk.

## Supported setup

- Windows 10/11 x64
- Python 3.12 for the control service
- Miniforge/Conda with Python 3.11 for Pink/Ruckig kinematics
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
- Pink inverse kinematics and Ruckig rate limiting for Cartesian teleoperation
- Shadow teleoperation before hardware movement
- Structured runtime logs and watchdog-assisted service recovery
- Unit tests that use fake hardware and send no CAN frames

## Repository layout

```text
config/          Runtime and teleoperation safety configuration
motion/          Safety validation and Cartesian teleoperation
nero_backend/    NERO/AGX hardware adapter and CAN recovery
scripts/         HTTP service, watchdog, and restart helper
shared/          Shared schemas
supervisor/      Lease, authority, lifecycle, and logging logic
tests/           Hardware-free unit tests
vendor/          Pinned local snapshots of upstream AgileX dependencies
web/console/     Browser console
```

See [Windows deployment](docs/DEPLOYMENT_WINDOWS.md), [Architecture](docs/ARCHITECTURE.md), [Safety](docs/SAFETY.md), and [HTTP API](docs/API.md) for details.

## Installation

Clone the repository and open PowerShell in its root:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd neroAgilex-pico-teleop
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

Do not copy `.venv` or `.conda` from another computer. Both environments contain machine-specific paths and native binaries and must be recreated locally. The control service uses `.venv`; the isolated Pinocchio/Pink solver uses `.conda/nero-kinematics`.

The project includes pinned local snapshots of the official AgileX [`pyAgxArm`](https://github.com/agilexrobotics/pyAgxArm) and [`python-can-agx-cando`](https://github.com/agilexrobotics/python-can-agx-cando) projects because the verified NERO v120 integration contains parser behavior that must remain reproducible. Their licenses are preserved; see [Third-party notices](THIRD_PARTY_NOTICES.md). See [repository portability](docs/REPOSITORY_PORTABILITY.md) for the complete GitHub handoff boundary.

## Start the console

Connect the CANDO adapter and NERO CAN cable, clear the workspace, then run:

```powershell
.\run_console.cmd
```

Open <http://127.0.0.1:8765/>. The service binds to localhost by default. Do not expose it directly to a public or untrusted network.

The first teleoperation session remains Shadow by default. Review live feedback and limits before explicitly enabling hardware control in the page.

## Tests

Tests use fake hardware and do not access USB-CAN:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This frozen hardware-validated release preserves its test sources byte-for-byte. The complete discovery command may not terminate because one legacy real-time-loop test has no test-side timeout. It is therefore not used as an automated CI gate for this release. This limitation does not alter the runtime control path.

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

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For security or safety-sensitive reports, follow [SECURITY.md](SECURITY.md).

Original console code is licensed under the [MIT License](LICENSE). Bundled third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
