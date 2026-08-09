# Repository portability

This repository is self-contained for the desktop Web-control program: all application source, browser assets, configuration, robot description, and the verified AgileX/CANDO SDK snapshots are stored beneath the repository root.

## Files that must be committed

- `config/`, `motion/`, `nero_backend/`, `scripts/`, `shared/`, `supervisor/`, `web/`, and `tests/`
- `vendor/`, including the NERO URDF, `pyAgxArm`, `python-can-agx-cando`, and the CANDO Windows DLLs
- `requirements.txt`, `environment-kinematics.yml`, `setup.ps1`, `setup-kinematics.ps1`, `run_console.cmd`, and `run_console_watchdog.cmd`
- Documentation and third-party licence notices

The control service resolves its runtime paths from the repository root. `config/teleop.json` references the kinematics interpreter, solver script, and URDF with repository-relative paths.

## Local files that must not be committed

- `.venv/`: the machine-local Python 3.12 control environment
- `.conda/`: the machine-local Pinocchio/Pink environment
- `runtime/`: PID files, logs, watchdog state, and other service output
- `_local_tmp/`: local-only archived or diagnostic files

These paths are intentionally ignored by Git. Recreate the first two environments on each computer; do not copy them between machines.

## Bootstrap on a new computer

```powershell
git clone <YOUR_REPOSITORY_URL>
cd neroAgilex-pico-teleop
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\setup-kinematics.ps1
.\run_console.cmd
```

The workstation still needs its normal external prerequisites: Python 3.12, Conda/Miniforge, and the CANDO Windows driver. Those are installed software or drivers, not project files. Hardware use also requires a connected supported NERO arm and CANDO adapter.

## Scope boundary

`pico_unity_client/` is an optional, separate Unity project for the PICO companion application. It is not required to run the desktop Web console, HTTP API, shadow mode, or hardware control. Keep it as a separate repository or intentionally add it later only when distributing the APK source as well; it is not automatically moved or included in the Web-control handoff.

## Before publishing

Run the following from the repository root:

```powershell
git status --short
git diff --check
.\.venv\Scripts\python.exe -m unittest tests.test_control_http_bootstrap
```

Confirm that only intended source, configuration, documentation, and vendored dependency changes are staged. Never commit real `runtime/` files, generated environments, CAN logs containing sensitive operational data, or local diagnostic artifacts.
