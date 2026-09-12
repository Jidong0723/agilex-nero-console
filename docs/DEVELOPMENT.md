# Development guide

## Runtime boundary

The repository intentionally uses two local, ignored environments:

- `.venv` / Python 3.12 is the only process allowed to start the HTTP service, watchdog, reset helper, SDK, and CAN transport owner.
- `.conda/nero-kinematics` / Python 3.11 is used only by the isolated Pink/Pinocchio solver.

Do not launch `scripts/nero_control_server.py` with a global Python. Use `run_console.cmd`, `python -m nero_console serve`, or `nero-console serve` from the project `.venv`.

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\setup-kinematics.ps1
.\.venv\Scripts\python.exe -m nero_console doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite uses fakes and shadow transport. Do not add a test that opens real CAN hardware to CI.

## Vendored CANDO baseline

`vendor/python-can-agx-cando` is synchronized with AgileX upstream commit
[`b222c4027ad4f6599f7634c72c67184619177972`](https://github.com/agilexrobotics/python-can-agx-cando/commit/b222c4027ad4f6599f7634c72c67184619177972).
Keep the upstream RX/TX concurrency model intact; ownership of CAN writes is
enforced by this project's `HardwareTxOwner`, not by private locks in CANDO.

## Dependency groups

- `requirements/control.txt`: runtime and local SDK dependencies.
- `requirements/pi05.txt`: optional pi0.5 camera/inference adapter dependencies.
- `requirements/dev.txt`: development test dependency entry point.
- `environment-kinematics.yml`: only the Pinocchio/Pink Conda environment.
