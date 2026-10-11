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

## OSC maintenance tuning

The maintenance panel separates hardware CPV settings from 12 software tracking,
IK-weight and arrival parameters. Software edits use one group-setting button;
the optional “同时保存为启动配置” checkbox also persists them to `config/osc.json`.
Without that checkbox, a restart reloads the saved startup values.

`POST /api/osc/tuning` accepts `values` (all 12 keys), `save_startup` (boolean),
and `expected_revision`. The current values, saved values and approved ranges
are published in `/api/osc/state` under `transport.osc_tuning`.
The panel displays arrival distances in mm, angles in degrees and dwell in ms;
the API uses metres, radians and seconds. `motion/osc_tuning.py` defines the
allowlist and ranges; control frequency, hard limits, stale-feedback thresholds
and TCP calibration are excluded.

Each servo cycle captures one immutable tuning revision for both Pink requests
and arrival checks. Applying a new group resets arrival dwell on the next cycle.
Saving uses atomic file replacement before publishing the new running revision;
a validation or disk error leaves the running group unchanged. The disk write
does not hold the cycle snapshot lock. Offline coverage:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_osc_tuning tests.test_osc_direct_cpv
node --test tests/console_output_mode.test.js
```

## Hardware CPV group maintenance

The hardware CPV table separates last hardware readback, editable CV and paired
ACC/DCC values, and service startup configuration. Refresh readback explicitly
while idle in HOLD; normal UI polling does not query firmware or replace drafts.
The single setting button sends `POST /api/osc/cpv-parameters` with
`values: {cv_rad_s, acc_dcc_rad_s2}` and boolean `save_startup` (default false).
Both values must be finite positive numbers within the configured OSC limits.
The backend validates the complete group and confirmed fresh-feedback HOLD with
no OSC session before writing, serializes against control handoffs, and verifies
all seven joints plus unchanged PP/KP/KI after writing.

Unchecked applies to current operation without changing service startup config;
mode-entry restoration preserves this runtime choice. Checked additionally saves
`control_service.cpv_profile_defaults` in `config/runtime.json`. Service reset
reloads the saved profile after firmware initialization. This distinction is
about service configuration, not a guarantee of volatile firmware storage.
Firmware writes are sequential and cannot be atomic or rolled back as a group:
a partial failure is reported and never saved as startup config. A save failure
can leave hardware changed; inspect readback before retrying.

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
