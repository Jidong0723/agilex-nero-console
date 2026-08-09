# Windows deployment

This release targets Windows 10/11 x64, NERO firmware v120, and the AgileX CANDO USB-CAN adapter. Other combinations are unverified.

## Prerequisites

- Python 3.12 x64, available through the `py` launcher or an explicit path.
- Miniforge or another Conda installation with the `conda` command available in PowerShell.
- The CANDO Windows driver and a connected NERO CAN cable.
- A clear workspace and an accessible physical emergency stop or power switch.

## Clone and install

```powershell
git clone <YOUR_REPOSITORY_URL>
cd neroAgilex-pico-teleop
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\setup-kinematics.ps1
```

Replace `<YOUR_REPOSITORY_URL>` with the published GitHub repository URL.

The two environments are intentionally separate:

- `.venv` runs the Web service, NERO SDK, CANDO backend, and Ruckig control loop.
- `.conda/nero-kinematics` runs Pinocchio and Pink in an isolated subprocess.

Do not upload or copy either environment between computers. Virtual environments contain absolute paths and platform-specific binaries. Recreate them from `requirements.txt` and `environment-kinematics.yml` on every workstation.

## Start

```powershell
.\run_console.cmd
```

Open <http://127.0.0.1:8765/>. The service intentionally binds only to localhost. Do not expose this HTTP service directly to a LAN or the public Internet.

Start with Shadow mode. Confirm that joint feedback, the solver, workspace limits, and the TCP display are valid before selecting hardware control.

## Solver offline

Check the solver environment directly:

```powershell
.\.conda\nero-kinematics\python.exe -c "import pinocchio, pink; print(pinocchio.__version__, pink.__version__)"
```

Expected versions are Pinocchio `4.1.0` and Pink `4.3.0`. If the executable is missing, open a Miniforge PowerShell prompt and rerun `setup-kinematics.ps1`. If `conda` is not recognized, initialize Conda for PowerShell or invoke the script from a Miniforge prompt.

## Control environment checks

```powershell
.\.venv\Scripts\python.exe -c "import numpy, ruckig, pyAgxArm, agx_cando; print('control imports OK')"
```

If CANDO cannot be opened, verify the Windows driver, USB connection, architecture (x64), and that no second NERO control process owns the adapter. The service only supports one USB-CAN owner.

## Safe first run

1. Power the arm in a clear, supported pose.
2. Open the console and verify fresh seven-axis feedback.
3. Use Shadow teleoperation first.
4. Test HOLD and FREEDRIVE without a payload.
5. Enter hardware teleoperation at low speed and make only a small free-space movement.

Software protection does not replace the physical emergency stop. Stop if feedback, solver state, joint limits, or motion direction are inconsistent.
