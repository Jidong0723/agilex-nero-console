# Contributing

Contributions are welcome, especially documentation, fake-hardware tests, browser usability, and support for clearly identified NERO firmware versions.

## Development setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Pull-request requirements

1. Keep USB-CAN ownership inside the control service.
2. Preserve lease, authority-epoch, deadman, stale-feedback, HOLD, and operator-preemption behavior.
3. Add a hardware-free regression test for behavioral changes.
4. Do not commit environments, runtime logs, device serial numbers, credentials, or captured user data.
5. Document the NERO model, firmware, end effector, adapter, and safety setup for hardware results.
6. Never describe untested hardware behavior as verified.

Changes that can move hardware should first be exercised with fake hardware, then Shadow, then a constrained low-speed setup with an operator and physical stop available.

## Style

- Target Python 3.12 for the control service.
- Prefer explicit state and failure handling over implicit recovery.
- Keep HTTP handlers thin; ownership and safety decisions belong in the supervisor/backend layers.
- Avoid changing vendored dependencies unless the upstream revision and local reason are documented.
