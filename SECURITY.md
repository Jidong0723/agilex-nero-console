# Security and safety policy

## Supported version

Only the latest commit on the default branch is supported. Hardware behavior is currently verified only on the configuration described in the README.

## Reporting

For a vulnerability or defect that could cause unsafe motion, loss of control authority, stale-command execution, or unintended network exposure, use GitHub's private security advisory feature. Do not publish exploit or unsafe-motion details before a mitigation is available.

For ordinary bugs and feature requests, open an issue with hardware writes disabled whenever possible.

## Operational scope

- The HTTP service is designed for localhost use. It has no authentication suitable for hostile networks.
- Shadow and fake-hardware tests do not prove a setup is safe for real movement.
- Physical workspace, payload, tool geometry, firmware, cabling, and emergency-stop access remain the operator's responsibility.
- A software HOLD is not a physical emergency stop.
