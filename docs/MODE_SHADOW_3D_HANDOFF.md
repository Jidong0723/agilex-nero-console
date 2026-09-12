# OSC Shadow and Hardware Handoff

OSC has two execution modes: `shadow` and `hardware`. The mode determines
whether the final safe joint command is sent to the Shadow CPV plant or the
single hardware CAN writer; it does not encode an input device.

Input adapters run in the HTTP process. WebAdapter, π0.5, and PICO may retain
device-local anchors, relative poses, camera frames, policy state, and pairing
information. Their only robot-facing output is an absolute base-frame TCP
target sent through the OSC API.

```text
AdapterRuntime -> narrow OSC client -> OperationalSpaceController
                                      -> Pink/Ruckig/Safety
                                      -> Shadow plant or CAN
```

An OSC session is owned by its `client_id`, `session_id`, monotonic command
sequence, and heartbeat lease. `hold`, heartbeat expiry, authority changes,
and faults clear the active absolute target and use the same braking path.
The next accepted absolute TCP target may safely resume from `HOLD_READY`.
