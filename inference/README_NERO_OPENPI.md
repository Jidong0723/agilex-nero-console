# NERO OpenPI inference

For the optional H20 inference-time RTC service, new AutoDL instance, timing
contract and no-motion validation results, see [RTC deployment guide](README_RTC.md).
Ordinary inference remains the default.

The console owns one OpenPI MessagePack/WebSocket client for both local 5090
and AutoDL presets. Start the ordinary policy server using its deployment's
existing checkpoint and matching normalization assets. The console does not
launch a model server, SSH tunnel or robot motion when saving settings.

For local/WSL deployments, expose the policy port to the Windows host.

## AutoDL: two windows

Replace every `<placeholder>` with the actual deployment value. The console's
AutoDL preset normally connects to `127.0.0.1:8001`; use its **saved local port**
in window A. The remote model port may differ from the local port.

Window A, Windows PowerShell: establish the tunnel and keep the window open.
An idle tunnel normally prints no continuous output. Forward setup errors remain
visible; a listening port alone doesn't prove that the remote policy is ready.

```powershell
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -L '127.0.0.1:8001:127.0.0.1:<policy-port>' -p '<ssh-port>' '<user>@<remote-host>'
```

Window B, Windows PowerShell: log into the same AutoDL instance.

```powershell
ssh -p '<ssh-port>' '<user>@<remote-host>'
```

Then in **window B's remote Bash shell**, start the ordinary OpenPI server:

```bash
cd -- "<openpi-project-directory>" || exit 1
export PATH="$HOME/.local/bin:$PATH"
export BOTO_CONFIG="$HOME/.boto"
exec uv run --frozen --no-sync scripts/serve_policy.py \
  --port "<policy-port>" \
  policy:checkpoint \
  --policy.config="<training-config-name>" \
  --policy.dir="<checkpoint-directory>"
```

Use the deployment's existing checkpoint and matching normalization assets.
The remote `--port` must match window A's `<policy-port>`, while SSH `-p` is the
SSH login port. The forwarding target `127.0.0.1` refers to the remote machine;
the WebSocket client's `127.0.0.1` refers to the console host.

If the model is already running, only window A is needed. The console establishes
WebSocket automatically; no third terminal is needed. Closing window A removes
the tunnel. Stopping console output doesn't stop SSH or the remote policy.
H/K settings never select a checkpoint or change the loaded model's horizon.
Deployment information and passwords aren't persisted by this help UI.

## Connection diagnostics

The two cards have the same meaning for both presets:

- **Service entry**: address-specific local listener inspection, or TCP evidence
  from a real connection attempt. Inspection failure is unknown, not unavailable.
- **Model WebSocket**: WebSocket handshake plus a valid OpenPI metadata object.

A green entry with a failed WebSocket is expected if the local tunnel is up but
the remote server isn't ready, the port targets a different service, or metadata
is invalid. Check window B's startup logs and the exact WebSocket error. For an
unavailable entry, inspect SSH errors, local port conflicts and the saved endpoint.
For startup failures, check the project path, uv environment, training config,
checkpoint and normalization assets.

TCP evidence is collected on the socket subsequently used for WebSocket; there
are no extra connect-and-close probes. Listener checks don't gate real connection
attempts, so WSL forwarding isn't blocked by a missing local listener-table entry.
Evidence expires after ten seconds without refresh, and endpoint changes discard
old results. A non-loopback AutoDL address means direct connection; the local SSH
tunnel instructions only apply after saving a loopback endpoint.

The separate 7D/H badge validates the data contract. An established WebSocket
can remain green while the badge rejects incompatible metadata or actions.

## Data contract

The server sends MessagePack metadata first. Declare `state_dim: 7`,
`valid_action_dim: 7` (or `action_dim: 7`) and `action_horizon: H` when supported.
Missing declarations remain pending until the console validates an actual
response. Explicit incompatible declarations prevent execution.

Every request contains `observation/image` and `observation/wrist_image`
(`uint8[224,224,3]`, RGB, letterboxed by the shared camera resource),
`observation/state` (`float32[7]`) and `prompt`. State is base-frame position
(metres), rotation vector (radians), and normalized absolute gripper opening.
There is no redundant negative gripper component, image encoding negotiation,
or real-time chunking context. A server requiring state8 must be updated to
the state7 contract; the console does not silently switch formats.

Responses contain finite `actions[H,7]`: translation and rotation relative to
the observation TCP at chunk start, followed by absolute gripper opening ratio.
The console validates the whole chunk and executes its first K rows through
OSC. Both H and K are positive integers, K <= H. Changing H in the UI sets the
expected response length; it does not change a loaded model's prediction horizon.

`config/pi05.json` schema `nero.inference.v2` stores endpoint and H/K once per
preset. Legacy configurations migrate in memory on load, retaining the active
runtime endpoint and lengths; the first successful save writes the canonical
structure. `/api/pi05/state` still publishes derived model/connection/execution
fields for existing consumers. `POST /api/pi05/config` accepts a preset,
`policy_endpoint: {host, port}`, and paired
`execution: {expected_chunk_steps: H, replan_steps: K}`. Settings are validated
and atomically saved before publication; running/priming output must be stopped
before changing endpoint or H/K. Each preset remembers its last saved values.

Use preview inference to validate compatibility before enabling output. Test
deployments with a no-motion request first. Input and output validation does
not establish that checkpoint normalization or task behavior is correct.

Offline checks use fake policies, loopback WebSocket servers and fake OSC only:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_inference*.py"
node --test tests/test_inference_startup.cjs tests/inference_panel.test.cjs
```

Removed experimental inference services and proxies remain recoverable from Git
history. Removal from this checkout does not stop or update external deployments.
