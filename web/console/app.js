(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  function ensureSelectionControls() {
    const legacy = $("mode");
    if (!$("execution-mode") && legacy) {
      legacy.id = "execution-mode";
      legacy.innerHTML = '<option value="shadow">影子模式</option><option value="hardware">真机模式</option>';
      const label = legacy.closest("label");
      if (label) {
        label.firstChild.textContent = "执行目标";
        const inputLabel = document.createElement("label");
        inputLabel.innerHTML = '输入适配器<select id="input-adapter"><option value="web">WebAdapter · 网页摇杆</option></select>';
        label.parentElement?.insertBefore(inputLabel, label.nextSibling);
      }
    }
  }
  ensureSelectionControls();
  const keys = new Set(["KeyW", "KeyS", "KeyQ", "KeyE", "KeyR", "KeyF", "KeyA", "KeyD"]);
  const clientId = (() => {
    const key = "nero.console.client.v2";
    let value = sessionStorage.getItem(key);
    if (!value) {
      value = crypto.randomUUID?.() || `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      sessionStorage.setItem(key, value);
    }
    return value;
  })();

  const state = {
    status: null,
    teleop: null,
    broker: null,
    action: null,
    latestSequence: 0,
    requestGeneration: 0,
    refreshBusy: false,
    intentBusy: false,
    intentPending: false,
    heartbeatBusy: false,
    heartbeatFailures: 0,
    oscSequence: 0,
    xy: [0, 0],
    right: [0, 0],
    rightMode: "zy",
    webAdapterActive: false,
    oscAnchor: null,
    relativePose: { position_m: [0, 0, 0], orientation_xyzw: [0, 0, 0, 1] },
    lastPoseTick: performance.now(),
    resetPendingUntil: 0,
    keys: new Set(),
    sticks: new Map(),
  };

  const canvas = $("workspace");
  const ctx = canvas?.getContext("2d");
  const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const fixed = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";

  async function api(path, method = "GET", body, timeout = method === "GET" ? 900 : 7000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload.data;
    } finally {
      clearTimeout(timer);
    }
  }

  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  async function reloadWhenControlServiceReturns() {
    // Give the old process time to exit first. Polling too early could reload
    // the page from the instance that is about to be terminated.
    await delay(1200);
    while (Date.now() < state.resetPendingUntil) {
      try {
        await api("/api/health", "GET", undefined, 3000);
        window.location.reload();
        return;
      } catch (_) {
        await delay(400);
      }
    }
    state.resetPendingUntil = 0;
    $("maintenance-result").textContent = "\u5df2\u63d0\u4ea4\u91cd\u7f6e\uff0c\u8bf7\u5237\u65b0\u9875\u9762\u3002";
    phase("\u63a7\u5236\u670d\u52a1\u6062\u590d\u8d85\u65f6", true);
  }

  function phase(text, error = false) {
    for (const id of ["phase-result", "operator-phase-label"]) {
      const element = $(id);
      if (element) {
        element.textContent = text;
        element.style.color = error ? "var(--red)" : "var(--amber)";
      }
    }
  }

  function badge(id, text, tone = "neutral") {
    const element = $(id);
    if (element) {
      element.textContent = text;
      element.className = `badge ${tone}`;
    }
  }

  function session() {
    return state.teleop?.session || { state: "IDLE", id: null, client_id: null, mode: null, execution_mode: null, input_source: null };
  }

  const isShadowSession = (current) => (current.execution_mode || current.mode) === "shadow";

  function canSendNonzero(allowHoldReadyResume = false) {
    const current = session();
    const diagnostic = state.teleop?.diagnostics || {};
    const shadow = isShadowSession(current);
    if (allowHoldReadyResume && diagnostic.trajectory_state === "HOLD_READY" &&
        current.state === "ACTIVE" && current.client_id === clientId) {
      // Let the backend resynchronise Ruckig and restore TRACKING from the
      // current client's fresh deadman packet.
      return true;
    }
    const common = current.state === "ACTIVE" &&
      current.client_id === clientId &&
      state.teleop?.execution?.accepting_targets === true;
    if (!common) {
      return false;
    }
    return (shadow || state.teleop?.authority?.servo_mode === "TRACKING") &&
      diagnostic.trajectory_state === "RUNNING";
  }

  function permissionReason() {
    const current = session();
    if (current.state !== "ACTIVE") return "会话未处于 ACTIVE";
    if (current.client_id !== clientId) return "会话属于其他客户端";
    if (state.teleop?.diagnostics?.trajectory_state === "FAULT") return `FAULT · ${state.teleop.diagnostics.trajectory_brake_reason || "轨迹故障"}`;
    if (state.teleop?.execution?.accepting_targets !== true) return "等待 OSC 输入权限";
    if (!isShadowSession(current) && state.teleop?.authority?.servo_mode !== "TRACKING") return "等待进入 TRACKING";
    if (state.teleop?.diagnostics?.trajectory_state !== "RUNNING") return "等待遥操轨迹运行";
    return "";
  }

  function resetWebAdapter() {
    state.webAdapterActive = false;
    state.oscAnchor = null;
    state.intentPending = false;
    resetInput(false);
  }

  function nextOscSequence() {
    // OSC requires a strictly increasing sequence, not consecutive integers.
    // Use the local clock as a fresh-session floor so an in-flight response
    // or a delayed status snapshot can never roll this browser backward.
    state.oscSequence = Math.max(state.oscSequence + 1, Date.now());
    return state.oscSequence;
  }

  async function reanchorWebAdapter() {
    const current = session();
    if (current.state !== "ACTIVE" || current.client_id !== clientId) return phase(`WebAdapter 输入已拦截：${permissionReason()}`, true);
    const target = state.teleop?.command?.target_tcp;
    if (!target?.position_m || !target?.orientation_xyzw) return phase("OSC 尚未提供当前 TCP 位姿", true);
    state.oscAnchor = { position_m: [...target.position_m], orientation_xyzw: [...target.orientation_xyzw] };
    state.webAdapterActive = true;
    resetInput(false);
    state.lastPoseTick = performance.now();
    updateInputView(); render();
  }

  function velocity() {
    const axis = (positive, negative) => (state.keys.has(positive) ? 0.5 : 0) - (state.keys.has(negative) ? 0.5 : 0);
    const result = [state.xy[0], state.xy[1], axis("KeyW", "KeyS"), axis("KeyE", "KeyQ"), axis("KeyF", "KeyR"), axis("KeyD", "KeyA")];
    if (state.sticks.get("right")?.pointerId != null) {
      if (state.rightMode === "zy") {
        result[2] = -state.right[1];
        result[5] = state.right[0];
      } else {
        result[3] = state.right[0];
        result[4] = -state.right[1];
      }
    }
    return result.map((item) => Math.max(-1, Math.min(1, finite(item))));
  }

  function quatMultiply(a, b) {
    const [x1, y1, z1, w1] = a; const [x2, y2, z2, w2] = b;
    return [w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2];
  }

  function quatFromEuler(roll, pitch, yaw) {
    const cr=Math.cos(roll/2), sr=Math.sin(roll/2), cp=Math.cos(pitch/2), sp=Math.sin(pitch/2), cy=Math.cos(yaw/2), sy=Math.sin(yaw/2);
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy];
  }

  function eulerFromQuat([x, y, z, w]) {
    const roll=Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));
    const pitch=Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x))));
    const yaw=Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
    return [roll,pitch,yaw];
  }

  function integrateRelativePose(dt) {
    if (!state.webAdapterActive) return;
    const axes = velocity(); const scale = Number($("scale")?.value || 1);
    const p = state.relativePose.position_m;
    for (let i=0; i<3; i+=1) p[i] += axes[i] * 0.06 * dt;
    const increment = quatFromEuler(axes[3]*0.45*dt, axes[4]*0.45*dt, axes[5]*0.45*dt);
    state.relativePose.orientation_xyzw = quatMultiply(state.relativePose.orientation_xyzw, increment);
    const norm=Math.hypot(...state.relativePose.orientation_xyzw);
    state.relativePose.orientation_xyzw = state.relativePose.orientation_xyzw.map((value)=>value/norm);
    void scale;
  }

  function resetInput(sendZero = false) {
    for (const entry of state.sticks.values()) {
      entry.pointerId = null;
      entry.knob.style.transform = "translate(-50%, -50%)";
    }
    state.xy = [0, 0];
    state.right = [0, 0];
    state.keys.clear();
    state.relativePose = { position_m: [0, 0, 0], orientation_xyzw: [0, 0, 0, 1] };
    updateInputView();
    if (sendZero) requestIntent(true);
  }

  function updateInputView() {
    const names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"];
    const intentLabel = document.querySelector(".intent-readout > span");
    if (intentLabel) intentLabel.textContent = "WebAdapter 相对输入 ΔPose";
    const output = $("input-readout");
    const values = [...state.relativePose.position_m, ...eulerFromQuat(state.relativePose.orientation_xyzw)];
    if (output) output.textContent = values.map((value, index) => `${names[index]} ${value.toFixed(3)}`).join(" · ");
  }

  function updateRightLabels() {
    const element = $("right-label");
    if (!element) return;
    element.innerHTML = state.rightMode === "zy"
      ? "<span>Yaw−</span><span>Z+</span><span>Yaw+</span><span>Z−</span>"
      : "<span>Roll−</span><span>Pitch+</span><span>Roll+</span><span>Pitch−</span>";
  }

  function backendPhase() {
    const transport = state.teleop?.transport || {};
    const broker = state.teleop?.authority || {};
    const teleop = state.teleop || {};
    const action = teleop.active_action;
    if (broker.hardware_mode === "FAULT" || broker.arm_writer === "SAFETY" || broker.safety_state === "FAULT" || teleop.diagnostics?.trajectory_state === "FAULT") {
      return `FAULT · ${transport.reason || broker.reason || teleop.diagnostics?.trajectory_brake_reason || "安全停车"}`;
    }
    if (broker.arm_writer === "MODE_TRANSITION") return "MODE_TRANSITION";
    if (action?.type) return `${String(action.type).toUpperCase()} · 执行中`;
    if (broker.servo_mode) return broker.servo_mode;
    if (teleop.session?.state === "ACTIVE") return teleop.execution?.accepting_targets ? "ACTIVE · OSC 已就绪" : "ACTIVE · 等待 OSC";
    return broker.hardware_mode || "IDLE";
  }

  async function sendIntent() {
    const current = session();
    if (current.state !== "ACTIVE" || current.client_id !== clientId) return;
    if (!state.webAdapterActive) return;
    const anchor = state.oscAnchor || state.teleop?.command?.target_tcp;
    if (!anchor?.position_m || !anchor?.orientation_xyzw) return;
    const sequence = nextOscSequence();
    const targetPose = {
      position_m: anchor.position_m.map((value, index) => value + state.relativePose.position_m[index]),
      orientation_xyzw: quatMultiply(anchor.orientation_xyzw, state.relativePose.orientation_xyzw),
    };
    const result = await api("/api/osc/command", "POST", {
      session_id: current.id,
      client_id: clientId,
      sequence,
      type: "track_tcp",
      acknowledgement_only: true,
      payload: { target_pose: targetPose },
    }, 3000);
    if (result?.state) {
      const resultSequence = Number(result.state.state_sequence || result.state.session?.sequence || 0);
      if (resultSequence >= state.latestSequence) {
        state.latestSequence = resultSequence;
        state.teleop = result.state;
      }
    }
    if (!result?.ok) {
      const acceptedSequence = Number(result?.result?.accepted_sequence);
      if (Number.isFinite(acceptedSequence)) state.oscSequence = Math.max(state.oscSequence, acceptedSequence);
      const reason = result?.result?.reason || result?.state?.diagnostics?.trajectory_brake_reason || "OSC 未接受该目标";
      throw new Error(`OSC 拒绝：${reason}`);
    }
    if (result?.result?.accepted_sequence != null && state.teleop?.command) {
      state.oscSequence = Math.max(state.oscSequence, Number(result.result.accepted_sequence));
      state.teleop.command.sequence = result.result.accepted_sequence;
    }
  }

  function requestIntent() {
    if (state.intentBusy) {
      state.intentPending = true;
      return;
    }
    state.intentBusy = true;
    sendIntent().catch((error) => {
      if (state.webAdapterActive) phase(`意图发送失败：${error.message}`, true);
    }).finally(() => {
      state.intentBusy = false;
      if (state.intentPending) {
        state.intentPending = false;
        requestIntent();
      }
    });
  }

  function attachStick(name) {
    const element = $(`${name}-stick`);
    const knob = $(`${name}-knob`);
    if (!element || !knob) return;
    const entry = { element, knob, pointerId: null };
    state.sticks.set(name, entry);
    const update = (event) => {
      const rect = element.getBoundingClientRect();
      let x = (event.clientX - rect.left) / rect.width * 2 - 1;
      let y = (event.clientY - rect.top) / rect.height * 2 - 1;
      const length = Math.hypot(x, y);
      if (length > 1) { x /= length; y /= length; }
      if (name === "xy") state.xy = [x, y]; else state.right = [x, y];
      knob.style.transform = `translate(calc(-50% + ${x * 68}px), calc(-50% + ${y * 68}px))`;
      updateInputView();
      requestIntent();
    };
    element.addEventListener("pointerdown", (event) => {
      if (entry.pointerId != null) return;
      event.preventDefault();
      entry.pointerId = event.pointerId;
      element.setPointerCapture?.(event.pointerId);
      update(event);
    });
    element.addEventListener("pointermove", (event) => {
      if (event.pointerId === entry.pointerId) { event.preventDefault(); update(event); }
    });
    const release = (event) => {
      if (event.pointerId !== entry.pointerId) return;
      entry.pointerId = null;
      if (name === "xy") state.xy = [0, 0]; else state.right = [0, 0];
      knob.style.transform = "translate(-50%, -50%)";
      updateInputView();
      requestIntent();
    };
    element.addEventListener("pointerup", release);
    element.addEventListener("pointercancel", release);
    element.addEventListener("lostpointercapture", release);
  }

  function rotationFromQuat(orientation) {
    if (!Array.isArray(orientation) || orientation.length !== 4) return null;
    const values = orientation.map(Number);
    if (!values.every(Number.isFinite)) return null;
    const [x, y, z, w] = values;
    const norm = Math.hypot(x, y, z, w);
    if (!Number.isFinite(norm) || norm < 1e-9) return null;
    const nx = x / norm;
    const ny = y / norm;
    const nz = z / norm;
    const nw = w / norm;
    return [
      [1 - 2 * (ny * ny + nz * nz), 2 * (nx * ny - nz * nw), 2 * (nx * nz + ny * nw)],
      [2 * (nx * ny + nz * nw), 1 - 2 * (nx * nx + nz * nz), 2 * (ny * nz - nx * nw)],
      [2 * (nx * nz - ny * nw), 2 * (ny * nz + nx * nw), 1 - 2 * (nx * nx + ny * ny)],
    ];
  }

  function drawWorkspace(teleop) {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const tcp = teleop?.diagnostics?.pink?.tcp || {};
    const links = Array.isArray(tcp.link_positions_m) ? tcp.link_positions_m : [];
    const currentPose = teleop?.execution?.current_tcp_pose || null;
    const position = Array.isArray(currentPose?.position_m) && currentPose.position_m.length === 3 && currentPose.position_m.every((value) => Number.isFinite(Number(value)))
      ? currentPose.position_m.map(Number)
      : (Array.isArray(tcp.position_m) && tcp.position_m.length === 3 && tcp.position_m.every((value) => Number.isFinite(Number(value))) ? tcp.position_m.map(Number) : null);
    const targetPose = teleop?.command?.target_tcp || null;
    const targetPosition = Array.isArray(targetPose?.position_m) && targetPose.position_m.length === 3 && targetPose.position_m.every((value) => Number.isFinite(Number(value)))
      ? targetPose.position_m.map(Number) : null;
    const targetRotation = rotationFromQuat(targetPose?.orientation_xyzw);
    const workspace = teleop?.workspace || {};
    const min = Array.isArray(workspace.min_xyz_m) ? workspace.min_xyz_m.map(Number) : [-0.45, -0.15, -0.02];
    const max = Array.isArray(workspace.max_xyz_m) ? workspace.max_xyz_m.map(Number) : [0.45, 0.60, 0.70];
    const minZ = Number.isFinite(Number(workspace.min_flange_z_m)) ? Number(workspace.min_flange_z_m) : min[2];
    const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - minZ, 0.1);
    const scale = Math.min(canvas.width, canvas.height) * 0.72 / span;
    const project = ([x, y, z]) => ({
      x: canvas.width * 0.53 + (Number(x) - Number(y)) * scale * 0.72,
      y: canvas.height * 0.84 - (Number(z) - minZ) * scale * 0.88 - (Number(x) + Number(y)) * scale * 0.28,
    });
    const line = (a, b, color, width = 1, dash = []) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.restore();
    };
    const polygon = (points, fill, stroke) => {
      ctx.save(); ctx.beginPath(); points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
    };
    const plane = [[min[0], min[1], minZ], [max[0], min[1], minZ], [max[0], max[1], minZ], [min[0], max[1], minZ]].map(project);
    polygon(plane, "rgba(111, 150, 158, .12)", "rgba(126, 171, 177, .55)");
    for (let i = 1; i < 5; i += 1) {
      const x = min[0] + (max[0] - min[0]) * i / 5;
      const y = min[1] + (max[1] - min[1]) * i / 5;
      line(project([x, min[1], minZ]), project([x, max[1], minZ]), "rgba(126,171,177,.18)");
      line(project([min[0], y, minZ]), project([max[0], y, minZ]), "rgba(126,171,177,.18)");
    }
    const origin = project([0, 0, minZ]);
    line(origin, project([0.08, 0, minZ]), "#ff817a", 2);
    line(origin, project([0, 0.08, minZ]), "#59d9a2", 2);
    line(origin, project([0, 0, minZ + 0.08]), "#78bdf0", 2);
    ctx.save(); ctx.font = "11px ui-monospace, Consolas, monospace"; ctx.fillStyle = "#8ba09e"; ctx.fillText(`最低高度面  Z=${minZ.toFixed(3)} m`, 16, 22); ctx.restore();
    for (let index = 1; index < links.length; index += 1) line(project(links[index - 1]), project(links[index]), index === links.length - 1 ? "#e7eef3" : "#71878b", index === links.length - 1 ? 6 : 5);
    if (targetPosition) {
      const targetCenter = project(targetPosition);
      if (position) line(project(position), targetCenter, "rgba(240, 197, 106, .85)", 2, [7, 5]);
      ctx.save();
      ctx.strokeStyle = "#f0c56a";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(targetCenter.x, targetCenter.y, 10, 0, Math.PI * 2); ctx.stroke();
      ctx.font = "bold 11px ui-monospace, Consolas, monospace";
      ctx.fillStyle = "#f0c56a";
      ctx.fillText("T_target", targetCenter.x + 13, targetCenter.y + 15);
      ctx.restore();
      if (targetRotation) {
        const targetAxisColors = ["#ff817a", "#59d9a2", "#78bdf0"];
        for (let axis = 0; axis < 3; axis += 1) {
          const endpoint = [
            targetPosition[0] + Number(targetRotation[0][axis]) * 0.07,
            targetPosition[1] + Number(targetRotation[1][axis]) * 0.07,
            targetPosition[2] + Number(targetRotation[2][axis]) * 0.07,
          ];
          line(targetCenter, project(endpoint), targetAxisColors[axis], 2, [6, 4]);
        }
      }
    }
    ctx.save();
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.fillStyle = "#8ba09e";
    ctx.fillText("\u25cf \u5b9e\u9645 TCP", 16, canvas.height - 28);
    ctx.fillStyle = "#f0c56a";
    ctx.fillText("\u25cb \u76ee\u6807 T_ref", 16, canvas.height - 12);
    ctx.restore();
    if (!position) return;
    const center = project(position);
    ctx.save(); ctx.fillStyle = "#59d9a2"; ctx.beginPath(); ctx.arc(center.x, center.y, 8, 0, Math.PI * 2); ctx.fill(); ctx.font = "11px ui-monospace, Consolas, monospace"; ctx.fillStyle = "#e7eef3"; ctx.fillText("TCP", center.x + 12, center.y - 10); ctx.restore();
    const rotation = rotationFromQuat(currentPose?.orientation_xyzw) || (Array.isArray(tcp.rotation) && tcp.rotation.length === 3 ? tcp.rotation : null);
    const axisColors = ["#ff817a", "#59d9a2", "#78bdf0"];
    const axisNames = ["X", "Y", "Z"];
    if (rotation && rotation.every(row => Array.isArray(row) && row.length === 3 && row.every(Number.isFinite))) {
      for (let axis = 0; axis < 3; axis += 1) {
        const endpoint = [position[0] + Number(rotation[0][axis]) * 0.07, position[1] + Number(rotation[1][axis]) * 0.07, position[2] + Number(rotation[2][axis]) * 0.07];
        const projected = project(endpoint); line(center, projected, axisColors[axis], 3);
        ctx.save(); ctx.font = "bold 11px ui-monospace, Consolas, monospace"; ctx.fillStyle = axisColors[axis]; ctx.fillText(axisNames[axis], projected.x + 4, projected.y - 4); ctx.restore();
      }
    }
  }

  function render() {
    const teleop = state.teleop || {};
    const broker = teleop.authority || {};
    const transport = teleop.transport || {};
    const command = teleop.command || {};
    const execution = teleop.execution || {};
    const robot = teleop.robot || {};
    const current = session();
    const diagnostic = teleop.diagnostics || {};
    const timing = diagnostic.timing || {};
    // A successful /api/status response already proves that the service is
    // online. This flag describes only the USB-CAN robot transport.
    badge("connection", transport.connected ? "\u786c\u4ef6\u5df2\u8fde\u63a5" : "\u786c\u4ef6\u672a\u8fde\u63a5", transport.connected ? "ok" : "warn");
    const modeLabel = current.state === "ACTIVE" ? `WebAdapter + ${isShadowSession(current) ? "影子" : "真机"}` : "未接入";
    badge("teleop-mode", modeLabel, current.state === "ACTIVE" && isShadowSession(current) ? "ok" : current.state === "ACTIVE" ? "warn" : "neutral");
    badge("session-state", current.state || "IDLE", current.state === "ACTIVE" ? "ok" : "neutral");
    badge("feedback-state", Number.isFinite(finite(timing.feedback_age_s, NaN)) ? `反馈 ${Math.round(timing.feedback_age_s * 1000)} ms` : "反馈 --", transport.can_health?.ok ? "ok" : "warn");
    badge("safety-state", `安全 ${broker.safety_state || diagnostic.trajectory_state || "--"}`, broker.safety_state === "FAULT" ? "fault" : "neutral");
    badge("writer-state", `Writer ${broker.arm_writer || "--"}`, broker.arm_writer === "SERVO" ? "ok" : "neutral");
    badge("servo-state", `Servo ${broker.servo_mode || "--"}`, broker.servo_mode === "TRACKING" ? "ok" : "neutral");
    const inputStatus = execution.accepting_targets ? "WebAdapter 已就绪" : current.state === "ACTIVE" ? "等待 OSC 输入权限" : "WebAdapter 未接入";
    const livePhase = backendPhase();
    phase(state.last_error || livePhase || inputStatus, Boolean(state.last_error));
    $("solver-badge").textContent = teleop.solver?.running ? "求解器在线" : "求解器空闲";
    $("solver-badge").className = `badge ${teleop.solver?.running ? "ok" : "neutral"}`;
    $("status-age").textContent = Number.isFinite(finite(timing.feedback_age_s, NaN)) ? `反馈 ${Math.round(timing.feedback_age_s * 1000)} ms` : "反馈 --";
    $("control-mode").textContent = `${broker.hardware_mode || "DISCONNECTED"} · ${broker.control_role || "NONE"}`;
    $("controller-mode").textContent = robot.arm_status?.ctrl_mode ?? "--";
    $("active-action").textContent = teleop.active_action?.type || "无";
    $("reason").textContent = transport.reason || diagnostic.safety_gate?.reason || diagnostic.trajectory_brake_reason || "--";
    const grip = teleop.gripper || {};
    $("gripper-width-value").textContent = grip.width_m == null ? "--" : `${fixed(grip.width_m * 1000, 1)} mm`;
    $("gripper-force").textContent = grip.force_n == null ? "--" : `${fixed(grip.force_n, 2)} N`;
    $("gripper-driver").textContent = grip.status?.foc_status?.driver_enable_status ? "已使能" : "--";
    const formatPose = (pose) => {
      if (!Array.isArray(pose) || pose.length !== 6 || !pose.every((value) => Number.isFinite(Number(value)))) return "--";
      const values = pose.map(Number);
      return `位置 [${values.slice(0, 3).map((value) => `${value.toFixed(4)} m`).join(", ")}] · 姿态 [${values.slice(3).map((value) => `${value.toFixed(4)} rad`).join(", ")}]`;
    };
    $("tcp-pose-values").textContent = formatPose(robot.tcp_pose);
    const formatAbsolutePose = (pose) => {
      const position = pose?.position_m;
      const orientation = pose?.orientation_xyzw;
      if (!Array.isArray(position) || position.length !== 3 || !Array.isArray(orientation) || orientation.length !== 4) return "--";
      const rpy = eulerFromQuat(orientation).map((value) => `${value.toFixed(3)} rad`);
      return `位置 [${position.map((value) => `${Number(value).toFixed(3)} m`).join(", ")}] · RPY [${rpy.join(", ")}]`;
    };
    const referenceLine = $("tcp-reference-readout");
    if (referenceLine) referenceLine.textContent = `当前 TCP：${formatAbsolutePose(execution.current_tcp_pose)} · 目标 TCP：${formatAbsolutePose(command.target_tcp)}`;
    const joints = robot.joint_angles_rad || [];
    $("joints").innerHTML = joints.map((value, index) => `<div><span>J${index + 1}</span><strong>${fixed(value * 180 / Math.PI, 2)}°</strong></div>`).join("") || "<span class='muted'>无关节反馈</span>";
    $("input-age").textContent = Number.isFinite(finite(current.last_input_age_s, NaN)) ? `${fixed(current.last_input_age_s, 2)} s` : "--";
    $("loop-rate").textContent = String(diagnostic.loop_count ?? "--");
    $("cpv-count").textContent = String(execution.output_count ?? 0);
    if ($("cpv-count").previousElementSibling) $("cpv-count").previousElementSibling.textContent = "OSC 输出批次";
    $("condition").textContent = Number.isFinite(finite(diagnostic.pink?.condition_number, NaN)) ? fixed(diagnostic.pink.condition_number, 1) : "--";
    $("period-ms").textContent = Number.isFinite(finite(timing.actual_dt_s, NaN)) ? `${fixed(timing.actual_dt_s * 1000, 1)} ms` : "--";
    $("solver-age").textContent = Number.isFinite(finite(timing.solver_age_s, NaN)) ? `${fixed(timing.solver_age_s * 1000, 0)} ms` : "--";
    $("feedback-age").textContent = Number.isFinite(finite(timing.feedback_age_s, NaN)) ? `${fixed(timing.feedback_age_s * 1000, 0)} ms` : "--";
    const tcpError = diagnostic.tcp_error || {};
    $("tcp-position-error").textContent = Number.isFinite(finite(tcpError.position_norm_m, NaN)) ? `${fixed(tcpError.position_norm_m * 1000, 1)} mm` : "--";
    $("tcp-orientation-error").textContent = Number.isFinite(finite(tcpError.orientation_angle_rad, NaN)) ? `${fixed(tcpError.orientation_angle_rad * 180 / Math.PI, 1)}°` : "--";
    $("gate-state").textContent = timing.gate_limited === true ? "限速" : timing.gate_ok === true ? "通过" : timing.gate_ok === false ? "拒绝" : "--";
    $("trajectory-state").textContent = diagnostic.trajectory_state || "--";
    const active = current.state === "ACTIVE";
    // A visibility change or a recoverable request error deliberately clears
    // only this browser's adapter state.  The OSC session may still be
    // ACTIVE and owned by this client, so keep a path to re-anchor/reconnect
    // instead of stranding the UI behind a disabled button.
    $("start").disabled = current.state === "STARTING" || (active && state.webAdapterActive);
    $("start").textContent = active && !state.webAdapterActive ? "重新接入 WebAdapter" : "接入 WebAdapter";
    $("stop").disabled = !active;
    if ($("reanchor")) {
      $("reanchor").disabled = !active || diagnostic.trajectory_state === "BRAKING" || diagnostic.trajectory_state === "FAULT";
      $("reanchor").textContent = state.webAdapterActive ? "重锚定 WebAdapter" : "初始化 WebAdapter";
    }
    const picoLine = $("pico-connection");
    if (picoLine) picoLine.textContent = "WebAdapter 在浏览器内将摇杆增量转换为基座系绝对 TCP 目标，并只调用 OSC track_tcp。";
    $("hold").disabled = !transport.connected;
    $("freedrive").disabled = !transport.connected;
    // Reset is served by the independent localhost watchdog and sends no
    // robot command. It must remain available precisely when the backend is
    // disconnected, initializing, or stuck.
    $("reset-control").disabled = Date.now() < state.resetPendingUntil;
    drawWorkspace(teleop);
    renderHierarchy(teleop, broker, transport, robot, diagnostic, current);
  }

  async function refresh() {
    if (state.refreshBusy) return;
    state.refreshBusy = true;
    const generation = state.requestGeneration;
    const resetPending = Date.now() < state.resetPendingUntil;
    const statusTimeout = resetPending ? 5000 : 900;
    try {
      const osc = await api("/api/osc/state", "GET", undefined, statusTimeout);
      if (generation !== state.requestGeneration) return;
      const sequence = Number(osc.state_sequence || osc.session?.sequence || 0);
      if (sequence < state.latestSequence) return;
      state.latestSequence = sequence;
      state.teleop = osc;
      if (resetPending) {
        state.resetPendingUntil = 0;
        $("maintenance-result").textContent = "\u63a7\u5236\u670d\u52a1\u5df2\u6062\u590d\u3002";
      }
      render();
    } catch (error) {
      if (Date.now() < state.resetPendingUntil) {
        phase("控制服务正在重启…");
        return;
      }
      if (generation === state.requestGeneration) phase(`服务不可用：${error.message}`, true);
    } finally {
      state.refreshBusy = false;
    }
  }

  async function connectWebAdapter() {
    state.requestGeneration += 1;
    resetWebAdapter();
    $("start").disabled = true;
    phase("正在接入 WebAdapter…");
    const executionMode = $("execution-mode").value;
    try {
      const result = await api("/api/osc/session/start", "POST", { execution_mode: executionMode, client_id: clientId }, 10000);
      const osc = result.state;
      const sequence = Number(osc?.state_sequence || osc?.session?.sequence || 0);
      // A control-service restart resets state_sequence to zero.  This is a
      // fresh authoritative response to the user's explicit connect action,
      // so it must replace any snapshot from the previous process instance.
      state.latestSequence = sequence;
      state.teleop = osc;
      state.oscSequence = Math.max(state.oscSequence, Number(osc.command?.sequence || osc.session?.sequence || 0));
      if (osc?.session?.state === "ACTIVE") {
        await reanchorWebAdapter();
        phase("WebAdapter 已接入 OSC", false);
      } else {
        phase("WebAdapter 未接入 OSC", true);
      }
      render();
      await refresh();
    } catch (error) {
      phase(`WebAdapter 接入失败：${error.message}`, true);
      await refresh();
    }
  }

  async function disconnectWebAdapter() {
    state.requestGeneration += 1;
    resetWebAdapter();
    try { state.teleop = (await api("/api/osc/session/stop", "POST", { reason: "WebAdapter disconnected" })).state; render(); } catch (error) { phase(`WebAdapter 断开失败：${error.message}`, true); }
  }

  function stopWebAdapterForPageExit(reason) {
    const current = session();
    if (current.state === "ACTIVE" && current.client_id === clientId && current.id) {
      const payload = JSON.stringify({ reason });
      const body = new Blob([payload], { type: "application/json" });
      // Page lifecycle handlers cannot rely on pending promises completing.
      // sendBeacon keeps the OSC HOLD request alive while the document is
      // being hidden or discarded, so a hidden tab never leaves a live arm
      // session that its throttled timers can no longer maintain.
      if (!navigator.sendBeacon("/api/osc/session/stop", body)) {
        void api("/api/osc/session/stop", "POST", { reason }, 1500).catch(() => {});
      }
    }
    resetWebAdapter();
  }

  async function heartbeat() {
    const current = session();
    if (state.heartbeatBusy || current.state !== "ACTIVE" || current.client_id !== clientId || !current.id) return;
    state.heartbeatBusy = true;
    try {
      const result = await api("/api/osc/session/heartbeat", "POST", {
        session_id: current.id,
        client_id: clientId,
      }, 3000);
      const osc = result?.state;
      if (!osc) return;
      const sequence = Number(osc.state_sequence || osc.session?.sequence || 0);
      if (sequence >= state.latestSequence) {
        state.latestSequence = sequence;
        state.teleop = osc;
        state.oscSequence = Number(osc.command?.sequence || osc.session?.sequence || 0);
        render();
      }
      state.heartbeatFailures = 0;
    } catch (error) {
      phase(`OSC 会话心跳失败：${error.message}`, true);
      state.heartbeatFailures += 1;
      if (state.heartbeatFailures >= 2) {
        // Do not leave an ACTIVE backend session after this browser has lost
        // the ability to renew its ownership lease.  The endpoint performs
        // the official CPV-to-Follower/HOLD handoff and returns a fresh UI
        // snapshot that makes reconnect immediately available.
        resetWebAdapter();
        try {
          const result = await api("/api/osc/session/stop", "POST", { reason: "WebAdapter heartbeat lost" }, 3000);
          state.teleop = result.state;
          state.latestSequence = Number(result.state?.state_sequence || 0);
          render();
        } catch (stopError) {
          phase(`WebAdapter 安全断开失败：${stopError.message}`, true);
          void refresh();
        }
      }
    } finally {
      state.heartbeatBusy = false;
    }
  }

  function renderHierarchy(teleop, broker, transport, robot, diagnostic, current) {
    const cards = document.querySelectorAll(".layer-card");
    if (cards.length >= 4) {
      const copy = [
        ["CONTROL SUPERVISOR", "控制监督", "会话 · 运行模式 · 安全 · 反馈"],
        ["ARM WRITE AUTHORITY", "唯一写入权", "当前谁可以写入 ARM"],
        ["CONTINUOUS MOTION", "连续运动", "伺服阶段 · 实际执行通道"],
        ["NERO BACKEND", "NERO 后端", "硬件模式 · 控制角色 · 反馈"],
      ];
      cards.forEach((card, index) => {
        const [label, title, note] = copy[index];
        const name = card.querySelector(".layer-name");
        const heading = card.querySelector("h2");
        const paragraph = card.querySelector(".layer-copy p:last-child");
        if (name) name.textContent = label;
        if (heading) heading.textContent = title;
        if (paragraph) paragraph.textContent = note;
      });
    }
    const sessionText = current.state ? `${current.state} · ${current.mode || "--"}` : "IDLE";
    const phase = $("phase-result");
    if (phase) phase.textContent = sessionText;
    const writer = $("writer-state");
    if (writer) writer.textContent = broker.arm_writer || "NONE";
    const epoch = $("servo-state");
    if (epoch) {
      const label = epoch.parentElement?.querySelector("span");
      if (label) label.textContent = "Epoch";
      epoch.textContent = broker.control_epoch == null ? "--" : String(broker.control_epoch);
    }
    const motion = $("solver-badge");
    if (motion) {
      motion.textContent = `${broker.servo_mode || "SUSPENDED"} · ${broker.command_stream || "NONE"}`;
      motion.className = `badge ${broker.servo_mode === "TRACKING" ? "ok" : "neutral"}`;
    }
    const backend = $("control-mode");
    if (backend) backend.textContent = `${broker.hardware_mode || "DISCONNECTED"} · ${broker.control_role || "NONE"}`;
    const modeLabel = cards[3]?.querySelector(".layer-value span");
    if (modeLabel) modeLabel.textContent = "Hardware / Role";
    const authorityLabel = cards[1]?.querySelector(".layer-value span");
    if (authorityLabel) authorityLabel.textContent = "Writer / Epoch";
    const motionLabel = cards[2]?.querySelector(".layer-value span");
    if (motionLabel) motionLabel.textContent = "Servo / Stream";
    void teleop; void transport; void robot; void diagnostic;
  }

  function permissionReason() {
    const current = session();
    const broker = state.teleop?.authority || {};
    const diagnostic = state.teleop?.diagnostics || {};
    if (current.state !== "ACTIVE") return "会话未处于 ACTIVE";
    if (current.client_id !== clientId) return "会话属于其他客户端";
    if (broker.hardware_mode === "FAULT" || diagnostic.trajectory_state === "FAULT") return `FAULT · ${broker.reason || diagnostic.trajectory_brake_reason || "轨迹故障"}`;
    if (state.teleop?.execution?.accepting_targets !== true) return "后端未开启 OSC 输入权限";
    if (!isShadowSession(current) && broker.servo_mode !== "TRACKING") return `等待进入 ${broker.servo_mode || "TRACKING"}`;
    if (diagnostic.trajectory_state !== "RUNNING") return `等待轨迹恢复（${diagnostic.trajectory_state || "unknown"}）`;
    return "";
  }

  attachStick("xy");
  attachStick("right");
  $("execution-mode")?.addEventListener("change", () => { resetWebAdapter(); render(); });
  $("right-mode")?.addEventListener("change", () => { state.rightMode = $("right-mode").value; resetInput(true); updateRightLabels(); });
  $("scale")?.addEventListener("input", () => { $("scale-value").textContent = `${Math.round(Number($("scale").value) * 100)}%`; });
  window.addEventListener("keydown", (event) => { if (!keys.has(event.code) || event.repeat || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return; event.preventDefault(); state.keys.add(event.code); updateInputView(); });
  window.addEventListener("keyup", (event) => { if (!keys.has(event.code)) return; event.preventDefault(); state.keys.delete(event.code); updateInputView(); });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") stopWebAdapterForPageExit("WebAdapter page hidden");
  });
  window.addEventListener("pagehide", () => stopWebAdapterForPageExit("WebAdapter page unload"));
  $("start").onclick = connectWebAdapter;
  $("stop").onclick = disconnectWebAdapter;
  $("reanchor")?.addEventListener("click", () => reanchorWebAdapter().catch((error) => phase(`WebAdapter 重锚定失败：${error.message}`, true)));
  const oscCommand = (type, payload = {}) => {
    const current = session();
    const sequence = nextOscSequence();
    return api("/api/osc/command", "POST", { session_id: current.id, client_id: clientId, sequence, type, payload });
  };
  $("hold").onclick = () => { resetWebAdapter(); $("hold").disabled = true; oscCommand("hold", { reason: "operator requested HOLD" }).then((result) => { state.teleop = result.state; render(); }).catch((error) => phase(`HOLD失败：${error.message}`, true)); };
  $("freedrive").onclick = () => {
    // Clear the local UI only.  FREEDRIVE must not first submit a zero
    // teleop intent, because that would start a P1/Ruckig braking path before
    // the dedicated direct Leader transition reaches the backend.
    resetWebAdapter();
    $("freedrive").disabled = true;
    oscCommand("freedrive", { reason: "operator requested FREEDRIVE" })
      .then((result) => { state.teleop = result.state; render(); })
      .catch((error) => phase(`FREEDRIVE失败：${error.message}`, true));
  };
  $("gripper-open").onclick = () => oscCommand("gripper", { mode: "open", force_n: Number($("gripper-force-input").value) });
  $("gripper-grip").onclick = () => oscCommand("gripper", { mode: "grip", force_n: Number($("gripper-force-input").value) });
  $("gripper-position").onclick = () => oscCommand("gripper", { mode: "position", width_m: Number($("gripper-width-input").value) / 1000, force_n: Number($("gripper-force-input").value) });
  $("reset-control").onclick = async () => {
    try {
      state.resetPendingUntil = Date.now() + 10000;
      $("reset-control").disabled = true;
      $("maintenance-result").textContent = "\u6b63\u5728\u8bf7\u6c42\u670d\u52a1\u91cd\u7f6e\u2026";
      try {
        // Prefer the independent watchdog. If it has not started, the main
        // HTTP process can still launch the same external hard-reset helper.
        await api("http://127.0.0.1:8767/api/reset", "POST", {}, 1500);
      } catch (_) {
        await api("/api/control/reset", "POST", {}, 1500);
      }
      $("maintenance-result").textContent = "\u91cd\u7f6e\u5df2\u63d0\u4ea4\uff0c\u6b63\u5728\u7b49\u5f85\u670d\u52a1\u6062\u590d\u3002";
      phase("\u63a7\u5236\u670d\u52a1\u6b63\u5728\u91cd\u542f\u2026");
      void reloadWhenControlServiceReturns();
    } catch (error) {
      state.resetPendingUntil = 0;
      $("reset-control").disabled = false;
      $("maintenance-result").textContent = `\u91cd\u7f6e\u5931\u8d25\uff1a${error.message}`;
    }
  };
  updateRightLabels();
  updateInputView();
  refresh();
  setInterval(refresh, 500);
  setInterval(heartbeat, 1000);
  setInterval(() => {
    const now = performance.now();
    const dt = Math.min(0.05, Math.max(0, (now - state.lastPoseTick) / 1000));
    state.lastPoseTick = now;
    if (!state.webAdapterActive) return;
    // Reanchoring is local-only. A zeroed input must leave OSC in HOLD_READY;
    // only a non-zero adapter input starts or updates absolute TCP tracking.
    if (!velocity().some((value) => Math.abs(value) > 1e-6)) return;
    integrateRelativePose(dt);
    updateInputView();
    requestIntent();
  }, 20);
})();
