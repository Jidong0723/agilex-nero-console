// Execute the console's real functions against a fake DOM and HTTP transport.
// No running service, browser, SDK or CAN connection is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../web/console/app.js"), "utf8");
const startup = source.indexOf('  attachStick("xy");');
assert.ok(startup > 0, "console startup marker must exist");

function fixture(mode = "cpv") {
  const nodes = new Map();
  const element = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      value: id === "execution-mode" ? "hardware" : id === "input-adapter" ? "web" : "",
      style: {}, classList: { toggle(name, enabled) { this[name] = Boolean(enabled); }, add() {}, remove() {} },
      setAttribute() {}, replaceChildren() {}, remove() {}, closest() { return null; },
      getContext() { return null; },
    });
    return nodes.get(id);
  };
  const requests = [];
  const context = vm.createContext({
    document: { getElementById: element, querySelector: () => null, querySelectorAll: () => [] },
    sessionStorage: { getItem: () => "browser-test" },
    performance: { now: () => 0 }, crypto: {}, AbortController, setTimeout, clearTimeout,
    fetch: async (url, options) => {
      const body = options.body ? JSON.parse(options.body) : null;
      requests.push({ url, body });
      const data = await handler(url, body);
      return { ok: true, json: async () => ({ ok: true, data }) };
    },
  });
  vm.runInContext(source.slice(0, startup) + `
    const renderCollection = renderDataset;
    drawWorkspace = renderHierarchy = renderPi05 = renderPico = renderDataset = renderCameraControls = () => {};
    globalThis.consoleTest = { state, render, updateOscState, connectWebAdapter, disconnectWebAdapter,
      reanchorWebAdapter, sendIntent, heartbeat, refresh, selectOutputMode, resetWebAdapter,
      renderDataset: renderCollection, datasetPreview, startDataset };
  })();`, context);
  const app = context.consoleTest;
  let handler = async (url) => url === "/api/osc/state" ? app.state.osc : {};
  const snapshot = (active = true, sequence = 1, output = mode, id = "session-a") => ({
    state_sequence: sequence, output_mode: output,
    output_switch: { allowed: true, reason: null },
    session: active ? { id, state: "ACTIVE", client_id: "browser-test", execution_mode: "hardware" } : null,
    authority: { control_epoch: 1, servo_mode: "TRACKING", hardware_mode: "HOLD" },
    execution: { accepting_targets: active }, diagnostics: { trajectory_state: "HOLD_READY" },
    command: { target_tcp: { position_m: [0.1, 0.2, 0.3], orientation_xyzw: [0, 0, 0, 1] } },
  });
  app.updateOscState(snapshot());
  return { app, nodes, requests, snapshot, route: (fn) => { handler = fn; } };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

for (const mode of ["cpv", "impedance"]) {
  test(`${mode}: connect and joystick submit the same mode-neutral requests`, async () => {
    const f = fixture(mode);
    f.app.state.osc.session = null;
    f.route(async (url) => {
      if (url === "/api/osc/session/start") return { state: f.snapshot() };
      if (url === "/api/osc/command") return { ok: true, result: { accepted_sequence: 10 } };
      return url === "/api/osc/state" ? f.app.state.osc : {};
    });
    await f.app.connectWebAdapter();
    assert.equal(f.app.state.webAdapterActive, true);
    assert.equal(f.app.state.webAdapterSessionId, "session-a");
    f.app.state.relativePose.position_m[0] = 0.002;
    await f.app.sendIntent();
    assert.deepEqual(f.requests.find(r => r.url.endsWith("/start")).body,
      { execution_mode: "hardware", client_id: "browser-test", input_source: "web" });
    const command = f.requests.find(r => r.url === "/api/osc/command").body;
    assert.equal(command.type, "track_tcp");
    assert.equal(command.payload.target_pose.position_m[0], 0.10200000000000001);
    assert.equal("output_mode" in command, false);
  });

  test(`${mode}: same selection is a no-op, rejected selection preserves input`, async () => {
    const f = fixture(mode);
    await f.app.reanchorWebAdapter();
    const anchor = f.app.state.oscAnchor;
    await f.app.selectOutputMode(mode);
    assert.equal(f.requests.length, 0);
    f.route(async (url) => {
      if (url === "/api/osc/output-mode") throw new Error("HOLD required");
      return f.app.state.osc;
    });
    await f.app.selectOutputMode(mode === "cpv" ? "impedance" : "cpv");
    assert.equal(f.app.state.webAdapterActive, true);
    assert.equal(f.app.state.oscAnchor, anchor);
  });

  test(`${mode}: successful switch ends input through generic session state`, async () => {
    const f = fixture(mode);
    await f.app.reanchorWebAdapter();
    f.app.state.keys.add("KeyW");
    const next = mode === "cpv" ? "impedance" : "cpv";
    f.route(async () => ({ state: f.snapshot(false, 2, next) }));
    await f.app.selectOutputMode(next);
    assert.equal(f.app.state.osc.output_mode, next);
    assert.equal(f.app.state.webAdapterActive, false);
    assert.equal(f.app.state.oscAnchor, null);
    assert.equal(f.app.state.keys.size, 0);
    assert.deepEqual(f.requests.map(r => r.url), ["/api/osc/output-mode"]);
    assert.equal(f.nodes.get("start").disabled, false);
  });
}

test("backend availability alone enables/disables mode buttons", () => {
  const f = fixture();
  for (const allowed of [false, true]) {
    f.app.state.osc.output_switch.allowed = allowed;
    f.app.render();
    assert.equal(f.nodes.get("output-cpv").disabled, !allowed);
    assert.equal(f.nodes.get("output-impedance").disabled, !allowed);
  }
});

test("overview shows output mode while hardware feedback owns the age badge", () => {
  for (const mode of ["cpv", "impedance"]) {
    const f = fixture(mode);
    f.app.state.osc.transport = { can_health: { ok: true }, hardware_feedback: { feedback_age_s: .003 } };
    f.app.state.osc.execution.feedback_age_s = .080;
    f.app.state.osc.session.execution_mode = "shadow";
    f.app.render();
    assert.equal(f.nodes.get("output-mode-badge").textContent, mode === "cpv" ? "控制器 CPV" : "控制器 阻抗");
    assert.equal(f.nodes.get("status-age").textContent, "反馈 3 ms");
    assert.equal(f.nodes.get("status-age").className, "badge ok");
    f.app.state.osc.session.execution_mode = "hardware";
    f.app.render();
    assert.equal(f.nodes.get("status-age").className, "badge warn");
    f.app.state.osc.session = null;
    f.app.state.osc.transport.can_health.ok = false;
    f.app.render();
    assert.equal(f.nodes.get("status-age").className, "badge warn");
  }
});

test("mode actions live in OSC card and collection uses the shared bilingual heading", () => {
  const html = fs.readFileSync(path.join(__dirname, "../web/console/index.html"), "utf8");
  const oscCard = html.slice(html.indexOf('<section class="panel osc-card"'), html.indexOf('<p id="tcp-reference-readout"'));
  for (const id of ["output-cpv", "output-impedance", "output-mode-note"]) {
    assert.ok(oscCard.includes(`id="${id}"`));
    assert.equal(html.split(`id="${id}"`).length - 1, 1);
  }
  assert.equal(html.includes('id="feedback-state"'), false);
  assert.equal(html.includes('<section class="panel" aria-label="OSC 输出模式">'), false);
  assert.ok(source.includes('<p class="section-label">DATA COLLECTION</p><h2>数据采集</h2>'));
  assert.equal(source.includes("TCP–VLA 原始数据采集"), false);
});

test("session replacement and authority epoch invalidate anchors regardless of output mode", async () => {
  for (const mode of ["cpv", "impedance"]) {
    const f = fixture(mode);
    await f.app.reanchorWebAdapter();
    f.app.updateOscState(f.snapshot(true, 2, mode, "session-b"));
    f.app.render();
    assert.equal(f.app.state.webAdapterActive, false);
    await f.app.reanchorWebAdapter();
    f.app.state.osc.authority.control_epoch += 1;
    f.app.render();
    assert.equal(f.app.state.oscAnchor, null);
  }
});

test("late connect response cannot undo a disconnect; duplicate connect is suppressed", async () => {
  const f = fixture();
  const start = deferred();
  f.app.state.osc.session = null;
  f.route(async (url) => url.endsWith("/start") ? start.promise : { state: f.snapshot(false, 2) });
  const connecting = f.app.connectWebAdapter();
  await f.app.connectWebAdapter();
  f.app.render();
  assert.equal(f.nodes.get("start").disabled, true);
  await f.app.disconnectWebAdapter();
  start.resolve({ state: f.snapshot(true, 1) });
  await connecting;
  assert.equal(f.app.state.webAdapterActive, false);
  assert.equal(f.app.state.osc.session, null);
  assert.equal(f.requests.filter(r => r.url.endsWith("/start")).length, 1);
});

test("late intent rejection cannot restore a switched session's old safe target", async () => {
  const f = fixture();
  await f.app.reanchorWebAdapter();
  const intent = deferred();
  f.route(async (url) => url === "/api/osc/command" ? intent.promise : { state: f.snapshot(false, 2, "impedance") });
  const sending = f.app.sendIntent();
  await f.app.selectOutputMode("impedance");
  intent.resolve({ ok: false, state: f.snapshot(true, 1), result: {
    recoverable: true, safe_target_pose: f.snapshot().command.target_tcp,
  } });
  await sending;
  assert.equal(f.app.state.osc.session, null);
  assert.equal(f.app.state.oscAnchor, null);
});

test("late heartbeat cannot resurrect a stopped session or stop a new one", async () => {
  for (const fails of [false, true]) {
    const f = fixture();
    await f.app.reanchorWebAdapter();
    f.app.state.heartbeatFailures = 1;
    const beat = deferred();
    f.route(() => beat.promise);
    const pending = f.app.heartbeat();
    f.app.updateOscState(f.snapshot(false, 2));
    f.app.render();
    if (fails) beat.reject(new Error("old heartbeat failed"));
    else beat.resolve({ state: f.snapshot(true, 1) });
    await pending;
    assert.equal(f.app.state.osc.session, null);
    assert.equal(f.requests.length, 1);
  }
});

test("a delayed poll cannot overwrite an authoritative switch response", async () => {
  const f = fixture();
  await f.app.reanchorWebAdapter();
  const poll = deferred();
  f.route(async (url) => url === "/api/osc/state" ? poll.promise : { state: f.snapshot(false, 3, "impedance") });
  const refreshing = f.app.refresh(false);
  await f.app.selectOutputMode("impedance");
  poll.resolve(f.snapshot(true, 2));
  await refreshing;
  assert.equal(f.app.state.osc.output_mode, "impedance");
  assert.equal(f.app.state.webAdapterActive, false);
});

test("collection uses actual source, shared cameras and explicit target rates for every adapter", () => {
  const f = fixture();
  f.nodes.set("input-adapter", { value: "pi05" });
  f.app.state.dataset = { raw_camera_hz: 20, raw_robot_state_hz: 50,
    input_context: { control_source: "web", connected: true, output_mode: "impedance" },
    camera_sources: { external: { available: true, frame_available: true }, wrist: { available: false, frame_available: false } } };
  f.app.renderDataset();
  assert.match(f.nodes.get("dataset-source-summary").innerHTML, /网页摇杆/);
  assert.match(f.nodes.get("dataset-source-summary").innerHTML, /外部 RGB/);
  assert.match(f.nodes.get("dataset-content-summary").innerHTML, /20 Hz/);
  assert.match(f.nodes.get("dataset-content-summary").innerHTML, /50 Hz/);
  assert.equal(f.nodes.get("dataset-result").classList["dataset-warning"], true);
  assert.equal(f.nodes.get("dataset-start").disabled, false);
  f.app.state.dataset.input_context.connected = false;
  f.app.renderDataset();
  assert.equal(f.nodes.get("dataset-start").disabled, true);
});

test("camera, robot and command-history drops are visibly highlighted", () => {
  const f = fixture();
  f.app.state.dataset = { recording: true, duration_s: 3, raw_camera_drops: 2, raw_robot_drops: 1,
    osc_input_gaps: 4, osc_input_count: 75, raw_camera_hz: 20, raw_robot_state_hz: 50 };
  f.app.renderDataset();
  assert.equal(f.nodes.get("dataset-drops").classList["dataset-alert"], true);
  assert.equal(f.nodes.get("dataset-frames").classList["dataset-alert"], true);
  assert.equal(f.nodes.get("dataset-robot-rate").classList["dataset-alert"], true);
  assert.match(f.nodes.get("dataset-drops").textContent, /相机 2.*状态 1.*输入 4/);
  assert.match(f.nodes.get("dataset-inputs").textContent, /75/);
});

test("collection start does not use the selected UI tab as provenance", async () => {
  const f = fixture();
  f.route(async () => ({ recording: true }));
  await f.app.startDataset();
  const request = f.requests.find(r => r.url === "/api/dataset/start");
  assert.equal("control_source" in request.body, false);
});
