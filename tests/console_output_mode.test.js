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
      style: {}, classList: { toggle() {}, add() {}, remove() {} },
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
    drawWorkspace = renderHierarchy = renderPi05 = renderPico = renderDataset = renderCameraControls = () => {};
    globalThis.consoleTest = { state, render, updateOscState, connectWebAdapter, disconnectWebAdapter,
      reanchorWebAdapter, sendIntent, heartbeat, refresh, selectOutputMode, resetWebAdapter };
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
      { execution_mode: "hardware", client_id: "browser-test" });
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
