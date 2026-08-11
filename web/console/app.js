(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  function ensureSelectionControls() {
    const legacy = $("mode");
    if (!$("execution-mode") && legacy) {
      legacy.id = "execution-mode";
      legacy.innerHTML = '<option value="shadow">影子模式（只看虚拟机械臂）</option><option value="hardware">真机模式（控制真实机械臂）</option>';
      const label = legacy.closest("label");
      if (label) {
        label.firstChild.textContent = "控制模式";
        const inputLabel = document.createElement("label");
        inputLabel.innerHTML = '控制方式<select id="input-adapter"><option value="web">网页摇杆</option><option value="pi05">π0.5 自动控制</option><option value="pico">PICO 4 Ultra 手柄遥控</option></select>';
        label.parentElement?.insertBefore(inputLabel, label.nextSibling);
      }
    }
    const diagnostics = document.querySelector(".diagnostics");
    if (diagnostics && !$("observed-source")) {
      diagnostics.insertAdjacentHTML("beforeend", '<div><span>反馈来源</span><strong id="observed-source">--</strong></div><div><span>关节目标误差</span><strong id="joint-target-error">--</strong></div>');
    }
  }
  ensureSelectionControls();
  document.querySelector(".diagnostics-panel h2")?.replaceChildren("运动诊断");
  const gripperLabel = document.querySelector(".gripper-card .section-label");
  if (gripperLabel) gripperLabel.textContent = "DISCRETE ACTION";
  document.getElementById("gripper-result")?.remove();
  document.querySelector(".intent-readout span")?.replaceChildren("当前摇杆输入");
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
    osc: null,
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
    pi05: null,
    pi05Cameras: [],
    pico: null,
  };

  function buildPi05Card() {
    if ($("pi05-panel")) return;
    const panel = document.createElement("section");
    panel.id = "pi05-panel"; panel.className = "pi05-panel hidden";
    panel.innerHTML = `<div class="pi05-observation"><div class="pi05-head"><span class="pi05-index">01</span><div><strong>Observation</strong><small>π0.5 多模态输入</small></div></div><div class="pi05-cameras"><label>外部 RGB<input id="pi05-external-index" type="number" min="0" max="32" value="0"></label><label>腕部 RGB<input id="pi05-wrist-index" type="number" min="0" max="32" value="1"></label></div><div class="pi05-views"><div class="pi05-view"><span>外部视角</span><img id="pi05-external-frame" alt="外部 RGB 实时画面"><b id="pi05-external-preview">等待画面</b></div><div class="pi05-view"><span>腕部视角</span><img id="pi05-wrist-frame" alt="腕部 RGB 实时画面"><b id="pi05-wrist-preview">等待画面</b></div></div><label class="pi05-prompt">Prompt<textarea id="pi05-prompt" maxlength="500">place the fixed block into the fixed box</textarea></label></div><div class="pi05-arrow">→<small>推理中</small></div><div class="pi05-inference"><div class="pi05-head"><span class="pi05-index">02</span><div><strong>π0.5 Inference</strong><small>持续重规划</small></div></div><div class="pi05-orb">π</div><strong id="pi05-model-state">未连接</strong><div class="pi05-metrics"><span>状态<b id="pi05-run-state">IDLE</b></span><span>推理耗时<b id="pi05-inference-ms">--</b></span><span>生成序号<b id="pi05-chunk-length">0</b></span><span>执行动作<b id="pi05-executed">0</b></span></div><div class="pi05-actions"><button id="pi05-cameras" class="button" type="button">初始化相机</button><button id="pi05-start" class="button primary" type="button">启动 π0.5</button><button id="pi05-stop" class="button quiet" type="button">停止</button></div><p id="pi05-result" class="result">Action Chunk 将转换为绝对 TCP 目标，并只通过 OSC track_tcp 输出。</p></div>`;
    document.querySelector(".osc-panel")?.append(panel);
    for (const id of ["pi05-external-index", "pi05-wrist-index"]) {
      const input = $(id); const select = document.createElement("select");
      select.id = id; select.setAttribute("aria-label", id.includes("external") ? "外部 RGB 相机" : "腕部 RGB 相机");
      input?.replaceWith(select);
    }
    panel.querySelector(".pi05-cameras")?.insertAdjacentHTML("afterend", `<div class="camera-power"><span id="camera-power-pi05">相机状态：检查中</span><button id="camera-open-pi05" class="button" type="button">打开相机</button><button id="camera-close-pi05" class="button quiet" type="button">关闭相机</button></div>`);
    panel.insertAdjacentHTML("afterbegin", `<section class="pi05-connection"><div class="pi05-connection-title"><span>02 · 连接状态</span><small id="pi05-connection-message">正在检测 SSH 与 OpenPI 连接</small></div><div class="pi05-connection-nodes"><article id="pi05-connection-ssh"></article><i>→</i><article id="pi05-connection-policy"></article></div><details class="pi05-help"><summary>连接帮助 <small>启动顺序与可复制命令</small></summary><div></div></details></section>`);
    const help = panel.querySelector(".pi05-help");
    help.innerHTML = `<summary><span>连接帮助</span><small>启动顺序与可复制命令</small></summary><div class="help-content"><article class="help-step"><div class="help-title"><b>1</b><strong>启动 NERO 控制服务</strong></div><p>确认 NERO 已通电、CANDO USB-CAN 已连接，且没有其他程序占用设备。</p><div class="command"><code>..\\neroAgilex-control-console\\run_console.cmd</code><button type="button" data-copy="..\\neroAgilex-control-console\\run_console.cmd">复制</button></div><p>打开 <a href="http://127.0.0.1:8765/" target="_blank" rel="noopener">127.0.0.1:8765</a>，确认持续显示最新 7 轴反馈。若显示 HTTP 502，通常是未通电、USB-CAN/驱动未连接、设备被占用或控制服务环境异常；先恢复有效反馈，第一个绿灯才会亮起。</p></article><article class="help-step"><div class="help-title"><b>2</b><strong>建立 SSH 本地转发</strong></div><p>在新的 PowerShell 窗口执行。输入 AutoDL 密码后保留该窗口；连接完成后仍可直接在远程终端输入命令。</p><div class="command"><code>ssh -t \`\n  -o LogLevel=QUIET \`\n  -o ServerAliveInterval=15 \`\n  -o ServerAliveCountMax=3 \`\n  -L 8000:127.0.0.1:8000 \`\n  -p 38341 \`\n  root@connect.bjb1.seetacloud.com</code><button type="button" data-copy="ssh -t \`\n  -o LogLevel=QUIET \`\n  -o ServerAliveInterval=15 \`\n  -o ServerAliveCountMax=3 \`\n  -L 8000:127.0.0.1:8000 \`\n  -p 38341 \`\n  root@connect.bjb1.seetacloud.com">复制</button></div><p>实例地址、端口或密码变更时，请使用 AutoDL 当前提供的信息。本机验证：<code>Test-NetConnection 127.0.0.1 -Port 8000</code>；看到 <code>TcpTestSucceeded : True</code> 即隧道已建立。</p></article><article class="help-step"><div class="help-title"><b>3</b><strong>启动 AutoDL π0.5 policy server</strong></div><p>在第 2 步登录后的 AutoDL 终端执行，并保持终端运行：</p><div class="command"><code>cd /root/autodl-tmp/libero_openpi_loop/openpi\nexport PATH=&quot;$HOME/.local/bin:$PATH&quot;\nexport BOTO_CONFIG=&quot;$HOME/.boto&quot;\nuv run --frozen --no-sync scripts/serve_policy.py --env LIBERO</code><button type="button" data-copy="cd /root/autodl-tmp/libero_openpi_loop/openpi\nexport PATH=&quot;$HOME/.local/bin:$PATH&quot;\nexport BOTO_CONFIG=&quot;$HOME/.boto&quot;\nuv run --frozen --no-sync scripts/serve_policy.py --env LIBERO">复制</button></div><p>SSH 仅转发端口，不会自动启动 policy server。</p></article><div class="help-ready"><strong>4. 刷新面板</strong><span>保持控制服务、SSH 隧道和 policy server 都在运行，刷新 <a href="http://127.0.0.1:8765/" target="_blank" rel="noopener">127.0.0.1:8765</a>。正常顺序：NERO 控制服务 → SSH 本地转发 → π0.5 WebSocket，三个状态均为绿色。</span></div></div>`;
    // The control service is already running when this card is shown. Keep
    // this help focused on the two remote-connection steps only.
    help.querySelector(".help-step")?.remove();
    help.querySelector(".help-ready")?.remove();
    help.querySelectorAll(".help-step").forEach((step, index) => {
      const number = step.querySelector(".help-title b");
      if (number) number.textContent = String(index + 1);
    });
    const policyStep = help.querySelectorAll(".help-step")[1];
    policyStep?.querySelector("p")?.replaceChildren("在第 1 步登录后的 AutoDL 终端执行，并保持终端运行：");
    help.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", async () => { await navigator.clipboard.writeText(button.dataset.copy || ""); const original = button.textContent; button.textContent = "已复制"; setTimeout(() => { button.textContent = original; }, 1200); }));
    panel.querySelector(".pi05-inference")?.insertAdjacentHTML("beforeend", `<div class="pi05-chunk"><span>Action Chunk（首步）</span><code id="pi05-action-first">等待推理结果</code></div>`);
    const actionControls = panel.querySelector(".pi05-actions");
    actionControls.className = "pi05-adapter-actions";
    panel.append(actionControls);
    panel.querySelector(".pi05-inference .pi05-index").textContent = "03";
    $("pi05-start").textContent = "接入 π0.5";
    $("pi05-stop").textContent = "断开 π0.5";
    // π0.5 follows the same compact input-adapter rhythm as the WebAdapter:
    // input first, diagnostics beside it, connection and actions below.
    const observation = panel.querySelector(".pi05-observation");
    const inference = panel.querySelector(".pi05-inference");
    const connection = panel.querySelector(".pi05-connection");
    const chunk = panel.querySelector(".pi05-chunk");
    observation?.querySelector(".pi05-head")?.remove();
    const prompt = observation?.querySelector(".pi05-prompt");
    observation?.querySelector(".pi05-views")?.after(prompt);
    inference?.querySelector(".pi05-orb")?.remove();
    observation?.querySelectorAll(".pi05-cameras label").forEach((label) => {
      label.classList.add("pi05-camera-select");
    });
    inference?.querySelector(".pi05-index")?.remove();
    $("pi05-cameras")?.remove();
    if (prompt) prompt.firstChild.textContent = "任务说明";
    const inferenceTitle = inference?.querySelector(".pi05-head strong");
    if (inferenceTitle) inferenceTitle.textContent = "π0.5 自动控制";
    const inferenceHint = inference?.querySelector(".pi05-head small");
    if (inferenceHint) inferenceHint.textContent = "根据相机画面生成下一步动作";
    const chunkTitle = chunk?.querySelector("span");
    if (chunkTitle) chunkTitle.textContent = "下一步动作";
    const inferenceStack = document.createElement("div");
    inferenceStack.className = "pi05-inference-stack";
    if (inference) inferenceStack.append(inference);
    if (chunk) inferenceStack.append(chunk);
    const row = document.createElement("div");
    row.className = "pi05-adapter-row";
    if (observation) row.append(observation);
    row.append(inferenceStack);
    panel.replaceChildren(row, connection, actionControls);
    $("pi05-start").textContent = "开始自动控制";
    $("pi05-stop").textContent = "停止自动控制";
    [".sticks", ".intent-readout", ".keyboard-map", ".session-actions", "#pico-connection"].forEach((selector) => document.querySelector(selector)?.setAttribute("data-web-adapter", ""));
  }

  function buildPicoCard() {
    if ($("pico-panel")) return;
    const panel = document.createElement("section");
    panel.id = "pico-panel"; panel.className = "pico-panel hidden";
    panel.innerHTML = `<div class="pico-head"><span class="pi05-index">P</span><div><strong>PICO 4 Ultra</strong><small>6D 遥操输入适配器</small></div><span id="pico-state" class="badge neutral">IDLE</span></div><section class="pico-camera-resource"><strong>公共相机观测</strong><div class="pi05-cameras"><label>外部 RGB<select id="pico-external-index" aria-label="外部 RGB 相机"></select></label><label>腕部 RGB<select id="pico-wrist-index" aria-label="腕部 RGB 相机"></select></label></div><div class="pi05-views"><div class="pi05-view"><span>外部视角</span><img id="pico-external-frame" alt="外部 RGB 实时画面"><b>等待画面</b></div><div class="pi05-view"><span>腕部视角</span><img id="pico-wrist-frame" alt="腕部 RGB 实时画面"><b>等待画面</b></div></div></section><div class="pico-pair"><div id="pico-qr" class="pico-qr"><span>等待配对</span></div><div><strong>一次性配对</strong><p id="pico-url">先接入 PICO Adapter 以生成二维码。</p><code id="pico-code">------</code><small>二维码或短码有效期内仅允许一个头显接入。</small></div></div><div class="pico-status"><span>连接 <b id="pico-connected">未连接</b></span><span>追踪 <b id="pico-tracking">--</b></span><span>Anchor <b id="pico-anchor">--</b></span><span>夹爪 <b id="pico-gripper">--</b></span></div><div class="pico-controls"><strong>手柄映射</strong><p>右手 Grip 按住：定义 Anchor 并控制 TCP；右手 Trigger：夹爪开度；左手 Menu：安全 HOLD。松开 Grip、追踪丢失或断连都会进入 HOLD。</p></div><div class="pico-actions"><button id="pico-start" class="button primary" type="button">接入 PICO Adapter</button><button id="pico-stop" class="button quiet" type="button">断开 PICO Adapter</button></div><p id="pico-result" class="result">PICO 只向 OSC 提交绝对 TCP 与标准夹爪/HOLD 指令。</p>`;
    document.querySelector(".osc-panel")?.append(panel);
    panel.querySelector(".pico-camera-resource .pi05-cameras")?.insertAdjacentHTML("afterend", `<div class="camera-power"><span id="camera-power-pico">相机状态：检查中</span><button id="camera-open-pico" class="button" type="button">打开相机</button><button id="camera-close-pico" class="button quiet" type="button">关闭相机</button></div>`);
    const picoSubtitle = panel.querySelector(".pico-head small");
    if (picoSubtitle) picoSubtitle.textContent = "用 PICO 手柄控制机械臂";
    const picoAnchor = $("pico-anchor")?.parentElement;
    if (picoAnchor) picoAnchor.firstChild.textContent = "控制起点 ";
    const picoControls = panel.querySelector(".pico-controls");
    if (picoControls) picoControls.innerHTML = "<strong>怎么控制</strong><p>按住右手 Grip：移动机械臂；右手 Trigger：开合夹爪；左手 Menu：立即停止。松开 Grip、追踪丢失或断开连接时，机械臂会自动停止。</p>";
    $("pico-start").textContent = "连接 PICO 手柄";
    $("pico-stop").textContent = "断开 PICO 手柄";
    [".sticks", ".intent-readout", ".keyboard-map", ".session-actions", "#pico-connection"].forEach((selector) => document.querySelector(selector)?.setAttribute("data-web-adapter", ""));
  }

  const canvas = $("workspace");
  const ctx = canvas?.getContext("2d");
  const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const fixed = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";
  const controllerModeLabel = (value) => {
    const code = Number(value);
    const labels = {
      0: "待机",
      1: "CAN 指令控制",
      2: "示教模式",
      3: "以太网控制",
      4: "Wi-Fi 控制",
      5: "远程控制",
      6: "联动示教输入",
      7: "离线轨迹",
      8: "TCP 控制",
    };
    return Number.isInteger(code) ? `${labels[code] || "未知控制模式"}（${code}）` : "--";
  };

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
    return state.osc?.session || { state: "IDLE", id: null, client_id: null, mode: null, execution_mode: null, input_source: null };
  }

  const isShadowSession = (current) => (current.execution_mode || current.mode) === "shadow";

  function canSendNonzero(allowHoldReadyResume = false) {
    const current = session();
    const diagnostic = state.osc?.diagnostics || {};
    const shadow = isShadowSession(current);
    if (allowHoldReadyResume && diagnostic.trajectory_state === "HOLD_READY" &&
        current.state === "ACTIVE" && current.client_id === clientId) {
      // Let the backend resynchronise Ruckig and restore TRACKING from the
      // current client's fresh deadman packet.
      return true;
    }
    const common = current.state === "ACTIVE" &&
      current.client_id === clientId &&
      state.osc?.execution?.accepting_targets === true;
    if (!common) {
      return false;
    }
    return (shadow || state.osc?.authority?.servo_mode === "TRACKING") &&
      diagnostic.trajectory_state === "RUNNING";
  }

  function permissionReason() {
    const current = session();
    if (current.state !== "ACTIVE") return "会话未处于 ACTIVE";
    if (current.client_id !== clientId) return "会话属于其他客户端";
    if (state.osc?.diagnostics?.trajectory_state === "FAULT") return `FAULT · ${state.osc.diagnostics.trajectory_brake_reason || "轨迹故障"}`;
    if (state.osc?.execution?.accepting_targets !== true) return "等待 OSC 输入权限";
    if (!isShadowSession(current) && state.osc?.authority?.servo_mode !== "TRACKING") return "等待进入 TRACKING";
    if (state.osc?.diagnostics?.trajectory_state !== "RUNNING") return "等待遥操轨迹运行";
    return "";
  }

  function resetWebAdapter() {
    state.webAdapterActive = false;
    state.oscAnchor = null;
    state.intentPending = false;
    resetInput(false);
  }

  const selectedAdapter = () => $("input-adapter")?.value || "web";
  function applyAdapterSelection() {
    const pi = selectedAdapter() === "pi05";
    const pico = selectedAdapter() === "pico";
    document.querySelectorAll("[data-web-adapter]").forEach((element) => element.classList.toggle("hidden", pi || pico));
    $("pi05-panel")?.classList.toggle("hidden", !pi);
    $("pico-panel")?.classList.toggle("hidden", !pico);
    const note = document.querySelector(".osc-panel .section-note");
    if (note) note.textContent = pi ? "π0.5 将双相机观测与 OSC 状态送入 OpenPI，并把 Action Chunk 转换为绝对 TCP 目标。" : pico ? "PICO Adapter 在 OSC 外部处理 Anchor，并只向 OSC 发送绝对 TCP 目标。" : "WebAdapter 将网页摇杆增量转换为基座系绝对 TCP 目标，并接入 OSC。";
    if (note) note.textContent = pi ? "π0.5 会查看两路相机画面，自动生成并执行下一步机械臂动作。" : pico ? "用 PICO 手柄遥控机械臂：按住右手 Grip 才会移动，松开即停止。" : "用网页摇杆控制机械臂；系统会自动把摇杆动作变成机械臂末端的目标位置。";
    if (pi || pico) {
      void loadSharedCameras();
      if (!state.cameras?.ready) void activateSharedCameras();
    }
    render();
  }

  function renderPico() {
    const pico = state.pico || {}; const gateway = pico.gateway || {};
    $("pico-state").textContent = pico.state || "IDLE";
    $("pico-connected").textContent = pico.connected ? "已连接" : "未连接";
    $("pico-tracking").textContent = pico.tracking_valid ? "有效" : "--";
    $("pico-anchor").textContent = pico.anchor_active ? "已定义" : "未定义";
    $("pico-gripper").textContent = Number.isFinite(Number(pico.gripper_position)) ? `${Math.round(Number(pico.gripper_position) * 100)}%` : "--";
    $("pico-code").textContent = gateway.pair_code || "------";
    $("pico-url").textContent = gateway.ws_url ? `头显将连接 ${gateway.ws_url}` : "先接入 PICO Adapter 以生成二维码。";
    $("pico-result").textContent = pico.last_error || gateway.error || (gateway.paired ? "PICO 已配对；按住右手 Grip 开始定义 Anchor。" : "使用头显 App 扫描此处二维码或输入短码。");
    // A compact visual token remains useful even before the companion scanner
    // is available; the app also accepts the displayed short code.
    const qr = $("pico-qr"); if (qr) qr.innerHTML = gateway.pair_code ? `<img src="/api/adapters/pico/pair.svg?v=${encodeURIComponent(gateway.pair_code)}" alt="PICO 一次性配对二维码">` : "等待配对";
    $("pico-start").disabled = gateway.paired || session().state === "ACTIVE" && selectedAdapter() !== "pico";
    $("pico-stop").disabled = !gateway.session_id && pico.state === "IDLE";
    $("pico-start").textContent = "连接 PICO 手柄";
    $("pico-stop").textContent = "断开 PICO 手柄";
    if (!gateway.ws_url) $("pico-url").textContent = "点击“连接 PICO 手柄”后，会在这里生成二维码。";
    if (!pico.last_error && !gateway.error) $("pico-result").textContent = gateway.paired ? "已配对。按住右手 Grip 后再移动手柄，即可控制机械臂。" : "在头显中扫描二维码，或输入这里显示的短码。";
  }

  async function startPico() {
    $("pico-start").disabled = true; phase("正在接入 PICO Adapter…");
    try {
      let current = session();
      if (current.state !== "ACTIVE") {
        const started = await api("/api/osc/session/start", "POST", { execution_mode: $("execution-mode").value, client_id: clientId }, 10000);
        state.osc = started.state; current = session();
      }
      state.pico = await api("/api/adapters/pico/pair", "POST", { session_id: current.id, client_id: clientId }, 10000);
      phase("PICO 配对已创建；在头显中扫描二维码。 "); render();
    } catch (error) { phase(`PICO 接入失败：${error.message}`, true); }
    finally { $("pico-start").disabled = false; }
  }

  async function stopPico() {
    try { state.pico = await api("/api/adapters/pico/disconnect", "POST", { reason: "Console disconnected PICO" }); state.osc = (await api("/api/osc/session/stop", "POST", { reason: "PICO Adapter disconnected" })).state; render(); }
    catch (error) { phase(`PICO 断开失败：${error.message}`, true); }
  }

  function renderPi05() {
    const pi = state.pi05 || {}; const config = pi.config || {}; const cameras = config.cameras || {}; const model = config.model || {};
    if ($("pi05-external-index") && document.activeElement !== $("pi05-external-index")) $("pi05-external-index").value = cameras.external?.index ?? 0;
    if ($("pi05-wrist-index") && document.activeElement !== $("pi05-wrist-index")) $("pi05-wrist-index").value = cameras.wrist?.index ?? 1;
    if ($("pi05-prompt") && document.activeElement !== $("pi05-prompt")) $("pi05-prompt").value = model.prompt || pi.prompt || "";
    $("pi05-model-state").textContent = pi.model_state || "UNKNOWN";
    $("pi05-run-state").textContent = pi.state || "IDLE";
    $("pi05-inference-ms").textContent = Number.isFinite(Number(pi.inference_ms)) ? `${Number(pi.inference_ms).toFixed(0)} ms` : "--";
    $("pi05-chunk-length").textContent = String(pi.action_chunk_length || 0);
    $("pi05-executed").textContent = String(pi.executed_steps || 0);
    const firstAction = Array.isArray(pi.action_chunk) && Array.isArray(pi.action_chunk[0]) ? pi.action_chunk[0] : null;
    $("pi05-action-first").textContent = firstAction ? `[${firstAction.map((value) => Number(value).toFixed(3)).join(", ")}]` : "等待推理结果";
    const result = $("pi05-result"); if (result) result.textContent = pi.last_error || "";
    $("pi05-external-preview").textContent = pi.camera_ready ? "模型输入 224 × 224 RGB" : "等待画面";
    $("pi05-wrist-preview").textContent = pi.camera_ready ? "模型输入 224 × 224 RGB" : "等待画面";
    const frameVersion = Number(pi.frame_version || state.cameras?.frame_version || 0);
    if (frameVersion) {
      $("pi05-external-frame").src = `/api/cameras/frame/external.jpg?v=${frameVersion}`;
      $("pi05-wrist-frame").src = `/api/cameras/frame/wrist.jpg?v=${frameVersion}`;
      $("pi05-external-frame").classList.add("ready"); $("pi05-wrist-frame").classList.add("ready");
    }
    if ($("pi05-cameras")) $("pi05-cameras").disabled = pi.state === "RUNNING";
    $("pi05-start").disabled = pi.state === "RUNNING" || !pi.camera_ready;
    $("pi05-stop").disabled = pi.state !== "RUNNING" && pi.state !== "ERROR";
    $("pi05-start").textContent = "开始自动控制";
    $("pi05-stop").textContent = "停止自动控制";
    renderPi05Connections(pi.connections || {});
  }

  function renderPi05Connections(connections) {
    const map = [["pi05-connection-ssh", connections.ssh_forward, "SSH 本地转发", "127.0.0.1:8000"], ["pi05-connection-policy", connections.policy, "π0.5 WebSocket", "OpenPI policy server"]];
    map.forEach(([id, item, fallbackLabel, fallbackEndpoint]) => { const element = $(id); if (!element) return; const stateName = item?.state || "bad"; element.className = `pi05-connection-node ${stateName}`; element.innerHTML = `<b>${item?.label || fallbackLabel}</b><small>${item?.endpoint || fallbackEndpoint}</small><em>${item?.message || "等待连接检测"}</em>`; });
    const message = $("pi05-connection-message"); if (message) message.textContent = connections.policy?.state === "ok" ? "连接完成，可以接入 π0.5" : "请依次检查 OSC 服务、SSH 转发和 OpenPI policy server";
  }

  function cameraConfigBody() { return { cameras: {
    external: { index: Number($(selectedAdapter() === "pico" ? "pico-external-index" : "pi05-external-index")?.value), width: 640, height: 480 },
    wrist: { index: Number($(selectedAdapter() === "pico" ? "pico-wrist-index" : "pi05-wrist-index")?.value), width: 640, height: 480 },
  }}; }

  async function loadSharedCameras() {
    try {
      const [result, cameras] = await Promise.all([api("/api/cameras/list", "GET", undefined, 12000), api("/api/cameras/state", "GET", undefined, 12000)]); state.pi05Cameras = result.cameras || []; state.cameras = cameras;
      const config = cameras.config || {}; const fill = (id, selected) => { const select = $(id); if (!select) return; select.innerHTML = state.pi05Cameras.map((camera) => `<option value="${Number(camera.index)}">${Number(camera.index)} · ${String(camera.name || "Camera")}</option>`).join("") || `<option value="${selected}">${selected} · 未检测到相机</option>`; select.value = String(selected); select.onchange = activateSharedCameras; };
      ["pi05-external-index", "pico-external-index"].forEach((id) => fill(id, config.external?.index ?? 0));
      ["pi05-wrist-index", "pico-wrist-index"].forEach((id) => fill(id, config.wrist?.index ?? 1));
    } catch (error) { const result = $("pi05-result"); if (result) result.textContent = `相机列表读取失败：${error.message}`; }
  }

  function refreshPi05Frames() {
    if ((selectedAdapter() !== "pi05" && selectedAdapter() !== "pico") || !state.cameras?.ready) return;
    const stamp = Date.now();
    ["pi05", "pico"].forEach((prefix) => { const external = $(`${prefix}-external-frame`), wrist = $(`${prefix}-wrist-frame`); if (external) { external.src = `/api/cameras/frame/external.jpg?t=${stamp}`; external.classList.add("ready"); } if (wrist) { wrist.src = `/api/cameras/frame/wrist.jpg?t=${stamp}`; wrist.classList.add("ready"); } });
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
    const target = state.osc?.command?.target_tcp;
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
    const transport = state.osc?.transport || {};
    const broker = state.osc?.authority || {};
    const osc = state.osc || {};
    const action = osc.active_action;
    if (broker.hardware_mode === "FAULT" || broker.arm_writer === "SAFETY" || broker.safety_state === "FAULT" || osc.diagnostics?.trajectory_state === "FAULT") {
      return `FAULT · ${transport.reason || broker.reason || osc.diagnostics?.trajectory_brake_reason || "安全停车"}`;
    }
    if (broker.arm_writer === "MODE_TRANSITION") return "MODE_TRANSITION";
    if (action?.type) return `${String(action.type).toUpperCase()} · 执行中`;
    if (broker.servo_mode) return broker.servo_mode;
    if (osc.session?.state === "ACTIVE") return osc.execution?.accepting_targets ? "ACTIVE · OSC 已就绪" : "ACTIVE · 等待 OSC";
    return broker.hardware_mode || "IDLE";
  }

  async function sendIntent() {
    const current = session();
    if (current.state !== "ACTIVE" || current.client_id !== clientId) return;
    if (!state.webAdapterActive) return;
    const anchor = state.oscAnchor || state.osc?.command?.target_tcp;
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
        state.osc = result.state;
      }
    }
    if (!result?.ok) {
      const acceptedSequence = Number(result?.result?.accepted_sequence);
      if (Number.isFinite(acceptedSequence)) state.oscSequence = Math.max(state.oscSequence, acceptedSequence);
      const safeTarget = result?.result?.safe_target_pose;
      if (result?.result?.recoverable && safeTarget?.position_m && safeTarget?.orientation_xyzw) {
        // A one-shot workspace rejection must not trap the browser in an
        // invalid accumulated relative pose. Re-anchor to OSC's last safe
        // absolute target while keeping the active session and lease intact.
        state.oscAnchor = {
          position_m: [...safeTarget.position_m],
          orientation_xyzw: [...safeTarget.orientation_xyzw],
        };
        resetInput(false);
      }
      const reason = result?.result?.reason || result?.state?.diagnostics?.trajectory_brake_reason || "OSC 未接受该目标";
      throw new Error(`OSC 拒绝：${reason}`);
    }
    if (result?.result?.accepted_sequence != null && state.osc?.command) {
      state.oscSequence = Math.max(state.oscSequence, Number(result.result.accepted_sequence));
      state.osc.command.sequence = result.result.accepted_sequence;
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

  function drawWorkspace(osc) {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const tcp = osc?.diagnostics?.pink?.tcp || {};
    const links = Array.isArray(tcp.link_positions_m) ? tcp.link_positions_m : [];
    const currentPose = osc?.execution?.measured_tcp_pose || osc?.transport?.hardware_feedback?.tcp_pose || null;
    const position = Array.isArray(currentPose?.position_m) && currentPose.position_m.length === 3 && currentPose.position_m.every((value) => Number.isFinite(Number(value)))
      ? currentPose.position_m.map(Number)
      : (Array.isArray(tcp.position_m) && tcp.position_m.length === 3 && tcp.position_m.every((value) => Number.isFinite(Number(value))) ? tcp.position_m.map(Number) : null);
    const targetPose = osc?.command?.target_tcp || null;
    const targetPosition = Array.isArray(targetPose?.position_m) && targetPose.position_m.length === 3 && targetPose.position_m.every((value) => Number.isFinite(Number(value)))
      ? targetPose.position_m.map(Number) : null;
    const targetRotation = rotationFromQuat(targetPose?.orientation_xyzw);
    const workspace = osc?.workspace || {};
    const min = Array.isArray(workspace.min_xyz_m) ? workspace.min_xyz_m.map(Number) : [-0.45, -0.15, -0.02];
    const max = Array.isArray(workspace.max_xyz_m) ? workspace.max_xyz_m.map(Number) : [0.45, 0.60, 0.70];
    const minZ = Number.isFinite(Number(workspace.min_tcp_z_m)) ? Number(workspace.min_tcp_z_m) : min[2];
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
    const osc = state.osc || {};
    const broker = osc.authority || {};
    const transport = osc.transport || {};
    const command = osc.command || {};
    const execution = osc.execution || {};
    const hardwareFeedback = transport.hardware_feedback || {};
    const current = session();
    const diagnostic = osc.diagnostics || {};
    const timing = diagnostic.timing || {};
    const executionFeedbackAge = finite(execution.feedback_age_s, NaN);
    const hardwareFeedbackAge = finite(hardwareFeedback.feedback_age_s, NaN);
    const timingFeedbackAge = finite(timing.feedback_age_s, NaN);
    const feedbackAgeS = Number.isFinite(executionFeedbackAge) ? executionFeedbackAge : Number.isFinite(timingFeedbackAge) ? timingFeedbackAge : hardwareFeedbackAge;
    // A successful /api/status response already proves that the service is
    // online. This flag describes only the USB-CAN robot transport.
    badge("connection", transport.connected ? "\u786c\u4ef6\u5df2\u8fde\u63a5" : "\u786c\u4ef6\u672a\u8fde\u63a5", transport.connected ? "ok" : "warn");
    const adapterName = selectedAdapter() === "pi05" ? "π0.5" : selectedAdapter() === "pico" ? "PICO 4 Ultra" : "WebAdapter";
    const modeLabel = current.state === "ACTIVE" ? `${adapterName} + ${isShadowSession(current) ? "影子" : "真机"}` : "未接入";
    badge("osc-mode", modeLabel, current.state === "ACTIVE" && isShadowSession(current) ? "ok" : current.state === "ACTIVE" ? "warn" : "neutral");
    badge("session-state", current.state || "IDLE", current.state === "ACTIVE" ? "ok" : "neutral");
    badge("feedback-state", Number.isFinite(feedbackAgeS) ? `反馈 ${Math.round(feedbackAgeS * 1000)} ms` : "反馈 --", transport.can_health?.ok ? "ok" : "warn");
    badge("safety-state", `安全 ${broker.safety_state || diagnostic.trajectory_state || "--"}`, broker.safety_state === "FAULT" ? "fault" : "neutral");
    badge("writer-state", `Writer ${broker.arm_writer || "--"}`, broker.arm_writer === "SERVO" ? "ok" : "neutral");
    badge("servo-state", `Servo ${broker.servo_mode || "--"}`, broker.servo_mode === "TRACKING" ? "ok" : "neutral");
    const inputStatus = execution.accepting_targets ? "WebAdapter 已就绪" : current.state === "ACTIVE" ? "等待 OSC 输入权限" : "WebAdapter 未接入";
    const livePhase = backendPhase();
    phase(state.last_error || livePhase || inputStatus, Boolean(state.last_error));
    $("solver-badge").textContent = osc.solver?.running ? "求解器在线" : "求解器空闲";
    $("solver-badge").className = `badge ${osc.solver?.running ? "ok" : "neutral"}`;
    $("status-age").textContent = Number.isFinite(feedbackAgeS) ? `反馈 ${Math.round(feedbackAgeS * 1000)} ms` : "反馈 --";
    $("control-mode").textContent = `${broker.hardware_mode || "DISCONNECTED"} · ${broker.control_role || "NONE"}`;
    $("controller-mode").textContent = controllerModeLabel(hardwareFeedback.arm_status?.ctrl_mode);
    $("active-action").textContent = osc.active_action?.type || "无";
    $("reason").textContent = transport.reason || diagnostic.safety_gate?.reason || diagnostic.trajectory_brake_reason || "--";
    const grip = osc.gripper || {};
    $("gripper-width-value").textContent = grip.width_m == null ? "--" : `${fixed(grip.width_m * 1000, 1)} mm`;
    $("gripper-force").textContent = grip.force_n == null ? "--" : `${fixed(grip.force_n, 2)} N`;
    $("gripper-driver").textContent = grip.status?.foc_status?.driver_enable_status ? "已使能" : "--";
    const formatAbsolutePose = (pose) => {
      const position = pose?.position_m;
      const orientation = pose?.orientation_xyzw;
      if (!Array.isArray(position) || position.length !== 3 || !Array.isArray(orientation) || orientation.length !== 4) return "--";
      const rpy = eulerFromQuat(orientation).map((value) => `${value.toFixed(3)} rad`);
      return `位置 [${position.map((value) => `${Number(value).toFixed(3)} m`).join(", ")}] · RPY [${rpy.join(", ")}]`;
    };
    const hardwarePose = hardwareFeedback.tcp_pose || execution.measured_tcp_pose || null;
    $("tcp-pose-values").textContent = formatAbsolutePose(hardwarePose);
    const hardwareSource = hardwareFeedback.tcp_source || "none";
    const feedbackAgeMs = Number.isFinite(hardwareFeedbackAge) ? hardwareFeedbackAge : executionFeedbackAge;
    const feedbackSourceText = hardwareSource === "sdk_tcp_from_leader_fk" ? "SDK TCP · Leader FK" : hardwareSource === "sdk_tcp_from_follower_feedback" ? "SDK TCP · Follower" : "反馈未就绪";
    badge("hardware-feedback-source", feedbackSourceText, hardwarePose ? "ok" : "neutral");
    const feedbackNote = $("hardware-feedback-note");
    if (feedbackNote) feedbackNote.textContent = Number.isFinite(feedbackAgeMs)
      ? `${hardwareFeedback.joint_feedback_source === "leader" ? "FREEDRIVE · Leader 关节反馈" : "SDK 只读硬件反馈"} · 反馈龄期 ${Math.round(feedbackAgeMs * 1000)} ms`
      : "等待 OSC 控制循环发布有效反馈样本";
    const referenceLine = $("tcp-reference-readout");
    if (referenceLine) referenceLine.textContent = `实测 TCP（同一采样）：${formatAbsolutePose(execution.measured_tcp_pose)} · 目标 TCP（同一采样）：${formatAbsolutePose(command.target_tcp)}`;
    const joints = hardwareFeedback.joint_angles_rad || execution.measured_joint_state_rad || [];
    $("joints").innerHTML = joints.map((value, index) => `<div><span>J${index + 1}</span><strong>${fixed(value * 180 / Math.PI, 2)}°</strong></div>`).join("") || "<span class='muted'>无关节反馈</span>";
    $("input-age").textContent = Number.isFinite(finite(current.last_input_age_s, NaN)) ? `${fixed(current.last_input_age_s, 2)} s` : "--";
    $("loop-rate").textContent = String(diagnostic.loop_count ?? "--");
    $("cpv-count").textContent = String(execution.output_count ?? 0);
    if ($("cpv-count").previousElementSibling) $("cpv-count").previousElementSibling.textContent = "OSC 输出批次";
    $("condition").textContent = Number.isFinite(finite(diagnostic.pink?.condition_number, NaN)) ? fixed(diagnostic.pink.condition_number, 1) : "--";
    $("period-ms").textContent = Number.isFinite(finite(timing.actual_dt_s, NaN)) ? `${fixed(timing.actual_dt_s * 1000, 1)} ms` : "--";
    $("solver-age").textContent = Number.isFinite(finite(timing.solver_age_s, NaN)) ? `${fixed(timing.solver_age_s * 1000, 0)} ms` : "--";
    $("feedback-age").textContent = Number.isFinite(feedbackAgeS) ? `${fixed(feedbackAgeS * 1000, 0)} ms` : "--";
    // The headline error is deliberately physical/plant feedback against the
    // target bound to the same OSC sample, never Pink's predicted FK error.
    const tcpError = diagnostic.measured_tracking_error || diagnostic.tcp_error || {};
    $("tcp-position-error").textContent = Number.isFinite(finite(tcpError.position_norm_m, NaN)) ? `${fixed(tcpError.position_norm_m * 1000, 1)} mm` : "--";
    $("tcp-orientation-error").textContent = Number.isFinite(finite(tcpError.orientation_angle_rad, NaN)) ? `${fixed(tcpError.orientation_angle_rad * 180 / Math.PI, 1)}°` : "--";
    $("gate-state").textContent = timing.gate_limited === true ? "限速" : timing.gate_ok === true ? "通过" : timing.gate_ok === false ? "拒绝" : "--";
    $("trajectory-state").textContent = diagnostic.trajectory_state || "--";
    const source = execution.observed_source || "--";
    $("observed-source").textContent = source === "simulated_cpv_feedback" ? "影子 CPV 模拟" : source === "measured_can_feedback" ? "CAN 实测" : "--";
    const jointTargetError = execution.joint_target_error || {};
    $("joint-target-error").textContent = Number.isFinite(finite(jointTargetError.max_abs_rad, NaN)) ? `${fixed(jointTargetError.max_abs_rad * 180 / Math.PI, 2)}°` : "--";
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
    $("start").textContent = active && !state.webAdapterActive ? "重新连接网页摇杆" : "连接网页摇杆";
    $("stop").textContent = "断开网页摇杆";
    if ($("reanchor")) $("reanchor").textContent = "重新对齐摇杆";
    const picoLine = $("pico-connection");
    document.querySelector(".intent-readout span")?.replaceChildren("当前摇杆输入");
    if (picoLine) picoLine.textContent = "网页摇杆的动作会自动换算为机械臂末端的目标位置。";
    if (picoLine) picoLine.textContent = "WebAdapter 在浏览器内将摇杆增量转换为基座系绝对 TCP 目标，并只调用 OSC track_tcp。";
    if (picoLine) picoLine.textContent = "网页摇杆的动作会自动换算为机械臂末端的目标位置。";
    $("hold").disabled = !transport.connected;
    $("freedrive").disabled = !transport.connected;
    // Reset is served by the independent localhost watchdog and sends no
    // robot command. It must remain available precisely when the backend is
    // disconnected, initializing, or stuck.
    $("reset-control").disabled = Date.now() < state.resetPendingUntil;
    drawWorkspace(osc);
    renderHierarchy(osc, broker, transport, hardwareFeedback, diagnostic, current);
    renderPi05();
    renderPico();
    renderCameraControls();
  }

  async function refresh(includeAuxiliary = true) {
    if (state.refreshBusy) return;
    state.refreshBusy = true;
    const generation = state.requestGeneration;
    const resetPending = Date.now() < state.resetPendingUntil;
    const statusTimeout = resetPending ? 5000 : 900;
    try {
      const requests = [api("/api/osc/state", "GET", undefined, statusTimeout)];
      if (includeAuxiliary) {
        requests.push(
          api("/api/pi05/state", "GET", undefined, statusTimeout),
          api("/api/adapters/pico/state", "GET", undefined, statusTimeout),
        );
      }
      const [osc, pi05, pico] = await Promise.all(requests);
      if (generation !== state.requestGeneration) return;
      const sequence = Number(osc.state_sequence || osc.session?.sequence || 0);
      if (sequence < state.latestSequence) return;
      state.latestSequence = sequence;
      state.osc = osc;
      if (includeAuxiliary) {
        state.pi05 = pi05;
        state.pico = pico;
      }
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
      state.osc = osc;
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
    try { state.osc = (await api("/api/osc/session/stop", "POST", { reason: "WebAdapter disconnected" })).state; render(); } catch (error) { phase(`WebAdapter 断开失败：${error.message}`, true); }
  }

  function pi05ConfigBody() {
    return { model: { prompt: $("pi05-prompt")?.value || "" }, cameras: {
      external: { index: Number($("pi05-external-index")?.value), width: 640, height: 480 },
      wrist: { index: Number($("pi05-wrist-index")?.value), width: 640, height: 480 },
    }};
  }

  async function activateSharedCameras() {
    try {
      await api("/api/cameras/config", "POST", cameraConfigBody());
      state.cameras = await api("/api/cameras/activate", "POST", {} , 10000);
      renderPi05();
    } catch (error) { phase(`公共相机接入失败：${error.message}`, true); await refresh(); }
  }

  async function deactivateSharedCameras() {
    try {
      state.cameras = await api("/api/cameras/deactivate", "POST", {}, 10000);
      ["pi05", "pico"].forEach((prefix) => { ["external", "wrist"].forEach((source) => { const image = $(`${prefix}-${source}-frame`); if (image) { image.removeAttribute("src"); image.classList.remove("ready"); } }); });
      render();
    } catch (error) { phase(`关闭相机失败：${error.message}`, true); }
  }

  function renderCameraControls() {
    const ready = Boolean(state.cameras?.ready);
    ["pi05", "pico"].forEach((scope) => {
      const status = $(`camera-power-${scope}`); if (status) status.textContent = ready ? "相机已打开" : "相机已关闭";
      const open = $(`camera-open-${scope}`), close = $(`camera-close-${scope}`);
      if (open) open.disabled = ready; if (close) close.disabled = !ready;
    });
  }

  async function startPi05() {
    try {
      let current = session();
      if (current.state !== "ACTIVE" || current.client_id !== clientId) {
        const result = await api("/api/osc/session/start", "POST", { execution_mode: $("execution-mode").value, client_id: clientId }, 10000);
        state.osc = result.state; state.latestSequence = Number(result.state?.state_sequence || 0); current = session();
      }
      await api("/api/pi05/config", "POST", pi05ConfigBody());
      state.pi05 = await api("/api/pi05/start", "POST", { session_id: current.id, client_id: clientId }, 10000);
      phase("π0.5 已接入 OSC；持续推理与 Action Chunk 执行中"); render();
    } catch (error) { phase(`π0.5 启动失败：${error.message}`, true); await refresh(); }
  }

  async function stopPi05() {
    try {
      state.pi05 = await api("/api/pi05/stop", "POST", { reason: "operator stopped pi05 adapter" });
      const result = await api("/api/osc/session/stop", "POST", { reason: "pi05 adapter stopped" });
      state.osc = result.state; render();
    } catch (error) { phase(`π0.5 停止失败：${error.message}`, true); }
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
        state.osc = osc;
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
          state.osc = result.state;
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

  function renderHierarchy(osc, broker, transport, robot, diagnostic, current) {
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
    void osc; void transport; void robot; void diagnostic;
  }

  function permissionReason() {
    const current = session();
    const broker = state.osc?.authority || {};
    const diagnostic = state.osc?.diagnostics || {};
    if (current.state !== "ACTIVE") return "会话未处于 ACTIVE";
    if (current.client_id !== clientId) return "会话属于其他客户端";
    if (broker.hardware_mode === "FAULT" || diagnostic.trajectory_state === "FAULT") return `FAULT · ${broker.reason || diagnostic.trajectory_brake_reason || "轨迹故障"}`;
    if (state.osc?.execution?.accepting_targets !== true) return "后端未开启 OSC 输入权限";
    if (!isShadowSession(current) && broker.servo_mode !== "TRACKING") return `等待进入 ${broker.servo_mode || "TRACKING"}`;
    if (diagnostic.trajectory_state !== "RUNNING") return `等待轨迹恢复（${diagnostic.trajectory_state || "unknown"}）`;
    return "";
  }

  attachStick("xy");
  attachStick("right");
  buildPi05Card();
  buildPicoCard();
  void loadSharedCameras();
  $("input-adapter")?.addEventListener("change", applyAdapterSelection);
  $("execution-mode")?.addEventListener("change", () => { resetWebAdapter(); render(); });
  $("right-mode")?.addEventListener("change", () => {
    // Changing the axis mapping must not clear the active joystick or the
    // accumulated relative TCP pose. Only the interpretation of the right
    // stick changes; the current target remains continuous across the switch.
    state.rightMode = $("right-mode").value;
    updateRightLabels();
    updateInputView();
    if (state.webAdapterActive) requestIntent();
  });
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
  $("pi05-cameras")?.addEventListener("click", activateSharedCameras);
  $("pi05-start")?.addEventListener("click", startPi05);
  $("pi05-stop")?.addEventListener("click", stopPi05);
  $("pico-start")?.addEventListener("click", startPico);
  $("pico-stop")?.addEventListener("click", stopPico);
  ["pi05", "pico"].forEach((scope) => {
    $(`camera-open-${scope}`)?.addEventListener("click", activateSharedCameras);
    $(`camera-close-${scope}`)?.addEventListener("click", deactivateSharedCameras);
  });
  const oscCommand = (type, payload = {}) => {
    const current = session();
    const sequence = nextOscSequence();
    return api("/api/osc/command", "POST", { session_id: current.id, client_id: clientId, sequence, type, payload });
  };
  $("hold").onclick = () => { resetWebAdapter(); $("hold").disabled = true; oscCommand("hold", { reason: "operator requested HOLD" }).then((result) => { state.osc = result.state; render(); }).catch((error) => phase(`HOLD失败：${error.message}`, true)); };
  $("freedrive").onclick = () => {
    // Clear the local UI only.  FREEDRIVE must not first submit a zero
    // osc intent, because that would start a P1/Ruckig braking path before
    // the dedicated direct Leader transition reaches the backend.
    resetWebAdapter();
    $("freedrive").disabled = true;
    oscCommand("freedrive", { reason: "operator requested FREEDRIVE" })
      .then((result) => { state.osc = result.state; render(); })
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
  applyAdapterSelection();
  refresh();
  // OSC state is cached and published without an SDK read, so the console
  // can consume the same 50 Hz state cadence as the servo loop.
  setInterval(() => { void refresh(false); }, 20);
  setInterval(() => { void refresh(true); }, 500);
  setInterval(refreshPi05Frames, 200);
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
