/* Shared inference view: presets are configuration, never separate adapters. */
(function(root) {
  "use strict";
  const get = id => document.getElementById(id);
  const fields = ["inference-host", "inference-port", "inference-horizon", "inference-prefix", "inference-wait", "inference-mode", "inference-delay", "inference-execute-horizon"];
  let apiCall, acceptState, report, latest = null, busy = false, selected = null, notice = "";
  const drafts = new Map();
  const names = {local5090:"5090 本地", autodl:"AutoDL 云端"};
  function build() {
    if (get("pi05-panel")) return;
    const panel = document.createElement("section");
    panel.id = "pi05-panel"; panel.className = "pi05-panel hidden unified-inference";
    panel.innerHTML = `
      <section class="inference-settings">
        <div class="pi05-head"><div><strong>连接与模型设置</strong><small>两个连接预设，共用 NERO 7D 输入与动作协议</small></div><span id="inference-protocol" class="badge neutral">待连接验证</span></div>
        <div class="inference-settings-grid">
          <label>连接预设<select id="inference-profile"><option value="local5090">5090 本地</option><option value="autodl">AutoDL 云端</option></select></label>
          <label>推理方式<select id="inference-mode"><option value="normal">普通</option><option value="rtc">RTC 异步推理</option></select></label>
          <label class="inference-address">推理地址<input id="inference-host" autocomplete="off" placeholder="127.0.0.1"></label>
          <label>端口<input id="inference-port" type="number" min="1" max="65535" step="1"></label>
          <label>预测长度 H<input id="inference-horizon" type="number" min="1" step="1"></label>
          <label>每块执行步数 K<input id="inference-prefix" type="number" min="1" step="1"></label>
          <label>段末到位等待上限（ms）<input id="inference-wait" type="number" min="0" max="5000" step="1"></label>
          <label>RTC 推理延迟 inference_delay（步）<input id="inference-delay" type="number" min="0" max="19" step="1"></label>
          <label>RTC 执行长度 execute_horizon（步）<input id="inference-execute-horizon" type="number" min="1" max="19" step="1"></label>
        </div>
        <div class="inference-settings-footer"><small>H 必须匹配模型返回长度；修改这里不会改变模型本身。1 ≤ K ≤ H。</small><button id="inference-save" class="button" type="button">保存并应用</button></div>
        <p id="inference-save-result" class="result" role="status"></p>
        <p id="inference-mode-help" class="result">普通模式执行前 K 行；RTC 连续发送完整时间轴中的目标。</p>
      </section>
      <section class="pi05-connection"><div class="pi05-connection-title"><span>连接状态</span><small id="pi05-connection-message">等待连接</small></div><p id="inference-connection-active"></p><div class="pi05-connection-nodes"><article id="pi05-connection-ssh"></article><i aria-hidden="true">→</i><article id="pi05-connection-policy"></article></div></section>
      ${root.NeroInferenceHelp.markup()}
      <div class="pi05-adapter-row">
        <section class="pi05-observation"><div class="pi05-head"><div><strong>观测输入</strong><small>双 RGB · 224 × 224 · float32 状态7维</small></div></div>
          <div class="pi05-cameras"><label class="pi05-camera-select">外部 RGB<select id="pi05-external-index"></select></label><label class="pi05-camera-select">腕部 RGB<select id="pi05-wrist-index"></select></label></div>
          <div class="camera-power"><span id="camera-power-pi05">相机状态：检查中</span><button id="camera-open-pi05" class="button" type="button">打开相机</button><button id="camera-close-pi05" class="button quiet" type="button">关闭相机</button></div>
          <div class="pi05-views"><div class="pi05-view"><span>外部视角</span><img id="pi05-external-frame" alt="外部 RGB 实时画面"><b id="pi05-external-preview">等待画面</b></div><div class="pi05-view"><span>腕部视角</span><img id="pi05-wrist-frame" alt="腕部 RGB 实时画面"><b id="pi05-wrist-preview">等待画面</b></div></div>
          <div class="pi05-prompt"><label for="pi05-prompt">任务指令</label><textarea id="pi05-prompt" maxlength="500"></textarea><div class="pi05-prompt-footer"><small id="pi05-active-prompt">等待服务状态</small><button id="pi05-prompt-save" class="button" type="button">保存任务文本</button></div></div>
        </section>
        <div class="pi05-inference-stack"><section class="pi05-inference">
          <div class="pi05-head"><strong>运行状态</strong><strong id="pi05-model-state">未连接</strong></div>
          <div class="pi05-metrics"><span>状态<b id="pi05-run-state">IDLE</b></span><span>推理耗时<b id="pi05-inference-ms">--</b></span><span>Chunk 序号<b id="pi05-chunk-length">0</b></span><span>已执行行<b id="pi05-executed">0</b></span></div>
          <p id="inference-preview-state" class="result" role="status">动作预览：等待连接与观测</p>
          <p id="inference-control-state" class="result" role="status">控制状态：未开始</p>
          <p id="inference-control-message" class="result fault" role="alert" hidden></p>
          <p id="inference-chunk-source" class="result">暂无动作块</p>
        </section><div class="pi05-chunk"><span>动作预览</span><div class="pi05-action-chunk"><span>前 5 步 · 位置／旋转增量与夹爪开度</span><pre id="pi05-action-chunk-rows">等待推理结果</pre></div></div></div>
      </div>
      <div class="pi05-adapter-actions"><button id="pi05-start" class="button primary" type="button" disabled>开始推理控制</button><button id="pi05-stop" class="button quiet" type="button" disabled>停止推理控制</button><button id="inference-export" class="button quiet" type="button" disabled>导出日志</button></div>
      <p id="inference-start-reason" class="result" role="status"></p>
      <p id="pi05-result" class="result" role="status"></p>
      <details class="inference-details"><summary>技术详情</summary>
        <p id="inference-model-report" class="result"></p>
        <div class="inference-live-grid"><span>当前端点<code id="inference-live-endpoint">--</code></span><span>生效预设<b id="inference-live-profile">--</b></span><span>生效 H / K<b id="inference-live-lengths">--</b></span><span>最近返回步数<b id="inference-returned-steps">--</b></span></div>
        <p id="inference-protocol-detail" class="result"></p>
        <p id="inference-training-detail" class="result"></p>
        <p id="inference-rtc-state" class="result"></p>
        <p id="inference-chunk-detail" class="result"></p>
        <p id="inference-error-detail" class="result"></p>
        <code id="pi05-action-first"></code><code id="pi05-chunk-base"></code>
        <p id="pi05-rejection" class="result"></p><p id="pi05-websocket-diagnostic" class="result"></p>
        <p id="inference-log-state" class="result"></p>
      </details>`;
    document.querySelector(".osc-panel")?.append(panel);
  }
  function savedDraft(state, name) {
    const p = state.config.inference_profiles[name];
    return {host:p.host, port:String(p.port), horizon:String(p.expected_chunk_steps), prefix:String(p.replan_steps),
      wait:String(1000*(state.config.execution.arrival_wait_s ?? .1)),mode:state.config.execution.mode||"normal",
      inferenceDelay:String(state.config.execution.inference_delay??8),executeHorizon:String(state.config.execution.execute_horizon??10)};
  }
  function draft() {
    return {host:get(fields[0]).value,port:get(fields[1]).value,horizon:get(fields[2]).value,
      prefix:get(fields[3]).value,wait:get(fields[4]).value,mode:get(fields[5]).value,inferenceDelay:get(fields[6]).value,executeHorizon:get(fields[7]).value};
  }
  function fill(value) { [value.host,value.port,value.horizon,value.prefix,value.wait,value.mode,value.inferenceDelay,value.executeHorizon].forEach((v,i) => {get(fields[i]).value=v;}); }
  function dirty() {
    if (!latest || !selected) return true;
    return selected !== latest.config.inference_profiles.active || JSON.stringify(draft()) !== JSON.stringify(savedDraft(latest, selected));
  }
  function canStart() {
    const readiness=latest?.control_readiness?.[get("execution-mode")?.value||"shadow"];
    if(readiness && !readiness.allowed) return false;
    return Boolean(latest && !busy && !dirty() && !latest.connection_switching && !latest.execution_enabled &&
      !latest.priming && !["STOPPING","HOLD_PENDING","HOLD_UNCONFIRMED"].includes(latest.rtc?.state) && latest.camera_ready && latest.model_state === "CONNECTED" && latest.protocol_compatibility?.status === "compatible");
  }
  function controls() {
    const locked = busy || latest?.execution_enabled || latest?.priming || latest?.connection_switching;
    for (const id of [...fields,"inference-profile","inference-save"]) get(id).disabled = Boolean(locked);
    const rtc=get("inference-mode").value==="rtc";
    get("inference-prefix").disabled=Boolean(locked||rtc);
    get("inference-wait").disabled=Boolean(locked||rtc);
    get("inference-delay").disabled=Boolean(locked||!rtc);
    get("inference-execute-horizon").disabled=Boolean(locked||!rtc);
    get("pi05-prompt-save").disabled=Boolean(locked&&latest?.config.execution.mode==="rtc");
    get("inference-mode-help").textContent=rtc?"每步 50 ms。inference_delay：旧轨迹保留步数；execute_horizon：每轮总执行步数（含旧轨迹）。引导重叠长度 = H − execute_horizon。K 和段末到位等待不参与 RTC。":"普通模式：请求、等待、执行前 K 行；段末最多等待指定时间。RTC 参数仅用于 RTC。";
    get("pi05-start").disabled = !canStart();
    if (!busy) get("inference-save-result").textContent = notice || (dirty() ? "有未保存的设置；保存并验证模型响应后可开始控制" : "");
    get("inference-start-reason").textContent=startReason();
  }
  async function save() {
    if (busy || latest?.execution_enabled || latest?.priming) return;
    const value = draft(), H = Number(value.horizon), K = Number(value.prefix), port = Number(value.port), wait = Number(value.wait), inferenceDelay=Number(value.inferenceDelay), executeHorizon=Number(value.executeHorizon);
    if (!value.host.trim() || !Number.isSafeInteger(port) || port<1 || port>65535 ||
        !Number.isSafeInteger(H) || !Number.isSafeInteger(K) || K<1 || H<K ||
        !value.wait.trim() || !Number.isFinite(wait) || wait<0 || wait>5000 ||
        !value.inferenceDelay.trim() || !value.executeHorizon.trim() ||
        !Number.isInteger(inferenceDelay) || !Number.isInteger(executeHorizon) || inferenceDelay<0 || executeHorizon<1 || executeHorizon>=20 || inferenceDelay>executeHorizon || inferenceDelay+executeHorizon>20 || (value.mode==="rtc"&&H!==20)) {
      notice = "保存失败：RTC 要求 H20，0 ≤ inference_delay ≤ execute_horizon < 20，两者之和 ≤ 20";
      get("inference-save-result").textContent = notice;
      return false;
    }
    busy=true; controls();
    get("inference-save-result").textContent="正在保存并应用…";
    try {
      const state = await apiCall("/api/pi05/config","POST",{inference_profile:selected,
        policy_endpoint:{host:value.host.trim(),port},
        execution:{expected_chunk_steps:H,replan_steps:K,arrival_wait_s:wait/1000,mode:value.mode,inference_delay:inferenceDelay,execute_horizon:executeHorizon}},15000);
      drafts.delete(selected); latest=state; fill(savedDraft(state,selected));
      acceptState(state);
      notice="设置已保存；正在验证模型连接与响应";
      get("inference-save-result").textContent=notice;
      return true;
    } catch (error) {
      notice=`保存失败：${error.message}`;
      get("inference-save-result").textContent=notice;
      report(error.message,true); return false;
    } finally {busy=false; controlsAfterSave();}
  }
  function controlsAfterSave() {
    const message=get("inference-save-result").textContent;
    controls(); get("inference-save-result").textContent=message;
  }
  function initialize(api, receive, notify) {
    apiCall=api;acceptState=receive;report=notify;
    root.NeroInferenceHelp.bind();
    for (const id of fields) get(id).addEventListener("input",() => { notice=""; if(selected) drafts.set(selected,draft());controls();help(); });
    get("inference-mode").onchange=()=>{
      if(get("inference-mode").value==="rtc") get("inference-horizon").value="20";
      notice="";if(selected) drafts.set(selected,draft());controls();help();
    };
    get("inference-profile").onchange=() => {
      if (!latest) return;
      if (selected) drafts.set(selected,draft());
      selected=get("inference-profile").value;
      notice="";
      fill(drafts.get(selected) || savedDraft(latest,selected));controls();help();
    };
    get("inference-save").onclick=save;
    get("inference-export").onclick=async () => {
      try {
        const response=await fetch("/api/pi05/log/export",{cache:"no-store"});
        if(!response.ok) throw new Error(await response.text());
        const url=URL.createObjectURL(await response.blob()),link=document.createElement("a");
        link.href=url;link.download=/filename="([^"]+)"/.exec(response.headers.get("Content-Disposition")||"")?.[1]||"inference-log.zip";
        document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
      }catch(error){report(`导出失败：${error.message}`,true);}
    };
  }
  function help() {
    if (!selected) return;
    root.NeroInferenceHelp.render(selected,latest.config.inference_profiles[selected],dirty(),latest.config.execution);
    connections();
  }
  function startReason() {
    if(!latest) return "等待运行状态";
    if(latest.recovery?.stop_required) return "停止尚未确认：请点击“重试停止”。只读预览继续，确认前不能重新开始控制。";
    if(["STOPPING","HOLD_PENDING","HOLD_UNCONFIRMED"].includes(latest.rtc?.state)) return "正在确认停止；HOLD 确认前不能重新开始";
    if(latest.execution_enabled) return "正在控制";
    if(latest.priming) return "正在准备控制";
    if(busy || latest.connection_switching) return "正在应用连接设置";
    if(dirty()) return "请先保存并应用设置";
    const readiness=latest.control_readiness?.[get("execution-mode")?.value||"shadow"];
    if(readiness && !readiness.allowed && readiness.reason) return readiness.reason;
    if(latest.model_state!=="CONNECTED") return "等待模型 WebSocket 连接";
    if(!latest.camera_ready) return "请先打开相机，提供推理观测";
    if(latest.protocol_compatibility?.status!=="compatible") return latest.protocol_compatibility?.status==="incompatible"?latest.protocol_compatibility.message:"等待实际推理响应通过 7D／H 验证";
    return latest.last_error ? "控制故障已保留。预览继续；可手动重新开始，或点击“退出异常”。" : "";
  }
  function connections() {
    if(!latest) return;
    const active=latest.config.inference_profiles.active, p=latest.config.inference_profiles[active], items=latest.connections||{};
    const endpoint=items.policy?.endpoint||`${p.host}:${p.port}`;
    get("inference-connection-active").textContent=`当前生效：${names[active]} · ${endpoint}`+(selected!==active?`。正在编辑 ${names[selected]}；保存并应用后切换。`:(dirty()?"。编辑内容尚未应用。":""));
    for(const [id,item,label] of [["pi05-connection-ssh",items.service_entry,"服务入口"],["pi05-connection-policy",items.policy,"模型 WebSocket"]]) {
      const node=get(id), status=["ok","bad","warn","neutral"].includes(item?.state)?item.state:"neutral";
      node.className=`pi05-connection-node ${status}`;
      // All endpoint and diagnostic data is text, never HTML.
      node.replaceChildren();
      for(const [tag,text] of [["b",label],["small",item?.subtitle||""],["code",item?.endpoint||endpoint],["em",item?.message||"等待连接检测"]]) {
        const child=document.createElement(tag);child.textContent=text;node.append(child);
      }
      if(item?.checked_at) {const time=document.createElement("small");time.textContent=`检测时间 ${new Date(item.checked_at*1000).toLocaleTimeString()}`;node.append(time);}
    }
    const entry=items.service_entry;
    get("pi05-connection-message").textContent=latest.model_state==="CONNECTED"?"推理服务连接成功":(entry?.state==="ok"?"入口已就绪；等待推理服务连接":"先启动服务或建立转发；连接会自动重试");
  }
  function readableError(error, osc) {
    if(!error) return "";
    if(/RTC.*(v2|协议不兼容)|nero\.rtc\.v1/.test(error)) return "云端 RTC 版本不兼容，请更新推理服务后再启动控制。";
    const cpv=osc?.diagnostics?.mailboxes?.cpv || osc?.transport?.cpv_mailbox;
    const cause=cpv?.last_result?.error || "";
    if(/joint limits|joint.*limit|关节.*限位/i.test(error+cause)) return "关节限位保护已触发，控制已停止。请先检查当前姿态。";
    if(/write authority|tracking authority|PermissionError/.test(error)) return "控制权限已撤销，机械臂保持停止。请检查故障详情后重新建立控制会话。";
    if(/workspace/.test(error)) return "模型目标超出允许工作范围，控制已停止。";
    if(/measured TCP timestamp missing or stale|fresh measured TCP|measured TCP unavailable/i.test(error)) return "TCP 实测观测缺失或超过 100 ms，控制已停止。请检查本地反馈刷新链路。";
    if(/camera feedback|camera.*older|camera pair skew/i.test(error)) return "相机图像缺失、过期或双路时间差过大，控制已停止。";
    if(/hardware feedback is stale/i.test(error)) return "机械臂底层反馈超过 100 ms，控制已停止。";
    if(/gripper.*stopped updating/i.test(error)) return "夹爪实测反馈停止更新，控制已停止。";
    if(/feedback|stale|image\/TCP/i.test(error)) return "观测反馈不够新鲜，控制已停止。请展开技术详情确认具体来源。";
    if(/deadline|slot|runway|exhausted/i.test(error)) return "推理或控制未赶上执行节奏，控制已停止。";
    return "控制发生异常，已停止。可在技术详情查看原因，并通过右侧按钮退出异常或重试停止。";
  }
  function render(state, context={}) {
    if (!state?.config?.inference_profiles || !get("inference-save")) return;
    const wasClean=latest && selected && !dirty();
    latest=state;
    if (!selected) {
      selected=state.config.inference_profiles.active;get("inference-profile").value=selected;
      fill(savedDraft(state,selected));
    } else if (wasClean && selected===state.config.inference_profiles.active) {
      fill(savedDraft(state,selected));
    }
    const active=state.config.inference_profiles.active, p=state.config.inference_profiles[active];
    get("inference-live-profile").textContent=names[active];
    get("inference-live-endpoint").textContent=state.connections?.policy?.endpoint || `${p.host}:${p.port}`;
    get("inference-live-lengths").textContent=`${p.expected_chunk_steps} / ${p.replan_steps}`;
    get("inference-returned-steps").textContent=state.last_response_steps ?? "--";
    const compatibility=state.model_state==="CONNECTED"?(state.protocol_compatibility || {status:"pending",message:"待响应验证"}):{status:"pending",message:"待连接后验证 7D／H"};
    if (notice==="设置已保存；正在验证模型连接与响应" && compatibility.status==="compatible" && state.last_response_steps) notice="设置已保存，模型响应验证通过";
    get("inference-protocol").textContent=compatibility.status==="compatible"?"协议就绪":compatibility.status==="incompatible"?"协议不兼容":"待验证";
    get("inference-protocol-detail").textContent=compatibility.message;
    get("inference-protocol").className=`badge ${compatibility.status==="compatible"?"ok":compatibility.status==="incompatible"?"fault":"neutral"}`;
    const metadata=state.policy_metadata || {};
    get("inference-model-report").textContent=metadata.deployment_label || metadata.checkpoint_dir ? `模型报告：${metadata.deployment_label || metadata.checkpoint_dir}` : (p.checkpoint_label?`预设备注：${p.checkpoint_label}（模型未报告名称）`:"模型未报告名称");
    get("pi05-run-state").textContent=state.execution_enabled?"控制中":(state.priming?"准备控制":(state.last_error||state.state==="ERROR"?"控制故障":(state.session_id?"已停止控制":"未控制")));
    const recoverable=Boolean(state.last_error || context.startupError || state.recovery?.can_stop);
    const needsStop=Boolean(state.recovery?.stop_required || ["HOLD_PENDING","HOLD_UNCONFIRMED","STOPPING"].includes(state.rtc?.state));
    get("pi05-start").textContent=state.last_error&&!state.execution_enabled&&!state.priming?"重新开始推理控制":"开始推理控制";
    get("pi05-stop").disabled=!(state.execution_enabled || state.priming || recoverable || needsStop);
    get("pi05-stop").textContent=needsStop?"重试停止":recoverable&&!state.execution_enabled&&!state.priming?"退出异常":"停止推理控制";
    const rtc=state.rtc;
    const rtcNames={BOOTSTRAP:"初始化",RUNNING:"运行中",STOPPING:"正在停止",HOLD_PENDING:"等待保持确认",HOLD:"已保持",HOLD_UNCONFIRMED:"保持未确认",RELEASED:"已交出控制权"};
    get("inference-rtc-state").textContent=state.config.execution.mode==="rtc"?`RTC：${rtcNames[rtc?.state]||"待启动"} · 队列 ${rtc?.queue_remaining_ms??0} ms · 端到端 ${rtc?.end_to_end_ms?.toFixed(1)??"--"} ms · inference_delay=${state.config.execution.inference_delay??8} · execute_horizon=${state.config.execution.execute_horizon??10}${rtc?.reason?" · "+rtc.reason:""}${rtc?.manual_restart_required?" · 需手动重新开始":""}`:"";
    const preview=state.preview||{}, previewNames={WAITING:"等待观测",INFERENCING:"推理中",READY:"自动更新中",PAUSED:"控制期间暂停",ERROR:"预览失败"};
    get("inference-preview-state").textContent=`动作预览：${previewNames[preview.state]||"等待连接"}${preview.hz&&preview.state!=="PAUSED"?` · ${preview.hz.toFixed(1)} Hz`:""}${preview.error?" · "+preview.error:""}${preview.warning?" · "+preview.warning:""}`;
    get("inference-control-state").textContent=`控制状态：${state.execution_enabled?"控制中":state.priming?"准备控制":state.last_error?"已停止":context.startupError?"启动失败":"未控制"}`;
    const message=get("inference-control-message");
    const controlError=state.last_error||context.startupError;
    message.textContent=context.startupError&&!state.last_error?context.startupError:readableError(controlError,context.osc);
    message.hidden=!controlError;
    get("inference-training-detail").textContent=state.training_notice||"";
    get("inference-training-detail").hidden=!state.training_notice;
    get("inference-error-detail").textContent=[state.last_error,state.last_control_error?"上次控制故障："+state.last_control_error:null,context.startupError,preview.error,(context.osc?.diagnostics?.mailboxes?.cpv?.last_result||{}).error].filter(Boolean).join("\n");
    get("inference-chunk-source").textContent=state.chunk_updated_at?`${state.chunk_source==="control"?"控制动作":"动作预测预览"}${state.chunk_stale?" · 历史结果，已停止更新":state.chunk_source!=="control"?" · 仅预览，不执行":""}`:"等待模型预测";
    get("inference-chunk-detail").textContent=state.chunk_updated_at?`第 ${state.chunk_sequence} 块 · H=${state.action_chunk_length} · ${state.inference_ms?.toFixed(1)??"--"} ms · ${new Date(state.chunk_updated_at*1000).toLocaleTimeString()}${state.chunk_source!=="control"&&state.config.execution.mode==="rtc"?" · 普通 H20 采样，未进行 RTC 衔接":""}`:"";
    if(["HOLD_PENDING","HOLD_UNCONFIRMED","STOPPING"].includes(rtc?.state)) get("pi05-stop").disabled=false;
    get("inference-export").disabled=!state.log_export_ready;
    get("inference-export").hidden=false;
    get("inference-export").title=state.log_export_ready
      ? "导出最近一轮已结束的推理日志（服务重启后仍保留）"
      : state.inference_log?.write_error
        ? `日志不可导出：${state.inference_log.write_error}`
        : "暂无完整且已结束的推理日志";
    const log=state.inference_log;
    get("inference-log-state").textContent=state.log_error || (log ? `${log.run_id} · ${log.finished?"已结束":"记录中"}${log.write_error?" · "+log.write_error:""}` : "");
    get("inference-log-state").hidden=!get("inference-log-state").textContent;
    controls();help();get("inference-start-reason").textContent=startReason();
  }
  const api={build,initialize,render,save,canStart,readableError};
  if(typeof module!=="undefined" && module.exports) module.exports=api;
  else root.NeroInferencePanel=api;
})(typeof window!=="undefined"?window:globalThis);
