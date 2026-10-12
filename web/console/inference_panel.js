/* Shared inference view: presets are configuration, never separate adapters. */
(function(root) {
  "use strict";
  const get = id => document.getElementById(id);
  const fields = ["inference-host", "inference-port", "inference-horizon", "inference-prefix", "inference-wait"];
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
          <label class="inference-address">推理地址<input id="inference-host" autocomplete="off" placeholder="127.0.0.1"></label>
          <label>端口<input id="inference-port" type="number" min="1" max="65535" step="1"></label>
          <label>预测长度 H<input id="inference-horizon" type="number" min="1" step="1"></label>
          <label>每块执行步数 K<input id="inference-prefix" type="number" min="1" step="1"></label>
          <label>最大到达等待（ms）<input id="inference-wait" type="number" min="0" max="5000" step="1"></label>
        </div>
        <div class="inference-settings-footer"><small>H 必须匹配模型返回长度；修改这里不会改变模型本身。1 ≤ K ≤ H。</small><button id="inference-save" class="button" type="button">保存并应用</button></div>
        <p id="inference-save-result" class="result" role="status"></p>
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
          <div class="pi05-head"><div><strong>运行状态</strong><small id="inference-model-report">等待模型报告</small></div><strong id="pi05-model-state">未连接</strong></div>
          <div class="inference-live-grid"><span>当前端点<code id="inference-live-endpoint">--</code></span><span>生效预设<b id="inference-live-profile">--</b></span><span>生效 H / K<b id="inference-live-lengths">--</b></span><span>最近返回步数<b id="inference-returned-steps">--</b></span></div>
          <div class="pi05-scale"><label for="pi05-action-scale">动作 Δ 缩放 <output id="pi05-action-scale-value">1.00×</output></label><input id="pi05-action-scale" type="range" min="0" max="1" step="0.05" value="1"><small>仅缩放位置／旋转增量，夹爪比例不变</small></div>
          <div class="pi05-metrics"><span>状态<b id="pi05-run-state">IDLE</b></span><span>推理耗时<b id="pi05-inference-ms">--</b></span><span>Chunk 序号<b id="pi05-chunk-length">0</b></span><span>已执行行<b id="pi05-executed">0</b></span></div>
        </section><div class="pi05-chunk"><span>NERO 动作块</span><code id="pi05-action-first">等待推理结果</code><code id="pi05-chunk-base">起点 TCP --</code><small id="pi05-rejection">--</small><small id="pi05-websocket-diagnostic">等待连接诊断</small><div class="pi05-action-chunk"><span>动作预览 · 前 5 行（原始 Δ）</span><pre id="pi05-action-chunk-rows">等待推理结果</pre></div></div></div>
      </div>
      <div class="pi05-adapter-actions"><button id="pi05-start" class="button primary" type="button" disabled>开始推理输出</button><button id="pi05-stop" class="button quiet" type="button" disabled>停止输出</button><button id="inference-export" class="button quiet" type="button" disabled>导出日志</button></div>
      <p id="inference-start-reason" class="result" role="status"></p>
      <p id="pi05-result" class="result" role="status"></p><p id="inference-log-state" class="result"></p>`;
    document.querySelector(".osc-panel")?.append(panel);
  }
  function savedDraft(state, name) {
    const p = state.config.inference_profiles[name];
    return {host:p.host, port:String(p.port), horizon:String(p.expected_chunk_steps), prefix:String(p.replan_steps),
      wait:String(1000*(state.config.execution.arrival_wait_s ?? .1))};
  }
  function draft() {
    return {host:get(fields[0]).value,port:get(fields[1]).value,horizon:get(fields[2]).value,
      prefix:get(fields[3]).value,wait:get(fields[4]).value};
  }
  function fill(value) { [value.host,value.port,value.horizon,value.prefix,value.wait].forEach((v,i) => {get(fields[i]).value=v;}); }
  function dirty() {
    if (!latest || !selected) return true;
    return selected !== latest.config.inference_profiles.active || JSON.stringify(draft()) !== JSON.stringify(savedDraft(latest, selected));
  }
  function canStart() {
    return Boolean(latest && !busy && !dirty() && !latest.connection_switching && !latest.execution_enabled &&
      !latest.priming && latest.camera_ready && latest.model_state === "CONNECTED" && latest.protocol_compatibility?.status === "compatible");
  }
  function controls() {
    const locked = busy || latest?.execution_enabled || latest?.priming || latest?.connection_switching;
    for (const id of [...fields,"inference-profile","inference-save"]) get(id).disabled = Boolean(locked);
    get("pi05-start").disabled = !canStart();
    if (!busy) get("inference-save-result").textContent = notice || (dirty() ? "有未保存的设置；保存并验证模型响应后可开始输出" : "");
    get("inference-start-reason").textContent=startReason();
  }
  async function save() {
    if (busy || latest?.execution_enabled || latest?.priming) return;
    const value = draft(), H = Number(value.horizon), K = Number(value.prefix), port = Number(value.port), wait = Number(value.wait);
    if (!value.host.trim() || !Number.isSafeInteger(port) || port<1 || port>65535 ||
        !Number.isSafeInteger(H) || !Number.isSafeInteger(K) || K<1 || H<K ||
        !value.wait.trim() || !Number.isFinite(wait) || wait<0 || wait>5000) {
      notice = "保存失败：检查地址、端口、正整数 H/K 和到达等待（0–5000 ms）";
      get("inference-save-result").textContent = notice;
      return false;
    }
    busy=true; controls();
    get("inference-save-result").textContent="正在保存并应用…";
    try {
      const state = await apiCall("/api/pi05/config","POST",{inference_profile:selected,
        policy_endpoint:{host:value.host.trim(),port},
        execution:{expected_chunk_steps:H,replan_steps:K,arrival_wait_s:wait/1000}},15000);
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
    root.NeroInferenceHelp.render(selected,latest.config.inference_profiles[selected],dirty());
    connections();
  }
  function startReason() {
    if(!latest) return "等待运行状态";
    if(latest.execution_enabled) return "正在输出";
    if(latest.priming) return "正在准备输出";
    if(busy || latest.connection_switching) return "正在应用连接设置";
    if(dirty()) return "请先保存并应用设置";
    if(latest.model_state!=="CONNECTED") return "等待模型 WebSocket 连接";
    if(!latest.camera_ready) return "请先打开相机，提供推理观测";
    if(latest.protocol_compatibility?.status!=="compatible") return latest.protocol_compatibility?.status==="incompatible"?latest.protocol_compatibility.message:"等待实际推理响应通过 7D／H 验证";
    return "模型响应验证通过，可手动开始输出";
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
  function render(state) {
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
    if (notice==="设置已保存；正在验证模型连接与响应" && compatibility.status==="compatible") notice="设置已保存，模型响应验证通过";
    get("inference-protocol").textContent=compatibility.message+(compatibility.status==="compatible"?` · H=${p.expected_chunk_steps} / 实际=${state.last_response_steps ?? "--"} / 动作7维`:"");
    get("inference-protocol").className=`badge ${compatibility.status==="compatible"?"ok":compatibility.status==="incompatible"?"fault":"neutral"}`;
    const metadata=state.policy_metadata || {};
    get("inference-model-report").textContent=metadata.deployment_label || metadata.checkpoint_dir ? `模型报告：${metadata.deployment_label || metadata.checkpoint_dir}` : (p.checkpoint_label?`预设备注：${p.checkpoint_label}（模型未报告名称）`:"模型未报告名称");
    get("pi05-run-state").textContent=state.execution_enabled?"输出中":(state.priming?"准备中":(state.last_error||state.state==="ERROR"?"错误":(state.session_id?"已停止":"未启动")));
    get("pi05-stop").disabled=!(state.execution_enabled || state.priming);
    get("inference-export").disabled=!state.log_export_ready;
    const log=state.inference_log;
    get("inference-log-state").textContent=state.log_error || (log ? `${log.run_id} · ${log.finished?"已结束":"记录中"}${log.write_error?" · "+log.write_error:""}` : "日志：尚未开始");
    controls();help();get("inference-start-reason").textContent=startReason();
  }
  const api={build,initialize,render,save,canStart};
  if(typeof module!=="undefined" && module.exports) module.exports=api;
  else root.NeroInferencePanel=api;
})(typeof window!=="undefined"?window:globalThis);
