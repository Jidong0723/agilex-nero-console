/* UI extension limited to the inference panel; OSC maintenance is untouched. */
window.NeroInferencePanel = (() => {
  const get = (id) => document.getElementById(id);
  let busy = false;
  let apiCall;
  let acceptState;
  let report;
  let cloudHelpHtml;
  const profile = () => get('input-adapter')?.value === 'pi05-local' ? 'local5090' : 'autodl';
  const update = async (body) => {
    busy = true;
    get('pi05-start').disabled = true;
    try {
      const state = await apiCall('/api/pi05/config', 'POST', body, 10000);
      for (const id of ['inference-host','inference-port','inference-wait']) delete get(id).dataset.dirty;
      acceptState(state);
      return true;
    } catch (error) { report(error.message, true); return false; }
    finally { busy = false; }
  };
  function initialize(api, receive, notify) {
    apiCall = api; acceptState = receive; report = notify;
    cloudHelpHtml = document.querySelector('#pi05-panel .pi05-help').innerHTML;
    const selector = get('input-adapter');
    selector.querySelector('option[value="pi05"]').insertAdjacentHTML('afterend', '<option value="pi05-local">5090本地推理</option>');
    const settings = document.createElement('div');
    settings.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:12px';
    settings.innerHTML = '<label>推理地址<input id="inference-host" value="127.0.0.1"></label><label>端口<input id="inference-port" type="number" min="1" max="65535" value="8001"></label><button id="inference-save-endpoint" class="button" type="button">保存连接</button><label>最大到达等待（ms）<input id="inference-wait" type="number" min="0" max="5000" step="10" value="150"></label><button id="inference-save-wait" class="button" type="button">保存等待</button><p id="inference-log-state" class="result" style="flex-basis:100%"></p>';
    document.querySelector('#pi05-panel .pi05-inference').append(settings);
    get('pi05-stop').insertAdjacentHTML('afterend', '<button id="inference-export" class="button" type="button" disabled>导出日志</button>');
    for (const id of ['inference-host','inference-port','inference-wait']) get(id).addEventListener('input', () => { get(id).dataset.dirty = '1'; });
    get('inference-save-endpoint').onclick = () => update({inference_profile:profile(),policy_endpoint:{host:get('inference-host').value,port:Number(get('inference-port').value)}});
    get('inference-save-wait').onclick = () => update({execution:{arrival_wait_s:Number(get('inference-wait').value)/1000}});
    get('inference-export').onclick = async () => {
      try {
        const response = await fetch('/api/pi05/log/export', {cache:'no-store'});
        if (!response.ok) throw new Error(await response.text());
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement('a');
        link.href = url;
        link.download = /filename="([^"]+)"/.exec(response.headers.get('Content-Disposition') || '')?.[1] || 'inference-log.zip';
        document.body.append(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      } catch (error) { report(`导出失败：${error.message}`, true); }
    };
  }
  async function selectChanged() {
    if (!['pi05','pi05-local'].includes(get('input-adapter').value)) return;
    const selector = get('input-adapter'); selector.disabled = true;
    const ok = await update({inference_profile:profile()});
    if (!ok) {
      try {
        const current = await apiCall('/api/pi05/state','GET');
        selector.value = current.config?.inference_profiles?.active === 'local5090' ? 'pi05-local' : 'pi05';
        acceptState(current);
      } catch (error) { report(`无法确认连接配置：${error.message}`,true); }
    }
    selector.disabled = false;
  }
  function render(state) {
    if (!get('inference-export')) return;
    const config = state?.config || {};
    const local = config.inference_profiles?.active === 'local5090';
    const name = local ? '5090本地推理' : 'AutoDL云端推理';
    const help = document.querySelector('#pi05-panel .pi05-help');
    const horizon = Number(config.execution?.expected_chunk_steps || 20);
    const startCommand = horizon === 10
      ? 'Start-ScheduledTask -TaskName NERO_11649_LOCAL_INFERENCE_20261011'
      : 'wsl -d Ubuntu-22.04 -u root -- bash /opt/nero_h20_inference_test_20261010/start_h20_test.sh';
    const helpMode = local ? `local-${horizon}` : 'cloud';
    if (help && help.dataset.inferenceHelp !== helpMode) {
      help.innerHTML = local ? `<summary><span>5090本地连接帮助</span><small>Windows → WSL · 无需SSH隧道</small></summary><div class="help-content"><article class="help-step"><div class="help-title"><b>1</b><strong>检查本地模型端口</strong></div><div class="command"><code>Test-NetConnection 127.0.0.1 -Port 8000</code></div><p>无需云服务器SSH转发。</p></article><article class="help-step"><div class="help-title"><b>2</b><strong>必要时启动本地H${horizon}模型</strong></div><p>仅在服务未运行时，在5090机器PowerShell执行：</p><div class="command"><code>${startCommand}</code></div><p>${horizon === 10 ? '当前为11649检查点及配套归一化' : '当前为H20第8000步检查点及配套归一化'}，输出${horizon}×7，执行前${horizon/2}步。已连接时不要重复启动服务。</p></article><article class="help-step"><div class="help-title"><b>3</b><strong>接入控制台</strong></div><p>选择“5090本地推理”，地址<code>127.0.0.1</code>、端口<code>8000</code>。连接确认后再自行开始推理。</p><p>模型服务启动不会启动机械臂运动；不要让其他转发占用8000端口。</p></article></div>` : cloudHelpHtml;
      help.dataset.inferenceHelp = helpMode;
    }
    if (local) {
      get('pi05-connection-message').textContent = state.connections?.policy?.state === 'ok'
        ? '5090本地模型已连接（Windows → WSL），无需SSH转发'
        : '检查5090本地H20服务和127.0.0.1:8000，见本地连接帮助';
    }
    get('pi05-start').textContent = `开始 ${name}`;
    get('pi05-stop').textContent = `停止 ${name}`;
    const title = document.querySelector('#pi05-panel .pi05-inference .pi05-head strong');
    const hint = document.querySelector('#pi05-panel .pi05-inference .pi05-head small');
    if (title) title.textContent = name;
    if (hint) hint.textContent = state.policy_metadata?.deployment_label || config.model?.checkpoint_label || '';
    for (const [id,value] of [['inference-host',config.model?.host],['inference-port',config.model?.port],['inference-wait',1000*(config.execution?.arrival_wait_s ?? .15)]]) {
      const input = get(id);
      if (input !== document.activeElement && !input.dataset.dirty) input.value = String(value ?? '');
    }
    if (busy || state.connection_switching) get('pi05-start').disabled = true;
    get('inference-export').disabled = !state.log_export_ready;
    const log = state.inference_log;
    get('inference-log-state').textContent = state.log_error || (log ? `${log.run_id} · ${log.finished ? '已结束' : '记录中'}${log.write_error ? ` · 写入错误：${log.write_error}` : ''}${log.lost_records ? ` · 丢失${log.lost_records}条` : ''}` : '日志：尚未开始');
  }
  return {initialize,selectChanged,render};
})();
