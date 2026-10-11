/* Shared cloud/local inference extras. No OSC command or hardware API calls. */
(() => {
  const select=document.getElementById('input-adapter');
  const panel=document.getElementById('pi05-panel');
  if (!select || !panel) return;
  const option=document.createElement('option');
  option.value='pi05-local';option.textContent='5090本地推理';
  select.add(option,select.querySelector('option[value="pico"]'));
  const form=document.createElement('div');form.className='pi05-scale';
  form.innerHTML='<label>推理服务地址<input id="inference-host" value="127.0.0.1"></label><label>端口<input id="inference-port" type="number" min="1" max="65535" value="8000"></label><button id="inference-endpoint-save" class="button" type="button">保存连接</button><label>最大到达等待（ms）<input id="inference-wait" type="number" min="0" max="5000" step="10" value="100"></label><button id="inference-wait-save" class="button" type="button">保存等待设置</button><small>本地与云端共用；进入3 mm／2°容差即继续，超时继续，0表示不等待。</small><small id="inference-extension-status"></small>';
  panel.querySelector('.pi05-inference').append(form);
  const exportButton=document.createElement('button');exportButton.className='button';
  exportButton.textContent='导出日志';exportButton.id='inference-log-export';
  document.getElementById('pi05-stop').after(exportButton);
  const $=id=>document.getElementById(id);
  let busy=false,propagate=false;
  let modelConnected=false;
  const profile=()=>select.value==='pi05-local'?'local5090':'autodl';
  const message=text=>{$('inference-extension-status').textContent=text;};
  async function api(path,body) {
    const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const result=await r.json();if(!r.ok || !result.ok)throw new Error(result.error || '请求失败');return result.data;
  }
  function renderProfile() {
    const local=profile()==='local5090',name=local?'5090本地推理':'AutoDL云端推理';
    const title=panel.querySelector('.pi05-inference .pi05-head strong');if(title)title.textContent=name;
    if (!$('pi05-start').disabled)$('pi05-start').textContent='开始 '+name;
    $('pi05-stop').textContent='停止 '+name;
    if(local) {
      for(const id of ['pi05-connection-ssh','pi05-connection-policy']) {
        const node=$(id);if(!node)continue;
        node.className=`pi05-connection-node ${modelConnected?'ok':'bad'}`;
        node.replaceChildren();
        for(const [tag,text] of [['b',id.endsWith('ssh')?'5090本地直连':'5090 Policy WebSocket'],
          ['small',`${$('inference-host').value}:${$('inference-port').value}`],['em',modelConnected?'已连接；无需SSH转发':'等待本地服务连接']]) {
          const element=document.createElement(tag);element.textContent=text;node.append(element);
        }
      }
    }
  }
  async function choose(edit=false) {
    if(busy)return;busy=true;
    try {
      const request={profile:profile()};
      if(edit)Object.assign(request,{host:$('inference-host').value.trim(),port:Number($('inference-port').value)});
      const state=await api('/api/pi05/config',{inference_profile:request});
      const endpoint=state.config.inference_profiles[profile()];
      $('inference-host').value=endpoint.host;$('inference-port').value=endpoint.port;
      message('连接已保存；没有启动运动');
      propagate=true;select.dispatchEvent(new Event('change',{bubbles:true}));propagate=false;
    } catch(error){message(error.message);}finally{busy=false;renderProfile();}
  }
  select.addEventListener('change',event=>{
    if(propagate || !['pi05','pi05-local'].includes(select.value))return;
    event.stopImmediatePropagation();void choose();
  },true);
  $('pi05-start').addEventListener('click',event=>{
    if(busy){event.stopImmediatePropagation();message('等待连接设置完成后再开始');}
  },true);
  $('inference-endpoint-save').onclick=()=>choose(true);
  $('inference-wait-save').onclick=async()=>{
    const value=$('inference-wait').value.trim(),ms=Number(value);
    if(!value || !Number.isFinite(ms) || ms<0 || ms>5000){message('请输入0–5000 ms');return;}
    try{await api('/api/pi05/config',{execution:{arrival_wait_s:ms/1000}});message(`最大等待已保存：${ms} ms（云端与本地共用）`);}catch(error){message(error.message);}
  };
  exportButton.onclick=async()=>{
    exportButton.disabled=true;
    try {
      const r=await fetch('/api/pi05/log/export');
      if(!r.ok){const error=await r.json();throw new Error(error.error || '请先停止一轮推理');}
      const url=URL.createObjectURL(await r.blob());const link=document.createElement('a');
      link.href=url;link.download=r.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] || 'inference-log.zip';
      link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);message('上一轮日志已导出');
    }catch(error){message(error.message);}finally{exportButton.disabled=false;}
  };
  // Correct help for simultaneous local service + independent cloud tunnel.
  const help=panel.querySelector('.pi05-help');
  if(help)help.innerHTML=help.innerHTML.replace('-L 8000:127.0.0.1:8000','-L 8001:127.0.0.1:8000').replace('-Port 8000','-Port 8001');
  setInterval(()=>{
    if(!['pi05','pi05-local'].includes(select.value))return;
    renderProfile();
    api('/api/pi05/state').then(state=>{modelConnected=state.model_state==='CONNECTED';}).catch(()=>{modelConnected=false;});
  },1000);
  api('/api/pi05/state').then(state=>{
    modelConnected=state.model_state==='CONNECTED';
    $('inference-wait').value=Math.round((state.config.execution.arrival_wait_s ?? .1)*1000);
  }).catch(()=>{});
})();
