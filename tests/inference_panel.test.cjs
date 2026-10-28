const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');

function fixture() {
  const nodes = new Map(), calls = [];
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, {value:'',textContent:'',disabled:false,listeners:{},children:[],
      replaceChildren() {this.children=[];},append(child) {this.children.push(child);},
      addEventListener(name, fn) {this.listeners[name]=fn;}});
    return nodes.get(id);
  };
  const context = vm.createContext({window:{},document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>({})},setTimeout,URL});
  vm.runInContext(fs.readFileSync(require.resolve('../web/console/inference_help.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(require.resolve('../web/console/inference_panel.js'),'utf8'),context);
  const panel = context.window.NeroInferencePanel;
  let state = {model_state:'CONNECTED',camera_ready:true,protocol_compatibility:{status:'compatible',message:'验证通过'},
    config:{inference_profiles:{active:'local5090',local5090:{host:'127.0.0.1',port:8000,expected_chunk_steps:13,replan_steps:4},
      autodl:{host:'127.0.0.1',port:9911,expected_chunk_steps:20,replan_steps:9}},execution:{arrival_wait_s:.1}},
    connections:{policy:{endpoint:'ws://127.0.0.1:8000'}},last_response_steps:13};
  let handler = async (url,method,body) => {
    state=structuredClone(state);state.config.inference_profiles.active=body.inference_profile;
    Object.assign(state.config.inference_profiles[body.inference_profile],body.policy_endpoint,body.execution);
    Object.assign(state.config.execution,body.execution);
    state.protocol_compatibility={status:'pending',message:'待响应验证'};
    return state;
  };
  panel.initialize(async (...args)=>{calls.push(args);return handler(...args);},s=>{state=s;panel.render(s);},()=>{});
  panel.render(state);
  const input = (id,value) => {element(id).value=value;element(id).listeners.input();};
  return {panel,nodes,calls,input,state:()=>state,element,route:fn=>{handler=fn;}};
}

test('unified preset selection preserves drafts, uses saved H/K and blocks unsaved start',async()=>{
  const f=fixture();assert.equal(f.panel.canStart(),true);
  f.input('inference-horizon','17');f.input('inference-prefix','6');
  f.panel.render(f.state());assert.equal(f.element('inference-horizon').value,'17');
  assert.equal(f.panel.canStart(),false);
  f.element('inference-profile').value='autodl';f.element('inference-profile').onchange();
  assert.equal(f.element('inference-port').value,'9911');assert.equal(f.element('inference-horizon').value,'20');
  assert.match(f.element('inference-help-command').textContent,/-L '127.0.0.1:9911:127.0.0.1:<policy-port>'/);
  f.element('inference-profile').value='local5090';f.element('inference-profile').onchange();
  assert.equal(f.element('inference-horizon').value,'17');
  await f.panel.save();
  assert.deepEqual(JSON.parse(JSON.stringify(f.calls[0][2])),{inference_profile:'local5090',policy_endpoint:{host:'127.0.0.1',port:8000},
    execution:{expected_chunk_steps:17,replan_steps:6,arrival_wait_s:.1,mode:'normal',inference_delay:8,execute_horizon:10}});
  assert.equal(f.panel.canStart(),false);
});

test('invalid groups and save failure retain editable values',async()=>{
  const f=fixture();f.input('inference-prefix','14');await f.panel.save();
  assert.equal(f.calls.length,0);
  f.input('inference-prefix','4');f.input('inference-host','my-model');
  f.route(async()=>{throw new Error('disk full');});await f.panel.save();
  f.panel.render(f.state());assert.equal(f.element('inference-host').value,'my-model');
  assert.match(f.element('inference-save-result').textContent,/disk full/);
  assert.equal(f.state().config.inference_profiles.local5090.host,'127.0.0.1');
  assert.equal(f.panel.canStart(),false);
});

test('RTC selects H20, preserves drafts, locks normal-only controls and saves mode',async()=>{
  const f=fixture();
  f.element('inference-mode').value='rtc';f.element('inference-mode').onchange();
  assert.equal(f.element('inference-horizon').value,'20');
  assert.equal(f.element('inference-prefix').disabled,true);
  assert.equal(f.element('inference-wait').disabled,true);
  assert.equal(f.element('inference-delay').disabled,false);
  assert.equal(f.element('inference-execute-horizon').disabled,false);
  f.panel.render(f.state());
  assert.equal(f.element('inference-mode').value,'rtc');
  assert.equal(f.panel.canStart(),false);
  await f.panel.save();
  assert.equal(f.calls[0][2].execution.mode,'rtc');
  assert.equal(f.calls[0][2].execution.expected_chunk_steps,20);
  const s=structuredClone(f.state());s.rtc={state:'HOLD_UNCONFIRMED'};
  s.protocol_compatibility={status:'compatible',message:'RTC'};f.panel.render(s);
  assert.equal(f.panel.canStart(),false);
  assert.equal(f.element('pi05-stop').disabled,false);
});

test('active output and priming lock settings; connection alone cannot enable output',()=>{
  const f=fixture();
  for(const flag of ['execution_enabled','priming']) {
    const state=structuredClone(f.state());state[flag]=true;f.panel.render(state);
    assert.equal(f.element('inference-profile').disabled,true);
    assert.equal(f.element('inference-save').disabled,true);
    assert.equal(f.element('pi05-stop').disabled,false);
  }
  const state=structuredClone(f.state());state.execution_enabled=false;state.priming=false;
  state.protocol_compatibility={status:'incompatible',message:'state8 is incompatible'};f.panel.render(state);
  assert.equal(f.panel.canStart(),false);
  assert.equal(f.element('inference-protocol').textContent,'协议不兼容');
  assert.equal(f.element('inference-protocol-detail').textContent,'state8 is incompatible');
});

test('entry and websocket cards show independent evidence and drafts never relabel active connection',()=>{
  const f=fixture(), state=structuredClone(f.state());
  state.model_state='DISCONNECTED';state.websocket_error='HTTP 200';
  state.connections={service_entry:{state:'ok',endpoint:'127.0.0.1:8000',message:'已有监听'},
    policy:{state:'bad',endpoint:'ws://127.0.0.1:8000',message:'HTTP 200'}};
  f.panel.render(state);
  assert.match(f.element('pi05-connection-ssh').className,/ ok$/);
  assert.match(f.element('pi05-connection-policy').className,/ bad$/);
  assert.equal(f.element('inference-protocol').className,'badge neutral');
  f.element('inference-profile').value='autodl';f.element('inference-profile').onchange();
  assert.match(f.element('inference-connection-active').textContent,/当前生效：5090 本地/);
  assert.match(f.element('inference-connection-active').textContent,/正在编辑 AutoDL/);
  f.input('inference-port','12345');
  assert.match(f.element('inference-help-command').textContent,/:9911:/);
  assert.match(f.element('inference-help-draft').textContent,/已保存值/);
});

test('automatic preview is separate from hardware control readiness and shows stale chunks',()=>{
  const f=fixture(),state=structuredClone(f.state());
  state.preview={state:'READY',sequence:7,updated_at:Date.now()/1000,hz:3.8};
  state.chunk_source='preview';state.chunk_sequence=7;state.action_chunk_length=13;
  state.chunk_updated_at=Date.now()/1000;state.chunk_stale=true;state.inference_ms=190;
  state.control_readiness={shadow:{allowed:false,reason:'缺少有效观测'},hardware:{allowed:false,reason:'训练契约未核验'}};
  f.panel.render(state);
  assert.equal(f.panel.canStart(),false);
  assert.match(f.element('inference-preview-state').textContent,/自动更新中.*3.8 Hz/);
  assert.match(f.element('inference-control-state').textContent,/未控制/);
  assert.match(f.element('inference-chunk-source').textContent,/动作预测预览.*历史结果/);
});

test('RTC hardware start is enabled with valid observations even when training provenance is unverified',()=>{
  const f=fixture(),s=structuredClone(f.state());
  s.config.execution.mode='rtc';s.config.inference_profiles.local5090.expected_chunk_steps=20;
  s.policy_metadata={training_contract_verified:false};
  s.control_readiness={hardware:{allowed:true,reason:null}};
  s.training_notice='训练时间对齐未核验；不阻止手动控制';
  f.element('execution-mode').value='hardware';
  // Use a clean initial render so the saved RTC draft is authoritative.
  f.panel.render(s);
  assert.equal(f.panel.canStart(),true);
  assert.doesNotMatch(f.element('inference-control-state').textContent,/未核验/);
  assert.match(f.element('inference-training-detail').textContent,/未核验/);
});

test('control faults remain visible in plain Chinese while raw diagnostics and empty logs stay out of the main status',()=>{
  const f=fixture(),s=structuredClone(f.state());
  const raw='Model chunk execution rejected: PermissionError: OSC command rejected: cannot resume from HOLD_READY (SERVO write authority is unavailable)';
  s.last_error=raw;s.training_notice='训练契约未核验';
  f.panel.render(s,{osc:{diagnostics:{mailboxes:{cpv:{last_result:{error:'CPV reference moves outside joint limits'}}}}}});
  assert.match(f.element('inference-control-message').textContent,/关节限位保护/);
  assert.equal(f.element('inference-control-message').hidden,false);
  assert.equal(f.element('inference-control-state').textContent,'控制状态：已停止');
  assert.match(f.element('inference-error-detail').textContent,/SERVO write authority/);
  assert.equal(f.element('inference-log-state').textContent,'');
  assert.equal(f.element('inference-log-state').hidden,true);
  assert.equal(f.element('inference-export').hidden,false);
  assert.equal(f.element('inference-export').disabled,true);
  assert.match(f.panel.readableError(raw),/控制权限已撤销/);
});

test('export stays visible and a restored completed log enables download',()=>{
  const f=fixture(),s=structuredClone(f.state());
  s.log_export_ready=false;
  f.panel.render(s);
  assert.equal(f.element('inference-export').hidden,false);
  assert.equal(f.element('inference-export').disabled,true);
  s.log_export_ready=true;
  s.inference_log={run_id:'20261027T155555-92dbe55b',finished:true,export_ready:true};
  f.panel.render(s);
  assert.equal(f.element('inference-export').hidden,false);
  assert.equal(f.element('inference-export').disabled,false);
  assert.match(f.element('inference-export').title,/重启后仍保留/);
  s.log_export_ready=false;s.inference_log.write_error='disk full';
  f.panel.render(s);
  assert.equal(f.element('inference-export').disabled,true);
  assert.match(f.element('inference-export').title,/disk full/);
});

test('startup rejection stays visible through successful preview refreshes',()=>{
  const f=fixture(),s=structuredClone(f.state());
  const context={startupError:'模型推理启动失败：云端 RTC 协议不兼容，需要 nero.rtc.v2'};
  f.panel.render(s,context);
  f.panel.render({...s,last_error:null,chunk_sequence:99},context);
  assert.equal(f.element('inference-control-message').hidden,false);
  assert.match(f.element('inference-control-message').textContent,/启动失败.*v2/);
  assert.match(f.element('inference-error-detail').textContent,/启动失败/);
});

test('fault recovery always offers stop retry or exit while preview stays independent',()=>{
  const f=fixture(),s=structuredClone(f.state());
  s.last_error='injected control fault';s.execution_enabled=false;s.priming=false;
  s.preview={state:'READY',hz:4,sequence:8};
  s.rtc={state:'HOLD_UNCONFIRMED'};
  s.recovery={stop_required:true,can_stop:true};
  s.control_readiness={shadow:{allowed:false,reason:'请重试停止'}};
  f.panel.render(s);
  assert.equal(f.element('pi05-stop').disabled,false);
  assert.equal(f.element('pi05-stop').textContent,'重试停止');
  assert.equal(f.element('pi05-start').disabled,true);
  assert.match(f.element('inference-preview-state').textContent,/自动更新中/);
  s.rtc={state:'HOLD',hold_confirmed:true};s.recovery={stop_required:false,can_stop:true};
  s.control_readiness={shadow:{allowed:true,reason:null}};
  f.panel.render(s);
  assert.equal(f.element('pi05-stop').textContent,'退出异常');
  assert.equal(f.element('pi05-start').disabled,false);
  assert.equal(f.element('pi05-start').textContent,'重新开始推理控制');
});
