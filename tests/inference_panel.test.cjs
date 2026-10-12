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
    state.config.execution.arrival_wait_s=body.execution.arrival_wait_s;
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
    execution:{expected_chunk_steps:17,replan_steps:6,arrival_wait_s:.1}});
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
  assert.equal(f.element('inference-protocol').textContent,'state8 is incompatible');
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
