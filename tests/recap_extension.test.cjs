const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const startup=require('../web/console/inference_startup.js');
const nodes=new Map();let moves=0;
function n(id){if(!nodes.has(id))nodes.set(id,{id,value:'',disabled:false,style:{},dataset:{},parentNode:null,
 classList:{values:new Set(),toggle(k,v){if(v)this.values.add(k);else this.values.delete(k);},remove(k){this.values.delete(k);}},
 appendChild(node){node.parentNode=this;node.previousElementSibling=null;moves++;},insertAdjacentElement(where,node){if(where==='afterend'){node.parentNode=this.parentNode;node.previousElementSibling=this;}else if(where==='beforebegin'){node.parentNode=this.parentNode;this.previousElementSibling=node;}},
 addEventListener(type,fn){this[type]=fn;},querySelector(selector){return n(this.id+selector);},
 set innerHTML(s){for(const x of s.matchAll(/id="([^"]+)"/g))n(x[1]);}});return nodes.get(id);}
n('input-adapter').value='web';n('pi05-start').disabled=false;
const pico=n('pico-panel'),cam=n('pico-panel.pico-camera-resource');cam.parentNode=pico;cam.changeHandler=()=>{};const originalHandler=cam.changeHandler;
let collection={state:'IDLE',recording:false},session={state:'IDLE'},calls=[],requests=0;
const root={NeroInferenceStartup:startup};
const ctx={window:root,document:{getElementById:n,createElement:()=>n('created-'+nodes.size)},setInterval(){},setTimeout(){},URL,fetch(){throw Error('not used');},console};
vm.runInNewContext(fs.readFileSync(require.resolve('../web/console/recap_extension.js'),'utf8'),ctx);
root.NeroRecap.init({clientId:'c',session:()=>session,mode:()=> 'shadow',context:()=> 'recap',acceptOsc:value=>{session=value.session;},phase:()=>{},
 api:async(path,method,body)=>{calls.push(path);requests++;
  if(path==='/api/osc/session/start')return {state:{session:{state:'ACTIVE',id:'s',client_id:'c',execution_mode:'shadow'}}};
  if(path==='/api/osc/state')return {session};
  if(path==='/api/recap2/start')collection={state:'MODEL',recording:true};
  if(path==='/api/recap2/takeover')collection={state:'HUMAN',recording:true};
  if(path==='/api/recap2/return')collection={state:'MODEL',recording:true};
  if(path==='/api/recap2/finish')collection={state:'FINISHED',recording:false,export_ready:true};
  return collection;}});
(async()=>{
 assert.equal(requests,0,'initialization must not call any backend');
 n('input-adapter').value='recap';root.NeroRecap.layout();
 assert.equal(cam.parentNode,n('recap-camera-area'));assert.equal(cam.changeHandler,originalHandler);
 assert(n('pi05-panel.pi05-adapter-actions').classList.values.has('hidden'));
 assert(!n('pi05-panel').classList.values.has('hidden'));
 n('recap-task').value='move';n('recap-scene').value='s01';n('recap-purpose').value='pilot';
 await n('recap-begin').onclick();assert.equal(n('recap-human').disabled,false);
 await n('recap-human').onclick();assert.equal(n('recap-model').disabled,false);
 await n('recap-model').onclick();await n('recap-success').onclick();
 assert.equal(n('recap-download').disabled,false);
 assert(!calls.includes('/api/pi05/start'));assert(!calls.includes('/api/osc/command'));
 n('input-adapter').value='pico';root.NeroRecap.layout();
 assert.equal(cam.parentNode,pico);assert(!n('pi05-panel.pi05-adapter-actions').classList.values.has('hidden'));
 console.log('RECAP lazy initialization, original cards, camera restoration and delegated lifecycle passed.');
})();
