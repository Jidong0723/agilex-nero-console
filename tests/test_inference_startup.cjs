const assert = require('node:assert/strict');
const {create} = require('../web/console/inference_startup.js');
const ready = {id:'new',state:'ACTIVE',client_id:'client',execution_mode:'hardware'};
function fixture() {
  const calls=[];let current={state:'IDLE'},context='same';
  const o={clientId:'client',mode:'hardware',session:()=>current,context:()=>context,
    acceptOsc:s=>{current=s.session;},acceptPi:()=>{},progress:()=>{},changed:()=>{},config:null,
    api:async(path,method,body,timeout)=>{
      calls.push({path,method,body,timeout});
      if(path==='/api/osc/session/start'){current=ready;return {state:{session:ready}};}
      if(path==='/api/osc/state')return {session:current};
      return {state:'PRIMING'};
    }};
  return {o,calls,setCurrent:s=>{current=s},change:()=>{context='different'}};
}
(async()=>{
  let f=fixture(),flow=create();
  assert.equal(await flow.run(f.o),true);
  assert.equal(f.calls.find(c=>c.path==='/api/osc/session/start').timeout,60000);
  assert.equal(f.calls.filter(c=>c.path==='/api/osc/session/start').length,1);
  assert(!f.calls.some(c=>c.path==='/api/pi05/config'));

  f=fixture();flow=create();f.setCurrent(ready);
  assert.equal(await flow.run(f.o),true);
  assert(!f.calls.some(c=>c.path==='/api/osc/session/start'));

  f=fixture();flow=create();const api=f.o.api;
  f.o.api=async(...args)=>{
    if(args[0]==='/api/osc/session/start'){
      f.calls.push({path:args[0]});f.setCurrent(ready);
      const e=new Error('timeout');e.name='AbortError';throw e;
    }return api(...args);
  };
  assert.equal(await flow.run(f.o),true);
  assert.equal(f.calls.filter(c=>c.path==='/api/osc/session/start').length,1);

  f=fixture();flow=create();let release;
  const base=f.o.api;
  f.o.api=async(...args)=>{
    if(args[0]==='/api/osc/session/start')return new Promise(resolve=>{release=()=>{f.setCurrent(ready);resolve({state:{session:ready}})}});
    return base(...args);
  };
  const pending=flow.run(f.o);
  assert.equal(await flow.run(f.o),false);
  flow.cancel();release();await assert.rejects(pending,/取消/);
  assert(!f.calls.some(c=>c.path==='/api/pi05/start'));
  assert(f.calls.some(c=>c.path==='/api/osc/session/stop'));

  f=fixture();flow=create();f.setCurrent(ready);const existingApi=f.o.api;
  f.o.api=async(...args)=>{
    if(args[0]==='/api/pi05/start'){flow.cancel();f.calls.push({path:args[0]});return {state:'RUNNING'};}
    return existingApi(...args);
  };
  await assert.rejects(flow.run(f.o),/取消/);
  assert(f.calls.some(c=>c.path==='/api/pi05/stop'));
  assert(!f.calls.some(c=>c.path==='/api/osc/session/stop'));

  f=fixture();flow=create();const foreignApi=f.o.api;
  f.o.api=async(...args)=>{
    if(args[0]==='/api/osc/session/start'){const e=new Error('timeout');e.name='AbortError';f.setCurrent({...ready,client_id:'other'});throw e;}
    return foreignApi(...args);
  };
  await assert.rejects(flow.run(f.o),/其他客户端/);
  assert(!f.calls.some(c=>c.path==='/api/pi05/start'||c.path==='/api/osc/session/stop'));

  f=fixture();flow=create();const failureApi=f.o.api;
  f.o.api=async(...args)=>{
    if(args[0]==='/api/pi05/start'){f.calls.push({path:args[0]});return {state:'ERROR',last_error:'bad action'};}
    return failureApi(...args);
  };
  await assert.rejects(flow.run(f.o),/bad action/);
  assert(f.calls.some(c=>c.path==='/api/pi05/stop'));
  assert(f.calls.some(c=>c.path==='/api/osc/session/stop'));
  assert.equal(flow.busy,false);
  console.log('7 startup scenarios passed; no real APIs or hardware used.');
})().catch(error=>{console.error(error);process.exitCode=1});
