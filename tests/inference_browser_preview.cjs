/* Real console + fake HTTP backend; no robot or external model connections. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const os = require('node:os');
const {chromium} = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '../web/console');
  const canonical = JSON.parse(fs.readFileSync(path.resolve(__dirname,'../config/pi05.json'),'utf8'));
  canonical.inference_profiles.active='local5090';
  canonical.execution.mode='normal';
  let pi, camera;
  const requests = [];
  function publish() {
    const config = structuredClone(canonical), p = config.inference_profiles[config.inference_profiles.active];
    config.model = {...config.model,host:p.host,port:p.port,checkpoint_label:p.checkpoint_label};
    Object.assign(config.execution,{expected_chunk_steps:p.expected_chunk_steps,replan_steps:p.replan_steps,max_chunk_steps:p.expected_chunk_steps});
    camera = {ready:true,frame_version:1,config:config.cameras,sources:{},devices:[]};
    pi = {config, camera, camera_ready:true,model_state:'CONNECTED',state:'RUNNING',
      execution_enabled:false,protocol_compatibility:{status:'compatible',message:'7D 输入／动作响应验证通过'},
      action_chunk:Array.from({length:p.expected_chunk_steps},()=>[.001,0,0,0,0,0,.5]),
      preview:{state:'READY',updated_at:Date.now()/1000,hz:4,sequence:7},
      chunk_source:'preview',chunk_sequence:7,action_chunk_length:p.expected_chunk_steps,
      chunk_updated_at:Date.now()/1000,chunk_stale:false,inference_ms:190,
      last_response_steps:p.expected_chunk_steps,policy_metadata:{deployment_label:'模拟模型 · state7/action7'},
      connections:{policy:{state:'ok',label:'模型 WebSocket',endpoint:`ws://${p.host}:${p.port}`,message:'已连接'},
        service_entry:{state:'ok',subtitle:config.inference_profiles.active==='autodl'?'AutoDL 本机转发入口':'5090 本地直连',endpoint:`${p.host}:${p.port}`,message:'已有服务监听',checked_at:Date.now()/1000},
        ssh_forward:{state:'ok',label:'模型连接入口',endpoint:`${p.host}:${p.port}`,message:'模拟连接'}},updated_at:Date.now()/1000};
  }
  publish();
  const osc = {state_sequence:1,output_mode:'cpv',session:null,authority:{hardware_mode:'HOLD',servo_mode:'SUSPENDED'},
    transport:{connected:false},execution:{},diagnostics:{},workspace:{},command:{},solver:{}};
  const server = http.createServer(async (req,res) => {
    if (req.url.startsWith('/api/')) {
      let raw=''; for await (const chunk of req) raw+=chunk;
      const body=raw?JSON.parse(raw):null;requests.push({path:req.url,method:req.method,body});
      let data={};
      if(req.url==='/api/pi05/config') {
        if(body.inference_profile) canonical.inference_profiles.active=body.inference_profile;
        const p=canonical.inference_profiles[canonical.inference_profiles.active];
        Object.assign(p,body.policy_endpoint||{});
        for(const key of ['expected_chunk_steps','replan_steps']) if(body.execution?.[key]) p[key]=body.execution[key];
        if(body.execution?.arrival_wait_s!==undefined) canonical.execution.arrival_wait_s=body.execution.arrival_wait_s;
        for(const key of ['mode','rtc_min_replan_steps']) if(body.execution?.[key]!==undefined) canonical.execution[key]=body.execution[key];
        publish(); data=pi;
      } else if(req.url.includes('/pi05/state')) data=pi;
      else if(req.url==='/api/osc/state'||req.url==='/api/osc/fast-state') data=osc;
      else if(req.url==='/api/health') data={http_ready:true,control_backend_ready:true};
      else if(req.url==='/api/status') data={connected:false,control:{mode:'HOLD'}};
      else if(req.url==='/api/cameras/list') data={cameras:[{index:0,name:'模拟外部相机'},{index:2,name:'模拟腕部相机'}]};
      else if(req.url.startsWith('/api/cameras/')) data=camera;
      else if(req.url.includes('/dataset/')) data={state:'IDLE'};
      res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({ok:true,data}));return;
    }
    const target=path.resolve(root, '.'+new URL(req.url,'http://local').pathname.replace(/\/$/,'/index.html'));
    if(!target.startsWith(root+path.sep)){res.writeHead(404);res.end();return;}
    try {
      const mime=target.endsWith('.js')?'text/javascript':target.endsWith('.css')?'text/css':'text/html';
      res.writeHead(200,{'Content-Type':mime});res.end(fs.readFileSync(target));
    }catch{res.writeHead(404);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const output=fs.mkdtempSync(path.join(os.tmpdir(),'nero-inference-preview-'));
  let browser;
  try {
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1100}}), errors=[];
    await page.route('**/*',route=>route.request().url().startsWith(base)?route.continue():route.abort());
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);await page.selectOption('#input-adapter','pi05');
    await page.addStyleTag({content:'.topbar{position:static!important}'});
    await page.waitForFunction(()=>document.querySelector('#inference-horizon').value==='10');
    assert.equal(await page.locator('#input-adapter option').allTextContents().then(x=>x.filter(v=>v.includes('模型推理')).length),1);
    await page.fill('#inference-horizon','13');await page.fill('#inference-prefix','4');
    await page.waitForTimeout(1300);
    assert.equal(await page.inputValue('#inference-horizon'),'13');assert.ok(await page.isDisabled('#pi05-start'));
    await page.click('#inference-save');await page.waitForFunction(()=>document.querySelector('#inference-live-lengths').textContent==='13 / 4');
    await page.selectOption('#inference-profile','autodl');await page.fill('#inference-port','9911');
    assert.match(await page.textContent('#inference-connection-active'),/当前生效：5090 本地/);
    assert.match(await page.textContent('#inference-help-draft'),/已保存值/);
    await page.click('#inference-save');await page.waitForFunction(()=>document.querySelector('#inference-live-profile').textContent==='AutoDL 云端');
    const regionOrder=await page.evaluate(()=>['.inference-settings','.pi05-connection','.pi05-help','.pi05-adapter-row'].map(selector=>document.querySelector('#pi05-panel '+selector).getBoundingClientRect().top));
    assert.ok(regionOrder.every((top,index)=>index===0||top>regionOrder[index-1]),'connection status must follow settings, before help and observations');
    await page.locator('.pi05-help>summary').click();
    assert.equal(await page.locator('.inference-help-card:visible').count(),2);
    assert.match(await page.textContent('#inference-help-command'),/127.0.0.1:9911:127.0.0.1:<policy-port>/);
    await page.evaluate(()=>{window.copied=[];Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>window.copied.push(text)}});});
    await page.click('[data-copy-command="inference-help-server"]');
    await page.waitForFunction(()=>document.querySelector('#inference-copy-result').textContent.includes('已复制'));
    assert.equal(await page.evaluate(()=>window.copied[0]),await page.textContent('#inference-help-server'));
    await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async()=>{throw Error('denied')}}});document.execCommand=()=>false;});
    await page.click('[data-copy-command="inference-help-command"]');
    await page.waitForFunction(()=>document.querySelector('#inference-copy-result').textContent.includes('复制失败'));
    pi.model_state='DISCONNECTED';pi.websocket_error='HTTP 200: not a policy server';
    pi.connections.policy={...pi.connections.policy,state:'bad',message:pi.websocket_error};
    await page.waitForFunction(()=>document.querySelector('#pi05-connection-policy').classList.contains('bad'));
    assert.ok(await page.locator('#pi05-connection-ssh').evaluate(node=>node.classList.contains('ok')));
    assert.ok(await page.isDisabled('#pi05-start'));
    pi.model_state='CONNECTED';pi.connections.policy.state='ok';pi.connections.policy.message='已连接';pi.websocket_error=null;
    pi.protocol_compatibility={status:'incompatible',message:'state_dim=8, expected 7'};
    await page.waitForFunction(()=>document.querySelector('#inference-protocol-detail').textContent.includes('state_dim=8'));
    assert.ok(await page.locator('#pi05-connection-policy').evaluate(node=>node.classList.contains('ok')));
    assert.ok(await page.isDisabled('#pi05-start'));
    pi.protocol_compatibility={status:'compatible',message:'7D／H 验证通过'};
    await page.waitForFunction(()=>document.querySelector('#inference-protocol').classList.contains('ok'));
    await page.selectOption('#inference-mode','rtc');
    assert.equal(await page.inputValue('#inference-horizon'),'20');
    assert.ok(await page.isDisabled('#inference-prefix'));
    assert.equal(await page.locator('#pi05-action-scale').count(),0);
    assert.match(await page.locator('label').allTextContents().then(x=>x.join(' ')),/段末到位等待上限（ms）/);
    await page.click('#inference-save');
    await page.waitForFunction(()=>document.querySelector('#inference-mode').value==='rtc'&&!document.querySelector('#inference-save').disabled);
    pi.rtc={state:'RUNNING',queue_remaining_ms:650,end_to_end_ms:150,replan_interval_steps:7};
    await page.waitForFunction(()=>document.querySelector('#inference-rtc-state').textContent.includes('650 ms'));
    await page.locator('#pi05-panel').screenshot({path:path.join(output,'autodl-desktop.png')});
    await page.setViewportSize({width:390,height:844});
    await page.locator('#pi05-panel').screenshot({path:path.join(output,'autodl-mobile.png')});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'expanded AutoDL help must not overflow');
    const cards=await page.locator('.inference-help-card').evaluateAll(nodes=>nodes.map(node=>({x:node.getBoundingClientRect().x,y:node.getBoundingClientRect().y})));
    assert.equal(cards[0].x,cards[1].x);assert.ok(cards[1].y>cards[0].y);
    await page.setViewportSize({width:1440,height:1100});
    await page.selectOption('#inference-mode','normal');
    await page.click('#inference-save');
    await page.selectOption('#inference-profile','local5090');
    assert.equal(await page.inputValue('#inference-horizon'),'13');assert.equal(await page.inputValue('#inference-prefix'),'4');
    await page.click('#inference-save');
    await page.addStyleTag({content:'.topbar{position:static!important}'});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'desktop page must not overflow horizontally');
    await page.locator('#pi05-panel').screenshot({path:path.join(output,'desktop.png')});
    assert.equal(await page.locator('#pi05-start').textContent(),'开始推理控制');
    assert.equal(await page.locator('#pi05-stop').textContent(),'停止推理控制');
    assert.match(await page.locator('#inference-preview-state').textContent(),/自动更新中/);
    assert.match(await page.locator('#inference-chunk-source').textContent(),/动作预测预览/);
    assert.equal(await page.locator('.inference-details').getAttribute('open'),null);
    assert.equal(await page.locator('#inference-log-state').isVisible(),false);
    assert.equal(await page.locator('#inference-export').isVisible(),false);
    pi.training_notice='RTC训练契约未核验';
    pi.last_error='PermissionError: SERVO write authority is unavailable';
    osc.diagnostics={mailboxes:{cpv:{last_result:{error:'CPV reference moves outside joint limits'}}}};
    await page.waitForFunction(()=>document.querySelector('#inference-control-message').textContent.includes('关节限位保护'));
    assert.ok(await page.locator('#inference-control-message').isVisible());
    assert.equal(await page.locator('#inference-training-detail').isVisible(),false);
    assert.equal(await page.locator('#inference-error-detail').isVisible(),false);
    await page.locator('.inference-details summary').click();
    assert.ok(await page.locator('#inference-error-detail').isVisible());
    await page.locator('.inference-details summary').click();
    pi.last_error=null;
    await page.reload();await page.selectOption('#input-adapter','pi05');
    await page.waitForFunction(()=>document.querySelector('#inference-horizon').value==='13');
    await page.setViewportSize({width:390,height:844});
    await page.addStyleTag({content:'.topbar{position:static!important}'});
    await page.locator('#pi05-panel').screenshot({path:path.join(output,'mobile.png')});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'mobile page must not overflow horizontally');
    assert.deepEqual(errors,[]);
    assert.ok(!requests.some(r=>r.path.includes('/osc/command')||r.path.includes('/session/start')));
    console.log(JSON.stringify({passed:true,screenshots:output,errors,configurationRequests:requests.filter(r=>r.path==='/api/pi05/config').length}));
  }finally {
    if(browser) await browser.close();await new Promise(resolve=>server.close(resolve));
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
