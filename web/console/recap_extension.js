/* RECAP orchestration only. Original model/PICO cards own their connection UI. */
(function(root){
 'use strict';
 const $=id=>document.getElementById(id);
 let selected=()=>false,latest={},busy=false,startup,options,camera,home,card,polling=false;
 let startError='',wasSelected=false,exporting=false,exportMessage='';
 function showStartError(message){
  startError=message;
  $('recap-message').textContent=message;
 }
 function layout(){
  if(!card)return;
  const active=selected();
  card.classList.toggle('hidden',!active);
  if(active){
   $('pi05-panel').classList.remove('hidden');$('pico-panel').classList.remove('hidden');
   if(camera.parentNode!==$('recap-camera-area'))$('recap-camera-area').appendChild(camera);
  }else if(camera.previousElementSibling!==home)home.insertAdjacentElement('afterend',camera);
  $('pi05-panel').querySelector('.pi05-observation').classList.toggle('hidden',active);
  $('pi05-panel').querySelector('.pi05-adapter-actions').classList.toggle('hidden',active);
  $('pi05-panel').querySelector('.pi05-adapter-row').style.gridTemplateColumns=active?'1fr':'';
  $('recap-begin').disabled=busy||latest.recording;
  $('recap-human').disabled=busy||latest.state!=='MODEL';
  $('recap-model').disabled=busy||!['HUMAN','HUMAN_WAITING'].includes(latest.state);
  for(const key of ['success','failure'])$('recap-'+key).disabled=busy||!latest.recording;
  $('recap-abort').disabled=!latest.recording&&!busy;
  $('recap-download').disabled=exporting||!latest.export_ready;
  $('recap-download').textContent=exporting?'正在打包 / 下载…':'导出数据';
  $('recap-save-path').textContent=latest.data_path?'数据保存路径：'+latest.data_path:'数据保存路径：开始采集后显示';
  $('recap-export-path').textContent=exportMessage||(latest.export_file_path?'ZIP导出位置：'+latest.export_file_path:'');
  const labels=latest.labels||{}, quality=latest.raw_quality;
  const purposes={pilot:'试运行',train:'正式采集',demo:'补充示教',test:'独立测试'};
  const outcomes={success:'成功',failure:'失败',aborted:'人工中止',fault:'故障结束'};
  const validity=latest.recording?'数据有效性待检查':quality?.raw_valid===true?'数据有效':quality?.raw_valid===false?'数据无效':'数据有效性待检查';
  $('recap-labels').textContent=latest.data_path?`本轮数据标记｜场景：${labels.scene||'未填写'}｜用途：${purposes[labels.purpose]||labels.purpose||'未填写'}｜结果：${outcomes[labels.outcome]||labels.outcome||'尚未结束'}｜人工干预：${latest.human_intervened?'是':'否'}（接管 ${latest.interventions||0} 次）${labels.assisted_success?'｜人工帮助后成功':labels.autonomous_success?'｜自主成功':''}｜${validity}`:'本轮数据标记：尚无记录';
  for(const key of ['task','scene','purpose'])$('recap-'+key).disabled=busy||latest.recording;
  if(latest.recording){$('pico-start').disabled=true;$('pico-stop').disabled=true;}
 }
 async function refresh(){
  if(polling||(!selected()&&!latest.recording))return;
  polling=true;
  try{latest=await options.api('/api/recap2/state','GET',undefined,3000);
   const names={IDLE:'未采集',STARTING:'正在开始',PRIMING:'正在准备模型控制',MODEL:'模型控制',TO_HUMAN:'正在交给PICO',HUMAN_WAITING:'等待PICO连接',HUMAN:'PICO控制',TO_MODEL:'正在交还模型',STOPPING:'正在结束',FINISHED:'已结束',ERROR:'已停止'};
   const picoStatus=latest.pico_connected?(latest.pico_control_enabled?'PICO已连接':'PICO已连接待机'):'PICO未连接';
   if(!busy)$('recap-message').textContent=startError||`${names[latest.state]||latest.state} · ${picoStatus} · 接管 ${latest.interventions||0} 次${latest.error?' · '+latest.error:''}`;
  }catch(e){$('recap-message').textContent=e.message;}finally{polling=false;layout();}
 }
 async function act(endpoint,body={}){
  if(busy)return;
  startError='';
  busy=true;layout();$('recap-message').textContent='正在处理，请等待确认…';
  try{latest=await options.api('/api/recap2/'+endpoint,'POST',{client_id:options.clientId,...body},60000);}
  catch(e){options.phase(e.message,true);$('recap-message').textContent=e.message;}
  finally{busy=false;await refresh();layout();}
 }
 async function begin(){
  if(busy||latest.recording)return;
  const task=$('recap-task').value.trim(),scene=$('recap-scene').value.trim();
  if(!task||!scene){showStartError('请填写任务指令和场景编号');return;}
  startError='';
  busy=true;layout();
  const api=async(path,method,body,timeout)=>{
   if(path==='/api/pi05/start')return options.api('/api/recap2/start',method,{...body,task,scene,purpose:$('recap-purpose').value},timeout);
   if(path==='/api/pi05/stop')return options.api('/api/recap2/finish',method,{client_id:options.clientId,outcome:'aborted',reason:body?.reason},timeout);
   return options.api(path,method,body,timeout);
  };
  try{
   const pi=await options.api('/api/pi05/state','GET',undefined,5000);
   if(pi.model_state!=='CONNECTED')throw Error('请先连接模型');
   if(!pi.camera_ready)throw Error('请先打开双相机');
   if(pi.protocol_compatibility?.status==='incompatible')throw Error(pi.protocol_compatibility.message);
   await startup.run({...options,api,mode:options.mode(),config:null,
   context:()=>`${selected()}|${options.mode()}`,acceptPi:value=>{latest=value;},
   progress:text=>{$('recap-message').textContent=text;},changed:layout});}
  catch(e){options.phase(e.message,true);showStartError(e.message);}
  finally{busy=false;await refresh();layout();}
 }
 function init(o){
  options=o;startup=root.NeroInferenceStartup.create();selected=()=>$('input-adapter').value==='recap';
  const choice=document.createElement('option');choice.value='recap';choice.textContent='RECAP 学习工作台';$('input-adapter').appendChild(choice);
  card=document.createElement('section');card.id='recap-panel';card.className='panel hidden';
  card.innerHTML=`<h2>RECAP 学习工作台</h2><div id="recap-camera-area"></div>
   <div class="dataset-fields"><label>任务指令<input id="recap-task" maxlength="500"></label><label>场景编号<input id="recap-scene" maxlength="120"></label>
   <label>用途<select id="recap-purpose"><option value="pilot">试运行</option><option value="train">正式采集</option><option value="demo">补充示教</option><option value="test">独立测试</option></select></label></div>
   <div class="session-actions"><button class="button primary" id="recap-begin">开始本轮</button><button class="button" id="recap-human">PICO接管</button><button class="button" id="recap-model">交还模型</button>
   <button class="button" id="recap-success">成功结束</button><button class="button" id="recap-failure">失败结束</button><button class="button quiet" id="recap-abort">中止</button><button class="button quiet" id="recap-download">导出数据</button></div><p id="recap-message" role="status">尚未开始</p>`;
  $('pi05-panel').insertAdjacentElement('beforebegin',card);
  camera=$('pico-panel').querySelector('.pico-camera-resource');home=document.createElement('div');home.className='hidden';camera.insertAdjacentElement('beforebegin',home);
  $('recap-begin').onclick=begin;
  $('recap-human').onclick=()=>act('takeover');$('recap-model').onclick=()=>act('return');
  for(const result of ['success','failure','abort'])$('recap-'+result).onclick=()=>{
   if(startup.busy)startup.cancel();
   if(result==='abort'&&busy){void options.api('/api/recap2/finish','POST',{client_id:options.clientId,outcome:'aborted'},60000).then(refresh);return;}
   return act('finish',{outcome:result==='abort'?'aborted':result});
  };
  const paths=document.createElement('div');
  paths.innerHTML='<p id="recap-labels" style="overflow-wrap:anywhere;user-select:text"></p><p id="recap-save-path" style="overflow-wrap:anywhere;user-select:text"></p><p id="recap-export-path" role="status" style="overflow-wrap:anywhere;user-select:text"></p>';
  card.appendChild(paths);
  $('recap-download').onclick=async()=>{
   if(exporting)return;
   exporting=true;exportMessage='正在打包并下载，请稍候…';layout();
   try{const r=await fetch('/api/recap2/export',{cache:'no-store'});if(!r.ok)throw Error(await r.text());
    const url=URL.createObjectURL(await r.blob()),link=document.createElement('a');
    link.href=url;link.download=r.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1]||'recap_episode.zip';
    document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
    exportMessage='已请求浏览器下载。ZIP保存路径：'+latest.export_file_path+'；浏览器副本请查看下载列表。';
   }catch(e){exportMessage='导出失败：'+e.message;options.phase(exportMessage,true);}
   finally{exporting=false;layout();}
  };
  $('input-adapter').addEventListener('change',async()=>{
   const leaving=wasSelected&&!selected();wasSelected=selected();
   if(leaving){
    busy=true;
    try{latest=await options.api('/api/recap2/leave','POST',{client_id:options.clientId},60000);}
    catch(e){$('input-adapter').value='recap';wasSelected=true;showStartError(e.message);}
    finally{busy=false;}
   }
   layout();void refresh();
  });
  setInterval(refresh,500);layout();
  const learning=document.createElement('script');learning.src='recap_learning.js?v=20261112';
  learning.onload=()=>root.NeroRecapLearning.init({api:options.api,clientId:options.clientId,selected,collectionState:()=>latest});
  document.head.appendChild(learning);
 }
 root.NeroRecap={init,layout};
})(window);
