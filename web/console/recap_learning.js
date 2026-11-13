/* Additive workbench only: no changes to the existing collection/model/PICO UI. */
(function(root){'use strict';
const $=id=>document.getElementById(id);let options,busy=false,polling=false,status={},items=[],candidates=[],baseline=null,lastRemoved=null,lastCollectionKey=null;
const prefix='/api/recap2/learning';
const names={idle:'尚未开始',queued:'已排队',prepare:'审核与奖励',value_train:'价值模型更新',score:'N步优势',train:'策略更新',save:'保存',verify:'重新加载验证',completed:'完成',stopped:'已停止',error:'出错'};
async function api(route,method='GET',body){return options.api(prefix+route,method,body,15000)}
function message(text){$('rl-message').textContent=text}
async function act(fn){if(busy)return;busy=true;controls();try{await fn()}catch(e){message(e.message)}finally{busy=false;controls()}}
function controls(){if(!$('rl-start'))return;const recording=options.collectionState().recording,locked=busy||status.active||recording;
for(const id of ['rl-start','rl-review','rl-steps','rl-batch','rl-stride','rl-value-steps','rl-lookahead','rl-positive-percent','rl-existing','rl-initial','rl-model-path','rl-parent','rl-small-preset'])$(id).disabled=Boolean(locked);
$('rl-undo-remove').disabled=Boolean(locked||!lastRemoved);$('rl-rows').querySelectorAll('[data-remove]').forEach(button=>button.disabled=Boolean(locked));
$('rl-stop').disabled=busy||!status.active;$('rl-discard').disabled=busy||!status.active;$('rl-candidate-start').disabled=Boolean(locked||!candidates.length);
$('rl-candidate-stop').disabled=busy||recording||!status.candidate_running;
$('rl-use').disabled=Boolean(locked||!status.candidate_ready);$('rl-restore').disabled=Boolean(locked||!baseline);
$('rl-export').disabled=busy||status.active||!status.run_id;
}
function table(){const tbody=$('rl-rows');tbody.replaceChildren();
for(const row of items){const tr=document.createElement('tr');tr.dataset.id=row.id;
const select=document.createElement('input');select.type='checkbox';select.checked=row.approved;select.disabled=!row.eligible;select.dataset.role='approved';
const split=document.createElement('select');split.innerHTML='<option value="train">训练</option><option value="validation">验证</option>';split.value=row.split;split.dataset.role='split';split.disabled=!row.eligible||row.used;
const remove=document.createElement('button');remove.className='button quiet';remove.type='button';remove.textContent='移出列表';remove.dataset.remove=row.id;
remove.onclick=()=>act(async()=>{await review();const data=await api('/remove','POST',{id:row.id});lastRemoved=row.id;try{sessionStorage.setItem('nero.recap.learning.lastRemoved',lastRemoved)}catch(_){}items=data.episodes;table();message('已移出列表并取消纳入训练；原始文件保留，可撤销。')});
const outcome={success:'成功',failure:'任务失败',fault:'故障',aborted:'中止'};
for(const value of [select,row.name,row.task||'—',outcome[row.outcome]||row.outcome||'未结束',row.eligible?'可纳入':row.reasons.join('；'),split,remove]){
const td=document.createElement('td');if(value instanceof Node)td.appendChild(value);else td.textContent=value;tr.appendChild(td)}tbody.appendChild(tr)}
if(!items.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.textContent='暂无已结束记录';tr.appendChild(td);tbody.appendChild(tr)}
}
async function catalog(){const data=await api('/catalog');items=data.episodes;table()}
async function review(){const episodes=[...$('rl-rows').querySelectorAll('tr[data-id]')].map(tr=>({id:tr.dataset.id,approved:tr.querySelector('[data-role=approved]').checked,split:tr.querySelector('[data-role=split]').value}));
const data=await api('/review','POST',{episodes});items=data.episodes;table();message('本轮数据选择已保存；取消勾选的记录不进入本轮训练。')}
function plot(){const canvas=$('rl-loss'),d=devicePixelRatio||1;canvas.width=canvas.clientWidth*d;canvas.height=210*d;const c=canvas.getContext('2d');c.scale(d,d);const w=canvas.clientWidth,h=210,p=38;
const data=status.metrics||[],values=data.flatMap(r=>[r.loss,r.validation_loss]).filter(Number.isFinite),top=Math.max(...values,0)*1.1||1,last=Math.max(...data.map(r=>r.step),1);
c.font='11px sans-serif';for(let i=0;i<5;i++){const y=p+(h-2*p)*i/4;c.strokeStyle='#284144';c.beginPath();c.moveTo(p,y);c.lineTo(w-12,y);c.stroke();c.fillStyle='#91abaa';c.fillText((top*(1-i/4)).toFixed(3),0,y+3)}
for(const [key,color] of [['loss','#49d7af'],['validation_loss','#e8ba73']]){c.strokeStyle=color;c.lineWidth=1.7;c.beginPath();let start=true;for(const row of data){if(!Number.isFinite(row[key]))continue;const x=p+(w-p-15)*row.step/last,y=h-p-(h-2*p)*row[key]/top;start?c.moveTo(x,y):c.lineTo(x,y);start=false}c.stroke()}
c.fillStyle='#91abaa';c.fillText('0',p,h-10);c.fillText(String(last)+' 次更新',w-90,h-10);if(!data.length)c.fillText('策略更新后显示真实损失',p,h/2)}
function fillSelect(id,choices){const el=$(id),signature=JSON.stringify(choices);if(el.dataset.choices===signature)return;const selected=el.value;el.replaceChildren(...choices.map(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;return option}));if(choices.some(([value])=>value===selected))el.value=selected;el.dataset.choices=signature}
async function refresh(){if(polling||(!options.selected()&&!status.active))return;polling=true;
try{status=await api('/status');$('rl-state').textContent=(names[status.phase]||status.phase)+(['error','stopped'].includes(status.phase)?' · '+status.message:'');$('rl-state').title=status.message||'';
const collection=options.collectionState(),collectionKey=collection.recording?'recording':`${collection.data_path||''}:${collection.labels?.outcome||''}`;
if(lastCollectionKey!==null&&collectionKey!==lastCollectionKey&&!collection.recording)await catalog();lastCollectionKey=collectionKey;
const metrics=status.metrics||[],vm=status.value_metrics||[],last=metrics.at(-1),vlast=vm.at(-1);
$('rl-metrics').textContent=`策略 ${last?.step||0}步 · 价值 ${vlast?.step||0}步 · loss ${last?.loss?.toFixed(5)||'—'} · val ${metrics.filter(m=>Number.isFinite(m.validation_loss)).at(-1)?.validation_loss.toFixed(5)||'—'}`;
$('rl-path').textContent=status.directory||'';const report=status.policy_report;
$('rl-result').textContent=report?`验证损失 ${report.baseline_validation_loss.toFixed(5)} → ${report.candidate_validation_loss.toFixed(5)}`:'';
const ar=status.advantage_report,actual=ar?.train_autonomous?.actual_positive_fraction;
$('rl-advantage').textContent=ar?`自主正优势：目标 ${(ar.positive_fraction_target*100).toFixed(1)}%${Number.isFinite(actual)?` · 实际 ${(actual*100).toFixed(1)}%`:''}`:'';
$('rl-candidate-state').textContent=status.candidate_running?(status.candidate_ready?'服务就绪':'加载中'):'服务未启动';$('rl-candidate-state').title=status.candidate_running?`127.0.0.1:8010 · ${status.candidate_run}`:'';
const data=await api('/candidates');candidates=data.candidates;
fillSelect('rl-candidates',candidates.length?candidates.map(row=>[row.id,`${row.id} · ${row.report.step}步`]):[['','暂无已验证候选']]);$('rl-candidates').disabled=!candidates.length;
fillSelect('rl-initial',[['','11649（初始模型）'],['custom','自定义地址'],...candidates.map(row=>[row.id,row.id])]);
fillSelect('rl-parent',[['','不关联'],...candidates.map(row=>[row.id,'关联上一轮：'+row.id])]);
if(!$('rl-model-path').value&&!$('rl-model-path').dataset.edited&&status.default_policy_source)$('rl-model-path').value=status.default_policy_source;
$('rl-source').textContent=status.active&&status.training_policy_source?'本轮起点：'+status.training_policy_source:'';
plot();if($('rl-details').open){const log=await api('/log');$('rl-log').textContent=log.text||'尚无运行日志'}
}catch(e){$('rl-state').textContent=e.message}finally{polling=false;controls()}}
async function switchEndpoint(host,port,restore){const current=await options.api('/api/pi05/state','GET',undefined,5000);
if(options.collectionState().recording||current.execution_enabled||current.priming)throw Error('请先结束采集和模型输出');
if(!restore&&!baseline){const name=current.config.inference_profiles.active,profile=current.config.inference_profiles[name];if(!profile||!profile.host||!profile.port)throw Error('当前连接地址不完整，无法保存回切信息');baseline={profile:name,endpoint:{host:profile.host,port:profile.port}};try{sessionStorage.setItem('nero.recap.learning.baseline',JSON.stringify(baseline))}catch(_){}}
await options.api('/api/pi05/config','POST',{inference_profile:restore?baseline.profile:current.config.inference_profiles.active,policy_endpoint:{host,port}},15000);
message(restore?'已恢复原模型连接设置；未启动运动。':'已改为候选连接；请在原模型栏确认连接和H20协议，然后手动开始本轮。');controls()}
function init(o){if($('recap-learning'))return;options=o;
try{lastRemoved=sessionStorage.getItem('nero.recap.learning.lastRemoved')}catch(_){}
try{baseline=JSON.parse(sessionStorage.getItem('nero.recap.learning.baseline'));if(!baseline?.endpoint?.host||!Number.isInteger(baseline?.endpoint?.port))baseline=null}catch(_){}
const section=document.createElement('div');section.id='recap-learning';section.innerHTML=`
<style>
#recap-learning{border-top:1px solid var(--border,#294245);margin-top:18px;padding-top:16px}
#recap-learning h3{margin:0 0 12px}#recap-learning label{display:block}
#recap-learning .rl-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
#recap-learning .rl-grid input,#recap-learning .rl-grid select{width:100%;box-sizing:border-box}
#recap-learning details{border:1px solid var(--border,#294245);border-radius:8px;padding:10px 12px;margin:10px 0}
#recap-learning summary{cursor:pointer;font-weight:600}#recap-learning details[open]>summary{margin-bottom:10px}
#recap-learning .rl-scroll{overflow:auto;max-height:230px;margin:10px 0}
#recap-learning table{width:100%;border-collapse:collapse;font-size:12px}
#recap-learning td,#recap-learning th{border-bottom:1px solid #294245;padding:7px;text-align:left;max-width:260px;overflow-wrap:anywhere}
#recap-learning input[type=checkbox]{width:auto;display:inline}
#recap-learning canvas{width:100%;height:210px;display:block}
#recap-learning pre{white-space:pre-wrap;word-break:break-all;max-height:220px;overflow:auto;font-size:11px}
#recap-learning .rl-note{color:#95aeac;line-height:1.6;font-size:12px}
#recap-learning .rl-path{overflow-wrap:anywhere;user-select:text;font-size:11px}
#recap-learning .session-actions{gap:8px;margin:10px 0}#recap-learning p{margin:8px 0}
#recap-learning p:empty{display:none}
</style>
<h3>学习训练</h3>
<details id="rl-data"><summary>数据选择</summary>
<div class="session-actions"><button class="button quiet" id="rl-refresh">刷新</button><button class="button" id="rl-review">保存选择</button><button class="button quiet" id="rl-undo-remove">撤销移出</button></div>
<div class="rl-scroll"><table><thead><tr><th>纳入</th><th>编号</th><th>任务</th><th>结果</th><th>状态</th><th>用途</th><th>操作</th></tr></thead><tbody id="rl-rows"></tbody></table></div>
<label title="288段示教，沿用原80/20父组划分"><input id="rl-existing" type="checkbox"> 混合已有示教</label>
</details>
<label>训练起点<select id="rl-initial"><option value="">11649（初始模型）</option><option value="custom">自定义地址</option></select></label>
<label>模型地址<input id="rl-model-path" type="text" placeholder="5090机上的模型目录（包含params和assets）"></label><p id="rl-source" class="rl-path"></p>
<div class="rl-grid">
<label>策略更新<input id="rl-steps" type="number" min="1" max="1000000" value="200"></label>
<label>价值更新<input id="rl-value-steps" type="number" min="1" max="1000000" value="100"></label>
<label>有效batch<select id="rl-batch"><option>1</option><option>4</option><option>8</option><option selected>16</option><option>32</option><option>64</option></select></label>
<label title="仅控制自主样本；人工样本保持正条件">正优势（%）<input id="rl-positive-percent" type="number" min="0" max="100" step="0.1" value="40"></label>
</div>
<details id="rl-advanced"><summary>更多参数</summary>
<label>关联上一轮<select id="rl-parent"><option value="">首轮（11649）</option></select></label>
<div class="rl-grid"><label>锚点间隔<input id="rl-stride" type="number" min="1" max="200" value="20"></label>
<label>优势前瞻<input id="rl-lookahead" type="number" min="1" max="200" value="50"></label></div>
<button class="button quiet" id="rl-small-preset">试跑预设</button>
</details>
<div class="session-actions"><button class="button primary" id="rl-start">开始训练</button><button class="button quiet" id="rl-stop">保存并停止</button><button class="button quiet" id="rl-discard">不保存并停止</button><button class="button quiet" id="rl-export">导出报告</button></div>
<p id="rl-state" role="status">连接中…</p><p id="rl-metrics" class="rl-note"></p>
<details id="rl-details"><summary>曲线与日志</summary>
<p id="rl-result" class="rl-note"></p><p id="rl-advantage" class="rl-note"></p>
<canvas id="rl-loss"></canvas><p class="rl-note">绿色：训练 · 橙色：验证</p>
<p id="rl-path" class="rl-path"></p><pre id="rl-log"></pre>
</details>
<details id="rl-candidate-panel"><summary>候选模型</summary>
<label>推理候选模型<select id="rl-candidates"></select></label>
<div class="session-actions"><button class="button" id="rl-candidate-start">启动服务</button><button class="button quiet" id="rl-candidate-stop">关闭服务</button><button class="button" id="rl-use">使用此模型</button><button class="button quiet" id="rl-restore">恢复原模型</button></div>
<p id="rl-candidate-state" class="rl-note"></p>
</details>
<details id="rl-help"><summary>使用说明</summary>
<p class="rl-note">先保存数据选择，结束采集和控制会话，并关闭原推理服务释放显存。训练不会自动启动机械臂。</p>
<p class="rl-note">只显示成功或失败结束的记录。移出不删除原文件；本轮仅使用勾选的数据。每轮从所选训练起点加载权重。</p>
<p class="rl-note">正优势比例仅针对自主样本；人工示教与纠正保持正条件。同分时实际比例可有差异。20Hz下，前瞻50步为2.5秒。</p>
<p class="rl-note">候选先做影子与实机对照；验证损失不是成功率。切换模型不会自动运动。</p>
</details>
<p id="rl-message" role="status"></p>`;
$('recap-panel').appendChild(section);
$('rl-initial').onchange=()=>{const id=$('rl-initial').value;if(id==='custom'){$('rl-model-path').focus();return}const source=id?candidates.find(row=>row.id===id)?.checkpoint:status.default_policy_source;if(source){$('rl-model-path').value=source;delete $('rl-model-path').dataset.edited}};
$('rl-model-path').oninput=()=>{$('rl-model-path').dataset.edited='true';$('rl-initial').value='custom'};
$('rl-refresh').onclick=()=>act(catalog);$('rl-review').onclick=()=>act(review);
$('rl-undo-remove').onclick=()=>act(async()=>{const data=await api('/restore','POST',{id:lastRemoved});lastRemoved=null;try{sessionStorage.removeItem('nero.recap.learning.lastRemoved')}catch(_){}items=data.episodes;table();message('记录已恢复到列表，需重新勾选才纳入训练。')});
$('rl-small-preset').onclick=()=>{if(status.active||options.collectionState().recording)return;for(const [id,value] of [['rl-steps',200],['rl-value-steps',100],['rl-batch',16],['rl-stride',20],['rl-lookahead',50],['rl-positive-percent',40]])$(id).value=value;$('rl-existing').checked=false;message('已设置试跑参数：价值100次、策略200次、有效batch16、正优势目标40%；未启动训练。')};
$('rl-start').onclick=()=>act(async()=>{const raw=$('rl-positive-percent').value,percent=Number(raw);if(raw.trim()===''||!Number.isFinite(percent)||percent<0||percent>100)throw Error('正优势占比请输入0至100');await review();await api('/start','POST',{include_existing:$('rl-existing').checked,steps:Number($('rl-steps').value),value_steps:Number($('rl-value-steps').value),batch:Number($('rl-batch').value),stride:Number($('rl-stride').value),lookahead:Number($('rl-lookahead').value),positive_fraction:percent/100,policy_initial:$('rl-initial').value==='custom'?'':$('rl-initial').value,policy_source:$('rl-model-path').value.trim(),policy_parent:$('rl-parent').value});message('已开始本轮学习；未启动机械臂。');await refresh()});
$('rl-discard').onclick=()=>act(async()=>{message((await api('/stop','POST',{discard:true})).message);await refresh()});
$('rl-stop').onclick=()=>act(async()=>message((await api('/stop','POST',{})).message));
$('rl-export').onclick=()=>act(async()=>{const r=await fetch(prefix+'/export');if(!r.ok)throw Error(await r.text());const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download='recap_iteration_report.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000)});
$('rl-candidate-start').onclick=()=>act(async()=>{message((await api('/candidate/start','POST',{id:$('rl-candidates').value})).message);await refresh()});
$('rl-candidate-stop').onclick=()=>act(async()=>{message((await api('/candidate/stop','POST',{})).message);await refresh()});
$('rl-use').onclick=()=>act(()=>switchEndpoint('127.0.0.1',8010,false));
$('rl-restore').onclick=()=>act(()=>switchEndpoint(baseline.endpoint.host,baseline.endpoint.port,true));
$('input-adapter').addEventListener('change',()=>{if(options.selected()){void catalog().catch(e=>message(e.message));void refresh()}});
$('rl-details').addEventListener('toggle',refresh);window.addEventListener('resize',()=>{if(options.selected())plot()});
setInterval(refresh,3000);void catalog().catch(e=>message(e.message));void refresh();controls();
}
root.NeroRecapLearning={init};
})(window);
