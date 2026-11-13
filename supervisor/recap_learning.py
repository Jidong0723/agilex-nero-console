"""Additive learning workbench bridge; no hardware commands or state changes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CONSOLE_ROOT=Path(__file__).resolve().parents[1]
SOURCE_ROOT=CONSOLE_ROOT.parent/'recap_dataset_v2'
STATE_ROOT=CONSOLE_ROOT.parent/'recap_learning_state_20261112'
BACKEND='http://127.0.0.1:8791'
_lock=threading.RLock()


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return default


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def request(path,body=None,timeout=8):
    payload=json.dumps(body).encode() if body is not None else None
    req=Request(BACKEND+path,payload,headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=timeout) as response:return json.loads(response.read())
    except HTTPError as error:
        detail=read_error(error)
        raise RuntimeError(detail) from error
    except (URLError,TimeoutError) as error:
        raise RuntimeError('学习后端未启动或暂不可达（8791）；原采集功能不受影响') from error


def read_error(error):
    try:return json.loads(error.read()).get('error',str(error))
    except (ValueError,AttributeError):return str(error)


def catalog(include_removed=False):
    with _lock:return _catalog(include_removed)


def _catalog(include_removed=False):
    reviews=read(STATE_ROOT/'reviews.json',{})
    # A submitted job is not evidence that a parent trained a model. The
    # backend reports actual learning evidence plus any currently active job.
    # On backend failure retain existing locks rather than risk leakage.
    try:
        usage={row['path']:row for row in request('/api/cycle/usage')['episodes']}
    except RuntimeError:
        usage=None
    found=[]
    if not SOURCE_ROOT.exists():return found
    candidates=list(SOURCE_ROOT.iterdir())
    for category in ('dataset','failed'):
        parent=SOURCE_ROOT/category
        if parent.is_dir():candidates.extend(parent.iterdir())
    for folder in candidates:
        if not folder.is_dir() or folder.name in ('dataset','failed'):continue
        meta=read(folder/'episode.json')
        if not isinstance(meta,dict):continue
        # Only explicit completed task outcomes belong in the training list.
        if not meta.get('finalized') or meta.get('outcome') not in ('success','failure'):continue
        relative=folder.relative_to(SOURCE_ROOT).as_posix()
        identifier=hashlib.sha256(relative.encode()).hexdigest()[:24]
        quality=(meta.get('raw_result') or {}).get('quality') or {}
        reasons=[]
        if not meta.get('finalized'):reasons.append('采集未结束')
        if meta.get('outcome') not in ('success','failure'):reasons.append('故障或中止，不作为策略任务失败')
        if meta.get('purpose')=='test':reasons.append('独立测试数据不参与训练')
        if quality.get('raw_valid') is not True:reasons.append('原始数据有效性未通过')
        if meta.get('errors'):reasons.append('含硬件、记录或导出异常')
        if not (folder/'transitions.jsonl').is_file():reasons.append('缺少控制来源时间线')
        raw=folder/'raw'
        if not raw.is_dir() or not any(raw.rglob('robot_states.jsonl')):reasons.append('缺少机械臂原始数据')
        review=reviews.get(identifier,{})
        if usage is not None:
            seen=usage.get(windows_to_wsl(folder))
            if seen and review.get('split',seen['split'])!=seen['split']:
                raise ValueError('已参与训练的父组划分与历史不一致：'+folder.name)
            changed=dict(review,used=bool(seen))
            if changed!=review:
                reviews[identifier]=changed;review=changed
        if review.get('removed') and not include_removed:continue
        found.append(dict(id=identifier,name=folder.name,path=str(folder),task=meta.get('task'),
            outcome=meta.get('outcome'),purpose=meta.get('purpose'),scene=meta.get('scene'),
            interventions=meta.get('interventions',0),raw_valid=quality.get('raw_valid'),eligible=not reasons,
            reasons=reasons,approved=review.get('approved',False),split=review.get('split','train'),
            removed=bool(review.get('removed')),used=bool(review.get('used')),
            started=meta.get('started_perf_ns',0)))
    if usage is not None:
        with _lock:save(STATE_ROOT/'reviews.json',reviews)
    return sorted(found,key=lambda row:row['started'],reverse=True)


def idle(runtime):
    collection=getattr(runtime,'recap_extension',None)
    if collection and collection.state()['recording']:raise RuntimeError('请先结束当前采集轮次')
    adapters=runtime.require_adapters()
    pi=adapters.pi05.snapshot();pico=adapters.pico.snapshot()
    if pi.get('execution_enabled') or pi.get('priming'):raise RuntimeError('请先停止模型运动输出')
    if pico.get('control_enabled') and pico.get('connected'):raise RuntimeError('请先停止 PICO 控制输出')
    session=adapters.osc.fast_state().get('session') or {}
    if session.get('state') not in ('IDLE',None):raise RuntimeError('请先通过原有按钮结束控制会话；训练不会自动停止机械臂')


def windows_to_wsl(path):
    path=Path(path).resolve()
    if not path.is_relative_to(SOURCE_ROOT.resolve()):raise ValueError('Episode outside collection root')
    text=path.as_posix()
    return '/mnt/'+text[0].lower()+text[2:]


def get(handler,path):
    if path.endswith('/catalog'):return handler._json_ok(dict(episodes=catalog(),existing_demonstrations=dict(parents=100,tasks=288,train_parents=80,validation_parents=20)))
    route=path.removeprefix('/api/recap2/learning')
    if route in ('/status','/candidates','/log'):
        return handler._json_ok(request('/api/cycle'+route))
    if route=='/export':
        with urlopen(BACKEND+'/api/cycle/export',timeout=30) as response:
            payload=response.read()
        handler.send_response(200);handler.send_header('Content-Type','application/zip')
        handler.send_header('Content-Disposition','attachment; filename="recap_iteration_report.zip"')
        handler.send_header('Content-Length',str(len(payload)));handler.end_headers();handler.wfile.write(payload);return
    raise ValueError('Unknown learning endpoint')


def post(handler,path,body):
    route=path.removeprefix('/api/recap2/learning')
    if route in ('/remove','/restore'):
        with _lock:
            identifier=body.get('id')
            available={row['id']:row for row in catalog(include_removed=True)}
            if identifier not in available:raise ValueError('该已结束记录不存在，请刷新列表')
            row=available[identifier];reviews=read(STATE_ROOT/'reviews.json',{})
            review=dict(reviews.get(identifier,dict(split=row['split'],used=False)))
            review.update(approved=False,removed=route=='/remove')
            reviews[identifier]=review;save(STATE_ROOT/'reviews.json',reviews)
        return handler._json_ok(dict(ok=True,episodes=catalog(),raw_files_deleted=False))
    if route=='/review':
        with _lock:
            available={row['id']:row for row in catalog()};previous=read(STATE_ROOT/'reviews.json',{})
            for item in body.get('episodes',[]):
                if item.get('id') not in available:raise ValueError('Unknown episode')
                if item.get('approved') and not available[item['id']]['eligible']:raise ValueError('该记录不可进入训练：'+available[item['id']]['name'])
                if item.get('split') not in ('train','validation'):raise ValueError('Invalid split')
                old=previous.get(item['id'])
                if old and old.get('used') and old['split']!=item['split']:raise ValueError('已用于训练的父组不能跨集合移动')
                previous[item['id']]=dict(approved=bool(item.get('approved')),split=item['split'],used=bool(old and old.get('used')))
            save(STATE_ROOT/'reviews.json',previous)
        return handler._json_ok(dict(ok=True,episodes=catalog()))
    if route=='/start':
        idle(handler.runtime)
        selected=[row for row in catalog() if row['approved'] and row['eligible']]
        payload={key:body[key] for key in ('steps','batch','stride','value_steps','lookahead','include_existing','policy_parent','policy_initial','policy_source','positive_fraction') if key in body}
        payload['selected_episodes']=[dict(path=windows_to_wsl(row['path']),split=row['split']) for row in selected]
        result=request('/api/cycle/start',payload)
        with _lock:
            reviews=read(STATE_ROOT/'reviews.json',{})
            for row in selected:reviews[row['id']]['used']=True
            save(STATE_ROOT/'reviews.json',reviews)
        return handler._json_ok(result)
    if route in ('/stop','/candidate/start','/candidate/stop'):
        if route.startswith('/candidate/'):idle(handler.runtime)
        return handler._json_ok(request('/api/cycle'+route,body))
    raise ValueError('Unknown learning endpoint')


def assert_collection_allowed():
    try:active=request('/api/cycle/status',timeout=.15).get('active')
    except RuntimeError:return  # learning service availability never disables original collection
    if active:raise RuntimeError('学习迭代正在运行，请待保存停止后开始新采集')
