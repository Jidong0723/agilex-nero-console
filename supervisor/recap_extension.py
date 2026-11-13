"""Opt-in collection orchestration; no SDK, CAN or control-loop patching.

This module is constructed only on a RECAP HTTP request. All motion is sent
through the repository's existing adapter/session APIs. No initialization hook.
"""
from __future__ import annotations
import json, threading, time, uuid, zipfile, io
from pathlib import Path


class Collection:
    def __init__(self, adapters, gateway, root, recorder_factory=None):
        self.a, self.gateway, self.root = adapters, gateway, Path(root)
        self.factory = recorder_factory
        self.operation = threading.Lock()
        self.cancel = threading.Event()
        self.lock = threading.RLock()
        self.phase, self.error = 'IDLE', None
        self.meta, self.folder, self.recorder = {}, None, None
        self.model_logs, self.export_path = [], None
        self.pico_owner = None
        self.export_lock = threading.Lock()
        self.root.mkdir(parents=True,exist_ok=True)
        for category in ('dataset','failed'):(self.root/category).mkdir(exist_ok=True)
        # Restore only a finished recording for download, never its controls.
        if self.root.is_dir():
            candidates=list(self.root.iterdir())
            for category in ('dataset','failed'):candidates.extend((self.root/category).iterdir())
            for folder in sorted(candidates, key=lambda p:p.stat().st_mtime_ns, reverse=True):
                if not folder.is_dir(): continue
                try:
                    meta = json.loads((folder/'episode.json').read_text(encoding='utf8'))
                except (OSError, ValueError): continue
                if meta.get('finalized') and meta.get('outcome'):
                    self.folder = self.export_path = folder
                    self.meta = meta
                    self.phase = 'FINISHED'
                    break

    def state(self):
        with self.lock:
            shown=self.phase
            if self.phase=='HUMAN' and not self.a.pico.snapshot().get('connected'):shown='HUMAN_WAITING'
            if self.phase=='MODEL' and self.a.pi05.snapshot().get('priming'):shown='PRIMING'
            return dict(state=shown, recording=self.phase in {
                'STARTING','MODEL','TO_HUMAN','HUMAN','TO_MODEL','STOPPING'},
                interventions=self.meta.get('interventions',0), error=self.error,
                export_ready=self.export_path is not None,
                data_path=str(self.folder.resolve()) if self.folder else None,
                export_file_path=str(self.export_path.with_suffix('.zip').resolve()) if self.export_path else None,
                labels={key:self.meta.get(key) for key in ('task','scene','purpose','outcome','autonomous_success','assisted_success')},
                human_intervened=self.meta.get('interventions',0)>0,
                raw_quality=(self.meta.get('raw_result') or {}).get('quality'),
                pico_connected=bool(self.a.pico.snapshot().get('connected')),
                pico_control_enabled=self.a.pico.snapshot().get('control_enabled', True))

    def connect_pico(self, body):
        with self.operation:
            client = str(body.get('client_id', ''))
            session = self.a.osc.fast_state().get('session') or {}
            if (session.get('state') != 'ACTIVE' or session.get('id') != body.get('session_id')
                    or session.get('client_id') != client):
                raise PermissionError('PICO standby requires the current client session')
            if self.state()['recording']:
                self._owner(client)
                if self.phase != 'MODEL': raise RuntimeError('Cannot reconnect during handover or PICO control')
            self.a.pico.set_control_enabled(False)
            self.pico_owner = client
            result = self.gateway.start_session(session['id'], client)
            self.a.osc.bind_source(session['id'], client, 'pi05')
            return result

    def leave(self, client):
        if self.pico_owner and client != self.pico_owner:
            raise PermissionError('PICO standby belongs to another browser')
        if self.state()['recording']: self.finish(client, 'aborted', 'RECAP mode left')
        with self.operation:
            self.a.pico.set_control_enabled(False)
            self.gateway.invalidate()
            self.a.pico.set_control_enabled(True)
            self.pico_owner = None
            return self.state()

    def _save(self, event):
        if self.folder is None: return
        now=time.perf_counter_ns()
        with (self.folder/'transitions.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(perf_ns=now,event=event,state=self.phase),ensure_ascii=False)+'\n')
        temp=self.folder/'episode.json.tmp'
        temp.write_text(json.dumps(self.meta,ensure_ascii=False,indent=2),encoding='utf8')
        temp.replace(self.folder/'episode.json')

    def _owner(self, client):
        if client != self.meta.get('client_id'): raise PermissionError('Collection belongs to another browser')

    def _session(self):
        s=(self.a.osc.fast_state().get('session') or {})
        if s.get('id') != self.meta.get('session_id') or s.get('client_id') != self.meta.get('client_id') or s.get('state') != 'ACTIVE':
            raise RuntimeError('Collection control session is no longer active')
        return s

    def _wait_hold(self):
        deadline=time.monotonic()+5
        while True:
            if self.cancel.is_set(): raise RuntimeError('Collection operation cancelled')
            self._session()
            p=self.a.pi05.snapshot()
            pending=p.get('recovery',{}).get('old_run_pending',False)
            # Do not bind a new source while an old model request/row is active.
            row_busy=False
            lock=getattr(self.a.pi05,'_row_lock',None)
            if lock is not None:
                if lock.acquire(blocking=False): lock.release()
                else: row_busy=True
            g=p.get('gripper_transport') or {}
            ready=g.get('state') not in {'sending','queued','pending','running'}
            state=self.a.osc.state()
            if (not pending and not row_busy and ready and
                (state.get('diagnostics') or {}).get('trajectory_state')=='HOLD_READY'): return
            if time.monotonic()>=deadline: raise RuntimeError('Existing stop/HOLD has not completed; no handover granted')
            self.cancel.wait(.02)

    def _archive_model(self):
        if not self.meta.get('model_started'): return
        name,payload=self.a.pi05.export_inference_log()
        dest=self.folder/f'model_{len(self.model_logs):03d}.zip'
        dest.write_bytes(payload);self.model_logs.append(dest.name)
        self.meta['model_logs']=list(self.model_logs)
        self.meta['model_started']=False

    def start(self, body):
        if not self.operation.acquire(False): raise RuntimeError('Collection is switching')
        try:
            if self.state()['recording']: raise RuntimeError('An episode is already active')
            task=str(body.get('task','')).strip();scene=str(body.get('scene','')).strip()
            if not task or not scene: raise ValueError('Task and scene are required')
            if body.get('purpose','pilot') not in {'pilot','train','demo','test'}: raise ValueError('Invalid purpose')
            pi=self.a.pi05.snapshot();pico=self.a.pico.snapshot()
            if pi.get('execution_enabled') or pi.get('priming'):
                raise RuntimeError('Stop the original model/PICO control before starting a collection')
            if self.a.dataset.state().get('recording'): raise RuntimeError('Original dataset recorder is already active')
            metadata=pi.get('policy_metadata') or {}
            if (metadata.get('state_dim'),metadata.get('valid_action_dim'),metadata.get('action_horizon')) != (7,7,20):
                raise RuntimeError('Connected policy must declare state7/action7/H20')
            if not self.a.cameras.snapshot().get('ready'): raise RuntimeError('Open both original cameras first')
            self.meta=dict(task=task,scene=scene,purpose=body.get('purpose','pilot'),
                session_id=body.get('session_id'),client_id=body.get('client_id'),
                started_perf_ns=time.perf_counter_ns(),interventions=0,finalized=False,
                started_monotonic_ns=time.monotonic_ns(),
                policy_metadata=metadata,contract=dict(state_dim=7,valid_action_dim=7,horizon=20,fps=20,
                action='base-relative observation-start TCP targets and absolute gripper'))
            self._session()
            self.cancel=threading.Event();self.error=None;self.export_path=None;self.model_logs=[]
            if pico.get('session_id'):
                if pico['session_id'] != self.meta['session_id']:
                    self.gateway.rebind_standby(self.meta['session_id'], self.meta['client_id'])
                else:
                    self.a.pico.set_control_enabled(False)
                self.pico_owner=self.meta['client_id']
                self._wait_hold()
            else:
                self.a.pico.set_control_enabled(False)
                self.gateway.start_session(self.meta['session_id'], self.meta['client_id'])
                self.a.osc.bind_source(self.meta['session_id'], self.meta['client_id'], 'pi05')
                self.pico_owner=self.meta['client_id']
            self.folder=self._new_episode_folder()
            self.meta['episode_name']=self.folder.name
            self.phase='STARTING';self._save('begin')
            if self.meta['purpose']!='demo' and (pi.get('prompt') or (pi.get('config') or {}).get('model',{}).get('prompt'))!=task:
                self.a.pi05.update_config({'model':{'prompt':task}})
                deadline=time.monotonic()+10
                while True:
                    if self.cancel.is_set():raise RuntimeError('Start cancelled')
                    self._session()
                    current=self.a.pi05.snapshot()
                    policy=current.get('policy_metadata') or {}
                    if (current.get('model_state')=='CONNECTED' and
                        (current.get('protocol_compatibility') or {}).get('status')=='compatible' and
                        (policy.get('state_dim'),policy.get('valid_action_dim'),policy.get('action_horizon'))==(7,7,20)):
                        self.meta['policy_metadata']=policy
                        break
                    if time.monotonic()>=deadline:raise RuntimeError('Model handshake did not complete after saving task text')
                    self.cancel.wait(.05)
            if self.factory is None:
                from supervisor.tcp_vla_dataset_recorder import TcpVlaDatasetRecorder
                factory=TcpVlaDatasetRecorder
            else: factory=self.factory
            self.recorder=factory(self.a.osc,self.a.cameras,self.folder/'raw',context_provider=self.a.osc.input_context)
            self.a.osc.bind_source(self.meta['session_id'],self.meta['client_id'],'pi05')
            raw=self.recorder.start({'task':scene,'description':task})
            self.meta['raw_initial_path']=raw.get('episode_dir')
            if self.cancel.is_set(): raise RuntimeError('Start cancelled')
            if self.meta['purpose']=='demo':
                self.phase='TO_HUMAN';self.gateway.start_session(self.meta['session_id'],self.meta['client_id'])
                self.a.osc.bind_source(self.meta['session_id'], self.meta['client_id'], 'pico')
                self.a.pico.set_control_enabled(True)
                self.meta['interventions']=1;self.phase='HUMAN'
            else:
                self.a.pi05.start(self.meta['session_id'],self.meta['client_id'])
                self.meta['model_started']=True
                self.phase='MODEL'
            if self.cancel.is_set(): raise RuntimeError('Start cancelled')
            self._save('ready')
            threading.Thread(target=self._watch,args=(self.folder,),name='recap-session-watch',daemon=True).start()
            return self.state()
        except Exception as e:
            self.error = str(e)
            if self.phase in {'STARTING','TO_HUMAN','MODEL','HUMAN'}: self._end('fault',str(e))
            raise
        finally: self.operation.release()

    def switch(self, client, human):
        self._owner(client)
        if not self.operation.acquire(False): raise RuntimeError('Already switching')
        try:
            if self.phase != ('MODEL' if human else 'HUMAN'): raise RuntimeError('Invalid handover state')
            self._session();self.phase='TO_HUMAN' if human else 'TO_MODEL';self._save('handover_requested')
            if human:
                model_osc = self.a.pi05.osc
                previous = model_osc.preserve_session_on_hold
                model_osc.preserve_session_on_hold = True
                try:
                    self.a.pi05.stop('RECAP takeover')
                finally:
                    model_osc.preserve_session_on_hold = previous
                self._wait_hold();self._archive_model()
                self.gateway.start_session(self.meta['session_id'],self.meta['client_id'])
                if self.a.pico.snapshot().get('connected'):
                    self.a.osc.bind_source(self.meta['session_id'], self.meta['client_id'], 'pico')
                    self.a.pico.set_control_enabled(True)
                self.meta['interventions']+=1;self.phase='HUMAN'
            else:
                self.a.pico.set_control_enabled(False);self._wait_hold()
                self.a.pi05_start(self.meta['session_id'],self.meta['client_id'])
                self.meta['model_started']=True
                self.phase='MODEL'
            if self.cancel.is_set(): raise RuntimeError('Handover cancelled')
            self._save('handover_complete');return self.state()
        except Exception as e:
            self._end('fault',str(e));raise
        finally:self.operation.release()

    def finish(self, client, outcome, reason=''):
        if outcome not in {'success','failure','aborted'}:raise ValueError('Invalid outcome')
        if not self.state()['recording']:return self.state()
        self._owner(client);self.cancel.set()
        with self.operation:
            if not self.state()['recording']:return self.state()
            self._end(outcome,reason);return self.state()

    def _end(self, outcome, reason):
        self.phase='STOPPING';self.meta['ended_perf_ns']=time.perf_counter_ns()
        errors=[]
        try:
            p=self.a.pi05.snapshot()
            if p.get('session_id')==self.meta['session_id'] and p.get('client_id')==self.meta['client_id']:
                self.a.pi05.stop('RECAP ended')
        except Exception as e:errors.append(str(e))
        try:
            if self.gateway.status().get('session_id')==self.meta['session_id']:
                self.a.pico.set_control_enabled(False)
        except Exception as e:errors.append(str(e))
        try:self._archive_model()
        except Exception as e:errors.append('model log export: '+str(e))
        if self.recorder:
            try:self.meta['raw_result']=self.recorder.stop('completed' if outcome=='success' else 'failed',reason,
                context_end_perf_ns=self.meta['ended_perf_ns'])
            except Exception as e:errors.append('raw collection: '+str(e))
            self.recorder=None
        self.meta.update(outcome=outcome,reason=reason,errors=errors,finalized=True,
            ended_monotonic_ns=time.monotonic_ns(),
            autonomous_success=outcome=='success' and not self.meta['interventions'],
            assisted_success=outcome=='success' and bool(self.meta['interventions']))
        trace=getattr(self.a,'pico_trace_logger',None)
        if trace is not None:
            try:
                # Copy the original receive/accepted-target journal. No new
                # observer, monkeypatch, SDK access or CAN reader is installed.
                deadline=time.monotonic()+2
                while not trace._queue.empty() and time.monotonic()<deadline:time.sleep(.02)
                time.sleep(.05)
                raw=Path(trace.path).read_bytes()
                (self.folder/'original_pico_trace.jsonl').write_bytes(raw)
                self.meta['pico_trace_scope']='original process journal; filter by started/ended_monotonic_ns'
                self.meta['pico_trace_queue_remaining']=trace._queue.qsize()
                if trace._queue.qsize():errors.append('Original PICO trace still has pending records')
                for line in raw.splitlines():json.loads(line)
            except Exception as e:errors.append('PICO trace export: '+str(e))
        self.error='; '.join(([reason] if outcome=='fault' and reason else [])+errors) or None
        self.phase='ERROR' if self.error else 'FINISHED'
        self._route_episode()
        self._save('ended');self.export_path=self.folder

    def _watch(self, folder):
        while self.folder == folder and self.state()['recording']:
            time.sleep(.2)
            if self.folder != folder:return
            if self.phase not in {'MODEL','HUMAN'}:continue
            try:
                self._session()
                if self.phase=='MODEL':
                    p=self.a.pi05.snapshot()
                    if p.get('last_error') or not (p.get('execution_enabled') or p.get('priming')):
                        raise RuntimeError(p.get('last_error') or 'Model output stopped')
                else:
                    p=self.a.pico.snapshot()
                    if p.get('connected') and not p.get('control_enabled', True):
                        if self.operation.acquire(False):
                            try:
                                if self.phase=='HUMAN' and not self.cancel.is_set():
                                    self._session();self._wait_hold()
                                    self.a.osc.bind_source(self.meta['session_id'],self.meta['client_id'],'pico')
                                    self.a.pico.set_control_enabled(True)
                                    self._save('pico_connection_ready')
                            finally:self.operation.release()
                    if p.get('state')=='IDLE' or p.get('last_error'):raise RuntimeError(p.get('last_error') or 'PICO receiver stopped')
            except Exception as e:
                if self.operation.acquire(False):
                    try:
                        if self.phase in {'MODEL','HUMAN'}:self.cancel.set();self._end('fault',str(e))
                    finally:self.operation.release()

    def _new_episode_folder(self):
        self.root.mkdir(parents=True,exist_ok=True)
        indices=[]
        for parent in (self.root,self.root/'dataset',self.root/'failed'):
            parent.mkdir(exist_ok=True)
            for path in parent.iterdir():
                name=path.stem if path.is_file() and path.suffix=='.zip' else path.name
                if name.startswith('episode-') and name[8:].isdigit(): indices.append(int(name[8:]))
        index=max(indices,default=0)+1
        while True:
            folder=self.root/'failed'/f'episode-{index:05d}'
            if folder.with_suffix('.zip').exists():
                index+=1;continue
            try:
                folder.mkdir()
                return folder
            except FileExistsError:index+=1

    def _route_episode(self):
        category='dataset' if self.meta.get('outcome')=='success' else 'failed'
        old=self.folder.resolve();target=(self.root/category/old.name).resolve()
        root=self.root.resolve()
        if not old.is_relative_to(root) or not target.is_relative_to(root):
            raise RuntimeError('Episode path outside collection root')
        if old!=target:
            if target.exists():raise RuntimeError('Episode destination already exists')
            old.rename(target)
            self.folder=target
            raw=self.meta.get('raw_result') or {}
            if raw.get('episode_dir'):
                path=Path(raw['episode_dir']).resolve()
                if path.is_relative_to(old):raw['episode_dir']=str(target/path.relative_to(old))
        self.meta['storage_category']=category

    def prepare_export(self):
        with self.export_lock:
            with self.lock:
                if self.state()['recording'] or not self.export_path:
                    raise RuntimeError('Finish the episode before export')
                folder = self.export_path
            target = folder.with_suffix('.zip')
            if target.is_file(): return target
            temp = target.with_suffix('.zip.tmp')
            try:
                with zipfile.ZipFile(temp,'w',allowZip64=True) as archive:
                    for file in folder.rglob('*'):
                        if file.is_file():
                            compression = zipfile.ZIP_STORED if file.suffix.lower() in {'.jpg','.jpeg','.png','.mp4','.zip'} else zipfile.ZIP_DEFLATED
                            archive.write(file,file.relative_to(folder),compress_type=compression)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
            return target

    def export(self):
        target = self.prepare_export()
        return target.name, target.read_bytes()


_creation=threading.Lock()
def manager(handler):
    with _creation:
        value=getattr(handler.runtime,'recap_extension',None)
        if value is None:
            value=Collection(handler.runtime.require_adapters(),handler.pico_gateway,
                Path(__file__).resolve().parents[2]/'recap_dataset_v2')
            handler.runtime.recap_extension=value
        return value

def get(handler,path):
    if path.startswith('/api/recap2/learning/'):
        from supervisor import recap_learning
        return recap_learning.get(handler,path)
    if path=='/api/recap2/state':return handler._json_ok(manager(handler).state())
    if path=='/api/recap2/export':
        target=manager(handler).prepare_export()
        handler.send_response(200);handler.send_header('Content-Type','application/zip')
        handler.send_header('Content-Disposition',f'attachment; filename="{target.name}"')
        handler.send_header('Content-Length',str(target.stat().st_size));handler.end_headers()
        with target.open('rb') as stream:
            while block := stream.read(1024*1024): handler.wfile.write(block)
        return
    raise ValueError('Unknown collection endpoint')

def post(handler,path,body):
    if path.startswith('/api/recap2/learning/'):
        from supervisor import recap_learning
        return recap_learning.post(handler,path,body)
    if path=='/api/recap2/start':
        from supervisor import recap_learning
        recap_learning.assert_collection_allowed()
    m=manager(handler);client=str(body.get('client_id',''))
    if path=='/api/recap2/start':result=m.start(body)
    elif path=='/api/recap2/connect-pico':result=m.connect_pico(body)
    elif path=='/api/recap2/leave':result=m.leave(client)
    elif path=='/api/recap2/takeover':result=m.switch(client,True)
    elif path=='/api/recap2/return':result=m.switch(client,False)
    elif path=='/api/recap2/finish':result=m.finish(client,body.get('outcome','aborted'),body.get('reason',''))
    else:raise ValueError('Unknown collection endpoint')
    return handler._json_ok(result)

def guard(runtime,path,body=None):
    m=getattr(runtime,'recap_extension',None)
    if m is None or not m.state()['recording']:return
    if path=='/api/osc/command' and (body or {}).get('type')=='hold':return
    if path in {'/api/pi05/start','/api/pi05/config','/api/adapters/pico/connect',
        '/api/osc/session/start','/api/osc/command','/api/dataset/start','/api/cameras/config',
        '/api/cameras/deactivate','/api/adapters/pico/mapping','/api/adapters/pico/sensitivity'}:
        raise RuntimeError('Finish RECAP collection before starting or reconfiguring another controller')
