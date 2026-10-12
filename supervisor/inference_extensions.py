"""Optional inference connections, passive journals and bounded TCP handoff.

No robot object, transport ownership, solver or motion command is defined here.
Both cloud and local inference use this same implementation.
"""
from __future__ import annotations
import copy
import io
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid
import zipfile
import numpy as np
from shared.schemas import jsonable


def safe_log_value(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,list):return [safe_log_value(x) for x in value]
    if isinstance(value,dict):return {str(k):safe_log_value(v) for k,v in value.items()}
    return value


def connection_config(config, request):
    profiles=copy.deepcopy(config.get('inference_profiles') or {
        'active':'autodl','autodl':{'host':'127.0.0.1','port':8001},
        'local5090':{'host':'127.0.0.1','port':8000}})
    name=request.get('profile')
    if name not in ('autodl','local5090'):raise ValueError('Unknown inference profile')
    endpoint=dict(profiles[name])
    endpoint.update({k:request[k] for k in ('host','port') if k in request})
    host=str(endpoint['host']).strip()
    port=endpoint['port']
    if not host or any(c.isspace() for c in host) or isinstance(port,bool) or not 1<=int(port)<=65535:
        raise ValueError('Invalid inference endpoint')
    endpoint={'host':host,'port':int(port)}
    profiles[name]=endpoint;profiles['active']=name
    result=copy.deepcopy(config);result['inference_profiles']=profiles
    result['model'].update(host=host,port=int(port))
    # Model and execution contracts are shared. Profiles only select endpoints.
    result.setdefault('connection',{}).update(policy_host=host,policy_port=int(port),ssh_forward_port=int(port))
    return result


def wait_deadline(stop_event, deadline):
    """Cancelable QPC deadline: Windows coarse ticks must not shorten a step."""
    while not stop_event.is_set():
        remaining=deadline-time.perf_counter()
        if remaining<=0:return False
        if stop_event.wait(remaining):return True
    return True


def pose_error(pose,target):
    try:
        a=np.asarray(pose['position_m'],float);b=np.asarray(target['position_m'],float)
        q=np.asarray(pose['orientation_xyzw'],float);r=np.asarray(target['orientation_xyzw'],float)
        if a.shape!=(3,) or b.shape!=(3,) or q.shape!=(4,) or r.shape!=(4,):return math.inf,math.inf
        if not all(np.isfinite(x).all() for x in (a,b,q,r)) or min(np.linalg.norm(q),np.linalg.norm(r))<1e-12:return math.inf,math.inf
        cosine=abs(float(q@r)/(np.linalg.norm(q)*np.linalg.norm(r)))
        return float(np.linalg.norm(a-b)),2*math.acos(min(1.,cosine))
    except (KeyError,TypeError,ValueError):return math.inf,math.inf


def tcp_delta(before,after):
    if before is None:return None
    try:
        q=np.asarray(after['orientation_xyzw'],float);r=np.asarray(before['orientation_xyzw'],float)
        q=q/np.linalg.norm(q);r=r/np.linalg.norm(r)
        u,v=q[:3],-r[:3];xyz=q[3]*v+r[3]*u+np.cross(u,v);w=q[3]*r[3]-u@v
        if w<0:xyz,w=-xyz,-w
        n=np.linalg.norm(xyz);rot=xyz/n*2*np.arctan2(n,w) if n>1e-12 else np.zeros(3)
        return {'translation_base_m':(np.array(after['position_m'])-before['position_m']).tolist(),'rotvec_base_rad':rot.tolist()}
    except (ValueError,KeyError,TypeError):return None


class InferenceJournal:
    def __init__(self,root,metadata):
        self.run_id=time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        self.path=Path(root)/self.run_id/'run.jsonl';self.path.parent.mkdir(parents=True)
        self.lock=threading.RLock();self.queue=queue.Queue();self.closed=False;self.error=None;self.counts={}
        self.stop_event=threading.Event();self.previous_pose=None;self.previous_source=None
        self.writer=threading.Thread(target=self._write,daemon=True);self.writer.start()
        self.sampler=None
        self.append('run_start',metadata=metadata,target_tcp_sample_hz=50)
    def append(self,kind,**fields):
        with self.lock:
            if self.closed:return
            record=safe_log_value(jsonable({'record_type':kind,'wall_time_ns':time.time_ns(),
                    'recorded_perf_counter_ns':time.perf_counter_ns(),**fields}))
            self.counts[kind]=self.counts.get(kind,0)+1;self.queue.put(record)
    def _write(self):
        try:
            with self.path.open('w',encoding='utf-8',newline='\n') as f:
                while True:
                    row=self.queue.get()
                    if row is None:break
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        except Exception as exc:self.error=str(exc)
    def start_sampling(self,reader):
        def collect():
            deadline=time.perf_counter()
            while not self.stop_event.is_set():
                try:
                    state=reader();execution=state.get('execution') or {}
                    pose=execution.get('measured_tcp_pose')
                    source=execution.get('sample_monotonic_ns') or execution.get('control_sample_id')
                    fresh=pose is not None and source is not None and source!=self.previous_source
                    delta=tcp_delta(self.previous_pose,pose) if fresh else None
                    if fresh:self.previous_pose=copy.deepcopy(pose);self.previous_source=source
                    self.append('tcp_sample',measured_tcp_pose=pose,tcp_delta_base=delta,fresh_source_sample=fresh,
                        source_sample_monotonic_ns=execution.get('sample_monotonic_ns'),control_sample_id=execution.get('control_sample_id'),
                        measured_joint_state_rad=execution.get('measured_joint_state_rad'),joint_velocity_rad_s=execution.get('joint_velocity_rad_s'),
                        feedback_age_s=execution.get('feedback_age_s'),target_tcp=(state.get('command') or {}).get('target_tcp'),
                        gripper_width_m=(state.get('gripper') or {}).get('width_m'))
                except Exception as exc:self.append('tcp_sample_error',error=str(exc))
                deadline+=.02
                delay=deadline-time.perf_counter()
                if delay>0:time.sleep(delay)
                else:deadline=time.perf_counter()
        self.sampler=threading.Thread(target=collect,daemon=True);self.sampler.start()
    def finish(self,reason):
        self.stop_event.set()
        if self.sampler and self.sampler is not threading.current_thread():self.sampler.join(timeout=2)
        with self.lock:
            if self.closed:return
            self.append('run_end',reason=reason);self.closed=True;self.queue.put(None)
        self.writer.join(timeout=3)
        if self.writer.is_alive():self.error='Journal writer did not finish'
    def summary(self):return {'run_id':self.run_id,'completed':self.closed,'write_error':self.error,'counts':dict(self.counts),'tcp_sample_target_hz':50}
    def export(self):
        if not self.closed or self.error:raise RuntimeError('Stop inference and finish log writing before exporting')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(self.path,'run.jsonl');z.writestr('summary.json',json.dumps(self.summary(),ensure_ascii=False))
        return self.run_id+'.zip',buffer.getvalue()


def wait_for_arrival(adapter,target,session_id,client_id):
    with adapter.lock:settings=dict(adapter.config['execution'])
    budget=float(settings.get('arrival_wait_s',0.0))
    started=time.perf_counter();deadline=started+budget
    adapter._record_inference('tcp_arrival_wait_started',target_tcp=target,max_wait_s=budget)
    while not adapter.stop_event.is_set():
        with adapter.lock:
            if not adapter.state.get('execution_enabled'):return None
            state=copy.deepcopy(adapter._osc_snapshot)
        session=state.get('session') or {}
        if session.get('id')!=session_id or session.get('client_id')!=client_id or session.get('state')!='ACTIVE':return None
        execution=state.get('execution') or {};command=state.get('command') or {}
        pos,angle=pose_error(execution.get('measured_tcp_pose'),target)
        target_pos,target_angle=pose_error(command.get('target_tcp'),target)
        generation=command.get('target_generation')
        reached=(generation is not None and generation==execution.get('target_generation')
                 and target_pos<1e-9 and target_angle<1e-6
                 and pos<=float(settings.get('arrival_position_tolerance_m',.003))
                 and angle<=float(settings.get('arrival_orientation_tolerance_rad',math.radians(2))))
        if reached or budget==0 or time.perf_counter()>=deadline:
            adapter._record_inference('tcp_arrival_reached' if reached else 'tcp_arrival_wait_bypassed' if budget==0 else 'tcp_arrival_wait_timeout',
                target_tcp=target,measured_tcp_pose=execution.get('measured_tcp_pose'),position_error_m=pos if math.isfinite(pos) else None,
                orientation_error_rad=angle if math.isfinite(angle) else None,wait_s=time.perf_counter()-started)
            return state
        adapter.stop_event.wait(min(.01,max(0,deadline-time.perf_counter())))
    return None
