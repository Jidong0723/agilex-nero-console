"""Read-only inference diagnostics. This module never calls a control API."""
from __future__ import annotations
import copy
import io
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
import uuid
import zipfile


def quaternion_delta(now, before):
    def unit(q):
        if len(q) != 4 or not all(math.isfinite(float(x)) for x in q): raise ValueError('Invalid quaternion')
        n=math.sqrt(sum(float(x)**2 for x in q))
        if n < 1e-12: raise ValueError('Zero quaternion')
        return [float(x)/n for x in q]
    a,b=unit(now),unit(before)
    u,v=a[:3],[-x for x in b[:3]]
    xyz=[a[3]*v[i]+b[3]*u[i]+u[(i+1)%3]*v[(i+2)%3]-u[(i+2)%3]*v[(i+1)%3] for i in range(3)]
    w=a[3]*b[3]-sum(x*y for x,y in zip(u,v))
    if w<0: xyz=[-x for x in xyz]; w=-w
    n=math.sqrt(sum(x*x for x in xyz))
    return [x*2*math.atan2(n,w)/n for x in xyz] if n>1e-12 else [0.,0.,0.]


def distance(actual, target):
    a,b=actual['position_m'],target['position_m']
    if len(a)!=3 or len(b)!=3 or not all(math.isfinite(float(x)) for x in [*a,*b]): raise ValueError('Invalid translation')
    r=quaternion_delta(actual['orientation_xyzw'],target['orientation_xyzw'])
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))),math.sqrt(sum(x*x for x in r))


def await_target(read_cache, target, identity, sequence, previous_id, deadline, cancelled):
    result={'outcome':'timeout','detail':'no matching fresh feedback'}
    started=time.perf_counter_ns()
    while not cancelled.is_set() and time.perf_counter()<deadline:
        frame=read_cache(); s=frame.get('session') or {}; cmd=frame.get('command') or {}; e=frame.get('execution') or {}
        if (s.get('id'),s.get('client_id'))!=identity or s.get('state')!='ACTIVE':
            result.update(outcome='cancelled',detail='session changed'); break
        sample=e.get('control_sample_id')
        current = cmd.get('sequence') == sequence
        current = current and cmd.get('target_generation') is not None and e.get('target_generation') == cmd.get('target_generation')
        fresh = sample is not None and previous_id is not None and sample > previous_id
        if current and fresh and e.get('measured_tcp_pose'):
            try:
                cp,cr=distance(cmd['target_tcp'],target)
                if cp<1e-8 and cr<1e-7:
                    pe,re=distance(e['measured_tcp_pose'],target)
                    result.update(position_error_m=pe,orientation_error_rad=re,detail='outside tolerance')
                    if pe<=.003 and re<=math.radians(2):
                        result.update(outcome='arrived',detail='within tolerance'); break
            except (KeyError,TypeError,ValueError): result['detail']='invalid cached pose'
        cancelled.wait(min(.005,max(0.,deadline-time.perf_counter())))
    if cancelled.is_set(): result.update(outcome='cancelled',detail='output stopped')
    result['wait_started_perf_ns']=started
    result['wait_finished_perf_ns']=time.perf_counter_ns()
    result['blocking_ms']=(result['wait_finished_perf_ns']-started)/1e6
    return result


def serializable(value):
    if isinstance(value,float) and not math.isfinite(value): return {'non_finite':str(value)}
    if isinstance(value,dict): return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serializable(v) for v in value]
    if hasattr(value,'tolist'): return serializable(value.tolist())
    return value


class CompletedRun:
    """Read-only handle to a finalized on-disk run; never starts worker threads."""
    def __init__(self, path, summary):
        self.path = Path(path)
        self.id = self.path.name
        self._summary = copy.deepcopy(summary)
        if not (self.path/'events.jsonl').is_file():
            self._summary.update(export_ready=False, write_error='Completed run events.jsonl is missing')

    def status(self):
        return copy.deepcopy(self._summary)

    def zip_bytes(self):
        if not self.status().get('export_ready'):
            raise RuntimeError('Only fully written, loss-free completed runs can be exported')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in ('events.jsonl', 'summary.json'):
                bundle.write(self.path/name, name)
        return self.id+'.zip', buffer.getvalue()


def latest_completed_run(directory):
    """Restore the newest finalized run once at startup, not on every UI poll.

    A summary is written only by finish(), after the writer has been joined.
    Running/partial runs are not exported. A finalized run with a recorded
    write failure remains visible but disabled; do not silently hide data loss
    by falling back to an older successful run.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not re.fullmatch(r'\d{8}T\d{6}-[0-9a-f]{8}', path.name) or not path.is_dir() or path.is_symlink():
            continue
        try:
            summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict) or summary.get('run_id') != path.name or summary.get('finished') is not True:
            continue
        summary['export_ready'] = (summary.get('export_ready') is True
                                   and summary.get('write_error') is None
                                   and summary.get('lost_records') == 0)
        return CompletedRun(path, summary)
    return None


class RunCapture:
    def __init__(self, directory, metadata, cached_feedback):
        self.id=time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        self.path=Path(directory)/self.id
        self.path.mkdir(parents=True)
        self.events=queue.Queue(maxsize=12000)
        self.write_lock=threading.Lock()
        self.stop_sampling=threading.Event()
        self.closed=False; self.error=None; self.lost=0
        self.emit_count=0; self.slow_deepcopy_count=0; self.max_deepcopy_ms=0.0
        self.json_count=0; self.slow_json_count=0; self.max_json_ms=0.0
        self.write_count=0; self.max_write_ms=0.0
        self.sample_count=0; self.fresh_sample_count=0; self.reused_sample_count=0
        self.sample_first_ns=None; self.sample_last_ns=None
        self.max_sample_interval_ms=0.; self.max_cache_read_ms=0.
        self.missed_sampling_periods=0
        self.read_cache=cached_feedback
        self.writer=threading.Thread(target=self._writer,daemon=True,name='inference-event-writer')
        self.sampler=threading.Thread(target=self._sampler,daemon=True,name='inference-cache-50hz')
        self.writer.start()
        self.emit('run_start',metadata=metadata,sampling_target_hz=50)
        self.sampler.start()

    def emit(self, event, **payload):
        # Copy caller-owned data outside the enqueue lock. Large model chunks
        # must not hold up the independent feedback sampler.
        copy_started=time.perf_counter_ns()
        payload=copy.deepcopy(payload)
        copy_ms=(time.perf_counter_ns()-copy_started)/1e6
        self.emit_count+=1; self.max_deepcopy_ms=max(self.max_deepcopy_ms,copy_ms)
        if copy_ms>=5.0: self.slow_deepcopy_count+=1
        with self.write_lock:
            if self.closed: return
            item={'event':event,'perf_ns':time.perf_counter_ns(),'wall_ns':time.time_ns(),**payload}
            try: self.events.put_nowait(item)
            except queue.Full: self.lost+=1

    def _writer(self):
        try:
            with (self.path/'events.jsonl').open('w',encoding='utf-8') as stream:
                while True:
                    item=self.events.get()
                    if item is None: break
                    serialise_started=time.perf_counter_ns()
                    line=json.dumps(serializable(item),ensure_ascii=False,allow_nan=False)+'\n'
                    json_ms=(time.perf_counter_ns()-serialise_started)/1e6
                    self.json_count+=1; self.max_json_ms=max(self.max_json_ms,json_ms)
                    if json_ms>=5.0: self.slow_json_count+=1
                    write_started=time.perf_counter_ns()
                    stream.write(line)
                    self.write_count+=1
                    self.max_write_ms=max(self.max_write_ms,(time.perf_counter_ns()-write_started)/1e6)
                stream.flush()
        except Exception as exc: self.error=f'{type(exc).__name__}: {exc}'

    def _sampler(self):
        deadline=time.perf_counter(); last_key=None; last_pose=None; last_control=None
        while not self.stop_sampling.is_set():
            sampled_ns=time.perf_counter_ns()
            try:
                frame=self.read_cache(); execution=frame.get('execution') or {}
                cache_read_ms=(time.perf_counter_ns()-sampled_ns)/1e6
                control=execution.get('control_sample_id')
                revision=execution.get('feedback_revision')
                basis='feedback_revision' if revision is not None else 'control_sample_fallback'
                key=('feedback',revision) if revision is not None else ('control',control,execution.get('sample_monotonic_ns'))
                known=revision is not None or control is not None or execution.get('sample_monotonic_ns') is not None
                changed=known and key!=last_key
                pose=execution.get('measured_tcp_pose'); delta=None
                if pose and last_pose and changed:
                    delta=[a-b for a,b in zip(pose['position_m'],last_pose['position_m'])]+quaternion_delta(pose['orientation_xyzw'],last_pose['orientation_xyzw'])
                self.emit('feedback_sample',**frame,fresh_source_sample=changed,tcp_delta_base=delta,
                          sample_started_perf_ns=sampled_ns,cache_read_ms=cache_read_ms,
                          fresh_control_sample=control is not None and control!=last_control,freshness_basis=basis,
                          feedback_status='missing' if not known else 'new' if changed else 'reused',
                          reference_is_can_transmission=False,
                          can_record_available=(frame.get('last_cpv_dispatch') is not None))
                self.sample_count+=1
                self.fresh_sample_count+=int(changed)
                self.reused_sample_count+=int(known and not changed)
                self.max_cache_read_ms=max(self.max_cache_read_ms,cache_read_ms)
                if self.sample_first_ns is None: self.sample_first_ns=sampled_ns
                if self.sample_last_ns is not None:
                    self.max_sample_interval_ms=max(self.max_sample_interval_ms,(sampled_ns-self.sample_last_ns)/1e6)
                self.sample_last_ns=sampled_ns
                if changed:
                    last_key=key
                    if pose: last_pose=pose
                last_control=control
            except Exception as exc: self.emit('feedback_sample_error',error=str(exc))
            deadline+=.02
            now=time.perf_counter()
            if deadline<now:
                missed=int((now-deadline)/.02)+1
                self.missed_sampling_periods+=missed
                deadline+=missed*.02
            self.stop_sampling.wait(max(0.,deadline-time.perf_counter()))

    def finish(self, reason):
        if self.closed: return
        self.stop_sampling.set(); self.sampler.join(timeout=2)
        self.emit('run_end',reason=reason)
        with self.write_lock: self.closed=True
        if self.writer.is_alive():
            try: self.events.put(None,timeout=1)
            except queue.Full: self.error='log finalization queue is full'
            self.writer.join(timeout=5)
        if self.writer.is_alive(): self.error='log writer did not finish'
        try: (self.path/'summary.json').write_text(json.dumps(self.status(),ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as exc: self.error=f'{type(exc).__name__}: {exc}'

    def status(self):
        span=(self.sample_last_ns-self.sample_first_ns)/1e9 if self.sample_first_ns is not None and self.sample_last_ns is not None else 0.
        return {'run_id':self.id,'finished':self.closed,'write_error':self.error,'lost_records':self.lost,
                'logging_diagnostics':{'emit_count':self.emit_count,
                'slow_deepcopy_count':self.slow_deepcopy_count,'max_deepcopy_ms':self.max_deepcopy_ms,
                'json_count':self.json_count,'slow_json_count':self.slow_json_count,
                'max_json_ms':self.max_json_ms,'write_count':self.write_count,
                'max_write_ms':self.max_write_ms},
                'sampling_target_hz':50.,'feedback_record_count':self.sample_count,
                'actual_record_hz':(self.sample_count-1)/span if span>0 else None,
                'fresh_feedback_count':self.fresh_sample_count,'reused_feedback_count':self.reused_sample_count,
                'fresh_feedback_hz':max(0,self.fresh_sample_count-1)/span if span>0 else None,
                'max_sample_interval_ms':self.max_sample_interval_ms,'max_cache_read_ms':self.max_cache_read_ms,
                'missed_sampling_periods':self.missed_sampling_periods,
                'export_ready':self.closed and not self.writer.is_alive() and self.error is None and self.lost==0}

    def zip_bytes(self):
        if not self.status()['export_ready']: raise RuntimeError('Only fully written, loss-free completed runs can be exported')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in ('events.jsonl','summary.json'): bundle.write(self.path/name,name)
        return self.id+'.zip',buffer.getvalue()
