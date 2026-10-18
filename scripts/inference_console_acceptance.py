"""Validate the running console's native preview/shadow path, never hardware mode."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import uuid


def api(path, body=None):
    request = Request('http://127.0.0.1:8765'+path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if not result['ok']:
        raise RuntimeError(str(result))
    return result['data']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preview','shadow'],required=True)
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.seconds<=600:
        p.error('seconds must be 1..600')
    initial=api('/api/osc/state')
    first=api('/api/pi05/state')
    if initial['session']['state']!='IDLE' or first['execution_enabled'] or first['priming']:
        raise RuntimeError('Console must be idle; refusing to replace an operator session')
    baseline=initial['transport']['cpv_mailbox']['sent_count']
    run={'mode':args.mode,'passed':False,'samples':[],'source':'native RGB cache / production adapter',
         'initial_executed_steps':first['executed_steps']}
    identity=None
    started=time.perf_counter()
    try:
        if args.mode=='shadow':
            client='acceptance-'+uuid.uuid4().hex
            session=api('/api/osc/session/start',{'execution_mode':'shadow','client_id':client,'input_source':'pi05'})['session']
            identity={'session_id':session['id'],'client_id':client}
            api('/api/pi05/start',identity)
        while time.perf_counter()-started<args.seconds:
            state=api('/api/pi05/state')
            osc=api('/api/osc/state')
            if osc['transport']['cpv_mailbox']['sent_count']!=baseline:
                raise RuntimeError('Unexpected hardware CPV mailbox activity')
            if identity:
                if osc['session'].get('id')!=identity['session_id'] or osc['session'].get('execution_mode')!='shadow':
                    raise RuntimeError('Shadow ownership lost')
                api('/api/osc/session/heartbeat',identity)
                if state.get('last_error') and 'error' not in run:
                    run['error']=state['last_error']
                    run['fault_elapsed_s']=time.perf_counter()-started
                    run['steps_at_fault']=state['executed_steps']
                if 'error' in run and state['executed_steps']!=run['steps_at_fault']:
                    raise RuntimeError('Control continued after fault')
            elif osc['session']['state']!='IDLE' or state['execution_enabled'] or state['priming']:
                raise RuntimeError('Preview acquired control unexpectedly or operator changed session')
            sample={k:state.get(k) for k in ('preview','rtc','chunk_source','chunk_sequence','inference_ms','executed_steps','last_error','control_readiness')}
            sample['elapsed_s']=time.perf_counter()-started
            run['samples'].append(sample)
            if len(run['samples'])%20==0:
                print(json.dumps(sample),flush=True)
            time.sleep(.5)
        run['passed']='error' not in run and state['chunk_sequence']>first['chunk_sequence']+5
        if identity:
            run['passed'] &= state['executed_steps']>=20
    except Exception as exc:
        run['error']=str(exc)
    finally:
        run['elapsed_s']=time.perf_counter()-started
        if identity:
            osc=api('/api/osc/state')
            if osc['session'].get('id')==identity['session_id'] and osc['session'].get('execution_mode')=='shadow':
                stopped=api('/api/pi05/stop',{'reason':'bounded shadow acceptance complete'})
                run['hold_confirmed']=stopped.get('rtc',{}).get('hold_confirmed')
                api('/api/osc/session/stop',{'reason':'bounded shadow acceptance complete'})
                run['passed'] &= bool(run['hold_confirmed'])
        final=api('/api/osc/state')
        run['hardware_mailbox_sent_delta']=final['transport']['cpv_mailbox']['sent_count']-baseline
        run['final_session']=final['session']
        run['final_model_state']=api('/api/pi05/state')
        run['control_steps']=run['final_model_state']['executed_steps']-run['initial_executed_steps']
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(run,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in run.items() if k not in ('samples','final_model_state')}),flush=True)
    return 0 if run['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
