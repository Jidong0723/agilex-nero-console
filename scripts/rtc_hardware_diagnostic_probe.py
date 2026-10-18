"""Explicitly authorized, single-run native RTC hardware diagnostic (max 10 s)."""
import argparse
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
import uuid


def api(path, body=None, timeout=4):
    request = Request('http://127.0.0.1:8765'+path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not result['ok']:
        raise RuntimeError(str(result))
    return result['data']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-hardware', action='store_true', required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 10:
        parser.error('seconds must be >0 and <=10')
    initial = api('/api/osc/state')
    first = api('/api/pi05/state')
    if initial['session']['state'] != 'IDLE' or first['execution_enabled'] or first['priming']:
        raise RuntimeError('Refusing to replace an existing control session')
    if first['config']['execution']['mode'] != 'rtc':
        raise RuntimeError('RTC must already be saved; probe never changes model settings')
    if not first['control_readiness']['hardware']['allowed']:
        raise RuntimeError(str(first['control_readiness']['hardware']))
    if initial['authority']['safety_state'] != 'NORMAL':
        raise RuntimeError('Hardware safety is not NORMAL')
    run = {'seconds_limit': args.seconds, 'samples': [], 'passed': False}
    identity = None
    stop_lock = threading.Lock()
    done = threading.Event()
    started = time.perf_counter()

    def stop_owned():
        with stop_lock:
            if done.is_set():
                return
            try:
                current = api('/api/osc/state')['session']
                if identity and current.get('id') == identity['session_id'] and current.get('client_id') == identity['client_id']:
                    try:
                        stopped = api('/api/pi05/stop', {'reason': 'bounded RTC hardware diagnostic stop'}, timeout=6)
                        run['hold_confirmed'] = (stopped.get('rtc') or {}).get('hold_confirmed')
                    finally:
                        # Recheck ownership before releasing the session.
                        if api('/api/osc/state')['session'].get('id') == identity['session_id']:
                            api('/api/osc/session/stop', {'reason': 'bounded RTC hardware diagnostic complete'}, timeout=6)
                else:
                    run['ownership_lost'] = True
            except Exception as exc:
                run['stop_error'] = str(exc)
            finally:
                done.set()

    timer = None
    try:
        client = 'rtc-hardware-diagnostic-'+uuid.uuid4().hex
        session = api('/api/osc/session/start', {'execution_mode': 'hardware', 'client_id': client, 'input_source': 'pi05'}, timeout=10)['session']
        identity = {'session_id': session['id'], 'client_id': client}
        # The independent timer covers bootstrap and control, even if start blocks.
        started = time.perf_counter()
        timer = threading.Timer(args.seconds, stop_owned)
        timer.daemon = True
        timer.start()
        api('/api/pi05/start', identity, timeout=6)
        while not done.wait(.2):
            state = api('/api/pi05/state')
            sample = {k: state.get(k) for k in ('state', 'rtc', 'rtc_observation_health', 'executed_steps', 'last_error', 'inference_log')}
            sample['elapsed_s'] = time.perf_counter()-started
            run['samples'].append(sample)
            if state.get('last_error') or not (state.get('execution_enabled') or state.get('priming')):
                run['error'] = state.get('last_error') or 'Control stopped before deadline'
                break
    except Exception as exc:
        run['error'] = str(exc)
    finally:
        stop_owned()
        if timer:
            timer.cancel()
        run['elapsed_s'] = time.perf_counter()-started
        final = api('/api/osc/state')
        state = api('/api/pi05/state')
        run['final_session'] = final['session']
        run['final_authority'] = final['authority']
        run['final_rtc'] = state.get('rtc')
        run['inference_log'] = state.get('inference_log')
        run['hardware_mailbox_sent_delta'] = final['transport']['cpv_mailbox']['sent_count']-initial['transport']['cpv_mailbox']['sent_count']
        run['passed'] = not run.get('error') and not run.get('stop_error') and bool(run.get('hold_confirmed')) and final['session']['state'] == 'IDLE'
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(run, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in run.items() if k != 'samples'}), flush=True)
    return 0 if run['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
