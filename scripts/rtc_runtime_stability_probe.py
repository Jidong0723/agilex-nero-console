"""No-motion deployment probe: normal/first/later calls, tokens and fixed noise."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supervisor.openpi_client import OpenPIClient


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    saved=np.load(ROOT/'runtime/rtc_episode16_20261021/samples/observations.npz')
    state=saved['6_state'];base=json.loads(str(saved['6_context']))
    prefix=[*state[:3].tolist(),*Rotation.from_rotvec(state[3:6]).as_quat().tolist(),float(state[6])]
    base.update(inference_delay=6,execute_horizon=6,prefix_attention_horizon=14)
    origin=base['timeline_origin_perf_ns'];stamp=base['observation_perf_ns']
    base['freeze_until']=(stamp+300_000_000-origin+49_999_999)//50_000_000
    base['freeze_until_perf_ns']=origin+base['freeze_until']*50_000_000
    out=ROOT/'runtime/rtc_runtime_stability_20261021';out.mkdir(exist_ok=True)
    report=dict(hardware_commands=0,rows=[],busy_rejections=0,
        limitations='Saved RGB/state and repeated fixed references; no motion, task rollout or CUDA graph profiler')
    client=OpenPIClient('127.0.0.1',18762,90);report['metadata']=client.metadata
    try:
        for repeat in range(args.rounds):
            for prompt_id,count in enumerate((1,15,45)):
                observation={'observation/state':state,'observation/image':saved['6_image'],
                    'observation/wrist_image':saved['6_wrist_image'],
                    'prompt':'Pick up the yellow cube. '*count}
                for phase in ('normal','bootstrap_hold','replan'):
                    for fixed in (False,True):
                        payload=dict(observation)
                        if phase!='normal':
                            context=dict(base,phase=phase,absolute_prefix=[prefix]*(20 if phase=='bootstrap_hold' else 14),
                                         prefix_attention_horizon=20 if phase=='bootstrap_hold' else 14)
                            if phase=='bootstrap_hold':
                                context.update(observation_tick=0,observation_perf_ns=origin,
                                    freeze_until=6,freeze_until_perf_ns=origin+300_000_000)
                            payload['rtc_context']=context
                        if fixed:payload['rtc_diagnostic_noise_seed']=11649
                        for attempt in range(15):
                            began=time.perf_counter()
                            try:reply=client.infer(payload);break
                            except RuntimeError as exc:
                                if 'busy' not in str(exc).lower() or attempt==14:raise
                                report['busy_rejections']+=1;client.close();time.sleep(.15)
                                client=OpenPIClient('127.0.0.1',18762,90)
                        row=dict(repeat=repeat,prompt_id=prompt_id,phase=phase,fixed_noise=fixed,
                            elapsed_ms=(time.perf_counter()-began)*1000,timing=reply.get('policy_timing'),
                            returned_shape=list(np.shape(reply['actions'])))
                        assert row['returned_shape']==[20,7] and np.isfinite(reply['actions']).all()
                        if phase!='normal':assert row['timing']['prepared_before_call']
                        report['rows'].append(row)
                        print(json.dumps(row),flush=True)
                        (out/'probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:client.close()
    print('completed',len(report['rows']),'requests; zero hardware commands',flush=True)


if __name__=='__main__':main()
