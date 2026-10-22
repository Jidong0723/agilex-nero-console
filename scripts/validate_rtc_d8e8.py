"""No-motion deployed D8/E8 protocol and sampler check on recorded inputs."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supervisor.openpi_client import OpenPIClient


def main():
    out=ROOT/'runtime/rtc_d8e8_20261021';out.mkdir(parents=True,exist_ok=True)
    saved=np.load(ROOT/'runtime/rtc_episode13_optimization_20261021/deployed_exp/observations.npz')
    state=saved['6_state'];old=json.loads(str(saved['6_context']))['absolute_prefix']
    observation={'observation/state':state,'observation/image':saved['6_image'],
                 'observation/wrist_image':saved['6_wrist_image'],'prompt':str(saved['prompt'])}
    client=OpenPIClient('127.0.0.1',18762,60)
    report=dict(hardware_commands=0,metadata=client.metadata,rows=[],
                limitation='Recorded input with synthetic 8/8 protocol context; not a robot or original-run replay')
    def infer(payload):
        nonlocal client
        for attempt in range(20):
            try:return client.infer(payload)
            except RuntimeError as exc:
                if 'busy' not in str(exc).lower() or attempt==19:raise
                client.close();time.sleep(.11+.013*attempt);client=OpenPIClient('127.0.0.1',18762,60)
    try:
        metadata=client.metadata
        assert metadata['ready'] and metadata['prefix_attention_schedule']=='exp'
        assert metadata['num_steps']==20 and metadata['terminal_refinement_steps']==10
        assert metadata['rtc_requires_execution_within_prefix'] is False
        assert metadata['rtc_late_response_policy']=='guided_same_time_continuity_v1'
        assert metadata['rtc_warmup_timing']==dict(inference_delay=8,execute_horizon=8)
        normal=infer(dict(observation,rtc_diagnostic_noise_seed=0))
        assert np.asarray(normal['actions']).shape==(20,7)
        hold=[*map(float,state[:3]),*Rotation.from_rotvec(state[3:6]).as_quat().tolist(),float(state[6])]
        for phase,reference,end in [('bootstrap_hold',[hold]*20,20),('replan',old[:12],12)]:
            for number,seed in enumerate((None,0,1,2),1):
                context=dict(version='nero.rtc.v2',phase=phase,run_epoch=8,request_id=number,
                    observation_tick=0,plan_revision=1,freeze_until=8,
                    observation_perf_ns=1_000_000_000,timeline_origin_perf_ns=1_000_000_000,
                    freeze_until_perf_ns=1_400_000_000,action_row0_offset_s=0,period_s=.05,
                    inference_delay=8,execute_horizon=8,prefix_attention_horizon=end,absolute_prefix=reference)
                payload=dict(observation,rtc_context=context)
                if seed is not None:payload['rtc_diagnostic_noise_seed']=seed
                started=time.perf_counter();reply=infer(payload)
                elapsed=(time.perf_counter()-started)*1000
                actions=np.asarray(reply['actions']);rtc=reply['rtc'];sampler=rtc['sampler_config']
                assert actions.shape==(20,7) and np.isfinite(actions).all()
                assert rtc['inference_delay']==8 and rtc['execute_horizon']==8
                assert rtc['observation_perf_ns']==context['observation_perf_ns']
                assert sampler['prefix_attention_schedule']==('ones' if phase=='bootstrap_hold' else 'exp')
                assert sampler['num_steps']==20 and sampler['terminal_refinement_steps']==10
                assert len(sampler['integration_times'])==21
                residual=rtc['prefix_residuals'];assert len(residual)==end
                if phase=='replan':
                    expected=np.clip((7-np.arange(12))/5+1,0,1)
                    expected=expected*np.expm1(expected)/(np.e-1)
                    np.testing.assert_allclose([row['guidance_weight'] for row in residual],expected,atol=1e-6)
                report['rows'].append(dict(phase=phase,seed=seed,end_to_end_ms=elapsed,
                    noise_mode='production_rng' if seed is None else 'fixed_diagnostic_seed',
                    server_timing=reply['policy_timing'],
                    nominal_400ms_margin=400-elapsed,first8_position_max_mm=max(row['position_error_m'] for row in residual[:8])*1000,
                    predicted_gripper_min=float(actions[:,6].min()),predicted_gripper_max=float(actions[:,6].max()),
                    diagnostics=rtc))
                print(json.dumps({k:v for k,v in report['rows'][-1].items() if k!='diagnostics'}),flush=True)
    finally:
        client.close()
        (out/'deployed_check_production_rng.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
