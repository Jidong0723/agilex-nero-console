"""Replay saved episode observations through WebSocket; never import robot APIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.openpi_client import OpenPIClient
from inference.rtc.diagnostics import prefix_diagnostics
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, default=ROOT.parent/'dataset/failed_episodes/episode_000004')
    parser.add_argument('--events', type=Path, default=ROOT/'runtime/logs/inference_runs/20261021T093010-1241c74f/events.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/rtc_prefix_diagnostics_20261021')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--port', type=int, default=18762)
    parser.add_argument('--request-ids', default='6,12,17')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = [json.loads(line) for line in args.events.read_text(encoding='utf-8').splitlines()]
    frames = [json.loads(line) for line in (args.episode/'raw/camera_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    prompt = events[0]['metadata']['config']['model']['prompt']
    requests = {e['context']['request_id']: e for e in events if e['event'] == 'rtc_request'}
    manifest = []; payload = {}; inputs = []
    for rid in [int(value) for value in args.request_ids.split(',')]:
        request = requests[rid]; context = request['context']
        observation = {'observation/state': np.asarray(request['observation_state'], np.float32), 'prompt': prompt}
        image_info = []
        for source, key, stamp in zip(('external', 'wrist'), ('observation/image', 'observation/wrist_image'),
                                      request['observation_times']['image_perf_ns']):
            frame = min((f for f in frames if f['source'] == source), key=lambda f: abs(f['capture_monotonic_ns']-stamp))
            bgr = cv2.imread(str(args.episode/frame['image']))
            if bgr is None:
                raise ValueError('Saved image missing: '+frame['image'])
            observation[key] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
            image_info.append(dict(source=source, file=frame['image'], nearest_frame_offset_ms=(frame['capture_monotonic_ns']-stamp)/1e6))
        payload.update({f'{rid}_state': observation['observation/state'], f'{rid}_image': observation['observation/image'],
                        f'{rid}_wrist_image': observation['observation/wrist_image'], f'{rid}_context': np.asarray(json.dumps(context))})
        manifest.append(dict(request_id=rid, context=context, images=image_info))
        inputs.append((rid, observation, context))
    np.savez_compressed(args.output/'observations.npz', **payload, prompt=np.asarray(prompt))
    (args.output/'manifest.json').write_text(json.dumps(dict(episode=str(args.episode), events=str(args.events), samples=manifest,
        limitation='Exact logged state/reference/timestamps; nearest saved JPEG images, original inference RGB and noise were not logged.'), indent=2), encoding='utf-8')
    if args.prepare_only:
        print('Prepared '+str(args.output)); return
    client = OpenPIClient('127.0.0.1', args.port, 60)
    report = dict(metadata=client.metadata, hardware_commands=0, rows=[])
    try:
        for rid, observation, context in inputs:
            for seed in (0, 1, 2, 0):
                started = time.perf_counter()
                for attempt in range(20):
                    try:
                        reply = client.infer(dict(observation, rtc_context=context, rtc_diagnostic_noise_seed=seed))
                        break
                    except RuntimeError as exc:
                        if 'busy' not in str(exc).lower() or attempt == 19:
                            raise
                        client.close()
                        time.sleep(.11 + .013*attempt)
                        client = OpenPIClient('127.0.0.1', args.port, 60)
                rtc = reply['rtc']
                if rtc.get('schema_version') != 'nero.rtc.diagnostics.v1':
                    raise RuntimeError('Cloud has not loaded diagnostic wrapper')
                actions = np.asarray(reply['actions'], dtype=float)
                weights = np.asarray([r['guidance_weight'] for r in rtc['prefix_residuals']])
                check = prefix_diagnostics(actions, observation['observation/state'], context['absolute_prefix'][:len(weights)], weights,
                                            delay=context['inference_delay'], config=rtc['sampler_config'])
                np.testing.assert_allclose([r['position_error_m'] for r in rtc['prefix_residuals']],
                                           [r['position_error_m'] for r in check['prefix_residuals']], rtol=1e-5, atol=1e-7)
                splice = next((event for event in events if event['event'] == 'rtc_splice'
                               and event['observation_perf_ns'] == context['observation_perf_ns']), None)
                execution_metrics = None
                if splice is not None:
                    # The log's first_future_tick is the minimum queued key;
                    # it can still belong to retained OLD targets. Never use
                    # it as the replacement boundary before the frozen end.
                    cut = splice.get('replacement_tick', max(context['freeze_until'], splice['first_future_tick']))
                    previous = next((event for event in events if event['event'] == 'rtc_dispatch' and event['tick'] == cut-1), None)
                    rows = ((context['timeline_origin_perf_ns'] + np.arange(cut, cut+context['execute_horizon'])*50_000_000)
                            - context['observation_perf_ns'])/50_000_000
                    absolute_xyz = np.asarray(requests[rid]['base_tcp']['position_m']) + actions[:, :3]
                    new_xyz = np.column_stack([np.interp(rows, np.arange(20), absolute_xyz[:, k]) for k in range(3)])
                    reference = np.asarray(context['absolute_prefix'])
                    if rows[-1] <= len(reference)-1:
                        old_xyz = np.column_stack([np.interp(rows, np.arange(len(reference)), reference[:, k]) for k in range(3)])
                        execution_metrics = dict(first_row=float(rows[0]), last_row=float(rows[-1]),
                            same_time_seam_error_mm=float(np.linalg.norm(new_xyz[0]-old_xyz[0])*1000),
                            executed_prefix_max_error_mm=float(np.max(np.linalg.norm(new_xyz-old_xyz,axis=1))*1000),
                            executed_inside_max_step_mm=float(np.max(np.linalg.norm(np.diff(new_xyz,axis=0),axis=1))*1000))
                        if previous:
                            execution_metrics['recorded_old_to_new_seam_step_mm'] = float(np.linalg.norm(new_xyz[0]-np.asarray(previous['target_tcp']['position_m']))*1000)
                report['rows'].append(dict(request_id=rid, seed=seed, elapsed_ms=(time.perf_counter()-started)*1000,
                    diagnostics=rtc, actions=actions.tolist(), execution_metrics=execution_metrics,
                    server_timing=reply.get('policy_timing')))
                print(json.dumps(dict(request_id=rid, seed=seed, first6_mm=[round(r['position_error_m']*1000, 3) for r in rtc['prefix_residuals'][:6]])), flush=True)
        normal = client.infer(inputs[-1][1])
        report['normal_shape'] = list(np.asarray(normal['actions']).shape)
    finally:
        client.close()
        (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    repeat_errors = []
    for rid, _, _ in inputs:
        repeats = [r for r in report['rows'] if r['request_id'] == rid and r['seed'] == 0]
        repeat_errors.append(float(np.max(np.abs(np.asarray(repeats[0]['actions'])-np.asarray(repeats[1]['actions'])))))
    report['same_seed_max_abs_action_difference'] = max(repeat_errors)
    unique_rows = []
    seen = set()
    for row in report['rows']:
        key = (row['request_id'], row['seed'])
        if key not in seen:
            unique_rows.append(row)
            seen.add(key)
    errors = np.array([[r['position_error_m']*1000 for r in row['diagnostics']['prefix_residuals'][:6]] for row in unique_rows])
    report['first6_position_mm'] = dict(median=float(np.median(errors)), maximum=float(errors.max()), all_within_1mm=bool((errors<=1).all()))
    (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(inputs), figsize=(5*len(inputs), 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    for axis, (rid, _, _) in zip(axes, inputs):
        for row in [r for r in report['rows'] if r['request_id'] == rid][:3]:
            residuals = row['diagnostics']['prefix_residuals']
            axis.plot([r['row'] for r in residuals], [r['position_error_m']*1000 for r in residuals], 'o-', label='seed '+str(row['seed']))
        axis.axvspan(-.5, 5.5, alpha=.10, color='green', label='frozen rows 0..5')
        axis.axhline(1., color='red', linestyle='--', label='1 mm threshold')
        axis.set(title=f'{args.episode.name} / request {rid}', xlabel='New chunk row'); axis.grid(alpha=.25)
    axes[0].set_ylabel('Same-time new vs reference position error (mm)'); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output/'prefix_residuals.png', dpi=160); plt.close(fig)
    print(json.dumps(report['first6_position_mm']))


if __name__ == '__main__':
    main()
