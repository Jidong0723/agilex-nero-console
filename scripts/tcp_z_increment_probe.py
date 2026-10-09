"""Supervised, one-shot base-Z probe. Default is read-only preflight.

Run with the isolated nero-kinematics Python. No SDK writes except through
the existing OSC HTTP interface; no enable, mode change, or fault reset.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation
from motion.osc_kinematics_server import Solver


def timeline():
    return [(cycle * .7 + step * .05,
             (cycle * 5 + step + 1) * .005 if cycle < 5
             else (25 - (cycle - 5) * 5 - step - 1) * .005)
            for cycle in range(10) for step in range(5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--onsite-safe', action='store_true')
    parser.add_argument('--expected-cv', type=float)
    parser.add_argument('--http', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    if args.execute and not args.onsite_safe:
        parser.error('execution requires explicit onsite safety confirmation')
    folder = ROOT / 'runtime' / 'diagnostics' / datetime.now().strftime('tcp-z-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'config/osc.json').read_text(encoding='utf-8'))
    solver = Solver(ROOT / config['solver']['urdf'], config['tcp']['offset_from_link7_m'])
    client = 'tcp-z-probe-' + folder.name
    session_id = None
    stopped = threading.Event()
    failed = threading.Event()
    errors, commands, samples, states = [], [], [], []
    threads = []
    result = {'hardware_requested': args.execute, 'output_mode': 'cpv',
              'clock': 'Windows QPC / time.perf_counter_ns',
              'schedule_s': timeline(), 'completed': False}

    def request(path, body=None, timeout=2):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.http + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            envelope = json.load(response)
        if not envelope.get('ok'):
            raise RuntimeError(str(envelope))
        value = envelope['data']
        if value.get('ok') is False:
            raise RuntimeError(str(value.get('result', value)))
        return value

    def fault(message):
        errors.append(str(message))
        failed.set()

    def check_state(state, initial=False):
        authority = state.get('authority') or {}
        session = state.get('session') or {}
        if state.get('output_mode') != 'cpv':
            raise RuntimeError('output mode changed / is not CPV')
        if authority.get('safety_state') != 'NORMAL' or not authority.get('feedback_fresh'):
            raise RuntimeError('unsafe or stale hardware feedback: ' + str(authority))
        if initial:
            if session.get('id') or authority.get('hardware_mode') != 'HOLD':
                raise RuntimeError('preflight requires confirmed HOLD and no session')
        elif session.get('id') != session_id or session.get('client_id') != client:
            raise RuntimeError('control session lost; no further targets will be issued')

    def sensor():
        sample = request('/api/dataset/feedback', timeout=.4)
        now = time.perf_counter_ns()
        age = (now - int(sample['feedback_monotonic_ns'])) / 1e9
        if not 0 <= age <= .15 or sample.get('latest_feedback_age_s', 1) > .15:
            raise RuntimeError('CAN feedback stale: ' + str(age))
        sample = {key: sample.get(key) for key in (
            'feedback_monotonic_ns', 'feedback_revision', 'joint_position_rad',
            'joint_velocity_rad_s', 'target_tcp_pose', 'control_sample_id',
            'target_generation', 'motion_epoch', 'latest_feedback_age_s')}
        sample['poll_perf_ns'] = now
        return sample

    def recorder():
        deadline = time.perf_counter()
        while not stopped.is_set():
            try:
                samples.append(sensor())
            except Exception as exc:
                fault('feedback recorder: ' + str(exc))
                return
            deadline += .02
            stopped.wait(max(0, deadline - time.perf_counter()))
            if time.perf_counter() > deadline + .02:
                deadline = time.perf_counter()

    def watchdog():
        while not stopped.is_set():
            try:
                state = request('/api/osc/session/heartbeat',
                                {'client_id': client, 'session_id': session_id}, timeout=.8)['state']
                states.append({'perf_ns': time.perf_counter_ns(), 'state': state})
                check_state(state)
            except Exception as exc:
                fault('session watchdog: ' + str(exc))
                return
            stopped.wait(.2)

    def wait_until(deadline):
        while True:
            if failed.is_set():
                raise RuntimeError(errors[-1])
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            failed.wait(min(remaining, .01))

    try:
        before = request('/api/osc/state')
        check_state(before, initial=True)
        if args.execute:
            profile = request('/api/osc/cpv-parameters/read-only')['cpv_parameters']
            result['hardware_cpv_parameters'] = profile
            if profile['status'] != 'available':
                raise RuntimeError('complete CPV read-back required')
            if args.expected_cv is not None and any(abs(v - args.expected_cv) > 1e-6 for v in profile['values']['cv']):
                raise RuntimeError('hardware CV differs from the requested test setting')
            expected_acc = config['solver']['joint_acceleration_limit_rad_s2']
            if any(abs(v - expected_acc) > 1e-6 for name in ('acc', 'dcc') for v in profile['values'][name]):
                raise RuntimeError('hardware acceleration differs from the requested test setting')
        first = sensor()
        q0 = np.array(first['joint_position_rad'])
        if max(abs(v) for v in first['joint_velocity_rad_s']) > .005:
            raise RuntimeError('robot is not stationary')
        tcp = solver.fk(q0.tolist())
        anchor = np.array(tcp['position_m'])
        quaternion = Rotation.from_matrix(tcp['rotation']).as_quat().tolist()
        result['initial_tcp_m'] = anchor.tolist()
        result['initial_joints_rad'] = q0.tolist()
        limits, settings = config['limits'], config['solver']
        margin = np.array(config['safety_supervisor']['fixed_model_margin_rad'])
        lower = np.maximum(solver.hard_lower, config['hardware_limits']['lower_rad']) + margin
        upper = np.minimum(solver.hard_upper, config['hardware_limits']['upper_rad']) - margin
        for _, dz in timeline():
            target = anchor + [0, 0, dz]
            if np.any(target < limits['workspace_min_m']) or np.any(target > limits['workspace_max_m']):
                raise RuntimeError('trajectory outside configured workspace')
        # Read-only quasi-static reachability sweep, not a hardware simulation.
        q, velocity = q0.copy(), np.zeros(7)
        for dz in (.025, .05, .075, .1, .125):
            target = anchor + [0, 0, dz]
            for iteration in range(250):
                payload = dict(settings)
                payload.update(joint_angles_rad=q.tolist(), target_position_m=target.tolist(),
                    target_orientation_xyzw=quaternion, last_sent_joint_velocity_rad_s=velocity.tolist(),
                    joint_speed_limit_rad_s=[limits['joint_speed_rad_s']] * 7,
                    joint_acceleration_limit_rad_s2=[settings['joint_acceleration_limit_rad_s2']] * 7,
                    soft_lower_rad=lower.tolist(), soft_upper_rad=upper.tolist(),
                    posture_reference_rad=q0.tolist(), condition_limit=limits['singularity_condition_max'], dt_s=.02)
                try:
                    solved = solver.solve(payload)
                except Exception as exc:
                    raise RuntimeError(f'offline IK at dz={dz}, iteration={iteration}, '
                                       f'q={q.tolist()}, dq={velocity.tolist()}: {exc}') from exc
                if not solved.get('ok'):
                    raise RuntimeError('offline reachability: ' + str(solved))
                velocity = np.array(solved['pink_joint_velocity_rad_s'])
                q += velocity * .02
                pose = solver.fk(q.tolist())
                error = np.linalg.norm(np.array(pose['position_m']) - target)
                if error < .0005:
                    break
            if error > .001:
                raise RuntimeError(f'offline endpoint not reachable: dz={dz}, error={error}')
        result['offline_reachability_passed'] = True
        print(json.dumps({'preflight': 'passed', 'anchor_m': anchor.tolist(),
                          'peak_z_m': float(anchor[2] + .125), 'execute': args.execute}), flush=True)
        if not args.execute:
            return 0
        # Recheck immediately before claiming a session; no implicit takeover.
        check_state(request('/api/osc/state'), initial=True)
        fresh = sensor()
        if np.max(np.abs(np.array(fresh['joint_position_rad']) - q0)) > .002:
            raise RuntimeError('robot moved during preflight; refuse stale anchoring')
        started = request('/api/osc/session/start', {'client_id': client,
                          'execution_mode': 'hardware', 'input_source': 'tcp_z_increment_test'})
        session_id = started['session']['id']
        check_state(request('/api/osc/state'))
        for worker in (recorder, watchdog):
            thread = threading.Thread(target=worker, daemon=True)
            threads.append(thread)
            thread.start()
        epoch = time.perf_counter() + .5
        result['t0_perf_ns'] = round(epoch * 1e9)
        for index, (scheduled, dz) in enumerate(timeline(), 1):
            wait_until(epoch + scheduled)
            sent = time.perf_counter_ns()
            lag = sent / 1e9 - epoch - scheduled
            if lag > .025:
                raise RuntimeError(f'command scheduling late {lag:.3f}s; no catch-up burst')
            position = (anchor + [0, 0, dz]).tolist()
            body = {'client_id': client, 'session_id': session_id, 'sequence': index,
                    'type': 'track_tcp', 'acknowledgement_only': True,
                    'payload': {'target_pose': {'position_m': position, 'orientation_xyzw': quaternion}}}
            row = {'sequence': index, 'cycle': (index - 1) // 5 + 1,
                   'scheduled_s': scheduled, 'send_perf_ns': sent,
                   'target_z_m': position[2], 'delta_z_m': .005 if index <= 25 else -.005}
            commands.append(row)
            reply = request('/api/osc/command', body, timeout=.4)
            row.update(ack_perf_ns=time.perf_counter_ns(), reply=reply)
            if not reply.get('result', {}).get('accepted'):
                raise RuntimeError('target not accepted: ' + str(reply))
            if index % 5 == 0:
                print(f'cycle {row["cycle"]}/10 accepted; target Z={position[2]*1000:.2f} mm', flush=True)
        wait_until(epoch + 7.0)  # Includes the final 500-ms no-input pause.
        wait_until(epoch + 8.0)  # One additional second of observation, no new target.
        result['completed'] = True
    except Exception as exc:
        fault(str(exc))
        print('ABORT: ' + str(exc), flush=True)
    finally:
        stopped.set()
        # End only this probe's session. Never clear a fault or restore targets.
        if args.execute:
            try:
                current = request('/api/osc/state')
                states.append({'perf_ns': time.perf_counter_ns(), 'state': current})
                if (current.get('session') or {}).get('client_id') == client:
                    held = request('/api/osc/session/stop', {'reason': 'TCP Z probe complete/abort'})
                    result['stop_result'] = held
                    time.sleep(.3)
                    final = request('/api/osc/state')
                    result['final_authority'] = final.get('authority')
                    final_sensor = sensor()
                    result['final_sensor'] = final_sensor
                    authority = final.get('authority') or {}
                    result['hold_confirmed'] = (authority.get('hardware_mode') == 'HOLD'
                        and authority.get('feedback_fresh')
                        and max(abs(v) for v in final_sensor['joint_velocity_rad_s']) <= .005)
                    if not result['hold_confirmed']:
                        fault('stationary fresh-feedback HOLD not confirmed')
                elif session_id:
                    fault('session no longer owned; global stop not issued')
            except Exception as exc:
                fault('HOLD failed: ' + str(exc))
        for thread in threads:
            thread.join(timeout=2)
        result.update(errors=errors, commands_accepted=sum(
            bool(x.get('reply', {}).get('result', {}).get('accepted')) for x in commands),
                      sample_count=len(samples))
        # Preserve raw records before doing any offline model work.
        for name, rows in (('commands', commands), ('feedback-raw', samples), ('states', states)):
            with (folder / (name + '.jsonl')).open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        (folder / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        # FK is deferred until after HOLD, so it cannot disturb live timing.
        for sample in samples:
            sample['measured_tcp_m'] = solver.fk(sample['joint_position_rad'])['position_m']
        for name, rows in (('commands', commands), ('feedback', samples), ('states', states)):
            with (folder / (name + '.jsonl')).open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (folder / 'feedback.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['t_s', 'feedback_perf_ns', 'measured_x_m', 'measured_y_m', 'measured_z_m', 'feedback_revision'])
            for row in samples:
                writer.writerow([(row['feedback_monotonic_ns'] - result['t0_perf_ns']) / 1e9,
                                 row['feedback_monotonic_ns'], *row['measured_tcp_m'], row['feedback_revision']])
        (folder / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Artifacts: ' + str(folder), flush=True)
    return 0 if result['completed'] and not errors else (0 if not args.execute and not errors else 1)


if __name__ == '__main__':
    raise SystemExit(main())
