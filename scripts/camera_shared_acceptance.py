"""Read-only dual-camera/JPEG load acceptance; never imports robot controls.

Run with ``python -m scripts.camera_shared_acceptance --seconds 120 --output DIR``.
Temporarily releases the console cameras only when control/recording are idle.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np

from supervisor.camera_resource import SharedCameraResource
from supervisor.tcp_vla_dataset_recorder import TcpVlaDatasetRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error('seconds must be positive')

    def api(path, post=False):
        with urlopen(Request(args.url + path, data=b'{}' if post else None,
                             headers={'Content-Type': 'application/json'}), timeout=20) as response:
            body = json.load(response)
            if not body.get('ok'):
                raise RuntimeError(body)
            return body['data']

    pi = api('/api/pi05/state')
    dataset = api('/api/dataset/state')
    session = api('/api/osc/state').get('session') or {}
    if pi.get('execution_enabled') or pi.get('priming') or dataset.get('recording') or session.get('state') == 'ACTIVE':
        raise RuntimeError('Acceptance requires idle control and dataset recorder')
    previous = api('/api/cameras/state')
    args.output.mkdir(parents=True, exist_ok=False)
    resource = SharedCameraResource(previous['config'])
    stop = threading.Event()
    report = {'duration_s': args.seconds, 'clock': 'host perf_counter_ns after read; not sensor exposure',
              'sources': {}, 'preview_reads': 0, 'errors': []}
    encoder = SimpleNamespace(raw_jpeg_quality=95)
    threads = []
    try:
        api('/api/cameras/deactivate', True)
        opened = time.perf_counter()
        resource.activate()
        resource.read()
        report['warmup_s'] = time.perf_counter() - opened
        # Allow driver/exposure transitions to settle before measuring.
        time.sleep(2)
        report['capture'] = resource.snapshot()
        started = time.perf_counter_ns()
        initial = dict(resource.dataset_source_sequence)

        def consume(source):
            stamps, rows, luma, clipping = [], [], [], []
            last = initial[source]
            gaps = writes = failures = 0
            (args.output / 'raw' / 'images' / source).mkdir(parents=True)
            while not stop.is_set():
                for row in resource.dataset_frames_after(source, last, .05, 8):
                    gaps += max(0, row['sequence'] - last - 1)
                    last = row['sequence']
                    frame = row['frame']
                    try:
                        if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                            raise RuntimeError(f'invalid raw RGB {frame.shape} {frame.dtype}')
                        TcpVlaDatasetRecorder._encode_raw_frame(encoder, args.output, source, writes, frame)
                        writes += 1
                    except Exception as exc:
                        failures += 1
                        report['errors'].append(str(exc))
                    stamps.append(row['timestamp'])
                    luma.append(float(frame.mean()))
                    clipping.append(float(np.all(frame >= 250, axis=2).mean()))
                    rows.append({'sequence': last, 'timestamp_ns': row['timestamp']})
            intervals = np.diff(stamps) / 1e6
            result = {'frames': len(stamps), 'writes': writes, 'write_failures': failures,
                      'producer_sequence_gaps': gaps,
                      'actual_hz': (len(stamps)-1)*1e9/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0,
                      'interval_ms': {str(p): float(np.percentile(intervals, p)) for p in (50,95,99,100)} if len(intervals) else {},
                      'intervals_over_50ms': int((intervals > 50).sum()),
                      'mean_luma': float(np.mean(luma)) if luma else None,
                      'white_clipping_fraction': float(np.mean(clipping)) if clipping else None,
                      'over_50ms': [{'previous_ns': stamps[i], 'timestamp_ns': stamps[i+1], 'interval_ms': float(v)}
                                    for i,v in enumerate(intervals) if v > 50]}
            report['sources'][source] = result
            (args.output / f'{source}-timestamps.json').write_text(json.dumps(rows), encoding='utf-8')

        for source in ('external', 'wrist'):
            thread = threading.Thread(target=consume, args=(source,))
            thread.start(); threads.append(thread)
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            frames = resource.read()
            if any(frame.shape != (224,224,3) or frame.dtype != np.uint8 for frame in frames):
                report['errors'].append('invalid model RGB')
            report['preview_reads'] += 1
            stop.wait(.05)
        report['elapsed_s'] = (time.perf_counter_ns()-started)/1e9
        stop.set()
        for thread in threads: thread.join()
        report['final_capture'] = resource.snapshot()
        report['capture_settings_ok'] = all(s['capture'].get('configuration_ok')
                                           for s in report['final_capture']['sources'].values())
        report['passed'] = all(s['actual_hz'] >= 20 and not s['intervals_over_50ms']
                               and not s['producer_sequence_gaps'] and not s['write_failures']
                               for s in report['sources'].values()) and len(report['sources']) == 2 and not report['errors'] and report['capture_settings_ok']
        print(json.dumps({k:v for k,v in report.items() if k not in ('capture','final_capture')}, indent=2), flush=True)
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=10)
        resource.close()
        if previous['ready']:
            api('/api/cameras/activate', True)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
