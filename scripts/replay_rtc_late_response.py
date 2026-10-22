"""Reconstruct a recorded RTC queue and evaluate the last reply without motion."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.inference_rtc import Timeline, Target


def decode(rows):
    return [Target(tuple(row[:3]), tuple(row[3:7]), row[7]) for row in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    events = [json.loads(line) for line in (args.run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    pending = None; replay_errors = []; timeline = None; trials = []
    metadata = dict(prefix_attention_schedule='ones', rtc_requires_execution_within_prefix=True)
    for event in events:
        kind = event['event']
        if kind == 'rtc_bootstrap_alignment':
            context = next(e['context'] for e in events if e['event'] == 'rtc_request')
            timeline = Timeline(context['run_epoch'], context['timeline_origin_perf_ns'])
            targets = decode(event['scheduled_targets'])
            timeline.seed(targets, event['response_ready_perf_ns'], targets[0])
            timeline.revision = next(e['context']['plan_revision'] for e in events
                                     if e['event']=='rtc_request' and e['context']['phase']=='replan')
        elif kind == 'rtc_response' and event['context']['phase'] == 'replan':
            pending = event
            try:
                candidate, evidence = timeline.merge_response(event['context'], decode(event['absolute_targets']), event['perf_ns'], metadata)
                trials.append(dict(request_id=event['context']['request_id'], accepted=True, **evidence))
            except RuntimeError as exc:
                trials.append(dict(request_id=event['context']['request_id'], accepted=False, reason=str(exc)))
        elif kind == 'rtc_splice':
            timeline.targets = timeline.merged(pending['context'], decode(pending['absolute_targets']), now_ns=event['perf_ns'])
            timeline.revision += 1
        elif kind == 'rtc_dispatch':
            tick = event['tick']
            replay_errors.append(float(np.linalg.norm(np.subtract(timeline.targets[tick].position, event['target_tcp']['position_m']))*1000))
            timeline.next_tick = tick
            timeline.complete(tick)
    report = dict(run=str(args.run), hardware_commands=0, responses=len(trials),
                  replay_max_error_mm=max(replay_errors, default=0),
                  last_response=trials[-1], fault=[e for e in events if e['event']=='rtc_fault'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
