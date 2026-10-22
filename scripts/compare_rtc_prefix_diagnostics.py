"""Compare saved, matched-noise WebSocket responses without model/robot calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def unique(report):
    result = {}
    for row in report['rows']:
        result.setdefault((row['request_id'], row['seed']), row)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    args = ap.parse_args()
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    for name in ('checkpoint_dir', 'norm_stats_sha256', 'openpi_source_sha256'):
        if before['metadata'].get(name) != after['metadata'].get(name):
            raise ValueError('Model/assets/OpenPI changed across comparison: '+name)
    b, a = unique(before), unique(after)
    old_inputs = np.load(args.before.parent/'observations.npz')
    new_inputs = np.load(args.after.parent/'observations.npz')
    for key in old_inputs.files:
        np.testing.assert_array_equal(old_inputs[key], new_inputs[key])
    if b.keys() != a.keys():
        raise ValueError('Comparison must use the same observations and seeds')
    for key in b:
        for field in ('reference_sha256',):
            assert b[key]['diagnostics'][field] == a[key]['diagnostics'][field]
        assert b[key]['diagnostics']['sampler_config']['noise_sha256'] == a[key]['diagnostics']['sampler_config']['noise_sha256']
        if not b[key].get('execution_metrics') and a[key].get('execution_metrics'):
            rid = key[0]; metric = a[key]['execution_metrics']
            context = json.loads(str(new_inputs[f'{rid}_context']))
            rows = np.linspace(metric['first_row'], metric['last_row'], context['execute_horizon'])
            absolute = new_inputs[f'{rid}_state'][:3] + np.asarray(b[key]['actions'])[:, :3]
            ref = np.asarray(context['absolute_prefix'])
            sampled = np.column_stack([np.interp(rows, np.arange(20), absolute[:, k]) for k in range(3)])
            desired = np.column_stack([np.interp(rows, np.arange(len(ref)), ref[:, k]) for k in range(3)])
            b[key]['execution_metrics'] = dict(same_time_seam_error_mm=float(np.linalg.norm(sampled[0]-desired[0])*1000),
                executed_prefix_max_error_mm=float(np.max(np.linalg.norm(sampled-desired,axis=1))*1000),
                executed_inside_max_step_mm=float(np.max(np.linalg.norm(np.diff(sampled,axis=0),axis=1))*1000))
    ids = sorted(set(key[0] for key in a))
    fig, axes = plt.subplots(1, len(ids), figsize=(5*len(ids), 4.5), sharey=True, squeeze=False)
    for axis, rid in zip(axes[0], ids):
        for mapping, label, color in ((b, 'Before: exp / cap5 / uniform', 'tab:red'),
                                       (a, 'After: ones / cap1000 / refined', 'tab:blue')):
            errors = np.array([[r['position_error_m']*1000 for r in row['diagnostics']['prefix_residuals']]
                               for (request, _), row in mapping.items() if request == rid])
            axis.plot(np.arange(errors.shape[1]), np.median(errors, axis=0), 'o-', color=color, label=label)
            axis.fill_between(np.arange(errors.shape[1]), np.min(errors, axis=0), np.max(errors, axis=0), alpha=.12, color=color)
        axis.axhline(1., color='black', linestyle='--', label='1 mm threshold')
        axis.axvspan(-.5, 5.5, color='green', alpha=.06)
        axis.set(yscale='log', xlabel='New chunk row', title=f'episode 000004 / request {rid}')
        axis.grid(alpha=.2)
    axes[0, 0].set_ylabel('Same-time prefix error (mm, log scale)')
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.after.parent/'before_after.png', dpi=170); plt.close(fig)
    report = {}
    for label, mapping in (('before', b), ('after', a)):
        errors = np.array([[r['position_error_m']*1000 for r in row['diagnostics']['prefix_residuals'][:6]] for row in mapping.values()])
        report[label] = dict(first6_position_median_mm=float(np.median(errors)), first6_position_max_mm=float(np.max(errors)),
                             sampler_config=next(iter(mapping.values()))['diagnostics']['sampler_config'])
        metrics = [row['execution_metrics'] for row in mapping.values() if row.get('execution_metrics')]
        if metrics:
            report[label]['execution'] = {name: dict(median=float(np.median([m[name] for m in metrics])), maximum=float(max(m[name] for m in metrics)))
                                           for name in ('same_time_seam_error_mm', 'executed_prefix_max_error_mm', 'executed_inside_max_step_mm', 'recorded_old_to_new_seam_step_mm')
                                           if all(name in m for m in metrics)}
    (args.after.parent/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
