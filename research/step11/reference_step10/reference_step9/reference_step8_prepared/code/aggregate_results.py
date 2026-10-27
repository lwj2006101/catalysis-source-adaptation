"""Source-equal conditional inference; no significance from support repeats."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from server_guard import ROOT, require_server, write_json, utc
import run_history as experiment


PRIMARY_CONTROLS = ('composition_vopt', 'operating_vopt', 'kernel_vopt',
                    'robust_view_vopt', 'fixed_best_view')


def clustered_interval(frame, confidence, seed=806271, draws=20000):
    groups = sorted(frame.leakage_group.astype(str).unique())
    if len(groups) < 2:
        return None, None
    sums = np.array([frame.loc[frame.leakage_group.eq(g), 'delta'].sum() for g in groups])
    counts = np.array([frame.leakage_group.eq(g).sum() for g in groups])
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, draws, 500):
        take = rng.integers(0, len(groups), (min(500, draws - start), len(groups)))
        boot.extend(sums[take].sum(1) / counts[take].sum(1))
    tail = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(boot, [tail, 1 - tail]))


def aggregate(root, data_scope='historical_development'):
    paths = sorted((root / 'parts').glob('*/cells.csv.gz'))
    if not paths:
        raise ValueError('No actual results; no tables will be synthesized')
    all_cells = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    metrics = [c for c in all_cells if c.startswith(('mae__', 'mse__'))]
    keys = ['task', 'source', 'leakage_group', 'scenario', 'policy', 'k']
    per_seed = all_cells.groupby(keys + ['seed'], as_index=False)[metrics].mean()
    source = per_seed.groupby(keys, as_index=False)[metrics].mean()
    summary = source.groupby(['task', 'scenario', 'policy', 'k'], as_index=False)[metrics].mean()
    count = source.groupby(['task', 'scenario', 'policy', 'k'], as_index=False).agg(
        n_sources=('source', 'nunique'), n_leakage_groups=('leakage_group', 'nunique'))
    summary = summary.merge(count, validate='one_to_one')
    primary = []
    sample = source[(source.task == 'ocm_f0') & (source.scenario == 'random') & (source.k == 5)]
    candidate = sample[sample.policy == 'view_gate']
    for control in PRIMARY_CONTROLS:
        reference = sample[sample.policy == control]
        pair = candidate.merge(reference, on=['task','source','leakage_group','scenario','k'],
                               suffixes=('_candidate','_reference'), validate='one_to_one')
        pair['delta'] = pair.mae__adaptive_blend_candidate - pair.mae__adaptive_blend_reference
        if len(pair) != len(candidate) or not len(pair):
            raise ValueError('Primary comparison lost a source')
        low, high = clustered_interval(pair, 0.99)
        primary.append(dict(control=control, task='ocm_f0', scenario='random', k=5,
            candidate_mae=float(pair.mae__adaptive_blend_candidate.mean()),
            control_mae=float(pair.mae__adaptive_blend_reference.mean()),
            delta_mae=float(pair.delta.mean()), family_adjusted_ci_low=low,
            family_adjusted_ci_high=high, n_sources=len(pair),
            n_leakage_groups=pair.leakage_group.nunique(),
            adjusted_support_for_advantage=bool(high is not None and high < 0)))
    output = root / 'aggregated'
    output.mkdir(exist_ok=True)
    for frame, name in [(source,'per_source.csv'),(per_seed,'per_source_per_seed.csv.gz'),
                        (summary,'summary.csv'),(pd.DataFrame(primary),'primary_family.csv')]:
        experiment.save_csv(frame, output / name)
    write_json(output/'aggregation.json', dict(created_utc=utc(), cells=len(all_cells),
        units='percentage points of C2 yield', primary_family_size=5,
        uncertainty='conditional on historical fitted components; sources equally weighted; repeats and seeds averaged first',
        data_scope=data_scope, new_dataset_confirmation=data_scope!='historical_development',
        prospective_experiments=False, pipeline_refit_bootstrap=False))
    return pd.DataFrame(primary)


def main():
    require_server()
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default='history_v1')
    args = p.parse_args()
    print(aggregate(require_server(ROOT/'runs'/args.run_name)).to_string(index=False))


if __name__ == '__main__':
    main()
