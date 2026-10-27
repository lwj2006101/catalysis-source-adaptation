"""Test outcome-blind partitions on every actual target source and outer split.

The main audit separately reconstructs all support orders. This extra check
tests the legacy partitioner using the real released frames, with poisoned
responses, instead of relying only on a synthetic fixture.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import run_history as experiment
from server_guard import write_json, utc


def main():
    root = ROOT / 'runs/history_v1'
    comparisons = sources = eligible = 0
    for path in sorted((root / 'parts').glob('*/evaluated_input.csv.gz')):
        data = pd.read_csv(path)
        for source, sub in data.groupby('canonical_source_id', sort=True):
            original = experiment.b.designs(sub)
            eligible += bool(original)
            poisoned = sub.copy()
            poisoned['C2_yield_percent'] = np.arange(len(sub)) * 1e12 - 1e15
            changed = experiment.b.designs(poisoned)
            assert set(original) == set(changed)
            for scenario in original:
                for before, after in zip(original[scenario][:2], changed[scenario][:2]):
                    np.testing.assert_array_equal(before, after)
                    comparisons += 1
            sources += 1
    assert sources > 0
    write_json(root / 'verification/target_outcome_poison.json', {
        'passed': True, 'checked_utc': utc(), 'actual_source_outer_split_frames': sources,
        'eligible_source_outer_split_frames': eligible,
        'partition_index_arrays_compared': comparisons,
        'outcome_poison': 'replace every actual target response by extreme artificial values',
        'acquisition_signature': 'selection_orders accepts only historical-frozen geometry, row IDs, partitions, frozen gate and seed; no target responses',
        'limits': 'This tests implementation isolation; it does not establish unbiased source identities or independent new laboratories.'})
    print(json.dumps({'passed': True, 'source_outer_split_frames': sources,
                      'eligible_source_outer_split_frames': eligible,
                      'partition_arrays_compared': comparisons}))


if __name__ == '__main__':
    main()
