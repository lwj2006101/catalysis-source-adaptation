"""Re-aggregate every scalar prediction and reconstruct acquired supports."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from server_guard import ROOT, require_server, write_json, utc
import run_history as experiment
from view_selection import ViewGate


def audit(root):
    require_server(root)
    totals = dict(parts=0, prediction_values_reaggregated=0,
                  reconstructed_support_orders=0, reconstructed_prediction_rows=0)
    for part in sorted((root / 'parts').glob('*')):
        if not (part / 'completed.json').exists():
            raise ValueError(f'Incomplete part: {part.name}')
        info = json.loads((part / 'completed.json').read_text())
        assert set(info['history_training_groups']).isdisjoint(info['evaluation_groups'])
        parameters = json.loads((part / 'parameters.json').read_text())
        data = pd.read_csv(part / 'evaluated_input.csv.gz', keep_default_na=False)
        cells = pd.read_csv(part / 'cells.csv.gz')
        ledger = pd.read_csv(part / 'support_ledger.csv.gz')
        geometry = np.load(part / 'geometry.npz')
        bank = np.load(part / 'row_predictions.npz')
        pred, cell_ids, queries = bank['predictions'], bank['cell_id'], bank['query_index']
        assert tuple(bank['methods']) == experiment.METHODS
        counts = np.bincount(cell_ids, minlength=len(cells))
        assert np.all(counts > 0)
        error = pred - data.C2_yield_percent.to_numpy(float)[queries, None]
        absolute, square = np.zeros((len(cells), len(experiment.METHODS))), np.zeros((len(cells), len(experiment.METHODS)))
        np.add.at(absolute, cell_ids, np.abs(error))
        np.add.at(square, cell_ids, error * error)
        np.testing.assert_allclose(absolute / counts[:, None],
            cells[['mae__' + m for m in experiment.METHODS]], atol=1e-9, rtol=1e-11)
        np.testing.assert_allclose(square / counts[:, None],
            cells[['mse__' + m for m in experiment.METHODS]], atol=1e-8, rtol=1e-11)
        gate = ViewGate.from_dict(parameters['gate'])
        alphas = experiment.alpha_schedule(info['inherited_infos']['ocm_f1'])
        for (source, scenario), records in ledger.groupby(['source', 'scenario'], sort=True):
            first = records.iloc[0]
            query, pool = np.array(json.loads(first.query_indices)), np.array(json.loads(first.pool_indices))
            assert set(query).isdisjoint(pool)
            orders = experiment.selection_orders(geometry['z_f1'], data.row_id.to_numpy(str),
                query, pool, alphas, gate, parameters['gamma'],
                experiment.b.stable_seed('step8_target', source, scenario, info['seed']))
            for row in records.itertuples():
                original = np.array(json.loads(row.support_indices))
                found = [o for repeat, o, _ in orders[row.policy] if repeat == row.repeat]
                assert len(found) == 1
                np.testing.assert_array_equal(original, found[0])
                totals['reconstructed_support_orders'] += 1
        # Every deterministic-design cell and repeat-zero stochastic cell.
        for cell in cells[cells.repeat.eq(0)].itertuples():
            design = ledger.iloc[cell.design_id]
            support = np.array(json.loads(design.support_indices))[:cell.k]
            query = np.array(json.loads(design.query_indices))
            suffix = 'f0' if cell.task == 'ocm_f0' else 'f1'
            pp = experiment.predict(geometry['z_' + suffix], geometry['base_' + suffix],
                data.C2_yield_percent.to_numpy(float)[support], support, query,
                info['inherited_infos'][cell.task],
                parameters['controls'][cell.task][cell.policy][str(cell.k)], cell.k)
            stored = pred[cell_ids == cell.cell_id]
            np.testing.assert_allclose(np.column_stack(list(pp.values())), stored, atol=1e-8, rtol=1e-11)
            totals['reconstructed_prediction_rows'] += len(query)
        totals['parts'] += 1
        totals['prediction_values_reaggregated'] += int(pred.size)
    if not totals['parts']:
        raise ValueError('No completed experiment to audit')
    totals.update(passed=True, audited_utc=utc(),
                  independence='loss aggregation is independent; prediction reconstruction reuses prediction helpers; independent math checks are separate')
    write_json(root / 'verification/audit.json', totals)
    return totals


def main():
    require_server()
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', default='history_v1')
    args = p.parse_args()
    print(json.dumps(audit(require_server(ROOT / 'runs' / args.run_name)), indent=2))


if __name__ == '__main__':
    main()
