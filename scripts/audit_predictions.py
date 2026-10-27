"""Independently recompute archived final-extension cell errors, without fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'research/step11'

def audit(part, examples):
    with np.load(part/'row_predictions.npz', allow_pickle=False) as arrays:
        prediction = arrays['predictions']
        cell_id = arrays['cell_id'].astype(int)
        query = arrays['query_index'].astype(int)
        methods = arrays['methods'].tolist()
    data = pd.read_csv(part/'evaluated_input.csv.gz')
    cells = pd.read_csv(part/'cells.csv.gz').set_index('cell_id').sort_index()
    assert prediction.shape == (len(query), len(methods))
    assert np.isfinite(prediction).all()
    assert len(query) == len(cell_id)
    assert query.min() >= 0 and query.max() < len(data)
    truth = data['C2_yield_percent'].to_numpy(float)[query]
    assert np.isfinite(truth).all()
    ids = cells.index.to_numpy(int)
    counts = np.bincount(cell_id, minlength=int(ids.max())+1)
    assert (counts[ids] == cells['n_test'].to_numpy(int)).all()
    worst_mae = worst_mse = 0.
    for j, method in enumerate(methods):
        error = prediction[:,j]-truth
        mae = np.bincount(cell_id, weights=np.abs(error), minlength=len(counts))[ids]/counts[ids]
        mse = np.bincount(cell_id, weights=error**2, minlength=len(counts))[ids]/counts[ids]
        stored_mae = cells['mae__'+method].to_numpy(float)
        stored_mse = cells['mse__'+method].to_numpy(float)
        np.testing.assert_allclose(mae, stored_mae, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(mse, stored_mse, rtol=1e-10, atol=1e-8)
        worst_mae = max(worst_mae, float(np.max(np.abs(mae-stored_mae))))
        worst_mse = max(worst_mse, float(np.max(np.abs(mse-stored_mse))))
    ledger = pd.read_csv(part/'support_ledger.csv.gz')
    for row in ledger.itertuples():
        s = json.loads(row.support_indices)
        q = json.loads(row.query_indices)
        pool = json.loads(row.pool_indices)
        assert len(s) == len(set(s))
        assert set(s).issubset(pool) and set(s).isdisjoint(q)
        assert set(pool).isdisjoint(q)
    if not examples:
        first = ids[0]
        for i in np.flatnonzero(cell_id == first)[:10]:
            for j, method in enumerate(methods):
                examples.append({'archive_part':part.relative_to(ROOT).as_posix(),'cell_id':int(first),
                    'row_id':str(data.iloc[query[i]]['row_id']), 'predictor':method,
                    'measured_C2_yield_percent':float(truth[i]),
                    'predicted_C2_yield_percent':float(prediction[i,j]),
                    'absolute_error_percentage_points':float(abs(prediction[i,j]-truth[i]))})
    return {'part':part.relative_to(ROOT).as_posix(),'cells':len(cells),
            'query_prediction_rows':len(query),'scalar_predictions':int(prediction.size),
            'max_mae_difference':worst_mae,'max_mse_difference':worst_mse,
            'support_query_disjoint':True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='work/prediction_audit.json')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    assert output.is_relative_to(ROOT/'work'), 'Write audit reports under work/'
    parts = sorted((ARCHIVE/'runs/operator_history_v1/parts').glob('ocm_*'))
    parts += [ARCHIVE/'runs/operator_opened_v1/parts/cohort']
    assert len(parts) == 26
    examples = []
    records = [audit(p, examples) for p in parts]
    total = sum(r['scalar_predictions'] for r in records)
    assert total == 10_948_000
    report = {'passed':True,'new_model_fits':0,'new_predictions_generated':0,
              'verified_scalar_predictions':total,'parts':records,
              'note':'Read-only reaggregation of stored predictions against recorded measurements; not a fresh model run.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(examples).to_csv(output.parent/'prediction_examples.csv',index=False)
    print(json.dumps({'passed':True,'verified_scalar_predictions':total,
                      'max_mae_difference':max(r['max_mae_difference'] for r in records),
                      'max_mse_difference':max(r['max_mse_difference'] for r in records),
                      'report':str(output.relative_to(ROOT)),
                      'examples':str((output.parent/'prediction_examples.csv').relative_to(ROOT))},indent=2))

if __name__ == '__main__':
    main()
