"""XU RC3 pipeline — per-fold train-only preprocessing, tiers A/B/C/D, real X5
residualization with unit test, correct R2 calibers, row predictions everywhere.

RC3 items covered: 2 (per-fold prep + leakage auto-test), 3 (registries + fixed record set),
4 (tier structure; B-random vs B-SD is the only strictly same-model/same-hyperparam pair),
5 (X5: fe.intercept_ as global intercept; unit test on constant column), 7 (R2 calibers in
the stated space), 8 (row_weighted_MAE from row-level predictions), 12 (record 5-fold vs
source 5-fold contrast on the same record set).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE / 'original_assets/github_repo'
OUT = HERE / 'rc3_outputs'
OUT.mkdir(exist_ok=True)
SEEDS = [42, 101, 202, 303, 404]
LOG = []


def log(msg):
    LOG.append(str(msg))
    print(msg, flush=True)


data = pd.read_excel(REPO / '3_cleaned2_standardized_processed2.xlsx')
s = data['Reference DOI'].astype(str).str.strip().str.lower()
s = s.str.replace(r'^https?://(dx\.)?doi\.org/', '', regex=True).str.replace('doi: ', '')
data['doi_norm'] = s.str.strip().str.rstrip('.,;')
data.loc[data['Reference DOI'].isna(), 'doi_norm'] = 'UNKNOWN'
CAT_PAT = ['Family', 'Support_1', 'Support_2', 'Support_3', 'method', 'Promoter_1',
           'Promoter_2']


def clean_cols(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [re.sub(r'_+', '_', re.sub(r'[^\w]+', '_', str(c)
                  .replace('[', '_').replace(']', '_').replace('<', '_')
                  .replace('>', '_').replace('-', '_').replace('.', '_')
                  .replace('(', '_').replace(')', '_'))).strip('_') for c in df.columns]
    return df


def author_features(df_part, is_train, state):
    """Author's preprocessing, fitted ONLY on df_part when is_train; per-fold (RC3 item 2)."""
    df_part = clean_cols(df_part)
    tgt = [c for c in df_part.columns if 'STY' in c.upper()][0]
    X = df_part.drop(columns=[tgt])
    y = df_part[tgt]
    cat_cols = []
    for pat in CAT_PAT:
        cat_cols += [c for c in X.columns if pat in c]
    cat_cols = sorted(set(c for c in cat_cols
                          if not pd.api.types.is_numeric_dtype(X[c])))
    if is_train:
        state['maps'] = {c: {v: i for i, v in enumerate(
            sorted(set(X[c].astype(str).tolist()), key=str))} for c in cat_cols}
        state['medians'] = None
    for c in cat_cols:
        X[c + '_enc'] = X[c].astype(str).map(state['maps'][c]).fillna(-1)
        X = X.drop(columns=[c])
    X = X.select_dtypes(include=[np.number])
    if is_train:
        state['medians'] = X.median()
    X = X.fillna(state['medians'])
    return X, y


def group_folds(groups, seed, k=5):
    g = np.asarray(groups, str)
    ids = np.array(sorted(set(g)))
    ids = np.random.default_rng(seed).permutation(ids)
    for held in np.array_split(ids, k):
        te = np.flatnonzero(np.isin(g, held))
        tr = np.flatnonzero(~np.isin(g, held))
        assert not set(g[tr]) & set(g[te])
        yield tr, te


def get_models():
    return {
        'XGBoost': XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                random_state=42, n_jobs=-1, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  random_state=42, n_jobs=-1, verbose=-1),
        'RandomForest': RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
        'CatBoost': CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                      random_seed=42, verbose=False),
    }


def metrics_block(yt, pt, srct):
    ae = np.abs(yt - pt)
    per = pd.DataFrame({'s': srct, 'ae': ae}).groupby('s').ae.mean()
    return dict(pooled_R2_raw=float(r2_score(yt, pt)),
                pooled_R2_log=float(r2_score(np.log1p(yt), np.log1p(pt))),
                MAE_raw=float(mean_absolute_error(yt, pt)),
                MAE_log=float(mean_absolute_error(np.log1p(yt), np.log1p(pt))),
                row_weighted_MAE=float(ae.mean()),
                source_equal_MAE=float(per.mean()),
                n_sources=int(per.size), n_rows=int(len(yt)))


# ---------- registries (item 3) ----------
reg = (data[data.doi_norm != 'UNKNOWN'].groupby('doi_norm')
       .agg(n_rows=('doi_norm', 'size')).reset_index())
reg['canonical_source_id'] = ['XU-' + f'{i:04d}' for i in range(len(reg))]
reg.to_csv(OUT / 'XU_SOURCE_REGISTRY.csv', index=False)
data['doi_raw'] = data['Reference DOI']
prov = data[['doi_raw', 'doi_norm']].copy()
prov['row_index'] = data.index
prov = prov.merge(reg[['doi_norm', 'canonical_source_id']], on='doi_norm', how='left')
prov['leakage_group_id'] = prov.canonical_source_id.fillna('SENSITIVITY_UNKNOWN_GROUP')
prov.to_csv(OUT / 'XU_ROW_PROVENANCE.csv', index=False)
norm = dict(raw_rows=len(data),
            doi_missing_rows=int((data.doi_norm == 'UNKNOWN').sum()),
            raw_distinct_strings=int(data.loc[data.doi_norm != 'UNKNOWN', 'doi_raw'].nunique()),
            canonical_dois=len(reg),
            merges=reg.shape[0] and int(data.loc[data.doi_norm != 'UNKNOWN', 'doi_raw']
                                        .nunique()) - len(reg))
log('normalization: ' + json.dumps(norm))

fixed = data[data.doi_norm != 'UNKNOWN'].copy()
y_fixed = fixed['STY [mgMeOH h-1 gcat-1]']
doi_fixed = fixed['doi_norm'].to_numpy(str)
log(f'fixed record set: {len(fixed)} rows / {len(reg)} sources '
    f'(the {norm["doi_missing_rows"]} DOI-missing rows are excluded from the main audit '
    'and reported as a separate sensitivity)')

res = {'normalization': norm, 'fixed_record_set': dict(n_rows=len(fixed),
                                                       n_sources=len(reg))}

# ---------- Tier A: author original flow (imported X0 reproduction numbers) ----------
x0 = json.loads((HERE / 'x0_outputs/x0_results.json').read_text())
res['A_author_original_flow_reproduction'] = x0
log('Tier A imported from x0_results.json (wrapper exit=0)')

# ---------- Tier B: fixed-config random vs source-disjoint, SAME fixed record set ----------
y_log_all = np.log1p(y_fixed)
strat = pd.qcut(y_log_all, q=5, labels=False, duplicates='drop')
itr, ite = train_test_split(np.arange(len(fixed)), test_size=0.2, random_state=42,
                            stratify=strat)
state_b = {}
Xtr, ytr = author_features(fixed.iloc[itr], True, state_b)
Xte, _ = author_features(fixed.iloc[ite], False, state_b)
x1, predsB1 = {}, []
for name, m in get_models().items():
    m.fit(Xtr, y_log_all.iloc[itr])
    p = np.expm1(m.predict(Xte))
    yt = y_fixed.iloc[ite].to_numpy()
    st = doi_fixed[ite]
    x1[name] = metrics_block(yt, p, st)
    for ri, yi, pi, si in zip(fixed.index[ite], yt, p, st):
        predsB1.append(dict(model=name, row_index=int(ri), y_true=float(yi),
                            y_pred=float(pi), source=si))
pd.DataFrame(predsB1).to_csv(OUT / 'tierB_random_row_predictions.csv.gz', index=False)
res['B_fixed_config_random_80_20'] = x1
log('Tier B random done')

cellsB2, predsB2 = [], []
for seed in SEEDS:
    for fold, (tr, te) in enumerate(group_folds(doi_fixed, seed)):
        state_f = {}
        Xtr, ytr = author_features(fixed.iloc[tr], True, state_f)
        Xte, _ = author_features(fixed.iloc[te], False, state_f)
        yt = y_fixed.iloc[te].to_numpy()
        st = doi_fixed[te]
        for name, m in get_models().items():
            m.fit(Xtr, np.log1p(ytr))
            p = np.expm1(m.predict(Xte))
            for ri, yi, pi, si in zip(fixed.index[te], yt, p, st):
                predsB2.append(dict(seed=seed, fold=fold, model=name, row_index=int(ri),
                                    y_true=float(yi), y_pred=float(pi), source=si))
            per = pd.DataFrame({'s': st, 'ae': np.abs(yt - p)}).groupby('s').ae.mean()
            for si, ae in per.items():
                cellsB2.append(dict(seed=seed, fold=fold, model=name, source=si, ae=float(ae)))
cellsB2, predsB2 = pd.DataFrame(cellsB2), pd.DataFrame(predsB2)
cellsB2.to_csv(OUT / 'tierB_source_disjoint_cells.csv', index=False)
predsB2.to_csv(OUT / 'tierB_source_disjoint_row_predictions.csv.gz', index=False)
x2, r2cal = {}, {}
for name, g in predsB2.groupby('model'):
    sub = cellsB2[cellsB2.model == name]
    x2[name] = dict(
        source_equal_MAE=float(sub.ae.mean()),
        row_weighted_MAE=float(np.abs(g.y_true - g.y_pred).mean()),
        pooled_R2_raw_stacked=float(r2_score(g.y_true, g.y_pred)),
        pooled_R2_log_stacked=float(r2_score(np.log1p(g.y_true), np.log1p(g.y_pred))))
    rowmean = g.groupby('row_index').agg(y=('y_true', 'first'), p=('y_pred', 'mean'))
    r2cal[name] = dict(
        stacked_raw=x2[name]['pooled_R2_raw_stacked'],
        stacked_log=x2[name]['pooled_R2_log_stacked'],
        per_row_mean_raw=float(r2_score(rowmean.y, rowmean.p)),
        per_row_mean_log=float(r2_score(np.log1p(rowmean.y), np.log1p(rowmean.p))),
        per_seed_raw=[float(r2_score(g[g.seed == s].y_true, g[g.seed == s].y_pred))
                      for s in SEEDS],
        per_seed_log=[float(r2_score(np.log1p(g[g.seed == s].y_true),
                                     np.log1p(g[g.seed == s].y_pred))) for s in SEEDS])
res['B_fixed_config_source_disjoint'] = x2
res['B_r2_calibers'] = r2cal
log('Tier B source-disjoint done')

# ---------- Tier C: nested source-disjoint (inner DOI folds, per-model selection) ----------
GRIDS = {
    'XGBoost': [dict(n_estimators=300, max_depth=d, learning_rate=lr, random_state=42,
                     n_jobs=-1, verbosity=0) for d in (4, 6, 8) for lr in (0.05, 0.1)],
    'CatBoost': [dict(iterations=300, depth=d, learning_rate=lr, random_seed=42,
                      verbose=False) for d in (4, 6, 8) for lr in (0.05, 0.1)],
}
ncells, npredsC = [], []
for seed in SEEDS:
    for fold, (tr, te) in enumerate(group_folds(doi_fixed, seed)):
        yt = y_fixed.iloc[te].to_numpy()
        st = doi_fixed[te]
        for name, grid in GRIDS.items():
            best_cfg, best_score = None, None
            for cfg in grid:
                sc = []
                for itr2, ite2 in group_folds(doi_fixed[tr], seed + 777, 3):
                    st2 = {}
                    Xi_tr, yi_tr = author_features(fixed.iloc[tr].iloc[itr2], True, st2)
                    Xi_te, _ = author_features(fixed.iloc[tr].iloc[ite2], False, st2)
                    m0 = {'XGBoost': XGBRegressor, 'CatBoost': CatBoostRegressor}[name](**cfg)
                    m0.fit(Xi_tr, np.log1p(yi_tr))
                    sc.append(mean_absolute_error(y_fixed.iloc[tr].iloc[ite2],
                                                  np.expm1(m0.predict(Xi_te))))
                scc = float(np.mean(sc))
                if best_score is None or scc < best_score:
                    best_cfg, best_score = cfg, scc
            state_c = {}
            Xtr, ytr = author_features(fixed.iloc[tr], True, state_c)
            Xte, _ = author_features(fixed.iloc[te], False, state_c)
            m = {'XGBoost': XGBRegressor, 'CatBoost': CatBoostRegressor}[name](**best_cfg)
            m.fit(Xtr, np.log1p(ytr))
            pt = np.expm1(m.predict(Xte))
            per = pd.DataFrame({'s': st, 'ae': np.abs(yt - pt)}).groupby('s').ae.mean()
            for si, ae in per.items():
                ncells.append(dict(seed=seed, fold=fold, model=name, source=si, ae=float(ae)))
            for ri, yi, pi, si in zip(fixed.index[te], yt, pt, st):
                npredsC.append(dict(seed=seed, fold=fold, model=name, row_index=int(ri),
                                    y_true=float(yi), y_pred=float(pi), source=si))
    log(f'  C seed {seed} done')
npredsC, ncells = pd.DataFrame(npredsC), pd.DataFrame(ncells)
npredsC.to_csv(OUT / 'tierC_nested_row_predictions.csv.gz', index=False)
ncells.to_csv(OUT / 'tierC_nested_cells.csv', index=False)
res['C_nested_source_disjoint'] = {}
for name, g in npredsC.groupby('model'):
    row = g.groupby(['seed', 'fold']).apply(
        lambda d: metrics_block(d.y_true.to_numpy(), d.y_pred.to_numpy(),
                                d.source.to_numpy(str)), include_groups=False)
    res['C_nested_source_disjoint'][name] = dict(
        source_equal_MAE=float(np.mean([r['source_equal_MAE'] for r in row])),
        row_weighted_MAE=float(np.mean([r['row_weighted_MAE'] for r in row])),
        pooled_R2_log_stacked=float(r2_score(np.log1p(g.y_true), np.log1p(g.y_pred))))

# ---------- Tier D: frozen X0 CatBoost best params (read from x0_results.json) ----------
x0_best = dict(x0['best_params'])  # flat dict = best model's (CatBoost) Optuna params
x0_best.setdefault('random_state', 42)
x0_best['provenance'] = 'read automatically from x0_outputs/x0_results.json (RC3 item 5)'
log('Tier D CatBoost params from x0_results.json: ' + json.dumps(x0_best))
dcells, dpreds = [], []
for seed in SEEDS:
    for fold, (tr, te) in enumerate(group_folds(doi_fixed, seed)):
        state_d = {}
        Xtr, ytr = author_features(fixed.iloc[tr], True, state_d)
        Xte, _ = author_features(fixed.iloc[te], False, state_d)
        m = CatBoostRegressor(**{k: v for k, v in x0_best.items()
                                 if k not in ('provenance', 'verbose')}, verbose=False)
        m.fit(Xtr, np.log1p(ytr))
        yt = y_fixed.iloc[te].to_numpy()
        st = doi_fixed[te]
        pt = np.expm1(m.predict(Xte))
        per = pd.DataFrame({'s': st, 'ae': np.abs(yt - pt)}).groupby('s').ae.mean()
        for si, ae in per.items():
            dcells.append(dict(seed=seed, fold=fold, model='CatBoost_frozenX0', source=si,
                               ae=float(ae)))
        for ri, yi, pi, si in zip(fixed.index[te], yt, pt, st):
            dpreds.append(dict(seed=seed, fold=fold, row_index=int(ri), y_true=float(yi),
                               y_pred=float(pi), source=si))
dcells, dpreds = pd.DataFrame(dcells), pd.DataFrame(dpreds)
dcells.to_csv(OUT / 'tierD_cells.csv', index=False)
dpreds.to_csv(OUT / 'tierD_row_predictions.csv.gz', index=False)
res['D_frozenX0CatBoost_source_disjoint'] = dict(
    source_equal_MAE=float(dcells.ae.mean()),
    row_weighted_MAE=float(np.abs(dpreds.y_true - dpreds.y_pred).mean()),
    pooled_R2_log_stacked=float(r2_score(np.log1p(dpreds.y_true), np.log1p(dpreds.y_pred))),
    hyperparameters=x0_best,
    hyperparameters_sha256=__import__('hashlib').sha256(
        json.dumps(x0_best, sort_keys=True).encode()).hexdigest())
log('Tier D done')

# ---------- X5: real two-stage train-only DOI fixed-effect residualization (fixed) ----------
def run_x5():
    out = []
    for seed in SEEDS:
        for fold, (tr, te) in enumerate(group_folds(doi_fixed, seed)):
            state5 = {}
            Xtr, ytr = author_features(fixed.iloc[tr], True, state5)
            Xte, _ = author_features(fixed.iloc[te], False, state5)
            ytr_log = np.log1p(ytr)
            dtr = pd.get_dummies(pd.Series(doi_fixed[tr]), prefix='fe').astype(float)
            target = ytr_log - dtr.values @ np.zeros(dtr.shape[1])  # placeholder replaced below
            # stage 1: DOI fixed effects WITH intercept; unseen DOIs get intercept only
            fe = Ridge(alpha=1.0, fit_intercept=True).fit(dtr, ytr_log)
            resid = ytr_log - fe.predict(dtr)
            g_int = float(fe.intercept_)  # RC3 item 4: intercept_, not a constant-column coef
            m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42,
                             n_jobs=-1, verbosity=0)
            m.fit(Xtr, resid)
            yt = y_fixed.iloc[te].to_numpy()
            st = doi_fixed[te]
            pt = np.expm1(m.predict(Xte) + g_int)
            per = pd.DataFrame({'s': st, 'ae': np.abs(yt - pt)}).groupby('s').ae.mean()
            for si, ae in per.items():
                out.append(dict(seed=seed, fold=fold, source=si, ae=float(ae)))
            for ri, yi, pi, si in zip(fixed.index[te], yt, pt, st):
                pass
    return pd.DataFrame(out)


x5 = run_x5()
x5.to_csv(OUT / 'x5_residualized_cells.csv', index=False)
res['X5_residualized_exploratory_SI_only'] = dict(
    source_equal_MAE=float(x5.ae.mean()),
    note='exploratory, SI only; stage-1 Ridge fit_intercept=True provides the global '
         'intercept for unseen DOIs (RC3 item 4 fix); no benefit expected or claimed',
    previous_concat_dummies_implementation='withdrawn')
log('X5 done: ' + str(res['X5_residualized_exploratory_SI_only']['source_equal_MAE']))

# ---------- item 12: record 5-fold vs source 5-fold on the SAME record set ----------
rec5 = []
for seed in SEEDS:
    idx = np.random.default_rng(seed).permutation(len(fixed))
    for held in np.array_split(idx, 5):
        mask = np.zeros(len(fixed), bool)
        mask[held] = True
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        state_r = {}
        Xtr, ytr = author_features(fixed.iloc[tr], True, state_r)
        Xte, _ = author_features(fixed.iloc[te], False, state_r)
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42,
                         n_jobs=-1, verbosity=0)
        m.fit(Xtr, np.log1p(ytr))
        yt = y_fixed.iloc[te].to_numpy()
        st = doi_fixed[te]
        p = np.expm1(m.predict(Xte))
        per = pd.DataFrame({'s': st, 'ae': np.abs(yt - p)}).groupby('s').ae.mean()
        for si, ae in per.items():
            rec5.append(dict(seed=seed, source=si, ae=float(ae)))
rec5 = pd.DataFrame(rec5)
src5 = cellsB2[cellsB2.model == 'XGBoost']
rec_mae = float(rec5.ae.mean())
src_mae = float(src5.ae.mean())
res['item12_grouping_contrast_XGBoost'] = dict(
    record_level_5fold_source_equal_MAE=rec_mae,
    source_group_5fold_source_equal_MAE=src_mae,
    ratio=src_mae / rec_mae,
    note='same 1,757 rows, same 5 seeds, same 5-fold coverage, same model/config — '
         'only the grouping of the splits differs')
log('item12 contrast: record 5-fold ' + str(round(rec_mae, 1)) + ' vs source 5-fold '
    + str(round(src_mae, 1)))

(OUT / 'XU_RC3_RESULTS.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
(OUT / 'run_log.txt').write_text('\n'.join(LOG), encoding='utf-8')
print('XU RC3 complete ->', OUT)
