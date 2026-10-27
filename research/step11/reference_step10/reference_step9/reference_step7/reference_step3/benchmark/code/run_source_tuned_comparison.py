"""Source-disjoint benchmark with two additional deployable adaptation methods.

All model and adapter tuning sees outer-training sources only. The second-stage
protocol is retrospective: archival test results informed its research question.
The random-source tuning is frozen for both high-temperature and composition /
name-stratum stress panels. Query labels are accepted by scoring, not prediction.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "source_snapshot"
sys.path.insert(0, str(SOURCE / "code"))
from v51_contracts import (COMPOSITION_COLUMNS, RoleCompositionEncoder, TARGET_BOUNDS,
                           affine, cluster_bootstrap, composition_record, stable_seed)

TASK_INPUT = {"methanol": "model_input_revision.csv", "ocm_f0": "ocm_model_input_operating.csv",
              "ocm_f1": "ocm_paired_raw_chemistry.csv", "sty": "sty_model_input_operating.csv"}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
BUDGETS = (1, 2, 3, 5, 10)
METHODS = ("base", "target_mean", "offset_mean", "affine", "eb", "local_ridge",
           "temperature_ridge", "residual_ridge", "adaptive_blend")
GRID = ([dict(kind="ridge", alpha=a) for a in ALPHAS] +
        [dict(kind="gbdt", n=n, depth=d) for n in (100, 300) for d in (2, 3)] +
        [dict(kind="gbdt", n=500, depth=8)])
W0_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
C_GRID = ("inf", 10.0, 2.0, 0.5, 0.0)
VERSION = "step2-source-tuned-1.0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_task(task):
    cfg = json.loads((SOURCE / "benchmark_splits" / task / "task_config.json").read_text())
    d = pd.read_csv(SOURCE / "inputs" / TASK_INPUT[task]).reset_index(drop=True)
    assert d.row_id.is_unique
    if task.startswith("ocm"):
        chemistry = pd.read_csv(SOURCE / "inputs" / TASK_INPUT["ocm_f1"])
        keys = []
        for _, row in chemistry.iterrows():
            rec = composition_record(row)
            terms = [v for group in rec.values() for v in group.values()]
            if not terms or any(v[2] > 0 for v in terms):
                keys.append("")
            else:
                rec = {role: {token: [round(float(x), 6) for x in values]
                              for token, values in group.items()} for role, group in rec.items()}
                keys.append(json.dumps(rec, sort_keys=True, separators=(",", ":")))
        keymap = dict(zip(chemistry.row_id.astype(str), keys))
        d["__stratum"] = d.row_id.astype(str).map(keymap)
        cfg["stratum_kind"] = "reported_composition_pattern"
    elif "catalyst_name" in d:
        d["__stratum"] = d.catalyst_name.fillna("").astype(str).map(
            lambda x: re.sub(r"\s+", " ", x.strip().lower()))
        cfg["stratum_kind"] = "reported_catalyst_name"
    else:
        d["__stratum"] = ""
        cfg["stratum_kind"] = "unavailable"
    assert np.isfinite(d[cfg["targets"]].to_numpy(float)).all()
    return d, cfg


def group_folds(groups, seed, k=3):
    groups = np.asarray(groups, str)
    ids = np.random.default_rng(seed).permutation(np.array(sorted(set(groups))))
    for held in np.array_split(ids, k):
        test = np.flatnonzero(np.isin(groups, held))
        train = np.flatnonzero(~np.isin(groups, held))
        assert set(groups[train]).isdisjoint(groups[test])
        yield train, test


def fitted_transform(train, all_rows, cfg):
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    numeric.fit(train[cfg["features"]].to_numpy(float))
    z = numeric.transform(all_rows[cfg["features"]].to_numpy(float))
    vocabulary = None
    if cfg["use_composition"]:
        chemistry = RoleCompositionEncoder().fit(train[COMPOSITION_COLUMNS])
        z = np.column_stack([z, chemistry.transform(all_rows[COMPOSITION_COLUMNS])])
        vocabulary = {k: list(v) for k, v in chemistry.vocabulary_.items()}
    assert cfg["features"][0] == "Temperature_K"
    assert np.isfinite(z).all()
    return z, dict(numeric_medians=numeric[0].statistics_.tolist(),
                   numeric_mean=numeric[1].mean_.tolist(), numeric_scale=numeric[1].scale_.tolist(),
                   chemistry_vocabulary=vocabulary)


def estimator(cfg):
    if cfg["kind"] == "ridge":
        return Ridge(alpha=cfg["alpha"])
    return GradientBoostingRegressor(n_estimators=cfg["n"], max_depth=cfg["depth"],
                                    learning_rate=0.05, subsample=0.8,
                                    min_samples_split=3, random_state=42)


def source_mean(values, sources):
    return float(pd.DataFrame({"value": values, "source": sources}).groupby("source").value.mean().mean())


def random_design(g):
    rows = g.sort_values("row_id").index.to_numpy()
    nt = max(5, math.ceil(0.25 * len(rows)))
    order = np.random.default_rng(stable_seed("test", g.canonical_source_id.iloc[0])).permutation(rows)
    return order[:nt], order[nt:]


def designs(g):
    """Label-blind scenarios, all restricted to the original fixed main cohort."""
    test, pool = random_design(g)
    if len(pool) < 10:
        return {}
    output = {"random": (test, pool, {})}
    finite = g[np.isfinite(g.Temperature_K)]
    choices = []
    for cutoff in sorted(finite.Temperature_K.unique())[1:]:
        test = finite.index[finite.Temperature_K.ge(cutoff)].to_numpy()
        pool = finite.index[finite.Temperature_K.lt(cutoff)].to_numpy()
        if len(test) >= 5 and len(pool) >= 10:
            choices.append((abs(len(test) - math.ceil(0.25 * len(finite))), float(cutoff), test, pool))
    if choices:
        _, cutoff, test, pool = min(choices, key=lambda x: (x[0], x[1]))
        output["temperature_high"] = (test, pool, dict(temperature_cutoff=cutoff))
    known = g[g.__stratum.ne("")]
    choices = []
    for name, sub in known.groupby("__stratum", sort=True):
        test = sub.index.to_numpy()
        pool = known.index[known.__stratum.ne(name)].to_numpy()
        if len(test) >= 5 and len(pool) >= 10:
            choices.append((abs(len(test) - math.ceil(0.25 * len(known))), name, test, pool))
    if choices:
        _, name, test, pool = min(choices, key=lambda x: (x[0], x[1]))
        assert set(g.loc[test, "__stratum"]).isdisjoint(g.loc[pool, "__stratum"])
        output["stratum_holdout"] = (test, pool, dict(held_stratum=name))
    return output


def calibration_order(d, pool, source, scenario, repeat):
    pool = np.array(sorted(pool, key=lambda i: str(d.row_id.iloc[i])))
    seed = stable_seed("cal", source, repeat) if scenario == "random" else stable_seed("cal_shift", scenario, source, repeat)
    return np.random.default_rng(seed).permutation(pool)


def ridge_query_and_loo(za, ya, zt, alpha):
    """Exact dual Ridge with an unpenalized intercept and analytic LOO."""
    ya = np.asarray(ya, float)
    xc, mean_y = za - za.mean(axis=0), ya.mean(axis=0)
    centered_y = ya - mean_y
    gram = xc @ xc.T
    inv = np.linalg.solve(gram + alpha * np.eye(len(za)), np.eye(len(za)))
    coeff = inv @ centered_y
    query = mean_y + ((zt - za.mean(axis=0)) @ xc.T) @ coeff
    if len(ya) == 1:
        return query, None
    fitted = mean_y + gram @ coeff
    hdiag = 1.0 / len(ya) + np.einsum("ij,ji->i", gram, inv)
    denom = 1.0 - hdiag
    assert np.all(denom > 1e-12)
    loo = ya - (ya - fitted) / (denom[:, None] if ya.ndim == 2 else denom)
    return query, loo


def prior_fit(residual, sources):
    vals = [residual[sources == s] for s in sorted(set(sources))]
    counts = np.array([len(v) for v in vals])
    means = np.array([v.mean() for v in vals])
    denominator = max(int(np.sum(counts - 1)), 1)
    sigma2 = max(sum(float(((v - v.mean()) ** 2).sum()) for v in vals) / denominator, 1e-12)
    tau2 = max(float(np.var(means, ddof=1) - sigma2 * np.mean(1.0 / counts)), 0.0)
    return dict(mu=float(means.mean()), sigma2=sigma2, tau2=tau2)


def blend_weight(ya, local_loo, residual_loo, w0, c, dscale):
    if len(ya) == 1 or c == "inf":
        return float(w0)
    delta = residual_loo - local_loo
    numerator = float(np.dot(delta, ya - local_loo)) + float(c) * dscale * w0
    denominator = float(np.dot(delta, delta)) + float(c) * dscale
    return float(np.clip(numerator / denominator, 0, 1)) if denominator > 1e-12 else float(w0)


def tune_adapters(d, y, p, fold_z, inner_ids, label_scale, repeats, budgets):
    records = {k: [] for k in budgets}
    for source, g in d.groupby("canonical_source_id", sort=True):
        test, pool = random_design(g)
        if len(pool) < 10:
            continue
        j = int(inner_ids[g.index[0]])
        assert np.all(inner_ids[g.index] == j)
        z = fold_z[j]
        for rep in range(repeats):
            order = np.random.default_rng(stable_seed("pseudo_step2", source, rep)).permutation(
                sorted(pool, key=lambda i: str(d.row_id.iloc[i])))
            for k in budgets:
                support = order[:k]
                lr, rr, temp, af, ll, rl = [], [], [], [], [], []
                response = np.column_stack([y[support], y[support] - p[support]])
                for alpha in ALPHAS:
                    query, loo = ridge_query_and_loo(z[support], response, z[test], alpha)
                    lr.append(query[:, 0]); rr.append(p[test] + query[:, 1])
                    ll.append(None if loo is None else loo[:, 0])
                    rl.append(None if loo is None else p[support] + loo[:, 1])
                    tq, _ = ridge_query_and_loo(z[support, :1], y[support], z[test, :1], alpha)
                    temp.append(tq)
                    af.append(affine(p[support], y[support], p[test], alpha, label_scale[j]))
                records[k].append(dict(source=source, group=str(g.leakage_group_id.iloc[0]),
                                       yq=y[test], ya=y[support], local=lr, residual=rr,
                                       temperature=temp, affine=af, local_loo=ll, residual_loo=rl))
    hp, audit = {}, []
    for k, recs in records.items():
        assert recs, "No training pseudo-sources eligible for the fixed cohort"
        sources = [r["source"] for r in recs]
        groups = [r["group"] for r in recs]
        chosen, scores = {}, {}
        for kind in ("local", "residual", "temperature", "affine"):
            scores[kind] = []
            for ai, alpha in enumerate(ALPHAS):
                errors = [float(np.abs(r["yq"] - r[kind][ai]).mean()) for r in recs]
                by_source = pd.DataFrame({"source": sources, "mae": errors}).groupby("source").mae.mean()
                scores[kind].append(float(by_source.mean()))
                audit.extend(dict(k=k, method=kind, hyperparameter=str(alpha), source=s, mae=float(v))
                             for s, v in by_source.items())
            chosen[kind] = int(np.argmin(scores[kind]))
        li, ri = chosen["local"], chosen["residual"]
        errs = np.array([[float(np.abs(r["yq"] - ((1-w)*r["local"][li] + w*r["residual"][ri])).mean())
                          for w in W0_GRID] for r in recs])
        by_source = pd.DataFrame(errs, index=sources).groupby(level=0).mean()
        by_group = pd.Series(groups, index=sources).groupby(level=0).first()
        wi = int(np.argmin(by_source.mean().to_numpy()))
        w0 = W0_GRID[wi]
        diff_scales = []
        for r in recs:
            delta = ((r["residual_loo"][ri] - r["local_loo"][li]) if k > 1 else
                     (r["residual"][ri] - r["local"][li]))
            diff_scales.append(float(np.mean(delta ** 2)))
        by_source_scale = pd.Series(diff_scales, index=sources).groupby(level=0).mean()
        dscale = max(float(by_source_scale.mean()), 1e-12)
        loo_prior = {}
        for group in sorted(set(groups)):
            keep = by_group.ne(group)
            if keep.any():
                lw = W0_GRID[int(np.argmin(by_source.loc[keep].mean().to_numpy()))]
                ld = max(float(by_source_scale.loc[keep].mean()), 1e-12)
            else:
                lw, ld = w0, dscale
            loo_prior[group] = lw, ld
        c_scores = []
        for c in C_GRID:
            errors = []
            for r in recs:
                lw, ld = loo_prior[r["group"]]
                w = blend_weight(r["ya"], r["local_loo"][li], r["residual_loo"][ri], lw, c, ld)
                pred = (1-w)*r["local"][li] + w*r["residual"][ri]
                errors.append(float(np.abs(r["yq"] - pred).mean()))
            bs = pd.DataFrame({"source": sources, "mae": errors}).groupby("source").mae.mean()
            c_scores.append(float(bs.mean()))
            audit.extend(dict(k=k, method="blend_shrinkage", hyperparameter=str(c), source=s, mae=float(v))
                         for s, v in bs.items())
        hp[str(k)] = dict(local_alpha=ALPHAS[li], residual_alpha=ALPHAS[ri],
                          temperature_alpha=ALPHAS[chosen["temperature"]],
                          affine_lambda=ALPHAS[chosen["affine"]], blend_w0=w0,
                          blend_c=C_GRID[int(np.argmin(c_scores))], blend_dscale=dscale,
                          n_pseudo_sources=len(by_source), alpha_scores=scores,
                          w0_scores=by_source.mean().tolist(), c_scores=c_scores)
    return hp, audit


def train_select(d, cfg, target, outer_seed, fold, pseudo_repeats, budgets):
    y = d[target].to_numpy(float)
    sources = d.canonical_source_id.to_numpy(str)
    splits = list(group_folds(d.leakage_group_id, stable_seed("inner", f"{outer_seed}:{fold}")))
    oof = np.empty((len(d), len(GRID)))
    fold_z, fold_meta, scale = [], [], []
    inner_ids = np.empty(len(d), int)
    for j, (train, test) in enumerate(splits):
        z, meta = fitted_transform(d.iloc[train], d, cfg)
        fold_z.append(z); fold_meta.append(meta)
        scale.append(max(float(np.std(y[train])), 1e-8))
        inner_ids[test] = j
        for ci, candidate in enumerate(GRID):
            m = estimator(candidate).fit(z[train], y[train])
            oof[test, ci] = m.predict(z[test])
    errors = [source_mean(np.abs(y-oof[:, i]), sources) for i in range(len(GRID))]
    chosen = int(np.argmin(errors))
    p = oof[:, chosen]
    hp, adapter_audit = tune_adapters(d, y, p, fold_z, inner_ids, scale, pseudo_repeats, budgets)
    prior = prior_fit(y-p, sources)
    return GRID[chosen], hp, prior, p, inner_ids, dict(base_grid=GRID, base_scores=errors,
                                                   inner_preprocessors=fold_meta), adapter_audit


def predict_methods(za, ya, pa, zt, pt, hp, prior, label_scale):
    """Deployment entry point. No query outcomes are accepted."""
    local, lloo = ridge_query_and_loo(za, ya, zt, hp["local_alpha"])
    rq, rloo = ridge_query_and_loo(za, ya-pa, zt, hp["residual_alpha"])
    residual = pt + rq
    if rloo is not None:
        rloo = pa + rloo
    temp, _ = ridge_query_and_loo(za[:, :1], ya, zt[:, :1], hp["temperature_alpha"])
    w = blend_weight(ya, lloo, rloo, hp["blend_w0"], hp["blend_c"], hp["blend_dscale"])
    k = len(ya)
    ew = k * prior["tau2"] / (prior["sigma2"] + k * prior["tau2"])
    output = dict(base=pt, target_mean=np.full(len(pt), ya.mean()), offset_mean=pt+(ya-pa).mean(),
                  affine=affine(pa, ya, pt, hp["affine_lambda"], label_scale),
                  eb=pt+prior["mu"]+ew*((ya-pa).mean()-prior["mu"]), local_ridge=local,
                  temperature_ridge=temp, residual_ridge=residual,
                  adaptive_blend=(1-w)*local+w*residual)
    assert all(np.isfinite(q).all() for q in output.values())
    if k == 1:
        np.testing.assert_allclose(residual, output["offset_mean"], atol=1e-9)
        np.testing.assert_allclose(local, output["target_mean"], atol=1e-9)
    return output, w


def run_part(task, target, seed, fold, args, identity):
    start = time.monotonic()
    part = args.out / "parts" / f"{task}_{seed}_{fold}_{target}"
    part.mkdir(parents=True, exist_ok=True)
    if (part / "done.json").exists():
        old = json.loads((part / "done.json").read_text())
        assert old["identity"] == identity, f"Cache identity changed: {part.name}"
        return part.name
    d, cfg = load_task(task)
    ledger = pd.read_csv(SOURCE / "benchmark_splits" / task / "split_ledger.csv.gz")
    split = ledger[(ledger.outer_seed == seed) & (ledger.fold == fold)]
    tr_ids = set(split.loc[split.role.eq("train"), "row_id"].astype(str))
    te_ids = set(split.loc[split.role.eq("test"), "row_id"].astype(str))
    train = np.flatnonzero(d.row_id.astype(str).isin(tr_ids))
    test = np.flatnonzero(d.row_id.astype(str).isin(te_ids))
    assert len(train)+len(test) == len(d) and not tr_ids & te_ids
    assert set(d.leakage_group_id.iloc[train]).isdisjoint(d.leakage_group_id.iloc[test])
    training = d.iloc[train].reset_index(drop=True)
    model_cfg, hp, prior, oof, inner_ids, selection, audit = train_select(
        training, cfg, target, seed, fold, args.pseudo_repeats, args.budgets)
    z, preprocessor = fitted_transform(training, d, cfg)
    model = estimator(model_cfg).fit(z[train], d[target].to_numpy(float)[train])
    p = model.predict(z)
    label_scale = max(float(training[target].std(ddof=0)), 1e-8)
    train_oof = training[["row_id", "canonical_source_id", "leakage_group_id"]].copy()
    train_oof["inner_fold"], train_oof["y_true"], train_oof["base"] = inner_ids, training[target], oof
    train_oof.to_csv(part / "training_oof.csv.gz", index=False, compression="gzip")
    global_rows = d.iloc[test][["row_id", "canonical_source_id", "leakage_group_id", target]].copy()
    global_rows.rename(columns={target: "y_true"}, inplace=True)
    global_rows["base"] = p[test]
    global_rows.to_csv(part / "zero_all_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(audit).to_csv(part / "pseudo_source_scores.csv.gz", index=False, compression="gzip")
    cells, prediction_rows, design_rows, ledgers = [], [], [], []
    for source, g in d.iloc[test].groupby("canonical_source_id", sort=True):
        for scenario, (query, pool, meta) in designs(g).items():
            assert set(query).isdisjoint(pool) and len(pool) >= max(args.budgets)
            if scenario == "temperature_high":
                assert d.Temperature_K.iloc[query].min() > d.Temperature_K.iloc[pool].max()
            design_rows.append(dict(task=task, source=source, scenario=scenario,
                                    leakage_group=str(g.leakage_group_id.iloc[0]), n_source=len(g),
                                    n_test=len(query), n_pool=len(pool), n_excluded=len(g)-len(query)-len(pool),
                                    stratum_kind=cfg["stratum_kind"],
                                    test_ids=json.dumps(d.row_id.iloc[query].tolist()),
                                    pool_ids=json.dumps(d.row_id.iloc[pool].tolist()), **meta))
            yq = d[target].to_numpy(float)[query]
            for rep in range(args.repeats):
                order = calibration_order(d, pool, source, scenario, rep)
                if seed == args.seeds[0] and target == cfg["targets"][0]:
                    ledgers.append(dict(task=task, source=source, scenario=scenario, repeat=rep,
                                        ordered_pool_ids=json.dumps(d.row_id.iloc[order].tolist())))
                for k in args.budgets:
                    support = order[:k]
                    q, weight = predict_methods(z[support], d[target].to_numpy(float)[support], p[support],
                                                z[query], p[query], hp[str(k)], prior, label_scale)
                    row = dict(task=task, target=target, outer_seed=seed, fold=fold, source=source,
                               leakage_group=str(g.leakage_group_id.iloc[0]), scenario=scenario,
                               repeat=rep, k=k, n_test=len(query), blend_weight=weight)
                    for method, prediction in q.items():
                        lo, hi = TARGET_BOUNDS[target]
                        bounded = np.maximum(prediction, lo) if hi is None else np.clip(prediction, lo, hi)
                        row[f"mae__{method}"] = float(np.abs(yq-prediction).mean())
                        row[f"mae_bounded__{method}"] = float(np.abs(yq-bounded).mean())
                    cells.append(row)
                    if rep == 0:
                        for qi, index in enumerate(query):
                            prediction_rows.append({**{key: row[key] for key in
                                ("task", "target", "outer_seed", "fold", "source", "leakage_group", "scenario", "repeat", "k")},
                                "row_id": str(d.row_id.iloc[index]), "y_true": float(yq[qi]),
                                "support_ids": json.dumps(d.row_id.iloc[support].tolist()),
                                "blend_weight": weight, **{method: float(pred[qi]) for method, pred in q.items()}})
    pd.DataFrame(cells).to_csv(part / "cells.csv.gz", index=False, compression="gzip")
    pd.DataFrame(prediction_rows).to_csv(part / "row_predictions_repeat0.csv.gz", index=False, compression="gzip")
    pd.DataFrame(design_rows).to_csv(part / "designs.csv", index=False)
    pd.DataFrame(ledgers, columns=["task", "source", "scenario", "repeat", "ordered_pool_ids"]).to_csv(
        part / "support_ledger.csv.gz", index=False, compression="gzip")
    write_json(part / "done.json", dict(identity=identity, task=task, target=target, seed=seed, fold=fold,
                                        model=model_cfg, hyperparameters=hp, prior=prior,
                                        selection=selection, outer_preprocessor=preprocessor,
                                        n_train=len(train), n_outer_test=len(test), n_cells=len(cells),
                                        wall_seconds=time.monotonic()-start))
    print("DONE", part.name, "model", model_cfg, "seconds", round(time.monotonic()-start, 1), flush=True)
    return part.name


def aggregate(args, jobs):
    parts = [args.out / "parts" / f"{t}_{s}_{f}_{y}" for t, y, s, f in jobs]
    cells = pd.concat([pd.read_csv(p / "cells.csv.gz") for p in parts], ignore_index=True)
    metrics = [f"mae__{m}" for m in METHODS] + [f"mae_bounded__{m}" for m in METHODS]
    keys = ["task", "target", "scenario", "source", "leakage_group", "k"]
    assert not cells.duplicated(keys+["outer_seed", "repeat"]).any()
    per_seed = cells.groupby(keys+["outer_seed"], as_index=False)[metrics+["n_test", "blend_weight"]].mean()
    count = per_seed.groupby(keys).outer_seed.nunique()
    assert count.eq(len(args.seeds)).all()
    per_source = per_seed.groupby(keys, as_index=False)[metrics+["n_test", "blend_weight"]].mean()
    summary, contrasts = [], []
    panels = {name: per_source[per_source.scenario.eq(name)].copy() for name in per_source.scenario.unique()}
    for stress in ("temperature_high", "stratum_holdout"):
        if stress not in panels:
            continue
        eligible = per_source.loc[per_source.scenario.eq(stress), ["task", "target", "source"]].drop_duplicates()
        matched = per_source.loc[per_source.scenario.eq("random")].merge(eligible, on=["task", "target", "source"], validate="many_to_one")
        matched["scenario"] = "random_matched_"+stress
        panels["random_matched_"+stress] = matched
    all_panels = pd.concat(list(panels.values()), ignore_index=True)
    for (task, target, scenario, k), g in all_panels.groupby(["task", "target", "scenario", "k"], sort=True):
        for method in METHODS:
            cb = cluster_bootstrap(g, f"mae__{method}", draws=args.bootstrap_draws,
                                   seed=stable_seed(VERSION, task, target, scenario, k, method))
            interval = cb["ci95"] or [None, None]
            summary.append(dict(task=task, target=target, scenario=scenario, k=k, method=method,
                                source_equal_mae=cb["estimate"], ci_low=interval[0], ci_high=interval[1],
                                row_weighted_mae=float(np.average(g[f"mae__{method}"], weights=g.n_test)),
                                bounded_mae=float(g[f"mae_bounded__{method}"].mean()),
                                n_sources=len(g), n_groups=cb["n_groups"], n_query_rows=int(g.n_test.sum())))
        for method in ("residual_ridge", "adaptive_blend"):
            references = ["local_ridge", "temperature_ridge", "offset_mean", "affine", "eb", "base"]
            if method == "adaptive_blend":
                references.append("residual_ridge")
            for reference in references:
                delta = g[["source", "leakage_group"]].copy()
                delta["delta"] = g[f"mae__{method}"]-g[f"mae__{reference}"]
                cb = cluster_bootstrap(delta, "delta", draws=args.bootstrap_draws,
                                       seed=stable_seed("contrast", task, target, scenario, k, method, reference))
                interval = cb["ci95"] or [None, None]
                contrasts.append(dict(task=task, target=target, scenario=scenario, k=k, method=method,
                                      reference=reference, delta_mae=cb["estimate"], ci_low=interval[0],
                                      ci_high=interval[1], n_improved=int(delta.delta.lt(-1e-10).sum()),
                                      n_harmed=int(delta.delta.gt(1e-10).sum()), n_sources=len(delta)))
    cells.to_csv(args.out / "all_repeat_cells.csv.gz", index=False, compression="gzip")
    per_seed.to_csv(args.out / "per_source_per_seed.csv.gz", index=False, compression="gzip")
    all_panels.to_csv(args.out / "per_source_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out / "paired_contrasts.csv", index=False)
    design = pd.concat([pd.read_csv(p / "designs.csv") for p in parts], ignore_index=True)
    design.drop_duplicates(["task", "source", "scenario"]).to_csv(args.out / "scenario_designs.csv", index=False)
    orders = pd.concat([pd.read_csv(p / "support_ledger.csv.gz") for p in parts], ignore_index=True)
    orders.drop_duplicates(["task", "source", "scenario", "repeat"]).to_csv(args.out / "support_ledger.csv.gz", index=False, compression="gzip")
    row_predictions = pd.concat([pd.read_csv(p / "row_predictions_repeat0.csv.gz") for p in parts], ignore_index=True)
    row_predictions.to_csv(args.out / "row_predictions_repeat0.csv.gz", index=False, compression="gzip")
    parameters = [json.loads((p / "done.json").read_text()) for p in parts]
    write_json(args.out / "selected_parameters.json", parameters)
    write_json(args.out / "verification.json", dict(n_parts=len(parts), n_repeat_cells=len(cells),
                methods=list(METHODS), budgets=args.budgets, seeds=args.seeds, support_query_disjoint=True,
                outer_leakage_groups_disjoint=True, inner_leakage_groups_disjoint=True,
                query_outcomes_used_for_selection=False, all_methods_paired=True,
                expected_seed_count_per_source=len(args.seeds), scope="retrospective internal validation"))
    print(pd.DataFrame(summary).query("scenario=='random' and k==5").to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PACKAGE / "results")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--pseudo-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_INPUT), default=list(TASK_INPUT))
    args = parser.parse_args()
    assert all(k in BUDGETS for k in args.budgets)
    args.out.mkdir(parents=True, exist_ok=True)
    identity = dict(version=VERSION, code_hash=sha(__file__), contracts_hash=sha(SOURCE/"code"/"v51_contracts.py"),
                    data_hashes={name: sha(SOURCE/"inputs"/name) for name in TASK_INPUT.values()},
                    split_hashes={task: sha(SOURCE/"benchmark_splits"/task/"split_ledger.csv.gz") for task in TASK_INPUT},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    seeds=args.seeds, folds=args.folds, budgets=args.budgets,
                    repeats=args.repeats, pseudo_repeats=args.pseudo_repeats,
                    base_grid=GRID, alphas=ALPHAS, w0_grid=W0_GRID, c_grid=C_GRID,
                    stress_hyperparameters="frozen random-pseudo-source tuning",
                    design_status="locked before this run; informed by earlier archival results")
    # JSON arrays are lists; normalize before equality checks on a resumed run.
    identity = json.loads(json.dumps(identity))
    lock = args.out / "protocol_lock.json"
    if lock.exists():
        assert json.loads(lock.read_text())["identity"] == identity
    else:
        write_json(lock, dict(locked_utc=datetime.now(timezone.utc).isoformat(), identity=identity))
    jobs = []
    for task in args.tasks:
        _, cfg = load_task(task)
        jobs.extend((task, target, seed, fold) for target in cfg["targets"] for seed in args.seeds for fold in args.folds)
    print("START", len(jobs), "parts", "workers", args.workers, flush=True)
    Parallel(n_jobs=args.workers)(delayed(run_part)(t, y, s, f, args, identity) for t, y, s, f in jobs)
    aggregate(args, jobs)


if __name__ == "__main__":
    main()
