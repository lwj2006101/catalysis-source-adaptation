# Reproduction routes

Use Python 3.12. Pinned core packages are in the root `requirements.txt` and
the historical environment record in `research/step11/reports/execution_environment.json`.

## 1. Public-package integrity (offline, standard library)

`python scripts/verify_release.py` verifies the current manifest and every
historical archive record, accounting explicitly for published metadata
redactions and optional raw-asset omissions. The pre-publication, untouched
archive verification result is in `docs/original_archive_integrity_check.json`.
Historical checksums are not silently recomputed to erase provenance.
Historical integrity scripts expect the original unredacted archive; use the
root public verifier for this prepared public release.

## 2. Independent error recomputation (offline)

`python scripts/audit_predictions.py --output work/prediction_audit.json`
reads each final-extension prediction array, uses its query-row index to find
the recorded true C2 yield, independently computes per-cell absolute/squared
errors, and compares them with archived cells. It also checks support/query
separation and exports a small measured/predicted/error example CSV.
This recomputes errors from saved predictions; it does not retrain a model.

## 3. Fresh final-extension predictions

```bash
python scripts/reproduce.py smoke --destination work/smoke --workers 1
python scripts/reproduce.py compute --destination work/full --workers 4
```

The launcher copies experiment code and locked protocol into a new directory,
and links the read-only historical reference directory. Smoke generates one
historical fold and compares all saved arrays with its archived counterpart.
Compute generates all 25 historical parts and the 20-partition opened-family
extension and runs the archived independent extension audit. Reports are
written only to the chosen working directory. On Windows use WSL2/Linux for
this symlink-based launcher, or enable directory symlink privileges.
Both modes reuse frozen historical backbones, as the original extension did.

## 4. Earlier analyses (separate protocols)

| Analysis | Archived entry point / location |
| --- | --- |
| Five primary selection panels, source-adapted baselines | `research/step11/reference_step10/reference_step9/reference_step7/code/reproduce.py` |
| Source-tuned comparison and base data snapshot | `.../reference_step7/reference_step3/benchmark/code/run_source_tuned_comparison.py` |
| New OCM publications / opened-family curation | `research/step11/reference_step10/reference_step9/code/` and `analysis/curate_nishimura.py` |
| Predictor-matched kernel analysis | `research/step11/reference_step10/code/run_matched.py` |
| Public Suvarna/Xu original assets and audit | `.../reference_step7/reference_step3/benchmark/source_snapshot/reference_code/fetch_original_assets.py` and `rc3_pipeline.py` |

Read each preserved README and the current manuscript correction table before
running an earlier protocol. The reference snapshots include original source
rows, historical splits/models and result ledgers; upstream author-workflow
assets are fetched separately at recorded commits. Optional author workflows
may need `requirements-public-audits.txt` plus upstream dependencies.
Do not assume that the latest extension's `compute` command rebuilds every
historical study, original author repository, or main-text table.

The preserved upstream fetcher uses shallow Git clones before historical
checkout; if a server cannot resolve an old commit, explicitly fetch that
recorded commit into the cloned repository before checkout.

## 5. Paper build

The ready-to-read current PDFs are under `paper/pdfs/`. To compile current text:

```bash
python scripts/build_paper.py --language cn
python scripts/build_paper.py --language en
```

XeLaTeX is required. Build scripts copy editable sources into `work/paper-*`
before compilation so archived source/figures remain unchanged. The existing
combined supplementary PDF includes original Supplement A and extended B.
The original `paper/source/重新排版.py` also regenerates frozen figures and
supplements and is retained as a historical entry point; the public launcher
does not overwrite the current figure assets.
