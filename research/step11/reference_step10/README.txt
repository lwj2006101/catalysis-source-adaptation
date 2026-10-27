Step10 complete research release: predictor-matched acquisition under publication shift

This is an executed research extension, not a job specification awaiting a private server.
Root is the extracted step10 folder. Historical source-group fits and measured input provenance
are included through the complete, checksum-preserved reference_step9 release.
Scientific code resolves paths from its own location. No SSH connection or new data download
is needed for the included evaluations.

Evidence:
- 25 actual historical comparisons; 82,350 cells and 4,099,200 scalar query predictions.
- Three opened-cohort partitions; 1,830 cells and 5,696,180 scalar query predictions.
- 12 acquisitions, seven fixed predictors, two feature views, five nested budgets.
- Independent loss reaggregation passed for every scalar; 825,840 kernel values reconstructed.
- No new backbone fits. Kernel parameter validation, acquisition and predictions were executed.
- A six-predictor pilot is preserved separately; the seventh predictor was added before the full grid.
- The covariance-averaged absolute-risk primary candidate did not establish superiority.
- Catalyst holdout is an additional task within the same opened publication, not a new cohort.

Runtime used: Python 3.12.14, versions in requirements.txt, and pdflatex.
The archived joblib models require compatible scikit-learn dependencies.
Use approximately four CPU workers; the runs need no GPU.

Start with a read-only integrity check:
    python3 analysis/verify_release.py
This reads checkpoints and checksums. It does not count cached files as new fits.

To run a fresh full historical comparison in a new result directory:
    python3 code/run_matched.py --run-name history_reproduction_v1 --workers 4
    python3 analysis/audit_aggregate.py --run-name history_reproduction_v1
For a fresh one-fold portability check using the current seven-predictor version:
    python3 code/run_matched.py --run-name smoke_reproduction_v1 --seeds 42 --folds 0 --workers 1
    python3 analysis/audit_aggregate.py --run-name smoke_reproduction_v1
The archive's pilot_v1 uses its preserved six-predictor code and is a different version.

To independently recompute full-history kernel parameters and the opened application:
    python3 code/freeze_apply.py freeze --name full_history_reproduction_v1
    python3 code/freeze_apply.py apply --name nishimura_opened_reproduction_v1 --freeze-name full_history_reproduction_v1
    python3 analysis/audit_aggregate.py --run-name applications/nishimura_opened_reproduction_v1
The freeze entry verifies the archived historical completion/audit and the preceding frozen
pipeline, then actually retunes the historical kernel. It does not fit global backbones.
Keep existing result folders as evidence; use a new version when changing scientific code.

To rebuild the current presentation from the released actual results (in a working copy):
    python3 analysis/build_step10_results.py
    python3 analysis/build_chinese_progress.py
Run pdflatex twice on manuscript_en.tex and supplementary_en.tex from manuscripts/.
The table/figure builder does not rerun scientific predictions. Its figures are standard plots.
qa_pdfs.py uses a temporary image path from the original runtime; change that path if desired.

Arrays are lossless float64 NPZ; their keys and predictor order are defined in run_matched.py
and audit_aggregate.py. Sources are equally weighted after averaging draws then seeds.
Source/fold repetition and multiple query tasks do not create new independent sources.
No population interval is reported for the single opened experiment family.

The package includes previous complete manuscripts and scientific records, current manuscript
sources/PDFs, Chinese report, figures, frozen choices, all data ledgers, execution logs and hashes.
For research decisions and further methodological freedom, read AGENT_START_HERE.txt.
