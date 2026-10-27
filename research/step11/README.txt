Step11: completed scientific manuscript and submission preparation

Start with 00_第11步进度说明.txt and 01_文件与投稿路径索引.txt.
The current scientific master is manuscripts/manuscript_en.pdf. Its completed
scientific content is author-neutral: names, affiliations, contribution/funding
and conflict declarations cannot be inferred and are supplied by the authors.
The full SI is manuscripts/supplementary_en.pdf.

Use Python 3.12 and the pinned requirements.txt. Scientific runs need no GPU,
private server, SSH access or new data download. Use about four CPU workers.
The previous 56 fitted backbones are preserved, not counted as newly fitted.

Quick read-only verification:
    python3 analysis/reproduce.py check

Genuinely fresh one-fold portability check (new predictions, no cached completion):
    python3 analysis/reproduce.py smoke --destination tmp/new_smoke --workers 1

Fresh full computation in an empty new directory:
    python3 analysis/reproduce.py compute --destination runs/new_reproduction --workers 4
The archived evidence is preserved. A fresh run uses a symlink to the complete
read-only reference_step10 inputs, so do not move the reference directory alone.

Rebuild current paper tables/figures and both English PDFs:
    python3 analysis/reproduce.py paper
This presentation rebuild reads the archived actual runs; it is not a new fit.

Fill submission/author_metadata.json with actual information. Authors need to
confirm their own declarations and approval. Then build a separately named PDF:
    python3 analysis/finalize_author_info.py
It creates manuscripts/manuscript_submission.pdf, the corresponding signed SI,
and submission/cover_letter_submission.pdf; it retains the neutral masters.
No article has been submitted or repository record published by this package.

Every fixed-RBF prediction in this extension is reconstructed independently by
analysis/audit_extension.py using sklearn kernels/centering/KernelRidge. All scalar
losses are reaggregated. The proof is checked in verify_operator_math.py.
Source populations, folds, support orders, row predictions, parameter choices and
execution/audit records are all included. Twenty catalyst partitions share one
opened experimental family; they are not twenty independent data sources.

The full predecessor archive remains checksum-identical under reference_step10/.
Newly investigated unaccepted public datasets are not used as extra evidence.
Restricted new raw data and failed downloads are omitted from the deliverable.
The acquisition receipts/URLs and exclusion rationale are retained in reports.

Latest total: 10,948,000 new scalar predictions, 2,737,000 independent fixed-RBF
reconstructions, 25 historical parts and 20 within-family catalyst partitions.


2026-10-12 中英文阅读版合并：
manuscripts/第11步_论文英文原文.pdf：12页英文主文独立阅读版。
manuscripts/第11步_论文中文版.pdf：11页完整中文主文译文。
manuscripts/第11步_论文中文译文编辑源文件.zip：中文编辑源文件小包。
manuscripts/chinese_source/：已展开的中文LaTeX、图表和重建入口。
中文译文保留公式及数值，表格与图注已翻译；图内文字及文献沿用原文。
此更新合并阅读文件并修复恢复时发现的旧ZIP截断，不增加新的科学结论。
