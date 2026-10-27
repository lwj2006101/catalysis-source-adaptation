# 文献催化预测：来源隔离与少量实测结果适配

本研究回答三个问题：模型换一篇没见过的文献以后还能预测多准？先提供少量该来源的真实结果，能改善多少？同样只允许知道五条结果，选哪五条更有帮助？

## 先看什么

- 最新中文论文：`paper/pdfs/论文中文版_原稿逻辑整合修订.pdf`。
- 英文主文与完整补充材料：同一目录。
- 主选样结果：`results/primary_selection.csv`，对应最新正文表7。
- 简单解释：`docs/EXPLAINER_CN.md`。
- 完整科学档案：`research/step11/`。它包含实际计算代码、历史模型、输入、数据划分、支持记录选择、逐条预测、结果与审计记录。

## 三种不同操作

```bash
# 1. 检查公开包的每个文件及其与原归档的对应；只用Python标准库。
python scripts/verify_release.py

# 2. 从保存的真实结果和预测值重新计算最新扩展的逐单元误差。
python -m pip install -r requirements-minimal.txt
python scripts/audit_predictions.py --output work/prediction_audit.json

# 3. 真正生成一折新的预测；使用已有冻结输入和历史骨干模型。
python -m pip install -r requirements.txt
python scripts/reproduce.py smoke --destination work/smoke --workers 1
```

第1项是文件检查，第2项是误差复算，第3项才产生新的科学预测。三者不能混称为“全部实验重新跑通”。完整扩展运行、早期基准与公开论文复核的入口见 `docs/REPRODUCING.md`。CPU即可，原归档使用Python 3.12和固定版本依赖。

## 目前主要结果

- 基础比较中，五条新来源实测结果配合局部岭回归，使甲醇转化率平均误差由5.279降至2.429个百分点；同五条结果只取平均时为4.304。
- 同样五条标签、相同组合预测模型配置，查询V选样相对于随机选样，使STY误差由62.608降至50.274，使OCM组成设置的误差由3.081降至2.708，分别降低19.7%和12.1%。
- 组成既能作为预测输入，也能用于选择先获取哪些记录的结果；不同分析流程使用各自匹配对照，不能跨表混减。
- 附加实测数据的20次划分来自同一个实验家族；全部分析为已有实测记录上的回顾性计算，没有新增湿实验。

现有经典估计器和V最优设计本身不作为新算法贡献。研究重点是把跨来源评价、少量结果的使用、输入信息与测量安排组织成可复核的证据。

## 开源与上传

解压后上传本目录内的文件，仓库名可用 `catalysis-source-adaptation`。完整上传步骤见 `docs/GITHUB_UPLOAD_CN.md`。大结果文件适合使用Git或GitHub Desktop上传。

自有代码按MIT开放；第三方数据、文章、字体及论文保留各自条件，见 `THIRD_PARTY_NOTICES.md`。未明确单独再分发许可的原始CSV及一个未纳入证据的出版物补充文件保留来源和校验值，另行获取；核心已运行输入、模型和预测档案包含在内。
