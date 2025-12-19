## TiCard: Deployable EXPLAIN-only Residual Learning for Cardinality Estimation

## Qizhi Wang 1*

1* PingCAP, Data &amp; AI-Innovation Lab, Beijing, China.

Corresponding author(s). E-mail(s): qizhi.wang@pingcap.com;

## Abstract

Cardinality estimation is a key bottleneck for cost-based query optimization, yet deployable improvements remain difficult: classical estimators miss correlations, while learned estimators often require workload-specific training pipelines and invasive integration into the optimizer. This paper presents TiCard , a lowintrusion, correction-based framework that augments (rather than replaces) a database's native estimator. TiCard learns multiplicative residual corrections using EXPLAIN -only features, and uses EXPLAIN ANALYZE only for offline labels. We study two practical instantiations: (i) a Gradient Boosting Regressor for sub-millisecond inference, and (ii) TabPFN, an in-context tabular foundation model that adapts by refreshing a small reference set without gradient retraining. On TiDB with TPCH and the Join Order Benchmark, in a low-trace setting (263 executions total; 157 used for learning), TiCard improves operator-level tail accuracy substantially: P90 Q-error drops from 312.85 (native) to 13.69 (TiCard-GBR), and P99 drops from 37,974.37 to 3,416.50 (TiCard-TabPFN), while a join-only policy preserves near-perfect median behavior. We position TiCard as an AI4DB building block focused on deployability: explicit scope, conservative integration policies, and an integration roadmap from offline correction to in-optimizer use.

Cardinality estimation, Query optimization, ML-for-DB, AI4DB,

Keywords: In-context learning, TiDB, EXPLAIN

## 1 Introduction

Cardinality estimation (CE)-predicting the number of rows produced by each operator-is central to cost-based query optimization, affecting join ordering, physical operator choice, and memory management [1, 2]. Despite decades of work, CE

remains brittle in modern analytical workloads, primarily because independence assumptions and limited statistics struggle with multi-column predicates and cross-table correlations [3, 4].

From an AI4DB perspective, the challenge is not only improving accuracy but doing so in a way that is deployable : learned estimators can be accurate, yet are often costly to train, sensitive to workload drift, and require deep integration into the optimizer's enumeration loop [5-7]. In practice, database teams frequently prefer incremental, low-risk changes that preserve existing optimizer behavior and can be rolled out conservatively.

This paper proposes a pragmatic framing: treat the native optimizer as a strong prior and learn only its residual error . We introduce TiCard , a correction-based CE framework that learns multiplicative adjustments on top of the optimizer estimate. Crucially, TiCard's feature pipeline is derived from EXPLAIN only, enabling a lowintrusion path that leverages existing database interfaces. EXPLAIN ANALYZE is used solely for offline label collection.

## 1.1 Scope and deployability goals

We explicitly scope this work to the setting that is most actionable for deployment teams:

- Low intrusion: learn from existing interfaces ( EXPLAIN / EXPLAIN ANALYZE ) without requiring a new optimizer or deep runtime instrumentation.
- Safety controls: support conservative policies (e.g., join-only correction, blending with fallback) to preserve strong baseline behavior.
- Data efficiency: operate under a low-trace regime where executed-query labels are expensive (hundreds of executions, not thousands).
- Evaluation focus: we report offline , operator-level CE accuracy on collected plans; we do not claim end-to-end plan-quality or latency gains without full integration into the optimizer.

This scope is not a limitation to hide; it is a design choice motivated by deployability. We therefore also provide an integration roadmap that describes how to use TiCardstyle corrections inside a live optimizer, and where the engineering risks and overheads arise.

## 1.2 Contributions

Our main contributions are:

1. EXPLAIN-only correction formulation: We frame CE as learning multiplicative residual corrections using a leakage-free feature pipeline derived from EXPLAIN plans.
2. Deployable model choices: We study two complementary instantiationsTabPFN in-context learning (fast refresh without gradient retraining) and Gradient Boosting Regression (very fast inference).
3. Conservative integration policies: We define and evaluate practical policies (join-only correction, blending, and a two-stage design for zero-cardinality cases) aimed at controlling regressions.

Fig. 1 TiCard pipeline: plan extraction, EXPLAIN-only feature engineering, correction-target construction, and model setup/training and inference.

![Image](TiCard_e1da9db1-referenced_artifacts/img_001.png)

4. Empirical evaluation in a low-trace regime: On TiDB with TPC-H and JOB, we show large tail improvements at the operator level using only 157 training executions, and quantify setup/training and inference costs.
5. Integration roadmap: We outline a path from offline correction to online use, with overhead and risk considerations that matter for real deployments.

## 2 Background

## 2.1 Cardinality estimation and Q-error

CE quality is commonly measured by Q-error [8]:

<!-- formula-not-decoded -->

Q-error is scale-invariant and symmetric for over/underestimation. We apply a standard guard max( · , 1) to keep Q-error well-defined when the true or predicted cardinality is zero.

## 2.2 Plan interfaces: EXPLAIN and EXPLAIN ANALYZE

Many systems expose estimated and actual operator cardinalities through plan interfaces. In TiDB, EXPLAIN returns the chosen plan with estimated rows ( estRows ), and EXPLAIN ANALYZE executes the query and reports actual rows ( actRows ). TiCard uses EXPLAIN to construct features and EXPLAIN ANALYZE only to obtain labels.

## 3 TiCard: Correction-based CE with EXPLAIN-only Features

## 3.1 Overview

Given an operator node q in a query plan, let E ( q ) be the native optimizer estimate from EXPLAIN and C ( q ) be the true cardinality observed from EXPLAIN ANALYZE . TiCard learns a function f ( · ) that predicts a log-space correction target:

<!-- formula-not-decoded -->

At inference time, the corrected estimate is:

<!-- formula-not-decoded -->

This residual formulation leverages the native estimator as a prior and focuses learning capacity on systematic optimizer errors.

## 3.2 Feature engineering (leakage-free)

All model inputs are derived from EXPLAIN only. We extract per-operator features that capture local properties and global plan context, including:

- Native estimate: optimizer estimated output cardinality (and its log transform).
- Operator attributes: operator type (scan/filter/join/aggregation), table name, task type, and join/scan sub-types when applicable.
- Plan structure: depth/position features and ratios relative to total estimated rows.

We split datasets by query execution (not by operator) to avoid leakage: all operator nodes from a query belong to the same split. For preprocessing, we use onehot encoding for categorical features, standard scaling for numerical features, and simple missing-value handling (zero or unknown ). To reduce overfitting in the low-trace regime, we apply feature selection ( SelectKBest with k tuned on a validation set). For correction learning, we optionally stabilize the training target in Eq. 2 with a robust IQR-based clipping range; evaluation still reports on all test samples without filtering.

## 3.3 Model instantiations

TiCard-GBR uses gradient boosting regression [9] for maximum inference speed. This choice targets production-grade overhead constraints, where CE must be computed frequently.

TiCard-TabPFN uses TabPFN [10], a transformer pre-trained for tabular prediction, in an in-context learning mode. TabPFN adapts by refreshing the reference set (the setup data) without gradient retraining, which is attractive under workload drift when labels are available but retraining pipelines are costly.

## TabPFN regression setup.

Although TabPFN is often discussed in the context of classification, we use a regression variant ( TabPFNRegressor ) to predict continuous targets. Specifically, the target is the continuous log-space correction in Eq. 2 (or log(1 + act) for the direct ablation). We use a small ensemble ( n estimators=8 ) for throughput and run inference without fine-tuning.

## 3.3.1 Terminology: 'setup' vs. 'training'

To avoid ambiguity, we distinguish two learning paradigms:

- Setup / in-context learning (TabPFN): fit() stores the reference set for attention-based inference; no gradient descent or parameter updates occur.
- Training (GBR and neural baselines): model parameters are learned via standard optimization (boosting or backpropagation).

Throughout the paper, 'training executions' refer to the same labeled traces, but they are consumed as setup data for TabPFN and as training data for GBR/neural models.

## 3.4 Conservative integration policies

Correction is a design space rather than a single rule. To improve deployability, TiCard supports conservative policies that can be selected and tuned on a validation set:

- Join-only correction: apply corrections only to join operators and fall back to native estimates for other nodes, preserving strong median behavior.
- Two-stage zero handling: a classifier for zero vs. non-zero cardinality followed by regression, reducing instability around zero-cardinality operators.
- Blending: combine corrected and native estimates with a bounded rule (e.g., clamp correction factor within a range) to reduce risk from outliers.

## 4 Experimental Evaluation (Offline Operator-level)

## 4.1 Setup

System: TiDB v8.5.2. After loading datasets we run ANALYZE TABLE to populate statistics.

Benchmarks: TPC-H (scale factor 1) and JOB [4, 11]. For TPC-H we execute 22 query templates with multiple parameterizations (150 executions total); for JOB we execute 113 distinct queries (113 executions). Overall, we collect 263 query executions and 6114 operator samples.

Split: 157 executions for learning (3556 operator samples), 53 for validation (1369), and 53 for test (1189).

Baselines: native TiDB estimates, simple calibration baselines (group scaling and isotonic regression), a LiteCard-style local correction baseline [12], and a MADE (NeuroCard-architecture) correction baseline [6, 13] trained on our EXPLAIN-derived feature vectors (not NeuroCard's canonical data-driven pipeline). Note on LiteCard: our

Table 1 Q-error on the test set (1189 operator samples; no filtering).

| Model                         |    P90 | P99       |   Median | Mean     |
|-------------------------------|--------|-----------|----------|----------|
| TiDB Default                  | 312.85 | 37,974.37 |   1.003  | 3,045.79 |
| Scale (group)                 | 226.79 | 16,214.41 |   1.01   | 1,728.37 |
| Isotonic (group)              |  75.7  | 2,706.81  |   1.8247 | 131.20   |
| LiteCard (corr.)              |  55.29 | 2,355.90  |   1.8751 | 140.35   |
| TiCard-TabPFN (our)           |  13.82 | 3,416.50  |   1.0406 | 131.90   |
| TiCard-GBR (our)              |  13.69 | 3,812.02  |   1.3158 | 122.03   |
| TiCard-GBR (join- only, our)  |  70.54 | 4,576.90  |   1.0078 | 1,559.91 |
| MADE (NeuroCard arch., corr.) | 202.6  | 12,730.89 |   1.8363 | 988.50   |

P90

P99

Mean

Fig. 2 Q-error comparison across models (P90, P99, Mean; log scale).

![Image](TiCard_e1da9db1-referenced_artifacts/img_002.png)

baseline is an offline analogue of LiteCard's hierarchical local-correction idea (patternkeyed regressors with fallback). It does not reproduce LiteCard's full online learning loop or planner hooks; we use it to contextualize lightweight correction behavior under the same EXPLAIN-only, offline setting.

Metrics: Q-error statistics (median/mean/P90/P99), quality-band distribution, and wall-clock setup/training and inference time.

## 4.2 Overall results

TiCard substantially improves tail accuracy at the operator level, while preserving strong baseline behavior via conservative policies. In particular, TiCard-GBR reduces P90 Q-error from 312.85 to 13.69, and TiCard-TabPFN reduces P99 from 37,974.37 to 3,416.50. The join-only policy illustrates a deployability trade-off: it preserves TiDB's near-perfect median (1.0078) while still reducing tail errors, but gives up some of the full-correction improvements.

Table 2 Q-error distribution on the test set (1189 operator samples).

| Model                         | Excellent ( ≤ 2)   | Good (2-5]   | Fair (5-10]   | Poor (10-100]   | Terrible ( > 100)   |
|-------------------------------|--------------------|--------------|---------------|-----------------|---------------------|
| TiDB Default                  | 67.9%              | 5.8%         | 2.6%          | 11.2%           | 12.5%               |
| Scale (group)                 | 64.3%              | 10.2%        | 4.0%          | 9.2%            | 12.4%               |
| Isotonic (group)              | 52.6%              | 18.1%        | 9.1%          | 12.4%           | 7.9%                |
| TiCard-TabPFN (our)           | 78.2%              | 6.4%         | 3.7%          | 7.4%            | 4.3%                |
| TiCard-GBR (our)              | 72.8%              | 11.3%        | 4.1%          | 7.3%            | 4.5%                |
| LiteCard (corr.)              | 52.5%              | 20.9%        | 6.1%          | 12.4%           | 8.0%                |
| MADE (NeuroCard arch., corr.) | 52.4%              | 23.4%        | 3.4%          | 9.0%            | 11.8%               |

Fig. 3 Q-error quality bands (percentage of estimates in each range).

![Image](TiCard_e1da9db1-referenced_artifacts/img_003.png)

## 4.3 Distribution analysis

TiCard reduces catastrophic errors (Q-error &gt; 100) from 12.5% to about 4-5% and increases the share of excellent estimates ( ≤ 2). This matters for deployability because optimizer failures are often driven by tail events rather than median behavior.

## 4.4 Robustness and ablations (no new executions)

Review feedback often asks whether improvements rely on memorization via instancespecific categorical features, how corrections behave across operator types, and whether per-operator corrections translate into plan-level improvements. We address these with additional analyses computed on the same cached EXPLAIN / EXPLAIN ANALYZE traces (no new query executions).

## 4.4.1 Feature inventory and selection

Table 3 enumerates categorical feature cardinalities in our TiDB plan traces and the resulting feature dimensionality. After one-hot encoding, the learning pipeline uses a

Fig. 4 Cumulative distribution function (CDF) of Q-error (log scale).

![Image](TiCard_e1da9db1-referenced_artifacts/img_004.png)

Table 3 Feature inventory and selection (offline traces).

| Item                      |   Count | Notes                                  |
|---------------------------|---------|----------------------------------------|
| Operator types            |      18 | operator type categories               |
| Task types                |       2 | task type categories                   |
| Join types                |       5 | join type categories                   |
| Table identities          |       4 | extracted when available; often coarse |
| Feature dim (post-encode) |      14 | numeric features used by models        |
| Selected k                |      10 | chosen on validation set               |

compact feature matrix (14 numeric features after dropping IDs/labels). Validationtuned SelectKBest retains k = 10 features for correction mode; the selected set is dominated by native-estimate and plan-structure signals, consistent with the deployability goal of minimizing feature dependence.

For the default split, the 10 selected features are: optimizer est out , log est rows , plan depth , node position , relative position , est to total ratio , and join/scan indicators ( is join , is scan , is table scan , is hash join ).

## 4.4.2 Per-operator breakdown

Table 4 breaks down tail errors by coarse operator group. The largest reductions occur on joins (where correlation effects dominate), while scans/aggregations also improve substantially. This addresses the concern that improvements are confined to a narrow operator class.

## 4.4.3 Plan-level proxy: root-node cardinality

Although we do not integrate into TiDB's plan enumeration loop, we can report a plan-level proxy using the root operator's output cardinality (one root node per query execution). Table 5 shows that independent per-operator correction nevertheless

Table 4 Per-operator breakdown (test split; P90/P99 Q-error).

| Group       |   P90 (TiDB) | P99 (TiDB)   |   P90 (TiCard-GBR) | P99 (TiCard-GBR)   |
|-------------|--------------|--------------|--------------------|--------------------|
| Join        |      1815.32 | 45,568.47    |              59.02 | 6,263.02           |
| Scan        |        18.97 | 1,710.98     |               5.9  | 599.68             |
| Filter      |       109.51 | 2,937.37     |              21.14 | 1,367.13           |
| Aggregation |       363.18 | 3,514.45     |               7.01 | 307.34             |
| Other       |       902.32 | 219,959.70   |              23.01 | 12,863.68          |

Table 5 Root-node (plan output) Q-error across queries (53 executions in the test split).

| Model        |   Median |   P90 |    P99 |   Mean |
|--------------|----------|-------|--------|--------|
| TiDB Default |     1    | 55.46 | 989.68 |  62.29 |
| TiCard-GBR   |     1.26 |  2.38 |  23.11 |   2.79 |

Table 6 Query-level worst-case proxies over the 53 test executions.

| Metric                | Model        |   Median | P90      | P99        | Mean      |
|-----------------------|--------------|----------|----------|------------|-----------|
| Max operator Q-error  | TiDB Default |    90.93 | 4,560.41 | 710,991.63 | 29,826.90 |
| Max operator Q-error  | TiCard-GBR   |     5.49 | 1,668.87 | 16,466.88  | 1,050.21  |
| Max join-node Q-error | TiDB Default |    52.22 | 2,508.54 | 743,000.10 | 30,912.90 |
| Max join-node Q-error | TiCard-GBR   |     2.38 | 843.77   | 13,846.29  | 862.28    |

improves root-node tail errors dramatically, suggesting reduced error propagation at the plan output level.

## 4.4.4 Query-level worst-case (plan-aware) proxies

To further align evaluation with plan outcomes, we report query-level worst-case errors, which are more predictive of optimizer 'disasters' than operator-marginal averages. For each query execution, we compute (i) the maximum operator Q-error in the plan and (ii) the maximum join-node Q-error (since joins dominate plan sensitivity). Table 6 summarizes the distribution over the 53 test executions. These metrics are not a substitute for executing alternative plans, but they are plan-aware in the sense that they aggregate errors within a plan and emphasize worst-case nodes.

Plan-aware metrics such as P-error can correlate better with end-to-end outcomes in some settings; however, they typically require plan-cost models and/or enumerator access to define and evaluate alternative plans. Within our EXPLAIN-only scope, we therefore report root-node error (Table 5) and query-level worst-case proxies (Table 6) as partial evidence, and defer full planner-in-the-loop metrics to future work.

Table 7 Improvement factors over native TiDB (TiDB Q-error / TiCard Q-error).

| Metric   | TiCard-TabPFN   | TiCard-GBR   |
|----------|-----------------|--------------|
| P90      | 22.6 ×          | 22.9 ×       |
| P99      | 11.1 ×          | 10.0 ×       |
| Mean     | 23.1 ×          | 25.0 ×       |

Table 8 Setup/training and inference performance (test set inference over 1189 operator nodes).

| Model                         |   Setup/Training (s) |   Inference (s) | Samples/sec   |
|-------------------------------|----------------------|-----------------|---------------|
| TiCard-TabPFN (our)           |                 2.29 |          2.41   | 493           |
| TiCard-GBR (our)              |                 0.65 |          0.0022 | 550,931       |
| LiteCard (corr.)              |                 0.01 |          0.022  | 52,858        |
| MADE (NeuroCard arch., corr.) |                 5.87 |          0.38   | 3,097         |

## 4.4.5 TPC-H parameter holdout (template-stratified)

To address concerns about memorization on TPC-H due to repeated template executions, we run a template-stratified execution holdout: for each of the 22 templates, we hold out a subset of parameterizations as test and train on the remaining parameterizations (no template is entirely absent from training). On this holdout (96/25/29 executions for train/val/test), TiCard-GBR improves tail errors from P90 23.83 to 1.29 and P99 2,113.06 to 16.75 (median 1.01 to 1.05). This indicates that the correction signal generalizes across parameters rather than relying on exact query instances.

## 4.5 Improvement factors over the native optimizer

The largest gains occur on the same tail events where classical independence assumptions cause multiplicative error propagation (especially across joins). From a deployment standpoint, this is desirable: the primary goal is to reduce optimizer 'disasters' rather than to chase marginal improvements on already-correct estimates.

## 4.6 Efficiency: setup/training and inference

GBR provides very fast inference suitable for tight optimization loops, while TabPFN offers fast refresh without gradient retraining but at a much higher inference cost. For a plan with 50 operators, the node-level inference overhead is roughly ∼ 0.09ms (GBR) versus ∼ 101ms (TabPFN), suggesting different deployment positions: online optimization vs. offline analysis or amortized planning.

Table 9 Ablation on the test set (1189 operator samples).

| Model               | Mode       |    P90 | P99       |   Median | Mean     |
|---------------------|------------|--------|-----------|----------|----------|
| TiDB Default        | native     | 312.85 | 37,974.37 |   1.003  | 3,045.79 |
| TabPFN              | direct     |  16.64 | 2,280.87  |   1.0379 | 528.34   |
| TiCard-TabPFN (our) | correction |  13.82 | 3,416.50  |   1.0406 | 131.90   |
| GBR                 | direct     |  16.82 | 1,736.23  |   1.2956 | 106.84   |
| TiCard-GBR (our)    | correction |  13.69 | 3,812.02  |   1.3158 | 122.03   |

## 4.7 Ablation: direct prediction vs. correction

To isolate the effect of residual learning , we compare predicting cardinalities directly (log(1 + act)) with predicting correction targets (Eq. 2) under the same features and model family.

Direct prediction and correction are complementary: correction improves P90 for both models but can worsen the extreme tail (P99) relative to direct learning. This supports a deployability viewpoint: correction should be paired with conservative policies (e.g., join-only, blending) and validated for regressions.

## Zero-handling policy.

We implement a two-stage option (zero vs. non-zero classifier, then regression), but on our default split it does not improve and can slightly worsen P99 due to false positives (predicting zero when the true cardinality is non-zero). We therefore report results without the two-stage override by default and treat robust zero handling as future work; the primary stabilization used here is the log(1 + x ) target transform in Eq. 2.

## 4.8 Limitations of the offline study

Our evaluation is intentionally scoped:

- No end-to-end planner impact: we do not measure join-order changes or query latency improvements, which require integration into TiDB's plan enumeration and costing loop.
- Out-of-distribution patterns: novel operators or rare plan motifs can still produce large errors; uncertainty estimation and fallback policies are important for production use.
- Operator independence: we predict per-operator corrections independently; future work should model error propagation through the plan tree.

We emphasize that CE accuracy is necessary but not sufficient for end-to-end performance: plan quality also depends on the enumerator search space, cost model, and runtime behavior. To provide partial evidence without invasive integration, we include a plan-level proxy based on root-node cardinality (Table 5), which improves substantially and suggests reduced propagation at the plan output level [14].

## Optimizer evolution and model refresh.

Because TiCard conditions on the native estimate E ( q ), changes to the optimizer's estimator or cost model can shift the residual distribution. In practice this can be handled with lightweight refresh: GBR retrains in seconds on hundreds of executions, and TabPFN refreshes by replacing the in-context reference set. We recommend feature and model versioning tied to optimizer releases to detect and manage such drift.

## 5 Integration Roadmap: From Offline Correction to Online Use

To make TiCard actionable for ML-for-DB deployments, we outline an integration path that preserves optimizer control and enables staged rollout:

1. Label collection: select and execute representative queries (or sampled production queries) to collect EXPLAIN / EXPLAIN ANALYZE pairs.
2. Model training/refresh: train GBR periodically (seconds for hundreds of executions), or refresh TabPFN's reference set when drift is detected.
3. Inference placement: cache per-(sub)plan-node features and predictions; apply corrections at the cardinality estimation boundary, not by replacing the optimizer.
4. Safety controls: start with join-only correction and bounded blending; add uncertainty-based fallback once confidence estimates are available.
5. Measurement: evaluate plan stability (join-order changes), compilation overhead, and latency/throughput on shadow traffic before enabling corrections broadly.

This roadmap is consistent with the AI4DB principle that a learned component should be incremental , observable , and reversible .

## 5.1 Why we do not report end-to-end optimizer results in this paper

Reviewers often request end-to-end measurements (chosen plans, plan stability, and query latency). We agree these are important for deployment, but we intentionally do not include them here because they require a different artifact: an in-planner integration into TiDB's enumeration and costing loop. Concretely, end-to-end evaluation would require (i) injecting corrected cardinalities into the optimizer's cardinality interface for candidate plan enumeration, (ii) ensuring feature extraction and inference are available inside the planning critical path (including caching and concurrency), and (iii) defining safe rollback and gating policies. These changes are non-trivial engineering work and introduce confounders from the cost model, plan enumeration search space, and runtime effects [15]. As a result, end-to-end outcomes can be difficult to attribute cleanly to CE correction alone.

Instead, this paper focuses on a deployable building block and provides (a) strong operator-level tail reductions, (b) a plan-level proxy via root-node Q-error (Table 5) that partially reflects propagation effects [14], and (c) conservative policies (join-only, blending, fallback) intended to make a future in-planner integration low-risk. We

further specify a safe-injection layer (Section 5.3) to prevent semantic inconsistencies and reduce plan instability when integrating corrections into a cost model.

## 5.2 Minimal end-to-end study design (future work)

To directly address the end-to-end question without over-claiming, we outline a minimal study design that we plan to implement as future work:

- Shadow-mode planning: for each query, generate plans under (i) native CE and (ii) TiCard-corrected CE, log plan choices and estimated costs, but execute only the production plan initially.
- Attribution controls: keep the enumerator and cost model unchanged; vary only the cardinality interface to isolate the impact of CE corrections.
- Safety-first rollout: enable join-only correction and bounded blending first; expand scope only when plan stability and regressions are acceptable on shadow traffic.
- Metrics: plan changes (join order / operator choices), compile-time overhead, and execution-time distributions (median and tail latency), plus rollback triggers.

In TiDB, this study can be implemented with minimal operational risk by leveraging SQL Plan Management (SPM) to bind and unbind plans dynamically [16]. Concretely, once TiCard discovers a consistently better plan for a query (e.g., under join-only correction and bounded blending), the system can asynchronously bind that plan via SPM, monitor runtime/variance, and roll back by unbinding if regressions are detected. This mechanism provides a practical pathway to staged deployment without requiring immediate, invasive changes to the optimizer's plan enumeration logic.

This design complements our offline evidence and aligns with deployability goals: incremental adoption with clear observability and reversibility.

## 5.3 Safe injection: consistency constraints and stability guards

A common concern with per-operator learning is that naively injecting independent predictions into a cost model can increase plan instability or violate operator semantics (e.g., a filter producing more rows than its input). TiCard's design therefore assumes a safe-injection layer between model inference and the optimizer's cost model. This layer is lightweight (pure post-processing on estimated cardinalities) and can be applied before any planner-in-the-loop rollout.

## What can go wrong without guards.

Even if each node's prediction reduces marginal Q-error, a plan can still be harmed if (i) a few nodes receive extreme corrections, (ii) corrected cardinalities become inconsistent with operator semantics, or (iii) changes in a small subset of nodes flip join-order decisions. These are not model failures per se; they are integration risks.

## Guard families.

Werecommend four guard families that together address the above risks while preserving TiCard's 'augment, not replace' stance:

Table 10 Examples of semantics-aware constraints for safe injection (enforced on corrected cardinalities ̂ C ).

| Operator class                                                                        | Safe constraint (examples)                                                                                                                                                                                                                                                      |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Filter / Selection Limit Projection Aggregation (group-by) Inner join Left outer join | ̂ C out ← min( ̂ C out , ̂ C in ) ̂ C out ← min( ̂ C out , limit , ̂ C in ) ̂ C out ← ̂ C in (or ≤ if system allows dedup/other effects) ̂ C out ← min( ̂ C out , ̂ C in ) No general monotone bound holds; rely on bounded correction + join-only gating ̂ C out ← max( ̂ C out , ̂ C left ) |

- Semantic bounds (local): enforce operator-level invariants where they are unambiguous (Table 10).
- Scope gating (structural): apply corrections only to high-leverage operator classes first (e.g., joins) and fall back to native estimates elsewhere; expand scope only after shadow-mode evidence is satisfactory.
- Bounded correction (global): clamp correction factors to a safe range (e.g., exp(ˆ y ) ∈ [ c min , c max ]) or to a validation-calibrated quantile band, to prevent rare extreme outputs from dominating the cost model.
- Plan-change gating (decision-level): when corrected CE changes the chosen plan, require a minimum predicted benefit threshold and allow rollback (e.g., via TiDB SPM binding/unbinding) [16].

## Implementation sketch.

In a TiDB integration, these guards can be implemented as a deterministic postprocessing pass over the plan tree: apply bounded correction to each node, then enforce local semantic constraints where applicable (e.g., top-down for monotone operators such as filters/limits, and operator-specific rules for outer joins). This ensures corrected cardinalities are (a) non-negative, (b) within a validated range relative to the native estimate, and (c) consistent with basic operator semantics before the cost model consumes them.

## What we claim.

Our offline results evaluate the prediction quality of residual correction and conservative policies such as join-only and blending. The safe-injection constraints above are proposed integration mechanisms to prevent pathological planner behavior; they do not require new training data and are compatible with shadow-mode rollout.

## 6 Related Work

## 6.1 Traditional cardinality estimation

Histograms and independence-based estimators are the dominant classical approach [3, 17, 18], but cannot capture cross-table correlations. Sampling and sketching methods improve robustness at the cost of runtime overhead or limited query expressiveness [1921].

## 6.2 Learned and hybrid estimation

Query-driven learned estimators map query features to cardinalities (e.g., MSCN) [5], while data-driven approaches learn joint distributions (e.g., NeuroCard/DeepDB/BayesCard) [6, 22, 23]. Hybrid approaches and correction methods aim to combine learned models with native estimators [24-26]. LiteCard learns many lightweight local models keyed by repeated patterns with hierarchical fallback [12]. TiCard shares a correction-first philosophy but emphasizes EXPLAIN-only features and deployability-oriented model choices (GBR vs. in-context TabPFN).

## Transfer and benchmarks.

Recent work studies transfer and pretraining for CE (e.g., PRICE) [27] and highlights evaluation challenges and generalization gaps across schemas/workloads (e.g., CardBench) [28]. These directions are highly relevant to deployability, but they typically require richer query/predicate encodings or access to database contents/statistics beyond the EXPLAIN-only interface we target. As a result, we treat them as complementary baselines and discuss integration opportunities rather than claiming direct empirical comparability in our EXPLAIN-only, low-intrusion scope.

## Deployable hybrid/data-driven models.

Several recent estimators emphasize end-to-end integration and practical overhead, including FactorJoin (hybrid PGM framework for join queries) [29] and workloador data-aware SPN-based approaches [30]. Other work explores decoupled predicate modulation for joins [31] or learning under imperfect workloads [32]. These methods are highly relevant comparators from a deployability perspective; however, they typically require access to base-table data/statistics and query predicates, and their reported endto-end benefits arise from system-specific integration into the planner and/or execution feedback loops. TiCard targets a different deployment surface: EXPLAIN-only features and correction policies that can be inserted conservatively as an augmentation layer, with a roadmap for planner-in-the-loop evaluation.

## 6.3 ML for query optimization

Learned optimizers and cost models apply ML to plan selection or runtime prediction [7, 33, 34]. TiCard is complementary: it targets a narrow, deployable interface (cardinality corrections) that can be combined with other optimizer components.

## 7 Conclusion

TiCard reframes learned CE for deployability: instead of replacing the optimizer, it learns residual corrections over native estimates using EXPLAIN-only features. In an offline, operator-level study on TiDB with TPC-H and JOB under a low-trace regime, TiCard reduces tail errors dramatically (P90 from 312.85 to 13.69 with GBR; P99 from 37,974.37 to 3,416.50 with TabPFN) while enabling conservative integration policies such as join-only correction. We provide an integration roadmap to connect this offline evidence to online optimizer use, and view TiCard as a practical AI4DB component for incremental adoption in production databases.

Acknowledgements. This work was supported by PingCAP, the company behind TiDB.

## Declarations

Funding Not applicable.

Conflict of interest/Competing interests The author is employed by PingCAP. Ethics approval and consent to participate Not applicable. Consent for publication Not applicable.

Data availability Query plans used in the offline evaluation are cached under query plans/ in the accompanying repository. Large raw datasets (TPC-H tables, IMDB) follow their original licenses and are not redistributed.

Materials availability Not applicable.

Code availability Source code and scripts are available at https://github.com/ Icemap/TiCard.

Author contribution Qizhi Wang: conceptualization, implementation, evaluation, and writing.

## References

- [1] Selinger, P.G., Astrahan, M.M., Chamberlin, D.D., Lorie, R.A., Price, T.G.: Access path selection in a relational database management system. In: Proceedings of the 1979 ACM SIGMOD International Conference on Management of Data. SIGMOD '79, pp. 23-34 (1979). https://doi.org/10.1145/582095.582099
- [2] Leis, V., Gubichev, A., Mirchev, A., Boncz, P., Kemper, A., Neumann, T.: How good are query optimizers, really? Proceedings of the VLDB Endowment 9 (3), 204-215 (2015) https://doi.org/10.14778/2850583.2850594
- [3] Ioannidis, Y.: The history of histograms (abridged). Proceedings of the VLDB Endowment, 19-30 (2003) https://doi.org/10.1016/B978-012722442-8/50011-2
- [4] Leis, V., Radke, B., Gubichev, A., Kemper, A., Neumann, T.: Query optimization through the looking glass, and what we found running the join order benchmark. The VLDB Journal 27 (5), 643-668 (2018) https://doi.org/10.1007/ s00778-017-0480-7

| [5]   | Kipf, A., Kipf, T., Radke, B., Leis, V., Boncz, P., Kemper, A.: Learned cardinalities: Estimating correlated joins with deep learning. In: 9th Bien- nial Conference on Innovative Data Systems Research. CIDR '19 (2019). http://cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf   |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [6]   | Yang, Z., Kamsetty, A., Luan, S., Liang, E., Duan, Y., Chen, X., Stoica, I.: Neurocard: One cardinality estimator for all tables. Proceedings of the VLDB Endowment 14 (1), 61-73 (2020) https://doi.org/10.14778/3421424.3421427                                                   |
| [7]   | Marcus, R., Negi, P., Mao, H., Tatbul, N., Alizadeh, M., Kraska, T.: Bao: Making learned query optimization practical. In: Proceedings of the 2021 International Conference on Management of Data. SIGMOD '21, pp. 1275-1288 (2021). https: //doi.org/10.1145/3448016.3452838       |
| [8]   | Moerkotte, G., Neumann, T., Steidl, G.: Preventing bad plans by bounding the impact of cardinality estimation errors. Proceedings of the VLDB Endowment 2 (1), 982-993 (2009) https://doi.org/10.14778/1687627.1687738                                                              |
| [9]   | Friedman, J.H.: Greedy function approximation: A gradient boosting machine. Annals of Statistics 29 (5), 1189-1232 (2001) https://doi.org/10.1214/aos/ 1013203451                                                                                                                   |
| [10]  | Hollmann, N., M¨ uller, S., Eggensperger, K., Hutter, F.: Tabpfn: A transformer that solves small tabular classification problems in a second. Nature 637 , 319-326 (2025) https://doi.org/10.1038/s41586-024-08328-6 . Also appeared in ICLR 2023                                  |
| [11]  | Transaction Processing Performance Council: TPC-H Benchmark Specification. Accessed: 2025-10-15 (2024). http://www.tpc.org/tpch/                                                                                                                                                    |
| [12]  | Yi, Z., Abu-el-Haija, S., Wang, Y., Vemparala, T., Chronis, Y., Gan, Y., Burrows, M., Binnig, C., Perozzi, B., Marcus, R., Ozcan, F.: Is it Bigger than a Breadbox: Efficient Cardinality Estimation for Real World Workloads (2025). https://arxiv. org/abs/2510.03386             |
| [13]  | Germain, M., Gregor, K., Murray, I., Larochelle, H.: Made: Masked autoen- coder for distribution estimation. In: Proceedings of the 32nd Interna- tional Conference on Machine Learning. ICML '15, pp. 881-889 (2015). http://proceedings.mlr.press/v37/germain15.pdf               |
| [14]  | Ioannidis, Y.E., Christodoulakis, S.: On the propagation of errors in the size of join results. In: Proceedings of the 1991 ACM SIGMOD International Conference on Management of Data. SIGMOD '91, pp. 268-277 (1991). https://doi.org/10. 1145/115790.115835                       |
| [15]  | Lan, H., Bao, Z., Peng, Y.: A survey on advancing the dbms query optimizer: Cardinality estimation, cost model, and plan enumeration. Data Science and Engineering 6 , 86-101 (2021) https://doi.org/10.1007/s41019-020-00149-7                                                     |

| [16]   | PingCAP: TiDB SQL Plan Management (SPM). Accessed: 2025-12-16 (2025). https://docs.pingcap.com/tidb/stable/sql-plan-management/                                                                                                                                                                                      |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [17]   | Poosala, V., Ioannidis, Y.E.: Selectivity estimation without the attribute value independence assumption. In: Proceedings of the 1997 International Conference on Very Large Data Bases. VLDB '96, pp. 486-495 (1996). http://www.vldb.org/conf/1997/P486.PDF                                                        |
| [18]   | Bruno, N., Chaudhuri, S., Gravano, L.: Stholes: A multidimensional workload- aware histogram. In: Proceedings of the 2001 ACM SIGMOD International Conference on Management of Data. SIGMOD '01, pp. 211-222 (2001). https: //doi.org/10.1145/375663.375686                                                          |
| [19]   | Haas, P.J., Naughton, J.F., Seshadri, S., Stokes, L.: Sampling-based estimation of the number of distinct values of an attribute. In: Proceedings of the 21st International Conference on Very Large Data Bases. VLDB '95, pp. 311-322 (1995). http://www.vldb.org/conf/1995/P311.PDF                                |
| [20]   | Chaudhuri, S., Motwani, R., Narasayya, V.: Random sampling for histogram construction: How much is enough? In: Proceedings of the 1998 ACM SIGMOD International Conference on Management of Data. SIGMOD '98, pp. 436-447 (1998). https://doi.org/10.1145/276304.276343                                              |
| [21]   | Cormode, G., Muthukrishnan, S.: An improved data stream summary: The count- min sketch and its applications. Journal of Algorithms 55 (1), 58-75 (2005) https: //doi.org/10.1016/j.jalgor.2003.12.001                                                                                                                |
| [22]   | Hilprecht, B., Schmidt, A., Kulessa, M., Molina, A., Kersting, K., Binnig, C.: Deepdb: Learn from data, not from queries! Proceedings of the VLDB Endowment 13 (7), 992-1005 (2020) https://doi.org/10.14778/3384345.3384349                                                                                         |
| [23]   | Kipf, A., Vorona, D., M¨ uller, J., Kipf, T., Radke, B., Leis, V., Boncz, P., Kemper, A., Neumann, T.: Estimating cardinalities with deep sketches. In: Proceedings of the 2019 ACM SIGMOD International Conference on Management of Data. SIGMOD '19, pp. 1937-1940 (2019). https://doi.org/10.1145/3299869.3320240 |
| [24]   | Wang, J., Chai, C., Liu, J., Li, G.: Face: A normalizing flow based cardinality estimator. Proceedings of the VLDB Endowment 15 (1), 72-84 (2021) https: //doi.org/10.14778/3485450.3485458                                                                                                                          |
| [25]   | Wu, P., Cong, G.: A unified deep model of learning from both data and queries for cardinality estimation. In: Proceedings of the 2021 International Conference on Management of Data. SIGMOD '21, pp. 2009-2022 (2021). https://doi.org/10. 1145/3448016.3452830                                                     |

- [26] Negi, P., Wu, Z., Kipf, A., Tatbul, N., Marcus, R., Madden, S., Kraska, T.,

Alizadeh, M.: Robust query driven cardinality estimation under changing workloads. Proceedings of the VLDB Endowment 16 (6), 1520-1533 (2023) https: //doi.org/10.14778/3583140.3583164

- [27] Zeng, T., Lan, J., Ma, J., Wei, W., Zhu, R., Li, P., Ding, B., Lian, D., Wei, Z., Zhou, J.: PRICE: A Pretrained Model for Cross-Database Cardinality Estimation (2024). https://arxiv.org/abs/2406.01027
- [28] Chronis, Y., Wang, Y., Gan, Y., Abu-El-Haija, S., Lin, C., Binnig, C., ¨ Ozcan, F.: CardBench: A Benchmark for Learned Cardinality Estimation in Relational Databases (2024). https://arxiv.org/abs/2408.16170
- [29] Wu, Z., Negi, P., Alizadeh, M., Kraska, T., Madden, S.: Factorjoin: A new cardinality estimation framework for join queries. Proceedings of the ACM on Management of Data (2023). SIGMOD/PACMMOD
- [30] Liu, J., Fan, J., Liu, T., Zeng, K., Wang, J., Liu, Q., Ye, T., Tang, N.: A Unified Model for Cardinality Estimation by Learning from Data and Queries via SumProduct Networks (2025). https://arxiv.org/abs/2505.08318
- [31] Zhang, K., Wang, H., Li, Z., Lu, Y., Li, Y., Yan, Y., Guan, Y.: DistJoin: A Decoupled Join Cardinality Estimator based on Adaptive Neural Predicate Modulation (2025). https://arxiv.org/abs/2503.08994
- [32] Wu, P., Kang, R., Zhang, T., Chen, J., Marcus, R., Ives, Z.G.: Data-Agnostic Cardinality Learning from Imperfect Workloads (2025). https://arxiv.org/abs/ 2506.16007
- [33] Yang, Z., Liang, E., Kamsetty, A., Wu, C., Duan, Y., Chen, X., Abbeel, P., Hellerstein, J.M., Krishnan, S., Stoica, I.: Balsa: Learning a query optimizer without expert demonstrations. In: Proceedings of the 2022 International Conference on Management of Data. SIGMOD '22, pp. 931-944 (2022). https: //doi.org/10.1145/3514221.3517885
- [34] Pathak, U., Mankodi, A.: Redefining Cost Estimation in Database Systems: The Role of Execution Plan Features and Machine Learning (2025). https://arxiv.org/ abs/2510.05612