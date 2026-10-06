# From Reaching Motion to Clinical Scores

### Feature engineering vs. deep representation learning

**An MSc thesis in Computer Science by Itay Mizikov.**

Can video-derived 3D reaching kinematics estimate established clinical scores after stroke—and how do three different modeling pipelines compare?

This study compares **feature engineering**, **reconstruction-based self-supervised learning**, and **supervised deep learning** under a shared participant-grouped evaluation. It is a step toward video-based quantitative assessment, not a validated clinical product.

[Research](#research) · [Data](#data-without-disclosing-patient-records) · [Models](#three-modeling-pipelines) · [Results](#main-results) · [Tools](#tools-and-engineering) · [Privacy](#privacy-and-release-scope)

> **At a glance:** 36 stroke participants · 13 controls for secondary classification · 3 modeling paradigms · 5 participant-grouped outer folds.
>
> **Main finding:** Feature engineering had the strongest observed score-estimation performance among the three canonical pipelines. The evaluated deep-learning pipelines did not demonstrate an advantage over it. Timing and session structure provided a strong reference.

## Research

**Research question:** How do feature-engineering and deep representation-learning pipelines compare in their ability to assess post-stroke motor impairment from 3D reaching kinematics?

The main task is **concurrent clinical-score estimation**:

- **FMA-UE (0–66):** Fugl–Meyer Assessment of the Upper Extremity, an impairment-level outcome.
- **ARAT (0–57):** Action Research Arm Test, an activity-level outcome.

Clinical scores provide interpretable reference outcomes but limited detail about how movement is produced. This work evaluates approaches to estimating those scores; developing a complementary movement measure and establishing sensitivity to recovery require further research.

## Data, without disclosing patient records

Only study-level counts and aggregate model results are presented here.

| Main score-estimation task | Secondary cohort-discrimination task |
|---|---|
| 36 stroke participants | 36 stroke participants + 13 healthy controls |
| 52 participant-visits for FMA-UE; 51 for ARAT | One pooled prediction per participant |
| Separately administered clinical assessments | Healthy-versus-stroke label; not a diagnostic validation |

<details>
<summary><strong>Explore the acquisition and processing workflow</strong></summary>

1. **Task:** seated reaching toward five vertical targets, adopted from the parent study protocol. The clinical scales were administered separately; the task is not an abbreviated FMA-UE or ARAT test.
2. **Capture:** two synchronized RGB cameras at 150 fps and 1280 × 1024 resolution.
3. **Reconstruction:** OpenPose 2D keypoints, stereo calibration, and triangulation to 3D arm/trunk trajectories.
4. **Segmentation:** manual reach boundaries in MATLAB, using video, wrist velocity, and reconstructed trajectories. A reach runs from movement onset to target contact, excluding the return movement.
5. **Modeling:** each pipeline preprocesses and represents the shared reaching recordings, then aggregates information to the clinical evaluation unit.

Repeated reaches and sessions are nested within participant-visits. They are **not independent patients** or independently scored clinical assessments.

</details>

## Three modeling pipelines

| Pipeline | Representation | Canonical score estimator |
|---|---|---|
| **Feature engineering** | 490 reach descriptors covering timing, speed, geometry, smoothness, joint motion, and trunk compensation; summarized by target within visit | Training-fold feature selection and tuned Ridge regression; rod predictions averaged to visits |
| **Self-supervised learning (SSL)** | Temporal convolutional autoencoder trained to reconstruct motion without clinical labels; corrected 64-dimensional reach representation | Frozen embeddings aggregated to visits, followed by fixed Ridge regression |
| **Supervised deep learning** | Spatial-temporal graph convolutional network (ST-GCN), trained with clinical labels and hierarchical multiple-instance learning | Canonical **Track P / Path B**: aggregated frozen embeddings and fixed Ridge probes |

These are comparisons of **complete pipelines**: preprocessing, aggregation, estimators, and tuning also differ. Feature engineering also uses a learned estimator.

<details>
<summary><strong>Explore modeling choices and supporting experiments</strong></summary>

- **Supervision:** reconstruction-based learning versus clinical-score supervision, including a temporal convolutional control model.
- **Architecture and prediction route:** graph versus temporal models; end-to-end prediction versus probes on frozen embeddings.
- **Aggregation and input:** session/target pooling, input-length sensitivity, and feature parsimony.
- **Robustness and extensions:** training-cohort-size analysis, hybrid models, exploratory severity prediction, and acquisition-structure diagnostics for classification.

The supervised **Track C** uses coordinates; **Track P** adds derived channels, target conditioning, and training augmentation. **Path A** predicts through the end-to-end head; **Path B** fits a downstream estimator on frozen embeddings. The table above reports the canonical Track-P Path-B route—not the temporal control or another ablation.

These analyses retain their distinct descriptive, diagnostic, sensitivity, robustness, or exploratory roles; they are not all confirmatory tests.

</details>

## Main results

Out-of-fold performance on the aligned stroke participant-visits. **MAE** is mean absolute error in clinical-score points (lower is better); **ρ** is Spearman rank correlation (higher is better).

| Pipeline / reference | FMA-UE MAE ↓ | FMA-UE ρ ↑ | ARAT MAE ↓ | ARAT ρ ↑ |
|---|---:|---:|---:|---:|
| Feature engineering | 10.09 | 0.560 | 13.33 | 0.624 |
| Self-supervised, corrected | 17.38 | 0.294 | 24.47 | 0.360 |
| Supervised ST-GCN, Track P / Path B | 12.01 | 0.297 | 13.87 | 0.409 |
| *Duration reference* | 9.88 | 0.628 | 13.30 | 0.585 |
| *Outer-training mean reference* | 13.08 | −0.209 | 17.62 | −0.097 |

**What this means:** observed ordering is not blanket statistical superiority. The FMA-UE MAE difference between SSL and feature engineering was **7.28 points** (paired 95% CI **2.37–13.71**; Holm-adjusted **p = 0.027**). The corresponding ARAT difference did not remain significant after correction; feature-engineering versus canonical supervised MAE intervals included zero on both outcomes.

Timing references contextualize performance; they are not a fourth representation paradigm or a pure measure of impairment. The prespecified test did not support lower error for the canonical supervised pipeline than the duration reference.

<details>
<summary><strong>Explore the evaluation design and interpretation</strong></summary>

- **Shared evaluation:** five frozen outer folds grouped by participant. All visits from one person remain in the same fold; comparisons use identical held-out observations for each outcome.
- **Model selection:** pipeline-specific selection uses outer-training data, with nested clinical selection for canonical feature-engineering regression. Tuning budgets and procedures are not identical across pipelines.
- **Complementary metrics:** MAE, rank correlation, pooled out-of-fold R², and calibration. Rank association alone does not establish accurate prediction on the clinical-score scale: corrected SSL had FMA-UE ρ = 0.294 but R² = −1.98.
- **Uncertainty:** 2,000 participant-cluster bootstrap replicates and 10,000 paired participant-cluster permutations, with within-family multiplicity correction. Resampling uses frozen predictions, not repeated model fitting. Regression metrics weight labelled visits equally.
- **Confirmatory scope:** the planned hypothesis required both lower supervised error than duration and improvement from adding that representation to duration. The first condition was tested and not supported; the second, combined-model arm was not executed. Other hybrid experiments do not replace that arm.
- **Generalization limits:** a small, single-source cohort without external validation; no direct optoelectronic validation of reconstructed motion. Segmentation agreement was checked only on a limited subset. A global, outcome-independent reach-length QC threshold was set before cross-validation rather than fold-locally.

</details>

<details>
<summary><strong>Explore secondary healthy-versus-stroke discrimination</strong></summary>

| Pipeline / reference | Participant ROC-AUC | 95% bootstrap CI |
|---|---:|---|
| Feature engineering | 0.976 | 0.932–1.000 |
| Self-supervised, corrected | 0.850 | 0.712–0.953 |
| Supervised ST-GCN, Track P / Path B | 0.891 | 0.782–0.969 |
| Duration reference | 0.944 | 0.872–0.991 |
| Protocol reference | 0.911 | 0.821–0.981 |

Timing and protocol alone were highly discriminative. Acquisition differences between groups therefore limit interpretation: **high AUC is not evidence of a stroke-specific movement signature or clinical diagnostic readiness**. Corrected SSL classification is post hoc and descriptive. No frozen cross-paradigm AUC contrast survived Holm adjustment; a separate mechanistic Track-C versus feature-engineering contrast did.

</details>

## Tools and engineering

| Tool | Role in the research |
|---|---|
| **Python · NumPy · pandas · SciPy** | Kinematic processing, feature construction, analysis, and statistical evaluation |
| **scikit-learn** | Classical estimators, preprocessing, grouped cross-validation, and clinical probes |
| **PyTorch** | Temporal autoencoder, ST-GCN, temporal control, and hierarchical deep-learning training |
| **OpenPose** | 2D pose estimation before stereo reconstruction |
| **MATLAB** | Manual reach segmentation and inspection of trajectories and wrist velocity |
| **SLURM · GPU computing** | Scheduled training and experiment execution |
| **Matplotlib · LaTeX · Git** | Result visualization, thesis production, and version control |

The technical contribution is the adaptation and comparative evaluation of these pipelines for hierarchical clinical-score estimation, supported by reference models and targeted diagnostics—not a claim of a new neural architecture.

## Privacy and release scope

**This is a public research showcase, not the private research repository or a runnable reproduction package.** All numerical results above are aggregate summaries from the thesis, not individual records.

This release contains no participant videos or photographs, clinical records, individual scores or predictions, identifiers, visit dates, motion trajectories, embeddings, trained weights, notebooks, logs, or private Git history. The full thesis and research code are not included in this release.

The repository intentionally starts with newly authored documentation only. Reproducing the reported experiments requires the restricted research data and implementation, neither of which is distributed here. Any future code or asset release requires a separate privacy and permissions review; `.gitignore` alone is not a privacy guarantee.
