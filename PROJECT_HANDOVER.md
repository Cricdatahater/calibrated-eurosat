# Calibrated EuroSAT — Project Handover

Last updated: 23 September 2026

## Objective

Build a reproducible computer-vision study on the original EuroSAT RGB dataset that compares:

1. handcrafted statistical image features;
2. frozen ImageNet-pretrained CNN representations;
3. a fine-tuned ResNet18;
4. predictive uncertainty and probability calibration.

The portfolio goal is to show a coherent progression from statistical modelling to modern computer vision, with fixed data splits, reproducible evaluation, model comparisons, and honest limitations.

## Current status

### Completed: Stage 1 — data audit and EDA

- Loaded the official EuroSAT RGB dataset with Torchvision.
- Verified 27,000 RGB images, 10 classes, and 64 × 64 resolution.
- Recorded the class-to-index mapping.
- Measured class frequencies and found a modest maximum-to-minimum imbalance ratio of 1.5:1.
- Created and saved representative class examples.
- Extracted per-image handcrafted features:
  - RGB channel means;
  - RGB channel standard deviations;
  - brightness;
  - contrast;
  - grayscale entropy.
- Verified that the feature table contains no missing or non-finite values.
- Checked exact file duplicates with MD5; none were found.
- Created a fixed stratified 70/15/15 train/validation/test split using seed 42.
- Verified that the three partitions are mutually disjoint and collectively contain all observations.
- Saved both numeric split indices and portable CSV manifests.
- Saved the split checksum.
- Generated class-distribution, class-example, brightness, contrast, entropy, and feature-correlation figures.
- Added written EDA findings, data provenance, and the geographic-generalization limitation to the notebook.

### Completed: Stage 2 — statistical baseline

- Reused the fixed train/validation/test manifests without regenerating the split.
- Established a most-frequent-class dummy benchmark.
- Trained standardized multinomial logistic-regression models across a validation-tuned regularization grid.
- Selected `C = 100` using validation macro-F1, then froze the configuration.
- Refit the selected pipeline on training plus validation data and evaluated the test set once.
- Saved validation and test metrics, per-class reports, predictions, coefficients, confusion matrices, calibration metrics, and the experiment summary.
- Recorded the statistical model as an interpretable baseline rather than a final remote-sensing solution.

Final statistical-baseline performance:

| Partition | Accuracy | Macro-F1 | Log loss | Multiclass Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Validation | 0.740 | 0.727 | 0.755 | 0.366 | 0.017 |
| Test | 0.748 | 0.735 | 0.737 | 0.360 | 0.025 |

### Completed: Stage 3 — frozen ResNet18 embeddings

- Enabled a Kaggle GPU environment and loaded the official EuroSAT RGB data.
- Reused the fixed split manifests and generated 512-dimensional embeddings
  from an ImageNet-pretrained frozen ResNet18.
- Searched six logistic-regression regularization values using the training and
  validation partitions only.
- Selected and froze `C = 0.01` using the declared primary criterion of
  validation macro-F1.
- Kept the test partition locked during model selection.
- Refit the selected pipeline on the combined training and validation
  embeddings and evaluated the test partition once.
- Saved the Kaggle notebook, validation search, test classification report,
  confusion matrix, experiment summary, and baseline comparison.

Frozen ResNet18 validation selection:

| C | Accuracy | Macro-F1 | Log loss | Multiclass Brier | ECE |
|---:|---:|---:|---:|---:|---:|
| **0.01** | **0.939** | **0.937** | **0.183** | **0.091** | **0.019** |
| 0.001 | 0.933 | 0.931 | 0.258 | 0.117 | 0.080 |
| 0.1 | 0.931 | 0.928 | 0.212 | 0.103 | 0.023 |
| 1 | 0.920 | 0.917 | 0.394 | 0.131 | 0.055 |
| 10 | 0.916 | 0.913 | 0.852 | 0.151 | 0.072 |
| 100 | 0.916 | 0.913 | 1.397 | 0.158 | 0.077 |

Final frozen-ResNet18 performance:

| Partition | Accuracy | Macro-F1 | Log loss | Multiclass Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Validation | 0.939 | 0.937 | 0.183 | 0.091 | 0.019 |
| Test | 0.949 | 0.947 | 0.160 | 0.079 | 0.021 |

Relative to the handcrafted baseline, test accuracy improved by 20.2 percentage
points and macro-F1 by 21.2 percentage points. PermanentCrop, River, and Highway
remain the weakest frozen-CNN classes by F1; SeaLake, Residential, and
Industrial are strongest.

## Verified dataset facts

- Dataset: EuroSAT RGB
- Source: official EuroSAT release by Helber et al.
- DOI: `10.5281/zenodo.7711097`
- Samples: 27,000
- Classes: 10
- Image format: RGB
- Resolution: 64 × 64

Class sizes:

| Class | Images |
|---|---:|
| AnnualCrop | 3,000 |
| Forest | 3,000 |
| HerbaceousVegetation | 3,000 |
| Highway | 2,500 |
| Industrial | 2,500 |
| Pasture | 2,000 |
| PermanentCrop | 2,500 |
| Residential | 3,000 |
| River | 2,500 |
| SeaLake | 3,000 |

## Fixed experimental split

| Partition | Images | Fraction |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Test | 4,050 | 15% |

- Method: stratified image-level split
- Random seed: 42
- Split checksum: `f97c4ec9a27435a932662d5a8b707255`
- Index file: `data/splits/split_indices.json`
- Portable manifests:
  - `data/splits/train.csv`
  - `data/splits/validation.csv`
  - `data/splits/test.csv`

Do not regenerate or alter this split to improve model results.

## Dataset normalization statistics

Statistics were calculated over all EuroSAT RGB pixels scaled to `[0, 1]`:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| Red | 0.344376 | 0.202661 |
| Green | 0.380291 | 0.136897 |
| Blue | 0.407770 | 0.115550 |

For final experimental discipline, normalization used by a model should be estimated from the training split only. For ImageNet-pretrained models, compare or justify use of the normalization bundled with the selected pretrained weights.

## Initial EDA findings

- The imbalance is modest; weighted loss or oversampling is not initially justified. Report macro-F1 in addition to accuracy.
- Forest and SeaLake are generally darker and lower-contrast than built-environment classes.
- Industrial has the highest average brightness, contrast, and entropy, indicating greater local visual complexity.
- SeaLake has the lowest entropy and contrast, consistent with comparatively uniform image content.
- Likely difficult class pairs include:
  - AnnualCrop vs PermanentCrop;
  - Highway vs River;
  - Industrial vs Residential.
- Handcrafted colour and texture features provide partial class separation but overlap substantially, making them suitable for a statistical baseline rather than a complete solution.
- No exact duplicate files were found. Near-duplicate or geographically related patches have not been ruled out.

## Important methodological limitation

The evaluation uses a fixed stratified image-level split. Geographic grouping metadata is not incorporated. Results therefore estimate performance on held-out EuroSAT images and must not be described as generalization to unseen geographic regions.

## Environment recorded by the audit

- Python: 3.12.7
- PyTorch: 2.14.0+cpu
- Torchvision: 0.29.0+cpu
- NumPy: 1.26.4
- pandas: 2.2.2
- Platform: Windows 11

The NumPy version is deliberately pinned below 2 because the original Anaconda environment contained compiled SciPy ecosystem packages that were incompatible with NumPy 2.5.3.

## Key artifacts

- Notebooks:
  - `Notebooks/01_data_audit.ipynb`
  - `Notebooks/02_statistical_baseline.ipynb`
  - `Notebooks/03_frozen_resnet18_embeddings.ipynb`
  - `Notebooks/04_finetuned_resnet18.ipynb`
  - `Notebooks/05_temperature_scaling.ipynb`
- Machine-readable audit: `reports/data_audit.json`
- Image-level feature table: `reports/image_statistics.csv`
- Duplicate report: `reports/duplicate_report.csv`
- Split balance report: `reports/split_class_distribution.csv`
- Figures: `reports/figures/`
- Fixed split definitions: `data/splits/`
- Statistical results: `results/statistical_baseline/`
- Frozen ResNet18 validation search and audit: `results/frozen_resnet18/`
- Fine-tuned ResNet18 histories, predictions, metrics, and figures:
  `results/fine_tuned_resnet18/`
- Temperature-scaling logits, metrics, predictions, bootstrap intervals, and
  figures: `results/temperature_scaling/`
- Fitted statistical pipeline (local, Git-ignored): `models/statistical_logistic_regression.joblib`
- Fine-tuned neural checkpoints (local, Git-ignored):
  `models/fine_tuned_resnet18/`

## Completed: Stage 5 — temperature scaling

Stage 5 calibrates the locked Stage 4 Candidate A checkpoint without retraining
or changing the classifier. The complete method and audit record are in
`STAGE5_PROTOCOL.md`; the archival notebook is
`Notebooks/05_temperature_scaling.ipynb`.

### Locked inputs and method

- Checkpoint SHA-256:
  `3f6a701aca082fa675ba229ddfbae0139346e3cbff84ea02ff0617ab73c5d657`.
- Split checksum: `f97c4ec9a27435a932662d5a8b707255`.
- Deterministic `IMAGENET1K_V1` evaluation transforms were retained.
- Batch size 64 and CUDA mixed-precision inference reproduce the Stage 4 path.
- One positive scalar temperature was fitted by minimizing validation-only
  multiclass negative log-likelihood.
- The fitted temperature `1.8226799937` was frozen before test metrics were
  calculated.
- All 4,050 validation and 4,050 test observations have saved logits and unique
  dataset indices.

### Stage 5 results

| Partition | Calibration | Accuracy | Macro-F1 | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| Validation | Before | 0.978519 | 0.977565 | 0.088523 | 0.036730 | 0.013305 |
| Validation | After | 0.978519 | 0.977565 | 0.069421 | 0.035502 | 0.005596 |
| Test | Before | 0.979506 | 0.978540 | 0.074072 | 0.033437 | 0.012213 |
| Test | After | 0.979506 | 0.978540 | 0.060098 | 0.031412 | 0.002788 |

The positive temperature preserves class ordering; every test prediction is
unchanged. Paired 2,000-resample bootstrap differences (`calibrated -
uncalibrated`) were:

- log loss: `-0.013974`, 95% interval `[-0.022496, -0.006348]`;
- multiclass Brier score: `-0.002026`, 95% interval
  `[-0.003563, -0.000492]`;
- 15-bin ECE: `-0.009425`, 95% interval `[-0.011544, -0.003110]`.

All intervals exclude zero on the fixed image-level test set. They do not
establish calibration under geographic or other distribution shift.

### Stage 5 audit status — 21 September 2026

- The saved Kaggle version completed successfully on a Tesla T4.
- The checkpoint and split checksums match the locked records.
- The uncalibrated test accuracy and macro-F1 reproduce Stage 4 exactly.
- The downloaded artifact ZIP passed extraction and numerical consistency
  checks.
- The test prediction table has 4,050 unique rows, 27 columns, and no missing
  values.
- Trained checkpoints remain local and Git-ignored.

## Completed: Stage 4 — fine-tuned ResNet18

The cleaned archival notebook is `Notebooks/04_finetuned_resnet18.ipynb`.
It initializes from `IMAGENET1K_V1`, preserves the fixed manifests, compares
two registered fine-tuning candidates using validation data only, and evaluates
the selected checkpoint once on test.

### Stage 4 registration status — 20 September 2026

- The complete pre-training protocol is recorded in `STAGE4_PROTOCOL.md`.
- Stage 4 training and its locked test evaluation are complete.
- The Kaggle notebook `04-finetuned-resnet18.ipynb` has been created and its
  setup/data-pipeline cells have run successfully on a Tesla T4 GPU.
- The saved split was verified at 18,900/4,050/4,050 samples with checksum
  `f97c4ec9a27435a932662d5a8b707255`; all 27,000 manifest paths were present.
- The train, validation, and test loaders passed a 64-image smoke test with
  tensor shape `[64, 3, 224, 224]` and labels in the expected 10-class range.
- The registered three-epoch head-only warm-up completed successfully. Only
  `fc.weight` and `fc.bias` (5,130 parameters) were trainable.
- Warm-up validation macro-F1 improved from `0.910109` to `0.925971`; validation
  log loss improved from `0.289788` to `0.219391`. Epoch 3 is the selected
  shared warm-up checkpoint for both fine-tuning candidates.
- Candidate A (`layer4` + head) completed all 12 epochs. Its selected epoch is
  12 with validation macro-F1 `0.977565` and log loss `0.088523`.
- Candidate B (full fine-tuning) stopped after epoch 11 with patience 4. Its
  selected epoch is 7 with validation macro-F1 `0.977003` and log loss
  `0.074390`.
- Candidate A is the locked Stage 4 selection because the predeclared primary
  criterion is higher validation macro-F1. Candidate B's lower log loss is only
  a tie-breaker and the macro-F1 values are not tied.
- Candidate A was frozen before test evaluation with checkpoint SHA-256
  `3f6a701aca082fa675ba229ddfbae0139346e3cbff84ea02ff0617ab73c5d657`.
- The one-time Stage 4 test evaluation is complete: accuracy `0.979506`,
  macro-F1 `0.978540`, log loss `0.074072`, multiclass Brier score `0.033437`,
  and 15-bin top-label ECE `0.012213`.
- Local audit of the downloaded notebook confirmed that model selection was
  frozen before the test loop and that the test set was evaluated once.
- The downloaded notebook was cleaned and renamed to
  `Notebooks/04_finetuned_resnet18.ipynb`. The saved diagnostic error and
  duplicate recovery cells were removed, verified Candidate A/B histories were
  consolidated into the archival record, and explanatory Markdown was added.
- The registered project protocol compares partial (`layer4` + head) and full
  fine-tuning after a shared three-epoch head warm-up.
- Candidate selection is validation-only; the known Stage 3 test result must not
  influence any Stage 4 choice.
- The proposed recipe is a project-specific controlled extension until the
  target research paper and its published fine-tuning settings are audited.
- The complete Kaggle artifact ZIP passed CRC and numerical consistency checks.
  Checkpoints are stored locally under the Git-ignored `models/` directory;
  reproducible metrics, histories, predictions, summaries, and vector figures
  are stored under `results/fine_tuned_resnet18/`.

### Required workflow

1. Load the saved split manifests without modification.
2. Use training-only stochastic augmentation and deterministic validation/test
   preprocessing; document both transform pipelines.
3. Start from `ResNet18_Weights.IMAGENET1K_V1` and replace the classification
   head for the 10 EuroSAT classes.
4. Choose the head-only warm-up, unfreezing policy, learning rates, optimizer,
   scheduler, regularization, and early-stopping rule using validation data only.
5. Save the best validation checkpoint and its complete training history.
6. Freeze the full configuration before the single test evaluation.
7. Save per-class metrics, probabilities, confusion matrix, calibration outputs,
   environment metadata, and experiment configuration.

### Metrics

- accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- multiclass log loss;
- multiclass Brier score;
- confusion matrix;
- one-vs-rest calibration curves where practical.

### Comparison goals

- Compare the fine-tuned model directly with both completed baselines on the same split.
- Check whether PermanentCrop, River, and Highway improve.
- Report both classification quality and probability quality.
- Keep the image-level/geographic-generalization limitation explicit.

## Completed: Stage 6 — paired model comparison

The archival notebook is `Notebooks/06_paired_model_comparison.ipynb`.
It aligns the three models' saved predictions by `dataset_index` against the
locked 4,050-image test manifest. The frozen-model prediction CSV from the
original Kaggle run is now at `results/frozen_resnet18/test_predictions.csv`.
The reproducible result tables and method record are under
`results/paired_model_comparison/`.

- The fine-tuned ResNet18 has the best uncalibrated test accuracy (`0.979506`),
  macro-F1 (`0.978540`), log loss (`0.074072`), and multiclass Brier score
  (`0.033437`) among the three registered models.
- Against frozen ResNet18, its paired accuracy difference is `+0.030123`
  (2,000-resample bootstrap 95% interval `[0.023704, 0.036790]`); its
  macro-F1 difference is `+0.031485` (`[0.024878, 0.038559]`).
- Its log-loss difference from frozen ResNet18 is `-0.085727`
  (`[-0.102065, -0.068015]`), and its Brier difference is `-0.045199`
  (`[-0.053137, -0.037320]`).
- On discordant test images, only frozen ResNet18 is correct 30 times and only
  fine-tuned ResNet18 is correct 152 times. The two-sided exact McNemar
  p-value is `7.70e-21`.
- The comparison uses the original uncalibrated Stage 4 predictions; Stage 5
  temperature scaling is evaluated separately and does not change class
  predictions.
- The intervals treat images as independent. The fixed image-level split does
  not establish performance in unseen geographic regions, and no test result
  should be used for further model or calibration tuning.

## Completed: Stage 7 — qualitative failure-case analysis

The archival notebook is `Notebooks/07_failure_case_analysis.ipynb`. It joins
the handcrafted, frozen ResNet18, and fine-tuned ResNet18 predictions to the
locked test manifest by `dataset_index`, then reviews aggregate error counts and
selected original image patches. Outputs are stored under
`results/qualitative_failure_analysis/`.

### Recorded findings

- Fine-tuned ResNet18 makes 83 errors on the 4,050-image test split.
- The largest class error counts are PermanentCrop (21/375), Pasture (16/300),
  HerbaceousVegetation (13/450), and River (11/375). Highway has 8/375 errors;
  Forest and SeaLake have none on this fixed split.
- The most frequent fine-tuned confusion pairs are PermanentCrop → AnnualCrop
  (11), Pasture → AnnualCrop (7), PermanentCrop → HerbaceousVegetation (6),
  Pasture → Forest (5), River → Highway (5), and HerbaceousVegetation → Pasture
  (5).
- Relative to frozen ResNet18, fine-tuning corrects 152 errors, introduces 30
  errors, and leaves both models wrong on 53 images.
- The review set contains 18 deliberately selected cases: six preselected
  PermanentCrop/River/Highway examples, six of the remaining highest-confidence
  fine-tuned errors, and six of the remaining highest-confidence corrections.
- The completed `case_notes.csv` records visible features, possible reasons for
  confusion, apparent image or label ambiguity, and confidence in each human
  interpretation.

### Interpretation limits

The selected cases are not a random sample and cannot estimate the prevalence
of a visual failure mechanism. Notes about field geometry, narrow linear
features, vegetation, or built-up patterns are hypotheses based on 64 × 64 RGB
patches; they do not reveal a model's causal decision process and must not be
used to relabel the data or tune a model on the test set. The image-level split
still does not establish generalization to unseen geographic regions.

## Planned later stages

1. Final research report and portfolio presentation.

## Immediate next action

Prepare the final research report and portfolio presentation. Consolidate the
fixed experimental protocol, model progression, paired comparisons,
calibration results, qualitative examples, reproducibility instructions, and
geographic-generalization limitation. Perform editorial and artifact checks;
do not introduce new model selection or tune against the locked test set.
