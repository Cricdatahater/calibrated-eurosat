# Calibrated EuroSAT — Project Handover

Last updated: 19 September 2026

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
- Machine-readable audit: `reports/data_audit.json`
- Image-level feature table: `reports/image_statistics.csv`
- Duplicate report: `reports/duplicate_report.csv`
- Split balance report: `reports/split_class_distribution.csv`
- Figures: `reports/figures/`
- Fixed split definitions: `data/splits/`
- Statistical results: `results/statistical_baseline/`
- Frozen ResNet18 validation search and audit: `results/frozen_resnet18/`
- Fitted statistical pipeline (local, Git-ignored): `models/statistical_logistic_regression.joblib`

## Next milestone: Stage 4 — fine-tuned ResNet18

Create `Notebooks/04_finetuned_resnet18.ipynb` on the GPU platform. Initialize from the same `IMAGENET1K_V1` checkpoint, preserve the fixed manifests, and tune the training procedure using validation data only.

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

## Planned later stages

1. Calibration analysis and temperature scaling.
2. Paired comparison using bootstrap confidence intervals and McNemar's test.
3. Qualitative failure-case analysis.
4. Final README and portfolio presentation.

## Immediate next action

Design the validation-only fine-tuning protocol before starting Stage 4 training; the completed Stage 3 test result is now locked and must not guide Stage 4 hyperparameter selection.
