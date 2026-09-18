# Calibrated EuroSAT — Project Handover

Last updated: 18 September 2026

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
- Machine-readable audit: `reports/data_audit.json`
- Image-level feature table: `reports/image_statistics.csv`
- Duplicate report: `reports/duplicate_report.csv`
- Split balance report: `reports/split_class_distribution.csv`
- Figures: `reports/figures/`
- Fixed split definitions: `data/splits/`
- Statistical results: `results/statistical_baseline/`
- Fitted statistical pipeline (local, Git-ignored): `models/statistical_logistic_regression.joblib`

## Next milestone: Stage 3 — frozen CNN embeddings

Create `Notebooks/03_frozen_resnet18_embeddings.ipynb`. Use an ImageNet-pretrained ResNet18 as a fixed feature extractor and train a linear classifier on its embeddings.

### Required workflow

1. Load the three saved split manifests; do not create a new split.
2. Use the preprocessing transforms associated with the selected pretrained ResNet18 weights.
3. Freeze all CNN parameters and extract one embedding per image without gradient tracking.
4. Cache embeddings with dataset indices, labels, weight identifier, and preprocessing metadata.
5. Fit any scaler and the linear classifier using training embeddings only.
6. Select classifier regularization using validation macro-F1 and log loss.
7. Keep the test partition locked until the representation and classifier settings are frozen.
8. Save metrics, predictions, confusion matrices, calibration outputs, and experiment metadata.

### Metrics

- accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- multiclass log loss;
- multiclass Brier score;
- confusion matrix;
- one-vs-rest calibration curves where practical.

### Comparison goals

- Compare the frozen-CNN classifier directly with the handcrafted-feature baseline on the same split.
- Check whether visually ambiguous class pairs improve.
- Report both classification quality and probability quality.
- Keep the image-level/geographic-generalization limitation explicit.

## Planned later stages

1. Fine-tuned ImageNet-pretrained ResNet18.
2. Calibration analysis and temperature scaling.
3. Paired comparison using bootstrap confidence intervals and McNemar's test.
4. Qualitative failure-case analysis.
5. Final README and portfolio presentation.

## Immediate next action

Confirm GPU access, then build the frozen ResNet18 embedding baseline using the existing split manifests.
