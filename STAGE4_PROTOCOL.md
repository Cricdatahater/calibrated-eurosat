# Stage 4 Registered Protocol — Fine-tuned ResNet18

Protocol registered: 20 September 2026  
Status: completed and locally audited; Git push pending  
Canonical progress record: `PROJECT_HANDOVER.md`

## Research question

Does supervised adaptation of an ImageNet-pretrained ResNet18 to EuroSAT improve
classification and probability quality over the completed frozen-embedding
baseline?

Stage 3 achieved test accuracy `0.949383`, macro-F1 `0.947055`, and log loss
`0.159799`. Those test results are known, so they must not be used to change the
Stage 4 architecture, augmentation, optimizer, learning rates, epoch budget,
checkpoint rule, or candidate-selection rule.

## Reproduction status

The fine-tuning recipe below is a pre-registered project protocol, not yet a
verified reproduction of a specific paper. Before describing Stage 4 as a paper
reproduction, record the paper title/DOI and audit its published architecture,
preprocessing, augmentation, optimizer, learning rates, batch size, training
duration, freezing policy, and evaluation protocol. Paper-specified settings
take precedence; every deviation must be documented.

## Fixed data protocol

- Dataset: EuroSAT RGB, 27,000 images, 10 classes.
- Existing stratified split: 70% train, 15% validation, 15% test.
- Training images: 18,900.
- Validation images: 4,050.
- Test images: 4,050.
- Split seed: `42`.
- Split checksum: `f97c4ec9a27435a932662d5a8b707255`.
- Reuse `data/splits/*.csv`; do not generate a new split.
- Test data is evaluated once after candidate selection and checkpoint freezing.

## Notebook and output locations

- Notebook: `Notebooks/04_finetuned_resnet18.ipynb`.
- Results: `results/fine_tuned_resnet18/`.
- Local checkpoints: `models/fine_tuned_resnet18/` (Git-ignored).

## Reproducibility controls

- Seed Python, NumPy, PyTorch, and all CUDA devices with `42`.
- Disable cuDNN benchmarking and enable deterministic behavior with warnings.
- Record Python, PyTorch, Torchvision, CUDA, GPU, and scikit-learn versions.
- Pin `ResNet18_Weights.IMAGENET1K_V1`; do not use the moving `DEFAULT` alias.
- Use mixed precision on CUDA and record whether it was enabled.
- Preserve dataset indices in all prediction exports.

## Image transformations

Training only:

1. `RandomResizedCrop(224, scale=(0.85, 1.0))`;
2. `RandomHorizontalFlip(p=0.5)`;
3. `RandomVerticalFlip(p=0.5)`;
4. conversion to tensor;
5. ImageNet mean/std normalization.

Validation and test use the deterministic transforms bundled with
`IMAGENET1K_V1`. Do not apply random augmentation outside the training set.
Strong colour jitter, arbitrary-angle rotation, MixUp, CutMix, class weighting,
and label smoothing are excluded from this registered baseline.

## Model initialization

1. Load ResNet18 with `IMAGENET1K_V1` weights.
2. Replace the 1,000-class layer with `nn.Linear(512, 10)`.
3. Initialize the new head once using seed 42.
4. Save the initial state so both fine-tuning candidates start identically.

## Phase A — fixed head warm-up

- Freeze the complete backbone.
- Train only the new 10-class head for exactly 3 epochs.
- Loss: standard cross-entropy.
- Optimizer: AdamW.
- Head learning rate: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `64`.
- Gradient clipping: maximum norm `1.0`.
- Save the epoch-3 warm-up checkpoint; both candidates begin from it.

## Phase B — predeclared candidates

### Candidate A: partial fine-tuning

- Trainable: `layer4` and `fc`.
- Frozen: stem and `layer1`–`layer3`.
- Learning rate for `layer4`: `1e-4`.
- Learning rate for `fc`: `5e-4`.

### Candidate B: full fine-tuning

- Trainable: all parameters.
- Learning rate for stem and `layer1`–`layer3`: `1e-5`.
- Learning rate for `layer4`: `5e-5`.
- Learning rate for `fc`: `5e-4`.

### Shared candidate settings

- Begin from the identical epoch-3 warm-up checkpoint.
- Maximum fine-tuning epochs: `12`.
- Loss: standard cross-entropy.
- Optimizer: AdamW.
- Weight decay: `1e-4`.
- Scheduler: `ReduceLROnPlateau` on validation log loss.
- Scheduler factor: `0.2`; scheduler patience: `2`.
- Early-stopping/checkpoint monitor: validation macro-F1.
- Early-stopping patience: `4`.
- Gradient clipping: maximum norm `1.0`.
- Do not stop or change settings in response to test performance.

## Candidate-selection rule

For each candidate, retain its checkpoint with the highest validation macro-F1.
Select the Stage 4 model using this fixed order:

1. higher validation macro-F1;
2. if equal at stored precision, lower validation log loss;
3. if still equal, Candidate A because it alters fewer pretrained parameters.

Once selected, freeze the candidate name, epoch, checkpoint hash, and complete
configuration. Do not refit on train plus validation because neural-network
early stopping depends on validation feedback and retraining would change the
optimization trajectory.

## Metrics

Record on validation at every epoch and once on test:

- cross-entropy/log loss;
- accuracy;
- macro-F1;
- per-class precision, recall, and F1 for the final model;
- multiclass Brier score;
- top-label ECE with 15 equal-width bins;
- confusion matrix for the final test evaluation.

The test evaluation must export class probabilities for all 10 classes.

## Required artifacts

```text
results/fine_tuned_resnet18/
├── experiment_summary.json
├── candidate_comparison.csv
├── training_history_candidate_a.csv
├── training_history_candidate_b.csv
├── frozen_selection.json
├── checkpoint_manifest.json
├── environment.json
├── test_predictions.csv
├── test_classification_report.csv
├── test_confusion_matrix.csv
├── test_calibration_bins.csv
├── test_metrics.json
└── figures/
    ├── training_curves.svg
    ├── candidate_validation_comparison.svg
    ├── test_confusion_matrix.svg
    └── test_calibration_curve.svg
```

Each prediction row must include `dataset_index`, true and predicted class
indices/names, confidence, and all 10 class probabilities. These fields are
required for later calibration, paired bootstrap intervals, McNemar's test, and
failure-case analysis.

Per-epoch validation metrics were exported for both candidates. Validation
probabilities were not retained in the downloaded Kaggle artifact set; any
future validation-logit requirement must be produced in a new, explicitly
documented calibration run without altering the locked Stage 4 result.

## Notebook structure

1. Objective and registered protocol.
2. Runtime and deterministic seeds.
3. Dataset acquisition and fixed-split verification.
4. Training and evaluation transforms.
5. Datasets and data loaders.
6. Model and optimizer construction helpers.
7. Metric and training-loop helpers.
8. Three-epoch head warm-up.
9. Candidate A fine-tuning.
10. Candidate B fine-tuning.
11. Validation comparison and frozen selection.
12. Single locked test evaluation.
13. Per-class report, confusion matrix, and baseline comparison.
14. Artifact export.
15. Findings and limitations.

## Execution checklist

- [x] Stage 4 protocol registered before training.
- [ ] Paper identity and published training recipe recorded, if applicable.
- [x] Kaggle notebook created with GPU and Internet enabled (Tesla T4).
- [x] Environment and fixed split verified.
- [x] Transform pipelines implemented and loader smoke test passed.
- [x] Head warm-up completed and checkpoint saved (epoch 3; validation
  macro-F1 `0.925971`, validation log loss `0.219391`).
- [x] Candidate A completed (best epoch 12; validation macro-F1 `0.977565`,
  log loss `0.088523`).
- [x] Candidate B completed (best epoch 7; validation macro-F1 `0.977003`,
  log loss `0.074390`).
- [x] Candidate A selected using validation only under the predeclared primary
  macro-F1 rule.
- [x] Configuration and checkpoint frozen (Candidate A, epoch 12; SHA-256
  `3f6a701aca082fa675ba229ddfbae0139346e3cbff84ea02ff0617ab73c5d657`).
- [x] Test evaluated once (accuracy `0.979506`, macro-F1 `0.978540`, log loss
  `0.074072`, Brier `0.033437`, 15-bin ECE `0.012213`).
- [x] Required artifacts downloaded to the project and numerically verified.
- [ ] Cleaned notebook and reproducible non-checkpoint artifacts pushed to GitHub.

## Current restart point

Review and commit the cleaned notebook, protocol, histories, predictions,
metrics, summaries, and vector figures. Checkpoints remain local and Git-ignored.
Then begin the separate validation-only temperature-scaling stage.
