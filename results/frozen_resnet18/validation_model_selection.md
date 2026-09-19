# Frozen ResNet18 model-selection audit

## Verified status

The downloaded Kaggle notebook was inspected on 19 September 2026. It records a
complete validation-only regularization search followed by a single locked test
evaluation. The test partition was not used to select the representation or the
logistic-regression regularization strength.

## Frozen choice

`C = 0.01` was selected for the standardized multinomial logistic-regression
classifier on the 512-dimensional frozen ResNet18 embeddings.

The declared selection rule was highest validation macro-F1, followed by lower
validation log loss and then smaller `C`. The selected configuration achieved:

- validation accuracy: `0.939012`;
- validation macro-F1: `0.936953`;
- validation log loss: `0.182669`;
- validation multiclass Brier score: `0.091367`;
- validation top-label ECE: `0.018778`.

The same candidate was best on all five reported validation metrics. Larger
values of `C` progressively worsened log loss and calibration, consistent with
weaker regularization producing increasingly overconfident predictions.

## Locked test result

After freezing `C`, the pipeline was refitted on the combined training and
validation embeddings and evaluated once on 4,050 test images:

- test accuracy: `0.949383`;
- test macro-F1: `0.947055`;
- test log loss: `0.159799`;
- test multiclass Brier score: `0.078636`;
- test top-label ECE: `0.020928`.

## Controls verified in the notebook

1. The original fixed split manifests and checksum are reused unchanged.
2. Dataset indices are unique within partitions and disjoint across partitions.
3. The ImageNet preprocessing bundled with the ResNet18 weights is used.
4. The backbone is frozen, placed in evaluation mode, and called under
   `torch.inference_mode()`.
5. `StandardScaler` is fitted on training embeddings only during model
   selection through a scikit-learn pipeline.
6. Test embeddings are extracted only after the selected `C` is frozen.
7. Fit time is recorded for operational reference and is not a selection
   criterion.

## Corrections applied during review

- Added narrative Markdown for the objective, protocol, each experimental
  stage, findings, and limitations; the downloaded notebook originally
  contained code cells only.
- Replaced the moving `ResNet18_Weights.DEFAULT` alias with the exact
  `IMAGENET1K_V1` weight enum used by the recorded checkpoint.
- Added deterministic seeds and CUDA reproducibility controls.
- Added explicit assertions that the backbone is frozen and in evaluation mode.
- Expanded the experiment summary with validation metrics, weight and transform
  identifiers, software versions, CUDA/GPU metadata, and the one-time test flag.
- Removed unused retained candidate models and temporary file-download/debug
  cells.
- Removed misleading saved `0%` widget displays while preserving final outputs.

## Remaining limitations

- Validation ECE is descriptive, not an unbiased estimate of final calibration,
  because the same validation partition selected the classifier.
- The image-level split does not establish geographic generalization and may
  place nearby or visually related Sentinel-2 patches in different partitions.
- The representation is frozen; the effect of end-to-end EuroSAT fine-tuning
  remains to be evaluated.
- The machine-readable CSV values in this repository are synchronized from the
  saved notebook output. Large embeddings and fitted model binaries remain
  intentionally excluded from Git.
