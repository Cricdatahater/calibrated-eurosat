# Frozen ResNet18 validation model selection

## Status

This is a validation-stage result. The test partition remains locked. The
reported values were transcribed from the Kaggle output screenshot supplied on
19 September 2026; the original Kaggle CSV and notebook have not yet been
committed to this repository.

## Frozen choice

`C = 0.01` is selected for the standardized multinomial logistic-regression
classifier on the 512-dimensional frozen ResNet18 embeddings.

The declared selection rule was highest validation macro-F1, followed by lower
validation log loss and then smaller `C`. The selected configuration achieved:

- accuracy: `0.939012`;
- macro-F1: `0.936953`;
- log loss: `0.182669`;
- multiclass Brier score: `0.091367`;
- top-label expected calibration error: `0.018778`.

The same candidate was best on all five reported validation metrics. Larger
values of `C` progressively worsened log loss and calibration, consistent with
weaker regularization producing increasingly overconfident predictions.

## Methodological audit

No error is visible in the selection table itself, but the following controls
must be verified in the Kaggle notebook before the result is treated as fully
reproducible:

1. The original fixed split manifests and checksum must be reused unchanged.
2. `StandardScaler` must be fitted on training embeddings only during model
   selection; validation embeddings may only be transformed by that fitted
   scaler.
3. The ResNet18 backbone must use an explicitly recorded ImageNet weight enum,
   the transforms bundled with those weights, `eval()` mode, frozen parameters,
   and inference without gradient tracking.
4. Dataset indices must remain attached to embeddings, and the three partitions
   must be checked for disjointness.
5. `C = 0.01` is now frozen. Refining the grid after viewing the test results is
   prohibited. Any optional refinement around `0.01` must happen now, use only
   validation data, and be documented as an additional search.
6. Validation ECE is descriptive, not an unbiased estimate of final calibration,
   because the same validation partition selected the classifier. Temperature
   scaling later requires a clearly defined calibration protocol.
7. Test embeddings and metrics must be generated only after the full pipeline is
   frozen, and the test set must be evaluated once.
8. The image-level split does not establish geographic generalization and may
   contain nearby or visually related Sentinel-2 patches across partitions.

Training time was recorded for operational reference only and was not used as a
selection criterion.

## Immediate next step

Fit the selected pipeline on training plus validation embeddings, extract the
locked test embeddings with the unchanged backbone and preprocessing, evaluate
the test partition once, and save the complete Kaggle notebook, environment
metadata, predictions, per-class report, confusion matrix, fitted pipeline, and
experiment summary.
