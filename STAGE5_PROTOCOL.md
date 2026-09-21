# Stage 5 Protocol and Completed Record — Temperature Scaling

Protocol fixed: 21 September 2026  
Status: completed and locally audited  
Canonical progress record: `PROJECT_HANDOVER.md`

## Research question

Can validation-only scalar temperature scaling improve the probability quality
of the locked Stage 4 ResNet18 without changing its classifications?

## Locked inputs

- Model: Stage 4 Candidate A (`layer4` + classifier fine-tuning), epoch 12.
- Checkpoint SHA-256:
  `3f6a701aca082fa675ba229ddfbae0139346e3cbff84ea02ff0617ab73c5d657`.
- Split checksum: `f97c4ec9a27435a932662d5a8b707255`.
- Validation images: 4,050.
- Test images: 4,050.
- Preprocessing: deterministic transforms from
  `ResNet18_Weights.IMAGENET1K_V1`.
- Inference path: batch size 64 with CUDA automatic mixed precision, matching
  the archived Stage 4 evaluation.

The classifier weights, split, preprocessing, and class mapping are not changed.

## Calibration method

1. Regenerate validation logits from the locked checkpoint.
2. Fit one positive scalar temperature `T` by minimizing validation multiclass
   negative log-likelihood.
3. Optimize `log(T)` with bounded scalar optimization, corresponding to
   `0.05 <= T <= 10`.
4. Freeze the fitted temperature before calculating calibrated test metrics.
5. Transform logits as `z / T` before softmax.

The positive scalar preserves the ordering of class logits, so predicted
classes, accuracy, macro-F1, and the confusion matrix must remain unchanged.

## Metrics

- multiclass log loss;
- multiclass Brier score;
- 15-bin equal-width top-label expected calibration error (ECE);
- accuracy and macro-F1 as classification-invariance checks;
- reliability and confidence plots.

Paired bootstrap intervals use 2,000 seed-42 resamples of the test observations.
Each reported difference is `calibrated - uncalibrated`; negative values favor
temperature scaling for all three probability metrics.

## Completed result

The fitted temperature is `1.8226799937`.

| Partition | Calibration | Accuracy | Macro-F1 | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| Validation | Before | 0.978519 | 0.977565 | 0.088523 | 0.036730 | 0.013305 |
| Validation | After | 0.978519 | 0.977565 | 0.069421 | 0.035502 | 0.005596 |
| Test | Before | 0.979506 | 0.978540 | 0.074072 | 0.033437 | 0.012213 |
| Test | After | 0.979506 | 0.978540 | 0.060098 | 0.031412 | 0.002788 |

Test bootstrap differences and percentile intervals:

| Metric | Difference | 95% interval |
|---|---:|---:|
| Log loss | -0.013974 | [-0.022496, -0.006348] |
| Multiclass Brier score | -0.002026 | [-0.003563, -0.000492] |
| Top-label ECE | -0.009425 | [-0.011544, -0.003110] |

All three intervals exclude zero. All 4,050 test classifications are unchanged.

## Artifacts

```text
Notebooks/05_temperature_scaling.ipynb
results/temperature_scaling/
├── bootstrap_intervals.csv
├── calibration_summary.json
├── environment.json
├── metric_comparison.csv
├── temperature.json
├── test_calibration_bins_calibrated.csv
├── test_calibration_bins_uncalibrated.csv
├── test_logits.npz
├── test_predictions_calibrated.csv
├── validation_logits.npz
└── figures/
    └── reliability_and_confidence.svg
```

The trained checkpoint remains local and Git-ignored.

## Audit checks completed

- [x] Checkpoint SHA-256 matches the locked Stage 4 checkpoint.
- [x] Split checksum matches the fixed experimental split.
- [x] Validation and test logit arrays each contain 4,050 unique observations.
- [x] Logits are finite and have 10 class columns.
- [x] Uncalibrated test accuracy and macro-F1 reproduce Stage 4 exactly.
- [x] Calibrated and uncalibrated predicted classes are identical.
- [x] Temperature was fitted using validation data only.
- [x] Test metrics were calculated only after the temperature was frozen.
- [x] Prediction export contains 4,050 unique rows and no missing values.
- [x] Kaggle artifact archive passed extraction and numerical consistency checks.

## Limitations

- ECE depends on the chosen number and placement of bins.
- A single fixed validation split was used to estimate the temperature.
- Bootstrap intervals describe uncertainty on this image-level test split; they
  do not establish geographic generalization.
- Calibration on the in-domain EuroSAT split does not imply calibration after
  distribution shift.

## Next stage

Perform paired model comparisons across the handcrafted, frozen-CNN, and
fine-tuned predictions using bootstrap confidence intervals and McNemar's test,
followed by qualitative failure-case analysis.
