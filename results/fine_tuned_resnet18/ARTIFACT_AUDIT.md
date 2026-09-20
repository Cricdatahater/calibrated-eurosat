# Stage 4 artifact audit

Audit date: 20 September 2026

The downloaded Kaggle ZIP and its extracted artifacts passed the following
checks:

- ZIP CRC test reported no damaged member.
- Candidate A checkpoint SHA-256 matches the frozen-selection record:
  `3f6a701aca082fa675ba229ddfbae0139346e3cbff84ea02ff0617ab73c5d657`.
- Candidate A history contains 12 epochs and reproduces best epoch 12 with
  validation macro-F1 `0.977565` and log loss `0.088523`.
- Candidate B history contains 11 epochs and reproduces best epoch 7 with
  validation macro-F1 `0.977003` and log loss `0.074390`.
- Test predictions contain 4,050 unique dataset indices, exactly matching the
  committed fixed test manifest.
- Every prediction contains 10 probabilities that sum to one within numeric
  serialization tolerance, and the stored predicted class equals `argmax`.
- Accuracy, log loss, multiclass Brier score, and 15-bin ECE recomputed from
  the prediction CSV agree with `test_metrics.json` within `2e-6`.
- The confusion matrix recomputed from predictions exactly matches the exported
  matrix.
- The cleaned notebook contains a monotonic execution record and no saved error
  outputs. Its archival note explains that Kaggle restarted between training
  and final evaluation and that candidate outputs were consolidated from the
  verified histories.

The large `.pth` files are retained locally under the Git-ignored `models/`
directory. Their hashes and sizes are recorded in `checkpoint_manifest.json`.
