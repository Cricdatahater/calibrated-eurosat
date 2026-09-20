# Stage 4 validation selection

Both candidates began from the identical epoch-3 head-only warm-up checkpoint.

| Candidate | Trainable parameters | Best epoch | Validation macro-F1 | Validation log loss |
|---|---|---:|---:|---:|
| A | `layer4` + `fc` | 12 | 0.977565 | 0.088523 |
| B | Full network | 7 | 0.977003 | 0.074390 |

Candidate A was selected because validation macro-F1 was the preregistered primary criterion. Candidate B's lower log loss is a tie-breaker only, and macro-F1 was not tied. The selection was frozen before the single test evaluation.
