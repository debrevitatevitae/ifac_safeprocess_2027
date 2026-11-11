# The Gamma stage fits held-out residuals

The staged MC-Dropout models (`staged+mcd`, `staged+mcd+proxy`) in `compare.py` fit their Gamma variance stage on **held-out residuals**: the mean's squared residuals on the validation split, which the mean stage never trained on. This holds for both Gamma fits, the one after the mean stage and the refit to the posterior predictive's mean.

## Why

- In-sample residuals understate the held-out error, so a Gamma network fit to them predicts too small a variance.
- The k-fold prototype (#3) showed out-of-fold residuals fix this: on the Gaussian staged model, 95% coverage went from 0.46 to 0.92 and NLL from 10.2 to 4.95.

## Decisions

- **Holdout split, not k-fold.** The existing train/validation split is reused, with `validation_fraction` raised from 0.2 to 0.3 so the Gamma stage sees ~30 units. k-fold cross-fitting and fold ensembles were not adopted.
- **The split is re-drawn per seed** (`config.seed + i`), with the standardization re-fitted on each seed's train split, so `--seeds` checks robustness to the split as well as to initialization.
- **Same data budget for the joint model.** `joint+mcd` trains on train ∪ validation, with no validation monitoring.
- **Gamma regularization stays** (`gamma_hidden_channels` 4, `gamma_weight_decay` 0.1): the Gamma stage has small data and no validation set of its own.
- **No pooled residuals.** Pooling every fold's residuals into one Gamma net never improved CRPS or NLL in the prototype, and widened the intervals.

## Consequences

- The figure draws the first seed's train-split units. They are in-sample for every model, so the figure is for visualization only.
- Early stopping remains out of scope.
