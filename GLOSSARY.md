# Glossary

**Archive cache**:
The persistent, on-disk copy of the downloaded CMAPSS zip, kept across process runs so the dataset is fetched from NASA at most once rather than once per loader call or per run.
_Avoid_: Temp directory, download folder

**Cache directory**:
The directory holding the archive cache, resolved from `CMAPSS_CACHE_DIR` if set, otherwise a stdlib-computed per-user cache location (honoring `XDG_CACHE_HOME`, else `~/.cache/cmapss`).

**Archive override**:
A user-supplied path to a manually obtained copy of the CMAPSS zip, provided via the `CMAPSS_ARCHIVE` environment variable or an optional loader parameter. When present, it seeds the archive cache and no download is attempted.
_Avoid_: Manual download, local copy

**Fetching**:
Obtaining the CMAPSS archive and placing it in the archive cache — via network download (with retries) or via an archive override. Distinct from *loading*, which reads sensor columns out of an archive already in the cache.
_Avoid_: Downloading (too narrow — excludes the override path)

**Loading**:
Reading a specific split (train, test, or ground-truth RUL) out of the archive cache into a DataFrame/array. Assumes fetching has already happened, and triggers it transparently if it hasn't.

**Lifetime**:
A run-to-failure unit's total number of cycles, i.e. its remaining useful life at cycle 0 — `max(time_cycles)`. Defined only for units from the train split, whose trajectories run to failure; a truncated test trajectory has no observable lifetime, only a ground-truth remaining life at its last observed cycle.
_Avoid_: Duration, age, total RUL

**Linear RUL**:
The unclipped remaining-useful-life target `max(time_cycles) - t`, decreasing by exactly one per cycle. This is what the models are trained on, as opposed to the widely used piecewise-constant convention that caps RUL at a constant (typically 125 or 130) early in life. Naming it explicitly keeps a reader from assuming the capped convention and misreading the targets and curves.
_Avoid_: RUL (ambiguous between the two conventions), piecewise RUL

**Full window**:
A window whose every row is an observed cycle of the unit, as opposed to a *padded window*, whose leading rows repeat the unit's first observed reading to make a short trajectory reach the window length. A padded window's remaining-useful-life target is still computed from real cycles, so it is a usable training sample; but its sensor history is fabricated, so a prediction made from it has no support in the data.
_Avoid_: Complete window, valid window, real window

**RUL curve**:
For a single unit, its linear RUL and the model's predictive RUL plotted against time cycles. The predicted curve begins at the unit's first full window, since no prediction exists before `window_length` cycles of history have accumulated.
_Avoid_: RUL plot, degradation curve

**Lifetime-spanning units**:
The shortest-, lower-median-, and longest-lifetime units of a set of run-to-failure trajectories, drawn side by side so that a reader can tell whether the model's accuracy and its stated uncertainty behave consistently across units that fail early and units that survive several times as long. Ordered ascending by lifetime with the unit ID as tie-break, and the median is the lower of the two central entries, so the same data always yields the same three units.
_Avoid_: Representative units, extreme units, best/worst units (none of the three is chosen for how well the model does on it)

**Predictive interval**:
The central 95% interval of a model's posterior predictive, from the 2.5% and 97.5% quantiles of its mixture over dropout draws, left unclipped at zero, so that a band extending below zero remains visible as evidence of the Gaussian assumption breaking down near end of life. It counts both the aleatoric noise and the spread of the draws; drawn as the *total band*.
_Avoid_: Confidence interval (this is an interval over a predicted value, not over a parameter estimate)

**Epistemic band**:
The 2.5% and 97.5% quantiles, at each window, of a posterior predictive's draws' means, ignoring their aleatoric variance: how much the dropout draws disagree about the mean, i.e. the model's uncertainty about the function itself. It lies inside the predictive interval, but the gap between the two is not the aleatoric part, since quantile widths do not add.
_Avoid_: Model uncertainty interval, credible interval

**Epistemic fraction**:
A model's epistemic share of its predictive variance, averaged over the test units: per unit, `Var_epi / (Var_epi + Var_ale)`, where the epistemic variance is the variance of the draws' means and the aleatoric variance is the draws' average aleatoric variance; by the law of total variance, the two add up to the mixture's variance. It is zero for a single-draw model, such as one with dropout rate 0, and approaches 1 when the draws' disagreement dominates the predicted noise. Unlike the *epistemic band*, it splits the variance exactly, since variances add where quantile widths do not.
_Avoid_: Epistemic ratio (ambiguous with epistemic over aleatoric), model uncertainty share

**Held-out residuals**:
A mean model's squared residuals on windows it never trained on: in `fd001_base.py`, the validation split's. The staged model's Gamma variance stage fits these rather than the in-sample residuals on the train split, which understate the held-out error and so give too small a variance.
_Avoid_: Validation residuals (ambiguous with monitoring), out-of-fold residuals (no folds here), OOF

**Calibration curve**:
A model's empirical coverage on a test set plotted against the nominal confidence level, one point per level: for each level `c`, the fraction of targets inside its central interval at `c`, from the `(1 - c) / 2` and `(1 + c) / 2` quantiles of its posterior predictive. A calibrated model follows the diagonal; a curve below it is overconfident, one above it underconfident.
_Avoid_: Reliability diagram (usually means binned predicted probability against observed frequency in classification), quantile calibration (the one-sided `P(y ≤ q_p)` against `p`)
