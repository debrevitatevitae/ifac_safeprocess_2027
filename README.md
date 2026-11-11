# ifac_safeprocess_2027

Code for the results submitted to the [IFAC SafeProcess 2027 conference](https://conferences.ifac-control.org/safeprocess2027/), built on [`stagereg`](https://github.com/GroupiSP/stagereg).

Install the environment with `uv sync --all-groups` and run the tests with `uv run pytest`.

## NASA CMAPSS FD001

The code in `src/ifac_safeprocess_2027/cmapss/` and the `scripts/cmapss/compare.py` script focus on predicting the remaining useful life (RUL) of the simulated jet engine units that compose the NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset, available from the [NASA CMAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8).

In particular, the scope is restricted to the FD001 dataset, which consists of 100 train and 100 test time trajectories. The train trajectories are run-to-failure, while the test trajectories are partial histories for which the RUL labels are provided in a separate file.

### Description of the files

```text
scripts/cmapss/compare.py          the entry point: trains, scores and plots the models
src/ifac_safeprocess_2027/cmapss/
├── data.py                        fetching and loading FD001
├── preprocessing.py               splitting, standardization and windowing
├── model.py                       the 1D-CNN networks
├── pipeline.py                    configuration, training and test scoring
└── plots.py                       RUL curves
```

#### `data.py`

Fetches the CMAPSS archive and loads the FD001 splits into pandas
DataFrames. The train/test DataFrames retain only the unit ID, time cycles,
and sensors 11, 12, 4, 7, 15, 20, 21, 2, and 17; the operational settings
and other sensors are discarded.

The archive is kept in a persistent on-disk cache, so it is downloaded from
NASA at most once rather than once per run. Two environment variables
override the defaults:

- `CMAPSS_CACHE_DIR` — the directory holding the cached archive.
  Defaults to a per-user cache location (`XDG_CACHE_HOME` if set, otherwise
  `~/.cache/cmapss`).
- `CMAPSS_ARCHIVE` — the path to a manually obtained copy of the
  CMAPSS zip. When set, it seeds the cache and no download is attempted,
  which is useful when the network is unavailable or untrusted.

Run the module with:

```shell
uv run python -m ifac_safeprocess_2027.cmapss.data
```

The `main` function displays all engine trajectories in seaborn line plots,
with one facet per selected sensor and independent sensor scales.

#### `preprocessing.py`

Turns the loaded DataFrames into the arrays the models consume:
`split_by_unit` holds out whole units for validation, `fit_standardization`
and `apply_standardization` standardize the sensor columns with statistics
fitted on the training subset only, `build_windows` produces sliding windows
with their aligned RUL targets, and `build_last_windows` produces the single
trailing window per test unit required by the FD001 test protocol. Units
shorter than the window length are left-padded by repeating their first row.
`build_unit_rul_curve` returns a single unit's `UnitRulCurve`: its
windows with their aligned linear RUL and window-end cycles, its lifetime,
and the true linear RUL over every observed cycle. The two time axes are
named apart — `window_cycles` is what a predicted curve is drawn against,
`trajectory_cycles` what the truth is drawn against — so a caller never
re-derives either one. A unit shorter than the window length is rejected
rather than padded.
`select_lifetime_spanning_units` picks the shortest-, median-, and
longest-lifetime units of a set of trajectories, a unit's lifetime being
its maximum observed time cycle. Units are ordered ascending by lifetime
with the unit ID as tie-break and the median is the lower of the two
central entries, so the same data always yields the same three units and
the figure is reproducible.

#### `model.py`

Contains the networks the MC-Dropout models are built from: the staged
model's mean network (`Cnn1DMeanModel`) and narrower Gamma variance network
(`Cnn1DGammaModel`), and the joint model's network (`Cnn1DGaussianModel`):
one backbone of the same kind, followed by a Gaussian head predicting mean
and scale together. The mean and joint networks take a dropout rate.

#### `plots.py`

Builds the RUL curves. `plot_validation_rul_curves` takes the
standardized validation trajectories, the feature columns, a sequence of
`LabelledModel`s (a label and a trained model each, predicting a posterior
predictive), the three units to draw, and the window length, and draws an
N×3 grid: one row per model, labelled with its name, and one column per
unit. `compare.py` passes the lifetime-spanning units of the validation
subset, so the columns are its shortest-, median-, and longest-lifetime
unit in that order; the module itself decides how a RUL curve looks and
never which units it looks at. Spanning the lifetime range that way lets a
reader tell whether the model's accuracy and its stated uncertainty behave
consistently across units that fail early and units that survive several
times as long.

Each panel shows one unit's true linear RUL across its whole trajectory,
the posterior predictive's mixture mean, a total band (the mixture's
central 95% interval, from its quantiles) and a narrower epistemic band
(the 2.5% and 97.5% quantiles of the draws' means at each window); see
[Reading the total and epistemic bands](#reading-the-total-and-epistemic-bands).
Each curve comes from one evaluation over the unit's windows, so every
draw is the same function along it. The columns have independent x limits,
so the bands stay readable in the short-lived unit instead of being
squashed by the long-lived one's axis range, while each column shares its
x-axis across rows so that every model's curve for a unit sits on the same
time axis; y limits are independent per panel. Each column is titled with
its unit ID, the role it plays, and that unit's lifetime in cycles, and one
shared figure-level legend names the elements drawn. The predicted mean and
its bands start at each unit's first full window, since no prediction
exists before `window_length` cycles of history have accumulated; the
bands are left unclipped at zero, so a band dipping below zero stays
visible as evidence of the Gaussian assumption breaking down near end of
life. With a save path the figure is written there, otherwise it is
displayed.

The plotted units are drawn from the **held-out validation subset**, never
from the units the model trained on: those units are never fitted, so the
curves are evidence of generalization rather than of memorized fit. They
also cannot come from the FD001 test split, whose trajectories are
truncated rather than run to failure — a unit's lifetime, and hence its
true RUL at every cycle, is only observable for the run-to-failure train
split the validation subset is carved out of.

#### `pipeline.py`

The pipeline `compare.py` builds on; it has no entry point of its own.
`CmapssConfig` holds every setting. `prepare_cmapss_windows` loads the
FD001 train/test splits, standardizes and windows them, and returns a
frozen `PreparedCmapssData` carrying the windowed arrays together with the
standardized trajectories they were built from and the standardization
statistics fitted on the training subset, so a downstream consumer can
re-window a unit under exactly the feature scaling the models were trained
with. The training and validation RUL targets are standardized too, with
the training targets' mean and standard deviation
(`target_standardization`); the test RUL stays in cycles. Every model
trains on the standardized targets and is scored and plotted through
`CyclesPredictive`, which maps its posterior predictive back to cycles.

The staged training functions are `train_mean_model`, which takes an
optional `dropout_rate` and per-window `objective`, and `train_gamma_model`,
which trains the Gamma network on the mean network's squared residuals.
Both take their event sinks as a parameter, so each caller decides how
verbose training is. The Gamma network is narrower than the mean network
(`gamma_hidden_channels`, default 4 against `hidden_channels` 16) and trains
with AdamW weight decay (`gamma_weight_decay`, default 0.1); see
[the variance network](#the-variance-network). Next to them sit the
MC-Dropout helpers: the joint predictive wrapper `McDropoutPredictive`
(built by `build_mc_dropout_predictive`), the inference-only posterior-stage
runner `build_staged_mcd_posterior`, which returns a
`PosteriorStagePredictive`, `refit_gamma_to_posterior_mean`, which re-fits
the Gamma network to that predictive's mean, and the proxy-variance
objective `ProxyVarianceNLL`.

`evaluate_test_metrics` scores a posterior-predictive model against the
official FD001 test split (`RUL_FD001.txt`), one window per test unit
ending at its last observed cycle, on its exact mixture: RMSE, 95%
coverage from the mixture's quantiles, and the mixture's closed-form CRPS,
which carries no sampling noise.

#### `compare.py`

Compares three **MC-Dropout models** on the same prepared FD001 data: a
single training (joint) against separate trainings (staged), and the
staged model with a proxy-variance mean stage. Each is trained once with
dropout active and predicts with dropout active, as a posterior predictive
over `num_draws` dropout-mask sets. The three models per seed, in column
order:

| Column | Model | How it is trained | Predictive distribution |
| --- | --- | --- | --- |
| `joint+mcd` | joint MC-Dropout | one Gaussian-head CNN with dropout, on the Gaussian NLL | posterior predictive, each draw with its own aleatoric variance |
| `staged+mcd` | staged MC-Dropout | a mean stage with dropout, a Gamma stage on its residuals, an inference-only posterior stage (`MCDropout`), then the Gamma network re-fitted to the posterior predictive's mean | posterior predictive, all draws sharing the re-fitted aleatoric variance |
| `staged+mcd+proxy` | staged MC-Dropout proxy-variance | as `staged+mcd`, with the mean stage trained on `ProxyVarianceNLL` | as `staged+mcd` |

The data is prepared once, with the configured seed, so the
train/validation split is the same for every seed; seed index `i`
initializes every model, and its dropout masks, from `seed + i`.
`compare_models` returns one `SeedComparison` per seed, holding the trained
models and their test metrics: RMSE, 95% interval coverage, closed-form
CRPS and NLL, each on the exact mixture (mixture log-density, coverage from
mixture quantiles, the mixture's closed-form CRPS), through
`PosteriorPredictivePredictor`.

Run it with:

```shell
uv run python scripts/cmapss/compare.py
```

Training is silent. The script prints one line per (seed, model) run as it
finishes, then a table with one row per metric and one column per model,
each cell the `mean ± std` over seeds. The `±` is the sample standard
deviation over seeds (`ddof=1`), and 0 for a single seed. Pass `--seeds` to
trade runtime for confidence (default 5):

```shell
uv run python scripts/cmapss/compare.py --seeds 10
```

Finally it plots the first seed's RUL curves for the shortest-, median-,
and longest-lifetime held-out validation units: one row per model, in
column order, with a shared time axis in each column. Using the first seed
keeps the figure reproducible. Pass `--plot-path` to save that figure
instead of displaying it:

```shell
uv run python scripts/cmapss/compare.py --plot-path comparison.png
```

Note that the joint network has one backbone where the staged model has two
(one per stage), so it has roughly half the staged model's parameters. The
comparison holds the architecture and the per-model epoch budget fixed, not
the parameter count.

##### How the MC-Dropout models are trained

Dropout is the only regularizer of the joint network and of the staged
mean network: neither has a prior, weight decay or early stopping. The
joint and staged paths differ only in single against separate training.

`joint+mcd` is one Gaussian-head CNN trained with dropout, with
`run_supervised` on the Gaussian NLL for `joint_epochs` epochs at
`learning_rate`. Its prediction evaluates the network under `num_draws`
dropout-mask sets, each one applied to the whole batch, and keeps each
draw's own mean and aleatoric variance: a posterior predictive scored on its
exact mixture. The masks come from a fixed key derived from the seed, so
test scoring and the figure use the same draws.

`staged+mcd` trains its mean network with dropout, for `mean_epochs`
epochs at `learning_rate`, then a Gamma network on its squared residuals. A
`PosteriorStage` with the `MCDropout` inference method then does inference
only: its optimizer is `optax.set_to_zero()`, so the posterior is the
trained mean network, unchanged, under random dropout masks.
`PosteriorStage` needs at least one epoch, so it runs one, which only
evaluates the training loss. The posterior predictive's mean, the average
over the draws, is not the mean network with dropout off that the Gamma
network was fitted to, so `refit_gamma_to_posterior_mean` trains a fresh
Gamma network, from the same initialization and with the same width and
weight decay, on the squared residuals of the posterior predictive's mean.
The prediction is the stage's posterior predictive with `num_draws` draws
under the same fixed, seed-derived key, every draw sharing the re-fitted
aleatoric variance.

##### Reading the total and epistemic bands

Each row of the figure draws the mixture mean and two bands around it:

- the **total band**, the posterior predictive's central 95% interval
  (from the mixture's quantiles): where the next observed RUL should fall,
  counting both the aleatoric noise and the spread of the draws;
- the narrower **epistemic band**, the 2.5% and 97.5% quantiles of the
  draws' means at each window: how much the dropout draws disagree about
  the mean, i.e. the model's uncertainty about the function itself.

The total band is wider because it also counts the aleatoric noise, but
the gap between the two bands is not the aleatoric part: quantile widths
do not add. The variances do: the predictive variance is the mean of the
draws' aleatoric variances plus the variance of the draws' means. An
epistemic band that is narrow next to the total band still says that most
of the uncertainty is noise in the data rather than lack of knowledge.

##### Defaults

The MC-Dropout settings live in `CmapssConfig` (`pipeline.py`):

| Field | Default | Source |
| --- | --- | --- |
| `dropout_rate` | 0.5 | chosen |
| `num_draws` | 100 | fixed by the spec (GroupiSP/stagereg#173) |
| `gamma_hidden_channels` | 4 | chosen, [below](#the-variance-network) |
| `gamma_weight_decay` | 0.1 | chosen, [below](#the-variance-network) |
| `proxy_variance_floor_cycles` | 5.0 | chosen, see [the proxy-variance mean stage](#the-proxy-variance-mean-stage) |

##### The variance network

The test split holds only each unit's last window, so it says little about
early-life uncertainty. The held-out validation units span whole lifetimes,
and on them a Gamma network as wide as the mean network, trained with plain
Adam, is overconfident: in-sample residuals understate the held-out error,
and a flexible Gamma network fits them closely, giving a spiky,
overconfident variance.

The defaults therefore narrow the Gamma network to 4 channels and train it
with AdamW weight decay 0.1, which shrinks it toward a constant variance;
`staged+mcd` and `staged+mcd+proxy` then re-fit it to the posterior
predictive's mean. A much larger decay flattens the band to almost a
constant.

##### The proxy-variance mean stage

`staged+mcd` trains its mean on the squared error, which carries no
information about how uncertain the RUL is. Dropout then drops the channels
that track degradation as freely near the end of life as the healthy
channels early in life, although the time to failure grows more certain as
failure nears.

`staged+mcd+proxy` is `staged+mcd` with its mean stage trained and
validated on `ProxyVarianceNLL` (`pipeline.py`): the Gaussian NLL of the mean
under a fixed, fictitious scale `ProxyVarianceSchedule` (`preprocessing.py`)
that depends on the window's true RUL `r`:

```text
sigma(r) = clip(sigma_0 * r / r_ref, floor, sigma_0)
```

| Term | Value |
| --- | --- |
| `sigma_0` | population std of the training units' lifetimes (never below `floor`) |
| `r_ref` | mean lifetime of the training units |
| `floor` | `proxy_variance_floor_cycles`, 5 cycles by default |

All three are computed in cycles from the training split and divided by the
target scale. Since the scale is fixed, the NLL's gradient with respect to
the mean is the squared error's weighted by `1 / sigma(r)**2`. End-of-life
windows weigh up to `(sigma_0 / floor)**2`, about 85 times, more than
early-life ones. Under dropout the expected weighted error includes the
spread of the dropout masks, so that spread is penalized most near the end
of life: the epistemic band should narrow there and stay wide early in
life. No weight normalization is needed, since Adam is close to invariant
to the loss scale.

The schedule is an aleatoric story, a time to failure growing more certain,
used to shape the epistemic spread of the dropout masks. The Gamma
variance, fitted and re-fitted afterwards exactly as for `staged+mcd`,
models the aleatoric part independently, so the two can overlap near the
end of life. This is a deliberate choice of the experiment.
