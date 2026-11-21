# Code for the paper "Stage-based training of probabilistic neural networks for prognostics: an application to the NASA CMAPSS dataset"

The repository contains the code for the results of the paper submitted to the [IFAC SafeProcess 2027 conference](https://conferences.ifac-control.org/safeprocess2027/), built on the [`stagereg` library](https://github.com/GroupiSP/stagereg) for stage-based probabilistic regression.

Install the environment with the [uv package manager](https://docs.astral.sh/uv/):

```shell
uv sync --all-groups
```

Run the scripts with:

```shell
uv run python scripts/cmapss/<script_name>.py
```

Every script takes `--jobs N` to train its independent runs (seeds, and the swept values in the sensitivity scripts) in `N` processes, up to the number of CPU cores. The results do not depend on `N`.

## Download the NASA C-MAPSS Dataset

These results are prognostics prediction on the NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset, described in [(Saxena et al., 2008)](https://ieeexplore.ieee.org/document/4711414).

To download the dataset in your local machine and store it in a cache directory, run the following command, previous to running the scripts:

```shell
uv run python -m ifac_safeprocess_2027.cmapss.data
```

Running the script also visualizes the sensor trajectories for the FD001/train sub-dataset. The sensors selected for the study are those with the highest-scoring prognostic indices and they are known from literature, e.g. [(Ramasso and Saxena, 2014)](https://papers.phmsociety.org/index.php/phmconf/article/view/2512).

## Description of the scripts

The files in `scripts/cmapss` can be run to reproduce the results of the paper. Below is a description of each script.

### `fd001_base.py`

Trains two different models on the FD001/train sub-dataset and compares their performance on the FD001/test units. It also visualizes the predictions and the uncertainty estimates for a few test units in the train set.

The models compared are:

- Joint trained LSTM with MC dropout (baseline)
- Stage-based trained LSTM mean with MC dropout, and a small CNN for the variance

The metrics scored are:

- Root Mean Squared Error (RMSE)
- 95% interval coverage
- Continuous Ranked Probability Score (CRPS)
- Negative Log-Likelihood (NLL)

### `fd001_calibration.py`

Trains the same two models as `fd001_base.py` at each dropout rate in `--rates` (default 0.3 and 0.9, the same seeds at every rate) and draws their calibration curves on FD001/test in one figure: the empirical coverage of each model's central interval against its nominal confidence level (10% to 90% in steps of 10%, and 95%), mean ± std over seeds, against the diagonal of ideal calibration. The colour marks the model, the line style the dropout rate. It also prints the coverage per level as a table, one column per model and rate. `--jobs` trains the (rate, seed) runs in parallel, so up to `len(rates) × seeds` processes help.

```shell
uv run python scripts/cmapss/fd001_calibration.py --seeds 5 [--rates 0.3 0.9] [--jobs 10] [--plot-path calibration.png]
```

### `fd001_dropout_sensitivity.py`

Analyzes the sensitivity of the joint and staged models to the dropout rate used in the MC dropout layers of the LSTM. It plots RMSE, 95% coverage, NLL and the epistemic fraction (the epistemic share of the predictive variance, see `GLOSSARY.md`) against the rate, in a figure sized for the full text width of a two-column paper. At rate 0 each model takes a single draw, so the epistemic fraction is left out there.

### `fd001_train_units_sensitivity.py`

Analyzes the sensitivity of the joint and staged models to the units used to train the networks, at dropout rate 0.9. It plots 95% coverage and the epistemic fraction, stacked over the fraction of training units, in a figure sized for one column of a two-column paper.

### `fd003_ood.py`

Studies the out-of-distribution (OOD) performance of the baseline and stage-trained model. The models are trained as usual on the FD001/train sub-dataset, but evaluated on the FD003/test sub-dataset, which contains two different failure modes. The metrics scored are:

- Root Mean Squared Error (RMSE)
- 95% interval coverage
- Continuous Ranked Probability Score (CRPS)
- Ratio between the 95% interval width on FD003 test units and the 95% interval width on FD001 test units.
