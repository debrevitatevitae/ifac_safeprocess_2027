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

Trains three different models on the FD001/train sub-dataset and compares their performance on the FD001/test units. It also visualizes the predictions and the uncertainty estimates for a few test units in the train set.

The models compared are:

- Joint trained CNN with MC dropout (baseline)
- Stage-based trained CNN with MC dropout
- Stage-based trained CNN with MC droput and variance proxy for the mean stage.

The metrics scored are:

- Root Mean Squared Error (RMSE)
- 95% interval coverage
- Continuous Ranked Probability Score (CRPS)
- Negative Log-Likelihood (NLL)

### `fd001_dropout_sensitivity.py`

Analyzes the sensitivity of each of the metrics in `fd001_base.py` to the dropout rate used in the MC dropout layers of the CNN.

### `fd001_train_units_sensitivity.py`

Analyzes the sensitivity of each of the metrics in `fd001_base.py` to the units used to train the CNN.

### `fd003_ood.py`

Studies the out-of-distribution (OOD) performance of the baseline and stage-trained model. The models are trained as usual on the FD001/train sub-dataset, but evaluated on the FD003/test sub-dataset, which contains two different failure modes. The metrics scored are:

- Root Mean Squared Error (RMSE)
- 95% interval coverage
- Continuous Ranked Probability Score (CRPS)
- Ratio between the 95% interval width on FD003 test units and the 95% interval width on FD001 test units.
