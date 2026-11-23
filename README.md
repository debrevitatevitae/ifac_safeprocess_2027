# Code for the paper "Stage-based training of probabilistic neural networks for prognostics: an application to the NASA CMAPSS dataset"

[![DOI](https://zenodo.org/badge/1453820781.svg)](https://doi.org/10.5281/zenodo.24443665)

The repository contains the code for the results of the paper submitted to the [IFAC SafeProcess 2027 conference](https://conferences.ifac-control.org/safeprocess2027/), built on the [`stagereg` library](https://github.com/GroupiSP/stagereg) for stage-based probabilistic regression.

## Reproduce the results

The results are reproduced in an [Apptainer](https://apptainer.org/) image that holds the locked dependencies and the NASA C-MAPSS dataset, so no download happens at run time. The scripts are not in the image: they come from this repository, which is bound over the image's `/code` at run time.

Apptainer runs on Linux. On macOS or Windows, run it in a Linux virtual machine, e.g. with [Lima](https://lima-vm.io/) or WSL2.

### 1. Get the image

Download `uv-cmapss.sif` from its [Zenodo record](https://doi.org/10.5281/zenodo.24465173), picking the version whose name matches the release tag of your checkout (`git describe --tags`):

```shell
curl -L -o uv-cmapss.sif "https://zenodo.org/records/24465174/files/uv-cmapss.sif?download=1"
apptainer test uv-cmapss.sif
```

The published image is built for `amd64` (x86_64). On an `arm64` machine, [build it yourself](#build-the-image-yourself). `apptainer test` checks that the dependencies import and the dataset loads.

### 2. Run a script

From the root of the repository:

```shell
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_base.py --seeds 5 --jobs 5 --plot-path img/rul_fd001.png
```

- `-B .:/code` binds the repository over `/code`. It is required: without it, the image has no scripts to run.
- The script path is relative to `scripts/`, so it is `cmapss/fd001_base.py`, not `scripts/cmapss/fd001_base.py`.
- Always pass `--plot-path`. The image has no display, so without it the figure is silently dropped. Paths are relative to the repository root; `img/` is ignored by git.
- The run refuses to start if the repository's `uv.lock` differs from the one the image was built from. Use the image matching your checkout, or build one.

### Parallel runs

Every script takes `--jobs N` to train its independent runs (one per seed, and per swept value in the calibration and sensitivity scripts) in `N` processes. Each run keeps about one CPU core busy, so:

- `N` beyond the number of independent runs does not help, e.g. 5 for `fd001_base.py --seeds 5`, 10 for `fd001_calibration.py --seeds 5` with its two default rates;
- the image uses the host's cores: keep `N` at or below the cores you have, or the cores allocated to your job on a shared machine.

The results do not depend on `N`.

### Build the image yourself

To build the image from [`uv-cmapss.def`](uv-cmapss.def), e.g. on an `arm64` machine or after changing the dependencies:

```shell
apptainer build --fakeroot uv-cmapss.sif uv-cmapss.def
```

This needs `--fakeroot` (or root) and network access to NASA's data server, since the dataset is fetched at build time. The base image is pinned by digest. An image built on `arm64` may differ from the published results in the last digits.

## Without Apptainer

For development, or to run the scripts without the image, install the environment with the [uv package manager](https://docs.astral.sh/uv/):

```shell
uv sync --all-groups
```

Fetch the C-MAPSS dataset into the cache directory (`~/.cache/cmapss`, or `CMAPSS_CACHE_DIR` if set); otherwise the first script to load it fetches it:

```shell
uv run python -m ifac_safeprocess_2027.cmapss.data
```

This also visualizes the sensor trajectories of the FD001/train sub-dataset; pass `--fetch` to only fetch. Then run a script with `uv run python scripts/cmapss/<script_name>.py`, with the same options as above.

## The NASA C-MAPSS dataset

These results are prognostics predictions on the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, described in [(Saxena et al., 2008)](https://ieeexplore.ieee.org/document/4711414). The sensors selected for the study are those with the highest-scoring prognostic indices and they are known from literature, e.g. [(Ramasso and Saxena, 2014)](https://papers.phmsociety.org/index.php/phmconf/article/view/2512).

## Description of the scripts

The files in `scripts/cmapss` reproduce the results of the paper. Below is a description of each script, with an example command.

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

```shell
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_base.py --seeds 10 --jobs 10 --plot-path img/rul_fd001.png
```

### `fd001_calibration.py`

Trains the same two models as `fd001_base.py` at each dropout rate in `--rates` (default 0.3 and 0.9, the same seeds at every rate) and draws their calibration curves on FD001/test in one figure: the empirical coverage of each model's central interval against its nominal confidence level (10% to 90% in steps of 10%, and 95%), mean ± std over seeds, against the diagonal of ideal calibration. The colour marks the model, the line style the dropout rate. It also prints the coverage per level as a table, one column per model and rate. `--jobs` trains the (rate, seed) runs in parallel, so up to `len(rates) × seeds` processes help.

```shell
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_calibration.py --seeds 5 --jobs 10 --plot-path img/calibration_fd001.pdf
```

### `fd001_dropout_sensitivity.py`

Analyzes the sensitivity of the joint and staged models to the dropout rate used in the MC dropout layers of the LSTM. It plots RMSE, 95% coverage, NLL and the epistemic fraction (the epistemic share of the predictive variance, see `GLOSSARY.md`) against the rate, in a figure sized for the full text width of a two-column paper. At rate 0 each model takes a single draw, so the epistemic fraction is left out there.

```shell
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_dropout_sensitivity.py --seeds 5 --jobs 30 --plot-path img/fd001_dropout_sensitivity.pdf
```

### `fd001_train_units_sensitivity.py`

Analyzes the sensitivity of the joint and staged models to the units used to train the networks, at dropout rate 0.9. It plots 95% coverage and the epistemic fraction, stacked over the fraction of training units, in a figure sized for one column of a two-column paper.

```shell
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_train_units_sensitivity.py --seeds 5 --jobs 28 --plot-path img/fd001_train_units_sensitivity.pdf
```

## Citation

If you use this code, please cite it through its Zenodo archive. The DOI below
always resolves to the latest release; each release also has its own version DOI,
listed on the [Zenodo record](https://doi.org/10.5281/zenodo.24443665).

```bibtex
@misc{tostibalducci2027code,
  author       = {Tosti Balducci, Giorgio},
  title        = {{ifac\_safeprocess\_2027}: Code for ``Stage-based training of
                  probabilistic neural networks for prognostics: an application
                  to the NASA CMAPSS dataset''},
  year         = {2026},
  publisher    = {Zenodo},
  howpublished = {\url{https://github.com/debrevitatevitae/ifac_safeprocess_2027}},
  doi          = {10.5281/zenodo.24443665}
}
```

The citation metadata is also in [`CITATION.cff`](CITATION.cff), which GitHub
renders as a "Cite this repository" button. The paper's reference will be added
here once it is published.
