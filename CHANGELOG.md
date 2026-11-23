# Changelog

All notable changes to `ifac_safeprocess_2027` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `CITATION.cff`, with the paper as preferred citation, and project metadata
  (description, author name, license, keywords, URLs) in `pyproject.toml`.
- The Zenodo DOI (10.5281/zenodo.24443665): a badge and a Citation section with
  a BibTeX entry in `README.md`, and the concept and version DOIs in
  `CITATION.cff`.
- The Apptainer image for `ifac-safeprocess-pre`, published on Zenodo
  (concept DOI 10.5281/zenodo.24465173), linked from `README.md` and listed in
  `CITATION.cff`.
- `docs/releasing-the-image.md`: how to build, check and publish the Apptainer
  image on Zenodo.
- A guard in the Apptainer runscript: it refuses to run when the bound
  repository's `uv.lock` differs from the image's, or when the repository is not
  bound to `/code`.

### Changed

- `README.md`: the Apptainer image is the way to reproduce the results; running
  with `uv` directly moves to a "Without Apptainer" section.
- `uv-cmapss.def` pins the base image by digest
  (`uv:0.9.30-python3.13-bookworm-slim`), the one the published image was built
  from.

### Removed

- The FD003 out-of-distribution script (`scripts/cmapss/fd003_ood.py`), which
  trained on FD001/train and scored on FD003/test.

## [ifac-safeprocess-pre] - 2026-11-22

### Added

- The NASA CMAPSS FD001 scripts (`scripts/cmapss/`) and their tests
  (`tests/cmapss/`), moved from `stagereg`'s CMAPSS example. The archive
  environment variables are now `CMAPSS_CACHE_DIR` and `CMAPSS_ARCHIVE`, and
  the default cache directory is `~/.cache/cmapss`.

### Removed

- The staged MC-Dropout proxy-variance model (`staged+mcd+proxy`), with
  `ProxyVarianceNLL`, `ProxyVarianceSchedule`, `fit_proxy_variance_schedule`,
  `unit_lifetimes` and the `proxy_variance_*` config fields. It showed no clear
  benefit once the mean network became an LSTM, and is out of the paper's
  scope. `fd001_base.py` now compares `joint+mcd` and `staged+mcd`; the
  dropout, train-units and FD003 OOD scripts use `staged+mcd` as their
  `"staged"` model. `train_mean_model` and `_train_staged_mcd` drop their
  `objective` parameter: the mean stage always trains on the squared error.

### Changed

- The staged MC-Dropout models' Gamma stage fits **held-out residuals**, on the
  validation split, in both its fits (ADR 0001). The joint MC-Dropout model
  trains on train ∪ validation, with no validation monitoring.
- `validation_fraction` defaults to 0.3 (was 0.2).
- `scripts/cmapss/compare.py` is renamed `fd001_base.py`, and the FD003 OOD prototype
  `prototype_fd003_ood.py` is renamed `fd003_ood.py`.
- `fd001_base.py` re-draws the train/validation split per seed; `compare_models`
  takes a `prepare` callable and each `SeedComparison` keeps its prepared data.
- The figure draws the first seed's lifetime-spanning **train-split** units.
- `fd003_ood.py` drops `--metrics-plot-path`: `--plot-path` now saves both
  figures, the metrics one as `<stem>_metrics<suffix>`, so the metrics figure is
  no longer lost when only `--plot-path` is passed.

[Unreleased]: https://github.com/debrevitatevitae/ifac_safeprocess_2027/compare/ifac-safeprocess-pre...HEAD
[ifac-safeprocess-pre]: https://github.com/debrevitatevitae/ifac_safeprocess_2027/releases/tag/ifac-safeprocess-pre
