# Changelog

All notable changes to `ifac_safeprocess_2027` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- The NASA CMAPSS FD001 scripts (`scripts/cmapss/`) and their tests
  (`tests/cmapss/`), moved from `stagereg`'s CMAPSS example. The archive
  environment variables are now `CMAPSS_CACHE_DIR` and `CMAPSS_ARCHIVE`, and
  the default cache directory is `~/.cache/cmapss`.

### Changed

- The staged MC-Dropout models' Gamma stage fits **held-out residuals**, on the
  validation split, in both its fits (ADR 0001). The joint MC-Dropout model
  trains on train ∪ validation, with no validation monitoring.
- `validation_fraction` defaults to 0.3 (was 0.2).
- `compare.py` re-draws the train/validation split per seed; `compare_models`
  takes a `prepare` callable and each `SeedComparison` keeps its prepared data.
- The figure draws the first seed's lifetime-spanning **train-split** units.
