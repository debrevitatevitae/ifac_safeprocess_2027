# Changelog

All notable changes to `ifac_safeprocess_2027` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- The NASA CMAPSS FD001 scripts (`scripts/cmapss/`) and their tests
  (`tests/cmapss/`), moved from `stagereg`'s CMAPSS example. The archive
  environment variables are now `CMAPSS_CACHE_DIR` and `CMAPSS_ARCHIVE`, and
  the default cache directory is `~/.cache/cmapss`.
