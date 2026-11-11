# AGENTS.md

`ifac_safeprocess_2027` is a repo with the code for the results submitted to the [IFAC SafeProcess 2027 conference](https://conferences.ifac-control.org/safeprocess2027/).

## The why

Reproducibility: running the repo scripts should produce the same results showed in the paper through the years.

## The what

- The repo builds on top of the Python library [GroupiSP/stagereg](https://github.com/GroupiSP/stagereg).
- It contains scripts to reproduce tables and figures in the paper.
- It uses `uv` to manage the virtual environment and dependencies.
- Documentation is left to the `README.md` file.

## The how

Before presenting changes for review, run

- `uv run prek run --all-files`
- `uv run pytest`
