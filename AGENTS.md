# AGENTS.md

`ifac_safeprocess_2027` is a repo with the code for the results submitted to the [IFAC SafeProcess 2027 conference](https://conferences.ifac-control.org/safeprocess2027/).

## The why

Conduct experiments to compare traditional joint training of MVE networks and stage based training [(Detlefsen, 2019)](http://arxiv.org/abs/1906.03260), [(Sluijterman, 2024)](https://www.sciencedirect.com/science/article/pii/S0925231224007008), [(Yi, 2025)](http://arxiv.org/abs/2505.02743), implemented by `stagereg` in a prognostics setting. This setting is the C-MAPSS dataset, for now limited to the FD001 subset [(Saxena, 2020)](https://papers.phmsociety.org/index.php/ijphm/article/view/2236).

## The what

- The repo builds on top of the Python library [GroupiSP/stagereg](https://github.com/GroupiSP/stagereg).
- It contains scripts to reproduce tables and figures in the paper.
- It uses `uv` to manage the virtual environment and dependencies.
- Documentation is left to the `README.md` file.

## The how

Before presenting changes for review, run

- `uv run prek run --all-files`
- `uv run pytest`

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`debrevitatevitae/ifac_safeprocess_2027`), managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
