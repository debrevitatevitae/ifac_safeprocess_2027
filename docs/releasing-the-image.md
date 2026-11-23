# Releasing the Apptainer image

The image `uv-cmapss.sif` holds the locked dependencies and the C-MAPSS dataset. It is published on Zenodo, one version per code release, so that readers can run the scripts without building it. Release a new image version whenever `uv.lock` or `uv-cmapss.def` changes, or when the results are regenerated for a new code release.

The image version is named after the code release tag it pairs with (e.g. `ifac-safeprocess-pre`). The runscript refuses a checkout whose `uv.lock` differs from the image's, so a stale image fails loudly instead of running old dependencies.

## 1. Build

On an `amd64` Linux host with Apptainer (the published image is `amd64`), from the repository root at the release tag:

```shell
git checkout <tag>
apptainer build --fakeroot uv-cmapss.sif uv-cmapss.def
```

The build fetches the dataset from NASA and ends by running `%test`.

## 2. Check

```shell
apptainer test uv-cmapss.sif
apptainer inspect uv-cmapss.sif          # build date, base image name and digest
apptainer exec uv-cmapss.sif cat /opt/uv.lock.sha256
sha256sum uv.lock                        # must match the line above
apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_base.py --seeds 1 --plot-path img/smoke.png
```

If the results are regenerated with this image, run the scripts with it before uploading, so the published image is the one behind the numbers.

## 3. Upload to Zenodo

The image has its own Zenodo record ([10.5281/zenodo.24465173](https://doi.org/10.5281/zenodo.24465173)), separate from the code's (the code is archived automatically from GitHub releases; release assets are not).

First release only, create the record:

1. On [zenodo.org](https://zenodo.org), **New upload**.
2. **Reserve DOI** (under *Digital Object Identifier*) before uploading, so the DOI can go into the README first.
3. Upload `uv-cmapss.sif`.
4. Fill in:
   - *Resource type*: Software
   - *Title*: Apptainer image for ifac_safeprocess_2027
   - *Version*: the code release tag, e.g. `ifac-safeprocess-pre`
   - *Description*: the [template below](#description-template), filled in.
   - *License*: BSD 3-Clause, as the code. The dataset inside keeps NASA's terms; say so in the description.
   - *Related works*: "Is supplement to" the code's concept DOI, 10.5281/zenodo.24443665.
   - *Keywords*: apptainer, prognostics, C-MAPSS, remaining useful life, uncertainty quantification
5. **Publish**. A published file cannot be changed, only superseded by a new version.

Later releases: open the record, **New version**, replace the file, update *Version* and the description, **Publish**. The concept DOI keeps resolving to the latest version.

## 4. Update the repository

- `README.md`: the concept DOI and the download URL (`https://zenodo.org/records/<record-id>/files/uv-cmapss.sif?download=1`, with the new version's record ID).
- `CITATION.cff`: the image's concept DOI under `identifiers`, if not there yet.
- `CHANGELOG.md`: note the new image version.

## Description template

Zenodo's description box is a rich-text editor: copy the template from the rendered Markdown preview (in VS Code, `Cmd+Shift+V`) or from GitHub, not from the source, so bold and lists survive the paste. Fill in the `[...]` fields from `apptainer inspect` and `apptainer exec uv-cmapss.sif uv --version`; update the dependency list if `pyproject.toml` changed. The paper's title stays until the paper's own reference replaces it.

> **Apptainer image for ifac_safeprocess_2027**
>
> This Apptainer image is the runtime environment for the code of the paper *"Stage-based training of probabilistic neural networks for prognostics: an application to the NASA CMAPSS dataset"*, submitted to IFAC SafeProcess 2027. The code is archived separately at https://doi.org/10.5281/zenodo.24443665 and developed at https://github.com/debrevitatevitae/ifac_safeprocess_2027.
>
> **Contents**
>
> - The Python environment from the code's `uv.lock`: JAX (CPU), `stagereg` [version], NumPy, pandas, scikit-learn, Matplotlib, seaborn
> - The NASA C-MAPSS dataset, fetched at build time, so no download happens at run time
>
> The image holds no scripts: they come from the code repository, which is bound into the image at run time.
>
> **Usage** (Linux, Apptainer ≥ 1.3, from the root of a checkout of the matching code release)
>
> `apptainer test uv-cmapss.sif`
>
> `apptainer run -B .:/code uv-cmapss.sif cmapss/fd001_base.py --seeds 5 --plot-path img/rul_fd001.png`
>
> See the repository README for every script and its options.
>
> **Versions**
>
> Each version of this record pairs with the code release of the same name. This version: `[release tag]`, [what it was used for, e.g. "the image used to produce the results submitted for review"].
>
> **Build details**
>
> - Architecture: amd64 (x86_64). On arm64, build the image from `uv-cmapss.def` in the code repository.
> - Built: [build date], Apptainer [version]
> - Base image: `ghcr.io/astral-sh/uv` (uv [version], Python [version], Debian [version]), digest `[sha256:...]`
>
> **Dataset credit**
>
> The C-MAPSS data were produced by NASA and are distributed through the NASA Prognostics Center of Excellence Data Set Repository. They are included unmodified and remain under their original terms. If you use them, cite: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation", *International Conference on Prognostics and Health Management (PHM 2008)*, IEEE, 2008. https://doi.org/10.1109/PHM.2008.4711414
>
> **License**
>
> The image's build recipe and the code are under the BSD 3-Clause License. Third-party software in the image and the C-MAPSS dataset keep their own licenses and terms.

Values for the first version (`ifac-safeprocess-pre`): `stagereg` 0.7.0; built 2026-11-20 with Apptainer 1.5.5; uv 0.9.30, Python 3.13.11, Debian 12; digest `sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca`.
