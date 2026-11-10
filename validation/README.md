# Validation against HYSPLIT

The numbers in `docs/sampling.rst`, `docs/vertical.rst` and `docs/development.rst` that say
arl-met agrees with HYSPLIT (`prfsig`, `prfprs`, `prfecm`, `metlvl`, `zindex`, `adv3nt`, the
11-file table) come from the programs in this folder. They are run by hand, like the slow test;
CI does not run them.

## How it works

HYSPLIT is Fortran, and its routines cannot be called from Python. Each *driver* in `fortran/`
is a short Fortran program linked against HYSPLIT's own `libhysplit.a`: it reads an ARL file,
calls the HYSPLIT routines, and prints their results. The `compare.py` scripts run arl-met on
the same inputs and compare.

| Driver | Calls | Used by |
|---|---|---|
| `fortran/hyrun.f90` | `METLVL`, `PRFSIG`/`PRFPRS`/`PRFECM` on HYSPLIT's internal levels, `ZINDEX`, `ADV3NT`, and `hymodelc`'s `PARTICLE_STILT.DAT` interpolation | `sampling/` |
| `fortran/hyorder.f90` | the same profile routines at the point's own height above each grid point (the order of operations without `model_top`) | `sampling/` |
| `fortran/hyref.f90` | `METINI`/`METSET`, `PAKINP`, the map routines, and the profile routines on real columns | `files/` |

## Layout

```
validation/
  build.sh              builds the three drivers and xtrct_grid into build/ (not tracked)
  fortran/              hyrun.f90, hyorder.f90, hyref.f90
  sampling/             sample_points at heights against HYSPLIT's internal levels
    files.tsv             the 11 met files (product, file name)
    compare.py            one file -> one JSON; reads HYSPLIT's values from hysplit/ if there
    run.sh                every file in files.tsv
    summarize.py          prints the JSON as the docs table
    hysplit/              HYSPLIT's values, one .npz per file (the Fortran output, float32)
    results/              one JSON per file
  files/                labels, grid, values and profile columns of real files against hyref
    files.tsv             the met files (product, file name)
    compare.py, run.sh, summarize.py
    results/              one JSON per file
```

## Running

The met files are not in the repository. `files.tsv` gives each file's name and product; put
the files in one folder (`DATA_DIR`) under those names. NOAA's archive (`arlmet.archives`)
serves GDAS 1° and 0.5°, GFS 0.25°, NAM 12 km, NAMS, HRRR and HRRR v1. The WRF files
(`wrfout_*`) are the Lin group's own `wrf2arl` output, and the `crop` files are `extract_subset`
crops of HRRR; neither can be downloaded.

Compare arl-met with the stored HYSPLIT values (no HYSPLIT needed; run it in a job):

```bash
DATA_DIR=/folder/with/the/met/files validation/sampling/run.sh /tmp/out validation/sampling/hysplit
python validation/sampling/summarize.py /tmp/out
```

Run HYSPLIT again (needs a HYSPLIT library you built, and `gfortran`):

```bash
HYSPLIT_LIB=/path/to/hysplit/library validation/build.sh
DATA_DIR=... validation/sampling/run.sh /tmp/out /tmp/hysplit-values
DATA_DIR=... validation/files/run.sh /tmp/files-out
```

The points are fixed (numpy seed 29 in `sampling/`, seeds 0 to 2 in `files/`), so a run
gives the same points every time. `files/` always needs the drivers: it checks bytes and
decoded fields too, and stores no HYSPLIT values.

## HYSPLIT version

The drivers were linked against `libhysplit.a` built from HYSPLIT **v5.3.2** (the `v5.3.2`
commit of the Lin group's HYSPLIT checkout; later commits there change `advpnt.f`,
`emspnt.f` and `hymodelc.F`, none of which a driver calls). No CHPC module is used.
`build.sh` also builds HYSPLIT's `xtrct_grid` from the same checkout; `tests/test_subset_slow.py`
compares `extract_subset` with it (set `ARLMET_XTRCT_GRID` to use another binary), so that test
is skipped, in CI too, where `validation/build/xtrct_grid` does not exist.

## Stored results

`sampling/results/` and `files/results/` are the JSON that `compare.py` wrote on
2026-11-10 with arl-met at `dc305733` (`main`). They are the numbers the docs quote, and a rerun
on a later arl-met overwrites them. `sampling/hysplit/` holds HYSPLIT's side, so a change to
arl-met can be compared with HYSPLIT without rebuilding anything.
