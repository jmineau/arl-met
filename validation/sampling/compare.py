"""arlmet.sample_points at heights against HYSPLIT's own library on its internal levels.

usage: python -I compare.py PATH OUT_JSON [--npt N] [--time I] [--zdata Z] [--hysplit FILE.npz]

The interpreter running this script is the arlmet under test. For one time step (the middle one
unless --time): N random interior positions (numpy seed 29, so the same points every run) plus 20
grid nodes, at HEIGHTS above the point's ground.

--hysplit FILE.npz: if FILE exists, HYSPLIT's values are read from it and the Fortran drivers are
not run (``results/`` holds these, so the comparison needs no HYSPLIT build); if not, the drivers
run (build them with ``build.sh``) and write FILE.

References, all on HYSPLIT's library:
- hyrun (fortran/hyrun): the run's internal grid (METLVL for CONTROL top Z, ZMDL = max(Z, 25000)),
  PRFSIG/PRFPRS/PRFECM on the internal levels at each of the four nodes (extrapolating above the
  data), then ZINDEX + ADV3NT ("zindex": the advection path) and hymodelc's PARTICLE_STILT.DAT
  interpolation ("lin").
- hyorder: mode 0, output levels at the point's height above each node's ground
  (what arlmet does without model_top); mode 1, the same with the sigma scaling only.

arlmet runs with model_top=None ("none": each grid point's column at the point's own height) and with model_top=Z ("top": HYSPLIT's internal levels).
"""

import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import arlmet

HERE = Path(__file__).resolve().parent
HYRUN = HERE.parent / "build" / "hyrun"
HYORDER = HERE.parent / "build" / "hyorder"
HEIGHTS = [2, 3.7, 5, 10, 13, 20, 27, 30, 50, 64, 75, 100, 150, 200, 300, 333, 500, 750, 1000,
           1234, 1500, 2000, 3000, 4321, 5000, 8000, 12000, 16000, 20000, 24000]
NAMES = ["UWND", "VWND", "TEMP", "pressure"]


def _xy_lines(xy):
    return [f"{x:.7f} {y:.7f}" for x, y in xy]


def run_hyorder(path, t1, xy, mode, work):
    lines = [str(Path(path).parent) + "/", Path(path).name, str(t1), str(len(HEIGHTS)),
             " ".join(str(h) for h in HEIGHTS), str(mode), str(len(xy))] + _xy_lines(xy)
    r = subprocess.run([str(HYORDER)], input="\n".join(lines) + "\n", text=True, cwd=work,
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    out = np.loadtxt(Path(work) / "out.txt", comments="#")
    vals = out[:, 2:6].reshape(len(xy), len(HEIGHTS), 4)
    return {name: vals[:, :, q] for q, name in enumerate(NAMES)}


def run_hyrun(path, t1, xy, zdata, work):
    lines = [str(Path(path).parent) + "/", Path(path).name, str(t1), str(len(HEIGHTS)),
             " ".join(str(h) for h in HEIGHTS), str(zdata), str(len(xy))] + _xy_lines(xy)
    r = subprocess.run([str(HYRUN)], input="\n".join(lines) + "\n", text=True, cwd=work,
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    out = np.loadtxt(Path(work) / "out.txt", comments="#")
    pts = np.loadtxt(Path(work) / "points.txt", ndmin=2)
    lv = open(Path(work) / "levels.txt").read().split()
    levels = {"nlvl": int(lv[0]), "aa": float(lv[1]), "bb": float(lv[2]), "cc": float(lv[3]),
              "zmdl": float(lv[4]), "heights": [float(v) for v in lv[5:]]}
    n = len(xy)
    zidx = out[:, 2:6].reshape(n, len(HEIGHTS), 4)
    lin = out[:, 8:12].reshape(n, len(HEIGHTS), 4)
    zx = out[:, 7].reshape(n, len(HEIGHTS))
    return pts, levels, ({k: zidx[:, :, q] for q, k in enumerate(NAMES)},
                         {k: lin[:, :, q] for q, k in enumerate(NAMES)}, zx)


def save_hysplit(path, pts, levels, true_z, true_lin, zx, ref0, ref1, seconds):
    """HYSPLIT's values for the points, as float32 in one compressed file (what the Fortran printed)."""
    arrays = {"pts": pts, "zx": zx, "seconds": np.array(seconds), "levels_json": np.array(json.dumps(levels))}
    for label, d in (("zindex", true_z), ("lin", true_lin), ("order0", ref0), ("order1", ref1)):
        for name in NAMES:
            arrays[f"{label}__{name}"] = np.asarray(d[name], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_hysplit(path):
    z = np.load(path)
    get = lambda label: {n: z[f"{label}__{n}"].astype(float) for n in NAMES}  # noqa: E731
    return (z["pts"], json.loads(str(z["levels_json"])), (get("zindex"), get("lin"), z["zx"]),
            get("order0"), get("order1"), float(z["seconds"]))


def stats(a, h):
    a = np.asarray(a, dtype=float); h = np.asarray(h, dtype=float)
    ok = np.isfinite(a) & np.isfinite(h)
    res = {"n": int(ok.sum()), "n_nan_arlmet": int((~np.isfinite(a) & np.isfinite(h)).sum())}
    if ok.any():
        d = np.abs(a[ok] - h[ok])
        res.update(max_abs=float(d.max()), mean_abs=float(d.mean()),
                   p99_abs=float(np.percentile(d, 99)))
    return res


def by_height(a, h):
    out = {}
    for q, z in enumerate(HEIGHTS):
        d = np.abs(a[:, q] - h[:, q])
        n_nan = int((~np.isfinite(a[:, q]) & np.isfinite(h[:, q])).sum())
        d = d[np.isfinite(d)]
        out[str(z)] = [float(d.max()), float(d.mean()), n_nan] if d.size else [None, None, n_nan]
    return out


BANDS = {"below_10m": lambda z: z < 10, "10m_5km": lambda z: (z >= 10) & (z <= 5000),
         "8_24km": lambda z: z >= 8000}


def bands(a, h):
    z = np.array(HEIGHTS, dtype=float)
    return {b: stats(a[:, sel(z)], h[:, sel(z)]) for b, sel in BANDS.items()}


def main():
    path, out_json = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    npt = int(args[args.index("--npt") + 1]) if "--npt" in args else 200
    zdata = float(args[args.index("--zdata") + 1]) if "--zdata" in args else 25000.0
    warnings.simplefilter("ignore")
    f = arlmet.File(path)
    times = list(f.times)
    ti = int(args[args.index("--time") + 1]) if "--time" in args else len(times) // 2
    t = times[ti]
    g = f.grid
    rng = np.random.default_rng(29)
    m = 3 if g.nx > 10 else 0
    x0 = rng.uniform(m, g.nx - 1 - m, npt); y0 = rng.uniform(m, g.ny - 1 - m, npt)
    nx_ = rng.integers(m, g.nx - 1 - m, 20).astype(float); ny_ = rng.integers(m, g.ny - 1 - m, 20).astype(float)
    x0 = np.concatenate([x0, nx_]); y0 = np.concatenate([y0, ny_])
    xy = list(zip(x0 + 1, y0 + 1))
    hysplit_npz = Path(args[args.index("--hysplit") + 1]) if "--hysplit" in args else None
    if hysplit_npz is not None and hysplit_npz.exists():
        pts, levels, (true_z, true_lin, zx), ref0, ref1, t_h = load_hysplit(hysplit_npz)
    else:
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as work:
            t_h = time.perf_counter()
            pts, levels, (true_z, true_lin, zx) = run_hyrun(path, ti + 1, xy, zdata, work)
            t_h = time.perf_counter() - t_h
            ref0 = run_hyorder(path, ti + 1, xy, 0, work)
            ref1 = run_hyorder(path, ti + 1, xy, 1, work)
        if hysplit_npz is not None:
            save_hysplit(hysplit_npz, pts, levels, true_z, true_lin, zx, ref0, ref1, t_h)
    lat, lon = pts[:, 3], pts[:, 4]
    nh = len(HEIGHTS)
    rows = pd.DataFrame({"lon": np.repeat(lon, nh), "lat": np.repeat(lat, nh),
                         "z": np.tile(np.array(HEIGHTS, dtype=float), len(lon)), "time": t})
    has_top = "model_top" in inspect.signature(arlmet.sample_points).parameters
    out = {"path_name": Path(path).name, "source": f.source, "flag": f.vertical_axis.flag,
           "nx": g.nx, "ny": g.ny, "nz": len(f.vertical_axis.levels), "arlmet": arlmet.__file__,
           "time": str(t), "npt": npt, "heights": HEIGHTS, "zdata": zdata, "levels": levels,
           "has_shgt": (0, "SHGT") in f[t], "has_u10m": (0, "U10M") in f[t],
           "hyrun_seconds": t_h, "vars": {}}
    if has_top:
        from arlmet.ops.sample import _InternalLevels
        try:
            il = _InternalLevels.for_axis(f.vertical_axis, zdata)
            out["levels_arlmet"] = {"zmdl": il.zmdl, "nlvl": len(il.heights),
                                    "max_abs_height_diff": float(np.max(np.abs(
                                        il.heights - np.array(levels["heights"])))) if len(il.heights) == levels["nlvl"] else None}
        except Exception as e:  # noqa: BLE001
            out["levels_arlmet"] = f"{type(e).__name__}: {e}"
    res = {}
    for label, kw in (("none", {"model_top": None}), ("top", {"model_top": zdata})):
        if label == "top" and not has_top:
            continue
        t0 = time.perf_counter()
        try:
            s = arlmet.sample_points(f, rows, NAMES, z_kind="agl", **kw)
        except Exception as e:  # noqa: BLE001
            out[f"error_{label}"] = f"{type(e).__name__}: {e}"
            continue
        out[f"seconds_{label}"] = time.perf_counter() - t0
        res[label] = {n: s[n].to_numpy(dtype=float).reshape(len(lon), nh) for n in NAMES}
    for name in NAMES:
        v = {}
        for label, r in res.items():
            a = r[name]
            v[label] = {
                "lin_between": stats(a[:npt], true_lin[name][:npt]),
                "lin_nodes": stats(a[npt:], true_lin[name][npt:]),
                "zindex_between": stats(a[:npt], true_z[name][:npt]),
                "lin_by_height": by_height(a[:npt], true_lin[name][:npt]),
                "vs_hyorder0_between": stats(a[:npt], ref0[name][:npt]),
                "lin_bands": bands(a[:npt], true_lin[name][:npt]),
            }
        v["effect_sigma"] = stats(ref1[name][:npt], ref0[name][:npt])
        v["effect_levels"] = stats(true_lin[name][:npt], ref1[name][:npt])
        v["effect_all"] = stats(true_lin[name][:npt], ref0[name][:npt])
        v["effect_all_by_height"] = by_height(ref0[name][:npt], true_lin[name][:npt])
        v["zindex_vs_lin"] = stats(true_z[name][:npt], true_lin[name][:npt])
        v["effect_sigma_bands"] = bands(ref1[name][:npt], ref0[name][:npt])
        v["effect_levels_bands"] = bands(true_lin[name][:npt], ref1[name][:npt])
        v["zindex_vs_lin_by_height"] = by_height(true_z[name][:npt], true_lin[name][:npt])
        v["effect_sigma_by_height"] = by_height(ref1[name][:npt], ref0[name][:npt])
        out["vars"][name] = v
    json.dump(out, open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
