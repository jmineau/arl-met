"""Compare arlmet's reading of one real ARL file with HYSPLIT's own library (fortran/hyref).

usage: compare_file.py PATH OUT_JSON [--values N] [--columns]

Checks, each against hyref (HYSPLIT's METINI/METSET, PAKINP, map and profile routines):
  labels : every index record (all fields, levels, variables, checksums) and every data record
           header, as metset/metinp read them; arlmet's record layout (level, variable, DIF pairs)
  grid   : METINI's grid (projection fields, nx/ny/nz, flag, delta, first/last time) and lat/lon of
           sample cells (CXY2LL/GBL2LL) and grid positions of sample lat/lons (CLL2XY/GBL2XY)
  values : full-field unpack of a sample of records (raw, and parent + DIF as metinp adds them)
  column : PRFSIG/PRFPRS/PRFECM profiles at a few cells against arlmet.sample_points at the
           same heights (nearest grid point, z_kind="agl"): pressure, TEMP
"""

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
from arlmet.record import Header
from arlmet.record import IndexRecord

HYREF = Path(__file__).resolve().parent.parent / "build" / "hyref"
f32 = np.float32


def run_hyref(path, mode, extra_lines, workdir):
    d = str(Path(path).parent) + "/"
    stdin = f"{d}\n{Path(path).name}\n{mode}\n" + "".join(f"{x}\n" for x in extra_lines)
    r = subprocess.run([str(HYREF)], input=stdin, text=True, cwd=workdir, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"hyref {mode} failed: {r.stdout[-2000:]} {r.stderr[-2000:]}")
    return r


def parse_labels(fn):
    steps, recs = [], {}
    cur = None
    for line in open(fn):
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "INDX":
            cur = {"irec": int(tok[1]), "ymdh": tuple(int(t) for t in tok[2:6]), "ic": int(tok[6]),
                   "mn": int(tok[7]), "cgrid": tok[8], "model": tok[9], "icx": int(tok[10]),
                   "nndx": int(tok[11]), "levels": []}
            steps.append(cur)
        elif tok[0] == "proj":
            cur["proj"] = [float(t) for t in tok[1:]]
        elif tok[0] == "nxyz_flag_lenh":
            cur["nx"], cur["ny"], cur["nz"], cur["flag"], cur["lenh"] = (int(t) for t in tok[1:])
        elif tok[0] == "level":
            nv = int(tok[3])
            vs = [(tok[4 + 2 * k], int(tok[5 + 2 * k])) for k in range(nv)]
            cur["levels"].append((float(tok[2]), vs))
        elif tok[0] == "REC":
            recs[int(tok[1])] = {"ymdh": tuple(int(t) for t in tok[2:6]), "ic": int(tok[6]),
                                 "il": int(tok[7]), "varb": tok[8], "nexp": int(tok[9]),
                                 "prec": float(tok[10]), "var1": float(tok[11])}
        elif tok[0] == "BADLABEL":
            recs[int(tok[1])] = {"bad": line}
    return steps, recs


def check_labels(path, f, work, out):
    t0 = time.time()
    run_hyref(path, "labels", [], work)
    steps, recs = parse_labels(Path(work) / "labels_out.txt")
    reclen = f.record_length
    problems = []
    fh = f.handle
    # index records, in file order (arlmet's _steps include skipped repeats)
    ar_steps = list(f._steps)
    if len(ar_steps) != len(steps):
        problems.append(f"n index records: arlmet {len(ar_steps)} hysplit {len(steps)}")
    names = ["pole_lat", "pole_lon", "tangent_lat", "tangent_lon", "grid_size", "orientation",
             "cone_angle", "sync_x", "sync_y", "sync_lat", "sync_lon", "reserved"]
    n_rec_checked = 0
    for (t, pos, nbytes), hs in zip(ar_steps, steps):
        irec = pos // reclen + 1
        if irec != hs["irec"]:
            problems.append(f"index record number arlmet {irec} hysplit {hs['irec']}")
            continue
        ix = IndexRecord.from_position(fh, position=pos)
        yy, mm, dd, hh = hs["ymdh"]
        h = ix.header
        if (h.year % 100, h.month, h.day, h.hour) != (yy, mm, dd, hh):
            problems.append(f"INDX {irec} date arlmet {h.year}-{h.month}-{h.day} {h.hour} hysplit {hs['ymdh']}")
        tt = pd.Timestamp(year=h.year, month=mm, day=dd, hour=hh) + pd.Timedelta(minutes=hs["mn"])
        if ix.time != tt or t != tt:
            problems.append(f"INDX {irec} time arlmet {ix.time}/{t} hysplit {tt}")
        if ix.source != hs["model"].strip():
            problems.append(f"INDX {irec} source {ix.source!r} vs {hs['model']!r}")
        if ix.forecast != hs["icx"] or ix.minutes != hs["mn"]:
            problems.append(f"INDX {irec} forecast/min arlmet {ix.forecast}/{ix.minutes} hysplit {hs['icx']}/{hs['mn']}")
        for n_, v in zip(names, hs["proj"]):
            if f32(getattr(ix, n_)) != f32(v):
                problems.append(f"INDX {irec} {n_} arlmet {getattr(ix, n_)!r} hysplit {v!r}")
        if (ix.nx % 1000, ix.ny % 1000, ix.nz, ix.vertical_flag) != (hs["nx"], hs["ny"], hs["nz"], hs["flag"]):
            problems.append(f"INDX {irec} nx/ny/nz/flag arlmet {(ix.nx, ix.ny, ix.nz, ix.vertical_flag)} hysplit {(hs['nx'], hs['ny'], hs['nz'], hs['flag'])}")
        if ix.index_length != hs["lenh"]:
            problems.append(f"INDX {irec} index_length arlmet {ix.index_length} stored {hs['lenh']}")
        if hs["nndx"] != 1:
            problems.append(f"INDX {irec} spans {hs['nndx']} records in HYSPLIT")
        if len(ix.levels) != len(hs["levels"]):
            problems.append(f"INDX {irec} n levels {len(ix.levels)} vs {len(hs['levels'])}")
        rno = irec + 1
        for lv, (hgt, vs) in zip(ix.levels, hs["levels"]):
            if f32(lv.height) != f32(hgt):
                problems.append(f"INDX {irec} level {lv.level} height arlmet {lv.height!r} hysplit F6.2 {hgt!r}")
            av = [(n, vi.checksum) for n, vi in lv.variables]
            if av != vs:
                problems.append(f"INDX {irec} level {lv.level} vars arlmet {av} hysplit {vs}")
            for name, _ in vs:
                r = recs.get(rno)
                if r is None or "bad" in r:
                    problems.append(f"record {rno}: hysplit label unreadable {r}")
                else:
                    raw = f._read_at((rno - 1) * reclen, Header.N_BYTES)
                    ah = Header.from_bytes(raw)
                    n_rec_checked += 1
                    if r["varb"] != name or r["il"] != lv.level:
                        problems.append(f"record {rno}: label {r['varb']} lvl {r['il']} but index says {name} lvl {lv.level}")
                    if (ah.year % 100, ah.month, ah.day, ah.hour) != r["ymdh"]:
                        problems.append(f"record {rno} date arlmet {ah} hysplit {r['ymdh']}")
                    if r["ymdh"] != hs["ymdh"]:
                        problems.append(f"record {rno} date {r['ymdh']} differs from its index {hs['ymdh']}")
                    if ah.forecast != r["ic"] or ah.level != r["il"] or ah.variable != r["varb"] or ah.exponent != r["nexp"]:
                        problems.append(f"record {rno} fields arlmet {(ah.forecast, ah.level, ah.variable, ah.exponent)} hysplit {(r['ic'], r['il'], r['varb'], r['nexp'])}")
                    if f32(ah.precision) != f32(r["prec"]) or f32(ah.initial_value) != f32(r["var1"]):
                        problems.append(f"record {rno} prec/var1 arlmet {ah.precision!r}/{ah.initial_value!r} hysplit {r['prec']!r}/{r['var1']!r}")
                rno += 1
    # arlmet's record objects map to the right records (level, variable), DIFs to a parent on their level
    layout = []
    for t in f.times:
        rs = f[t]
        for rec in rs:
            rno = rec.position // reclen + 1
            r = recs.get(rno)
            if r is None or r.get("varb") != rec.variable or r.get("il") != rec.level:
                layout.append(f"{t} {rec.variable} lvl {rec.level} -> record {rno} {r}")
            for d in rec.diffs:
                dn = d.position // reclen + 1
                rd = recs.get(dn)
                if rd is None or rd.get("il") != rec.level or not rd.get("varb", "").startswith("DIF"):
                    layout.append(f"DIF of {rec.variable} lvl {rec.level} -> record {dn} {rd}")
    problems += layout
    forecasts = sorted({r["ic"] for r in recs.values() if "ic" in r})
    out["labels"] = {"n_index": len(steps), "n_data_records_checked": n_rec_checked,
                     "record_forecasts": forecasts[:20], "index_forecasts": sorted({s["icx"] for s in steps})[:20],
                     "problems": problems[:60], "n_problems": len(problems), "seconds": round(time.time() - t0, 1)}
    return steps, recs


def check_grid(path, f, work, out):
    g = f.grid
    nx, ny = g.nx, g.ny
    rng = np.random.default_rng(0)
    cells = [(1, 1), (nx, 1), (1, ny), (nx, ny), ((nx + 1) // 2, (ny + 1) // 2)]
    cells += [(int(i), int(j)) for i, j in zip(rng.integers(1, nx + 1, 300), rng.integers(1, ny + 1, 300))]
    frac = [(float(x), float(y)) for x, y in zip(rng.uniform(1, nx, 100), rng.uniform(1, ny, 100))]
    xy = [(float(i), float(j)) for i, j in cells] + frac
    coords = g.calculate_coords()
    lon = np.asarray(coords["lon"][1]); lat = np.asarray(coords["lat"][1])
    # sample lat/lons for the inverse: cell centers' lat/lon from arlmet
    if g.is_latlon:
        lls = [(float(lat[j - 1]), float(lon[i - 1])) for i, j in cells[:100]]
    else:
        lls = [(float(lat[j - 1, i - 1]), float(lon[j - 1, i - 1])) for i, j in cells[:100]]
    extra = [len(xy)] + [f"{x} {y}" for x, y in xy] + [len(lls)] + [f"{a!r} {b!r}" for a, b in lls]
    run_hyref(path, "grid", extra, work)
    gl = {}
    xy2ll, ll2xy, levels = [], [], []
    for line in open(Path(work) / "grid_out.txt"):
        tok = line.split()
        if tok[0] == "xy2ll":
            xy2ll.append([float(t) for t in tok[1:]])
        elif tok[0] == "ll2xy":
            ll2xy.append([float(t) for t in tok[1:]])
        elif tok[0] == "level":
            levels.append(float(tok[2]))
        else:
            gl[tok[0]] = tok[1:]
    xy2ll = np.array(xy2ll); ll2xy = np.array(ll2xy)
    problems = []
    # arlmet lat/lon of the same positions
    if g.is_latlon:
        p = g.projection
        a_lat = p.sync_lat + (xy2ll[:, 1] - p.sync_y) * p.tangent_lat
        a_lon = p.sync_lon + (xy2ll[:, 0] - p.sync_x) * p.tangent_lon
    else:
        import pyproj
        tr = pyproj.Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True)
        step = g.projection.grid_size * 1000.0
        a_lon, a_lat = tr.transform((xy2ll[:, 0] - 1) * step, (xy2ll[:, 1] - 1) * step)
    dlon = ((a_lon - xy2ll[:, 3] + 180) % 360) - 180
    dlat = a_lat - xy2ll[:, 2]
    # cell-center coordinates arlmet puts in the Dataset
    ci = np.array(cells)
    if g.is_latlon:
        ds_lat = lat[ci[:, 1] - 1]; ds_lon = lon[ci[:, 0] - 1]
    else:
        ds_lat = lat[ci[:, 1] - 1, ci[:, 0] - 1]; ds_lon = lon[ci[:, 1] - 1, ci[:, 0] - 1]
    n = len(cells)
    ds_dlon = ((ds_lon - xy2ll[:n, 3] + 180) % 360) - 180
    ds_dlat = ds_lat - xy2ll[:n, 2]
    # inverse: arlmet fractional indices (0-based) vs CLL2XY (1-based)
    fx, fy = g.fractional_indices(ll2xy[:, 1], ll2xy[:, 0])
    ex = fx + 1 - ll2xy[:, 2]
    if g.is_latlon:
        ex = ((ex + nx / 2) % nx) - nx / 2 if g.wraps_lon else ex
    ey = fy + 1 - ll2xy[:, 3]
    # km per degree for reporting
    out["grid"] = {
        "hysplit": {k: v for k, v in gl.items()},
        "max_abs_dlat_deg": float(np.nanmax(np.abs(dlat))), "max_abs_dlon_deg": float(np.nanmax(np.abs(dlon))),
        "dataset_coord_max_dlat": float(np.nanmax(np.abs(ds_dlat))), "dataset_coord_max_dlon": float(np.nanmax(np.abs(ds_dlon))),
        "inverse_max_dx_cells": float(np.nanmax(np.abs(ex))), "inverse_max_dy_cells": float(np.nanmax(np.abs(ey))),
    }
    hp = [float(v) for v in gl["proj"]]
    names = ["pole_lat", "pole_lon", "tangent_lat", "tangent_lon", "grid_size", "orientation",
             "cone_angle", "sync_x", "sync_y", "sync_lat", "sync_lon", "reserved"]
    for n_, v in zip(names, hp):
        av = getattr(g.projection, n_, None)
        if n_ == "grid_size" and g.is_latlon:
            out["grid"]["metini_latlon_size_km"] = v  # METINI computes it for lat/lon grids
            continue
        if av is not None and f32(av) != f32(v):
            problems.append(f"projection {n_} arlmet {av!r} METINI {v!r}")
    hnx, hny, hnz = (int(v) for v in gl["nxyz"])
    if (hnx, hny) != (nx, ny) or hnz != len(f.vertical_axis.levels):
        problems.append(f"nx ny nz arlmet {(nx, ny, len(f.vertical_axis.levels))} METINI {(hnx, hny, hnz)}")
    zf, recper, delta = (int(v) for v in gl["zflag_recper_delta_offset_reclen_endrec"][:3])
    if zf != f.vertical_axis.flag:
        problems.append(f"flag arlmet {f.vertical_axis.flag} METINI {zf}")
    if not np.array_equal(np.float32(levels), np.float32(f.vertical_axis.levels)):
        problems.append(f"levels arlmet {list(f.vertical_axis.levels)} METINI {levels}")
    times = f.times
    if len(times) > 1:
        steps_min = sorted({int((b - a).total_seconds() // 60) for a, b in zip(times[:-1], times[1:])})
        if steps_min != [delta]:
            problems.append(f"time spacing arlmet {steps_min} min, METINI delta {delta} (HYSPLIT assumes one spacing)")
    fy_, fm, fd, fh, fmn = (int(v) for v in gl["first"][:5])
    if (times[0].year % 100, times[0].month, times[0].day, times[0].hour, times[0].minute) != (fy_, fm, fd, fh, fmn):
        problems.append(f"first time arlmet {times[0]} METINI {gl['first']}")
    ly, lm, ld, lh, lmn = (int(v) for v in gl["last"][:5])
    if (times[-1].year % 100, times[-1].month, times[-1].day, times[-1].hour, times[-1].minute) != (ly, lm, ld, lh, lmn):
        problems.append(f"last time arlmet {times[-1]} METINI {gl['last']}")
    out["grid"]["problems"] = problems


def check_values(path, f, work, out, n_random):
    reclen = f.record_length
    nrec_file = f.size // reclen
    times = f.times
    pick = set()
    # every record of the first and last time step, plus random records
    for t in {times[0], times[-1]}:
        for rec in f[t]:
            pick.add(rec.position // reclen + 1)
            for d in rec.diffs:
                pick.add(d.position // reclen + 1)
    by_rno = {}
    for t in times:
        for rec in f[t]:
            by_rno[rec.position // reclen + 1] = (rec, None)
            for d in rec.diffs:
                by_rno[d.position // reclen + 1] = (d, rec)
    rng = np.random.default_rng(1)
    all_rnos = sorted(by_rno)
    if n_random:
        pick |= set(rng.choice(all_rnos, size=min(n_random, len(all_rnos)), replace=False).tolist())
    # keep the job's disk use bounded: chunks of 40 records
    pick = sorted(p for p in pick if p in by_rno)
    nx, ny = f.grid.nx, f.grid.ny
    n_equal = n_diff = 0
    worst = []
    hy_raw = {}
    for start in range(0, len(pick), 40):
        chunk = pick[start:start + 40]
        run_hyref(path, "values", [len(chunk)] + chunk, work)
        arr = np.fromfile(Path(work) / "values_out.bin", dtype=">f4").reshape(len(chunk), ny, nx)
        for k, rno in enumerate(chunk):
            rec, parent = by_rno[rno]
            mine = rec._read_own(None)
            ref = arr[k]
            if mine is None:
                # arlmet reads it as missing; HYSPLIT skips such a record
                worst.append(f"record {rno} {rec.variable} missing in arlmet (forecast {rec.header.forecast})")
                continue
            if np.array_equal(mine, ref):
                n_equal += 1
            else:
                n_diff += 1
                dd = np.abs(mine.astype(float) - ref)
                worst.append(f"record {rno} {rec.variable} lvl {rec.level}: {int((dd > 0).sum())} cells differ, max {dd.max():.3g}")
            if parent is None:
                hy_raw[rno] = ref.copy() if rec.diffs else None
            else:
                hy_raw[rno] = ref.copy()
    # parent + DIF as metinp adds them, for pairs whose records were both read
    n_dif_pairs = n_dif_equal = 0
    for rno, (rec, parent) in by_rno.items():
        if parent is not None or not rec.diffs or rno not in hy_raw:
            continue
        drn = [d.position // reclen + 1 for d in rec.diffs]
        if not all(d in hy_raw for d in drn):
            continue
        ref = hy_raw[rno]
        for d in drn:
            ref = ref + hy_raw[d]
        mine = rec.read()
        n_dif_pairs += 1
        if np.array_equal(mine, ref):
            n_dif_equal += 1
        else:
            worst.append(f"parent+DIF record {rno} {rec.variable}: max {np.nanmax(np.abs(mine - ref)):.3g}")
    out["values"] = {"n_records": len(pick), "n_file_records": int(nrec_file), "bit_equal": n_equal,
                     "differ": n_diff, "dif_pairs_checked": n_dif_pairs, "dif_pairs_equal": n_dif_equal,
                     "notes": worst[:30]}


def check_columns(path, f, work, out):
    g = f.grid
    nx, ny = g.nx, g.ny
    ax = f.vertical_axis
    if ax.flag not in (1, 2, 4):
        out["column"] = {"skipped": f"flag {ax.flag}"}
        return
    times = f.times
    ti = len(times) // 2
    t = times[ti]
    # step index in file order, as HYSPLIT counts it
    order = [s[0] for s in f._steps]
    tstep = order.index(t) + 1
    rng = np.random.default_rng(2)
    cells = [((nx + 1) // 2, (ny + 1) // 2)] + [(int(i), int(j)) for i, j in zip(rng.integers(2, nx, 5), rng.integers(2, ny, 5))]
    run_hyref(path, "column", [tstep, len(cells)] + [f"{i} {j}" for i, j in cells], work)
    coords = g.calculate_coords()
    lon = np.asarray(coords["lon"][1]); lat = np.asarray(coords["lat"][1])
    res = []
    for k, (i, j) in enumerate(cells, start=1):
        col = np.loadtxt(Path(work) / f"column_{k}.txt", comments="#")
        head = open(Path(work) / f"column_{k}.txt").readline().split()
        z, p, a = col[:, 0], col[:, 1], col[:, 2]
        keep = (z > 0) & (z <= 15000)
        z, p, a = z[keep], p[keep], a[keep]
        z, p, a = z[::4], p[::4], a[::4]  # every 20 m
        if g.is_latlon:
            plon, plat = float(lon[i - 1]), float(lat[j - 1])
        else:
            plon, plat = float(lon[j - 1, i - 1]), float(lat[j - 1, i - 1])
        pts = pd.DataFrame({"lon": plon, "lat": plat, "z": z, "time": t})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                s = arlmet.sample_points(f, pts, ["pressure", "TEMP"], z_kind="agl", method="nearest")
            except Exception as e:
                res.append({"cell": (i, j), "error": f"{type(e).__name__}: {e}"})
                continue
        dp = s["pressure"].to_numpy() - p
        dt = s["TEMP"].to_numpy() - a
        bands = {}
        for lo, hi in [(0, 100), (100, 1000), (1000, 3000), (3000, 15000)]:
            m = (z > lo) & (z <= hi)
            if m.any():
                bands[f"{lo}-{hi}"] = {"max_abs_dp_hPa": float(np.nanmax(np.abs(dp[m]))) if np.isfinite(dp[m]).any() else None,
                                       "mean_dp": float(np.nanmean(dp[m])) if np.isfinite(dp[m]).any() else None,
                                       "max_abs_dT_K": float(np.nanmax(np.abs(dt[m]))) if np.isfinite(dt[m]).any() else None,
                                       "n_nan": int(np.isnan(dp[m]).sum()), "n": int(m.sum())}
        res.append({"cell": (i, j), "lonlat": (plon, plat), "flags_zqutps": head[4:10], "zsfc_p0": head[10:12],
                    "z_top": float(z.max()) if len(z) else None, "bands": bands})
    out["column"] = {"time": str(t), "step": tstep, "cells": res}


def main():
    path = sys.argv[1]
    out_json = sys.argv[2]
    n_random = int(sys.argv[sys.argv.index("--values") + 1]) if "--values" in sys.argv else 40
    do_cols = "--columns" in sys.argv
    out = {"path_name": Path(path).name, "size": os.path.getsize(path)}
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as work:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                f = arlmet.File(path)
                out["arlmet"] = {"source": f.source, "flag": f.vertical_axis.flag, "nlev": len(f.vertical_axis.levels),
                                 "levels": [float(x) for x in f.vertical_axis.levels], "offset": f.vertical_axis.offset,
                                 "nx": f.grid.nx, "ny": f.grid.ny, "projection": repr(f.grid.projection.params),
                                 "times": [str(f.times[0]), str(f.times[-1])], "n_times": len(f.times),
                                 "check": f.check()[:20]}
            out["arlmet"]["warnings"] = [str(x.message)[:300] for x in w][:10]
        except Exception as e:
            out["arlmet_error"] = f"{type(e).__name__}: {e}"
            json.dump(out, open(out_json, "w"), indent=1, default=str)
            return
        for name, fn in [("labels", lambda: check_labels(path, f, work, out)),
                         ("grid", lambda: check_grid(path, f, work, out)),
                         ("values", lambda: check_values(path, f, work, out, n_random)),
                         ("column", lambda: check_columns(path, f, work, out) if do_cols else None)]:
            try:
                fn()
            except Exception as e:
                import traceback
                out[name + "_error"] = traceback.format_exc()[-3000:]
            json.dump(out, open(out_json, "w"), indent=1, default=str)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
