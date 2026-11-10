"""summarize.py DIR : per file in files.tsv, |arlmet - HYSPLIT (lin path)| max|mean between nodes for 'none' and 'top' (DIR holds the JSON that compare.py wrote), and the effect sizes."""
import json, sys
from pathlib import Path
H = Path(sys.argv[1])
lab = sys.argv[2] if len(sys.argv) > 2 else ""
for line in open(Path(__file__).parent / "files.tsv").read().splitlines():
    prod, name = line.split("\t")[:2]
    p = H / f"{lab}{prod}__{name}.json"
    if not p.exists():
        print(prod, "missing"); continue
    d = json.load(open(p))
    lv = d["levels"]
    print(f"{prod} {d['path_name']} flag {d['flag']} nz {d['nz']} SHGT {d['has_shgt']} U10M {d['has_u10m']} | HYSPLIT nlvl {lv['nlvl']} aa {lv['aa']} bb {lv['bb']} cc {lv['cc']} | arlmet levels {d.get('levels_arlmet')}")
    for k in ("error_none", "error_top"):
        if k in d: print("  ", k, d[k])
    for lab2 in ("none", "top"):
        if f"seconds_{lab2}" not in d: continue
        parts = []
        for v in ("UWND", "VWND", "TEMP", "pressure"):
            b = d["vars"][v][lab2]["lin_between"]; n = d["vars"][v][lab2]["lin_nodes"]
            parts.append(f"{v} {b.get('max_abs', float('nan')):.3g}|{b.get('mean_abs', float('nan')):.2g} nan{b['n_nan_arlmet']} (node {n.get('max_abs', float('nan')):.2g})")
        print(f"  {lab2:4s} {d[f'seconds_{lab2}']:.2f}s " + "; ".join(parts))
    for eff in ("effect_sigma", "effect_levels", "effect_all", "zindex_vs_lin"):
        print(f"  {eff:13s} " + "; ".join(f"{v} {d['vars'][v][eff].get('max_abs', float('nan')):.3g}|{d['vars'][v][eff].get('mean_abs', float('nan')):.2g}" for v in d["vars"]))
