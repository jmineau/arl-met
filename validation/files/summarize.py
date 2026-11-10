"""summarize.py DIR : print what compare.py found for each file (DIR holds its JSON)."""
import json, glob, sys
from pathlib import Path
for fn in sorted(glob.glob(f"{sys.argv[1]}/*__*.json")):
    d = json.load(open(fn)); name = Path(fn).stem
    print("=====", name)
    if "arlmet_error" in d: print("  ARLMET ERROR", d["arlmet_error"][:300]); continue
    a = d["arlmet"]; print(f"  src={a['source']} flag={a['flag']} nlev={a['nlev']} nx={a['nx']} ny={a['ny']} times={a['times']} n={a['n_times']} offset={a['offset']} check={a['check'][:3]} warn={a['warnings'][:2]}")
    for k in ("labels_error","grid_error","values_error","column_error"):
        if k in d: print("  ", k, d[k][-600:])
    L = d.get("labels", {}); print(f"  labels: idx={L.get('n_index')} recs={L.get('n_data_records_checked')} problems={L.get('n_problems')} {L.get('problems', [])[:4]} fc_idx={L.get('index_forecasts')} fc_rec={L.get('record_forecasts')}")
    G = d.get("grid", {}); print(f"  grid: dlat={G.get('max_abs_dlat_deg')} dlon={G.get('max_abs_dlon_deg')} ds={G.get('dataset_coord_max_dlat')},{G.get('dataset_coord_max_dlon')} inv={G.get('inverse_max_dx_cells')},{G.get('inverse_max_dy_cells')} problems={G.get('problems')}")
    V = d.get("values", {}); print(f"  values: n={V.get('n_records')} equal={V.get('bit_equal')} differ={V.get('differ')} dif={V.get('dif_pairs_equal')}/{V.get('dif_pairs_checked')} notes={V.get('notes', [])[:3]}")
    C = d.get("column", {})
    if "skipped" in C: print("  column skipped", C["skipped"])
    errs = [c for c in C.get("cells", []) if "error" in c]
    if errs: print("  column errors", len(errs), errs[0]["error"][:200])
    agg = {}
    for c in C.get("cells", []):
        for k, v in c.get("bands", {}).items():
            a = agg.setdefault(k, [0, 0, 0, 0])
            a[0] = max(a[0], v["max_abs_dp_hPa"] or 0); a[1] = max(a[1], v["max_abs_dT_K"] or 0); a[2] += v["n_nan"]; a[3] += v["n"]
    if agg: print("  column (max over cells): " + " ".join(f"{k}: dp {a[0]:.4f} dT {a[1]:.4f} nan {a[2]}/{a[3]}" for k, a in agg.items()))
