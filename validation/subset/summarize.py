"""Print the subset results (results/*.json, or the folder given) as Markdown tables.

usage: summarize.py [RESULTS_DIR]
"""

import json
import sys
from pathlib import Path

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
for path in sorted(folder.glob("*.json")):
    r = json.loads(path.read_text())
    print(f"### {r['case']}: {r['file']} at {r['time']}, box {r['bbox']}, {r['levels']} levels\n")
    print(
        f"xtrct_grid {r['xtrct_grid_seconds']:.1f} s, extract_subset "
        f"{r['extract_subset_seconds']:.1f} s; source {r['source_bytes'] / 1e9:.2f} GB, "
        f"crops {r['xtrct_grid_bytes'] / 1e6:.1f} MB (xtrct_grid) and "
        f"{r['extract_subset_bytes'] / 1e6:.1f} MB (extract_subset)\n"
    )
    print("| Variable | xtrct_grid error | extract_subset error | Between the crops | Precision |")
    print("|---|---|---|---|---|")
    for name, v in r["variables"].items():
        print(
            f"| {name} | {v['xtrct_grid_error']:.4g} | {v['extract_subset_error']:.4g} "
            f"| {v['crop_difference']:.4g} | {v['precision']:.4g} |"
        )
    print()
