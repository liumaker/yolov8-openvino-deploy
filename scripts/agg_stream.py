"""Aggregate results/stream_<model>_s<N>_r<R>.json into a median table."""
import glob
import json
import re
import statistics

rows: dict[tuple[str, int], list[dict]] = {}
for f in glob.glob("results/stream_*.json"):
    d = json.load(open(f, encoding="utf-8"))
    m = re.match(r".*stream_(fp32|int8)_s(\d+)_r(\d+)", f)
    if not m:
        continue
    rows.setdefault((m.group(1), int(m.group(2))), []).append(d)

print("| model | streams | fps median | fps min-max | cpu proc % | infer ms median | post ms |")
print("|---|---|---|---|---|---|---|")
for (model, s), ds in sorted(rows.items()):
    fps = [d["fps_total"] for d in ds]
    print(f"| {model} | {s} | {statistics.median(fps):.1f} | {min(fps):.1f}–{max(fps):.1f} | "
          f"{statistics.median([d['cpu_proc_pct_mean'] for d in ds]):.0f} | "
          f"{statistics.median([d['infer_ms_mean'] for d in ds]):.1f} | "
          f"{statistics.median([d['post_ms_mean'] for d in ds]):.1f} |")
