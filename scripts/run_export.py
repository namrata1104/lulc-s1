"""Run the full GEE export (same as notebook 01, section 4) as a background job.

    nohup python scripts/run_export.py > export.log 2>&1 &
    tail -f export.log

Resumable: tiles that already exist are skipped.
Add --no-temporal to skip the optional training-regions-in-2025/26 split (test_temporal).
Add --workers=1 when the Earth Engine project is in restricted mode (quota used up).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, gee_data as G  # noqa: E402

G.init()
jobs = C.export_jobs(include_temporal="--no-temporal" not in sys.argv)
print(f"{len(jobs)} area-months", flush=True)

logs, t0 = [], time.time()
workers = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--workers=")), 4))
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(G.export_aoi_month, n, a, m): (n, m) for n, a, m in jobs}
    for i, f in enumerate(as_completed(futures), 1):
        n, m = futures[f]
        try:
            log = f.result()
        except Exception as e:
            log = dict(aoi=n, month=m, status=f"error: {e}")
        logs.append(log)
        print(f"[{i}/{len(jobs)}] {time.time() - t0:6.0f}s  {n} {m}: {log['status']}", flush=True)

logs = pd.DataFrame(logs).sort_values(["aoi", "month"])
logs.to_csv(C.DATA_DIR / "export_log.csv", index=False)
print(logs.status.value_counts().to_string(), flush=True)
print("DONE", flush=True)
