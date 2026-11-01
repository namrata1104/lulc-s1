"""Run the full GEE export (same as notebook 01, section 4) as a background job.

    nohup python scripts/run_export.py > export.log 2>&1 &
    tail -f export.log

Resumable: tiles that already exist are skipped.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, gee_data as G  # noqa: E402

G.init()
jobs = [(n, a, m) for n, a in C.AOIS.items() for m in C.EXPORT_MONTHS]
print(f"{len(jobs)} area-months", flush=True)

logs, t0 = [], time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
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
