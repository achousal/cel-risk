"""Aggregate 16 panel x model cells into a PRAUC matrix.

Reads results/panel_model_matrix/pmm_p<panel>_m<model>/<model>/aggregated/all_test_metrics.csv
and writes:
  - operations/panel_model_matrix/analysis/panel_model_prauc.csv     (long form)
  - operations/panel_model_matrix/analysis/panel_model_prauc_wide.csv (matrix form)
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "results" / "panel_model_matrix"
OUT = PROJECT_ROOT / "operations" / "panel_model_matrix" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = ["LinSVM_cal", "LR_EN", "RF", "XGBoost"]
MODELS = ["LinSVM_cal", "LR_EN", "RF", "XGBoost"]


def _cell_metrics(panel: str, model: str) -> dict[str, float] | None:
    run_id = f"pmm_p{panel}_m{model}"
    metrics_csv = RESULTS / run_id / f"run_{run_id}" / model / "aggregated" / "all_test_metrics.csv"
    if not metrics_csv.exists():
        return None
    rows = list(csv.DictReader(metrics_csv.open()))

    def agg(key: str) -> tuple[float, float] | None:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        if not vals:
            return None
        return (statistics.mean(vals), statistics.pstdev(vals))

    out: dict[str, float] = {"n_splits": float(len(rows))}
    for k in ("prauc", "auroc", "brier_score", "calibration_slope"):
        m = agg(k)
        if m is not None:
            out[f"{k}_mean"], out[f"{k}_std"] = m
    return out


def main() -> None:
    long_rows: list[dict[str, object]] = []
    wide: dict[str, dict[str, str]] = {panel: {} for panel in PANELS}
    for panel in PANELS:
        for model in MODELS:
            m = _cell_metrics(panel, model)
            if m is None:
                long_rows.append({"panel": panel, "model": model, "status": "missing"})
                wide[panel][model] = "NA"
                continue
            long_rows.append({"panel": panel, "model": model, "status": "ok", **m})
            wide[panel][model] = f"{m['prauc_mean']:.3f}±{m['prauc_std']:.3f}"

    long_path = OUT / "panel_model_prauc.csv"
    keys = sorted({k for r in long_rows for k in r.keys()})
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(long_rows)
    print(f"wrote {long_path}")

    wide_path = OUT / "panel_model_prauc_wide.csv"
    with wide_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel \\ model", *MODELS])
        for panel in PANELS:
            w.writerow([panel, *[wide[panel].get(m, "NA") for m in MODELS]])
    print(f"wrote {wide_path}")


if __name__ == "__main__":
    main()
