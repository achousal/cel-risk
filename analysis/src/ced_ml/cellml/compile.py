"""Library wrapper around operations/cellml/compile_factorial.py.

Delegates to the exact same filesystem compilation logic so we preserve
the column semantics used by validate_tree.R. Adds experiment-level
bookkeeping: registry update, output CSV path under experiments/<name>/.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pandas as pd

from ced_ml.cellml.registry import update_status

logger = logging.getLogger(__name__)


def _load_compile_module(repo_root: Path):
    """Dynamically import operations/cellml/compile_factorial.py.

    operations/ is not a Python package — it's a sibling of analysis/ —
    so we load the module directly via importlib.
    """
    compile_py = repo_root / "operations" / "cellml" / "compile_factorial.py"
    if not compile_py.exists():
        raise FileNotFoundError(f"compile_factorial.py not found at {compile_py}")
    spec = importlib.util.spec_from_file_location("cellml_compile_factorial", compile_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_experiment(
    name: str,
    experiment_dir: Path,
    results_root: Path,
    *,
    repo_root: Path,
) -> pd.DataFrame:
    """Walk results/cellml/<name>/ and compile a single CSV.

    Parameters
    ----------
    name : str
        Experiment name (used in registry updates).
    experiment_dir : Path
        experiments/<name>/ — contains recipes/cell_manifest.csv.
    results_root : Path
        Where the per-cell test_metrics_summary.csv files live.
    repo_root : Path
        Repository root; needed to load the compile_factorial module.

    Returns
    -------
    pd.DataFrame
        Compiled table written to experiments/<name>/compiled.csv.
    """
    manifest_csv = experiment_dir / "recipes" / "cell_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"cell manifest missing: {manifest_csv}")

    output_path = experiment_dir / "compiled.csv"
    mod = _load_compile_module(repo_root)
    df = mod.compile_factorial(manifest_csv, results_root, output_path)

    if df is not None and not df.empty:
        best = None
        for col in ("summary_prauc_mean", "summary_auprc_mean", "summary_auroc_mean"):
            if col in df.columns:
                best = float(df[col].max())
                break
        update_status(
            name,
            status="compiled",
            best_prauc=best if best is not None else "",
        )
    else:
        update_status(name, status="compiled_empty")
    return df
