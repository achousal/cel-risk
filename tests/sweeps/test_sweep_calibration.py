"""Unit tests for sweep calibration mechanics.

These tests use deterministic stub objective functions -- no Optuna, no
Minerva, no real dataset. The goal is to lock in Rules 5, 10-15, and the
cache / abort behavior of the orchestrator gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from operations.cellml.sweeps.calibration_schema import (
    CalibrationConfidence,
    CalibrationConfig,
    CalibrationResult,
    ProposedSweepParams,
)
from operations.cellml.sweeps.sweep_calibration import (
    CalibrationError,
    _confidence,
    _n_calib,
    detect_plateau,
    load_cached_calibration,
    run_calibration,
)
from operations.cellml.sweeps.sweep_schema import (
    ParameterDef,
    ParameterType,
    SweepConstraints,
    SweepSpec,
    SweepType,
)


# ---------- helpers ----------


def _toy_spec(**overrides) -> SweepSpec:
    spec = SweepSpec(
        id="test_sweep",
        question="?",
        sweep_type=SweepType.config_only,
        parameter_space={
            "alpha": ParameterDef(type=ParameterType.float_range, low=0.0, high=1.0),
            "beta": ParameterDef(type=ParameterType.float_range, low=0.0, high=1.0),
        },
        constraints=SweepConstraints(
            max_iterations=5,
            max_wall_hours=1.0,
            seeds=[0, 1],
            improvement_threshold=0.005,
        ),
        data_path=None,  # tests bypass fingerprinting
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


def _fast_config(**overrides) -> CalibrationConfig:
    base = CalibrationConfig(
        required=True,
        min_trials=10,
        max_trials=30,
        trials_per_dim=10,
        absolute_min_cap=5,
        absolute_max_cap=100,
        min_patience=2,
        max_patience=10,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------- Rule 5: n_calib sizing ----------


def test_n_calib_respects_min_trials_floor():
    cfg = _fast_config(min_trials=20, trials_per_dim=2)
    assert _n_calib(cfg, space_dim=2) == 20  # 2*2=4 < 20 floor


def test_n_calib_respects_max_trials_ceiling():
    cfg = _fast_config(max_trials=25, trials_per_dim=50)
    assert _n_calib(cfg, space_dim=5) == 25  # 50*5=250 > 25 ceiling


def test_n_calib_uses_heuristic_in_range():
    cfg = _fast_config(min_trials=5, max_trials=100, trials_per_dim=10)
    assert _n_calib(cfg, space_dim=4) == 40


# ---------- Rule 10: plateau detection ----------


def test_detect_plateau_early_knee():
    # best rises fast, stabilizes at trial 3
    trace = [0.70, 0.80, 0.85, 0.86, 0.86, 0.86, 0.86]
    knee = detect_plateau(trace, improvement_threshold=0.005)
    assert knee == 3


def test_detect_plateau_never_stabilizes():
    # improvements continue through the final trial
    trace = [0.70, 0.72, 0.74, 0.77, 0.81, 0.90]
    knee = detect_plateau(trace, improvement_threshold=0.005)
    # Final trial always satisfies |x - x| <= thresh, so it always finds
    # SOMETHING -- but for rising curves the knee lands at the last trial,
    # which the confidence layer treats as low.
    assert knee == len(trace) - 1


def test_detect_plateau_empty_returns_none():
    assert detect_plateau([], improvement_threshold=0.005) is None


# ---------- Rule 14: confidence ----------


def test_confidence_high_when_early():
    assert _confidence(plateau_trial=5, n_calib=30) == CalibrationConfidence.high


def test_confidence_medium_when_late():
    # 25/30 = 0.83 > 0.7 -> medium (but still < n_calib)
    assert _confidence(plateau_trial=25, n_calib=30) == CalibrationConfidence.medium


def test_confidence_low_when_not_found():
    # Rule 14: plateau_trial == n_calib (curve still rising at budget end)
    # or None -> low confidence. 29 < 30 is still medium.
    assert _confidence(plateau_trial=30, n_calib=30) == CalibrationConfidence.low


def test_confidence_low_when_none():
    assert _confidence(plateau_trial=None, n_calib=30) == CalibrationConfidence.low


# ---------- Rule 12-13: cap / patience bounds ----------


def test_run_calibration_emits_proposed_for_clean_plateau(tmp_path: Path):
    spec = _toy_spec()
    cfg = _fast_config(min_trials=15, max_trials=15, trials_per_dim=15)

    # Objective plateaus at trial 4 at 0.90
    def objective(params):  # noqa: ARG001
        step = objective.calls
        objective.calls += 1
        if step < 4:
            return 0.70 + 0.05 * step
        return 0.90

    objective.calls = 0

    def suggester(idx: int):
        return {"alpha": 0.1 * idx, "beta": 0.2}

    result = run_calibration(
        spec=spec,
        config=cfg,
        space_hash="spacehash",
        dataset_fingerprint="fingerprint",
        param_suggester=suggester,
        objective_fn=objective,
        calibration_dir=tmp_path / "calibration",
        subsample_rows_used=500,
    )

    assert result.is_usable()
    assert result.plateau_trial == 4
    assert result.proposed is not None
    assert result.proposed.n_trials_cap >= 5  # absolute_min_cap
    assert result.proposed.confidence == CalibrationConfidence.high
    # artifacts written
    assert (tmp_path / "calibration" / f"{result.calibration_id}.json").exists()
    assert (tmp_path / "calibration" / f"{result.calibration_id}.parquet").exists()


# ---------- Rule 15: abort on min_viable_objective ----------


def test_run_calibration_aborts_when_below_floor(tmp_path: Path):
    spec = _toy_spec()
    cfg = _fast_config(min_viable_objective=0.95)  # unreachable

    def objective(params):  # noqa: ARG001
        return 0.80

    def suggester(idx: int):
        return {"alpha": 0.0, "beta": 0.0}

    result = run_calibration(
        spec=spec,
        config=cfg,
        space_hash="sh",
        dataset_fingerprint="fp",
        param_suggester=suggester,
        objective_fn=objective,
        calibration_dir=tmp_path / "cal",
        subsample_rows_used=100,
    )
    assert result.aborted is True
    assert result.proposed is None
    assert "min_viable_objective" in (result.abort_reason or "")


# ---------- Rule 3: cache lifetime ----------


def test_load_cached_calibration_returns_fresh(tmp_path: Path):
    calibration_dir = tmp_path / "cal"
    calibration_dir.mkdir()
    cid = "sweep__sh__fp"

    result = CalibrationResult(
        calibration_id=cid,
        sweep_id="sweep",
        space_hash="sh",
        dataset_fingerprint="fp",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_calib_requested=10,
        n_calib_executed=10,
        sampler="tpe",
        calib_seed=0,
        subsample_rows_used=100,
        wall_seconds_total=1.0,
        plateau_trial=3,
        plateau_value=0.9,
        noise_sigma=0.001,
        noise_warning=False,
        proposed=ProposedSweepParams(
            n_trials_cap=20,
            patience=5,
            calibration_id=cid,
            confidence=CalibrationConfidence.high,
        ),
        aborted=False,
        abort_reason=None,
    )
    (calibration_dir / f"{cid}.json").write_text(result.model_dump_json())
    loaded = load_cached_calibration(calibration_dir, cid, cache_days=30)
    assert loaded is not None
    assert loaded.is_usable()


def test_load_cached_calibration_returns_none_when_stale(tmp_path: Path):
    calibration_dir = tmp_path / "cal"
    calibration_dir.mkdir()
    cid = "sweep__sh__fp"

    old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(timespec="seconds")
    result_dict = {
        "calibration_id": cid, "sweep_id": "sweep", "space_hash": "sh",
        "dataset_fingerprint": "fp", "created_utc": old_time,
        "n_calib_requested": 10, "n_calib_executed": 10, "sampler": "tpe",
        "calib_seed": 0, "subsample_rows_used": 100, "wall_seconds_total": 1.0,
        "plateau_trial": 3, "plateau_value": 0.9, "noise_sigma": 0.001,
        "noise_warning": False, "proposed": None, "aborted": False,
        "abort_reason": None,
    }
    (calibration_dir / f"{cid}.json").write_text(json.dumps(result_dict))
    assert load_cached_calibration(calibration_dir, cid, cache_days=30) is None


# ---------- fingerprint helper ----------


def test_dataset_fingerprint_stable_and_prevalence_sensitive(tmp_path: Path):
    from ced_ml.data.fingerprint import dataset_fingerprint

    df1 = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
    path1 = tmp_path / "d.parquet"
    df1.to_parquet(path1)

    fp1 = dataset_fingerprint(path1, label_col="y", seeds=[0, 1])
    fp1_again = dataset_fingerprint(path1, label_col="y", seeds=[0, 1])
    assert fp1 == fp1_again  # stable

    # different seed set -> different fingerprint
    fp2 = dataset_fingerprint(path1, label_col="y", seeds=[0, 2])
    assert fp1 != fp2

    # different prevalence -> different fingerprint
    df2 = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 0, 1]})
    path2 = tmp_path / "d2.parquet"
    df2.to_parquet(path2)
    fp3 = dataset_fingerprint(path2, label_col="y", seeds=[0, 1])
    assert fp1 != fp3


def test_warm_start_top_k_ordered_best_first(tmp_path: Path):
    """run_calibration should export top-k trials sorted best-first."""
    spec = _toy_spec()
    cfg = _fast_config(
        min_trials=6, max_trials=6, trials_per_dim=6,
        warm_start_top_k=3,
    )

    # Deliberately non-monotone so sorting is actually exercised.
    scores = [0.80, 0.95, 0.70, 0.90, 0.85, 0.60]

    def objective(params):  # noqa: ARG001
        i = len(objective.log)
        objective.log.append(i)
        return scores[i]

    objective.log = []

    def suggester(idx: int):
        return {"alpha": 0.1 * idx, "beta": 0.2 * idx}

    result = run_calibration(
        spec=spec, config=cfg,
        space_hash="sh", dataset_fingerprint="fp",
        param_suggester=suggester, objective_fn=objective,
        calibration_dir=tmp_path / "cal", subsample_rows_used=100,
    )
    assert result.proposed is not None
    wsp = result.proposed.warm_start_points
    assert len(wsp) == 3
    # Best first: scores 0.95, 0.90, 0.85 -> indices 1, 3, 4
    assert [p.calibration_trial_idx for p in wsp] == [1, 3, 4]
    assert [p.objective for p in wsp] == [0.95, 0.90, 0.85]
    # Params recovered as dicts
    assert all(isinstance(p.params, dict) for p in wsp)
    assert wsp[0].params == {"alpha": 0.1, "beta": 0.2}


def test_warm_start_deduplicates_identical_params(tmp_path: Path):
    """TPE convergence to a single best value must not waste warm-start slots."""
    spec = _toy_spec()
    cfg = _fast_config(
        min_trials=6, max_trials=6, trials_per_dim=6,
        warm_start_top_k=3,
    )

    # All trials return the same high score; suggester emits only 2 distinct
    # param dicts. Expect exactly 2 warm-start points, not 3 (padded).
    def objective(params):  # noqa: ARG001
        return 0.90

    def suggester(idx: int):
        # Two distinct configs, alternating
        return {"alpha": 0.1, "beta": 0.2} if idx % 2 == 0 else {"alpha": 0.5, "beta": 0.5}

    result = run_calibration(
        spec=spec, config=cfg,
        space_hash="sh", dataset_fingerprint="fp",
        param_suggester=suggester, objective_fn=objective,
        calibration_dir=tmp_path / "cal", subsample_rows_used=100,
    )
    assert result.proposed is not None
    wsp = result.proposed.warm_start_points
    assert len(wsp) == 2  # not 3 — deduped
    distinct = {json.dumps(p.params, sort_keys=True) for p in wsp}
    assert len(distinct) == 2


def test_enqueue_warm_start_seeds_parent_study():
    """enqueue_warm_start pushes points onto a parent Optuna study."""
    import optuna

    from operations.cellml.sweeps.calibration_schema import (
        ProposedSweepParams,
        WarmStartPoint,
    )
    from operations.cellml.sweeps.optuna_driver import enqueue_warm_start

    proposed = ProposedSweepParams(
        n_trials_cap=20,
        patience=5,
        calibration_id="fake__id",
        confidence=CalibrationConfidence.high,
        warm_start_points=[
            WarmStartPoint(params={"x": 0.5}, objective=0.9, calibration_trial_idx=3),
            WarmStartPoint(params={"x": 0.7}, objective=0.85, calibration_trial_idx=1),
        ],
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    n = enqueue_warm_start(study, proposed)
    assert n == 2

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return -((x - 0.5) ** 2)

    study.optimize(objective, n_trials=3)
    # First 2 trials should be the enqueued ones, carrying user_attrs
    first_two = sorted(study.trials, key=lambda t: t.number)[:2]
    sources = {t.user_attrs.get("source") for t in first_two}
    assert "calibration_warmstart" in sources
    # Enqueued x values are honored
    xs = {round(t.params["x"], 6) for t in first_two}
    assert 0.5 in xs and 0.7 in xs


def test_calibration_wall_hours_rolled_into_total(tmp_path: Path):
    """Flag 2 / Rule 20: ensure_calibration() updates _total_wall_hours."""
    from unittest.mock import patch

    from operations.cellml.sweeps.sweep_orchestrator import SweepOrchestrator

    spec = _toy_spec()
    spec.data_path = None  # we'll short-circuit the data load

    # required=false path writes a skipped result with 0 hours -- that's
    # not what we want. Use required=true + monkeypatched run path.
    spec.calibration = _fast_config(min_trials=4, max_trials=4, trials_per_dim=4)

    orch = SweepOrchestrator(
        spec=spec,
        project_root=tmp_path,
        ledger_dir=tmp_path / "ledger",
    )

    # Build a fake result the orchestrator can accept
    fake_result = CalibrationResult(
        calibration_id="fake",
        sweep_id=spec.id,
        space_hash="sh",
        dataset_fingerprint="fp",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_calib_requested=4, n_calib_executed=4,
        sampler="tpe", calib_seed=0, subsample_rows_used=100,
        wall_seconds_total=3600.0,  # exactly 1 hour
        plateau_trial=2, plateau_value=0.9,
        noise_sigma=0.0, noise_warning=False,
        proposed=ProposedSweepParams(
            n_trials_cap=10, patience=5,
            calibration_id="fake", confidence=CalibrationConfidence.high,
        ),
        aborted=False, abort_reason=None,
    )

    with patch("operations.cellml.sweeps.sweep_orchestrator.dataset_fingerprint", return_value="fp"), \
         patch("operations.cellml.sweeps.sweep_orchestrator.space_hash", return_value="sh"), \
         patch("operations.cellml.sweeps.sweep_orchestrator.load_cached_calibration", return_value=None), \
         patch("operations.cellml.sweeps.sweep_orchestrator.run_calibration", return_value=fake_result), \
         patch("operations.cellml.sweeps.sweep_orchestrator.build_surrogate_objective", return_value=(lambda p: 0.9, 100)):
        # Spec needs a data_path for the gate -- set it to any string
        spec.data_path = "fake.parquet"
        assert orch._total_wall_hours == 0.0
        orch.ensure_calibration()
        assert orch._total_wall_hours == pytest.approx(1.0)


def test_optuna_driver_suggest_tell_roundtrip():
    """OptunaDriver produces valid params and accepts tell()."""
    from operations.cellml.sweeps.optuna_driver import OptunaDriver

    spec = _toy_spec()
    driver = OptunaDriver.from_spec(spec, sampler="random", seed=42)
    params = driver.suggest(0)
    assert set(params.keys()) == {"alpha", "beta"}
    assert 0.0 <= params["alpha"] <= 1.0
    assert 0.0 <= params["beta"] <= 1.0
    driver.tell(0, params, 0.85)
    # Second trial still works
    params2 = driver.suggest(1)
    assert set(params2.keys()) == {"alpha", "beta"}


def test_surrogate_objective_end_to_end(tmp_path: Path):
    """Surrogate + OptunaDriver + run_calibration on a tiny synthetic dataset.

    Verifies the full default wiring works in-process. Does NOT assert
    on absolute AUROC values -- only that calibration produces a usable
    result and writes artifacts.
    """
    import numpy as np
    import pandas as pd

    from operations.cellml.sweeps.optuna_driver import OptunaDriver
    from operations.cellml.sweeps.surrogate_objective import (
        build_surrogate_objective,
    )

    # Build a 400-row toy dataset with 10 signal features + 5 noise
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=(n, 10))
    noise = rng.normal(size=(n, 5))
    y = (signal[:, 0] + signal[:, 1] - signal[:, 2] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame(
        np.hstack([signal, noise]),
        columns=[f"p{i}_resid" for i in range(15)],
    )
    df["CeD_comparison"] = y
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path)

    # Spec that points at the toy dataset; parameter space is the LR C
    # regularization strength so the surrogate has something to tune.
    spec = SweepSpec(
        id="surrogate_test",
        question="?",
        sweep_type=SweepType.config_only,
        parameter_space={
            "C": ParameterDef(type=ParameterType.float_range, low=0.01, high=10.0),
        },
        constraints=SweepConstraints(
            max_iterations=3, max_wall_hours=1.0, seeds=[0],
            improvement_threshold=0.005,
        ),
        data_path=str(data_path.relative_to(tmp_path)),
        label_col="CeD_comparison",
    )
    config = _fast_config(
        min_trials=6, max_trials=10, trials_per_dim=10,
        subsample_rows=300,
        absolute_min_cap=3, absolute_max_cap=50,
    )

    driver = OptunaDriver.from_spec(spec, sampler="random", seed=0)
    objective, rows_used = build_surrogate_objective(
        spec, config, project_root=tmp_path,
    )
    assert rows_used == 300

    result = run_calibration(
        spec=spec,
        config=config,
        space_hash="sh",
        dataset_fingerprint="fp",
        param_suggester=driver.suggest,
        objective_fn=objective,
        calibration_dir=tmp_path / "calibration",
        subsample_rows_used=rows_used,
        on_result=driver.tell,
    )

    assert result.n_calib_executed > 0
    assert not result.aborted
    assert result.proposed is not None
    assert 0.0 <= result.plateau_value <= 1.0  # AUROC bounds
    artifact = tmp_path / "calibration" / f"{result.calibration_id}.parquet"
    assert artifact.exists()


def test_dataset_fingerprint_raises_on_missing_label(tmp_path: Path):
    from ced_ml.data.fingerprint import dataset_fingerprint

    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "d.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="not found"):
        dataset_fingerprint(path, label_col="missing")
