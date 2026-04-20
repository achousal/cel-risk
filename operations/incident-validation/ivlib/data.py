"""Data loading and locked dev/test split for incident validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import (
    CONTROL_LABEL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    PROTEIN_SUFFIX,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_and_split(cfg: Any) -> dict:
    """Load data, define groups, create locked test/dev split."""
    logger.info("Loading data from %s", cfg.data_path)
    df = read_proteomics_file(str(cfg.data_path))
    logger.info("Loaded %d samples, %d columns", len(df), len(df.columns))

    protein_cols = [c for c in df.columns if c.endswith(PROTEIN_SUFFIX)]
    logger.info("Found %d protein columns", len(protein_cols))

    incident_mask = df[TARGET_COL] == INCIDENT_LABEL
    prevalent_mask = df[TARGET_COL] == PREVALENT_LABEL
    control_mask = df[TARGET_COL] == CONTROL_LABEL

    n_inc = incident_mask.sum()
    n_prev = prevalent_mask.sum()
    n_ctrl = control_mask.sum()
    logger.info("Groups: %d incident, %d prevalent, %d controls", n_inc, n_prev, n_ctrl)

    # Locked test set: 20% of incident + 20% of controls, stratified by sex
    ic_df = df[incident_mask | control_mask].copy()
    ic_df["_binary"] = (ic_df[TARGET_COL] == INCIDENT_LABEL).astype(int)

    sex_col = "sex"
    if sex_col in ic_df.columns:
        ic_df["_strat"] = ic_df["_binary"].astype(str) + "_" + ic_df[sex_col].astype(str)
    else:
        logger.warning("'sex' column not found; stratifying by outcome only")
        ic_df["_strat"] = ic_df["_binary"].astype(str)

    dev_idx, test_idx = train_test_split(
        ic_df.index,
        test_size=cfg.test_frac,
        stratify=ic_df["_strat"],
        random_state=cfg.split_seed,
    )

    test_set = set(test_idx)
    dev_incident_idx = np.array([i for i in df.index[incident_mask] if i not in test_set])
    dev_control_idx = np.array([i for i in df.index[control_mask] if i not in test_set])
    test_incident_idx = np.array([i for i in df.index[incident_mask] if i in test_set])
    test_control_idx = np.array([i for i in df.index[control_mask] if i in test_set])
    prevalent_idx = np.array(df.index[prevalent_mask])

    logger.info(
        "Test set: %d incident + %d controls = %d",
        len(test_incident_idx),
        len(test_control_idx),
        len(test_incident_idx) + len(test_control_idx),
    )
    logger.info(
        "Dev set: %d incident + %d controls + %d prevalent",
        len(dev_incident_idx),
        len(dev_control_idx),
        len(prevalent_idx),
    )

    return {
        "df": df,
        "protein_cols": protein_cols,
        "dev_incident_idx": dev_incident_idx,
        "dev_control_idx": dev_control_idx,
        "test_incident_idx": test_incident_idx,
        "test_control_idx": test_control_idx,
        "prevalent_idx": prevalent_idx,
    }
