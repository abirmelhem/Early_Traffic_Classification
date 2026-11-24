"""Training-pool balancing only; callers must validate partition separation first."""
import numpy as np
import pandas as pd

from .features import CLASSES


def validate_pool(frame):
    if frame.empty or set(frame.candidate_class) != set(CLASSES):
        raise ValueError("Fitting pool requires exactly the four research classes")
    if frame.flow_id.isna().any() or frame.flow_id.duplicated().any():
        raise ValueError("Original fitting observations require unique flow IDs")
    if frame.flow_id.astype(str).str.strip().eq("").any():
        raise ValueError("Empty fitting flow ID")


def oversample(frame, seed=42):
    """Retain originals, append replacement draws, balance to the largest class.

    Sort original identities before sampling so input row order is irrelevant.
    Use a seed shared across budgets to retain identical multiplicities across N.
    No estimator is constructed and the caller's frame is never mutated.
    """
    validate_pool(frame)
    original = frame.reset_index(drop=True).sort_values("flow_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    target = int(original.candidate_class.value_counts().max())
    parts = [original.copy()]
    for cls in CLASSES:
        group = original[original.candidate_class == cls]
        extra = target - len(group)
        if extra:
            parts.append(group.iloc[rng.choice(len(group), size=extra, replace=True)].copy())
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def balanced_sample_weights(frame):
    """Multiclass weights w_i = fitting_rows / (4 * fitting_class_count)."""
    validate_pool(frame)
    counts = frame.candidate_class.value_counts()
    return frame.candidate_class.map(len(frame) / (len(CLASSES) * counts)).to_numpy(dtype=float)
