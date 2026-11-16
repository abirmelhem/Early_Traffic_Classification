"""Explicit research predictor policy. Never select numeric columns implicitly."""
import numpy as np
import pandas as pd

PREDICTORS = ("mean_length", "std_length", "min_length", "max_length",
              "forward_ratio", "mean_interarrival", "std_interarrival")
CLASSES = ("CHAT", "MAIL", "VOIP", "FT")
BUDGETS = (3, 5, 10, 20)
DESIGN = "outer4_inner_holdout"


def predictor_frame(frame, columns=PREDICTORS):
    if tuple(columns) != PREDICTORS:
        raise ValueError("Predictors must exactly match the approved whitelist and order")
    missing = set(PREDICTORS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing approved predictors: {sorted(missing)}")
    result = frame.loc[:, list(PREDICTORS)].replace("", np.nan).apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Infinite predictor value")
    return result
