"""Retired historical runner; use imbalance_experiments for versioned executions."""
from .balancing import oversample


def run():
    raise RuntimeError("Historical runner retired. Use python -m src.ml.imbalance_experiments --model MODEL --method random_oversampling")


if __name__ == "__main__":
    run()
