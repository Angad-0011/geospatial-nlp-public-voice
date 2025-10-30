"""Toy causal inference example using DoWhy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ENABLE_CAUSAL
from utils_io import log

try:  # pragma: no cover
    import dowhy
except ImportError:
    dowhy = None


def run_toy_causal_example(seed: int = 42) -> dict:
    if not ENABLE_CAUSAL:
        log("ENABLE_CAUSAL=0 -> skipping causal demo")
        return {}
    if dowhy is None:
        log("dowhy not installed. Install via `pip install dowhy`. Skipping causal demo.")
        return {}

    rng = np.random.default_rng(seed)
    n = 200
    pollution = rng.normal(0, 1, size=n)
    civic_budget = 0.5 * pollution + rng.normal(0, 1, size=n)
    sentiment = -0.3 * pollution + 0.8 * civic_budget + rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        "pollution": pollution,
        "civic_budget": civic_budget,
        "sentiment": sentiment,
    })

    model = dowhy.CausalModel(
        data=df,
        treatment="civic_budget",
        outcome="sentiment",
        graph="""
        digraph {
            pollution -> civic_budget;
            pollution -> sentiment;
            civic_budget -> sentiment;
        }
        """,
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    result = {
        "estimand": str(identified_estimand),
        "estimate": float(estimate.value),
    }
    log(f"Estimated causal effect: {result['estimate']:.3f}")
    return result


if __name__ == "__main__":
    run_toy_causal_example()
