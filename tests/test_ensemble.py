from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cascade_dmr import EnsembleScorer


def _scorer(n_fragments):
    """Two samples (a, b), two DMRs and one anti-DMR, one permutation, one hyper-parameter set.

    n_fragments is keyed by (sample_id, roi_start).
    """
    rois = [(100, 200), (300, 400), (500, 600)]
    counts = pd.DataFrame(
        [
            {"permutation_label": 0, "sample_id": s, "ref": "chr1", "roi_start": a, "roi_end": b,
             "n_fragments": n_fragments.get((s, a), 0)}
            for s in ["a", "b"]
            for a, b in rois
        ]
    )
    calls = pd.DataFrame(
        {
            "permutation_label": [0, 0, 0],
            "ref": ["chr1"] * 3,
            "roi_start": [100, 300, 500],
            "roi_end": [200, 400, 600],
            "feature_label": ["dmr", "dmr", "antidmr"],
            "hyper_param_set_key": [0, 0, 0],
        }
    )
    annotations = SimpleNamespace(resolve_samples=lambda cv: pd.DataFrame({"sample_id": ["a", "b"]}))
    scorer = EnsembleScorer(
        annotations,
        SimpleNamespace(counts=counts),
        SimpleNamespace(calls=calls, grid=pd.DataFrame({"hyper_param_set_key": [0]})),
        cv=False,
    )
    scorer.score()
    return scorer.scores.set_index("sample_id")


def test_score_is_dmr_over_antidmr_counts():
    scores = _scorer({("a", 100): 6, ("a", 300): 4, ("a", 500): 5, ("b", 500): 5})

    assert scores.loc["a", "n_dmr_counts"] == 10
    assert scores.loc["a", "n_antidmr_counts"] == 5
    assert scores.loc["a", "score"] == pytest.approx(2.0)
    assert scores.loc["b", "score"] == pytest.approx(0.0)


def test_normalized_score_divides_by_number_of_regions():
    scores = _scorer({("a", 100): 6, ("a", 300): 4, ("a", 500): 5})

    # (10 counts / 2 DMRs) / (5 counts / 1 anti-DMR)
    assert scores.loc["a", "normalized_score"] == pytest.approx(1.0)


def test_zero_antidmr_counts_gives_no_call_not_infinity():
    scores = _scorer({("a", 100): 10})

    assert np.isnan(scores.loc["a", "score"])
    assert np.isnan(scores.loc["a", "normalized_score"])
    assert not np.isinf(scores[["score", "normalized_score"]].to_numpy()).any()
