import numpy as np
import pandas as pd

from cascade_dmr import DmrCaller


def _caller_with_stats(mean_pos, prevalence):
    """One region with a clean, low background. mean_neg = 0.99 so that
    log2((mean_pos + 0.01) / (mean_neg + 0.01)) = log2(mean_pos + 0.01) exactly."""
    stats = pd.DataFrame(
        {
            "ref": ["chr1"],
            "roi_start": [100],
            "roi_end": [200],
            "permutation_label": [0],
            "mean_pos": [mean_pos],
            "mean_neg": [0.99],
            "var_pos": [mean_pos],
            "var_neg": [0.99],
            "length": [100],
            "mean_neg_norm": [0.0099],
            "prevalence": [prevalence],
            "likelihood_ratio": [10.0],
        }
    )
    caller = DmrCaller.__new__(DmrCaller)
    caller.region_stats = stats
    caller.call_regions(
        n_features=[10],
        background_count_threshold=[1.0],
        background_dispersion_threshold=[3.0],
        antidmr_max_abs_log2_ratio=[0.25],
        dmr_min_abs_log2_ratio=[2.0],
        dmr_min_prevalence=[0.2],
    )
    return caller.calls.loc[caller.calls["feature_label"] == "dmr"]


def test_log2_ratio_exactly_at_threshold_is_not_a_dmr():
    # log2(3.99 + 0.01) = 2.0: the threshold is strict (>), so this region is excluded
    assert np.log2((3.99 + 0.01) / (0.99 + 0.01)) == 2.0
    assert _caller_with_stats(mean_pos=3.99, prevalence=0.5).empty


def test_log2_ratio_above_threshold_is_a_dmr():
    assert len(_caller_with_stats(mean_pos=4.5, prevalence=0.5)) == 1


def test_prevalence_exactly_at_minimum_is_a_dmr():
    # the prevalence threshold is inclusive (>=)
    assert len(_caller_with_stats(mean_pos=4.5, prevalence=0.2)) == 1


def test_prevalence_below_minimum_is_not_a_dmr():
    assert _caller_with_stats(mean_pos=4.5, prevalence=0.19).empty
