import pandas as pd

from cascade_dmr.windows import DynamicWindowSelector, _best_single_window, select_dynamic_windows


def _fragment_group(starts, ends, case_control_label, **meta):
    n = len(starts)
    group = pd.DataFrame(
        {
            "sample_id": [f"s{case_control_label}_{i}" for i in range(n)],
            "start": starts,
            "end": ends,
            "case_control_label": case_control_label,
            "n_cpgs": [2] * n,
        }
    )
    for key, value in meta.items():
        group[key] = value
    return group


def test_best_single_window_recovers_a_concentrated_case_signal():
    case_group = _fragment_group([99, 100, 100], [101, 102, 102], case_control_label=1)
    control_group = _fragment_group([0, 20, 40, 60, 80], [2, 22, 42, 62, 82], case_control_label=0)
    group = pd.concat([case_group, control_group], ignore_index=True)
    group["permutation_label"] = "0_0"
    group["region_id"] = "chr1"
    group["roi_id"] = "roi_1"
    group["ref"] = "chr1"
    group["roi_start"] = 0
    group["roi_end"] = 150
    group["zero_cpg_roi"] = False
    group["min_n_cpgs"] = 2
    group["n_cases"] = 3
    group["n_controls"] = 5

    result = (
        group.groupby(["region_id", "roi_id", "permutation_label"], group_keys=False)
        .apply(_best_single_window)
        .iloc[0]
    )

    assert 100 <= result["refined_end"] <= 105
    assert result["refined_end"] - result["refined_start"] < 50
    assert result["enrichment"] > 50


def test_select_dynamic_windows_end_to_end(synthetic_dataset):
    fragments = synthetic_dataset.fragments
    regions = synthetic_dataset.regions

    samples = synthetic_dataset.annotations.copy()
    samples["case_control_label"] = (samples["phenotype"] == "case").astype(int)
    samples["for_window_calibration"] = True
    samples["train_or_test"] = "train"
    samples["permutation_label"] = "0_0"

    windows = select_dynamic_windows(fragments, regions, samples, min_likelihood_ratio=1.5, min_enrichment=2.0)

    assert not windows.empty
    signal_windows = windows.loc[windows["roi_id"].isin(synthetic_dataset.signal_region_ids)]
    background_windows = windows.loc[~windows["roi_id"].isin(synthetic_dataset.signal_region_ids)]
    assert signal_windows["enrichment"].mean() > background_windows["enrichment"].mean()


def test_dynamic_window_selector_stores_result_on_instance(synthetic_dataset):
    samples_annotations = synthetic_dataset.annotations.copy()

    class _StubAnnotations:
        cv_annotations = None

        def __init__(self):
            df = samples_annotations.copy()
            df["case_control_label"] = (df["phenotype"] == "case").astype(int)
            df["for_window_calibration"] = True
            self.annotations = df

        def resolve_samples(self, cv):
            samples = self.annotations.reset_index()
            samples["permutation_label"] = "0_0"
            samples["train_or_test"] = "train"
            return samples

    selector = DynamicWindowSelector(synthetic_dataset.fragments, synthetic_dataset.regions, _StubAnnotations(), use_cv=False)
    selector.select(min_likelihood_ratio=1.0, min_enrichment=1.0)

    assert selector.windows is not None
    assert len(selector.windows) > 0
