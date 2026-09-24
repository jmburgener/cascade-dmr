from cascade_dmr import DmrCaller, DynamicWindowSelector, EnsembleScorer, RegionFeatureCounter
from cascade_dmr.synthetic import make_synthetic_dataset


def _run_pipeline(dataset, annotations):
    selector = DynamicWindowSelector(dataset.fragments, dataset.regions, annotations)
    selector.select(min_likelihood_ratio=1.5, min_enrichment=2.0)

    counter = RegionFeatureCounter(dataset.fragments, selector.windows)
    counter.count()

    caller = DmrCaller(annotations, counter)
    caller.compute_region_stats()
    caller.call_regions(
        n_features=[10],
        background_count_threshold=[1.0],
        background_dispersion_threshold=[3.0],
        antidmr_max_abs_log2_ratio=[0.5],
        dmr_min_abs_log2_ratio=[1.0],
        dmr_min_prevalence=[0.1],
    )

    scorer = EnsembleScorer(annotations, counter, caller)
    scorer.score()
    return caller, scorer


def test_pipeline_calls_true_signal_regions_and_separates_cases_from_controls():
    dataset = make_synthetic_dataset(n_cases=30, n_controls=30, n_signal_regions=5, n_background_regions=15, seed=7)

    from cascade_dmr import SampleAnnotations

    annotations = SampleAnnotations(dataset.annotations, project_id="pipeline_test")
    case_ids = dataset.annotations.loc[dataset.annotations["phenotype"] == "case", "sample_id"].tolist()
    control_ids = dataset.annotations.loc[dataset.annotations["phenotype"] == "control", "sample_id"].tolist()
    annotations.add_training_labels(case_ids, control_ids, case_ids, case_ids[:6] + control_ids[:6])
    annotations.add_cross_validation_folds(n_splits=3, n_repeats=2)

    caller, scorer = _run_pipeline(dataset, annotations)

    dmr_calls = caller.calls.loc[caller.calls["feature_label"] == "dmr"]
    assert len(dmr_calls) > 0

    signal_bounds = dataset.regions.loc[dataset.regions["roi_id"].isin(dataset.signal_region_ids), ["start", "end"]]
    recovered = dmr_calls.apply(
        lambda call: (
            (signal_bounds["start"] <= call["roi_start"]) & (call["roi_end"] <= signal_bounds["end"])
        ).any(),
        axis=1,
    )
    assert recovered.any()

    results = scorer.results
    test_rows = results.loc[results["train_or_test"] == "test"]
    case_scores = test_rows.loc[test_rows["case_control_label"] == 1, "normalized_score"]
    control_scores = test_rows.loc[test_rows["case_control_label"] == 0, "normalized_score"]
    assert case_scores.mean() > control_scores.mean()
