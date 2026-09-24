from __future__ import annotations

from cascade_dmr import (
    DmrCaller,
    DynamicWindowSelector,
    EnsembleScorer,
    RegionFeatureCounter,
    SampleAnnotations,
)
from cascade_dmr.synthetic import make_synthetic_dataset


def main() -> None:
    dataset = make_synthetic_dataset(n_cases=60, n_controls=60, n_signal_regions=8, n_background_regions=25, seed=42)
    print(f"Synthetic cohort: {len(dataset.annotations)} samples, {len(dataset.regions)} candidate regions")
    print(f"  ({len(dataset.signal_region_ids)} regions carry an injected case/control signal)")

    annotations = SampleAnnotations(dataset.annotations, project_id="demo")
    case_ids = dataset.annotations.loc[dataset.annotations["phenotype"] == "case", "sample_id"].tolist()
    control_ids = dataset.annotations.loc[dataset.annotations["phenotype"] == "control", "sample_id"].tolist()
    calibration_ids = case_ids[:10] + control_ids[:10]
    annotations.add_training_labels(case_ids, control_ids, case_ids, calibration_ids)
    annotations.add_cross_validation_folds(n_splits=5, n_repeats=3)
    print(f"Cross-validation: {annotations.cv_annotations['permutation_label'].nunique()} repeat/fold permutations")

    print("\nSelecting dynamic windows...")
    selector = DynamicWindowSelector(dataset.fragments, dataset.regions, annotations)
    selector.select(min_likelihood_ratio=2.0, min_enrichment=3.0)
    print(f"  {len(selector.windows)} region x permutation windows selected")

    print("Counting fragments in windows...")
    counter = RegionFeatureCounter(dataset.fragments, selector.windows)
    counter.count()

    print("Calling DMRs / anti-DMRs across a threshold grid...")
    caller = DmrCaller(annotations, counter)
    caller.compute_region_stats()
    caller.call_regions(
        n_features=[15, 30],
        background_count_threshold=[0.5, 1.0],
        background_dispersion_threshold=[2.0],
        antidmr_max_abs_log2_ratio=[0.5],
        dmr_min_abs_log2_ratio=[1.0],
        dmr_min_prevalence=[0.15],
    )
    print(f"  {len(caller.grid)} hyperparameter combinations evaluated")
    print(f"  {(caller.calls['feature_label'] == 'dmr').sum()} DMR calls, "
          f"{(caller.calls['feature_label'] == 'antidmr').sum()} anti-DMR calls (across all permutations/combinations)")

    print("\nScoring samples...")
    scorer = EnsembleScorer(annotations, counter, caller)
    scorer.score()

    best_key = caller.grid.loc[caller.grid["n_features"].idxmax(), "hyper_param_set_key"]
    results = scorer.results
    test_rows = results.loc[(results["train_or_test"] == "test") & (results["hyper_param_set_key"] == best_key)]
    case_scores = test_rows.loc[test_rows["case_control_label"] == 1, "normalized_score"]
    control_scores = test_rows.loc[test_rows["case_control_label"] == 0, "normalized_score"]

    print(f"\nHeld-out normalized_score (hyperparameter set {best_key}):")
    print(f"  cases:    mean={case_scores.mean():.2f}  median={case_scores.median():.2f}")
    print(f"  controls: mean={control_scores.mean():.2f}  median={control_scores.median():.2f}")


if __name__ == "__main__":
    main()
