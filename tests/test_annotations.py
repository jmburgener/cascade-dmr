from cascade_dmr import SampleAnnotations


def test_add_training_labels_balances_weights(synthetic_dataset):
    annotations = SampleAnnotations(synthetic_dataset.annotations, project_id="test")
    case_ids = synthetic_dataset.annotations.loc[
        synthetic_dataset.annotations["phenotype"] == "case", "sample_id"
    ].tolist()
    control_ids = synthetic_dataset.annotations.loc[
        synthetic_dataset.annotations["phenotype"] == "control", "sample_id"
    ].tolist()

    annotations.add_training_labels(
        case_ids=case_ids,
        control_ids=control_ids,
        high_confidence_case_ids=case_ids,
        window_calibration_ids=case_ids[:5] + control_ids[:5],
    )

    df = annotations.annotations
    assert set(df["case_control_label"]) == {0, 1}
    assert df.loc[df["case_control_label"] == 1, "sample_weight"].nunique() == 1
    assert df["for_window_calibration"].sum() == 10


def test_cross_validation_folds_cover_all_labeled_samples(synthetic_dataset):
    annotations = SampleAnnotations(synthetic_dataset.annotations, project_id="test")
    case_ids = synthetic_dataset.annotations.loc[
        synthetic_dataset.annotations["phenotype"] == "case", "sample_id"
    ].tolist()
    control_ids = synthetic_dataset.annotations.loc[
        synthetic_dataset.annotations["phenotype"] == "control", "sample_id"
    ].tolist()
    annotations.add_training_labels(case_ids, control_ids, case_ids, case_ids[:5] + control_ids[:5])
    annotations.add_cross_validation_folds(n_splits=4, n_repeats=2)

    cv = annotations.cv_annotations
    assert set(cv["permutation_label"]) == {f"{r}_{f}" for r in range(2) for f in range(4)}
    for permutation_label, group in cv.groupby("permutation_label"):
        assert set(group["train_or_test"]) == {"train", "test"}
        for _, sub in group.groupby("numeric_group"):
            assert sub["train_or_test"].nunique() == 1
