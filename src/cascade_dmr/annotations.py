from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED_COLUMNS = {"sample_id"}


class SampleAnnotations:
    def __init__(
        self,
        annotations: pd.DataFrame,
        project_id: str,
        grouping_id: str = "participant_id",
    ):
        annotations = annotations.copy()
        if annotations.index.name != "sample_id":
            if "sample_id" not in annotations.columns:
                raise ValueError("annotations must have a 'sample_id' column or be indexed by it")
            annotations = annotations.set_index("sample_id")
        self.project_id = project_id
        self.grouping_id = grouping_id
        self.annotations = annotations
        self.cv_annotations: pd.DataFrame | None = None

    def drop_by_id(self, sample_ids: list) -> None:
        present = [s for s in sample_ids if s in self.annotations.index]
        self.annotations = self.annotations.drop(present)

    def update_grouping(self, grouping_column: str) -> None:
        self.grouping_id = grouping_column

    def add_training_labels(
        self,
        case_ids: list,
        control_ids: list,
        high_confidence_case_ids: list,
        window_calibration_ids: list,
    ) -> None:
        df = self.annotations
        df["case_control_label"] = -1
        df.loc[df.index.intersection(case_ids), "case_control_label"] = 1
        df.loc[df.index.intersection(control_ids), "case_control_label"] = 0

        df["numeric_group"] = df.groupby(self.grouping_id).ngroup()
        if df["numeric_group"].eq(-1).any() or df["numeric_group"].isna().any():
            warnings.warn("Samples with missing grouping values found; assigning each its own group.")
            missing = df["numeric_group"].isin([-1]) | df["numeric_group"].isna()
            next_group = df["numeric_group"].max() + 1
            new_groups = range(next_group, next_group + missing.sum())
            df.loc[missing, "numeric_group"] = list(new_groups)

        n = len(df)
        n_pos = max((df["case_control_label"] == 1).sum(), 1)
        n_neg = max((df["case_control_label"] == 0).sum(), 1)
        df["sample_weight"] = 0.0
        df.loc[df["case_control_label"] == 1, "sample_weight"] = n / n_pos / 2
        df.loc[df["case_control_label"] == 0, "sample_weight"] = n / n_neg / 2

        df["high_confidence_label"] = False
        df.loc[df.index.intersection(high_confidence_case_ids), "high_confidence_label"] = True
        df.loc[df.index.intersection(control_ids), "high_confidence_label"] = True

        df["for_window_calibration"] = df.index.isin(window_calibration_ids)

        self.annotations = df

    def add_cross_validation_folds(self, n_splits: int = 5, n_repeats: int = 5) -> None:
        if "case_control_label" not in self.annotations.columns:
            raise RuntimeError("call add_training_labels() before add_cross_validation_folds()")

        labeled = self.annotations.loc[self.annotations["case_control_label"] != -1].copy()
        labeled = labeled.sort_index()

        folds = []
        for repeat in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            splits = cv.split(labeled, y=labeled["case_control_label"], groups=labeled["numeric_group"])
            for fold, (train_idx, test_idx) in enumerate(splits):
                fold_df = labeled.copy()
                fold_df["train_or_test"] = ""
                fold_df.iloc[train_idx, fold_df.columns.get_loc("train_or_test")] = "train"
                fold_df.iloc[test_idx, fold_df.columns.get_loc("train_or_test")] = "test"
                fold_df["permutation_label"] = f"{repeat}_{fold}"
                folds.append(fold_df)

        self.cv_annotations = pd.concat(folds).reset_index()

    def resolve_samples(self, cv: bool = True) -> pd.DataFrame:
        if cv:
            if self.cv_annotations is None:
                raise RuntimeError("call add_cross_validation_folds() first, or pass cv=False")
            return self.cv_annotations
        samples = self.annotations.reset_index()
        samples["permutation_label"] = "0_0"
        samples["train_or_test"] = "train"
        return samples
