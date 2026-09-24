from __future__ import annotations

import pandas as pd


class EnsembleScorer:
    def __init__(self, annotations, feature_counter, dmr_caller, cv: bool = True):
        self.counts = feature_counter.counts
        self.cv = cv
        self.annotations = annotations
        self.calls = dmr_caller.calls
        self.grid = dmr_caller.grid
        self.scores: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None

    def _region_sums(self, label: str, count_col: str, n_regions_col: str) -> pd.DataFrame:
        calls = self.calls.loc[self.calls["feature_label"] == label]
        n_regions = calls.groupby(["hyper_param_set_key", "permutation_label"]).size().rename(n_regions_col)

        joined = self.counts.merge(calls, on=["permutation_label", "ref", "roi_start", "roi_end"])
        sums = (
            joined.groupby(["permutation_label", "hyper_param_set_key", "sample_id"])["n_fragments"]
            .sum()
            .rename(count_col)
            .reset_index()
        )
        return sums.merge(n_regions, on=["hyper_param_set_key", "permutation_label"])

    def score(self) -> None:
        all_combos = (
            self.counts[["permutation_label"]]
            .drop_duplicates()
            .merge(self.counts[["sample_id"]].drop_duplicates(), how="cross")
            .merge(self.calls[["hyper_param_set_key"]].drop_duplicates(), how="cross")
        )

        dmr_sums = self._region_sums("dmr", "n_dmr_counts", "n_dmrs")
        antidmr_sums = self._region_sums("antidmr", "n_antidmr_counts", "n_antidmrs")

        merge_keys = ["permutation_label", "hyper_param_set_key", "sample_id"]
        scores = all_combos.merge(dmr_sums, on=merge_keys, how="left").merge(
            antidmr_sums, on=merge_keys, how="left"
        )
        for col in ["n_dmr_counts", "n_antidmr_counts", "n_dmrs", "n_antidmrs"]:
            scores[col] = scores[col].fillna(0)

        scores["score"] = scores["n_dmr_counts"] / scores["n_antidmr_counts"]
        scores["normalized_score"] = (scores["n_dmr_counts"] / scores["n_dmrs"]) / (
            scores["n_antidmr_counts"] / scores["n_antidmrs"]
        )
        self.scores = scores

        samples = self.annotations.resolve_samples(self.cv)
        join_cols = ["sample_id", "permutation_label"] if self.cv else ["sample_id"]
        self.results = scores.merge(samples, on=join_cols).merge(self.grid, on="hyper_param_set_key")
