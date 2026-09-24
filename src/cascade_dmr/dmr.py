from __future__ import annotations

import numpy as np
import pandas as pd

from cascade_dmr.features import region_stats

DMR_PARAMS = ["background_count_threshold", "background_dispersion_threshold", "dmr_min_abs_log2_ratio", "dmr_min_prevalence"]
ANTIDMR_PARAMS = ["background_count_threshold", "background_dispersion_threshold", "antidmr_max_abs_log2_ratio"]


def _cross_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.assign(_key=1).merge(right.assign(_key=1), on="_key").drop(columns="_key")


def parameter_grid(
    n_features: list[int],
    background_count_threshold: list[float],
    background_dispersion_threshold: list[float],
    antidmr_max_abs_log2_ratio: list[float],
    dmr_min_abs_log2_ratio: list[float],
    dmr_min_prevalence: list[float],
) -> pd.DataFrame:
    grid = pd.DataFrame({"n_features": n_features})
    for name, values in [
        ("background_count_threshold", background_count_threshold),
        ("background_dispersion_threshold", background_dispersion_threshold),
        ("antidmr_max_abs_log2_ratio", antidmr_max_abs_log2_ratio),
        ("dmr_min_abs_log2_ratio", dmr_min_abs_log2_ratio),
        ("dmr_min_prevalence", dmr_min_prevalence),
    ]:
        grid = _cross_join(grid, pd.DataFrame({name: values}))
    grid = grid.reset_index(drop=True)
    grid["hyper_param_set_key"] = grid.index
    return grid


class DmrCaller:
    def __init__(self, annotations, feature_counter, cv: bool = True):
        self.counts = feature_counter.counts
        self.cv = cv
        self.annotations = annotations
        self.region_stats: pd.DataFrame | None = None
        self.grid: pd.DataFrame | None = None
        self.calls: pd.DataFrame | None = None

    def compute_region_stats(self) -> None:
        samples = self.annotations.resolve_samples(self.cv)
        eligible = samples.loc[
            (samples["train_or_test"] == "train")
            & (~samples["for_window_calibration"])
            & ((samples["case_control_label"] == 0) | samples["high_confidence_label"])
        ]

        counts_per_perm = (
            eligible.groupby("permutation_label")["case_control_label"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={1: "n_cases", 0: "n_controls"})
            .reset_index()
        )
        eligible = eligible.merge(counts_per_perm, on="permutation_label")

        joined = self.counts.merge(eligible, on=["sample_id", "permutation_label"])
        stats = joined.groupby(["ref", "roi_start", "roi_end", "permutation_label"], group_keys=False).apply(
            region_stats
        )
        stats = stats.reset_index(drop=True)
        stats["length"] = stats["roi_end"] - stats["roi_start"]
        stats["mean_neg_norm"] = stats["mean_neg"] / stats["length"]
        stats["prevalence"] = stats["tp"] / (stats["tp"] + stats["fn"])
        self.region_stats = stats

    def call_regions(
        self,
        n_features: list[int] = (25,),
        background_count_threshold: list[float] = (0.5,),
        background_dispersion_threshold: list[float] = (1.5,),
        antidmr_max_abs_log2_ratio: list[float] = (0.25,),
        dmr_min_abs_log2_ratio: list[float] = (2.0,),
        dmr_min_prevalence: list[float] = (0.2,),
        ranking_metric: str = "likelihood_ratio",
        max_region_length: int = 300,
    ) -> None:
        if self.region_stats is None:
            raise RuntimeError("call compute_region_stats() first")

        self.grid = parameter_grid(
            list(n_features),
            list(background_count_threshold),
            list(background_dispersion_threshold),
            list(antidmr_max_abs_log2_ratio),
            list(dmr_min_abs_log2_ratio),
            list(dmr_min_prevalence),
        )

        stats = self.region_stats
        log2_ratio = np.abs(np.log2((stats["mean_pos"] + 0.01) / (stats["mean_neg"] + 0.01)))
        stats = stats.assign(_log2_ratio=log2_ratio, _dispersion_neg=stats["var_neg"] / stats["mean_neg"].replace(0, np.nan))
        stats["_dispersion_neg"] = stats["_dispersion_neg"].fillna(np.inf)
        stats["_dispersion_pos"] = (stats["var_pos"] / stats["mean_pos"].replace(0, np.nan)).fillna(np.inf)

        antidmr_pool = _cross_join(stats, self.grid[ANTIDMR_PARAMS].drop_duplicates())
        antidmr_pool = antidmr_pool.loc[
            (antidmr_pool["mean_neg_norm"] <= antidmr_pool["background_count_threshold"])
            & (antidmr_pool["_dispersion_neg"] < antidmr_pool["background_dispersion_threshold"])
            & (antidmr_pool["_dispersion_pos"] < antidmr_pool["background_dispersion_threshold"])
            & (antidmr_pool["_log2_ratio"] < antidmr_pool["antidmr_max_abs_log2_ratio"])
        ].assign(feature_label="antidmr")

        dmr_pool = _cross_join(stats, self.grid[DMR_PARAMS].drop_duplicates())
        dmr_pool = dmr_pool.loc[
            (dmr_pool["mean_neg_norm"] <= dmr_pool["background_count_threshold"])
            & (dmr_pool["_dispersion_neg"] < dmr_pool["background_dispersion_threshold"])
            & (dmr_pool["_log2_ratio"] > dmr_pool["dmr_min_abs_log2_ratio"])
            & (dmr_pool["length"] <= max_region_length)
            & (dmr_pool["prevalence"] >= dmr_pool["dmr_min_prevalence"])
        ].assign(feature_label="dmr")

        dmr_expanded = self.grid.merge(dmr_pool, on=DMR_PARAMS)
        dmr_expanded["feature_rank"] = dmr_expanded.groupby(["permutation_label", "hyper_param_set_key"])[
            ranking_metric
        ].rank(method="first", ascending=False)
        dmr_selected = dmr_expanded.loc[dmr_expanded["feature_rank"] <= dmr_expanded["n_features"]]

        antidmr_expanded = self.grid.merge(antidmr_pool, on=ANTIDMR_PARAMS)

        self.calls = pd.concat([dmr_selected, antidmr_expanded], ignore_index=True)
