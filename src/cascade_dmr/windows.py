from __future__ import annotations

import numpy as np
import pandas as pd

from cascade_dmr.fragments import validate_fragments

FRAGMENT_COUNT_PSEUDOCOUNT = 0.01
LIKELIHOOD_RATIO_PSEUDOCOUNT = 0.02


def _window_capture_matrix(midpoints, roi_start: float, window_min: int, window_max: int) -> np.ndarray:
    size = (window_max - window_min) * 2 + 1
    mat = np.zeros((size, size))
    for midpoint in midpoints:
        idx = int((midpoint - roi_start - window_min) * 2)
        if 0 <= idx < size - 1:
            mat[0 : idx + 1, idx + 1 : size] += 1
    return mat


def _prepare_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["midpoint"] = (group["start"] + group["end"] - 1) / 2
    if group["zero_cpg_roi"].iloc[0]:
        return group.loc[group["n_cpgs"] == 0]
    return group.loc[group["n_cpgs"] >= group["min_n_cpgs"].iloc[0]]


def _best_single_window(group: pd.DataFrame) -> pd.DataFrame:
    region_id, roi_id, permutation_label = group.name
    meta = group.iloc[0]
    filtered = _prepare_group(group)
    window_size = int(meta["roi_end"] - meta["roi_start"])
    n_cases, n_controls = meta["n_cases"], meta["n_controls"]

    case_mat = _window_capture_matrix(
        filtered.loc[filtered["case_control_label"] == 1, "midpoint"], meta["roi_start"], 0, window_size
    )
    control_mat = _window_capture_matrix(
        filtered.loc[filtered["case_control_label"] == 0, "midpoint"], meta["roi_start"], 0, window_size
    )
    ratio_mat = ((case_mat + FRAGMENT_COUNT_PSEUDOCOUNT) / n_cases) / (
        (control_mat + FRAGMENT_COUNT_PSEUDOCOUNT) / n_controls
    )
    for i in range(ratio_mat.shape[0]):
        ratio_mat[i, : i + 1] = 0

    best = np.unravel_index(np.argmax(ratio_mat), ratio_mat.shape)
    best_start = meta["roi_start"] + best[0] / 2
    best_end = meta["roi_start"] + best[1] / 2
    in_window = filtered.loc[(filtered["midpoint"] >= best_start) & (filtered["midpoint"] < best_end)]

    refined_case_samples = in_window.loc[in_window["case_control_label"] == 1, "sample_id"].nunique()
    refined_control_samples = in_window.loc[in_window["case_control_label"] == 0, "sample_id"].nunique()
    refined_case_frags = int((in_window["case_control_label"] == 1).sum())
    refined_control_frags = int((in_window["case_control_label"] == 0).sum())

    return pd.DataFrame(
        [
            {
                "permutation_label": permutation_label,
                "region_id": region_id,
                "roi_id": roi_id,
                "ref": meta["ref"],
                "roi_start": meta["roi_start"],
                "roi_end": meta["roi_end"],
                "refined_start": best_start,
                "refined_end": best_end,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "likelihood_ratio": (
                    (refined_case_samples + LIKELIHOOD_RATIO_PSEUDOCOUNT) / (n_cases + LIKELIHOOD_RATIO_PSEUDOCOUNT)
                )
                / ((refined_control_samples + LIKELIHOOD_RATIO_PSEUDOCOUNT) / (n_controls + LIKELIHOOD_RATIO_PSEUDOCOUNT)),
                "enrichment": (
                    (refined_case_frags + FRAGMENT_COUNT_PSEUDOCOUNT) / n_cases
                )
                / ((refined_control_frags + FRAGMENT_COUNT_PSEUDOCOUNT) / n_controls),
                "zero_cpg_roi": meta["zero_cpg_roi"],
            }
        ]
    )


def select_dynamic_windows(
    fragments: pd.DataFrame,
    regions: pd.DataFrame,
    samples: pd.DataFrame,
    min_likelihood_ratio: float = 5.0,
    min_enrichment: float = 10.0,
    min_n_cpgs: int = 2,
) -> pd.DataFrame:
    fragments = validate_fragments(fragments)
    calibration = samples.loc[samples["for_window_calibration"] & (samples["train_or_test"] == "train")]

    counts = (
        calibration.groupby("permutation_label")["case_control_label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={1: "n_cases", 0: "n_controls"})
        .reset_index()
    )
    calibration = calibration.merge(counts, on="permutation_label")

    regions = regions.rename(columns={"start": "roi_start", "end": "roi_end"}).copy()
    if "region_id" not in regions.columns:
        regions["region_id"] = regions["ref"]
    if "zero_cpg_roi" not in regions.columns:
        regions["zero_cpg_roi"] = regions["n_cpgs"] == 0

    frags_with_mid = fragments.merge(calibration, on="sample_id")
    frags_with_mid["midpoint"] = (frags_with_mid["start"] + frags_with_mid["end"] - 1) / 2
    merged = frags_with_mid.merge(
        regions[["region_id", "roi_id", "ref", "roi_start", "roi_end", "zero_cpg_roi", "n_cpgs"]],
        on="region_id",
        suffixes=("", "_roi"),
    )
    in_range = (merged["midpoint"] >= merged["roi_start"]) & (merged["midpoint"] < merged["roi_end"])
    merged = merged.loc[in_range].rename(columns={"n_cpgs_roi": "roi_n_cpgs"})
    merged["min_n_cpgs"] = min_n_cpgs

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "permutation_label",
                "region_id",
                "roi_id",
                "ref",
                "roi_start",
                "roi_end",
                "refined_start",
                "refined_end",
                "n_cases",
                "n_controls",
                "likelihood_ratio",
                "enrichment",
                "zero_cpg_roi",
            ]
        )

    results = merged.groupby(["region_id", "roi_id", "permutation_label"], group_keys=False).apply(_best_single_window)
    results = results.reset_index(drop=True)

    results["start"] = np.where(
        (results["likelihood_ratio"] > min_likelihood_ratio) & (results["enrichment"] > min_enrichment),
        results["refined_start"],
        results["roi_start"],
    )
    results["end"] = np.where(
        (results["likelihood_ratio"] > min_likelihood_ratio) & (results["enrichment"] > min_enrichment),
        results["refined_end"],
        results["roi_end"],
    )
    return results


class DynamicWindowSelector:
    def __init__(self, fragments: pd.DataFrame, regions: pd.DataFrame, annotations, use_cv: bool = True):
        self.fragments = validate_fragments(fragments)
        self.regions = regions
        self.annotations = annotations
        self.use_cv = use_cv
        self.windows: pd.DataFrame | None = None

    def select(self, min_likelihood_ratio: float = 5.0, min_enrichment: float = 10.0, min_n_cpgs: int = 2) -> None:
        self.windows = select_dynamic_windows(
            self.fragments,
            self.regions,
            self.annotations.resolve_samples(self.use_cv),
            min_likelihood_ratio=min_likelihood_ratio,
            min_enrichment=min_enrichment,
            min_n_cpgs=min_n_cpgs,
        )
