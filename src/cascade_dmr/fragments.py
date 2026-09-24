from __future__ import annotations

import pandas as pd

FRAGMENT_COLUMNS = {"sample_id", "ref", "start", "end", "mapq", "n_cpgs", "region_id"}


def validate_fragments(fragments: pd.DataFrame) -> pd.DataFrame:
    fragments = fragments.copy()
    if "region_id" not in fragments.columns:
        fragments["region_id"] = fragments["ref"]
    missing = FRAGMENT_COLUMNS.difference(fragments.columns)
    if missing:
        raise ValueError(f"fragments is missing required columns: {missing}")
    return fragments


def overlap_join(fragments: pd.DataFrame, regions: pd.DataFrame, region_extra_cols: list[str] = ()) -> pd.DataFrame:
    midpoint = (fragments["start"] + fragments["end"] - 1) / 2
    fragments = fragments.assign(_midpoint=midpoint)
    merged = fragments.merge(
        regions[["region_id", "roi_start", "roi_end", *region_extra_cols]],
        on="region_id",
        suffixes=("", "_roi"),
    )
    in_range = (merged["_midpoint"] >= merged["roi_start"]) & (merged["_midpoint"] < merged["roi_end"])
    return merged.loc[in_range].drop(columns="_midpoint")


def count_fragments_per_region(
    fragments: pd.DataFrame,
    regions: pd.DataFrame,
    max_frag_len: int = 150,
    min_frag_len: int = 40,
    min_n_cpg: int = 2,
    min_mapq: int = 20,
) -> pd.DataFrame:
    fragments = validate_fragments(fragments)
    frag_len = fragments["end"] - fragments["start"]
    filtered = fragments.loc[(frag_len <= max_frag_len) & (frag_len >= min_frag_len) & (fragments["mapq"] >= min_mapq)]

    joined = overlap_join(filtered, regions, region_extra_cols=["ref", "zero_cpg_roi", "permutation_label"])
    joined = joined.loc[(joined["n_cpgs"] >= min_n_cpg) | joined["zero_cpg_roi"]]

    counts = (
        joined.groupby(["region_id", "sample_id", "ref", "roi_start", "roi_end", "permutation_label"])
        .size()
        .reset_index(name="n_fragments")
    )
    return counts
