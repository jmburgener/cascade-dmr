from __future__ import annotations

import numpy as np
import pandas as pd

from cascade_dmr.fragments import count_fragments_per_region, validate_fragments

MEAN_RATIO_PSEUDOCOUNT = 0.005


class RegionFeatureCounter:
    def __init__(self, fragments: pd.DataFrame, regions: pd.DataFrame):
        self.fragments = validate_fragments(fragments)
        regions = regions.copy()
        if "start" in regions.columns:
            regions = regions.drop(columns=["roi_start", "roi_end"], errors="ignore")
            regions = regions.rename(columns={"start": "roi_start", "end": "roi_end"})
        self.regions = regions
        self.counts: pd.DataFrame | None = None

    def count(self, max_frag_len: int = 150, min_frag_len: int = 40, min_n_cpg: int = 2, min_mapq: int = 20) -> None:
        self.counts = count_fragments_per_region(
            self.fragments,
            self.regions,
            max_frag_len=max_frag_len,
            min_frag_len=min_frag_len,
            min_n_cpg=min_n_cpg,
            min_mapq=min_mapq,
        )


def region_stats(group: pd.DataFrame) -> pd.DataFrame:
    ref, roi_start, roi_end, permutation_label = group.name
    meta = group.iloc[0]
    n_cases, n_controls = meta["n_cases"], meta["n_controls"]
    tp = group.loc[group["case_control_label"] == 1, "sample_id"].nunique()
    fp = group.loc[group["case_control_label"] == 0, "sample_id"].nunique()
    tn = n_controls - fp
    fn = n_cases - tp

    pos_counts = group.loc[group["case_control_label"] == 1, "n_fragments"].tolist() + [0] * fn
    neg_counts = group.loc[group["case_control_label"] == 0, "n_fragments"].tolist() + [0] * tn
    mean_pos, mean_neg = np.mean(pos_counts), np.mean(neg_counts)

    return pd.DataFrame(
        [
            {
                "permutation_label": permutation_label,
                "region_id": meta["region_id"],
                "ref": ref,
                "roi_start": roi_start,
                "roi_end": roi_end,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "mean_pos": mean_pos,
                "mean_neg": mean_neg,
                "mean_ratio": (mean_pos + MEAN_RATIO_PSEUDOCOUNT) / (mean_neg + MEAN_RATIO_PSEUDOCOUNT),
                "var_pos": np.var(pos_counts),
                "var_neg": np.var(neg_counts),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "likelihood_ratio": ((tp + 1) / (n_cases + 2)) / ((fp + 1) / (n_controls + 2)),
            }
        ]
    )
