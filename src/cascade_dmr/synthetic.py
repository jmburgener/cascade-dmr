from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    annotations: pd.DataFrame
    regions: pd.DataFrame
    fragments: pd.DataFrame
    signal_region_ids: list[str]


def make_synthetic_dataset(
    n_cases: int = 40,
    n_controls: int = 40,
    n_signal_regions: int = 6,
    n_background_regions: int = 30,
    region_width: int = 300,
    background_rate: float = 1.5,
    signal_rate: float = 7.0,
    seed: int = 0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)

    sample_ids = [f"case_{i:03d}" for i in range(n_cases)] + [f"control_{i:03d}" for i in range(n_controls)]
    annotations = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "participant_id": sample_ids,
            "phenotype": ["case"] * n_cases + ["control"] * n_controls,
        }
    )

    n_regions = n_signal_regions + n_background_regions
    starts = np.arange(n_regions) * (region_width * 2)
    regions = pd.DataFrame(
        {
            "ref": "chr1",
            "start": starts,
            "end": starts + region_width,
            "n_cpgs": rng.integers(3, 15, n_regions),
        }
    )
    regions["region_id"] = regions["ref"]
    regions["roi_id"] = [f"region_{i:03d}" for i in range(n_regions)]
    signal_region_ids = regions["roi_id"].iloc[:n_signal_regions].tolist()

    fragments = []
    for sample_id in sample_ids:
        is_case = sample_id.startswith("case_")
        for _, region in regions.iterrows():
            width = region["end"] - region["start"]
            n_background = rng.poisson(background_rate)
            midpoints = rng.uniform(region["start"], region["end"], n_background).tolist()
            if is_case and region["roi_id"] in signal_region_ids:
                inner_start = region["start"] + width * 0.35
                inner_end = region["start"] + width * 0.65
                n_signal = rng.poisson(signal_rate)
                midpoints += rng.uniform(inner_start, inner_end, n_signal).tolist()
            for midpoint in midpoints:
                frag_len = rng.integers(60, 141)
                half = frag_len / 2
                fragments.append(
                    {
                        "sample_id": sample_id,
                        "ref": region["ref"],
                        "region_id": region["region_id"],
                        "start": midpoint - half,
                        "end": midpoint + half,
                        "mapq": rng.integers(30, 61),
                        "n_cpgs": rng.integers(2, 6),
                    }
                )

    return SyntheticDataset(
        annotations=annotations,
        regions=regions,
        fragments=pd.DataFrame(fragments),
        signal_region_ids=signal_region_ids,
    )
