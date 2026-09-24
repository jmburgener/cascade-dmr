from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"ref", "start", "end", "n_cpgs"}


class RegionTable:
    def __init__(self, regions: pd.DataFrame):
        missing = REQUIRED_COLUMNS.difference(regions.columns)
        if missing:
            raise ValueError(f"regions is missing required columns: {missing}")
        regions = regions.copy()
        if "zero_cpg_roi" not in regions.columns:
            regions["zero_cpg_roi"] = regions["n_cpgs"] == 0
        if "region_id" not in regions.columns:
            regions["region_id"] = regions["ref"]
        self.regions = regions

    def add_buffer(self, buffer_size: int) -> None:
        self.regions["start"] -= buffer_size
        self.regions["end"] += buffer_size

    def add_roi_id(self, column: str = "roi_id") -> None:
        self.regions[column] = (
            self.regions["ref"].astype(str)
            + ":"
            + self.regions["start"].round().astype(int).astype(str)
            + "-"
            + self.regions["end"].round().astype(int).astype(str)
        )

    def tile_large_regions(self, max_roi_size: int = 1000, step: int = 500) -> pd.DataFrame:
        df = self.regions.copy()
        length = df["end"] - df["start"]
        num_splits = np.where(length <= max_roi_size, 1, np.floor(length / step).astype(int))

        rows = []
        for (_, row), n in zip(df.iterrows(), num_splits):
            for split_index in range(int(n)):
                new_start = row["start"] + split_index * step
                new_end = min(new_start + max_roi_size, row["end"])
                new_row = row.copy()
                new_row["old_start"] = row["start"]
                new_row["old_end"] = row["end"]
                new_row["start"] = new_start
                new_row["end"] = new_end
                rows.append(new_row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def summarize(self) -> dict:
        return {
            "n_regions": len(self.regions),
            "n_cpg_regions": int((self.regions["n_cpgs"] > 0).sum()),
            "n_zero_cpg_regions": int((self.regions["n_cpgs"] == 0).sum()),
        }
