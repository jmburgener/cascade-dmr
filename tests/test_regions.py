import pandas as pd

from cascade_dmr import RegionTable


def test_add_roi_id_and_buffer():
    regions = pd.DataFrame({"ref": ["chr1", "chr1"], "start": [100, 500], "end": [200, 600], "n_cpgs": [5, 0]})
    table = RegionTable(regions)
    table.add_roi_id()
    table.add_buffer(10)

    assert table.regions["roi_id"].tolist() == ["chr1:100-200", "chr1:500-600"]
    assert table.regions["start"].tolist() == [90, 490]
    assert table.regions["end"].tolist() == [210, 610]
    assert table.regions["zero_cpg_roi"].tolist() == [False, True]


def test_tile_large_regions_splits_and_preserves_original_bounds():
    regions = pd.DataFrame({"ref": ["chr1"], "start": [0], "end": [1200], "n_cpgs": [10]})
    tiled = RegionTable(regions).tile_large_regions(max_roi_size=500, step=250)

    assert (tiled["end"] - tiled["start"] <= 500).all()
    assert (tiled["old_start"] == 0).all()
    assert (tiled["old_end"] == 1200).all()
    assert tiled["end"].max() == 1200


def test_summarize_counts_regions():
    regions = pd.DataFrame({"ref": ["chr1", "chr1"], "start": [0, 100], "end": [50, 150], "n_cpgs": [3, 0]})
    summary = RegionTable(regions).summarize()
    assert summary == {"n_regions": 2, "n_cpg_regions": 1, "n_zero_cpg_regions": 1}
