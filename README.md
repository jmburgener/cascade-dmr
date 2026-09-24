# cascade-dmr

Python package for calling differentially methylated regions (DMRs) from cell-free DNA fragment data and scoring samples against them. Coding sample, not an actual tool used in production.

Given fragment counts over regions of interest (ROI) for cases and controls, `DynamicWindowSelector` restricts each ROI down to whichever sub-window best separates cases and controls. `DmrCaller` performs a grid-search of threshold values to decide which regions are actually differentially methylated, and which are anti-DMRs/background. `EnsembleScorer` scores each sample as the ratio of DMR counts over vs. anti-DMR counts. `SampleAnnotations` runs everything through repeated grouped CV so results are out-of-fold.

`synthetic.py` fakes a cohort with a known signal so this runs without real data.

## Install

```
pip install -e ".[dev]"
```

## Run

```
python examples/end_to_end_demo.py
```

## Test

```
pytest
```

9 tests. Region/annotation/window helpers each get their own unit tests using synthetic data. DmrCaller and EnsembleScorer only get invoked through the end-to-end test in test_pipeline.py.
