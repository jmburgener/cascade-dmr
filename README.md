# cascade-dmr

Python package for calling differentially methylated regions (DMRs) from cell-free DNA fragment data and scoring samples against them. Coding sample, not an actual tool used in production.

Given fragment counts over regions of interest (ROI) for cases and controls, `DynamicWindowSelector` restricts each ROI down to whichever sub-window best separates cases and controls. `DmrCaller` performs a grid-search of threshold values to decide which regions are actually differentially methylated, and which are anti-DMRs/background. `EnsembleScorer` scores each sample as the ratio of DMR counts over anti-DMR counts. `SampleAnnotations` runs everything through repeated grouped CV so results are out-of-fold.

`synthetic.py` fakes a cohort with a known signal so this runs without real data.

## Setup and install

```
git clone https://github.com/jmburgener/cascade-dmr
cd cascade-dmr
python3 -m venv .venv
source .venv/bin/activate
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

16 tests. Region/annotation/window helpers each get their own unit tests using synthetic data. `DmrCaller` and `EnsembleScorer` have unit tests on hand-built inputs that pin the calling thresholds (|log2 ratio| is strict `>`, prevalence is inclusive `>=`) and the scoring edge case: a sample with zero anti-DMR counts gets `NaN` (no call), never `inf`. `test_pipeline.py` runs the whole thing end to end.
