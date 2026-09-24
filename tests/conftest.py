import pytest

from cascade_dmr.synthetic import make_synthetic_dataset


@pytest.fixture
def synthetic_dataset():
    return make_synthetic_dataset(n_cases=20, n_controls=20, n_signal_regions=4, n_background_regions=12, seed=1)
