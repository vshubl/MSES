import numpy as np
from mses.quantitative import create_pls_pipeline, calculate_lod


def test_pls_pipeline_creation():
    pipeline = create_pls_pipeline(n_components=2)
    assert pipeline is not None


def test_lod_calculation():
    syx = 0.01
    slope = 1.5
    lod = calculate_lod(syx, slope)
    expected = 3.0 * 0.01 / 1.5
    assert np.isclose(lod, expected)