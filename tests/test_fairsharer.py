import pytest
from Vermoegen import fair_sharer
import numpy as np

def test_fair_sharer():
    result = fair_sharer([800, 500, 100, 0], 2)
    expected = np.array([512, 644, 100, 144])

    assert np.array_equal(result, expected)
