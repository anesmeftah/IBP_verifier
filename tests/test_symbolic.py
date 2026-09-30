import numpy as np

from core.symbolic import SymbolicBound


def test_input_symbolic():

    symbolic = SymbolicBound.from_input(3)

    expected_A = np.eye(3)
    expected_b = np.zeros(3)

    assert np.allclose(symbolic.A, expected_A)
    assert np.allclose(symbolic.b, expected_b)