import numpy as np
from verifier import IBPVerifier


def test_relu_positive_interval():
    """
    Entire interval is positive.
    ReLU should keep the same bounds.
    """
    x = np.array([2.0, 3.0])
    eps = np.array([1.0, 1.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([1.0, 2.0])
    expected_upper = np.array([3.0, 4.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_negative_interval():
    """
    Entire interval is negative.
    ReLU output should be zero.
    """
    x = np.array([-3.0, -5.0])
    eps = np.array([1.0, 2.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0, 0.0])
    expected_upper = np.array([0.0, 0.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_cross_zero_interval():
    """
    Interval crosses zero.
    Output lower bound becomes zero.
    """
    x = np.array([0.0])
    eps = np.array([2.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0])
    expected_upper = np.array([2.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_exact_zero():
    """
    Input is exactly zero.
    """
    x = np.array([0.0])
    eps = np.array([0.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0])
    expected_upper = np.array([0.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_positive_boundary_zero():
    """
    Interval starts at zero and goes positive.
    """
    x = np.array([1.0])
    eps = np.array([1.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0])
    expected_upper = np.array([2.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_negative_boundary_zero():
    """
    Interval ends at zero.
    """
    x = np.array([-1.0])
    eps = np.array([1.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0])
    expected_upper = np.array([0.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_multiple_neurons():
    """
    Different ReLU states in the same layer:
    inactive, active, uncertain.
    """
    x = np.array([-3.0, 4.0, 0.0])
    eps = np.array([1.0, 1.0, 2.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0, 3.0, 0.0])
    expected_upper = np.array([0.0, 5.0, 2.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_large_values():
    """
    Large values should not create numerical issues.
    """
    x = np.array([1e10])
    eps = np.array([1e5])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([9999900000.0])
    expected_upper = np.array([10000100000.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_small_values_near_zero():
    """
    Very small interval around zero.
    """
    x = np.array([0.0])
    eps = np.array([1e-8])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0])
    expected_upper = np.array([1e-8])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)



def test_relu_random_mixed_interval():
    """
    Mixed interval containing negative, positive,
    and zero-crossing neurons.
    """
    x = np.array([-2.0, 5.0, 1.0, -4.0])
    eps = np.array([3.0, 1.0, 0.5, 10.0])

    verifier = IBPVerifier(Type="ReLU")

    expected_lower = np.array([0.0, 4.0, 0.5, 0.0])
    expected_upper = np.array([1.0, 6.0, 1.5, 6.0])

    result_lower, result_upper = verifier.forward(x=x, eps=eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)