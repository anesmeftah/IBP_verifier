import numpy as np
from verifier import IBPVerifier

def test_positive_weights():
    W = np.array([2.0])
    b = np.array([0.0])

    x = np.array([2.0])
    eps = np.array([1.0])

    expected_lower = np.array([2.0])
    expected_upper = np.array([6.0])

    verifier = IBPVerifier(weights=W , bias=b)

    result_lower , result_upper = verifier.forward(x , eps)


    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_negative_weights():
    W = np.array([-2.0])
    b = np.array([0.0])

    x = np.array([2.0])
    eps = np.array([1.0])

    expected_lower = np.array([-6.0])
    expected_upper = np.array([-2.0])

    verifier = IBPVerifier(weights=W , bias=b)

    result_lower , result_upper = verifier.forward(x , eps)


    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_zero_weights_with_bias():
    W = np.array([0.0])
    b = np.array([3.0])

    x = np.array([2.0])
    eps = np.array([1.0])

    expected_lower = np.array([3.0])
    expected_upper = np.array([3.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_positive_weights_with_bias():
    W = np.array([1.0])
    b = np.array([2.0])

    x = np.array([2.0])
    eps = np.array([1.0])

    expected_lower = np.array([3.0])
    expected_upper = np.array([5.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_mixed_weights():
    W = np.array([2.0, -3.0])
    b = np.array([1.0, 0.0])

    x = np.array([2.0, 2.0])
    eps = np.array([1.0, 1.0])

    expected_lower = np.array([1.0, -8.0])
    expected_upper = np.array([7.0, -2.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_identity_matrix():
    W = np.eye(2)
    b = np.zeros(2)

    x = np.array([2.0, -1.0])
    eps = np.array([1.0, 2.0])

    expected_lower = np.array([1.0, -3.0])
    expected_upper = np.array([3.0, 1.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_zero_interval():
    W = np.array([2.0])
    b = np.array([0.0])

    x = np.array([0.0])
    eps = np.array([0.0])

    expected_lower = np.array([0.0])
    expected_upper = np.array([0.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_identity_matrix_with_bias():
    W = np.eye(2)
    b = np.array([1.0, -2.0])

    x = np.array([2.0, 3.0])
    eps = np.array([1.0, 0.5])

    expected_lower = np.array([0.0, -1.5])
    expected_upper = np.array([4.0, 0.5])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_multiple_outputs():
    W = np.array([[2.0, -1.0], [1.0, 3.0]])
    b = np.array([1.0, -2.0])

    x = np.array([2.0, 1.0])
    eps = np.array([1.0, 0.5])

    expected_lower = np.array([1.0, -1.5])
    expected_upper = np.array([5.0, 4.5])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)


def test_multiple_inputs():
    W = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([0.0, 1.0])

    x = np.array([3.0, -2.0])
    eps = np.array([1.0, 1.0])

    expected_lower = np.array([2.0, -2.0])
    expected_upper = np.array([4.0, 4.0])

    verifier = IBPVerifier(weights=W, bias=b)

    result_lower, result_upper = verifier.forward(x, eps)

    np.testing.assert_allclose(result_lower, expected_lower)
    np.testing.assert_allclose(result_upper, expected_upper)