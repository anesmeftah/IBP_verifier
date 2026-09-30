import numpy as np

from core.symbolic import SymbolicBound
from core.ibp import IBPVerifier


def test_linear_sip_equals_ibp():

    W = np.array([
        [2, -3],
        [-1, 4]
    ], dtype=float)

    b = np.array([
        1,
        2
    ], dtype=float)


    x = np.array([0, 0], dtype=float)

    eps = np.array([1, 2], dtype=float)


    # ----------------
    # IBP
    # ----------------

    verifier = IBPVerifier(
        weights=W,
        bias=b
    )

    ibp_lower, ibp_upper = verifier.forward(
        x,
        eps
    )


    # ----------------
    # SIP
    # ----------------

    symbolic = SymbolicBound.from_input(
        input_dim=2
    )

    symbolic = symbolic.linear(
        W,
        b
    )


    x_lower = x - eps
    x_upper = x + eps


    sip_lower, sip_upper = symbolic.concretize(
        x_lower,
        x_upper
    )


    np.testing.assert_allclose(
        sip_lower,
        ibp_lower
    )

    np.testing.assert_allclose(
        sip_upper,
        ibp_upper
    )


def test_sip_relu_vs_ibp():

    W = np.array([
        [1, -2],
        [2, 1]
    ], dtype=float)


    b = np.array([
        0,
        -1
    ], dtype=float)


    x = np.array([0,0], dtype=float)

    eps = np.array([1,1], dtype=float)


    # ----------------
    # IBP
    # ----------------

    layers = [
        {
            "type":"Linear",
            "weights":W,
            "bias":b
        },
        {
            "type":"ReLU"
        }
    ]


    verifier = IBPVerifier(
        weights=W,
        bias=b
    )

    verifier.layers = layers


    ibp_lower, ibp_upper = verifier.forward(
        x,
        eps
    )


    # ----------------
    # SIP
    # ----------------

    symbolic = SymbolicBound.from_input(2)


    symbolic = symbolic.linear(
        W,
        b
    )


    x_lower = x-eps
    x_upper = x+eps


    lower, upper = symbolic.concretize(
        x_lower,
        x_upper
    )


    symbolic = symbolic.relu(
        lower,
        upper
    )


    sip_lower, sip_upper = symbolic.concretize(
        x_lower,
        x_upper
    )


    # Soundness
    assert np.all(
        sip_lower <= sip_upper
    )


    # SIP should be tighter or equal
    assert np.all(
        sip_lower >= ibp_lower
    )

    assert np.all(
        sip_upper <= ibp_upper
    )


