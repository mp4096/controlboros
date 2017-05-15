import controlboros
import numpy as np
import pytest


def test_dynamics_single_input():
    """Test dynamics equation with single input."""

    a = np.array([[1.0, 2.0], [0.0, 1.0]])
    b = np.array([[1.0], [3.0]])
    c = np.zeros((1, 2))

    s = controlboros.StateSpace(a, b, c)

    assert np.all(s.dynamics([1.0, 1.0], [1.0]) == np.array([4.0, 4.0]))


def test_dynamics_multiple_inputs():
    """Test dynamics equation with multiple inputs."""

    a = np.array([[1.0, 2.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [3.0, 2.0]])
    c = np.zeros((1, 2))

    s = controlboros.StateSpace(a, b, c)

    assert np.all(s.dynamics([1.0, 1.0], [1.0, 0.0]) == np.array([4.0, 4.0]))


def test_output_siso():
    """Test output equation with single input, single output."""

    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    c = np.array([[1.0, 1.0]])
    d = np.array([[2.0]])

    s = controlboros.StateSpace(a, b, c, d)

    assert s.output([1.0, 1.0], [1.0]) == np.array([4.0])


def test_output_simo():
    """Test output equation with single input, multiple outputs."""

    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    c = np.eye(2)
    d = np.array([[2.0], [3.0]])

    s = controlboros.StateSpace(a, b, c, d)

    assert np.all(s.output([1.0, 1.0], [1.0]) == np.array([3.0, 4.0]))


def test_output_miso():
    """Test output equation with multiple inputs, single output."""

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    c = np.array([[1.0, 1.0]])
    d = np.array([[2.0, 3.0]])

    s = controlboros.StateSpace(a, b, c, d)

    assert np.all(s.output([1.0, 1.0], [1.0, 1.0]) == np.array([7.0]))


def test_output_mimo():
    """Test output equation with multiple inputs, multiple outputs."""

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    c = np.eye(2)
    d = np.eye(2)

    s = controlboros.StateSpace(a, b, c, d)

    assert np.all(s.output([1.0, 1.0], [2.0, 3.0]) == np.array([3.0, 4.0]))


def test_invalid_dynamics_matrix_dimensions():
    """Test exception if A is not square."""
    a = np.zeros((2, 1))
    b = np.zeros((2, 1))
    c = np.zeros((1, 2))

    with pytest.raises(ValueError) as excinfo:
        s = controlboros.StateSpace(a, b, c)
    assert "Invalid matrix dimensions" in str(excinfo.value)


def test_invalid_input_matrix_dimensions():
    """Test exception if B and A have different number of rows."""
    a = np.zeros((2, 2))
    b = np.zeros((1, 1))
    c = np.zeros((1, 2))

    with pytest.raises(ValueError) as excinfo:
        s = controlboros.StateSpace(a, b, c)
    assert "Invalid matrix dimensions" in str(excinfo.value)


def test_invalid_output_matrix_dimensions():
    """Test exception if C and A have different number of columns."""
    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    c = np.zeros((1, 3))

    with pytest.raises(ValueError) as excinfo:
        s = controlboros.StateSpace(a, b, c)
    assert "Invalid matrix dimensions" in str(excinfo.value)


def test_invalid_feedthrough_matrix_dimensions():
    """Test exception if D does not match to B and C."""
    a = np.zeros((2, 2))
    b = np.zeros((2, 3))
    c = np.zeros((4, 2))
    d = np.zeros((4, 2))

    with pytest.raises(ValueError) as excinfo:
        s = controlboros.StateSpace(a, b, c, d)
    assert "Invalid matrix dimensions" in str(excinfo.value)


def test_human_friendly_form():
    """Test the __str__() method of a StateSpace object."""
    a = np.array([[1.0, 2.0], [0.0, 1.0]])
    b = np.array([[1.0, 2.0], [3.0, 2.0]])
    c = np.zeros((1, 2))

    s = controlboros.StateSpace(a, b, c)

    reference = \
        "LTI discrete-time system.\n\n" \
        "Dynamics matrix A:\n" \
        "[[ 1.  2.]\n" \
        " [ 0.  1.]]\n\n" \
        "Input matrix B:\n" \
        "[[ 1.  2.]\n" \
        " [ 3.  2.]]\n\n" \
        "Output matrix C:\n" \
        "[[ 0.  0.]]\n\n" \
        "Feedthrough matrix D:\n" \
        "[[ 0.  0.]]\n"

    assert s.__str__() == reference
