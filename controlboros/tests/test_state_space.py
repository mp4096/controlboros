import controlboros
import numpy as np


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
