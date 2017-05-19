"""Tests for the abstract system class."""
from controlboros import AbstractSystem
import numpy as np
import pytest

# Make AbstractSystem concrete
# pylint: disable=E0110
AbstractSystem.__abstractmethods__ = set()


def test_default_initial_value():
    """Use zero vector as a default initial value."""
    num_states = 10
    s = AbstractSystem(num_states)
    assert np.all(s.get_state() == np.zeros((num_states,)))


def test_state_getter_setter():
    """Test getter and setter for the internal state."""
    num_states = 10
    s = AbstractSystem(num_states)

    rand_state = np.random.rand(num_states)
    s.set_state(rand_state)

    assert np.all(s.get_state() == rand_state)


def test_state_setter_invalid_shape():
    """Test exception if trying to set incorrect state shape."""
    num_states = 4
    s = AbstractSystem(num_states)

    with pytest.raises(ValueError) as excinfo:
        s.set_state(np.zeros((5,)))
    assert "could not broadcast" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        s.set_state(np.zeros((4, 1)))
    assert "could not broadcast" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        s.set_state(np.zeros((2,)))
    assert "could not broadcast" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        s.set_state(np.zeros((2, 2)))
    assert "could not broadcast" in str(excinfo.value)


def test_set_state_to_zero():
    """Test set_state_to_zero() method."""
    num_states = 10
    s = AbstractSystem(num_states)
    s.set_state(np.ones((num_states,)))
    assert np.all(s.get_state() == np.ones((num_states,)))
    s.set_state_to_zero()
    assert np.all(s.get_state() == np.zeros((num_states,)))


def test_push_pure():
    """Test pure push method.

    This test is bogus, since abstract methods dynamics() and output()
    are undefined. But we can at least check if these methods are
    called "correctly".
    """
    s = AbstractSystem(2)
    assert s.push_pure(3.14, 2.71) == (None, None)


def test_push_stateful():
    """Test stateful push method.

    This test is bogus, since abstract methods dynamics() and output()
    are undefined. But we can at least check if these methods are
    called "correctly" and if the internal state is updated.
    """
    s = AbstractSystem(2)
    val = s.push_stateful(3.14)
    assert s.get_state() is None
    assert val is None
