"""Tests the system rate wrapper."""
from controlboros import StateSpaceBuilder, RateWrapper
import numpy as np
import pytest


# Define a threshold for comparing floating-point numbers
THRESHOLD = np.sqrt(np.finfo(np.float64).eps)


def test_wrapping_exception_non_integer():
    """Test exception if rate multiplier is not integer."""
    my_sys = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()

    with pytest.raises(ValueError) as excinfo:
        RateWrapper(my_sys, 1.1)
    assert "Rate multiplier must be an integer" in str(excinfo.value)


def test_wrapping_exception_leq_one():
    """Test exception if rate multiplier is less or equal 1."""
    my_sys = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()

    with pytest.raises(ValueError) as excinfo:
        RateWrapper(my_sys, 1)
    assert "Rate multiplier must be greater than 1" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        RateWrapper(my_sys, 0)
    assert "Rate multiplier must be greater than 1" in str(excinfo.value)


def test_wrapped_getter_setter():
    """Test if getters and setters are wrapped correctly."""
    num_states = 10
    num = np.ones((1,))
    den = np.ones((num_states + 1,))

    sys_tf = StateSpaceBuilder().from_tf(num, den)\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 2)

    rand_state = np.random.rand(num_states)
    s.set_state(rand_state)
    assert np.all(s.get_state() == rand_state)


def test_wrapped_set_state_to_zero():
    """Test wrapped set_state_to_zero() method."""
    num_states = 10
    num = np.ones((1,))
    den = np.ones((num_states + 1,))

    sys_tf = StateSpaceBuilder().from_tf(num, den)\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 2)

    s.set_state(np.ones((num_states,)))
    assert np.all(s.get_state() == np.ones((num_states,)))
    s.set_state_to_zero()
    assert np.all(s.get_state() == np.zeros((num_states,)))


def test_rate_wrapping():
    """Test the rate wrapping."""
    sys_tf = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 5)

    assert np.all(s.push_stateful([1.0]) == [0.0])  # t = 0.0 s
    assert np.all(s.push_stateful([1.0]) == [0.0])  # t = 0.2 s
    assert np.all(s.push_stateful([1.0]) == [0.0])  # t = 0.4 s
    assert np.all(s.push_stateful([1.0]) == [0.0])  # t = 0.6 s
    assert np.all(s.push_stateful([1.0]) == [0.0])  # t = 0.8 s
    assert np.all(
        np.abs(s.push_stateful([1.0]) - [1.0 - np.exp(-1.0)]) < THRESHOLD
        )  # t = 1.0 s


def test_wrapped_push_pure():
    """Test wrapped push_pure()."""
    sys_tf = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 5)
    new_state, output = s.push_pure([0.0], [1.0])
    assert np.all(np.abs(new_state - [1.0 - np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(output - [0.0]) < THRESHOLD)


def test_wrapped_dynamics():
    """Test wrapped dynamics()."""
    sys_tf = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 5)
    assert np.all(
        np.abs(s.dynamics([0.0], [1.0]) - [1.0 - np.exp(-1.0)]) < THRESHOLD
        )


def test_wrapped_output():
    """Test wrapped output()."""
    sys_tf = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 5)
    assert np.all(np.abs(s.output([4.2], [1.0]) - [4.2]) < THRESHOLD)


def test_reset_wrapper():
    """Test reset_wrapper()."""
    sys_tf = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    s = RateWrapper(sys_tf, 5)

    for _ in range(7):
        _ = s.push_stateful([1.0])
    assert s._counter == 2
    assert s._output_buffer is not None

    s.reset_wrapper()
    assert s._counter == 0
    assert s._output_buffer is None
