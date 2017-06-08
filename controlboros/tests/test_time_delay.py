"""Tests the time delay system."""
from controlboros import TimeDelay, FastSisoTimeDelay
import numpy as np
import pytest


def test_delay_exception_delay_non_integer():
    """Test exception if num samples is not integer."""
    with pytest.raises(ValueError) as excinfo:
        TimeDelay(1.1)
    assert "Number of delay samples must be an integer" in str(excinfo.value)


def test_delay_exception_delay_leq_zero():
    """Test exception if num samples is less or equal 0."""
    ref_msg = "Number of delay samples must be greater than 0"

    with pytest.raises(ValueError) as excinfo:
        TimeDelay(0)
    assert ref_msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        TimeDelay(-1)
    assert ref_msg in str(excinfo.value)


def test_delay_exception_dim_non_integer():
    """Test exception if signal dim is not integer."""
    with pytest.raises(ValueError) as excinfo:
        TimeDelay(1, dim=1.1)
    assert "Dimension of signal must be an integer" in str(excinfo.value)


def test_delay_exception_dim_leq_zero():
    """Test exception if signal dim is less or equal 0."""
    ref_msg = "Dimension of signal must be greater than 0"

    with pytest.raises(ValueError) as excinfo:
        TimeDelay(1, dim=0)
    assert ref_msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        TimeDelay(1, dim=-1)
    assert ref_msg in str(excinfo.value)


def test_delay_dynamics():
    """Test delay line dynamics."""
    td = TimeDelay(2, dim=2)
    assert np.all(
        td.dynamics(np.zeros((4,)), np.ones((2,))) == [0.0, 0.0, 1.0, 1.0]
        )


def test_delay_output():
    """Test delay line output."""
    td = TimeDelay(2, dim=2)
    assert np.all(
        td.output(np.array([1.0, 2.0, 3.0, 4.0]), None) == [1.0, 2.0]
        )


def test_delay_high_level():
    """Test delay line -- high-level test."""
    td = TimeDelay(3)
    assert np.all(td.push_stateful([1.0]) == [0.0])
    assert np.all(td.push_stateful([2.0]) == [0.0])
    assert np.all(td.push_stateful([3.0]) == [0.0])
    assert np.all(td.push_stateful([4.0]) == [1.0])
    assert np.all(td.push_stateful([5.0]) == [2.0])
    assert np.all(td.push_stateful([6.0]) == [3.0])


def test_fast_delay_exception_delay_non_integer():
    """Test exception if num samples is not integer."""
    with pytest.raises(ValueError) as excinfo:
        FastSisoTimeDelay(1.1)
    assert "Number of delay samples must be an integer" in str(excinfo.value)


def test_fast_delay_exception_delay_leq_zero():
    """Test exception if num samples is less or equal 0."""
    ref_msg = "Number of delay samples must be greater than 0"

    with pytest.raises(ValueError) as excinfo:
        FastSisoTimeDelay(0)
    assert ref_msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        FastSisoTimeDelay(-1)
    assert ref_msg in str(excinfo.value)


def test_fast_delay_dynamics():
    """Test fast delay line dynamics."""
    td = FastSisoTimeDelay(2)
    assert td.dynamics(1, 1) is None


def test_fast_delay_output():
    """Test fast delay line output."""
    td = FastSisoTimeDelay(2)
    assert td.output(1, 1) is None


def test_fast_delay_push_pure():
    """Test fast delay line stateless push function."""
    td = FastSisoTimeDelay(2)
    assert td.push_pure(1, 1) is None


def test_fast_delay_high_level():
    """Test fast delay line -- high-level test."""
    td = FastSisoTimeDelay(3)
    assert td.push_stateful(1.0) == 0.0
    assert td.push_stateful(2.0) == 0.0
    assert td.push_stateful(3.0) == 0.0
    assert td.push_stateful(4.0) == 1.0
    assert td.push_stateful(5.0) == 2.0
    assert td.push_stateful(6.0) == 3.0


def test_fast_delay_get_state():
    """Test fast delay line state getter."""
    td = FastSisoTimeDelay(3)
    assert td.push_stateful(1.0) == 0.0
    assert td.push_stateful(2.0) == 0.0
    assert td.push_stateful(3.0) == 0.0
    assert np.all(td.get_state() == [1.0, 2.0, 3.0])


def test_fast_delay_set_state():
    """Test fast delay line state setter."""
    td = FastSisoTimeDelay(3)
    td.set_state([1.0, 2.0, 3.0])
    assert td.push_stateful(4.0) == 1.0
    assert td.push_stateful(5.0) == 2.0
    assert td.push_stateful(6.0) == 3.0
    assert td.push_stateful(0.0) == 4.0
    assert td.push_stateful(0.0) == 5.0
    assert td.push_stateful(0.0) == 6.0


def test_fast_delay_set_state_to_zero():
    """Test fast delay line state-to-zero setter."""
    td = FastSisoTimeDelay(3)
    assert td.push_stateful(1.0) == 0.0
    assert td.push_stateful(2.0) == 0.0
    assert td.push_stateful(3.0) == 0.0
    td.set_state_to_zero()
    assert td.push_stateful(4.0) == 0.0
    assert td.push_stateful(5.0) == 0.0
    assert td.push_stateful(6.0) == 0.0
