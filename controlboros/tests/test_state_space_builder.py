"""Tests the LTI discrete-time system builder."""
from controlboros import StateSpaceBuilder
import numpy as np
import pytest


# Define a threshold for comparing floating-point numbers
THRESHOLD = np.sqrt(np.finfo(np.float64).eps)


def test_build_from_tf():
    """Test building from a transfer function."""
    my_sys = StateSpaceBuilder().from_tf([1.0], [1.0, 1.0])\
                                .discretise(1.0)\
                                .build()
    assert np.all(np.abs(my_sys.a - [np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.b - [1.0 - np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.c - [1.0]) < THRESHOLD)
    assert np.all(np.abs(my_sys.d - [0.0]) < THRESHOLD)


def test_build_from_zpk():
    """Test building from a ZPK representation."""
    my_sys = StateSpaceBuilder().from_zpk([], [-1.0], 1.0)\
                                .discretise(1.0)\
                                .build()
    assert np.all(np.abs(my_sys.a - [np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.b - [1.0 - np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.c - [1.0]) < THRESHOLD)
    assert np.all(np.abs(my_sys.d - [0.0]) < THRESHOLD)


def test_build_from_ss_no_feedthrough():
    """Test building from a cont. state-space model, default feedthrough."""
    my_sys = StateSpaceBuilder().from_ss([[-1.0]], [[1.0]], [[1.0]])\
                                .discretise(1.0)\
                                .build()
    assert np.all(np.abs(my_sys.a - [np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.b - [1.0 - np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.c - [1.0]) < THRESHOLD)
    assert np.all(np.abs(my_sys.d - [0.0]) < THRESHOLD)


def test_build_from_ss_with_feedthrough():
    """Test building from a cont. state-space model, specific feedthrough."""
    my_sys = StateSpaceBuilder().from_ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]])\
                                .discretise(1.0)\
                                .build()
    assert np.all(np.abs(my_sys.a - [np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.b - [1.0 - np.exp(-1.0)]) < THRESHOLD)
    assert np.all(np.abs(my_sys.c - [1.0]) < THRESHOLD)
    assert np.all(np.abs(my_sys.d - [0.0]) < THRESHOLD)


def test_build_from_ss_invalid_dims():
    """Test building from a cont. state-space model, invalid dimensions."""
    with pytest.raises(ValueError) as excinfo:
        StateSpaceBuilder().from_ss([[-1.0]], [[1.0]], [[1.0, 1.0]])
    assert "Invalid matrix dimensions" in str(excinfo.value)


def test_build_without_system():
    """Test exception attempting build without a system."""
    with pytest.raises(ValueError) as excinfo:
        StateSpaceBuilder().build()
    assert "Cannot build, no system specified" in str(excinfo.value)


def test_build_continuous_system():
    """Test exception attempting build of a continuous-time system."""
    with pytest.raises(ValueError) as excinfo:
        StateSpaceBuilder().from_tf([1.0], [1.0, 1.0]).build()
    assert "Cannot build, discretise the system first" in str(excinfo.value)
