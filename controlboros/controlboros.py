"""The Controlboros framework.

Abstract class for dynamical systems
====================================

:class:`AbstractSystem`
-----------------------

.. autoclass:: AbstractSystem
    :members:
    :undoc-members:
    :show-inheritance:

Generic discrete-time LTI models
================================

:class:`StateSpace`
-------------------

.. autoclass:: StateSpace
    :members:
    :undoc-members:
    :show-inheritance:

:class:`StateSpaceBuilder`
--------------------------

.. autoclass:: StateSpaceBuilder
    :members:
    :undoc-members:
    :show-inheritance:

Rate wrapper for multirate simulations
======================================

:class:`RateWrapper`
--------------------
.. autoclass:: RateWrapper
    :members:
    :undoc-members:
    :show-inheritance:

Some useful systems
===================

:class:`Time delay`
-------------------

.. autoclass:: TimeDelay
    :members:
    :undoc-members:
    :show-inheritance:
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import signal


def _state_space_dimensions_ok(a, b, c, d):
    r"""Check if matrices A, B, C, D form a valid state-space system.

    Parameters
    ----------
    a : (num_states, num_states) array_like
        state matrix :math:`\mathbf{A}`
    b : (num_states, num_inputs) array_like
        input matrix :math:`\mathbf{B}`
    c : (num_outputs, num_states) array_like
        output matrix :math:`\mathbf{C}`
    d : (num_outputs, num_inputs) array_like
        feedthrough matrix :math:`\mathbf{D}`

    Returns
    -------
    bool
        ``True`` if matrices $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$,
        $\mathbf{D}$ can be concatenated to the following block:

        .. math::

            \begin{bmatrix}
                \mathbf{A} & \mathbf{B} \\
                \mathbf{C} & \mathbf{D}
            \end{bmatrix}

        ``False``` otherwise.
    """
    a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)

    aa_ok = a.shape[0] == a.shape[1]
    ab_ok = a.shape[0] == b.shape[0]
    cd_ok = c.shape[0] == d.shape[0]
    ac_ok = a.shape[1] == c.shape[1]
    bd_ok = b.shape[1] == d.shape[1]

    return aa_ok and ab_ok and cd_ok and ac_ok and bd_ok


class AbstractSystem(metaclass=ABCMeta):
    """Abstract discrete-time systems."""

    def __init__(self, num_states):
        """Create an abstract system object.

        System state is always set to zero during initialisation.

        Parameters
        ----------
        num_states : int
            number of system state variables
        """
        self._state = np.zeros((num_states,))

    def get_state(self):
        """Get current system state.

        Returns
        -------
        (num_states,) ndarray
            current system state
        """
        return self._state

    def set_state(self, state):
        """Set current system state.

        Parameters
        ----------
        state : (num_states,) array_like
            new system state
        """
        self._state[:] = np.array(state)

    def set_state_to_zero(self):
        """Set current system state to zeros."""
        self._state[:] = 0.0

    def push_stateful(self, inp):
        """Push an input into system, get the output, update system state.

        Parameters
        ----------
        inp : (num_inputs,) array_like
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        self._state, output = self.push_pure(self._state, inp)
        return output

    def push_pure(self, state, inp):
        """Push an input into system, get the output and new state.

        Note
        ----
        This function does not affect the system state stored in the object!

        Parameters
        ----------
        state : (num_states,) array_like
            state vector at time :math:`k`

        inp : (num_inputs,) array_like
            input vector at time :math:`k`

        Returns
        -------
        new_state : (num_states,) ndarray
            state vector at time :math:`k + 1`

        output : (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        state = np.array(state)
        inp = np.array(inp)

        new_state = self.dynamics(state, inp)
        output = self.output(state, inp)
        return new_state, output

    @abstractmethod
    def dynamics(self, state, inp):
        """Abstract system dynamics.

        Note
        ----
        This must be a pure function!

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_states,) ndarray
            state vector at time :math:`k + 1`
        """
        pass

    @abstractmethod
    def output(self, state, inp):
        """Abstract system output.

        Note
        ----
        This must be a pure function!

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        pass


class StateSpace(AbstractSystem):
    """LTI discrete-time state-space systems."""

    def __init__(self, a, b, c, d=None):
        r"""Create an LTI discrete-time state-space system.

        Parameters
        ----------
        a : (num_states, num_states) array_like
            state matrix :math:`\mathbf{A}`

        b : (num_states, num_inputs) array_like
            input matrix :math:`\mathbf{B}`

        c : (num_outputs, num_states) array_like
            output matrix :math:`\mathbf{C}`

        d : None or (num_outputs, num_inputs) array_like
            feedthrough matrix :math:`\mathbf{D}`;
            defaults to :math:`\mathbf{0}` if not specified

        Raises
        ------
        ValueError
            If matrix dimensions are inconsistent, i.e. unable to form
            the following block matrix:

            .. math::

                \begin{bmatrix}
                    \mathbf{A} & \mathbf{B} \\
                    \mathbf{C} & \mathbf{D}
                \end{bmatrix}
        """
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        # Set ``self.num_states``
        super().__init__(self.a.shape[1])
        self.num_inputs = self.b.shape[1]
        self.num_outputs = self.c.shape[0]

        if d is None:
            self.d = np.zeros((self.num_outputs, self.num_inputs))
        else:
            self.d = np.array(d)

        if not _state_space_dimensions_ok(self.a, self.b, self.c, self.d):
            raise ValueError("Invalid matrix dimensions.")

    def dynamics(self, state, inp):
        r"""Linear discrete-time dynamics equation.

        .. math::

            \mathbf{x}[k + 1] = \mathbf{A} \mathbf{x}[k] +
                                \mathbf{B} \mathbf{u}[k]

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_states,) ndarray
            state vector at time :math:`k + 1`
        """
        return self.a @ state + self.b @ inp

    def output(self, state, inp):
        r"""Linear output equation.

        .. math::

            \mathbf{y}[k] = \mathbf{C} \mathbf{x}[k] +
                            \mathbf{D} \mathbf{u}[k]

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        return self.c @ state + self.d @ inp

    def __str__(self):
        """Get human-friendly representation of the object."""
        return "LTI discrete-time system.\n\n" \
            "State matrix A:\n{:s}\n\n" \
            "Input matrix B:\n{:s}\n\n" \
            "Output matrix C:\n{:s}\n\n" \
            "Feedthrough matrix D:\n{:s}\n".format(
                np.array_str(self.a),
                np.array_str(self.b),
                np.array_str(self.c),
                np.array_str(self.d),
                )


class StateSpaceBuilder():
    r"""Builder for LTI discrete-time state-space models.

    This class allows you to conveniently create discrete-time
    :class:`controlboros.StateSpace`
    objects from continuous-time transfer functions,
    zeros, poles, gain or state-space models.

    Examples
    --------
    Use a builder like this:

        >>> my_sys = StateSpaceBuilder().from_tf([1.0], [1.0, 1.5, 1.0])\
        ...                             .discretise(1.0e-3)\
        ...                             .build()

    """

    def __init__(self):
        """Create an empty object."""
        self._a = None
        self._b = None
        self._c = None
        self._d = None
        self._discrete = False

    def build(self):
        """Build a :class:`StateSpace` system.

        Returns
        -------
        controlboros.StateSpace
            built discrete-time LTI system
        """
        # Check if all system matrices are specified
        if any(m is None for m in [self._a, self._b, self._c, self._d]):
            raise ValueError("Cannot build, no system specified.")
        # Check if the system has been discretised
        if not self._discrete:
            raise ValueError("Cannot build, discretise the system first.")
        # If everything ok, return the built system
        return StateSpace(self._a, self._b, self._c, self._d)

    def from_tf(self, num, den):
        """Define a SISO system using the transfer function representation.

        Parameters
        ----------
        num : (len_numerator,) array_like
            transfer function numerator in descending exponent order

        den : (len_denominator,) array_like
            transfer function denominator in descending exponent order

        Returns
        -------
        controlboros.StateSpaceBuilder
            intermediate builder object
        """
        tf = signal.TransferFunction(num, den).to_ss()
        self._a, self._b, self._c, self._d = tf.A, tf.B, tf.C, tf.D
        return self

    def from_zpk(self, zeros, poles, gain):
        """Define a SISO system using the zeros, poles, gain representation.

        Parameters
        ----------
        zeros : (num_zeros,) array_like
            system zeros

        poles : (num_poles,) array_like
            system poles

        gain : float
            system gain

        Returns
        -------
        controlboros.StateSpaceBuilder
            intermediate builder object
        """
        zpk = signal.ZerosPolesGain(zeros, poles, gain).to_ss()
        self._a, self._b, self._c, self._d = zpk.A, zpk.B, zpk.C, zpk.D
        return self

    def from_ss(self, a, b, c, d=None):
        r"""Define a system using the state-space representation.

        Parameters
        ----------
        a : (num_states, num_states) array_like
            state matrix :math:`\mathbf{A}`

        b : (num_states, num_inputs) array_like
            input matrix :math:`\mathbf{B}`

        c : (num_outputs, num_states) array_like
            output matrix :math:`\mathbf{C}`

        d : None or (num_outputs, num_inputs) array_like
            feedthrough matrix :math:`\mathbf{D}`;
            defaults to :math:`\mathbf{0}` if not specified

        Returns
        -------
        controlboros.StateSpaceBuilder
            intermediate builder object
        """
        self._a, self._b, self._c = np.array(a), np.array(b), np.array(c)

        num_inputs, num_outputs = self._b.shape[1], self._c.shape[0]

        if d is None:
            self._d = np.zeros((num_outputs, num_inputs))
        else:
            self._d = np.array(d)

        if not _state_space_dimensions_ok(self._a, self._b, self._c, self._d):
            raise ValueError("Invalid matrix dimensions.")

        return self

    def discretise(self, dt, method="zoh", alpha=None):
        """Transform a continuous-time system into a discrete-time one.

        Parameters
        ----------
        dt : float
            the discretisation time step

        method : str, optional
            discretisation method, refer to the documentation for
            :func:`scipy.signal.cont2discrete`

        alpha : float within [0, 1], optional
            weighting parameter, refer to the documentation for
            :func:`scipy.signal.cont2discrete`

        Returns
        -------
        controlboros.StateSpaceBuilder
            intermediate builder object
        """
        self._a, self._b, self._c, self._d, _ = signal.cont2discrete(
            (self._a, self._b, self._c, self._d),
            dt,
            method=method,
            alpha=alpha,
            )
        self._discrete = True
        return self


class RateWrapper(AbstractSystem):
    """Wrap a system to simulate it at a smaller time step size.

    Suppose you want to run a simulation with a 1 ms time step.
    However, one system must be discretised at 10 ms. What should you do?
    You wrap your system into a :class:`controlboros.RateWrapper` object.
    Then your system is called only every 10 ms, and inbetween
    the most recent output is used.
    """

    def __init__(self, system, rate_multiplier):
        """Create a wrapped system.

        Parameters
        ----------
        system : obj derived from controlboros.AbstractSystem
            system to wrap

        rate_multiplier : int
            sample rate multiplier, i.e. for how many samples
            should the system's output be held constant;
            must be an integer greater than 1
        """
        if not isinstance(rate_multiplier, int):
            raise ValueError("Rate multiplier must be an integer.")

        if rate_multiplier <= 1:
            raise ValueError("Rate multiplier must be greater than 1.")

        self._output_buffer = None
        self._counter = 0
        self._system = system
        self.rate_multiplier = rate_multiplier

    def get_state(self):
        """Get current state of the wrapped system.

        Returns
        -------
        (num_states,) ndarray
            current system state
        """
        return self._system.get_state()

    def set_state(self, state):
        """Set current state of the wrapped system.

        Parameters
        ----------
        state : (num_states,) array_like
            new system state
        """
        self._system.set_state(state)

    def set_state_to_zero(self):
        """Set current state of the wrapped system to zeros."""
        self._system.set_state_to_zero()

    def push_stateful(self, inp):
        """Call wrapped system's push_stateful() or use buffered output.

        This function uses an internal counter to decide whether the wrapped
        system (and the buffered output) should be updated. If not, it just
        returns the buffered output.

        Parameters
        ----------
        inp : (num_inputs,) array_like
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        if self._counter >= self.rate_multiplier:
            self._counter = 0

        if self._counter == 0:
            self._output_buffer = self._system.push_stateful(inp)

        self._counter += 1
        return self._output_buffer

    def push_pure(self, state, inp):
        """Push an input into the wrapped system, get the output and new state.

        Note
        ----
        This function does not affect the states stored in the wrapper object
        and the wrapped system!

        Parameters
        ----------
        state : (num_states,) array_like
            state vector at time :math:`k`

        inp : (num_inputs,) array_like
            input vector at time :math:`k`

        Returns
        -------
        new_state : (num_states,) ndarray
            state vector at time :math:`k + 1`

        output : (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        return self._system.push_pure(state, inp)

    def dynamics(self, state, inp):
        """Call wrapped system's dynamics() method.

        Note
        ----
        This must be a pure function!

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_states,) ndarray
            state vector at time :math:`k + 1`
        """
        return self._system.dynamics(state, inp)

    def output(self, state, inp):
        """Call wrapped system's output() method.

        Note
        ----
        This must be a pure function!

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        return self._system.output(state, inp)

    def reset_wrapper(self):
        """Reset the wrapper counter and output."""
        self._output_buffer = None
        self._counter = 0


class TimeDelay(AbstractSystem):
    r"""Discrete time delay system.

    This object provides a :math:`m` samples long delay line.
    It supports scalar as well as vector-valued signals.

    Its state vector is given by the following vector
    of dimension :math:`mn`:

    .. math::

        \begin{bmatrix}
            x_1[k - m] \\
            \vdots \\
            x_n[k - m] \\
            x_1[k - 2] \\
            \vdots \\
            x_n[k - 2] \\
            x_1[k - 1] \\
            \vdots \\
            x_n[k - 1]
        \end{bmatrix}

    Note
    ----
    This implementation is not a circular buffer / FIFO queque,
    i.e. it is not the most efficient implementation. Instead, it stays
    in line with the Controlboros philosophy of stateful wrappers around
    pure functions, thus emphasising ease of debugging over efficiency.
    """

    def __init__(self, num_samples, dim=1):
        """Create a time delay line.

        Parameters
        ----------
        num_samples : int
            delay length (in samples), must be greater than 0

        dim : int, optional
            signal dimension, scalar by default
        """
        if not isinstance(num_samples, int):
            raise ValueError("Number of delay samples must be an integer.")

        if num_samples <= 0:
            raise ValueError("Number of delay samples must be greater than 0.")

        if not isinstance(dim, int):
            raise ValueError("Dimension of signal must be an integer.")

        if dim <= 0:
            raise ValueError("Dimension of signal must be greater than 0.")

        self.dim = dim
        super().__init__(num_samples*dim)

    def dynamics(self, state, inp):
        """Time line dynamics.

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_states,) ndarray
            state vector at time :math:`k + 1`
        """
        return np.concatenate((state[self.dim:], inp))

    def output(self, state, inp):
        """Time delay output.

        The output is the signal delayed by :math:`m` samples.

        Parameters
        ----------
        state : (num_states,) ndarray
            state vector at time :math:`k`

        inp : (num_inputs,) ndarray
            input vector at time :math:`k`

        Returns
        -------
        (num_outputs,) ndarray
            output vector at time :math:`k` *(sic!)*
        """
        return state[:self.dim]
