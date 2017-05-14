"""The Controlboros framework."""

from abc import ABCMeta, abstractmethod
import numpy as np


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
        self._state = np.array(state)

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
            dynamics matrix :math:`\mathbf{A}`
        b : (num_states, num_inputs) array_like
            input matrix :math:`\mathbf{B}`
        c : (num_outputs, num_states) array_like
            output matrix :math:`\mathbf{C}`
        d : None or (num_outputs, num_inputs) array_like
            feedthrough matrix :math:`\mathbf{D}`

            Defaults to :math:`\mathbf{0}` if not specified.

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

        aa_ok = self.a.shape[0] == self.a.shape[1]
        ab_ok = self.a.shape[0] == self.b.shape[0]
        cd_ok = self.c.shape[0] == self.d.shape[0]
        ac_ok = self.a.shape[1] == self.c.shape[1]
        bd_ok = self.b.shape[1] == self.d.shape[1]

        if not (aa_ok and ab_ok and cd_ok and ac_ok and bd_ok):
            raise ValueError("Invalid matrix dimensions.")

    def dynamics(self, state, inp):
        r"""Linear discrete-time dynamics equation.

        .. math::

            \mathbf{x}_{k + 1} = \mathbf{A} \mathbf{x}_k +
                                 \mathbf{B} \mathbf{u}_k

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

            \mathbf{y}_{k} = \mathbf{C} \mathbf{x}_k +
                             \mathbf{D} \mathbf{u}_k

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
            "Dynamics matrix A:\n{:s}\n\n" \
            "Input matrix B:\n{:s}\n\n" \
            "Output matrix C:\n{:s}\n\n" \
            "Feedthrough matrix D:\n{:s}\n".format(
                np.array_str(self.a),
                np.array_str(self.b),
                np.array_str(self.c),
                np.array_str(self.d),
                )
