# This code is part of the variational-lse-solver library.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pennylane as qml
import torch
import functools
from typing import Callable
from typing_extensions import override

from . import LocalBase
from ..cost_function_types import CostFunctionMode


class LocalDirect(LocalBase):
    def __init__(
            self,
            system: np.ndarray,
            coeffs: None,
            right_side: np.ndarray,
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        """
        Direct evaluation of the local loss.

        :param system: System matrix.
        :param coeffs: Ignored for direct evalaution.
        :param right_side:
        :param ansatz: The variational quantum circuit.
        :param data_qubits: Number of qubits in the VQC.
        :param mode: In which mode to run (i.e. in which form `system` is provided).
        :param imaginary: Whether to evaluate imaginary terms.
        """
        super().__init__(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)
        assert CostFunctionMode.MATRIX == mode

    @functools.cached_property
    def unitary_right_side(self):
        """ Return the unitary representation of the circuit implementation the right side of the LSE. """
        # noinspection PyCallingNonCallable, PyTypeChecker
        return torch.tensor(qml.matrix(self.right_side_fn, wire_order=range(self.data_qubits))(), dtype=torch.complex128)

    @functools.cached_property
    def local_term(self):
        """ Construct the unitary in the middel of equation (7) from https://quantum-journal.org/papers/q-2023-11-22-1188/
            required for evaluating the local loss. """
        _0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        _I = np.eye(2, dtype=np.float32)
        # construct unitary in the middle of equation (7)
        local_term = np.zeros((2 ** self.data_qubits, 2 ** self.data_qubits))
        for q in range(self.data_qubits):
            # subtract local_fn terms
            local_term_ = _0 if 0 == q else _I
            for q_ in range(1, self.data_qubits):
                local_term_ = np.kron(local_term_, _0) if q_ == q else np.kron(local_term_, _I)
            local_term += local_term_
        return torch.tensor(local_term / self.data_qubits, dtype=torch.complex128)

    @override
    def cost(self, weights: torch.tensor):
        """
        Calculates the local loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Local loss value (with grad_fn).
        """
        state_x = self.qnode_evolve_x()(weights)
        state_psi = torch.matmul(torch.tensor(self.system, dtype=torch.complex128), state_x)
        # this implements <x|H_L|x> from equation (6) in https://quantum-journal.org/papers/q-2023-11-22-1188/ directly
        # (without the identity part); it uses the decomposition <x|A^tU(1/n \sum|0_j><0_j| o 1_~j)U^tA|x> = <psi|U(...)|U^t|psi>
        loss_raw = torch.matmul(torch.matmul(self.unitary_right_side, self.local_term), torch.adjoint(self.unitary_right_side))
        return torch.abs(torch.dot(torch.conj(state_psi), torch.matmul(loss_raw, state_psi)))

    def qnode_evolve_x(self) -> Callable:
        """
        Quantum node that handles state evolution of the ansatz V(alpha).

        :return: Circuit handle implementing full state evolution.
        """
        dev = qml.device('default.qubit', wires=self.data_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_evolve_x(weights):
            """
            Circuit that handles state evolution if the ansatz V(alpha).

            :param weights: Parameters for the VQC.
            :return: Evolved state.
            """
            self.ansatz(weights)
            return qml.state()

        return circuit_evolve_x
