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
from typing import Callable
from typing_extensions import override

from . import NormBase
from ..cost_function_types import CostFunctionMode


class NormDirect(NormBase):
    """
    This class allows for direct evaluation of the norm.

    It computes <psi|psi> = A|V(alpha)>^t * A|V(alpha)> in a fully differentiable manner.
    """

    def __init__(
            self,
            system: np.ndarray,
            coeffs: None,
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        """
        Direct evaluation of norm.

        :param system: System matrix.
        :param coeffs: Ignored for direct evalaution.
        :param ansatz: The variational quantum circuit.
        :param data_qubits: Number of qubits in the VQC.
        :param mode: In which mode to run (i.e. in which form `system` is provided).
        :param imaginary: Whether to evaluate imaginary terms.
        """

        super().__init__(system, coeffs, ansatz, data_qubits, mode, imaginary)
        assert CostFunctionMode.MATRIX == mode

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Calculates the norm for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Norm value (with grad_fn).
        """
        # evolve the variational state |V(alpha)>
        state_x = self.qnode_evolve_x()(weights)
        # compose to |psi> = A|V(alpha)>
        state_psi = torch.matmul(torch.tensor(self.system, dtype=torch.complex128), state_x)
        # evaluate <psi|psi>
        norm = torch.abs(torch.dot(state_psi, torch.conj(state_psi)))
        return norm

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
