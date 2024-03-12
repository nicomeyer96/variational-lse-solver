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
import torch
import pennylane as qml
from typing import Callable
from typing_extensions import override

from . import LocalBase
from ..cost_function_types import CostFunctionMode


class LocalOverlap(LocalBase):
    """
    This class allows for evaluation of the local loss term via the Hadamard-overlap test.

    WORK IN PROGRESS
    """

    def __init__(
            self,
            system: list[str | np.ndarray | Callable],
            coeffs: list[float | complex],
            right_side: np.ndarray | Callable,
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        super().__init__(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)
        assert CostFunctionMode.MATRIX != mode
        # self.batched_encoded_system = self.generate_batched_encoded_system()
        # qubit mapping, first one is ancilla
        self.ancilla_qubit = 0
        self.data_qubits_map_upper = {qubit: qubit + 1 for qubit in range(data_qubits)}
        self.data_qubits_map_lower = {qubit: data_qubits + qubit + 1 for qubit in range(data_qubits)}

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        raise NotImplementedError('This method is currently under development. '
                                  'Please resort to the `direct` or `hadamard` evalaution method.')

    def qnode_overlap_local(self) -> Callable:
        dev = qml.device('default.qubit', wires=2 * self.data_qubits + 1)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_overlap_local(weights):
            raise NotImplementedError

        return circuit_overlap_local

    def generate_batched_encoded_system(self) -> dict:
        raise NotImplementedError
