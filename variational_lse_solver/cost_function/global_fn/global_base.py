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
import warnings
from abc import ABC
from typing import Callable

from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import generate_encoded_system


class GlobalBase(ABC):
    """
    This is a base class for the different methods of evaluating the global loss term.
    """

    def __init__(
            self,
            system: np.ndarray | list[str | np.ndarray | Callable],
            coeffs: list[float | complex] | None,
            right_side: np.ndarray | Callable,
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        self.system = system
        self.coeffs = coeffs
        self.right_side = right_side
        self.right_side_fn = right_side if callable(right_side) else self.qnode_right_side()
        self.ansatz = ansatz
        self.data_qubits = data_qubits
        self.mode = mode
        self.imaginary = imaginary
        self.encoded_system = generate_encoded_system(system, mode=mode)

    def cost(self, weights: torch.tensor) -> torch.tensor:
        pass

    def qnode_right_side(self) -> Callable:
        """
        Function that converts the right side of the LSE to a quantum circuit implementing a proportional state.

        :return: Circuit implementing |b>.
        """
        if callable(self.right_side):
            raise RuntimeError('The `right side` is already callable.')
        if not np.isclose(1.0, np.linalg.norm(self.right_side)):
            warnings.warn('The provided right side `b` does not have unit norm, will be normalized.')
            right_side = self.right_side.copy() / np.linalg.norm(self.right_side)
        else:
            right_side = self.right_side

        def circuit_right_side():
            qml.QubitStateVector(right_side, wires=range(int(np.log2(right_side.shape[0]))))

        return circuit_right_side
