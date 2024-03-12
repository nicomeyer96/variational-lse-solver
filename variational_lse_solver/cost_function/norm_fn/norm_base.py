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
from abc import ABC
from typing import Callable

from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import generate_encoded_system


class NormBase(ABC):
    """
    This is a base class for the different methods of evaluating the norm term.
    """

    def __init__(
            self,
            system: np.ndarray | list[str | np.ndarray | Callable],
            coeffs: list[float | complex] | None,
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        self.system = system
        self.coeffs = coeffs
        self.ansatz = ansatz
        self.data_qubits = data_qubits
        self.mode = mode
        self.imaginary = imaginary
        self.encoded_system = generate_encoded_system(system, mode=mode)

    def cost(self, weights: torch.tensor) -> torch.tensor:
        pass

