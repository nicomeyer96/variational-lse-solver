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

from . import GlobalBase
from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import controlled_system


class GlobalCoherent(GlobalBase):
    """
    This class allows for evaluation of the global loss term in a coherent manner, as described in
    https://pennylane.ai/qml/demos/tutorial_coherent_vqls/.

    One coherently computes <b|Ax(alpha)>, with access to a decomposition of A as sum c_m A_m,
    with sum c_m = 1 and c_m >= 0 for all terms.
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
        assert np.isclose(1.0, np.sum(np.square(coeffs)))
        # qubit mapping and control values (corresponding to binary indices), first log_2(m) many are ancilla
        self.ancilla_qubits = list(range(int(np.ceil(np.log2(len(coeffs))))))
        ancilla_qubits_control_values = [bin(b)[2:].zfill(len(self.ancilla_qubits))[::-1] for b in range(len(coeffs))]
        self.ancilla_qubits_control_values = [[int(b_) for b_ in b] for b in ancilla_qubits_control_values]
        self.data_qubits_map = {qubit: qubit + len(self.ancilla_qubits) for qubit in range(data_qubits)}
        # determine measurement indices of ground state of entire system and ancilla qubits
        self.index_ground_state_all = np.array(0)
        self.indices_ground_state_ancilla = np.arange(2 ** data_qubits)

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Coherently calculates the global loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Global loss value (with grad_fn).
        """

        # extract raw measurement probabilities
        measurement_probabilities = self.qnode_coherent_global()(weights)

        # extract probabilities of measuring the ground state of full / ancilla system
        ground_state_all = torch.gather(measurement_probabilities, dim=0, index=torch.tensor(self.index_ground_state_all))
        ground_state_ancilla = torch.sum(torch.gather(measurement_probabilities, dim=0,
                                                      index=torch.tensor(self.indices_ground_state_ancilla)))

        # return normalized value
        return ground_state_all / ground_state_ancilla

    def qnode_coherent_global(self) -> Callable:
        """
        Quantum node that realizes the coherent test for evaluating the global loss.

        :return: Circuit handle implementing the coherent realization of the global loss.
        """
        dev = qml.device('default.qubit', wires=len(self.ancilla_qubits) + self.data_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_coherent_global(weights):
            """
            Circuit that realizes the coherent test for evaluating the global loss term.

            :param weights: Parameters for the VQC.
            :return: Bistring measurement probabilities of entire system.
            """

            # >>> CIRCUIT CONSTRUCTION <<<

            # amplitude encoding of square-root pre-factors
            qml.AmplitudeEmbedding(self.coeffs, wires=self.ancilla_qubits, pad_with=0.0)

            # apply variational circuit ti data qubits
            qml.map_wires(self.ansatz, self.data_qubits_map)(weights)

            # apply controlled versions (with varying control values) of encoded systems A_m to data qubits
            for A_system_m, control_values in zip(self.encoded_system, self.ancilla_qubits_control_values):
                controlled_system(A_system_m, self.system, self.mode, self.ancilla_qubits, self.data_qubits_map,
                                  control_values=control_values)

            # apply adjoint unitary implementing right side to data qubits
            qml.map_wires(qml.adjoint(self.right_side_fn), self.data_qubits_map)()

            # adjoint amplitude encoding of square-root pre-factors
            qml.adjoint(qml.AmplitudeEmbedding)(self.coeffs, wires=self.ancilla_qubits, pad_with=0.0)

            # return bitstring measurement probabilities
            return qml.probs()

        return circuit_coherent_global
