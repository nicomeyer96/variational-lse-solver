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

from . import GlobalBase
from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import tensor_from_array, batch_dimension_reversal, controlled_system
from ... import gates


class GlobalHadamard(GlobalBase):
    """
    This class allows for evaluation of the global loss via the Hadamard test.

    One needs to compute \sum_m=[0,L-1] \sum_n=[0,L-1] c_m c_n^dagger gamma_mn,
    with gamma_mn as defined in https://quantum-journal.org/papers/q-2023-11-22-1188/.

    We re-formulate this to gamma_mn := gamma_m * gamma_n^dagger, with gamma_m = <0|U^dagger A_m V(alpha)|0>.
    In principle, we could exploit symmetries here, however, it is easier and faster to just evaluate all gamma_m
    explicitly and re-compose everything together, i.e:
    \sum_m=[0,L-1] \sum_n=[0,L-1] c_m c_n^dagger gamma_m gamma_n^dagger
    The imaginary parts of gamma_m are only relevant, if the product of coefficients also has an imaginary part.
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
        """
        Evaluation of global loss via Hadamard test.

        :param system: System matrix given as either Pauli strings, unitaries, or circuits.
        :param coeffs: Corresponding coefficients.
        :param right_side: Right side of the LSE.
        :param ansatz: The variational quantum circuit.
        :param data_qubits: Number of qubits in the VQC.
        :param mode: In which mode to run (i.e. in which form `system` is provided).
        :param imaginary: Whether to evaluate imaginary terms.
        """
        super().__init__(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)
        assert CostFunctionMode.MATRIX != mode
        self.batched_encoded_system = self.generate_batched_encoded_system()
        # qubit mapping, first one is ancilla
        self.ancilla_qubit = 0
        self.data_qubits_map = {qubit: qubit + 1 for qubit in range(data_qubits)}

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Calculates the global loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Global loss value (with grad_fn).
        """

        def calculate_gamma_batched(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all gamma_m in a batched manner. """
            return self.qnode_hadamard_global()(
                weights_,
                encoded_system=self.batched_encoded_system['batched_encoded_system'],
                real_imaginary_flag=tensor_from_array(np.ones(shape=(self.batched_encoded_system['batch_size'], 1)))
                if real_imaginary_flag_
                else tensor_from_array(np.zeros(shape=(self.batched_encoded_system['batch_size'], 1)))
            )

        def calculate_gamma_individual(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all gamma_m in a sequential manner. """
            batched_encoded_system = self.batched_encoded_system['batched_encoded_system']
            return torch.stack([
                self.qnode_hadamard_global()(
                    weights_,
                    encoded_system=encoded_system,
                    real_imaginary_flag=tensor_from_array([1.0]) if real_imaginary_flag_ else tensor_from_array([0.0])
                )
                for encoded_system
                in batched_encoded_system
            ])

        def calculate_gamma(weights_: torch.tensor) -> torch.tensor:
            """ Evaluate real (and optionally imaginary) global loss term. """
            calculate_gamma_fn = self.mode.mode_dependent_value(pauli=calculate_gamma_batched, unitary=calculate_gamma_batched,
                                                                circuit=calculate_gamma_individual, matrix=None)
            gamma_m_ = calculate_gamma_fn(weights_, real_imaginary_flag_=False)
            if self.imaginary:
                gamma_m_ = gamma_m_.type(torch.complex128)
                gamma_m_ += 1.j * calculate_gamma_fn(weights_, real_imaginary_flag_=True)
            return gamma_m_

        # compute gamma_m's (<0|U^tA_mV|0>)
        gamma_m = calculate_gamma(weights)
        assert gamma_m.shape[0] == len(self.system)

        # print(torch.outer(gamma_m, torch.conj(gamma_m)))

        # equation (16) of https://quantum-journal.org/papers/q-2023-11-22-1188/
        # Note: We extend equation (17) to gamma_mn = gamma_m * conjugate(gamma_n), with gamma_m = <0|U^tA_mV|0>,
        #       as it holds conjugate(gamma_m) = conjugate(<0|U^tA_mV|0>) = <0|V^tA_m^tU|0>.
        # IMPORTANT: One must avoid to call `abs(...)` on loss_raw, as done in https://pennylane.ai/qml/demos/tutorial_vqls/.
        #            This potentially introduces unwanted symmetries to the loss landscape and leads to faulty convergence.
        loss_raw = torch.sum(torch.mul(torch.outer(gamma_m, torch.conj(gamma_m)),
                                       torch.outer(self.batched_encoded_system['batched_factors'],
                                                   torch.conj(self.batched_encoded_system['batched_factors']))))
        return loss_raw

    def qnode_hadamard_global(self) -> Callable:
        """
        Quantum node that realizes the Hadamard test for evaluating the gamma_m constituting the global loss.

        :return: Circuit handle implementing the Hadamard test.
        """
        dev = qml.device('default.qubit', wires=self.data_qubits + 1)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_hadamard_global(weights, encoded_system, real_imaginary_flag):
            """
            Circuit that realizes the Hadamard test for evaluating the gamma_m constituting the global loss term.

            :param weights: Parameters for the VQC.
            :param encoded_system: Encoded representations of A_m.
            :param real_imaginary_flag: Whether to evaluate real or imaginary part.
            :return: Pauli-Z expectation value of ancilla qubit.
            """
            # revert batch dimension to end to ensure correct functionality
            if len(real_imaginary_flag.shape) == 2:
                real_imaginary_flag = torch.permute(real_imaginary_flag, (1, 0))
                encoded_system = batch_dimension_reversal(encoded_system, mode=self.mode)[0]

            # >>> CIRCUIT CONSTRUCTION <<<

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)
            # Sdg gate if evaluating imaginary part
            gates.param_s_dagger(real_imaginary_flag[0], wires=self.ancilla_qubit)

            # apply controlled version of variational circuit
            # noinspection PyTypeChecker
            qml.ctrl(qml.map_wires(self.ansatz, self.data_qubits_map), self.ancilla_qubit)(weights)

            # apply controlled version of encoded system A_m
            controlled_system(encoded_system, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map)

            # apply controlled version of unitary implementing right side
            # noinspection PyTypeChecker
            qml.ctrl(qml.adjoint(qml.map_wires(self.right_side_fn, self.data_qubits_map)), self.ancilla_qubit)()

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)

            # measure the ancilla qubit in Pauli-Z basis
            return qml.expval(qml.PauliZ(wires=self.ancilla_qubit))

        return circuit_hadamard_global

    def generate_batched_encoded_system(self) -> dict:
        """
        Generate encodings of the A_m required for evaluating the gamma_m constituting the global loss.

        Shapes of batched_encoded_system: (with batch_size = L)
            - [batch_size, num_qubits, 3] for `pauli` mode (one-hot encoded)
            - [batch_size, 2^num_qubits, 2^num_qubits] for `unitary` mode (full unitary)
            - [batch_size, 1] for `circuit` mode (circuit function index)

        :return: Dictionary containing encoded system and additional metadata.
        """
        # different data types for the different ways of system encoding
        dtype = self.mode.mode_dependent_value(torch.float32, torch.complex128, torch.int32, None)

        batched_data = {'batched_factors': tensor_from_array(self.coeffs, dtype=torch.complex128),
                        'batched_encoded_system': tensor_from_array(self.encoded_system, dtype=dtype),
                        'batch_size': len(self.system)}

        return batched_data
