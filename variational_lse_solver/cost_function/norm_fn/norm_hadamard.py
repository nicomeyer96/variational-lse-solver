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
from ..cost_function_helpers import tensor_from_array, batch_dimension_reversal, controlled_system
from ..cost_function_types import CostFunctionMode
from ... import gates


class NormHadamard(NormBase):
    """
    This class allows for evaluation of the norm via the Hadamard test.

    Following https://quantum-journal.org/papers/q-2023-11-22-1188/, one needs to compute
    \sum_m=[0,L-1] \sum_n=[0,L-1] c_m c_n^t beta_mn, with beta_mn = <0|V(alpha)^t A_m^t A_n V(alpha)|0>.

    By exploiting symmetries, we can re-formulate this to:
    \sum_m=[0,L-1] |c_m|^2 + 2 * \sum_m=[0,L-1] \sum_n=[m+1,L-1] REAL(c_m c_n^dagger beta_mn).
    As `REAL(...)` denotes the real part of the potentially complex number, the imaginary parts
    of beta_mn are only relevant, if the product of coefficients also has an imaginary part.
    """

    def __init__(
            self,
            system: list[str | np.ndarray | Callable],
            coeffs: list[float | complex],
            ansatz: Callable,
            data_qubits: int,
            mode: CostFunctionMode,
            imaginary: bool
    ):
        """
        Evaluation of norm via Hadamard test.

        :param system: System matrix given as either Pauli strings, unitaries, or circuits.
        :param coeffs: Corresponding coefficients.
        :param ansatz: The variational quantum circuit.
        :param data_qubits: Number of qubits in the VQC.
        :param mode: In which mode to run (i.e. in which form `system` is provided).
        :param imaginary: Whether to evaluate imaginary terms.
        """
        super().__init__(system, coeffs, ansatz, data_qubits, mode, imaginary)
        assert CostFunctionMode.MATRIX != mode
        self.batched_encoded_system = self.generate_batched_encoded_system()
        # qubit mapping, first one is ancilla
        self.ancilla_qubit = 0
        self.data_qubits_map = {qubit: qubit+1 for qubit in range(data_qubits)}

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Calculates the norm for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Norm value (with grad_fn).
        """
        def calculate_beta_batched(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all beta_mn in a batched manner. """
            return self.qnode_hadamard_norm()(
                weights_,
                encoded_system_m=self.batched_encoded_system['batched_encoded_system_m'],
                encoded_system_n=self.batched_encoded_system['batched_encoded_system_n'],
                real_imaginary_flag=tensor_from_array(np.ones(shape=(self.batched_encoded_system['batch_size'], 1)))
                if real_imaginary_flag_
                else tensor_from_array(np.zeros(shape=(self.batched_encoded_system['batch_size'], 1)))
            )

        def calculate_beta_individual(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all beta_mn in a sequential manner. """
            batched_encoded_system_m = self.batched_encoded_system['batched_encoded_system_m']
            batched_encoded_system_n = self.batched_encoded_system['batched_encoded_system_n']
            return torch.stack([
                self.qnode_hadamard_norm()(
                    weights_,
                    encoded_system_m=encoded_system_m,
                    encoded_system_n=encoded_system_n,
                    real_imaginary_flag=tensor_from_array([1.0]) if real_imaginary_flag_ else tensor_from_array([0.0])
                )
                for encoded_system_m, encoded_system_n
                in zip(batched_encoded_system_m, batched_encoded_system_n)
            ])

        def calculate_beta(weights_: torch.tensor) -> torch.tensor:
            """ Evaluate real (and optionally imaginary) norm term. """
            calculate_beta_fn = self.mode.mode_dependent_value(pauli=calculate_beta_batched, unitary=calculate_beta_batched,
                                                               circuit=calculate_beta_individual, matrix=None)
            beta_mn_ = calculate_beta_fn(weights_, real_imaginary_flag_=False)
            if self.imaginary:
                beta_mn_ = beta_mn_.type(torch.complex128)
                beta_mn_ += 1.j * calculate_beta_fn(weights_, real_imaginary_flag_=True)
            return beta_mn_

        # compute beta_mn`s, catch special case of decomposition length 1
        beta_mn = torch.tensor([1.0]) if 1 == len(self.coeffs) else calculate_beta(weights)
        if not 1 == len(self.coeffs):
            assert beta_mn.shape[0] == (len(self.coeffs) ** 2 - len(self.coeffs)) // 2

        # equation (14) of https://quantum-journal.org/papers/q-2023-11-22-1188/
        # Note: using symmetries: beta_mn = conjugate(beta_nm) => c_m c_n^t beta_mn + c_m^t c_n beta_nm = 2 Re(c_m c_n^t beta_mn)
        #       and using the fact that c_m c_m^t beta_mm = |c_m|^2
        norm = torch.abs(torch.add(torch.sum(torch.real(torch.mul(beta_mn, self.batched_encoded_system['batched_factors']))),
                                   self.batched_encoded_system['diagonal_value']))
        return norm

    def qnode_hadamard_norm(self) -> Callable:
        """
        Quantum node that realizes the Hadamard test for evaluating the beta_mn constituting the norm.

        :return: Circuit handle implementing the Hadamard test.
        """
        dev = qml.device('default.qubit', wires=self.data_qubits + 1)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_hadamard_norm(weights, encoded_system_m, encoded_system_n, real_imaginary_flag):
            """
            Circuit that realizes the Hadamard test for evaluating the beta_mn constituting the norm.

            :param weights: Parameters for the VQC.
            :param encoded_system_m: Encoded representations of A_m.
            :param encoded_system_n: Encoded representations of A_n.
            :param real_imaginary_flag: Whether to evaluate real or imaginary part.
            :return: Pauli-Z expectation value of ancilla qubit.
            """
            # revert batch dimension to end to ensure correct functionality
            if len(real_imaginary_flag.shape) == 2:
                real_imaginary_flag = torch.permute(real_imaginary_flag, (1, 0))
                encoded_system_m, encoded_system_n = batch_dimension_reversal(encoded_system_m,
                                                                              encoded_system_n, mode=self.mode)

            # >>> CIRCUIT CONSTRUCTION <<<

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)
            # Sdg gate if evaluating imaginary part
            gates.param_s_dagger(real_imaginary_flag[0], wires=self.ancilla_qubit)

            # apply variational circuit
            qml.map_wires(self.ansatz, self.data_qubits_map)(weights)

            # apply controlled version of encoded systems A_m
            controlled_system(encoded_system_m, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map)
            # apply controlled version of encoded system A_n^t
            controlled_system(encoded_system_n, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map, adjoint=True)

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)

            # measure the ancilla qubit in the Pauli-Z basis
            return qml.expval(qml.PauliZ(wires=self.ancilla_qubit))

        return circuit_hadamard_norm

    def generate_batched_encoded_system(self) -> dict:
        """
        Generate encodings of the A_m and A_n required for evaluating the beta_mn constituting the norm.

        Shapes of batched_encoded_system_m/n: (with batch_size = (L^2 - L) / 2)
            - [batch_size, num_qubits, 3] for `pauli` mode (one-hot encoded)
            - [batch_size, 2^num_qubits, 2^num_qubits] for `unitary` mode (full unitary)
            - [batch_size, 1] for `circuit` mode (circuit function index)

        :return: Dictionary containing encoded system and additional metadata.
        """

        # only consider upper block, lower block is symmetric beta_mn = conj(beta_nm), diagonal doesn't require circuit evaluations.
        # Set k = 1 to ignore the main diagonal, start 1 diagonal above.
        m_indices, n_indices = np.triu_indices(len(self.coeffs), k=1)

        # Compute 2 * c_m c_n^t (factor 2 arises from symmetry considerations above)
        batched_factors = 2 * np.array(self.coeffs)[m_indices] * np.conjugate(np.array(self.coeffs)[n_indices])

        # Select A_m's and A_n's
        batched_encoded_system_m = self.encoded_system[m_indices]
        batched_encoded_system_n = self.encoded_system[n_indices]

        # pre-factors |c_m|^2 of beta_mm (no circuit evaluations necessary, as beta_mm is always 1.0)
        diagonal_value = np.sum(np.square(np.abs(self.coeffs)))

        # different data types for the different ways of system encoding
        dtype = self.mode.mode_dependent_value(torch.float32, torch.complex128, torch.int32, None)

        batched_data = {'batched_factors': tensor_from_array(batched_factors, dtype=torch.complex128),
                        'batched_encoded_system_m': tensor_from_array(batched_encoded_system_m, dtype=dtype),
                        'batched_encoded_system_n': tensor_from_array(batched_encoded_system_n, dtype=dtype),
                        'diagonal_value': tensor_from_array(diagonal_value),
                        'batch_size': len(m_indices)}
        return batched_data
