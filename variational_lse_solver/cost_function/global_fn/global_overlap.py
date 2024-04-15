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
import itertools
import torch
import pennylane as qml
from typing import Callable
from typing_extensions import override

from . import GlobalBase
from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import tensor_from_array, batch_dimension_reversal, controlled_system


class GlobalOverlap(GlobalBase):
    """
    This class allows for evaluation of the global loss term via the Hadamard-overlap test.

    One needs to compute \sum_m=[0,L-1] \sum_n=[0,L-1] c_m c_n^dagger gamma_mn,
    with gamma_mn as defined in https://quantum-journal.org/papers/q-2023-11-22-1188/.

    By exploiting some symmetries, we can re-formulate this to:
    \sum_m=[0,L-1] |c_m|^2 REAL(gamma_mm) + 2 * \sum_m=[0,L-1] \sum_n=[m+1,L-1] REAL(c_m c_n^dagger gamma_mn).
    As `REAL(...)` denotes the real part of the potentially complex number, the imaginary parts
    of gamma_mn are only relevant, if the product of coefficients also has an imaginary part.
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
        self.batched_encoded_system = self.generate_batched_encoded_system()
        # qubit mapping, first one is ancilla
        self.ancilla_qubit = 0
        self.data_qubits_map_upper = {qubit: qubit + 1 for qubit in range(data_qubits)}
        self.data_qubits_map_lower = {qubit: data_qubits + qubit + 1 for qubit in range(data_qubits)}
        # bitstring indices for which parity of bitwise AND between bits of upper and lower register is even / odd
        bitstrings = np.array([''.join(bs) for bs in list(itertools.product('01', repeat=2 * data_qubits))])
        bitstrings = np.array([bin(int(bitstring[:self.data_qubits], 2) &
                                   int(bitstring[self.data_qubits:], 2))[2:].zfill(self.data_qubits) for bitstring in bitstrings])
        self.indices_even_parity = np.array([index for index, bitstring in enumerate(bitstrings) if 0 == bitstring.count('1') % 2])
        self.indices_odd_parity = np.array([index for index, bitstring in enumerate(bitstrings) if 0 != bitstring.count('1') % 2])

    @override
    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Calculates the global loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Local loss value (with grad_fn).
        """

        def calculate_gamma_batched(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all gamma_mn in a batched manner. """
            return self.qnode_overlap_global()(
                weights_,
                encoded_system_m=self.batched_encoded_system['batched_encoded_system_m'],
                encoded_system_n=self.batched_encoded_system['batched_encoded_system_n'],
                real_imaginary_flag=tensor_from_array(np.ones(shape=(self.batched_encoded_system['batch_size'], 1)))
                if real_imaginary_flag_
                else tensor_from_array(np.zeros(shape=(self.batched_encoded_system['batch_size'], 1)))
            )

        def calculate_gamma_individual(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all gamma_mn in a sequential manner. """
            batched_encoded_system_m = self.batched_encoded_system['batched_encoded_system_m']
            batched_encoded_system_n = self.batched_encoded_system['batched_encoded_system_n']
            return torch.stack([
                self.qnode_overlap_global()(
                    weights_,
                    encoded_system_m=encoded_system_m,
                    encoded_system_n=encoded_system_n,
                    real_imaginary_flag=tensor_from_array([1.0]) if real_imaginary_flag_ else tensor_from_array([0.0]),
                )
                for encoded_system_m, encoded_system_n
                in zip(batched_encoded_system_m, batched_encoded_system_n)
            ])

        def calculate_gamma(weights_: torch.tensor) -> torch.tensor:
            """ Evaluate real (and optionally imaginary) global loss term. """
            calculate_gamma_fn = self.mode.mode_dependent_value(pauli=calculate_gamma_batched,
                                                                unitary=calculate_gamma_batched,
                                                                circuit=calculate_gamma_individual, matrix=None)
            gamma_mn_ = calculate_gamma_fn(weights_, real_imaginary_flag_=False)
            if self.imaginary:
                gamma_mn_ = gamma_mn_.type(torch.complex128)
                gamma_mn_ += 1.j * calculate_gamma_fn(weights_, real_imaginary_flag_=True)
            return gamma_mn_

        # compute gamma_mn's
        gamma_mn = self.postprocess_bitstrings(calculate_gamma(weights))
        assert gamma_mn.shape[0] == (len(self.coeffs) ** 2 + len(self.coeffs)) // 2

        # equation (16) of https://quantum-journal.org/papers/q-2023-11-22-1188/
        # Note: using symmetries: gamma_mn = conjugate(gamma_nm) => c_m c_n^t gamma_mn + c_m^t c_n gamma_nm = 2 Re(c_m c_n^t gamma_mn);
        loss_raw = torch.sum(torch.real(torch.mul(gamma_mn, self.batched_encoded_system['batched_factors'])))
        return loss_raw

    def qnode_overlap_global(self) -> Callable:
        """
        Quantum node that realizes the Hadamard-overlap test for evaluating gamma_mn constituting the global loss.

        :return: Circuit handle implementing the Hadamard-overlap test.
        """
        dev = qml.device('default.qubit', wires=2 * self.data_qubits + 1)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_overlap_global(weights, encoded_system_m, encoded_system_n, real_imaginary_flag):
            """
            Circuit that realizes the Hadamard-overlap test for evaluating the gamma_m constituting the global loss term.

            :param weights: Parameters for the VQC.
            :param encoded_system_m: Encoded representations of A_m.
            :param encoded_system_n: Encoded representations of A_n.
            :param real_imaginary_flag: Whether to evaluate real or imaginary part.
            :return: Bistring measurement probabilities of entire system.
            """
            # revert batch dimension to end to ensure correct functionality
            if len(real_imaginary_flag.shape) == 2:
                real_imaginary_flag = torch.permute(real_imaginary_flag, (1, 0))
                encoded_system_m, encoded_system_n = batch_dimension_reversal(encoded_system_m, encoded_system_n,
                                                                              mode=self.mode)

            # >>> CIRCUIT CONSTRUCTION <<<

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)

            # apply variational circuit to upper data qubit register
            qml.map_wires(self.ansatz, self.data_qubits_map_upper)(weights)
            # apply unitary implementing right side to lower data qubit register
            qml.map_wires(self.right_side_fn, self.data_qubits_map_lower)()

            # apply controlled version of encoded system A_m to upper data qubit register
            controlled_system(encoded_system_m, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map_upper)
            # apply controlled version of encoded system A_n^t to lower data qubit register
            controlled_system(encoded_system_n, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map_lower, adjoint=True)

            # R_z(-pi/2) gate if evaluating imaginary part
            qml.RZ((-np.pi / 2) * real_imaginary_flag[0], wires=self.ancilla_qubit)
            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)

            # controlled-NOT gates between both data qubit registers
            for q in range(self.data_qubits):
                qml.CNOT(wires=(self.data_qubits_map_upper[q], self.data_qubits_map_lower[q]))
            # hadamard gates on upper data qubit register
            for q in range(self.data_qubits):
                qml.Hadamard(wires=self.data_qubits_map_upper[q])

            # return bitstring measurement probabilities
            return qml.probs()

        return circuit_overlap_global

    def postprocess_bitstrings(self, gamma_mn: torch.tensor) -> torch.tensor:
        """
        Post-process the bistring measurements received by Hadamard-overlap test to estimates of gamma_mn.

        This basically is a conditioned destructive SWAP test,
        as described in e.g. https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.052330

        :param gamma_mn: Tensor of bitstring measurement probabilities (of length (L^2 + L) / 2)
        :return: Values of gamma_mn (of length (L^2 + L) / 2)
        """

        p_0 = torch.subtract(torch.sum(torch.gather(gamma_mn, dim=1,
                                                    index=torch.tensor(self.indices_even_parity)
                                                    .repeat(gamma_mn.shape[0], 1)), dim=1),
                             torch.sum(torch.gather(gamma_mn, dim=1,
                                                    index=torch.tensor(self.indices_odd_parity)
                                                    .repeat(gamma_mn.shape[0], 1)), dim=1))

        p_1 = torch.subtract(torch.sum(torch.gather(gamma_mn, dim=1,
                                                    index=torch.tensor(self.indices_even_parity + 2 ** (2 * self.data_qubits))
                                                    .repeat(gamma_mn.shape[0], 1)), dim=1),
                             torch.sum(torch.gather(gamma_mn, dim=1,
                                                    index=torch.tensor(self.indices_odd_parity + 2 ** (2 * self.data_qubits))
                                                    .repeat(gamma_mn.shape[0], 1)), dim=1))

        return torch.subtract(p_0, p_1)

    def generate_batched_encoded_system(self) -> dict:
        """
        Generate encodings of the A_m and A_n required for evaluating the gamma_mn constituting the global loss.

        Shapes of batched_encoded_system_m/n: (with batch_size = (L^2 + L) / 2)
            - [batch_size, num_qubits, 3] for `pauli` mode (one-hot encoded)
            - [batch_size, 2^num_qubits, 2^num_qubits] for `unitary` mode (full unitary)
            - [batch_size, 1] for `circuit` mode (circuit function index)

        :return: Dictionary containing encoded system and additional metadata.
        """
        # Calculate the upper triangle indices (including the diagonals)
        # only consider upper block, lower block is symmetric delta_mn = conj(delta_nm);
        # diagonal is included once (see below);
        m_indices, n_indices = np.triu_indices(len(self.coeffs), k=0)

        # Compute c_m c_n^t
        batched_factors = np.array(self.coeffs)[m_indices] * np.conjugate(np.array(self.coeffs)[n_indices])
        # Multiply with two for m!=n (see symmetry considerations above)
        batched_factors[np.where(m_indices != n_indices)] *= 2

        # Select A_m's and A_n's
        batched_encoded_system_m = self.encoded_system[m_indices]
        batched_encoded_system_n = self.encoded_system[n_indices]

        # different data types for the different ways of system encoding
        dtype = self.mode.mode_dependent_value(torch.float32, torch.complex128, torch.int32, None)

        batched_data = {'batched_factors': tensor_from_array(batched_factors, dtype=torch.complex128),
                        'batched_encoded_system_m': tensor_from_array(batched_encoded_system_m, dtype=dtype),
                        'batched_encoded_system_n': tensor_from_array(batched_encoded_system_n, dtype=dtype),
                        'batch_size': len(m_indices)}

        return batched_data
