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

from . import LocalBase
from ..cost_function_types import CostFunctionMode
from ..cost_function_helpers import tensor_from_array, batch_dimension_reversal, controlled_system
from ... import gates


class LocalHadamard(LocalBase):
    """
    This class allows for evaluation of the local loss via the Hadamard test.

    One needs to compute \sum_m=[0,L-1] \sum_n=[0,L-1] 1/num_qubits \sum_j=[0,num_qubits-1] c_m c_n^dagger delta_mn^(j),
    with delta_mn as defined in https://quantum-journal.org/papers/q-2023-11-22-1188/.

    By exploiting some symmetries, we can re-formulate this to:
    1/num_qubits ( \sum_m=[0,L-1] \sum_j=[0,num_qubits] |c_m|^2 REAL(delta_mm^(j))
                    + 2 * \sum_m=[0,L-1] \sum_n=[m+1,L-1] \sum_j=[0,num_qubits-1] REAL(c_m c_n^dagger delta_mn^(j))).
    As `REAL(...)` denotes the real part of the potentially complex number, the imaginary parts
    of delta_mn are only relevant, if the product of coefficients also has an imaginary part.
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
        Evaluation of local loss via Hadamard test.

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
        Calculates the local loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Local loss value (with grad_fn).
        """

        def calculate_delta_batched(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all delta_mn in a batched manner. """
            return self.qnode_hadamard_local()(
                weights_,
                encoded_system_m=self.batched_encoded_system['batched_encoded_system_m'],
                encoded_system_n=self.batched_encoded_system['batched_encoded_system_n'],
                real_imaginary_flag=tensor_from_array(np.ones(shape=(self.batched_encoded_system['batch_size'], 1)))
                if real_imaginary_flag_
                else tensor_from_array(np.zeros(shape=(self.batched_encoded_system['batch_size'], 1))),
                local_flag=self.batched_encoded_system['batched_local_flag']
            )

        def calculate_delta_individual(weights_: torch.tensor, real_imaginary_flag_: bool) -> torch.tensor:
            """ Evaluate all delta_mn in a sequential manner. """
            batched_encoded_system_m = self.batched_encoded_system['batched_encoded_system_m']
            batched_encoded_system_n = self.batched_encoded_system['batched_encoded_system_n']
            batched_local_flag = self.batched_encoded_system['batched_local_flag']
            return torch.stack([
                self.qnode_hadamard_local()(
                    weights_,
                    encoded_system_m=encoded_system_m,
                    encoded_system_n=encoded_system_n,
                    real_imaginary_flag=tensor_from_array([1.0]) if real_imaginary_flag_ else tensor_from_array([0.0]),
                    local_flag=local_flag
                )
                for encoded_system_m, encoded_system_n, local_flag
                in zip(batched_encoded_system_m, batched_encoded_system_n, batched_local_flag)
            ])

        def calculate_delta(weights_: torch.tensor) -> torch.tensor:
            """ Evaluate real (and optionally imaginary) local loss term. """
            calculate_delta_fn = self.mode.mode_dependent_value(pauli=calculate_delta_batched, unitary=calculate_delta_batched,
                                                                circuit=calculate_delta_individual, matrix=None)
            delta_mn_ = calculate_delta_fn(weights_, real_imaginary_flag_=False)
            if self.imaginary:
                delta_mn_ = delta_mn_.type(torch.complex128)
                delta_mn_ += 1.j * calculate_delta_fn(weights_, real_imaginary_flag_=True)
            return delta_mn_

        # compute delta_mn's
        delta_mn = calculate_delta(weights)
        assert delta_mn.shape[0] == self.data_qubits * ((len(self.coeffs) ** 2 + len(self.coeffs)) // 2)

        # equation (18) of https://quantum-journal.org/papers/q-2023-11-22-1188/
        # Note: using symmetries: delta_mn^(j) = conjugate(delta_nm^(j)) => c_m c_n^t delta_mn^(j) + c_m^t c_n delta_nm^(j) = 2 Re(c_m c_n^t delta_mn^(j));
        # IMPORTANT: One must avoid to call `abs(...)` on loss_raw, as done in https://pennylane.ai/qml/demos/tutorial_vqls/.
        #            This potentially introduces unwanted symmetries to the loss landscape and leads to faulty convergence.
        loss_raw = torch.div(torch.sum(torch.real(torch.mul(delta_mn, self.batched_encoded_system['batched_factors']))), self.data_qubits)
        return loss_raw

    def qnode_hadamard_local(self) -> Callable:
        """
        Quantum node that realizes the Hadamard test for evaluating the delta_mn constituting the local loss.

        :return: Circuit handle implementing the Hadamard test.
        """
        dev = qml.device('default.qubit', wires=self.data_qubits + 1)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit_hadamard_local(weights, encoded_system_m, encoded_system_n, real_imaginary_flag, local_flag):
            """
            Circuit that realizes the Hadamard test for evaluating the delta_mn constituting the local loss term.

            :param weights: Parameters for the VQC.
            :param encoded_system_m: Encoded representations of A_m.
            :param encoded_system_n: Encoded representations of A_n.
            :param real_imaginary_flag: Whether to evaluate real or imaginary part.
            :param local_flag: Which qubit to target for local evalaution.
            :return: Pauli-Z expectation value of ancilla qubit.
            """
            # revert batch dimension to end to ensure correct functionality
            if len(real_imaginary_flag.shape) == 2:
                real_imaginary_flag = torch.permute(real_imaginary_flag, (1, 0))
                local_flag = torch.permute(local_flag, (1, 0))
                encoded_system_m, encoded_system_n = batch_dimension_reversal(encoded_system_m, encoded_system_n, mode=self.mode)

            # >>> CIRCUIT CONSTRUCTION <<<

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)
            # Sdg gate if evaluating imaginary part
            gates.param_s_dagger(real_imaginary_flag[0], wires=self.ancilla_qubit)

            # apply variational circuit
            qml.map_wires(self.ansatz, self.data_qubits_map)(weights)

            # apply controlled version of encoded system A_m
            controlled_system(encoded_system_m, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map)

            # apply controlled version of unitary implementing adjoint right side
            # noinspection PyTypeChecker
            qml.ctrl(qml.adjoint(qml.map_wires(self.right_side_fn, self.data_qubits_map)), self.ancilla_qubit)()

            # apply an ancilla-controlled Z-gate to the qubit indicated by `local_flag` (i.e. interactively to each qubit)
            for index, qubit_key in enumerate(self.data_qubits_map):
                # noinspection PyTypeChecker
                qml.ctrl(gates.param_pauli_z, self.ancilla_qubit)(local_flag[index], wires=self.data_qubits_map[qubit_key])

            # apply controlled version of unitary implementing right side
            # noinspection PyTypeChecker
            qml.ctrl(qml.map_wires(self.right_side_fn, self.data_qubits_map), self.ancilla_qubit)()

            # apply controlled version of encoded system A_n^t
            controlled_system(encoded_system_n, self.system, self.mode, self.ancilla_qubit, self.data_qubits_map, adjoint=True)

            # hadamard gate on ancilla
            qml.Hadamard(wires=self.ancilla_qubit)

            # measure the ancilla qubit in the Pauli-Z basis
            return qml.expval(qml.PauliZ(wires=self.ancilla_qubit))

        return circuit_hadamard_local

    def generate_batched_encoded_system(self) -> dict:
        """
        Generate encodings of the A_m and A_n required for evaluating the delta_mn constituting the local loss.

        Shapes of batched_encoded_system_m/n: (with batch_size = num_qubits * (L^2 + L) / 2)
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

        # Repeat for num_qubits to allow local evaluation
        batched_factors = np.repeat(batched_factors, self.data_qubits, axis=0)
        batched_encoded_system_m = np.repeat(batched_encoded_system_m, self.data_qubits, axis=0)
        batched_encoded_system_n = np.repeat(batched_encoded_system_n, self.data_qubits, axis=0)
        batched_local_flag = np.zeros((len(m_indices) * self.data_qubits, self.data_qubits))
        batched_local_indices = np.tile(np.arange(self.data_qubits), len(m_indices))
        batched_local_flag[np.arange(len(m_indices) * self.data_qubits),batched_local_indices] = 1.0

        # different data types for the different ways of system encoding
        dtype = self.mode.mode_dependent_value(torch.float32, torch.complex128, torch.int32, None)

        batched_data = {'batched_factors': tensor_from_array(batched_factors, dtype=torch.complex128),
                        'batched_encoded_system_m': tensor_from_array(batched_encoded_system_m, dtype=dtype),
                        'batched_encoded_system_n': tensor_from_array(batched_encoded_system_n, dtype=dtype),
                        'batched_local_flag': tensor_from_array(batched_local_flag, dtype=torch.int32),
                        'batch_size': len(m_indices) * self.data_qubits}

        return batched_data
