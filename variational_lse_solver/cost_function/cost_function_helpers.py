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

"""
This file contains several helper methods for realizing the cost function.
"""

import numpy as np
from typing import Callable
import pennylane as qml
import torch
import re

from .cost_function_types import CostFunctionMode
from .. import gates


# noinspection PyTypeChecker
def controlled_system(encoded_system: np.ndarray, system: list, mode: CostFunctionMode,
                      control_qubit: int, data_qubits_map: dict, adjoint: bool = False):
    """
    Realizes the controlled application of A_m, which are encoded in the following ways:
        - `pauli`-mode: one-hot values [num_qubits, 3] indicating which Pauli-term to apply to each qubit (all zero for `I`).
        - `unitaries`-mode: explicit unitary, i.e. matrix of shape 2^num_qubits x 2^num_qubits
        - `circuit`-mode: circuit index (within list `system`).

    :param encoded_system: Encoded A_m.
    :param system: Full decomposition of system matrix (only used for `circuit` mode).
    :param mode: Format of system decomposition.
    :param control_qubit: Index of control qubits (0 by default).
    :param data_qubits_map: Index of data qubits (1, ..., num_qubits by default).
    :param adjoint: Whether to use the adjoint A_m^t.
    """

    match mode:
        case CostFunctionMode.PAULI:
            assert len(data_qubits_map) == encoded_system.shape[0]
            # controlled version of A_m (as only Paulis we can just control them individually), self-adjoint
            for index, qubit_key in enumerate(data_qubits_map):
                qml.ctrl(gates.param_pauli_x, control_qubit)(-encoded_system[index, 0], wires=data_qubits_map[qubit_key])
                qml.ctrl(gates.param_pauli_y, control_qubit)(-encoded_system[index, 1], wires=data_qubits_map[qubit_key])
                qml.ctrl(gates.param_pauli_z, control_qubit)(-encoded_system[index, 2], wires=data_qubits_map[qubit_key])
        case CostFunctionMode.UNITARY:
            if adjoint:
                qml.ctrl(qml.adjoint(qml.QubitUnitary), control_qubit)(encoded_system, wires=list(data_qubits_map.values()))
            else:
                qml.ctrl(qml.QubitUnitary, control_qubit)(encoded_system, wires=list(data_qubits_map.values()))
        case CostFunctionMode.CIRCUIT:
            if adjoint:
                qml.ctrl(qml.adjoint(qml.map_wires(system[encoded_system[0]], data_qubits_map)), control_qubit)()
            else:
                qml.ctrl(qml.map_wires(system[encoded_system[0]], data_qubits_map), control_qubit)()
        case CostFunctionMode.MATRIX:
            raise RuntimeError('This function should not be called for the MATRIX mode.')
        case _:
            raise ValueError(f'Unknown mode: {mode}')


def generate_encoded_system(system: list[str | np.ndarray | Callable], mode: CostFunctionMode) -> np.ndarray | None:
    """
    For the different modes convert the provided input to a consistent format:
        - `pauli`-mode: list of one-hot values [L, num_qubits, 3] indicating which Pauli-term to apply to each qubit (all zero for `I`).
        - `unitaries`-mode: list of explicit unitary, i.e. L matrices of shape 2^num_qubits x 2^num_qubits.
        - `circuit`-mode: list of circuit indices of length L.

    :param system: Decomposed system matrix.
    :param mode: Format of provided system matrix.
    :return: Mode-dependent encoded system.
    """

    if mode == CostFunctionMode.PAULI:
        return np.array([pauli_string_to_encoded_system(pauli_string) for pauli_string in system])
    if mode == CostFunctionMode.UNITARY:
        return np.array(system)
    if mode == CostFunctionMode.CIRCUIT:
        return np.array([[index] for index in range(len(system))])
    if mode == CostFunctionMode.MATRIX:
        return None
    raise ValueError(f'Unknown mode: {mode}')


def pauli_string_to_encoded_system(pauli_string: str) -> np.ndarray:
    """
    Convert a Pauli operator string to a one-hot encoded representation.
    For a given Pauli string, this function generates a matrix, where each row corresponds to a qubit and each column
    corresponds to a Pauli operator ('X', 'Y', 'Z'). The matrix contains 1.0 at positions that match the Pauli operator
    in the input string, and 0.0 elsewhere.

    :param str pauli_string: String of 'X', 'Y', and 'Z' representing Pauli operators.
    :return: A binary matrix corresponding to the Pauli operators where rows are qubits and columns indicate 'X', 'Y', 'Z'.

    Example:
        pauli_string_to_a_component_m_matrix("XZYZ")
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
    """

    verify_pauli_string(pauli_string)
    num_qubits = len(pauli_string)

    # This is faster than list comprehension, np.vectorize and numpy char array broadcasting.
    indices = np.fromiter(map({'X': 0, 'Y': 1, 'Z': 2, 'I': 3}.get, pauli_string), dtype=np.int_)
    encoded_system = np.zeros((num_qubits, 4), dtype=np.float32)
    encoded_system[np.arange(num_qubits), indices] = 1.0
    return encoded_system[:, :3]


def verify_pauli_string(pauli_string: str):
    """ Check input format for pauli mode. """

    assert isinstance(pauli_string, str), "pauli_string is not a string"
    allowed_pattern = re.compile(r'^[IXYZ]+$')
    if not allowed_pattern.match(pauli_string):
        raise ValueError(f"The pauli string contains characters other than I, X, Y, Z.")


def tensor_from_array(array, dtype=torch.float32, requires_grad: bool = False) -> torch.tensor:
    """ Converts a numpy array to a PyTorch tensor. """

    return torch.tensor(np.array(array), dtype=dtype, requires_grad=requires_grad)


def batch_dimension_reversal(*components, mode: CostFunctionMode):
    """ For PennyLane to allow for batch-processing, in some cases the batch dimension has to be propagated to the end. """

    match mode:
        case CostFunctionMode.PAULI:
            return tuple(torch.permute(component, (*list(range(1, len(component.shape))), 0)) for component in components)
        case CostFunctionMode.UNITARY:
            return components
        case CostFunctionMode.CIRCUIT:
            raise RuntimeError('The `circuit` mode does not support batching.')
        case CostFunctionMode.MATRIX:
            raise RuntimeError('This function should not be called for the MATRIX mode.')
        case _:
            raise ValueError(f'Unknown mode: {mode}')
