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
This file contains several methods to set up and initialize the variational lse solver.
"""

import numpy as np
import warnings
from typing import Callable

from .cost_function.cost_function_types import CostFunctionMode, CostFunctionMethod, CostFunctionLoss


def mode_init(a_system: np.ndarray | list[np.ndarray] | list[str] | list[Callable]):
    """ Figure out in which format the input is provided. """
    if isinstance(a_system, np.ndarray): return CostFunctionMode.MATRIX
    if isinstance(a_system, list):
        if isinstance(a_system[0], str): return CostFunctionMode.PAULI
        if isinstance(a_system[0], np.ndarray): return CostFunctionMode.UNITARY
        if callable(a_system[0]): return CostFunctionMode.CIRCUIT
    raise ValueError(f'The supported modes are PAULI (with list[str] as a),'
                     f' UNITARY (with list[np.ndarray] as a),'
                     f' CIRCUIT (with list[Callable] as a), and'
                     f' MATRIX (with np.ndarray as a).')


def mode_pauli_validation(a_system: list[str], a_coeffs: list[float | complex], b_vector: np.ndarray | Callable,
                          data_qubits: int) -> int:
    """ Extract and validate number of data qubits for pauli mode. """
    expected_data_qubits = len(a_system[0])
    if not 0 == data_qubits: assert data_qubits == expected_data_qubits
    assert all(len(pauli_string) == expected_data_qubits for pauli_string in a_system)
    decomposition_validation(a_system, a_coeffs, b_vector, expected_data_qubits)
    return expected_data_qubits


def mode_unitary_validation(a_system: list[np.ndarray], a_coeffs: list[float | complex], b_vector: np.ndarray | Callable,
                            data_qubits: int) -> int:
    """ Extract and validate number of data qubits for unitary mode. """
    expected_data_qubits = int(np.log2(a_system[0].shape[0]))
    if not 0 == data_qubits: assert data_qubits == expected_data_qubits
    assert all(2 == len(unitaries.shape) for unitaries in a_system)
    assert all(unitaries.shape[0] == 2 ** expected_data_qubits for unitaries in a_system)
    assert all(unitaries.shape[1] == 2 ** expected_data_qubits for unitaries in a_system)
    decomposition_validation(a_system, a_coeffs, b_vector, expected_data_qubits)
    return expected_data_qubits


def mode_circuit_validation(a_system: list[np.ndarray], a_coeffs: list[float | complex], b_vector: np.ndarray | Callable,
                            data_qubits: int):
    """ Extract and validate number of data qubits for circuit mode. """
    if 0 == data_qubits:
        raise ValueError('Using CIRCUIT mode (with list[Callable] as a) requires providing data_qubits argument.')
    expected_data_qubits = data_qubits
    decomposition_validation(a_system, a_coeffs, b_vector, expected_data_qubits)
    return expected_data_qubits


def mode_matrix_validation(a_system: np.ndarray,  a_coeffs: list[float | complex], b_vector: np.ndarray | Callable,
                           data_qubits: int) -> int:
    """ Extract and validate number of data qubits for matrix mode. """
    expected_data_qubits = int(np.log2(a_system.shape[0]))
    if not 0 == data_qubits: assert data_qubits == expected_data_qubits
    assert 2 == len(a_system.shape)
    assert a_system.shape[0] == 2 ** expected_data_qubits and a_system.shape[1] == 2 ** expected_data_qubits
    if a_coeffs is not None:
        warnings.warn('Running in MATRIX mode (with np.ndarray as a), `coeffs` argument will be ignored).')
    if callable(b_vector):
        raise ValueError('Running in MATRIX mode (with np.ndarray as a) requires defining `b` argument as np.ndarray')
    assert 1 == len(b_vector.shape)
    assert expected_data_qubits == int(np.log2(b_vector.shape[0]))
    return expected_data_qubits


def decomposition_validation(a_system: list, a_coeffs: list, b_vector: np.ndarray | Callable, data_qubits: int):
    """ Test LSE system decomposition for compatibility. """
    assert len(a_system) == len(a_coeffs)
    if not callable(b_vector):
        assert 1 == len(b_vector.shape)
        assert data_qubits == int(np.log2(b_vector.shape[0]))


def method_init(method: str):
    """ Initialize loss computation method. """
    if 'direct' == method: return CostFunctionMethod.DIRECT
    if 'hadamard' == method: return CostFunctionMethod.HADAMARD
    if 'overlap' == method: return CostFunctionMethod.OVERLAP
    if 'coherent' == method: return CostFunctionMethod.COHERENT
    raise ValueError(f'The supported methods are DIRECT (with `direct` as method),'
                     f' HADAMARD (with `hadamard` as method),'
                     f' OVERLAP (with `overlap` as method), and'
                     f' COHERENT (with `coherent` as method).')


def method_direct_validate(mode: CostFunctionMode, loss: CostFunctionLoss, b_vector: np.ndarray | Callable):
    """ Validate setup for direct loss computation method. """
    if mode is not CostFunctionMode.MATRIX:
        raise ValueError('For DIRECT method `a` has to be provided as np.ndarray')
    if not isinstance(b_vector, np.ndarray):
        raise ValueError('For DIRECT method `b` has to be provided as np.ndarray')


def method_hadamard_validate(mode: CostFunctionMode, loss: CostFunctionLoss, b_vector: np.ndarray | Callable):
    """ Validate setup for hadamard loss computation method. """
    if mode is CostFunctionMode.MATRIX:
        raise ValueError('For HADAMARD method `a` has to be provided as list[str], list[np.ndarray], or list[Callable].')


def method_overlap_validate(mode: CostFunctionMode, loss: CostFunctionLoss, b_vector: np.ndarray | Callable):
    """ Validate setup for hadamard-overlap loss computation method. """
    if mode is CostFunctionMode.MATRIX:
        raise ValueError('For OVERLAP method `a` has to be provided as list[str], list[np.ndarray], or list[Callable].')


def method_coherent_validate(mode: CostFunctionMode, loss: CostFunctionLoss, b_vector: np.ndarray | Callable):
    """ Validate setup for coherent loss computation method. """
    if mode is CostFunctionMode.MATRIX:
        raise ValueError('For COHERENT method `a` has to be provided as list[str], list[np.ndarray], or list[Callable].')
    if loss is CostFunctionLoss.LOCAL:
        raise ValueError('The COHERENT method is only available for the global loss (`local` set to False).')


def init_imaginary_flag(coeffs: list[float | complex]) -> bool:
    """ Check if all pre-factors are real-values, which removes the necessity to compute imaginary loss terms. """
    return any([np.iscomplexobj(coeff) for coeff in coeffs])


def init_imaginary_flag_dummy(coeffs: list[float | complex]) -> bool:
    """ Dummy method for direct loss computation method. """
    return True
