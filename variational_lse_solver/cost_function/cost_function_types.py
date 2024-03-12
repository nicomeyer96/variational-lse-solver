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
This file contains several types used in the variational LSE solver implementation.
"""

from enum import Enum, auto


class CostFunctionMode(Enum):
    """ In which mode to run, depends on how the system matrix is provided. """

    PAULI = auto()
    UNITARY = auto()
    CIRCUIT = auto()
    MATRIX = auto()

    def mode_dependent_value(self, pauli, unitary, circuit, matrix):
        match self:
            case CostFunctionMode.PAULI:
                return pauli
            case CostFunctionMode.UNITARY:
                return unitary
            case CostFunctionMode.CIRCUIT:
                return circuit
            case CostFunctionMode.MATRIX:
                return matrix
            case _:
                raise ValueError(f'Unknown mode: {self}')


class CostFunctionMethod(Enum):
    """ Which method to use for evaluating the cost function. """

    DIRECT = auto()
    HADAMARD = auto()
    OVERLAP = auto()
    COHERENT = auto()

    def method_dependent_value(self, direct, hadamard, overlap, coherent):
        match self:
            case CostFunctionMethod.DIRECT:
                return direct
            case CostFunctionMethod.HADAMARD:
                return hadamard
            case CostFunctionMethod.OVERLAP:
                return overlap
            case CostFunctionMethod.COHERENT:
                return coherent
            case _:
                raise ValueError(f'Unknown method: {self}')


class CostFunctionLoss(Enum):
    """ Which loss function to use. """

    GLOBAL = auto()
    LOCAL = auto()

    def loss_dependent_value(self, global_loss, local_loss):
        match self:
            case CostFunctionLoss.GLOBAL:
                return global_loss
            case CostFunctionLoss.LOCAL:
                return local_loss
            case _:
                raise ValueError(f'Unknown loss: {self}')
