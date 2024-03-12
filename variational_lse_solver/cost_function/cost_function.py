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
This file contains a class handling the composition of multiple loss terms to the actual trainable loss.
"""

import numpy as np
import torch
from typing import Callable

from .cost_function_types import CostFunctionMode, CostFunctionMethod, CostFunctionLoss

from .norm_fn import NormDirect, NormHadamard
from .global_fn import GlobalDirect, GlobalHadamard, GlobalOverlap, GlobalCoherent
from .local_fn import LocalDirect, LocalHadamard, LocalOverlap


class CostFunction:
    """
    This class brings the different parts of the loss function together.
    """

    def __init__(
            self,
            system: np.ndarray | list[str | np.ndarray | Callable],
            coeffs: list[float | complex] | None,
            right_side: np.ndarray | Callable,
            ansatz: Callable,
            mode: CostFunctionMode,
            method: CostFunctionMethod,
            loss: CostFunctionLoss,
            data_qubits: int,
            imaginary: bool
    ):
        """
        Evalaution of normalized loss.

        :param system: System matrix given as either Pauli strings, unitaries, or circuits, or full matrix.
        :param coeffs: Corresponding coefficients.
        :param right_side: Right side of the LSE.
        :param ansatz: The variational quantum circuit.
        :param mode: In which mode to run (i.e. in which form `system` is provided).
        :param method: Which method to use for evalaution of the loss term.
        :param data_qubits: Number of qubits in the VQC.
        :param imaginary: Whether to evaluate imaginary terms.
        """

        self.loss = loss
        self.method = method

        # Set up the norm computation method
        # Note: The norm is computed with `hadamard` also for `overlap` method
        self.norm_fn = method.method_dependent_value(direct=self.init_norm_direct,
                                                     hadamard=self.init_norm_hadamard,
                                                     overlap=self.init_norm_hadamard,
                                                     coherent=self.init_norm_coherent)(system, coeffs, ansatz, data_qubits, mode, imaginary)

        # Set up the raw loss computation method
        match loss:
            case CostFunctionLoss.GLOBAL:
                self.loss_fn = (method.method_dependent_value(direct=self.init_loss_direct_global,
                                                              hadamard=self.init_loss_hadamard_global,
                                                              overlap=self.init_loss_overlap_global,
                                                              coherent=self.init_loss_coherent_global)
                                (system, coeffs, right_side, ansatz, data_qubits, mode, imaginary))
            case CostFunctionLoss.LOCAL:
                self.loss_fn = (method.method_dependent_value(direct=self.init_loss_direct_local,
                                                              hadamard=self.init_loss_hadamard_local,
                                                              overlap=self.init_loss_overlap_local,
                                                              coherent=self.init_loss_coherent_local)
                                (system, coeffs, right_side, ansatz, data_qubits, mode, imaginary))
            case _:
                raise ValueError(f'Loss {loss} unknown.')

    def cost(self, weights: torch.tensor) -> torch.tensor:
        """
        Calculates the normalized loss for given VQC parameters.

        :param weights: Weights for the VQC ansatz.
        :return: Normalized loss value (with grad_fn).
        """
        if CostFunctionMethod.COHERENT != self.method:  # The coherent method inherently computes the norm
            norm = self.norm_fn.cost(weights)
        loss_raw = self.loss_fn.cost(weights)
        if CostFunctionMethod.COHERENT == self.method:  # The coherent method inherently normalizes the loss
            return loss_raw
        # noinspection PyUnboundLocalVariable
        loss = torch.abs(torch.sub(1.0, torch.div(loss_raw, norm)))
        # if using local cost function, divide by 1/2 to be consistent with `direct` definition
        if CostFunctionLoss.LOCAL == self.loss and CostFunctionMethod.DIRECT != self.method:
            loss = torch.div(loss, 2.0)
        return loss

    @staticmethod
    def init_norm_direct(system: np.ndarray, coeffs: None, ansatz: Callable,
                         data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up direct computation of norm. """
        assert mode == CostFunctionMode.MATRIX
        return NormDirect(system, coeffs, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_norm_hadamard(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                           ansatz: Callable, data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up hadamard computation of norm. """
        assert mode != CostFunctionMode.MATRIX
        return NormHadamard(system, coeffs, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_norm_coherent(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                           ansatz: Callable, data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Dummy for coherent computation of norm. """
        assert mode != CostFunctionMode.MATRIX
        # The norm is not necessary for the `coherent` formulation, as it is inherently computed.
        return None

    @staticmethod
    def init_loss_direct_global(system: np.ndarray, coeffs: None,
                                right_side: np.ndarray, ansatz: Callable,
                                data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up direct computation of global loss. """
        assert mode == CostFunctionMode.MATRIX
        return GlobalDirect(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_hadamard_global(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                  right_side: np.ndarray | Callable, ansatz: Callable,
                                  data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up hadamard computation of global loss. """
        assert mode != CostFunctionMode.MATRIX
        return GlobalHadamard(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_overlap_global(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                 right_side: np.ndarray | Callable, ansatz: Callable,
                                 data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up hadamard-overlap computation of global loss. """
        assert mode != CostFunctionMode.MATRIX
        return GlobalOverlap(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_coherent_global(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                  right_side: np.ndarray | Callable, ansatz: Callable,
                                  data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up coherent computation of global loss. """
        assert mode != CostFunctionMode.MATRIX
        return GlobalCoherent(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_direct_local(system: np.ndarray, coeffs: None,
                               right_side: np.ndarray, ansatz: Callable,
                               data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up direct computation of local loss. """
        return LocalDirect(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_hadamard_local(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                 right_side: np.ndarray | Callable, ansatz: Callable,
                                 data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up hadamard computation of local loss. """
        return LocalHadamard(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_overlap_local(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                right_side: np.ndarray | Callable, ansatz: Callable,
                                data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Set up hadamard-overlap computation of local loss. """
        return LocalOverlap(system, coeffs, right_side, ansatz, data_qubits, mode, imaginary)

    @staticmethod
    def init_loss_coherent_local(system: list[str | np.ndarray | Callable], coeffs: list[float | complex],
                                 right_side: np.ndarray | Callable, ansatz: Callable,
                                 data_qubits: int, mode: CostFunctionMode, imaginary: bool):
        """ Dummy for coherent computation of local loss. """
        raise ValueError('The `coherent` method is only available for the `global` cost function.')
