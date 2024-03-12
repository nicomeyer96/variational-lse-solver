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
This file contains the main access point for using the variational LSE solver.
"""

import numpy as np
import torch
import pennylane as qml
import warnings
from typing import Callable
from tqdm import tqdm
import sys

from .utils import mode_init, mode_pauli_validation, mode_unitary_validation, mode_circuit_validation, mode_matrix_validation
from .utils import method_init, method_direct_validate, method_hadamard_validate, method_overlap_validate, method_coherent_validate
from .utils import init_imaginary_flag, init_imaginary_flag_dummy
from .circuits import dynamic_circuit

from .cost_function import CostFunction
from .cost_function.cost_function_types import CostFunctionMode, CostFunctionMethod, CostFunctionLoss


class VarLSESolver:
    """
    This class implements a variational LSE solver with customizable loss functions.
    """

    def __init__(
            self,
            a: np.ndarray | list[str | np.ndarray | Callable],
            b: np.ndarray | Callable,
            coeffs: list[float | complex] = None,
            ansatz: Callable = None,
            weights: tuple[int, ...] | np.ndarray = None,
            method: str = 'direct',
            local: bool = False,
            lr: float = 0.01,
            steps: int = 10000,
            epochs: int = 1,
            threshold: float = 1e-4,
            abort: int = 500,
            seed: int = None,
            data_qubits: int = 0,
    ):
        """
        Training variational LSE solver.

        :param a: System matrix given as either Pauli strings, unitaries, or circuits, or full matrix.
        :param b: Right side of the LSE.
        :param coeffs: Corresponding coefficients (optional when providing full system matrix).
        :param ansatz: The variational quantum circuit (optional, default is dynamic depth circuit).
        :param weights: Initial weights or shape of initial weights.
        :param method: Which method to use for computing the loss function.
        :param local: Whether to use to local or global loss formulation.
        :param lr: Learning rate to use for the Adam optimizer.
        :param steps: For how many steps to train the LSE solver.
        :param epochs: For how many epochs to train the LSE solver (increases circuit depth after each epoch).
        :param threshold: Early stopping criterion for loss value.
        :param abort: Number of steps after which to terminate if now loss improvement was observed.
        :param seed: Random seed to guarantee reproducible behaviour.
        :param data_qubits: Number of qubits required for encoding LSE.
        """

        mode: CostFunctionMode = mode_init(a)
        # determine number of required qubits for encoding the LSE
        self.data_qubits = mode.mode_dependent_value(pauli=mode_pauli_validation,
                                                     unitary=mode_unitary_validation,
                                                     circuit=mode_circuit_validation,
                                                     matrix=mode_matrix_validation)(a, coeffs, b, data_qubits)
        loss: CostFunctionLoss = CostFunctionLoss.LOCAL if local else CostFunctionLoss.GLOBAL
        method: CostFunctionMethod = method_init(method)
        method.method_dependent_value(direct=method_direct_validate,
                                      hadamard=method_hadamard_validate,
                                      overlap=method_overlap_validate,
                                      coherent=method_coherent_validate)(mode, loss, b)

        # set random seed (None by default)
        np.random.seed(seed)

        # set up variational ansatz, use dynamix depth circuit if None was provided
        self.ansatz, self.weights, self.dynamic_circuit, self.epochs = self.init_ansatz_and_weights(ansatz, weights, self.data_qubits, epochs)

        # determine whether it is necessary to evaluate imaginary terms (i.e. there are imaginary coefficients)
        imaginary = mode.mode_dependent_value(pauli=init_imaginary_flag, unitary=init_imaginary_flag,
                                              circuit=init_imaginary_flag, matrix=init_imaginary_flag_dummy)(coeffs)

        # determine training settings
        self.lr = lr
        self.opt = torch.optim.Adam([{'params': self.weights}], lr=lr)
        self.steps = steps
        if 0 > threshold > 1:
            raise ValueError('The `threshold` has to be in (0.0, 1.0).')
        self.threshold = threshold
        self.abort = abort

        # set up the actual cost function
        self.cost_function = CostFunction(a, coeffs, b, self.ansatz, mode, method, loss, self.data_qubits, imaginary)

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the LSE provided during initialization.

        :return: Solution of the LSE (proportional), Associated variational parameters
        """

        # current best weights (i.e. the ones producing the lowest loss)
        best_weights = self.weights.detach().numpy()

        for epoch in range(self.epochs):

            # best loss with corresponding step it was achieved in during this epoch
            best_loss, best_step = 1.0, 0

            # append additional layer to dynamic circuit (skip for first epoch) and re-register optimizer
            if 0 < epoch:
                print('Increasing circuit depth.', flush=True)
                new_weights = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, self.weights.shape[1]))
                weights = np.concatenate((best_weights, np.stack((new_weights,
                                                                  np.zeros((1, self.weights.shape[1])),
                                                                  -new_weights), axis=2)))
                self.weights = torch.tensor(weights, requires_grad=True)
                self.opt = torch.optim.Adam([{'params': self.weights}], lr=self.lr)

            # train until either maximum number of steps is reached, early stopping criteria is fulfilled,
            # or no loss function change in several consecutive steps (increase depth in this case)
            pbar = tqdm(range(self.steps), desc=f'Epoch {epoch+1}/{self.epochs}: ', file=sys.stdout)
            for step in pbar:
                self.opt.zero_grad()
                # compute loss
                loss = self.cost_function.cost(self.weights)
                # test is loss has improved beyond 0.1 * `threshold`
                # (ensures increasing depth when only negligible improvements are made)
                if loss.item() < best_loss and abs(loss.item() - best_loss) > 0.1 * self.threshold:
                    best_loss = loss.item()
                    best_step = step
                    best_weights = self.weights.detach().numpy()
                # test if stopping threshold has been reached
                if loss.item() < self.threshold:
                    pbar.close()
                    print(f'Loss of {loss.item():.10f} below stopping threshold.\nReturning solution.', flush=True)
                    return self.evaluate(best_weights), best_weights
                # if loss has not improved in the last `abort` steps terminate this epoch and increase depth
                if step - best_step >= self.abort:
                    pbar.close()
                    print(f'Loss has not improved in last {self.abort} steps.', flush=True) \
                        if epoch < self.epochs - 1 \
                        else print(f'Loss has not improved in last {self.abort} steps.\nReturning best solution.', flush=True)
                    break
                # log current loss to progress bar
                pbar.set_postfix({'best loss': best_loss, 'last improvement in step': best_step, 'loss': loss.item()})
                # determine gradients and update
                loss.backward()
                self.opt.step()
        return self.evaluate(best_weights), best_weights

    def evaluate(self, weights: np.array) -> np.array:
        """
        Return measurement probabilities for the state prepared as solution of the LSE.

        :param weights: Weights for the VQC ansatz.
        :return: Measurement probabilities for the state V(alpha)
        """
        return self.qnode_evaluate_x()(weights).detach().numpy()

    def qnode_evaluate_x(self) -> Callable:
        """
        Quantum node that evaluate V(alpha)

        :return: Circuit handle evaluating V(alpha)
        """
        dev = qml.device('default.qubit', wires=self.data_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit_evolve_x(weights):
            """
            Circuit that evaluates V(alpha)

            :param weights: Parameters for the VQC.
            """
            self.ansatz(weights)
            return qml.probs()

        return circuit_evolve_x

    @staticmethod
    def init_ansatz_and_weights(ansatz: Callable, weights: tuple[int, ...] | np.ndarray, data_qubits: int, epochs: int
                                ) -> tuple[Callable, torch.tensor, bool, int]:
        """
        Initialize variational weights.

        :param ansatz: Variational quantum circuit.
        :param weights: Explicit initial weights, shape of initial weights, or None for dynamic circuit.
        :param data_qubits: Number of qubits for implementing the LSE.
        :param epochs: Current epoch index, determines depth of dynamic circuit.
        :return: Variational circuit, initial weights, use dynamic ansatz, circuit depth
        """
        if ansatz is None:
            if weights is not None:
                warnings.warn('No explicit `ansatz` was selected, provided `weights` will be ignored.')
            weights = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, data_qubits, 3))
            return dynamic_circuit, torch.tensor(weights, requires_grad=True), True, epochs
        if not callable(ansatz):
            raise ValueError('The provided `ansatz` has to be Callable.')
        if epochs > 1:
            warnings.warn('Explicit `ansatz` was provided, `epochs` argument will be ignored.')
        if isinstance(weights, tuple):
            weights = np.random.uniform(low=0.0, high=2 * np.pi, size=weights)
            return ansatz, torch.tensor(weights, requires_grad=True), False, 1
        elif isinstance(weights, np.ndarray):
            return ansatz, torch.tensor(weights, requires_grad=True), False, 1
        else:
            raise ValueError('The `weights` have to be provided either explicitly as np.ndarray, or as an tuple indicating the shape.')
