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
import argparse

from variational_lse_solver import VarLSESolver


# unitary representation of Paulis
I_ = np.array([[1.0, 0.0], [0.0, 1.0]])
X_ = np.array([[0.0, 1.0], [1.0, 0.0]])
Y_ = np.array([[0.0, -1.j], [1.j, 0.0]])
Z_ = np.array([[1.0, 0.0], [0.0, -1.0]])


def reproduce_bravo_prieto(mode: str = 'pauli', method: str = 'hadamard',
                           local: bool = False, steps: int = 50) -> None:
    """
    This code reproduces an experiment to variationally solve a LSE, which was originally proposed in
    C. Bravo-Prieto et al., "Variational Quantum Linear Solver: A Hybrid Algorithm for Linear Systems", arXiv:1909.05820v2 (2020).
    The implementation uses the `variational-lse-solve` library.

    :param mode: decomposition mode of system (default: pauli)
    :param method: loss evaluation method (default: hadamard)
    :param local: use global or local cost function (default: global, i.e. local=False)
    :param steps: number of steps to train for (default: 50)
    """

    match mode:
        case 'pauli':
            # decomposition into Pauli strings
            a = ['III', 'XZI', 'XII']
        case 'unitary':
            # decomposition into corresponding unitaries
            # Note: This is just for demonstration, it is also possible to use unitaries that are not just Pauli strings
            a = [np.kron(I_, np.kron(I_, I_)), np.kron(X_, np.kron(Z_, I_)), np.kron(X_, np.kron(I_, I_))]
        case 'circuit':
            # decomposition into circuits implementing the underlying unitaries
            def III_fn(): pass
            def XZI_fn(): qml.PauliX(0); qml.PauliZ(1)  # noqa
            def XII_fn(): qml.PauliX(0)  # noqa
            a = [III_fn, XZI_fn, XII_fn]
        case _:
            raise ValueError(f'Mode {mode} unknown.')

    # right side of the LSE
    b = np.ones(8)/np.sqrt(8)
    # alternatively can also be provided as implementing circuit:
    def b_fn(): qml.Hadamard(0); qml.Hadamard(1); qml.Hadamard(2)  # noqa

    # set up the variational LSE solver with a dynamic circuit ansatz
    lse = VarLSESolver(
        a,  # system decomposition
        b_fn,  # right side, alternatively use `b_fn`
        coeffs=[1.0, 0.2, 0.2],  # coefficients for `a`
        method=method,  # computation method
        local=local,  # global or local loss
        steps=steps, lr=0.1,  # set number of steps and learning rate
        data_qubits=3  # only required for `circuit` mode
    )

    # train the variational LSE solver and report the found solution
    solution, _ = lse.solve()

    # compute normalized classical solution for comparison, therefore first re-compose system matrix A
    A = 1.0 * np.kron(I_, np.kron(I_, I_)) + 0.2 * np.kron(X_, np.kron(Z_, I_)) + 0.2 * np.kron(X_, np.kron(I_, I_))
    classical_solution = np.linalg.solve(A, b)
    normalized_classical_solution = np.square(classical_solution / np.linalg.norm(classical_solution))

    # print and compare solutions
    print('\nQuantum:\n|', end='')
    for s in solution:
        print(f' {s:.4f} |', end='')
    print('\nClassical:\n|', end='')
    for s in normalized_classical_solution:
        print(f' {s:.4f} |', end='')
    print()


def parse():
    parser = argparse.ArgumentParser()
    mode_choices = ['pauli', 'unitary', 'circuit']
    method_choices = ['hadamard', 'overlap', 'coherent']
    parser.add_argument('--mode', type=str, choices=mode_choices, default='pauli',
                        help='Which decomposition mode to use (defaults to `pauli`).')
    parser.add_argument('--method', type=str, choices=method_choices, default='hadamard',
                        help='Which loss evaluation method to use (defaults to `hadamard`).')
    parser.add_argument('--local', action='store_true',
                        help='Use local cost function.')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps to use (defaults to 100).')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    _args = parse()
    reproduce_bravo_prieto(mode=_args.mode, method=_args.method, local=_args.local, steps=_args.steps)
