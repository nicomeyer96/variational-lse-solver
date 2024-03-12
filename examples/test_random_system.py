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
import argparse

from variational_lse_solver import VarLSESolver


def test_for_random_system(size: int = 3, local: bool = False, threshold: float = 5e-5) -> None:
    """
    This code solves a random sparse LSE using the `variational-lse-solve` library.

    :param size: size of the variational system, matrix scales as `2^size x 2^size` (default: 3)
    :param local: use global or local cost function (default: global, i.e. local=False)
    """

    # dimensionality of the system
    dim = 2 ** size

    # determine random matrix
    # NOTE: In principle it is also possible to subsequently decompose this matrix in either Pauli strings or unitaries
    #       and use the variational-lse-solver library as demonstrated in the other example. However, here we want to
    #       employ the direct method (which is not suitable for hardware, but guarantees the same results in simulation).
    random_sparse_matrix = np.random.rand(dim, dim)
    if np.isclose(0.0, np.linalg.det(random_sparse_matrix)):
        raise RuntimeError('Random matrix is non-invertible, please re-run,')

    # set up the right side, we are using a uniform vector for demonstration
    b = np.ones(dim)/np.sqrt(dim)

    # set up the variational LSE solver with a dynamic circuit ansatz
    # Note: A different ansatz might me much better suited for the individual setups, this setting is just for demonstration
    lse = VarLSESolver(
        random_sparse_matrix,  # system decomposition
        b,  # right side, alternatively use `b_fn`
        coeffs=None,  # not needed for direct evaluation method (this is the default)
        method='direct',  # use direct evaluation method (this is the default)
        local=local,  # global or local loss
        epochs=10,  # dynamically increase the circuit size to a maximum of 10 layers
        threshold=threshold,  # consider this loss as sufficient to have found a good solution, reduce for better accuracy
    )

    # train the variational LSE solver and report the found solution
    solution, _ = lse.solve()

    # compute normalized classical solution for comparison
    classical_solution = np.linalg.solve(random_sparse_matrix, b)
    normalized_classical_solution = np.square(classical_solution / np.linalg.norm(classical_solution))

    # print and compare solutions
    print('\nQuantum:\n|', end='')
    for s in solution:
        print(f' {s:.4f} |', end='')
    print('\nClassical:\n|', end='')
    for s in normalized_classical_solution:
        print(f' {s:.4f} |', end='')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true',
                        help='Use local cost function.')
    parser.add_argument('--size', type=int, default=3,
                        help='Size of the system (defaults to 3, i.e. system matrix of size 8x8).')
    parser.add_argument('--threshold', type=float, default=1e-5,
                        help='Termination threshold (defaults to 1e-5).')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    _args = parse()
    test_for_random_system(size=_args.size, local=_args.local, threshold=_args.threshold)
