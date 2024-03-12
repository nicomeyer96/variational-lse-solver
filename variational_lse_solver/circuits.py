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
This file contains a dynamic circuit implementation uss as ansatz for preparing the solution of the LSE.
"""

import pennylane as qml
import torch


def dynamic_circuit(weights: torch.tensor) -> None:
    """
    Function that realizes the dynamic ansatz.

    This ansatz consists of parameterized Rotations R_z R_y R_z, followed by a nearest-neighbor CZ- entanglement
    structure. The width and depth of the circuit is inherently defined by the shape of the weights.

    :param weights: Weights of the ansatz, have to be of shape [depth, num_qubits, 3]
    """

    assert 3 == len(weights.shape) and 3 == weights.shape[2]
    # extract circuit size and depth from shape of provided weights
    depth, num_qubits = weights.shape[0], weights.shape[1]

    # initial layer
    for qubit in range(num_qubits):
        # parameterized rotations
        qml.Rot(*weights[0, qubit], wires=qubit)

    # consecutive layers
    for layer in range(1, depth):
        # nearest-neighbor CZ-entanglement
        for qubit in range(num_qubits - 1):
            qml.CZ(wires=(qubit, qubit + 1))
        if num_qubits > 2:
            qml.CZ(wires=(num_qubits - 1, 0))
        # parameterized rotations
        for qubit in range(weights.shape[1]):
            qml.Rot(*weights[layer, qubit], wires=qubit)
