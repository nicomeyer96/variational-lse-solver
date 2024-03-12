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
This file contains parameterized Pauli gates used when running in pauli mode.
"""

import numpy as np
import pennylane as qml
from typing import Sequence


def param_pauli_x(value: float, wires: int | Sequence[int]) -> None:
    """
    Apply a parameterized Pauli-X gate to the specified wires.

    The parameterized Pauli-X gate is implemented as a Hadamard gate followed by
    a parameterized Pauli-Z gate and another Hadamard gate. The parameterized Pauli-Z
    gate is controlled by the 'value' argument, which determines the rotation around
    the Z-axis. For a value of 0 or 1, the Pauli-X operation corresponds to the identity
    or the Pauli-X gate, respectively.
    """

    # We cannot simply use Rx, Ry, Rz as we need to watch for global_fn phases as we control the Paulis!
    # pX(0) := H * pZ(0) * H = H * I * H = I
    # pX(1) := H * pZ(1) * H = H * Z * H = X
    qml.Hadamard(wires=wires)
    param_pauli_z(value, wires=wires)
    qml.Hadamard(wires=wires)


def param_pauli_y(value: float, wires: int | Sequence[int]) -> None:
    """
    Apply a parameterized Pauli-Y gate to the specified wires.

    The parameterized Pauli-Y gate is implemented by applying a parameterized S gate,
    followed by a parameterized Pauli-X gate, and finally a parameterized S-dagger (Sdg) gate.
    The 'value' argument controls the parameter for the S and Sdg gates, which in turn control
    the rotation around the Z-axis. This sequence of operations results in a rotation around
    the Y-axis, effectively implementing the parameterized Pauli-Y gate.
    """

    # We cannot simply use Rx, Ry, Rz as we need to watch for global_fn phases as we control the Paulis!
    # pY(0) := pS(0) * pX(0) * pSdg(0) = I * I * I = I
    # pY(1) := pS(1) * pX(1) * PSdg(1) = S * X * Sdg = Y
    param_s(value, wires=wires)
    param_pauli_x(value, wires=wires)
    param_s_dagger(value, wires=wires)


def param_pauli_z(value: float, wires: int | Sequence[int]) -> None:
    """
    Apply a parameterized Pauli-Z gate to the specified wires.

    The parameterized Pauli-Z gate is implemented using a phase shift operation (U1)
    controlled by the 'value' argument. The phase shift is equivalent to a rotation
    around the Z-axis on the Bloch sphere. For a value of 0, the gate is the identity;
    for a value of 1, it corresponds to the Pauli-Z gate.
    """

    # We cannot simply use Rx, Ry, Rz as we need to watch for global_fn phases as we control the Paulis!
    # pZ(0) := U1(0) = PhaseShift(0) = ( 1   0  ) = I
    #                                  ( 0  e^0 )
    # pZ(1) := U1(pi) = PhaseShift(pi) = ( 1    0    ) = ( 1   0 ) = Z
    #                                    ( 0  e^i*pi )   ( 0  -1 )
    qml.U1(np.pi * value, wires=wires)


def param_s(value: float, wires: int | Sequence[int]) -> None:
    """
    Apply a parameterized S (phase) gate to the specified wires.

    The parameterized S gate introduces a phase shift that is a fraction of the
    full rotation around the Z-axis controlled by the 'value' argument. For a value
    of 0, the operation is the identity; for a value of 1, it corresponds to the
    standard S gate.
    """

    # We cannot simply use Rx, Ry, Rz as we need to watch for global_fn phases as we control the Paulis!
    # pS(0) := U1(0) = PhaseShift(0) = ( 1   0  ) = I
    #                                  ( 0  e^0 )
    # ps(1) := U1(pi/2) = PhaseShift(pi) = ( 1     0     ) = ( 1  0 ) = S
    #                                      ( 0  e^i*pi/2 )   ( 0  i )
    qml.U1(np.pi / 2 * value, wires=wires)


def param_s_dagger(value: float, wires: int | Sequence[int]) -> None:
    """
    Apply a parameterized S-dagger (Sdg) gate to the specified wires.

    The parameterized S-dagger gate applies the conjugate transpose of the
    parameterized S gate. It introduces a phase shift in the opposite direction
    with respect to the S gate, controlled by the 'value' argument. For a value of
    0, the gate is the identity; for a value of 1, it corresponds to the S-dagger gate.
    """

    # We cannot simply use Rx, Ry, Rz as we need to watch for global_fn phases as we control the Paulis!
    # pSdg(0) := U1(0) = PhaseShift(0) = ( 1   0  ) = I
    #                                    ( 0  e^0 )
    # pSdg(1) := U1(-pi/2) = PhaseShift(pi) = ( 1    0       ) = ( 1   0 ) = Sdg
    #                                         ( 0  e^i*-pi/2 )   ( 0  -i )
    qml.U1(-np.pi / 2 * value, wires=wires)
