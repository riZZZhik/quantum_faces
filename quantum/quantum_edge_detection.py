from math import pi

from qiskit.aqua.circuits.fourier_transform_circuits import FourierTransformCircuits


def quantum_adder(circuit, epsilon):
    qubits = circuit.qubits
    n_qubits = circuit.n_qubits
    FourierTransformCircuits.construct_circuit(circuit, qubits)
    for i in range(n_qubits):
        circuit.u1(float(2 * pi * epsilon) / 2 ** (i + 1), qubits[n_qubits - i - 1])
    FourierTransformCircuits.construct_circuit(circuit, qubits, inverse=True)


def quantum_edge_detection(circuit):  # gives you the quantum state where you have to measure the ancilla and obtain 0
    qubits = circuit.qubits
    ancilla = qubits[0]
    circuit.h(ancilla)
    quantum_adder(circuit, -1)
    circuit.h(ancilla)
    circuit.x(ancilla)
