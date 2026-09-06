# State-Aware Cancellation Of Gate Sequences Using Quantum Constant Propagation

This repository contains the implementation developed as part of my Bachelor's thesis at the Technical University of Munich.

The project extends **Quantum Constant Propagation (QCP)** with a layer-based optimization pass that detects and removes groups of gates whose combined effect leaves the quantum state unchanged.

## Overview

Quantum Constant Propagation optimizes quantum circuits by propagating information about the states of qubits through the circuit. This allows it to remove gates that have no effect on the current state and to simplify or remove controlled operations whose controls are known.

However, QCP analyzes gates individually and therefore cannot detect sequences of gates that are redundant only when considered together.

This project extends QCP by dividing the circuit into layers and storing snapshots of the abstract qubit states after every layer. Windows of consecutive layers are analyzed by comparing the states at their boundaries. If the relevant qubit states are equivalent up to global phase, gates inside the window can be removed while preserving the semantics of the circuit.

## Main Extensions

The implementation adds:

* layer-based detection of redundant gate sequences;
* comparison of abstract qubit states across layer boundaries up to global phase;
* expansion of candidate windows to identify larger removable regions;
* support for exact application of three- and four-qubit operations;
* support for controlled gates activated when all control qubits are in the zero state (`ctrl_state = 0`);



## Implementation

The implementation is written in Python and uses [Qiskit](https://github.com/Qiskit/qiskit) for quantum circuit representation and manipulation.

The analysis tracks abstract quantum states using a bounded representation. If the size of a tracked state exceeds the configured limit, the state is conservatively treated as unknown.

## Acknowledgements

This implementation is based on code originally provided by **Innocenzo Fulginiti** and extends the Quantum Constant Propagation implementation used in his work.

The corresponding implementation can be found here:

* [Branch-Aware Quantum Constant Propagation – GitHub](https://github.com/1nnocenzo/bqcp)
* [Branch-Aware Quantum Constant Propagation for Dynamic Quantum Circuits – arXiv:2606.02018](https://arxiv.org/abs/2606.02018)

The original implementation served as the starting point for the extensions developed in this Bachelor's thesis.

## Author

Konstantin Fedorov
Technical University of Munich
