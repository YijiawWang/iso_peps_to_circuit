#!/usr/bin/env python3
"""
Simulate the optimized circuit from gates/final_circuit_cz/info.txt
with noise and calculate NLL.

CZ gates are followed by a 2-qubit depolarizing noise channel.
"""

import torch
import numpy as np
import random
import os
import sys
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "circuit_optimization")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from optimize_unitary_gates import u3_matrix
from optimize_unitary_gates_cz import cz_matrix
import nll_decomposed

FINAL_CIRCUIT_INFO = os.path.join(BASE_DIR, "gates", "final_circuit_cz", "info.txt")

_RE_U3 = re.compile(
    r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)"
)
_RE_CZ = re.compile(r"op\d+:\s*cz\[(\d+),(\d+)\]")

# Pauli matrices
I = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)
Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)

# All 2-qubit Pauli operators (excluding I⊗I)
PAULI_PAIRS = [
    (I, X), (I, Y), (I, Z),
    (X, I), (X, X), (X, Y), (X, Z),
    (Y, I), (Y, X), (Y, Y), (Y, Z),
    (Z, I), (Z, X), (Z, Y), (Z, Z),
]


def parse_info_file(info_path: str):
    """Parse operations from info.txt file."""
    operations = []

    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("num_"):
                continue

            m_u3 = _RE_U3.match(line)
            if m_u3:
                theta = float(m_u3.group(1))
                phi = float(m_u3.group(2))
                lam = float(m_u3.group(3))
                qubit = int(m_u3.group(4))
                operations.append({
                    'type': 'u3',
                    'params': (theta, phi, lam),
                    'qubits': [qubit],
                })
                continue

            m_cz = _RE_CZ.match(line)
            if m_cz:
                q0 = int(m_cz.group(1))
                q1 = int(m_cz.group(2))
                operations.append({
                    'type': 'cz',
                    'params': (q0, q1),
                    'qubits': [q0, q1],
                })
                continue

    return operations


def apply_u3_gate(state: torch.Tensor, theta: float, phi: float, lam: float, qubit: int) -> torch.Tensor:
    gate = u3_matrix(theta, phi, lam)
    return nll_decomposed.apply_gate(state, gate, [qubit])


def apply_cz_gate(state: torch.Tensor, q0: int, q1: int) -> torch.Tensor:
    gate = cz_matrix()
    return nll_decomposed.apply_gate(state, gate, [q0, q1])


def apply_depolarizing_noise(state: torch.Tensor, qubit0: int, qubit1: int, p: float) -> torch.Tensor:
    """
    Apply 2-qubit depolarizing noise channel:
        ρ' = (1 - p) * ρ + (p / 15) * Σ_{i} P_i ρ P_i†

    where P_i are the 15 non-identity 2-qubit Pauli operators.
    Approximates the mixed state by its principal eigenvector.
    """
    if p == 0.0:
        return state

    state_vec = state.reshape(-1)
    rho = torch.outer(state_vec, state_vec.conj())
    rho = rho.reshape(2 ** 9, 2 ** 9)

    rho_noisy = (1.0 - p) * rho

    for pauli0, pauli1 in PAULI_PAIRS:
        pauli_full = torch.eye(1, dtype=torch.complex128)
        for q in range(9):
            if q == qubit0:
                pauli_full = torch.kron(pauli_full, pauli0)
            elif q == qubit1:
                pauli_full = torch.kron(pauli_full, pauli1)
            else:
                pauli_full = torch.kron(pauli_full, I)

        rho_pauli = pauli_full @ rho @ pauli_full.conj().T
        rho_noisy = rho_noisy + (p / 15.0) * rho_pauli

    eigenvals, eigenvecs = torch.linalg.eigh(rho_noisy)
    principal_idx = torch.argmax(eigenvals.real)
    state_noisy = eigenvecs[:, principal_idx]
    state_noisy = state_noisy / torch.norm(state_noisy)
    state_noisy = state_noisy.reshape(*([2] * 9))

    return state_noisy


def simulate_circuit(operations, cz_fidelity: float = 1.0):
    """
    Simulate a quantum circuit by applying operations sequentially with noise.

    Args:
        operations: List of gate operations from parse_info_file()
        cz_fidelity: Average gate fidelity for CZ gates (1.0 = no noise)
    """
    # p = (1 - F_avg) * (d-1)/d, d=4 for 2-qubit gate
    d = 4
    p = (1.0 - cz_fidelity) * (d - 1) / d

    state = torch.zeros(2 ** 9, dtype=torch.complex128)
    state[0] = 1.0
    state = state.reshape(*([2] * 9))

    for op in operations:
        if op['type'] == 'u3':
            theta, phi, lam = op['params']
            qubit = op['qubits'][0]
            state = apply_u3_gate(state, theta, phi, lam, qubit)
        elif op['type'] == 'cz':
            q0, q1 = op['params']
            state = apply_cz_gate(state, q0, q1)
            if p > 0.0:
                state = apply_depolarizing_noise(state, q0, q1, p)

    print("apply done!")
    return state


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cz_fidelity = 0.99

    print("=" * 70)
    print(f"Simulating optimized circuit with noise (CZ fidelity={cz_fidelity:.2%})")
    print("=" * 70)

    print(f"\nLoading circuit from: {FINAL_CIRCUIT_INFO}")
    if not os.path.exists(FINAL_CIRCUIT_INFO):
        raise FileNotFoundError(f"Circuit info file not found: {FINAL_CIRCUIT_INFO}")

    operations = parse_info_file(FINAL_CIRCUIT_INFO)
    print(f"Loaded {len(operations)} operations")

    u3_count = sum(1 for op in operations if op['type'] == 'u3')
    cz_count = sum(1 for op in operations if op['type'] == 'cz')
    print(f"  U3 gates: {u3_count}")
    print(f"  CZ gates: {cz_count}")

    print(f"\n=== Simulating circuit with noise (CZ fidelity={cz_fidelity:.2%}) ===")
    state = simulate_circuit(operations, cz_fidelity=cz_fidelity)
    probs = state.reshape(-1).abs() ** 2

    print("\n=== Calculating NLL on STANDARD_INDICES ===")
    nll = nll_decomposed.calculate_nll(state, nll_decomposed.STANDARD_INDICES)
    print(f"\nFinal NLL: {nll:.6f}")

    print("\n=== Probabilities for STANDARD_INDICES ===")
    for idx_bits in nll_decomposed.STANDARD_INDICES:
        idx_tensor = torch.tensor(idx_bits, dtype=torch.long)
        psi = state[tuple(idx_tensor.tolist())]
        prob = (psi.abs() ** 2).item()
        print(f"{idx_bits}  p={prob:.6f}")

    print("\n=== Sampling from probability distribution ===")
    num_shots = 1000
    samples = nll_decomposed.sample_from_probability(
        probs, num_shots=num_shots, num_qubits=9
    )
    print(f"Generated {len(samples)} samples from distribution")

    nll_decomposed.plot_image_grid(
        samples,
        rows=10,
        cols=10,
        title=f"Samples from noisy circuit (shots={num_shots}, CZ fid={cz_fidelity:.2%})",
        figsize=(5, 5),
        filename="../samples/samples_noisy_circuit_cz.png",
    )


if __name__ == "__main__":
    main()
