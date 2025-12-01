#!/usr/bin/env python3
"""
Simulate the optimized circuit from gates/final_circuit/info.txt
with noise and calculate NLL.

This is a noisy version of circuit_simulation.py, where CNOT gates
are followed by a depolarizing noise channel.
"""

import torch
import numpy as np
import random
import os
import sys
import re

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from optimize_unitary_gates import u3_matrix, cnot_matrix
import nll_decomposed

# Path to final circuit info.txt
FINAL_CIRCUIT_INFO = os.path.join(BASE_DIR, "gates", "final_circuit", "info.txt")

# Regex patterns for parsing info.txt
_RE_U3 = re.compile(
    r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)"
)
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")

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
    (Z, I), (Z, X), (Z, Y), (Z, Z)
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
                    'qubits': [qubit]
                })
                continue
            
            m_cx = _RE_CNOT.match(line)
            if m_cx:
                control = int(m_cx.group(1))
                target = int(m_cx.group(2))
                operations.append({
                    'type': 'cnot',
                    'params': (control, target),
                    'qubits': [control, target]
                })
                continue
    
    return operations


def apply_u3_gate(state: torch.Tensor, theta: float, phi: float, lam: float, qubit: int) -> torch.Tensor:
    """Apply U3 gate to a single qubit in the quantum state."""
    gate = u3_matrix(theta, phi, lam)
    return nll_decomposed.apply_gate(state, gate, [qubit])


def apply_cnot_gate(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Apply CNOT gate to two qubits in the quantum state."""
    gate = cnot_matrix()
    return nll_decomposed.apply_gate(state, gate, [control, target])


def apply_depolarizing_noise(state: torch.Tensor, qubit0: int, qubit1: int, p: float) -> torch.Tensor:
    """
    Apply 2-qubit depolarizing noise channel to the quantum state.
    
    The depolarizing channel is:
        ρ' = (1 - p) * ρ + (p / 15) * Σ_{i} P_i ρ P_i^†
    
    where P_i are the 15 non-identity 2-qubit Pauli operators.
    
    For computational efficiency, we compute the noisy state by:
    1. Convert to density matrix
    2. Apply depolarizing channel
    3. Extract principal eigenvector (approximation for mixed state)
    
    Args:
        state: Quantum state tensor with shape (2, 2, ..., 2) for n qubits
        qubit0: First qubit index
        qubit1: Second qubit index
        p: Error probability (related to fidelity: p = (1 - F_avg) * 3 / 4)
    
    Returns:
        Updated quantum state tensor after applying noise
    """
    if p == 0.0:
        return state
    
    # Convert state to density matrix: |ψ⟩⟨ψ|
    state_vec = state.reshape(-1)
    rho = torch.outer(state_vec, state_vec.conj())
    rho = rho.reshape(2 ** 9, 2 ** 9)
    
    # Apply depolarizing channel: ρ' = (1 - p) * ρ + (p / 15) * Σ P_i ρ P_i^†
    rho_noisy = (1.0 - p) * rho
    
    # Apply each Pauli error and sum
    for pauli0, pauli1 in PAULI_PAIRS:
        # Build full 9-qubit operator: I ⊗ ... ⊗ P ⊗ ... ⊗ I
        # Create identity operators for other qubits
        pauli_full = torch.eye(1, dtype=torch.complex128)
        for q in range(9):
            if q == qubit0:
                # Apply first part of 2-qubit Pauli to qubit0
                pauli_full = torch.kron(pauli_full, pauli0)
            elif q == qubit1:
                # Apply second part of 2-qubit Pauli to qubit1
                pauli_full = torch.kron(pauli_full, pauli1)
            else:
                # Identity for other qubits
                pauli_full = torch.kron(pauli_full, I)
        
        # Apply Pauli: P ρ P^†
        rho_pauli = pauli_full @ rho @ pauli_full.conj().T
        rho_noisy = rho_noisy + (p / 15.0) * rho_pauli
    
    # Convert back to pure state representation
    # Use principal eigenvector (largest eigenvalue) as approximation
    eigenvals, eigenvecs = torch.linalg.eigh(rho_noisy)
    principal_idx = torch.argmax(eigenvals.real)
    state_noisy = eigenvecs[:, principal_idx]
    state_noisy = state_noisy / torch.norm(state_noisy)
    state_noisy = state_noisy.reshape(*([2] * 9))
    
    return state_noisy


def simulate_circuit(operations, cnot_fidelity: float = 1.0):
    """
    Simulate a quantum circuit by applying operations sequentially with noise.
    
    Args:
        operations: List of gate operations from parse_info_file()
        cnot_fidelity: Average gate fidelity for CNOT gates (default: 1.0 = no noise)
    
    Returns:
        Final quantum state tensor
    """
    # Calculate error probability from fidelity
    # For 2-qubit gate: F_avg = 1 - p * d / (d - 1), where d = 4
    # => p = (1 - F_avg) * (d - 1) / d = (1 - F_avg) * 3 / 4
    d = 4  # dimension for 2-qubit gate
    p = (1.0 - cnot_fidelity) * (d - 1) / d
    
    state = torch.zeros(2 ** 9, dtype=torch.complex128)
    state[0] = 1.0
    state = state.reshape(*([2] * 9))
    
    for op in operations:
        if op['type'] == 'u3':
            theta, phi, lam = op['params']
            qubit = op['qubits'][0]
            state = apply_u3_gate(state, theta, phi, lam, qubit)
        elif op['type'] == 'cnot':
            control, target = op['params']
            state = apply_cnot_gate(state, control, target)
            # Apply depolarizing noise after CNOT
            if p > 0.0:
                state = apply_depolarizing_noise(state, control, target, p)
    
    print("apply done!")
    return state


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Noise parameters
    cnot_fidelity = 0.99  # 99% fidelity for CNOT gates
    
    print("=" * 70)
    print(f"Simulating optimized circuit with noise (CNOT fidelity={cnot_fidelity:.2%})")
    print("=" * 70)
    
    # Load operations from final_circuit/info.txt
    print(f"\nLoading circuit from: {FINAL_CIRCUIT_INFO}")
    if not os.path.exists(FINAL_CIRCUIT_INFO):
        raise FileNotFoundError(f"Circuit info file not found: {FINAL_CIRCUIT_INFO}")
    
    operations = parse_info_file(FINAL_CIRCUIT_INFO)
    print(f"Loaded {len(operations)} operations")
    
    # Count operations
    u3_count = sum(1 for op in operations if op['type'] == 'u3')
    cnot_count = sum(1 for op in operations if op['type'] == 'cnot')
    print(f"  U3 gates: {u3_count}")
    print(f"  CNOT gates: {cnot_count}")
    
    # Simulate circuit with noise
    print(f"\n=== Simulating circuit with noise (CNOT fidelity={cnot_fidelity:.2%}) ===")
    state = simulate_circuit(operations, cnot_fidelity=cnot_fidelity)
    probs = state.reshape(-1).abs() ** 2
    
    # Calculate NLL
    print("\n=== Calculating NLL on STANDARD_INDICES ===")
    nll = nll_decomposed.calculate_nll(state, nll_decomposed.STANDARD_INDICES)
    print(f"\nFinal NLL: {nll:.6f}")
    
    # Print probabilities for STANDARD_INDICES
    print("\n=== Probabilities for STANDARD_INDICES ===")
    for idx_bits in nll_decomposed.STANDARD_INDICES:
        idx_tensor = torch.tensor(idx_bits, dtype=torch.long)
        psi = state[tuple(idx_tensor.tolist())]
        p = (psi.abs() ** 2).item()
        print(f"{idx_bits}  p={p:.6f}")
    
    # Sample from probability distribution
    print("\n=== Sampling from probability distribution ===")
    num_shots = 1000
    samples = nll_decomposed.sample_from_probability(
        probs, num_shots=num_shots, num_qubits=9
    )
    print(f"Generated {len(samples)} samples from distribution")
    
    # Visualize samples
    nll_decomposed.plot_image_grid(
        samples,
        rows=10,
        cols=10,
        title=f"Samples from noisy circuit (shots={num_shots}, CX fid={cnot_fidelity:.2%})",
        figsize=(5, 5),
        filename="../samples/samples_noisy_circuit.png"
    )


if __name__ == "__main__":
    main()
