#!/usr/bin/env python3
"""
Simulate the optimized circuit from gates/final_circuit/info.txt
and calculate NLL (without noise).
"""

import torch
import numpy as np
import random
import os
import sys
import re

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "circuit_optimization")
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


def simulate_circuit(operations):
    """
    Simulate a quantum circuit by applying operations sequentially.
    
    Args:
        operations: List of gate operations from parse_info_file()
    
    Returns:
        Final quantum state tensor
    """
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
    
    print("apply done!")
    return state


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("Simulating optimized circuit from gates/final_circuit/info.txt")
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
    
    # Simulate circuit
    print("\n=== Simulating circuit (no noise) ===")
    state = simulate_circuit(operations)
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
    num_shots = 100
    samples = nll_decomposed.sample_from_probability(
        probs, num_shots=num_shots, num_qubits=9
    )
    print(f"Generated {len(samples)} samples from distribution")
    
    # Visualize samples
    nll_decomposed.plot_image_grid(
        samples,
        rows=10,
        cols=10,
        title=f"Samples from optimized circuit (shots={num_shots}, no noise)",
        figsize=(5, 5),
        filename="../samples/samples_final_circuit.png"
    )


if __name__ == "__main__":
    main()

