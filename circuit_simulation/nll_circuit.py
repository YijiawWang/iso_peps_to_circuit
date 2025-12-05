#!/usr/bin/env python3
"""
Simulate the NLL circuit using basic gates (U3+CX) from info.txt files.

This uses pure PyTorch (no Qiskit) and applies gates sequentially one by one,
matching the structure in info.txt files.
"""

import os
import re
import random
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# Make ../src importable (so we can import optimize_unitary_gates)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "circuit_optimization")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Make test/ importable so we can use nll_decomposed utilities
TEST_DIR = os.path.join(BASE_DIR, "circuit_simulation")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

import nll_decomposed
from optimize_unitary_gates import u3_matrix, cnot_matrix

DECOMP_BASE = os.path.join(BASE_DIR, "gates_2patterns", "decomposed_gates")

# Qubit pattern for each gate (same as in nll_unitary.py)
ag = [
    [4],
    [2, 5],
    [1, 2],
    [4, 7],
    [4, 5, 7],
    [0, 1, 4],
    [7, 8],
    [6, 7],
    [3, 4, 6],
]


# Order of gates (same logical order as nll_unitary.py)
gate_order = [8, 5, 2, 1, 4, 7, 6, 3, 0]

# Regex patterns for parsing info.txt
_RE_U3 = re.compile(
    r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)"
)
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")


def parse_info_file(info_path: str):
    """
    Parse num_qubits and op lines from info.txt for a single gate.
    
    Returns:
        tuple: (num_qubits, num_ops, op_lines)
    """
    num_qubits = None
    num_ops = None
    op_lines = []

    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("num_qubits:"):
                num_qubits = int(line.split(":", 1)[1].strip())
            elif line.startswith("num_ops:"):
                num_ops = int(line.split(":", 1)[1].strip())
            elif line.startswith("op"):
                op_lines.append(line)

    if num_qubits is None:
        raise ValueError(f"num_qubits not found in {info_path}")
    if num_ops is None:
        num_ops = len(op_lines)

    return num_qubits, num_ops, op_lines


def apply_u3_gate(state: torch.Tensor, theta: float, phi: float, lam: float, qubit: int) -> torch.Tensor:
    """
    Apply U3 gate to a single qubit in the quantum state.
    
    Uses nll_decomposed.apply_gate to ensure consistency.
    
    Args:
        state: Quantum state tensor with shape (2, 2, ..., 2) for n qubits
        theta, phi, lam: U3 gate parameters
        qubit: Target qubit index
    
    Returns:
        Updated quantum state tensor
    """
    gate = u3_matrix(theta, phi, lam)
    return nll_decomposed.apply_gate(state, gate, [qubit])


def apply_cnot_gate(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """
    Apply CNOT gate to two qubits in the quantum state.
    
    Uses nll_decomposed.apply_gate to ensure consistency.
    
    Args:
        state: Quantum state tensor with shape (2, 2, ..., 2) for n qubits
        control: Control qubit index
        target: Target qubit index
    
    Returns:
        Updated quantum state tensor
    """
    gate = cnot_matrix()
    return nll_decomposed.apply_gate(state, gate, [control, target])


def load_gates_from_info():
    """
    Load all gate operations from info.txt files.
    
    Returns:
        list: List of gate operations, each is a dict with keys:
            - 'type': 'u3' or 'cnot'
            - 'params': parameters (theta, phi, lam for u3; control, target for cnot)
            - 'qubits': list of qubit indices (global indices)
            - 'gate_idx': which gate_index this belongs to
    """
    operations = []
    
    for gate_idx in gate_order:
        info_path = os.path.join(DECOMP_BASE, f"gate_index{gate_idx}", "info.txt")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Missing info.txt for gate_index{gate_idx}")
        
        num_qubits, num_ops, op_lines = parse_info_file(info_path)
        local_to_global = ag[gate_idx]  # mapping: local qubit -> global qubit
        
        if len(local_to_global) != num_qubits:
            raise ValueError(
                f"Mismatch: info num_qubits={num_qubits}, but ag[{gate_idx}] has "
                f"{len(local_to_global)} entries: {local_to_global}"
            )
        
        for line in op_lines:
            m_u3 = _RE_U3.match(line)
            if m_u3:
                theta = float(m_u3.group(1))
                phi = float(m_u3.group(2))
                lam = float(m_u3.group(3))
                q_local = int(m_u3.group(4))
                q_global = local_to_global[q_local]
                
                operations.append({
                    'type': 'u3',
                    'params': (theta, phi, lam),
                    'qubits': [q_global],
                    'gate_idx': gate_idx
                })
                continue
            
            m_cx = _RE_CNOT.match(line)
            if m_cx:
                c_local = int(m_cx.group(1))
                t_local = int(m_cx.group(2))
                c_global = local_to_global[c_local]
                t_global = local_to_global[t_local]
                
                operations.append({
                    'type': 'cnot',
                    'params': (c_global, t_global),
                    'qubits': [c_global, t_global],
                    'gate_idx': gate_idx
                })
                continue
            
            print(f"[WARN] Unrecognized op line for gate_index{gate_idx}: {line}")
    
    return operations


def simulate_circuit(operations):
    """
    Simulate a quantum circuit by applying operations sequentially.
    
    Args:
        operations: List of gate operations from load_gates_from_info()
    
    Returns:
        Final quantum state tensor with shape (2,)*9
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
    
    print("Loading gate operations from info.txt files...")
    operations = load_gates_from_info()
    print(f"Loaded {len(operations)} operations from {len(gate_order)} gates")
    
    print("\n=== Simulating circuit with basic gates (U3+CX) from info.txt ===")
    state = simulate_circuit(operations)
    probs = state.reshape(-1).abs() ** 2
    
    # Reuse utilities from nll_decomposed
    print("\n=== Calculating NLL on STANDARD_INDICES ===")
    nll = nll_decomposed.calculate_nll(state, nll_decomposed.STANDARD_INDICES)
    print(f"NLL: {nll}")
    
    print("\n=== Sampling from probability distribution ===")
    samples = nll_decomposed.sample_from_probability(
        probs, num_shots=100, num_qubits=9
    )
    print(f"Generated {len(samples)} samples from distribution")
    
    # Plot using nll_decomposed utilities
    nll_decomposed.plot_image_grid(
        samples,
        rows=10,
        cols=10,
        title="Samples from info.txt-based circuit (pure PyTorch implementation)",
        figsize=(5, 5),
        filename="../samples/samples_decomposed_circuit.png"
    )


if __name__ == "__main__":
    main()
