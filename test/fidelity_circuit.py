#!/usr/bin/env python3
"""
Compute the fidelity between the full 9‑qubit circuits before and after
decomposition, when both act on the initial state |000...0>.

The "before" circuit is defined in nll_unitary.py (using unitary_gates),
and the "after" circuit is defined in nll_decomposed.py (using
gates/decomposed_gates gate_matrix.pt).

This uses pure PyTorch (no Qiskit).
"""

import random

import numpy as np
import torch

import nll_unitary
import nll_decomposed


def simulate_unitary_circuit():
    """Simulate original circuit (unitary_gates) on |0...0>."""
    gates = nll_unitary.load_unitary_gates()
    state = nll_unitary.simulate_circuit(
        gates, [nll_unitary.ag[i] for i in nll_unitary.gate_order]
    )
    return state


def simulate_decomposed_circuit():
    """Simulate decomposed circuit (decomposed_gates) on |0...0>."""
    gates = nll_decomposed.load_decomposed_gates()
    state = nll_decomposed.simulate_circuit(
        gates, [nll_decomposed.ag[i] for i in nll_decomposed.gate_order]
    )
    return state


def state_fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> float:
    """
    Fidelity between two pure states |ψ_a>, |ψ_b|:
        F = |<ψ_a | ψ_b>|^2
    """
    psi_a = state_a.reshape(-1).to(torch.complex128)
    psi_b = state_b.reshape(-1).to(torch.complex128)
    overlap = torch.conj(psi_a) @ psi_b
    F = (overlap.abs() ** 2).item()
    return float(F)


def main():
    # Ensure deterministic seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Simulating original (unitary) circuit...")
    state_u = simulate_unitary_circuit()

    print("Simulating decomposed circuit...")
    state_d = simulate_decomposed_circuit()

    print("Computing fidelity between the two final states...")
    F = state_fidelity(state_u, state_d)
    print(f"Fidelity F = |<psi_unitary | psi_decomposed>|^2 = {F:.12f}")


if __name__ == "__main__":
    main()
