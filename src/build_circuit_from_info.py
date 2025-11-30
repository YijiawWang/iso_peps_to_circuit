#!/usr/bin/env python3
"""
Build Qiskit circuit from decomposed gates info.txt files.
This is a standalone version that doesn't depend on construct_qiskit_circuit.py.
"""

import os
import re
from typing import List, Tuple
from qiskit import QuantumCircuit

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DECOMP_BASE = os.path.join(BASE_DIR, "gates", "decomposed_gates")

# Qubit pattern for each gate (same as in nll_circuit.py)
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

# Order of gates (same logical order as nll_circuit.py)
gate_order = [8, 5, 2, 1, 4, 7, 6, 3, 0]

_RE_U3 = re.compile(
    r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)"
)
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")


def parse_info_file(info_path: str) -> Tuple[int, int, List[str]]:
    """Parse num_qubits and op lines from info.txt for a single gate."""
    num_qubits = None
    num_ops = None
    op_lines: List[str] = []

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


def build_global_decomposed_circuit() -> QuantumCircuit:
    """
    Build a 9‑qubit QuantumCircuit corresponding to the full NLL circuit
    using the decomposed gate descriptions in info.txt.
    """
    qc = QuantumCircuit(9)

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
                qc.u(theta, phi, lam, q_global)
                continue

            m_cx = _RE_CNOT.match(line)
            if m_cx:
                c_local = int(m_cx.group(1))
                t_local = int(m_cx.group(2))
                c_global = local_to_global[c_local]
                t_global = local_to_global[t_local]
                # PyTorch's CNOT(control, target) corresponds to Qiskit's cx(control, target)
                qc.cx(c_global, t_global)
                continue

            # Ignore non-operation lines (like "optimization time: ...")
            if not line.startswith("op"):
                continue

    return qc


if __name__ == "__main__":
    qc = build_global_decomposed_circuit()
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total operations: {qc.size()}")
    print(f"Operation counts: {qc.count_ops()}")

