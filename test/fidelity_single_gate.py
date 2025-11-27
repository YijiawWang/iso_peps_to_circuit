#!/usr/bin/env python3
"""
Check, for each decomposed gate, that:

  1) Rebuilding the circuit from gates listed in
     ../gates/decomposed_gates/gate_index{i}/info.txt
     reproduces the saved gate_matrix.pt.

  2) (Optional) Report the distance between the original unitary_gates
     tensor{i}.pt and the decomposed gate via isometry.
"""

import os
import re
import sys
from typing import List, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Make ../src importable (so we can import optimize_unitary_gates)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from optimize_unitary_gates import contributed_index, get_isometry_matrix


def parse_info_file(info_path: str) -> Tuple[int, int, List[str]]:
    """
    Parse num_qubits and op lines from info.txt.

    Returns:
        num_qubits: int
        num_ops: int (from header)
        op_lines: list of 'opX: ...' strings
    """
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


_RE_U3 = re.compile(
    r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)"
)
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")


def build_circuit_from_info(num_qubits: int, op_lines: List[str]) -> QuantumCircuit:
    """
    Rebuild the decomposed circuit described in info.txt.
    """
    qc = QuantumCircuit(num_qubits)

    for line in op_lines:
        m_u3 = _RE_U3.match(line)
        if m_u3:
            theta = float(m_u3.group(1))
            phi = float(m_u3.group(2))
            lam = float(m_u3.group(3))
            q = int(m_u3.group(4))
            qc.u(theta, phi, lam, q)
            continue

        m_cx = _RE_CNOT.match(line)
        if m_cx:
            c = int(m_cx.group(1))
            t = int(m_cx.group(2))
            qc.cx(c, t)
            continue

        # Fallback: ignore unknown lines but print them
        print(f"[WARN] Unrecognized op line: {line}")

    return qc


def frobenius_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b).item())


def check_single_gate(idx: int, base_dir: str) -> None:
    """
    For a given tensor index idx:
      - load gate_matrix.pt and info.txt
      - rebuild circuit from info.txt
      - compare rebuilt matrix to saved gate_matrix.pt
      - also report distance to original unitary_gates tensor{idx}.pt (via isometry)
    """
    print("=" * 70)
    print(f"Checking tensor{idx}")

    dec_dir = os.path.join(base_dir, "gates", "decomposed_gates", f"gate_index{idx}")
    info_path = os.path.join(dec_dir, "info.txt")
    gate_mat_path = os.path.join(dec_dir, "gate_matrix.pt")

    if not os.path.exists(info_path) or not os.path.exists(gate_mat_path):
        print(f"[SKIP] Missing info or gate_matrix for tensor{idx}")
        return

    num_qubits, num_ops, op_lines = parse_info_file(info_path)
    print(f"  num_qubits={num_qubits}, num_ops(header)={num_ops}, parsed_ops={len(op_lines)}")

    # Load saved decomposed matrix
    dec_matrix = torch.load(gate_mat_path)
    if not dec_matrix.is_complex():
        dec_matrix = dec_matrix.to(torch.complex128)
    else:
        dec_matrix = dec_matrix.to(torch.complex128)
    print(f"  gate_matrix.pt shape: {tuple(dec_matrix.shape)}")

    # Rebuild circuit and get matrix
    qc = build_circuit_from_info(num_qubits, op_lines)
    op = Operator(qc)
    rec_matrix = torch.tensor(op.data, dtype=torch.complex128)
    print(f"  rebuilt circuit matrix shape: {tuple(rec_matrix.shape)}")

    diff_rec = frobenius_norm(rec_matrix, dec_matrix)
    print(f"  ||rebuilt - gate_matrix||_F = {diff_rec:.3e}")

    # Optional: compare to original unitary_gates tensor{i}.pt via isometry
    unit_path = os.path.join(base_dir, "gates", "unitary_gates", f"tensor{idx}.pt")
    if os.path.exists(unit_path):
        unit_tensor = torch.load(unit_path)
        if not unit_tensor.is_complex():
            unit_tensor = unit_tensor.to(torch.complex128)
        else:
            unit_tensor = unit_tensor.to(torch.complex128)
        k = unit_tensor.ndim // 2
        unit_matrix = unit_tensor.reshape(2 ** k, 2 ** k)

        ci = contributed_index[idx]
        V_target = get_isometry_matrix(unit_matrix, ci)
        V_dec = get_isometry_matrix(dec_matrix, ci)

        V_dag_V = torch.conj(V_target.T) @ V_dec
        trace = torch.trace(V_dag_V)
        n = V_target.shape[1]
        fid = torch.abs(trace) ** 2 / (n ** 2)
        print(f"  isometry fidelity with unitary_gates/tensor{idx}.pt: {float(fid.item())*100:.4f}%")
    else:
        print(f"  [WARN] original unitary tensor not found: {unit_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    indices = sorted(contributed_index.keys())

    print("Base dir:", base_dir)
    print("Tensor indices:", indices)

    for idx in indices:
        check_single_gate(idx, base_dir)


if __name__ == "__main__":
    main()


