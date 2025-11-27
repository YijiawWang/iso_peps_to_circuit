"""
Optimize a parameterized ansatz circuit to fit each unitary gate in
../gates/unitary_gates/tensor{i}.pt, using a U3+CX-based structure.

Results are saved under ../gates/decomposed_gates/gate_index{i}/:
  - gate_matrix.pt : decomposed full unitary matrix
  - params.npy     : optimized parameters
  - info.txt       : metadata and optimization summary (including gate list)
"""

import os
import time
from typing import Dict, Any, List
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from scipy.optimize import minimize
import time

contributed_index = {
    0: [1],
    1: [1,0],
    2: [1,0],
    3: [1,1],
    4: [1,1,0],
    5: [0,0,1],
    6: [1,0],
    7: [1,1],
    8: [0,0,0]
}

def get_isometry_matrix(matrix, ci):
    matrix = matrix.to(torch.complex128)
    tensor = matrix.reshape(*([2] * len(ci) * 2))
    einsum_str = '' 
    for i in range(len(ci)): einsum_str += chr(ord('A') + i) # 'AB...'
    for i in range(len(ci)): einsum_str += chr(ord('a') + i) # 'AB...ab...'
    einsum_str += ','
    for i in range(len(ci)):
        if ci[i] == 0:
            einsum_str += chr(ord('a') + i)
            einsum_str += ','
    einsum_str = einsum_str[:-1]  # 去掉最后一个逗号
    einsum_str += '->'
    for i in range(len(ci)): einsum_str += chr(ord('A') + i) # 'AB...ab..., a, -> AB...'
    for i in range(len(ci)):
        if ci[i] == 1:
            einsum_str += chr(ord('a') + i)
    id_vector = torch.tensor([1, 0], dtype=torch.complex128)
    # print(einsum_str)
    isometry_tensor = torch.einsum(einsum_str, tensor, *[id_vector for _ in range(ci.count(0))])
    isometry_matrix = isometry_tensor.reshape(2**len(ci), -1)
    return isometry_matrix


def loss_fn(predicted: torch.Tensor,
            target: torch.Tensor,
            ci: List[int]) -> torch.Tensor:
    """
    Frobenius‑norm based loss on the *isometry-reduced* matrices.
    """
    target_reduce = get_isometry_matrix(predicted, ci)
    predicted_reduce = get_isometry_matrix(target, ci)
    matrix = target_reduce.conj().T @ predicted_reduce
    dim = matrix.shape[0]
    eye = torch.eye(dim, dtype=torch.complex128, device=matrix.device)
    loss_matrix = (matrix - eye).abs() ** 2
    loss = loss_matrix.sum() / dim
    return loss


def create_ansatz_circuit(num_qubits: int = 3,
                          num_layers: int = 1):
    """
    CNOT‑based ansatz (similar spirit to optimize_gate5_cnot, but using U3):

      For 3 qubits:
        - Layer 0: U3(all) -> CNOT(0,1) -> U3(all) -> CNOT(1,2) -> U3(all)
        - Layer 1+ :        CNOT(0,1) -> U3(all) -> CNOT(1,2) -> U3(all)
        This avoids having two consecutive U3(all) blocks between layers.

      For 2 qubits: (U3 on all) -> CNOT(0,1) -> (U3 on all)
      For 1 qubit:  only U3 on that single qubit

    Here we directly use Qiskit's `u(theta, phi, lam)` (U3) as the generic
    single‑qubit rotation.
    """
    qc = QuantumCircuit(num_qubits)
    params: List[Parameter] = []

    if num_layers == 0:
        for q in range(num_qubits):
            tx = Parameter(f"θx_{q}")
            ty = Parameter(f"θy_{q}")
            tz = Parameter(f"θz_{q}")
            qc.u(tx, ty, tz, q)
            params.extend([tx, ty, tz])
    else:
        for layer in range(num_layers):
            # Only the first layer has the leading U3(all); later layers skip it
            if layer == 0:
                for q in range(num_qubits):
                    tx = Parameter(f"θx1_{layer}_{q}")
                    ty = Parameter(f"θy1_{layer}_{q}")
                    tz = Parameter(f"θz1_{layer}_{q}")
                    qc.u(tx, ty, tz, q)
                    params.extend([tx, ty, tz])

            if num_qubits >= 2:
                # CNOT(0,1)
                qc.cx(0, 1)

                # Second U3(all) block
                for q in range(num_qubits):
                    tx = Parameter(f"θx2_{layer}_{q}")
                    ty = Parameter(f"θy2_{layer}_{q}")
                    tz = Parameter(f"θz2_{layer}_{q}")
                    qc.u(tx, ty, tz, q)
                    params.extend([tx, ty, tz])

                if num_qubits >= 3:
                    # CNOT(1,2)
                    qc.cx(1, 2)

                    # Third U3(all) block
                    for q in range(num_qubits):
                        tx = Parameter(f"θx3_{layer}_{q}")
                        ty = Parameter(f"θy3_{layer}_{q}")
                        tz = Parameter(f"θz3_{layer}_{q}")
                        qc.u(tx, ty, tz, q)
                        params.extend([tx, ty, tz])

    return qc, params


def build_ansatz_circuit(params: np.ndarray,
                         num_qubits: int = 3,
                         num_layers: int = 1) -> QuantumCircuit:
    """
    Instantiate ansatz circuit with concrete parameter values.
    """
    qc = QuantumCircuit(num_qubits)
    idx = 0

    if num_layers == 0:
        for q in range(num_qubits):
            tx, ty, tz = params[idx:idx + 3]
            qc.u(tx, ty, tz, q)
            idx += 3
    else:
        for layer in range(num_layers):
            # Only layer 0 consumes the leading U3(all) parameters
            if layer == 0:
                for q in range(num_qubits):
                    tx, ty, tz = params[idx:idx + 3]
                    qc.u(tx, ty, tz, q)
                    idx += 3

            if num_qubits >= 2:
                # CNOT(0,1)
                qc.cx(0, 1)

                # Second U3(all) block
                for q in range(num_qubits):
                    tx, ty, tz = params[idx:idx + 3]
                    qc.u(tx, ty, tz, q)
                    idx += 3

                if num_qubits >= 3:
                    # CNOT(1,2)
                    qc.cx(1, 2)

                    # Third U3(all) block
                    for q in range(num_qubits):
                        tx, ty, tz = params[idx:idx + 3]
                        qc.u(tx, ty, tz, q)
                        idx += 3

    return qc


def circuit_to_isometry(circuit: QuantumCircuit,
                        ci: List[int]) -> torch.Tensor:
    """
    Convert quantum circuit to isometry matrix with ci constraint.
    """
    op = Operator(circuit)
    mat = op.data  # numpy array
    mat_t = torch.tensor(mat, dtype=torch.complex128)
    return get_isometry_matrix(mat_t, ci)


def fidelity_isometry(V_target: torch.Tensor,
                      V_ansatz: torch.Tensor) -> float:
    """
    Fidelity between two isometry matrices V1, V2 (m×n):
      F = |Tr(V1^† V2)|^2 / n^2
    """
    V_dag_V = torch.conj(V_target.T) @ V_ansatz
    trace = torch.trace(V_dag_V)
    n = V_target.shape[1]
    fid = torch.abs(trace) ** 2 / (n ** 2)
    return float(fid.item())


def objective_function(params: np.ndarray,
                       target_isometry: torch.Tensor,
                       num_qubits: int,
                       num_layers: int,
                       ci: List[int]) -> float:
    circuit = build_ansatz_circuit(params, num_qubits=num_qubits, num_layers=num_layers)
    ansatz_iso = circuit_to_isometry(circuit, ci)
    fid = fidelity_isometry(target_isometry, ansatz_iso)
    return float(1.0 - fid)


def _num_layers_for_gate_idx(gate_idx: int) -> int:
    """
    Heuristic choice of layers for tensor indices 0..8.
    """
    # You can tweak this heuristic later if needed.
    if gate_idx in (0, 1, 2, 6, 8):
        return 1
    elif gate_idx in (4,):
        return 4
    else:
        return 2


def optimize_gate(gate_idx: int,
                  max_iter: int = 500,
                  seed: int = 42) -> Dict[str, Any]:
    """
    Optimize ansatz to fit unitary_gates/tensor{gate_idx}.pt
    and save result under ../gates/decomposed_gates/gate_index{gate_idx}.
    """
    if gate_idx not in contributed_index:
        raise ValueError(f"gate_idx {gate_idx} not in contributed_index")

    ci = contributed_index[gate_idx]
    num_qubits = len(ci)
    num_layers = _num_layers_for_gate_idx(gate_idx)

    print("=" * 70)
    print(f"Optimizing ansatz to fit tensor{gate_idx}.pt (unitary_gates)")
    print(f"CI: {ci}, Layers: {num_layers}, Qubits: {num_qubits}")
    print("=" * 70)

    # 1. Load unitary gate tensor and reshape to matrix
    gate_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "gates",
        "unitary_gates",
        f"tensor{gate_idx}.pt",
    )
    gate_path = os.path.abspath(gate_path)
    print(f"\n1. Loading {gate_path} ...")
    tensor = torch.load(gate_path)
    if not tensor.is_complex():
        tensor = tensor.to(torch.complex128)
    else:
        tensor = tensor.to(torch.complex128)

    k = tensor.ndim // 2
    matrix = tensor.reshape(2 ** k, 2 ** k)
    print(f"   Tensor shape: {tuple(tensor.shape)}, matrix shape: {tuple(matrix.shape)}")

    # 2. Target isometry
    print(f"\n2. Computing target isometry matrix (ci={ci})...")
    target_isometry = get_isometry_matrix(matrix, ci)
    print(f"   Target isometry shape: {tuple(target_isometry.shape)}")

    # 3. Ansatz template
    print("\n3. Creating ansatz circuit template...")
    _, param_list = create_ansatz_circuit(num_qubits=num_qubits, num_layers=num_layers)
    num_params = len(param_list)
    print(f"   Number of layers: {num_layers}")
    print(f"   Number of parameters: {num_params}")

    # 4. Initial parameters
    np.random.seed(seed)
    initial_params = np.random.uniform(-np.pi, np.pi, num_params)

    best_fid = 0.0
    it_count = [0]

    def objective(x: np.ndarray) -> float:
        nonlocal best_fid
        loss = objective_function(
            x, target_isometry,
            num_qubits=num_qubits,
            num_layers=num_layers,
            ci=ci,
        )
        fid = 1.0 - loss
        if fid > best_fid:
            best_fid = fid
        it_count[0] += 1
        if it_count[0] % 10 == 0:
            print(f"   Iter {it_count[0]}: loss={loss:.6e}, fid={fid*100:.4f}%, best={best_fid*100:.4f}%")
        return loss

    print("\n4. Starting optimization (L-BFGS-B)...")
    print(f"   Max iterations: {max_iter}")
    start = time.time()
    result = minimize(
        objective,
        initial_params,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": True},
    )
    elapsed = time.time() - start

    # 5. Final evaluation
    print("\n5. Final evaluation...")
    final_circuit = build_ansatz_circuit(result.x, num_qubits=num_qubits, num_layers=num_layers)
    final_iso = circuit_to_isometry(final_circuit, ci)
    final_fid = fidelity_isometry(target_isometry, final_iso)

    op = Operator(final_circuit)
    final_matrix = torch.tensor(op.data, dtype=torch.complex128)
    loss_val = loss_fn(final_matrix, matrix, ci)

    print(f"   Optimization time: {elapsed:.2f}s")
    print(f"   Iterations: {result.nit}")
    print(f"   Final fidelity: {final_fid*100:.4f}%")
    print(f"   Best fidelity (during opt): {best_fid*100:.4f}%")
    print(f"   Final loss (1 - fidelity): {result.fun:.6e}")
    print(f"   Final loss_fn (Frobenius): {loss_val.item():.6e}")

    # 6. Isometry property check
    print("\n6. Isometry property check...")
    V_dag_V = torch.conj(final_iso.T) @ final_iso
    I = torch.eye(V_dag_V.shape[0], dtype=torch.complex128)
    iso_err = torch.norm(V_dag_V - I)
    print(f"   ||V^dagger V - I|| = {iso_err:.6e}")

    # 7. Extract gate decomposition info and save results
    print("\n7. Saving results...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gates", "decomposed_gates"))
    save_dir = os.path.join(base_dir, f"gate_index{gate_idx}")
    os.makedirs(save_dir, exist_ok=True)

    gate_matrix_path = os.path.join(save_dir, "gate_matrix.pt")
    torch.save(final_matrix, gate_matrix_path)
    print(f"   Saved gate matrix to: {gate_matrix_path}")

    params_path = os.path.join(save_dir, "params.npy")
    np.save(params_path, result.x)
    print(f"   Saved params to: {params_path}")

    # Build a human‑readable description of the decomposed circuit
    gate_descriptions: List[str] = []
    for idx, inst in enumerate(final_circuit.data):
        gate = inst.operation
        qubits = [q._index if hasattr(q, "_index") else q.index for q in inst.qubits]
        if gate.name in ("u", "u3"):
            # gate.params should be [theta, phi, lambda]
            if hasattr(gate, "params") and len(gate.params) >= 3:
                th, ph, lam = [float(p) for p in gate.params[:3]]
                gate_descriptions.append(
                    f"op{idx}: u3(theta={th:.6f}, phi={ph:.6f}, lam={lam:.6f}, qubit={qubits[0]})"
                )
            else:
                gate_descriptions.append(f"op{idx}: u3(qubit={qubits[0]})")
        elif gate.name in ("cx", "cnot"):
            if len(qubits) == 2:
                gate_descriptions.append(
                    f"op{idx}: cnot[{qubits[0]},{qubits[1]}]"
                )
            else:
                gate_descriptions.append(f"op{idx}: cnot{qubits}")
        else:
            gate_descriptions.append(f"op{idx}: {gate.name}{tuple(qubits)}")

    info_path = os.path.join(save_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"gate index: {gate_idx}\n")
        f.write(f"max_iter: {max_iter}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"ci: {ci}\n")
        f.write(f"final fidelity: {final_fid:.12f}\n")
        f.write(f"final loss (1 - fidelity): {result.fun:.12e}\n")
        f.write(f"final loss_fn (Frobenius): {loss_val.item():.12e}\n")
        f.write(f"iterations: {result.nit}\n")
        f.write(f"optimization time: {elapsed:.2f} seconds\n")
        f.write(f"num_qubits: {num_qubits}\n")
        f.write(f"num_params: {num_params}\n")
        f.write(f"num_ops: {len(gate_descriptions)}\n")
        # One line per decomposed gate, e.g.:
        #   op0: u3(theta=..., phi=..., lam=..., qubit=0)
        #   op1: cnot[0,1]
        for line in gate_descriptions:
            f.write(line + "\n")
    print(f"   Saved info to: {info_path}")

    return {
        "fidelity": final_fid,
        "loss": float(result.fun),
        "loss_fn": float(loss_val.item()),
        "iterations": int(result.nit),
        "time": float(elapsed),
        "params": result.x,
        "circuit": final_circuit,
        "isometry": final_iso,
        "gate_matrix": final_matrix,
    }


def cnot_per_gate(num_layers, num_qubits):
    if num_qubits == 1:
        return 0
    elif num_qubits == 2:
        return num_layers
    else:
        return 2 * num_layers

def optimize_all_gates(max_iter: int = 500,
                       seed: int = 42) -> Dict[int, Dict[str, Any]]:
    """
    Optimize all unitary_gates/tensor{i}.pt for i in contributed_index.keys().
    """
    all_results: Dict[int, Dict[str, Any]] = {}
    gate_indices = sorted(contributed_index.keys())

    print("=" * 70)
    print(f"Optimizing all unitary_gates tensors: {gate_indices}")
    print("=" * 70)

    for gi in gate_indices:
        print(f"\n{'=' * 70}")
        print(f"Processing tensor{gi}.pt")
        print(f"{'=' * 70}")
        try:
            res = optimize_gate(gi, max_iter=max_iter, seed=seed)
            all_results[gi] = res
        except Exception as e:
            print(f"Error optimizing tensor{gi}.pt: {e}")
            all_results[gi] = None

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL unitary_gates OPTIMIZATIONS")
    print("=" * 70)
    print(f"\n{'Idx':<6} {'Layers':<8} {'CI':<15} {'Fidelity':<12} {'Loss_fn':<12} {'Iter':<8}")
    print("-" * 70)

    for gi in gate_indices:
        res = all_results.get(gi)
        if res is None:
            print(f"{gi:<6} {'ERROR':<8} {'-':<15} {'-':<12} {'-':<12} {'-':<8}")
            continue
        ci = contributed_index[gi]
        layers = _num_layers_for_gate_idx(gi)
        print(f"{gi:<6} {layers:<8} {str(ci):<15} {res['fidelity']*100:>10.4f}%  "
              f"{res['loss_fn']:>10.6e}  {res['iterations']:>8}")

    return all_results


if __name__ == "__main__":
    optimize_all_gates(max_iter=500, seed=42)


    total = 0
    for idx in sorted(contributed_index.keys()):
        ci = contributed_index[idx]
        num_qubits = len(ci)
        num_layers = _num_layers_for_gate_idx(idx)
        c = cnot_per_gate(num_layers, num_qubits)
        total += c
        print(f"tensor{idx}: qubits={num_qubits}, layers={num_layers}, CNOTs={c}")

    print("TOTAL CNOTs across all tensors:", total)

