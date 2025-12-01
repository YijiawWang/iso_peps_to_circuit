"""
Optimize a parameterized ansatz circuit to fit each unitary gate in
../gates/unitary_gates/tensor{i}.pt, using a U3+CX-based structure.

This is a pure PyTorch implementation (no Qiskit) to ensure consistency
with PyTorch's tensor encoding.

Results are saved under ../gates/decomposed_gates/gate_index{i}/:
  - gate_matrix.pt : decomposed full unitary matrix
  - params.npy     : optimized parameters
  - info.txt       : metadata and optimization summary (including gate list)
"""

import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from scipy.optimize import minimize

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
    einsum_str = einsum_str[:-1]  # Remove last comma
    einsum_str += '->'
    for i in range(len(ci)): einsum_str += chr(ord('A') + i) # 'AB...ab..., a, -> AB...'
    for i in range(len(ci)):
        if ci[i] == 1:
            einsum_str += chr(ord('a') + i)
    id_vector = torch.tensor([1, 0], dtype=torch.complex128)
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


def u3_matrix(theta: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Build U3 gate matrix:
    U3(θ, φ, λ) = [
        [cos(θ/2), -e^(iλ) sin(θ/2)],
        [e^(iφ) sin(θ/2), e^(i(φ+λ)) cos(θ/2)]
    ]
    
    Args:
        theta, phi, lam: Scalar tensors or floats
    
    Returns:
        torch.Tensor: 2x2 complex matrix
    """
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64)
    if not isinstance(lam, torch.Tensor):
        lam = torch.tensor(lam, dtype=torch.float64)
    
    theta_half = theta / 2.0
    cos_theta = torch.cos(theta_half)
    sin_theta = torch.sin(theta_half)
    
    exp_i_lam = torch.exp(1j * lam)
    exp_i_phi = torch.exp(1j * phi)
    exp_i_phi_lam = torch.exp(1j * (phi + lam))
    
    matrix = torch.zeros(2, 2, dtype=torch.complex128)
    matrix[0, 0] = cos_theta
    matrix[0, 1] = -exp_i_lam * sin_theta
    matrix[1, 0] = exp_i_phi * sin_theta
    matrix[1, 1] = exp_i_phi_lam * cos_theta
    
    return matrix


def cnot_matrix() -> torch.Tensor:
    """
    Build CNOT gate matrix (control=0, target=1) using PyTorch's row-major encoding:
    CNOT = [
        [1, 0, 0, 0],  # |00⟩ -> |00⟩
        [0, 1, 0, 0],  # |01⟩ -> |01⟩
        [0, 0, 0, 1],  # |10⟩ -> |11⟩
        [0, 0, 1, 0]   # |11⟩ -> |10⟩
    ]
    
    Note: This matches PyTorch's row-major reshape order, not Qiskit's little-endian.
    
    Returns:
        torch.Tensor: 4x4 complex matrix
    """
    matrix = torch.zeros(4, 4, dtype=torch.complex128)
    matrix[0, 0] = 1.0  # |00⟩ -> |00⟩
    matrix[1, 1] = 1.0  # |01⟩ -> |01⟩
    matrix[2, 3] = 1.0  # |10⟩ -> |11⟩
    matrix[3, 2] = 1.0  # |11⟩ -> |10⟩
    return matrix


def apply_single_qubit_gate_to_matrix(state_matrix: torch.Tensor, gate: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    """
    Apply a single-qubit gate to a full state matrix.
    
    For a unitary matrix M (2^n x 2^n), applying gate U to qubit q:
    M' = (I ⊗ ... ⊗ U ⊗ ... ⊗ I) @ M
    
    We use movedim to move the target qubit to the first position, apply the gate,
    then move it back.
    
    Args:
        state_matrix: Full state matrix of shape (2**num_qubits, 2**num_qubits)
        gate: 2x2 gate matrix
        qubit: Target qubit index (0 to num_qubits-1)
        num_qubits: Total number of qubits
    
    Returns:
        Updated state matrix
    """
    # Reshape matrix to tensor: (2, 2, ..., 2, 2, 2, ..., 2)
    # First num_qubits dims are input, last num_qubits dims are output
    tensor = state_matrix.reshape(*([2] * (2 * num_qubits)))
    
    # Move qubit to first position in input side only
    # Input side: move qubit to position 0
    perm_in = list(range(num_qubits))
    perm_in.remove(qubit)
    perm_in.insert(0, qubit)
    
    # Output side: keep original order
    perm_out = [num_qubits + i for i in range(num_qubits)]
    
    perm = perm_in + perm_out
    tensor = tensor.permute(perm)
    
    # Reshape to (2, rest_in, 2**num_qubits)
    rest_in = 2 ** (num_qubits - 1)
    tensor = tensor.reshape(2, rest_in, 2 ** num_qubits)
    
    # Apply gate: U @ tensor
    # Apply U from left (input side): (2, rest_in, 2**num_qubits)
    tensor = torch.einsum('ij,jkl->ikl', gate, tensor)
    
    # Reshape back to tensor
    tensor = tensor.reshape(*([2] * (2 * num_qubits)))
    
    # Inverse permute
    inv_perm = [0] * (2 * num_qubits)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor = tensor.permute(inv_perm)
    
    # Reshape back to matrix
    state_matrix = tensor.reshape(2 ** num_qubits, 2 ** num_qubits)
    
    return state_matrix


def apply_two_qubit_gate_to_matrix(state_matrix: torch.Tensor, gate: torch.Tensor, qubit0: int, qubit1: int, num_qubits: int) -> torch.Tensor:
    """
    Apply a two-qubit gate to a full state matrix.
    
    For a unitary matrix M (2^n x 2^n), applying gate U to qubits q0, q1:
    M' = (I ⊗ ... ⊗ U ⊗ ... ⊗ I) @ M
    
    We use movedim to move the target qubits to the first two positions, apply the gate,
    then move them back. This works for any qubit positions, not just adjacent ones.
    
    Args:
        state_matrix: Full state matrix of shape (2**num_qubits, 2**num_qubits)
        gate: 4x4 gate matrix (assumes qubit0 is first, qubit1 is second in the gate)
        qubit0: First qubit index
        qubit1: Second qubit index
        num_qubits: Total number of qubits
    
    Returns:
        Updated state matrix
    """
    # Reshape matrix to tensor: (2, 2, ..., 2, 2, 2, ..., 2)
    # First num_qubits dims are input, last num_qubits dims are output
    tensor = state_matrix.reshape(*([2] * (2 * num_qubits)))
    
    # Move qubits to first two positions in input side only
    # Input side: move qubit0 and qubit1 to positions 0 and 1
    perm_in = list(range(num_qubits))
    perm_in.remove(qubit0)
    perm_in.remove(qubit1)
    perm_in = [qubit0, qubit1] + perm_in
    
    # Output side: keep original order
    perm_out = [num_qubits + i for i in range(num_qubits)]
    
    perm = perm_in + perm_out
    tensor = tensor.permute(perm)
    
    # Reshape to (4, rest_in, 2**num_qubits)
    rest_in = 2 ** (num_qubits - 2)
    tensor = tensor.reshape(4, rest_in, 2 ** num_qubits)
    
    # Apply gate: U @ tensor
    # Apply U from left (input side): (4, rest_in, 2**num_qubits)
    tensor = torch.einsum('ij,jkl->ikl', gate, tensor)
    
    # Reshape back to tensor
    tensor = tensor.reshape(*([2] * (2 * num_qubits)))
    
    # Inverse permute
    inv_perm = [0] * (2 * num_qubits)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor = tensor.permute(inv_perm)
    
    # Reshape back to matrix
    state_matrix = tensor.reshape(2 ** num_qubits, 2 ** num_qubits)
    
    return state_matrix


def build_ansatz_matrix(params: np.ndarray,
                       num_qubits: int = 3,
                       num_layers: int = 1) -> Tuple[torch.Tensor, List[str]]:
    """
    Build ansatz circuit matrix from parameters using pure PyTorch.
    
    Args:
        params: Parameter array (theta, phi, lam for each U3 gate)
        num_qubits: Number of qubits
        num_layers: Number of layers
    
    Returns:
        (matrix, gate_descriptions): Full unitary matrix and list of gate descriptions
    """
    params_t = torch.tensor(params, dtype=torch.float64)
    idx = 0
    
    # Start with identity matrix
    matrix = torch.eye(2 ** num_qubits, dtype=torch.complex128)
    gate_descriptions: List[str] = []
    op_idx = 0
    
    if num_layers == 0:
        for q in range(num_qubits):
            tx, ty, tz = params_t[idx:idx + 3]
            u3_gate = u3_matrix(tx, ty, tz)
            matrix = apply_single_qubit_gate_to_matrix(matrix, u3_gate, q, num_qubits)
            gate_descriptions.append(
                f"op{op_idx}: u3(theta={tx.item():.6f}, phi={ty.item():.6f}, lam={tz.item():.6f}, qubit={q})"
            )
            op_idx += 1
            idx += 3
    else:
        for layer in range(num_layers):
            # Only layer 0 has the leading U3(all)
            if layer == 0:
                for q in range(num_qubits):
                    tx, ty, tz = params_t[idx:idx + 3]
                    u3_gate = u3_matrix(tx, ty, tz)
                    matrix = apply_single_qubit_gate_to_matrix(matrix, u3_gate, q, num_qubits)
                    gate_descriptions.append(
                        f"op{op_idx}: u3(theta={tx.item():.6f}, phi={ty.item():.6f}, lam={tz.item():.6f}, qubit={q})"
                    )
                    op_idx += 1
                    idx += 3
            
            if num_qubits >= 2:
                # CNOT(0,1)
                cnot_gate = cnot_matrix()
                matrix = apply_two_qubit_gate_to_matrix(matrix, cnot_gate, 0, 1, num_qubits)
                gate_descriptions.append(f"op{op_idx}: cnot[0,1]")
                op_idx += 1
                
                # Second U3(all) block
                for q in range(num_qubits):
                    tx, ty, tz = params_t[idx:idx + 3]
                    u3_gate = u3_matrix(tx, ty, tz)
                    matrix = apply_single_qubit_gate_to_matrix(matrix, u3_gate, q, num_qubits)
                    gate_descriptions.append(
                        f"op{op_idx}: u3(theta={tx.item():.6f}, phi={ty.item():.6f}, lam={tz.item():.6f}, qubit={q})"
                    )
                    op_idx += 1
                    idx += 3
                
                if num_qubits >= 3:
                    # CNOT(1,2)
                    cnot_gate = cnot_matrix()
                    matrix = apply_two_qubit_gate_to_matrix(matrix, cnot_gate, 1, 2, num_qubits)
                    gate_descriptions.append(f"op{op_idx}: cnot[1,2]")
                    op_idx += 1
                    
                    # Third U3(all) block
                    for q in range(num_qubits):
                        tx, ty, tz = params_t[idx:idx + 3]
                        u3_gate = u3_matrix(tx, ty, tz)
                        matrix = apply_single_qubit_gate_to_matrix(matrix, u3_gate, q, num_qubits)
                        gate_descriptions.append(
                            f"op{op_idx}: u3(theta={tx.item():.6f}, phi={ty.item():.6f}, lam={tz.item():.6f}, qubit={q})"
                        )
                        op_idx += 1
                        idx += 3
    
    return matrix, gate_descriptions


def count_num_params(num_qubits: int, num_layers: int) -> int:
    """Count the number of parameters needed for the ansatz."""
    if num_layers == 0:
        return 3 * num_qubits
    else:
        # Layer 0: U3(all) + CNOT(0,1) + U3(all) [+ CNOT(1,2) + U3(all) if 3 qubits]
        # Layer 1+: CNOT(0,1) + U3(all) [+ CNOT(1,2) + U3(all) if 3 qubits]
        if num_qubits == 1:
            return 3 * num_qubits * num_layers
        elif num_qubits == 2:
            # Layer 0: 3*2 + 3*2 = 12
            # Layer 1+: 3*2 = 6 each
            return 12 + 6 * (num_layers - 1)
        else:  # num_qubits == 3
            # Layer 0: 3*3 + 3*3 + 3*3 = 27
            # Layer 1+: 3*3 + 3*3 = 18 each
            return 27 + 18 * (num_layers - 1)


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
    """Objective function for optimization."""
    matrix, _ = build_ansatz_matrix(params, num_qubits=num_qubits, num_layers=num_layers)
    ansatz_iso = get_isometry_matrix(matrix, ci)
    fid = fidelity_isometry(target_isometry, ansatz_iso)
    return float(1.0 - fid)


def _num_layers_for_gate_idx(gate_idx: int) -> int:
    """
    Heuristic choice of layers for tensor indices 0..8.
    """
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

    # 3. Count parameters
    num_params = count_num_params(num_qubits, num_layers)
    print(f"\n3. Ansatz structure:")
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
    final_matrix, gate_descriptions = build_ansatz_matrix(result.x, num_qubits=num_qubits, num_layers=num_layers)
    final_iso = get_isometry_matrix(final_matrix, ci)
    final_fid = fidelity_isometry(target_isometry, final_iso)
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

    # 7. Save results
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
            import traceback
            traceback.print_exc()
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
