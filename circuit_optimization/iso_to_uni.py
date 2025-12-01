#!/usr/bin/env python3
import os
from typing import List
import torch


def complete_isometry_to_unitary(V: torch.Tensor) -> torch.Tensor:
    print(V.dtype)
    m, k = V.shape
    if m < k:
        raise ValueError("--")
    if m == k:
        return V
    Q_full = torch.linalg.qr(V, mode='complete')[0]
    V_perp = Q_full[:, k:]
    U = torch.hstack([V, V_perp])
    return U


def iso_to_uni(iso_tensor: torch.Tensor, qubits: List[int]) -> torch.Tensor:
    """
    Convert an isometry tensor to a unitary gate.

    Args:
        iso_tensor: Isometry tensor to convert, the bond order is [p,l,r,d,u]
        qubits: List of qubit_ids, the order is [p,l,r,d,u],
                the value is negative if it's the input flow in the iso-PEPS graph,
                is 0 if there is no flow.

    Returns:
        Tensor with bonds permuted so that the logical bond order is
        [k1,k2,k3,-k1,-k2,-k3] (or truncated versions like [k1,k2,-k1,-k2] / [k1,-k1]).
        Here we **skip** those ids that appear only with negative sign (only_neg_ids).
    """
    assert iso_tensor.ndim == len(qubits), \
        f"iso_tensor.ndim ({iso_tensor.ndim}) must match len(qubits) ({len(qubits)})"

    # Collect positive / negative ids
    pos_ids = {abs(q) for q in qubits if q > 0}
    neg_ids = {abs(q) for q in qubits if q < 0}
    sorted_qubit_ids = sorted(neg_ids)

    # Map each id to its positive / negative axis index in the original tensor
    id_to_pos_axis = {}
    id_to_neg_axis = {}
    for axis, q in enumerate(qubits):
        if q > 0:
            _id = abs(q)
            id_to_pos_axis[_id] = axis
        elif q < 0:
            _id = abs(q)
            id_to_neg_axis[_id] = axis

    
    pos_axes = [id_to_pos_axis[_id] for _id in sorted_qubit_ids if _id in id_to_pos_axis]
    neg_axes = [id_to_neg_axis[_id] for _id in sorted_qubit_ids if _id in id_to_neg_axis]
    remaining_axes = [i for i in range(iso_tensor.ndim) if i not in pos_axes + neg_axes]
    print("pos_axes", pos_axes)
    print("neg_axes", neg_axes)
    print("remaining_axes", remaining_axes)
    perm = neg_axes + pos_axes + remaining_axes

    # Reorder tensor legs and reshape to a matrix:
    iso_matrix = iso_tensor.permute(perm).reshape(
        2 ** len(neg_axes),
        2 ** len(pos_axes),
    )
    uni_matrix = complete_isometry_to_unitary(iso_matrix)
    uni_tensor = uni_matrix.reshape(*([2] * (2 * len(neg_axes))))

    new_pos_axes = []
    real_bid = 2*len(neg_axes)-len(pos_axes)
    complement_bid = len(neg_axes)
    for qid in sorted_qubit_ids:
        if qid in pos_ids:
            new_pos_axes.append(real_bid)
            real_bid += 1
        else:
            new_pos_axes.append(complement_bid)
            complement_bid += 1
    
    new_perm = [i for i in range(len(neg_axes))] + new_pos_axes
    print(new_pos_axes)
    print(uni_tensor.shape)
    print(new_perm)
    uni_tensor = uni_tensor.permute(new_perm)
    return uni_tensor


def convert_all_iso_to_unitary(
    iso_folder: str = "../gates/iso_tensors",
    out_folder: str = "../gates/unitary_gates",
) -> None:
    """
    Read all iso-tensors from iso_folder, apply iso_to_uni, and save the resulting
    unitary tensors to out_folder. The qubits for each tensor are taken from
    ISO_TENSOR_QUBITS defined above.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    iso_dir = os.path.join(base_dir, iso_folder)
    out_dir = os.path.join(base_dir, out_folder)

    if not os.path.exists(iso_dir):
        print(f"[convert_all_iso_to_unitary] iso_folder not found: {iso_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(iso_dir) if f.endswith(".pt"))
    if not files:
        print(f"[convert_all_iso_to_unitary] No .pt files found in {iso_dir}")
        return

    if len(files) != len(ISO_TENSOR_QUBITS):
        print(
            f"[convert_all_iso_to_unitary] Warning: number of files ({len(files)}) "
            f"!= number of qubit patterns ({len(ISO_TENSOR_QUBITS)}). "
            f"Using min of the two."
        )

    n = min(len(files), len(ISO_TENSOR_QUBITS))
    print(f"[convert_all_iso_to_unitary] Converting {n} tensors...")

    for idx in range(n):
        fname = files[idx]
        fpath = os.path.join(iso_dir, fname)
        qubits = ISO_TENSOR_QUBITS[idx]

        try:
            iso_tensor = torch.load(fpath)
        except Exception as e:
            print(f"  [{idx}] {fname}: ERROR loading tensor -> {e}")
            continue

        if not isinstance(iso_tensor, torch.Tensor):
            print(f"  [{idx}] {fname}: object is not a torch.Tensor, got {type(iso_tensor)}")
            continue

        try:
            unitary_tensor = iso_to_uni(iso_tensor, qubits)
        except Exception as e:
            print(f"  [{idx}] {fname}: ERROR in iso_to_uni -> {e}")
            continue

        out_path = os.path.join(out_dir, fname)
        try:
            torch.save(unitary_tensor, out_path)
            print(
                f"  [{idx}] {fname}: saved unitary tensor to {out_path}, "
                f"shape={tuple(unitary_tensor.shape)}"
            )
        except Exception as e:
            print(f"  [{idx}] {fname}: ERROR saving unitary tensor -> {e}")


# For each iso_tensor, the corresponding qubits configuration (order [p,l,r,d,u])
ISO_TENSOR_QUBITS = [
    [-5, 0, 0, 0, 5],
    [-3, 0, 3, 0, -6],
    [-2, -3, 0, 0, 2],
    [-8, 0, 5, -5, 8],
    [-6, -5, 5, 6, -8],
    [-1, -5, 0, -2, 5],
    [-9, 0, 8, -8, 0],
    [-7, -8, 7, 8, 0],
    [-4, -7, 0, -5, 0],
]

# ISO_TENSOR_QUBITS = [
#     [-1, 0, 0, 0, 1],
#     [-2, 0, 2, 0, -1],
#     [-3,-2,0,0,3],
#     [-4,0,1,-1,4],
#     [-5,-1,5,1,-4],
#     [-6,-5,0,-3,6],
#     [-7,0,4,-4,0],
#     [-8,-4,8,4,0],
#     [-9,-8,0,-6,0]
# ]
if __name__ == "__main__":
    convert_all_iso_to_unitary()
