#!/usr/bin/env python3
"""
Test iso_to_uni:

For each iso_tensors/tensori.pt:
  1) Convert to unitary tensor via src.iso_to_uni.iso_to_uni.
  2) Take a slice of the resulting uni_tensor according to `INDICES[i]`.
  3) On the original iso_tensor, first squeeze all dims==1, then permute by `PERMUTES[i]`.
  4) Check the two results are (approximately) equal.
"""

import os
import sys

import torch


# Index patterns for the resulting uni_tensor
INDICES = [
    (slice(None), slice(None)),                           # 0: [:, :]
    (slice(None), slice(None), slice(None), 0),           # 1: [:, :, :, 0]
    (slice(None), slice(None), slice(None), 0),           # 2: [:, :, :, 0]
    (slice(None), slice(None), slice(None), slice(None)), # 3: [:, :, :, :]
    (slice(None), slice(None), slice(None),
     slice(None), slice(None), 0),                        # 4: [:, :, :, :, :, 0]
    (slice(None), slice(None), slice(None),
     0, 0, slice(None)),                                  # 5: [:, :, :, 0, 0, :]
    (slice(None), slice(None), slice(None), 0),           # 6: [:, :, :, 0]
    (slice(None), slice(None), slice(None), slice(None)), # 7: [:, :, :, :]
    (slice(None), slice(None), slice(None), 0, 0, 0),     # 8: [:, :, :, 0, 0, 0]
]                

# Permutations for the squeezed original iso_tensor
PERMUTES = [
    [0, 1],           # 0
    [0, 2, 1],        # 1
    [0, 1, 2],        # 2
    [2, 0, 1, 3],     # 3
    [1, 0, 4, 2, 3],  # 4
    [0, 2, 1, 3],     # 5
    [2, 0, 1],        # 6
    [0, 1, 2, 3],     # 7
    [0, 2, 1],        # 8
]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "..", "src")
    sys.path.insert(0, src_dir)

    from iso_to_uni import iso_to_uni, ISO_TENSOR_QUBITS

    iso_dir = os.path.join(base_dir, "..", "gates", "iso_tensors")

    # Files are expected to be tensor0.pt, tensor1.pt, ...
    files = [f"tensor{i}.pt" for i in range(len(INDICES))]

    print("iso_dir:", iso_dir)
    print("Testing", len(files), "tensors\n")

    for i, fname in enumerate(files):
        path = os.path.join(iso_dir, fname)
        if not os.path.exists(path):
            print(f"[{i}] {fname}: NOT FOUND, skip")
            continue

        print(f"[{i}] {fname}")
        iso_tensor = torch.load(path)
        qubits = ISO_TENSOR_QUBITS[i]

        # 1) Convert to unitary tensor
        uni_tensor = iso_to_uni(iso_tensor, qubits)

        # 2) Slice uni_tensor
        idx = INDICES[i]
        uni_slice = uni_tensor[idx]

        # 3) Squeeze original iso_tensor and permute
        iso_sq = iso_tensor.squeeze()
        perm = PERMUTES[i]
        iso_perm = iso_sq.permute(*perm)

        # 4) Compare shapes and values
        print("   uni_slice.shape:", tuple(uni_slice.shape))
        print("   iso_perm.shape :", tuple(iso_perm.shape))

        if uni_slice.shape != iso_perm.shape:
            print("   SHAPE MISMATCH")
            continue

        equal = torch.allclose(uni_slice, iso_perm, atol=1e-6, rtol=1e-6)
        print("   allclose:", equal)

    print("\nDone.")


if __name__ == "__main__":
    main()
