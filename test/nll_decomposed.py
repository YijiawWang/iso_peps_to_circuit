import torch
import numpy as np
import random
import matplotlib.pyplot as plt


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


def load_decomposed_gates():
    """
    Load all decomposed gate matrices from ../gates/decomposed_gates.

    Returns:
        list: List of gate matrices in gate_order sequence
    """
    gates = []
    for i in gate_order:
        gate_path = f"../gates/decomposed_gates/gate_index{i}/gate_matrix.pt"
        gate = torch.load(gate_path).detach()
        print(f"{i} ", end="")
        gates.append(gate)
    print()
    return gates


def apply_gate(state, gate, qubits):
    """
    Apply a quantum gate to specific qubits in a quantum state.

    Args:
        state: Quantum state tensor with shape (2, 2, ..., 2) for n qubits
        gate: Gate tensor or matrix to apply
        qubits: List of target qubit indices

    Returns:
        Updated quantum state tensor
    """
    # Ensure complex dtype
    gate = gate.to(torch.complex128)

    original_shape = state.shape
    print(gate.shape)
    print(qubits)

    # Move target qubits to the first dimensions
    state = state.movedim(qubits, list(range(len(qubits))))
    state = state.reshape(2 ** len(qubits), -1)  # (2**num_target_qubits, rest)

    # Reshape gate to matrix if given as tensor
    gate = gate.reshape(2 ** len(qubits), 2 ** len(qubits))

    state = gate @ state
    state = state.reshape(original_shape).movedim(list(range(len(qubits))), qubits)

    return state


def simulate_circuit(gates, apply_list):
    """
    Simulate a quantum circuit by applying gates sequentially.

    Args:
        gates: List of gate matrices
        apply_list: List of qubit indices for each gate (from ag)

    Returns:
        Final quantum state tensor
    """
    state = torch.zeros(2 ** 9, dtype=torch.complex128)
    state[0] = 1.0
    state = state.reshape(*([2] * 9))

    for gate, qubits in zip(gates, apply_list):
        state = apply_gate(state, gate, qubits)

    print("apply done!")
    return state


def calculate_nll(state, indices):
    """
    Calculate negative log-likelihood (NLL) for given state and indices.

    Args:
        state: Quantum state tensor
        indices: List of index patterns to evaluate

    Returns:
        float: Average NLL
    """
    indices = torch.tensor(indices, dtype=torch.long).reshape(-1, 9)
    nll = 0.0
    for idx in indices:
        psi = state[tuple(idx.tolist())]
        # print(psi.abs()**2, torch.log(psi.abs()**2 + 1e-12))
        if psi.abs() ** 2 > 0.3:
            print(idx.tolist(), psi.abs() ** 2)
        nll += -torch.log(psi.abs() ** 2 + 1e-12)
    avg_nll = nll.item() / len(indices)
    print("NLL:", avg_nll)
    return avg_nll


def sample_from_probability(probs, num_shots=100, num_qubits=9, seed=42):
    """
    Sample bitstrings from a probability distribution.

    Args:
        probs: Probability distribution tensor
        num_shots: Number of samples to generate
        num_qubits: Number of qubits
        seed: Random seed

    Returns:
        numpy.ndarray: Array of binary samples with shape (num_shots, num_qubits)
    """
    np.random.seed(seed)
    probs = probs.numpy()
    probs = probs / np.sum(probs)

    print(f"Sampling {num_shots} shots from distribution...")
    basis_states = np.arange(2 ** num_qubits)

    samples = np.random.choice(
        basis_states,
        size=num_shots,
        p=probs,
    )

    samples_binary = [
        [int(bit) for bit in bin(s)[2:].zfill(num_qubits)]
        for s in samples
    ]
    samples_binary = np.array(samples_binary)

    return samples_binary


def plot_samples(samples, filename="samples.png", grid_shape=(5, 6)):
    """
    Plot the first N samples (reshaped to 3x3 grids) into a single PNG.
    """
    if samples.shape[1] != 9:
        raise ValueError("plot_samples currently assumes 9 qubits (3x3 grid).")
    num_plots = grid_shape[0] * grid_shape[1]
    samples = samples[:num_plots]
    samples_to_plot = []
    for sample in samples:
        sample_to_plot = []
        for i in range(9):
            sample_to_plot.append(sample[QUBITS_MAPPING[i]])
        samples_to_plot.append(np.array(sample_to_plot))

    fig, axes = plt.subplots(
        grid_shape[0],
        grid_shape[1],
        figsize=(grid_shape[1] * 2, grid_shape[0] * 2),
    )
    fig.suptitle(f"First {num_plots} sampled bitstrings (3x3 grids)")

    for ax, sample in zip(axes.flat, samples_to_plot):
        grid = sample.reshape(3, 3)
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    for ax in axes.flat[len(samples_to_plot):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved samples visualization to {filename}")


def plot_image_grid(image_list: np.array,
                    rows: int=6,
                    cols: int=5,
                    figsize: tuple=(5, 7),
                    title: dict = None
                    ):
    """
    plot image grid
    """
    samples = image_list[:cols * rows]
    samples_to_plot = []
    for sample in samples:
        sample_to_plot = []
        for i in range(9):
            sample_to_plot.append(sample[QUBITS_MAPPING[i]])
        samples_to_plot.append(np.array(sample_to_plot).reshape(3, 3))
    assert len(samples_to_plot) > 0, "Received empty image list."
    if title is None:
        title = {}
        title['text'] = f"{len(samples_to_plot)} base image({rows}x{cols} grid)"
        title['fontsize'] = 12
        title['y'] = 0.99
    if len(title) == 0:
        title = {}
        title['text'] = ""
        title['fontsize'] = 12
        title['y'] = 0.99
    if type(title) is str:
        temp = title
        title = {}
        title['text'] = temp
        title['fontsize'] = 12
        title['y'] = 0.99
    fig, axes = plt.subplots(rows, cols, figsize=figsize) 
    fig.suptitle(title['text'], fontsize=title['fontsize'], y=title['y']) 
    for index, single_image_data in enumerate(samples_to_plot):
        if index >= rows * cols: 
            break
        row_idx = index // cols
        col_idx = index % cols       
        ax = axes[row_idx, col_idx]
        img_array = np.array(single_image_data)
        ax.imshow(img_array, cmap='binary', vmin=0, vmax=1) 
        ax.set_xticks([])  
        ax.set_yticks([])  
    for i in range(len(samples_to_plot), rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if rows > 1 or cols > 1 : 
            fig.delaxes(axes[row_idx,col_idx])
        else: 
            fig.delaxes(axes[i])
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
    plt.savefig(f'samples.png', dpi=300)
    plt.show()

# Same STANDARD_INDICES as in nll_unitary.py
STANDARD_INDICES = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1],
]

QUBITS_MAPPING = {
    0: 4,
    1: 2,
    2: 1,
    3: 7,
    4: 5,
    5: 0,
    6: 8,
    7: 6,
    8: 3,
}
if __name__ == "__main__":
    # Load decomposed gates and simulate circuit
    gates = load_decomposed_gates()
    print("\nGates loaded:", len(gates))

    print("\n=== Simulating circuit with decomposed gates ===")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    state = simulate_circuit(gates, [ag[i] for i in gate_order])
    probs = state.reshape(-1).abs() ** 2

    nll = calculate_nll(state, STANDARD_INDICES)
    print(f"NLL: {nll}")

    # Optional sampling / plotting:
    samples = sample_from_probability(probs, num_shots=100)
    print(f"Generated {len(samples)} samples from distribution")
    plot_image_grid(
    samples,
    rows = 10,
    cols = 10,
    title='',
    figsize=(5, 5)
    )


