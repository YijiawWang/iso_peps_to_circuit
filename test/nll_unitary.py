
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

ag = [
      [4], 
      [2,5], 
      [1,2],     
      [4, 7], 
      [4,5,7],  
      [0,1,4],    
      [7,8], 
      [6,7],  
      [3,4, 6],        
  ]
# gate_order = [0,3,6,7,4,1,2,5,8]
gate_order = [8,5,2,1,4,7,6,3,0]
# ag = [
#     [5, 7, 8],
#     [2, 4, 5],
#     [1, 2],
#     [0, 1],
#     [0, 3, 4],
#     [3, 7],
#     [3, 6],
#     [0, 3],
#     [0],
# ]
# gate_order =  [9, 6, 3, 2, 5, 8, 7, 4, 1]
def load_unitary_gates():
    """
    Load all unitary gate matrices from unitary_gates folder.
    
    Returns:
        list: List of gate matrices in gate_order sequence
    """
    gates = []
    for i in gate_order:
        gate = torch.load(f'../gates/unitary_gates/tensor{i}.pt').detach()
        print(i, end=', ')
        gates.append(gate)
    return gates


def apply_gate(state, gate, qubits):
    """
    Apply a quantum gate to specific qubits in a quantum state.
    
    Args:
        state: Quantum state tensor with shape (2, 2, ..., 2) for n qubits
        gate: Gate matrix to apply
        qubits: List of target qubit indices
    
    Returns:
        Updated quantum state tensor
    """
    # qubits: list of target qubit indices
    print(gate.shape)
    print(qubits)
    original_shape = state.shape
    gate = gate.to(state.dtype)
    # Move target qubits to the first dimensions
    state = state.movedim(qubits, list(range(len(qubits))))
    state = state.reshape(2**len(qubits), -1)  # (2**num_target_qubits, rest)
    gate = gate.reshape(2**len(qubits), 2**len(qubits))
    state = gate @ state
    state = state.reshape(original_shape).movedim(list(range(len(qubits))), qubits)
    
    return state


def simulate_circuit(gates, apply_list):
    """
    Simulate a quantum circuit by applying gates sequentially.
    
    Args:
        gates: List of gate matrices
        apply_list: List of qubit indices for each gate (from gates.apply)
    
    Returns:
        Final quantum state tensor
    """
    state = torch.zeros(2**9, dtype=torch.complex128)
    state[0] = 1.0
    state = state.reshape(*([2]*9))
    
    for i, (gate, qubits) in enumerate(zip(gates, apply_list)):
        state = apply_gate(state, gate, qubits)
    
    print('apply done!')
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
        if psi.abs()**2 > 0.3:
            print(idx.tolist(), psi.abs()**2)
        nll += -torch.log(psi.abs()**2 + 1e-12)
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
    
    print(f"正在从概率分布中进行 {num_shots} 次采样...")
    # Define basis states (0, 1, 2, ..., 2^N-1)
    basis_states = np.arange(2**num_qubits)
    
    # Sample using numpy
    samples = np.random.choice(
        basis_states, 
        size=num_shots, 
        p=probs
    )
    
    # Convert to binary representation
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
    samples_to_plot = samples[:num_plots]
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1]*2, grid_shape[0]*2))
    fig.suptitle(f"First {num_plots} sampled bitstrings (3x3 grids)")
    
    for ax, sample in zip(axes.flat, samples_to_plot):
        grid = sample.reshape(3, 3)
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    # Hide unused axes if samples fewer than num_plots
    for ax in axes.flat[len(samples_to_plot):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved samples visualization to {filename}")



# Standard indices for evaluation
STANDARD_INDICES = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1]
]


if __name__ == "__main__":
    # Example: Load gates and simulate circuit
    gates = load_unitary_gates()
    print("\nGates loaded:", len(gates))
    
    # Simulate circuit
    print("\n=== Simulating circuit ===")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = simulate_circuit(gates, [ag[i] for i in gate_order])
    # state = simulate_circuit(gates, ag)
    probs = state.reshape(-1).abs()**2
    nll = calculate_nll(state, STANDARD_INDICES)
    print(f"NLL: {nll}")
    
    # Sample from distribution
    # samples = sample_from_probability(probs, num_shots=100)
    # print(f"Generated {len(samples)} samples from distribution")
    # plot_samples(samples, filename="samples_first30.png", grid_shape=(5, 6))

