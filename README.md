## iso\_peps\_to\_circuit 

This directory contains the code that converts an iso-PEPS tensor network on a lattice into a quantum circuit, and then simulates / analyzes that circuit in Qiskit.

### Basic workflow


1. **Convert iso-PEPS tensors to unitary gates:**

   ```bash
   cd src
   python3 iso_to_uni.py          # writes tensors to ../gates/unitary_gates/
   ```

2. **Optimize per-gate U3+CX decompositions:**

   ```bash
   cd src
   python3 optimize_unitary_gates.py   # writes ../gates/decomposed_gates/gate_index{i}/
   ```

3. **Simulate the decomposed circuit, compute the nll and sample:**

   ```bash
   cd test
   python3 nll_decomposed.py
   # prints NLL and saves images to samples.png
   ```

4. **Build the exact decomposed circuit in Qiskit and inspect depth:**

   ```bash
   cd src
   python3 construct_qiskit_circuit.py
   # prints depth, total ops, and per-gate counts
   ```

### Dependencies

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Matplotlib
- Qiskit (core) and `qiskit-aer`
- Pyomo + SCIP (only required for `optimize_circuit_layout.py`)



