#!/usr/bin/env python3
"""
Simulate isoPEPS with IBM Brisbane Noise + Endianness Fix.
"""

import torch
import numpy as np
import os
import sys
import re

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
try:
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeBrisbane
    except ImportError:
        # Fallback if fake provider not found, use generic noise
        FakeBrisbane = None 

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "circuit_optimization")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import nll_decomposed
except ImportError:
    print("Warning: nll_decomposed module not found.")

FINAL_CIRCUIT_INFO = os.path.join(BASE_DIR, "gates", "final_circuit", "info.txt")

_RE_U3 = re.compile(r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)")
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")

def parse_info_file(info_path: str):
    operations = []
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("num_"): continue
            m_u3 = _RE_U3.match(line)
            if m_u3:
                operations.append({'type': 'u3', 'params': (float(m_u3.group(1)), float(m_u3.group(2)), float(m_u3.group(3))), 'qubits': [int(m_u3.group(4))]})
            m_cx = _RE_CNOT.match(line)
            if m_cx:
                operations.append({'type': 'cnot', 'params': (int(m_cx.group(1)), int(m_cx.group(2))), 'qubits': [int(m_cx.group(1)), int(m_cx.group(2))]})
    return operations

def build_qiskit_circuit(operations, num_qubits=9):
    qc = QuantumCircuit(num_qubits)
    for op in operations:
        if op['type'] == 'u3':
            qc.u(*op['params'], op['qubits'][0])
        elif op['type'] == 'cnot':
            qc.cx(*op['params'])
    qc.measure_all()
    return qc

def create_brisbane_noise_model():
    """Extract average noise parameters."""
    if FakeBrisbane is None:
        print("   [Warn] FakeBrisbane not found, using generic noise.")
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['u1', 'u2', 'u3', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.08, 2), ['cx'])
        return noise_model

    print("   [Info] Extracting calibration data from FakeBrisbane...")
    backend = FakeBrisbane()
    props = backend.properties()
    
    # Calculate Averages (Simplified)
    t1s = [props.t1(i) for i in range(backend.num_qubits) if props.t1(i)]
    t2s = [props.t2(i) for i in range(backend.num_qubits) if props.t2(i)]
    avg_t1 = np.mean(t1s) if t1s else 300e-6
    avg_t2 = np.mean(t2s) if t2s else 300e-6
    
    # Estimate Gate Errors
    cx_errs = [g.parameters[0].value for g in props.gates if g.gate == 'ecr']
    avg_2q_err = np.mean(cx_errs) if cx_errs else 0.008
    avg_1q_err = avg_2q_err / 10.0 # Heuristic

    print(f"   [Stats] T1: {avg_t1*1e6:.1f}us, CNOT Err: {avg_2q_err:.2%}")

    noise_model = NoiseModel()
    error_1q = depolarizing_error(avg_1q_err, 1)
    error_2q = depolarizing_error(avg_2q_err, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'ecr'])
    
    return noise_model

def simulate_circuit_logic(qc, noise_model=None):
    """
    Simulate with bitstring reversal to fix Endianness.
    """
    sim = AerSimulator(noise_model=noise_model)
    # Transpile (optimization_level=3 helps reduce depth even for ideal sim)
    tqc = transpile(qc, sim, optimization_level=3)
    
    shots = 10000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    
    full_probs = np.zeros(2**9)
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # === CRITICAL FIX: REVERSE BITSTRING ===
        # Qiskit: "q8 q7 ... q0" (q0 is last char)
        # PyTorch Tensor: Index 0 corresponds to q0.
        # So we need q0 to be the Most Significant Bit (first char).
        # Reversing the string makes q0 the first char.
        reversed_bitstring = bitstring[::-1]
        
        idx = int(reversed_bitstring, 2)
        full_probs[idx] = count / total_shots
        
    return torch.tensor(full_probs, dtype=torch.float64)

def main():
    if not os.path.exists(FINAL_CIRCUIT_INFO):
        raise FileNotFoundError(f"Not found: {FINAL_CIRCUIT_INFO}")
    
    operations = parse_info_file(FINAL_CIRCUIT_INFO)
    qc = build_qiskit_circuit(operations)

    # 1. Sanity Check: Noiseless Simulation
    print("\n" + "="*50)
    print("STEP 1: Noiseless Verification (Checking Logic)")
    print("="*50)
    probs_clean = simulate_circuit_logic(qc, noise_model=None)
    
    # Check top 3 probabilities
    top_indices = torch.argsort(probs_clean, descending=True)[:5]
    print("Top 5 States (Noiseless):")
    for idx in top_indices:
        # Convert idx back to bitstring for readability (Big Endian)
        bits = format(idx.item(), '09b')
        print(f"  State |{bits}> : p={probs_clean[idx]:.4f}")
        
    # 2. Realistic Simulation
    print("\n" + "="*50)
    print("STEP 2: Realistic Noise Simulation (Brisbane)")
    print("="*50)
    noise_model = create_brisbane_noise_model()
    probs_noisy = simulate_circuit_logic(qc, noise_model=noise_model)
    
    # Calculate NLL
    pseudo_state = torch.sqrt(probs_noisy).reshape(*([2] * 9))
    if 'nll_decomposed' in sys.modules:
        try:
            # Add epsilon and normalize
            probs_noisy_safe = probs_noisy + 1e-12
            probs_noisy_safe /= probs_noisy_safe.sum()
            pseudo_state_safe = torch.sqrt(probs_noisy_safe).reshape(*([2] * 9))
            
            nll = nll_decomposed.calculate_nll(pseudo_state_safe, nll_decomposed.STANDARD_INDICES)
            print(f"\nFinal NLL (Noisy): {nll:.4f}")
            
            # Save Samples
            samples = nll_decomposed.sample_from_probability(probs_noisy, 5000, 9)
            save_path = os.path.join(BASE_DIR, "samples", "brisbane_corrected.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            nll_decomposed.plot_image_grid(
                samples, rows=10, cols=10, title="IBM Brisbane (Corrected)", 
                figsize=(5,5), filename=save_path
            )
            print(f"Saved corrected samples to {save_path}")
            
        except Exception as e:
            print(f"NLL Error: {e}")

if __name__ == "__main__":
    main()