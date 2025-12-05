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
    """
    Extracts AVERAGE noise parameters from FakeBrisbane and builds
    a comprehensive noise model including T1/T2 relaxation.
    """
    # 尝试获取 FakeBrisbane，如果没有则使用通用噪声
    if FakeBrisbane is None:
        print("   [Warn] FakeBrisbane not found, using generic noise.")
        noise_model = NoiseModel()
        # 通用参数估算
        error_1q = depolarizing_error(0.0002, 1)
        error_2q = depolarizing_error(0.008, 2)
        # 这里为了简单，通用模式下只加去极化，或者你可以手动加通用的热弛豫
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'sx', 'x', 'rz', 'id'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'ecr'])
        return noise_model

    print("   [Info] Extracting calibration data from FakeBrisbane...")
    backend = FakeBrisbane()
    props = backend.properties()
    
    # 1. Calculate Average T1 and T2
    t1s = [props.t1(i) for i in range(backend.num_qubits) if props.t1(i)]
    t2s = [props.t2(i) for i in range(backend.num_qubits) if props.t2(i)]
    
    if not t1s: 
        avg_t1, avg_t2 = 280e-6, 150e-6 # fallback
    else:
        avg_t1 = np.mean(t1s)
        avg_t2 = np.mean(t2s)
        
    # [Safety Check] T2 cannot be greater than 2*T1 in physics
    if avg_t2 > 2 * avg_t1:
        avg_t2 = 2 * avg_t1
    
    # 2. Calculate Average Gate Errors (Depolarizing component)
    # SX gate (single qubit)
    sx_errs = []
    for i in range(backend.num_qubits):
        try:
            err = props.gate_error('sx', [i])
            if err: sx_errs.append(err)
        except: pass
    avg_1q_err = np.mean(sx_errs) if sx_errs else 0.0002

    # ECR/CNOT gate (two qubit)
    cx_errs = []
    for gate in props.gates:
        if gate.gate == 'ecr' or gate.gate == 'cx':
            cx_errs.append(gate.parameters[0].value)
    avg_2q_err = np.mean(cx_errs) if cx_errs else 0.008
    
    print(f"   [Brisbane Stats] Avg T1: {avg_t1*1e6:.2f} us")
    print(f"   [Brisbane Stats] Avg T2: {avg_t2*1e6:.2f} us")
    print(f"   [Brisbane Stats] Avg 1-qubit Gate Err: {avg_1q_err:.4%}")
    print(f"   [Brisbane Stats] Avg 2-qubit Gate Err: {avg_2q_err:.4%}")

    # 3. Define Gate Times (Crucial for Thermal Relaxation)
    # Based on typical IBM Eagle processor specs
    time_1q = 35e-9   # 35 ns for single qubit gates
    time_2q = 300e-9  # 300 ns for two qubit gates (ECR/CX)

    # 4. Build Noise Model
    noise_model = NoiseModel()
    
    # --- Single Qubit Errors ---
    # A. Depolarizing Error (Gate imperfection)
    depol_1q = depolarizing_error(avg_1q_err, 1)
    
    # B. Thermal Relaxation Error (Decay over time)
    # thermal_relaxation_error(t1, t2, time)
    therm_1q = thermal_relaxation_error(avg_t1, avg_t2, time_1q)
    
    # C. Combine (Compose)
    # Error_Total = Depolarizing * Thermal
    total_error_1q = depol_1q.compose(therm_1q)
    
    # Add to model for all 1-qubit gates
    noise_model.add_all_qubit_quantum_error(total_error_1q, ['u1', 'u2', 'u3', 'sx', 'x', 'rz', 'id'])
    
    # --- Two Qubit Errors ---
    # A. Depolarizing Error
    depol_2q = depolarizing_error(avg_2q_err, 2)
    
    # B. Thermal Relaxation Error
    # For a 2-qubit gate, relaxation happens on BOTH qubits independently.
    # We create a 1-qubit thermal error for the duration of the 2-qubit gate...
    therm_2q_single = thermal_relaxation_error(avg_t1, avg_t2, time_2q)
    # ...and then expand it to a tensor product (Error on Q1 ⊗ Error on Q2)
    therm_2q = therm_2q_single.expand(therm_2q_single)
    
    # C. Combine
    total_error_2q = depol_2q.compose(therm_2q)
    
    # Add to model for all 2-qubit gates
    noise_model.add_all_qubit_quantum_error(total_error_2q, ['cx', 'ecr'])
    
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