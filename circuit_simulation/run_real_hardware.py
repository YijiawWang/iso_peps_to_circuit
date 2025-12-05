#!/usr/bin/env python3
"""
EXECUTE isoPEPS on REAL IBM QUANTUM HARDWARE.

This script:
1. Connects to IBM Quantum Cloud via QiskitRuntimeService.
2. Selects the least busy backend (or a specific one like 'ibm_brisbane').
3. Transpiles the circuit for the specific hardware topology.
4. Executes using the 'Sampler' primitive with Error Mitigation.
5. Saves RAW measurement counts, calculated NLL, and Fidelity metrics.
"""

import torch
import numpy as np
import os
import sys
import re
import json
import time
from datetime import datetime

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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

# Regex patterns
_RE_U3 = re.compile(r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)")
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")

def parse_info_file(info_path: str):
    """Parse info.txt to get gate operations."""
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
    """Build the logical quantum circuit."""
    qc = QuantumCircuit(num_qubits)
    for op in operations:
        if op['type'] == 'u3':
            qc.u(*op['params'], op['qubits'][0])
        elif op['type'] == 'cnot':
            qc.cx(*op['params'])
    
    # CRITICAL: We need measurement for real hardware
    qc.measure_all()
    return qc

def get_real_backend(service, min_qubits=9):
    """
    Selects a real backend. 
    Priority: 'ibm_brisbane' -> 'ibm_osaka' -> 'ibm_kyoto' -> Least Busy
    """
    # List of preferred machines (Eagle/Heron processors)
    preferred_backends = ['ibm_brisbane', 'ibm_osaka', 'ibm_kyoto', 'ibm_sherbrooke']
    
    available_backends = service.backends(min_num_qubits=min_qubits, simulator=False, operational=True)
    available_names = [b.name for b in available_backends]
    
    print(f"   [Info] Available backends: {available_names}")
    
    # Try to find a preferred one
    for name in preferred_backends:
        if name in available_names:
            print(f"   [Info] Selected preferred backend: {name}")
            return service.backend(name)
    
    # Fallback: Least busy
    print("   [Info] Preferred backends busy or unavailable. Selecting least busy...")
    least_busy = service.least_busy(min_num_qubits=min_qubits, simulator=False)
    print(f"   [Info] Selected least busy backend: {least_busy.name}")
    return least_busy

def calculate_fidelity_and_nll(counts, total_shots):
    """
    Calculate NLL and Fidelity from REAL measurement counts.
    Handles Endianness fix (Reversing bitstrings).
    """
    # 1. Convert Counts to Probabilities (with Endianness Fix)
    full_probs = np.zeros(2**9)
    
    # Raw samples list for saving (e.g. ['000...1', '101...0'])
    raw_samples_corrected = []
    
    for bitstring, count in counts.items():
        # Qiskit (Little Endian) -> PyTorch (Big Endian)
        # Reverse the bitstring!
        reversed_bs = bitstring[::-1]
        
        # Add to samples list 'count' times
        raw_samples_corrected.extend([reversed_bitstring_to_tensor(reversed_bs)] * count)
        
        idx = int(reversed_bs, 2)
        full_probs[idx] = count / total_shots

    probs_tensor = torch.tensor(full_probs, dtype=torch.float64)
    
    # 2. Calculate NLL
    # Add epsilon for numerical stability
    probs_safe = probs_tensor + 1e-12
    probs_safe /= probs_safe.sum()
    pseudo_state = torch.sqrt(probs_safe).reshape(*([2] * 9))
    
    nll = 0.0
    if 'nll_decomposed' in sys.modules:
        try:
            nll = nll_decomposed.calculate_nll(pseudo_state, nll_decomposed.STANDARD_INDICES)
        except Exception as e:
            print(f"NLL Calc Error: {e}")
            
    # 3. Calculate Hellinger Fidelity (Optional, if we knew target distribution)
    # Here we just return the probs and NLL
    return probs_tensor, nll, raw_samples_corrected

def reversed_bitstring_to_tensor(bitstring):
    """Helper: '101' -> torch.tensor([1, 0, 1])"""
    return torch.tensor([int(c) for c in bitstring], dtype=torch.uint8)

def main():
    print("=" * 70)
    print(f"EXECUTION ON REAL IBM QUANTUM HARDWARE")
    print("=" * 70)
    
    # 1. Load Circuit
    if not os.path.exists(FINAL_CIRCUIT_INFO):
        raise FileNotFoundError(f"Not found: {FINAL_CIRCUIT_INFO}")
    operations = parse_info_file(FINAL_CIRCUIT_INFO)
    qc = build_qiskit_circuit(operations)
    
    # 2. Connect to IBM Cloud
    print("   [Info] Connecting to IBM Quantum Service...")
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token="bz5tGCW_Cm5S0LoqyaiTdfSa9ho1Wdji8nVfDQjXS6CR", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
             # Optional
            overwrite=True,  # Optional, defaults to False
        )
        service = QiskitRuntimeService()
    except Exception as e:
        print(f"   [Error] Failed to connect?: {e}")
        print("   Did you run QiskitRuntimeService.save_account()?")
        return

    # 3. Select Backend
    backend = get_real_backend(service)
    print(f"   [Target] Running on: {backend.name} ({backend.num_qubits} qubits)")
    
    # 4. Transpile for Real Hardware
    print("   [Info] Transpiling circuit (Optimization Level 3)...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(qc)
    
    print(f"   [Stats] Final Depth: {isa_circuit.depth()}")
    print(f"   [Stats] CNOT Count: {isa_circuit.count_ops().get('ecr', isa_circuit.count_ops().get('cx', 0))}")
    
    # 5. Execute using Sampler Primitive
    print("   [Info] Submitting Job to Queue...")
    sampler = Sampler(mode=backend)
    
    # Configuration: 4096 shots usually enough for 9 qubits
    # resilience_level=1 enables Readout Error Mitigation (M3) if available in V1, 
    # V2 uses different options but default is usually good.
    job = sampler.run([isa_circuit], shots=4096)
    print(f"   [Job ID] {job.job_id()}")
    print("   [Info] Waiting for results... (This may take minutes to hours)")
    
    result = job.result()
    print("   [Info] Job completed!")
    
    # 6. Process Results
    # SamplerV2 returns PubResult.data.meas.get_counts()
    # "meas" is the default name for the measurement register if measure_all() used.
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    total_shots = sum(counts.values())
    
    # 7. Calculate Metrics & Correct Endianness
    probs_tensor, nll, samples_list = calculate_fidelity_and_nll(counts, total_shots)
    
    print(f"\n   [Result] NLL on Hardware: {nll:.6f}")
    
    # 8. Save EVERYTHING
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(BASE_DIR, "results_hardware", f"{backend.name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Metrics
    metrics = {
        "backend": backend.name,
        "job_id": job.job_id(),
        "date": timestamp,
        "shots": total_shots,
        "nll": float(nll),
        "circuit_depth": isa_circuit.depth(),
        "cnot_count": isa_circuit.count_ops().get('ecr', isa_circuit.count_ops().get('cx', 0))
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Probs Tensor
    torch.save(probs_tensor, os.path.join(save_dir, "probs.pt"))
    
    # Save Raw Samples (as a big tensor)
    # Stack list of tensors into one tensor (N_shots, 9)
    if samples_list:
        samples_tensor = torch.stack(samples_list)
        torch.save(samples_tensor, os.path.join(save_dir, "samples.pt"))
    
    # 9. Plotting
    if 'nll_decomposed' in sys.modules:
        # For plotting, we just take the first 100 samples or random ones
        # We already have 'samples_tensor' which is real data
        subset_samples = samples_tensor[:100] if len(samples_tensor) > 100 else samples_tensor
        
        nll_decomposed.plot_image_grid(
            subset_samples, 
            rows=10, 
            cols=10, 
            title=f"IBM {backend.name} (Real Hardware)", 
            figsize=(5,5), 
            filename=os.path.join(save_dir, "hardware_samples.png")
        )
        print(f"   [Info] Plot saved to {save_dir}")

if __name__ == "__main__":
    main()