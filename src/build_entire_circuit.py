#!/usr/bin/env python3
"""
Build entire circuit by merging consecutive U3 gates on the same qubit.
Save the optimized circuit to gates/final_circuit folder.
"""

import os
import sys
from typing import List, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.circuit.library import UGate
import qiskit.qasm2 as qasm2
import numpy as np

# Add current directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from build_circuit_from_info import build_global_decomposed_circuit

# Euler decomposer for extracting U3 parameters from a matrix
euler_decomposer = OneQubitEulerDecomposer(basis='U3')


def merge_consecutive_u3_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Merge consecutive U3 gates on the same qubit.
    Two U3 gates can be merged if they act on the same qubit and there are no
    other operations acting on that qubit between them.
    
    Args:
        qc: Input quantum circuit
        
    Returns:
        Optimized quantum circuit with merged U3 gates
    """
    # Get all instructions
    instructions = list(qc.data)
    
    # Build optimized circuit
    qc_opt = QuantumCircuit(qc.num_qubits)
    
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        
        # Check if it's a U3 gate
        if inst.operation.name == 'u':
            # Get qubit index
            qubit = qc.qubits.index(inst.qubits[0])
            theta1, phi1, lam1 = inst.operation.params
            
            # Look ahead to find the next operation that affects this qubit
            # If it's also a U3 on the same qubit, we can merge them
            j = i + 1
            u3_gates_to_merge = [(theta1, phi1, lam1)]
            u3_indices_to_skip = [i]  # Track which instruction indices to skip
            
            while j < len(instructions):
                next_inst = instructions[j]
                
                # Get qubits affected by this instruction
                affected_qubits = [qc.qubits.index(q) for q in next_inst.qubits]
                
                if qubit in affected_qubits:
                    # This instruction affects our qubit
                    if next_inst.operation.name == 'u' and len(affected_qubits) == 1:
                        # It's a U3 gate on the same qubit, add to merge list
                        theta2, phi2, lam2 = next_inst.operation.params
                        u3_gates_to_merge.append((theta2, phi2, lam2))
                        u3_indices_to_skip.append(j)
                        j += 1
                    else:
                        # It's not a U3 or affects multiple qubits, stop merging
                        break
                else:
                    # This instruction doesn't affect our qubit, continue looking
                    j += 1
            
            # Merge all collected U3 gates
            if len(u3_gates_to_merge) > 1:
                # Start with the first gate
                theta_first, phi_first, lam_first = u3_gates_to_merge[0]
                u_first = UGate(theta_first, phi_first, lam_first)
                combined_matrix = Operator(u_first).data
                
                # Multiply remaining gates in order (each new gate applied after previous)
                # If gates are [U1, U2, U3], result is U3 @ U2 @ U1
                for theta, phi, lam in u3_gates_to_merge[1:]:
                    u_gate = UGate(theta, phi, lam)
                    op = Operator(u_gate)
                    combined_matrix = op.data @ combined_matrix
                
                # Extract U3 parameters from combined matrix
                decomp_circuit = euler_decomposer(combined_matrix)
                u3_gate = decomp_circuit.data[0].operation
                theta, phi, lam = u3_gate.params
                
                # Add merged U3 gate
                qc_opt.u(float(theta), float(phi), float(lam), qubit)
                
                # Add all instructions between the first and last merged U3 that don't affect our qubit
                # These are instructions from i+1 to j-1 that are not in u3_indices_to_skip
                for k in range(i + 1, j):
                    if k not in u3_indices_to_skip:
                        intermediate_inst = instructions[k]
                        qc_opt.append(intermediate_inst.operation, intermediate_inst.qubits, intermediate_inst.clbits)
                
                # Move to the instruction that stopped merging (or next instruction if j >= len)
                i = j
            else:
                # Single U3 gate, add as is
                qc_opt.u(float(theta1), float(phi1), float(lam1), qubit)
                i += 1
        else:
            # Not a U3 gate, add as is
            qc_opt.append(inst.operation, inst.qubits, inst.clbits)
            i += 1
    
    return qc_opt


def save_circuit_to_folder(qc: QuantumCircuit, output_dir: str):
    """
    Save circuit to the output directory in multiple formats.
    
    Args:
        qc: Quantum circuit to save
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as QASM
    qasm_path = os.path.join(output_dir, 'circuit.qasm')
    try:
        with open(qasm_path, 'w') as f:
            qasm2.dump(qc, f)
        print(f"Saved QASM to: {qasm_path}")
    except Exception as e:
        print(f"Could not save QASM: {e}")
    
    # Save as PDF
    try:
        pdf_path = os.path.join(output_dir, 'circuit_diagram.pdf')
        fig = qc.draw(output='mpl', fold=-1, scale=0.8)
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved PDF to: {pdf_path}")
    except Exception as e:
        print(f"Could not save PDF: {e}")
    
    # Save circuit info
    info_path = os.path.join(output_dir, 'circuit_info.txt')
    with open(info_path, 'w') as f:
        f.write("Optimized Circuit Information\n")
        f.write("=" * 70 + "\n")
        f.write(f"Number of qubits: {qc.num_qubits}\n")
        f.write(f"Circuit depth: {qc.depth()}\n")
        f.write(f"Total operations: {qc.size()}\n")
        f.write(f"Operation counts: {qc.count_ops()}\n")
    print(f"Saved circuit info to: {info_path}")
    
    # Save detailed gate information (info.txt format)
    gate_info_path = os.path.join(output_dir, 'info.txt')
    with open(gate_info_path, 'w') as f:
        f.write(f"num_qubits: {qc.num_qubits}\n")
        
        # Count operations
        instructions = list(qc.data)
        num_ops = len(instructions)
        f.write(f"num_ops: {num_ops}\n")
        f.write("\n")
        
        # Write each operation
        op_idx = 0
        for inst in instructions:
            if inst.operation.name == 'u':
                # U3 gate
                qubit = qc.qubits.index(inst.qubits[0])
                theta, phi, lam = inst.operation.params
                f.write(f"op{op_idx}: u3(theta={theta:.6f}, phi={phi:.6f}, lam={lam:.6f}, qubit={qubit})\n")
                op_idx += 1
            elif inst.operation.name == 'cx':
                # CNOT gate
                control = qc.qubits.index(inst.qubits[0])
                target = qc.qubits.index(inst.qubits[1])
                f.write(f"op{op_idx}: cnot[{control},{target}]\n")
                op_idx += 1
            else:
                # Other gates
                qubits = [qc.qubits.index(q) for q in inst.qubits]
                f.write(f"op{op_idx}: {inst.operation.name}{qubits}\n")
                op_idx += 1
    print(f"Saved gate information to: {gate_info_path}")


def main():
    print("Building original decomposed circuit...")
    qc_original = build_global_decomposed_circuit()
    
    print(f"\nOriginal circuit:")
    print(f"  Depth: {qc_original.depth()}")
    print(f"  Total operations: {qc_original.size()}")
    print(f"  Operation counts: {qc_original.count_ops()}")
    
    print("\nMerging consecutive U3 gates...")
    qc_optimized = merge_consecutive_u3_gates(qc_original)
    
    print(f"\nOptimized circuit:")
    print(f"  Depth: {qc_optimized.depth()}")
    print(f"  Total operations: {qc_optimized.size()}")
    print(f"  Operation counts: {qc_optimized.count_ops()}")
    
    # Verify fidelity between original and optimized circuits
    from qiskit.quantum_info import Statevector
    sv_orig = Statevector(qc_original)
    sv_opt = Statevector(qc_optimized)
    fidelity = abs(np.vdot(sv_orig.data, sv_opt.data))**2
    print(f"\nCircuit fidelity (original vs optimized): {fidelity:.10f}")
    if fidelity < 0.999999:
        print("WARNING: Optimized circuit does not match original circuit!")
        print("The optimization may have introduced errors.")
        return
    else:
        print("✓ Optimized circuit matches original circuit (fidelity = 1.0)")
    
    # Save to gates/final_circuit
    output_dir = os.path.join(BASE_DIR, "gates", "final_circuit")
    print(f"\nSaving optimized circuit to: {output_dir}")
    save_circuit_to_folder(qc_optimized, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

