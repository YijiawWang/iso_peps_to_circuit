#!/usr/bin/env python3
"""
EXECUTE isoPEPS on TIANYAN using NATIVE CQLIB (No Adapter).

Mechanism:
1. Parses info.txt from final_circuit_cz for 'u3' and 'cz' gates.
2. Manually decomposes 'u3(theta, phi, lam)' -> RZ(lam)-RY(theta)-RZ(phi).
3. Constructs a native cqlib.Circuit with CZ gates.
4. Submits directly via TianYanPlatform.
"""

import torch
import numpy as np
import os
import sys
import re
import json
import time
from datetime import datetime

# --- Native TianYan SDK Import ---
try:
    from cqlib import TianYanPlatform
    from cqlib.circuits import Circuit
except ImportError:
    print("Error: cqlib not found. Please run: pip install cqlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "circuit_optimization")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import nll_decomposed
except ImportError:
    pass

FINAL_CIRCUIT_INFO = os.path.join(BASE_DIR, "gates_2patterns", "final_circuit_cz", "info.txt")


MANUAL_MAPPING = {
    0: 12, 1: 7, 2: 1,
    3: 19, 4: 13, 5: 8,
    6: 25, 7: 20, 8: 14
}

def apply_mapping_to_qcis(logical_qcis: str, mapping: dict) -> str:
    """
    直接解析 QCIS 字符串，并将逻辑比特索引替换为物理比特索引。
    例如：'RY Q1 1.57' -> 'RY Q45 1.57'
    """
    mapped_lines = []
    
    # 按照空格分割，查找 Q 开头的参数并替换
    for line in logical_qcis.split('\n'):
        line = line.strip()
        if not line: continue
        
        parts = line.split()
        new_parts = []
        for part in parts:
            # 匹配 Q0, Q1, Q10 等格式
            # 注意：不区分大小写，通常 QCIS 是大写 Q
            if part.upper().startswith('Q') and part[1:].isdigit():
                logical_idx = int(part[1:])
                if logical_idx in mapping:
                    physical_idx = mapping[logical_idx]
                    new_parts.append(f"Q{physical_idx}")
                else:
                    # 如果有未映射的比特，保留原样或报错
                    new_parts.append(part)
            else:
                new_parts.append(part)
        
        mapped_lines.append(" ".join(new_parts))
        
    return "\n".join(mapped_lines)

# Regex patterns
_RE_U3 = re.compile(r"op\d+:\s*u3\(theta=([\-0-9.eE]+),\s*phi=([\-0-9.eE]+),\s*lam=([\-0-9.eE]+),\s*qubit=(\d+)\)")
_RE_CNOT = re.compile(r"op\d+:\s*cnot\[(\d+),(\d+)\]")
_RE_CZ = re.compile(r"op\d+:\s*cz\[(\d+),(\d+)\]")

def parse_info_file(info_path: str):
    operations = []
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("num_"): continue
            m_u3 = _RE_U3.match(line)
            if m_u3:
                operations.append({
                    'type': 'u3', 
                    'params': (float(m_u3.group(1)), float(m_u3.group(2)), float(m_u3.group(3))), 
                    'qubits': [int(m_u3.group(4))]
                })
                continue
            m_cx = _RE_CNOT.match(line)
            if m_cx:
                operations.append({
                    'type': 'cnot', 
                    'params': (int(m_cx.group(1)), int(m_cx.group(2))), 
                    'qubits': [int(m_cx.group(1)), int(m_cx.group(2))]
                })
                continue
            m_cz = _RE_CZ.match(line)
            if m_cz:
                operations.append({
                    'type': 'cz', 
                    'params': (int(m_cz.group(1)), int(m_cz.group(2))), 
                    'qubits': [int(m_cz.group(1)), int(m_cz.group(2))]
                })
                continue
    return operations

def build_native_cqlib_circuit(operations, num_qubits=9):
    """
    Constructs a cqlib Circuit by manually decomposing U3 gates.
    Qiskit U3(theta, phi, lam) is equivalent to: RZ(phi) * RY(theta) * RZ(lam)
    In circuit execution order (left to right): RZ(lam) -> RY(theta) -> RZ(phi)
    """
    # Initialize circuit with specific qubit indices
    qubit_indices = list(range(num_qubits))
    circuit = Circuit(qubits=qubit_indices)
    
    for op in operations:
        if op['type'] == 'cnot':
            control, target = op['params']
            circuit.cx(control, target)
        elif op['type'] == 'cz':
            q0, q1 = op['params']
            circuit.cz(q0, q1)
        elif op['type'] == 'u3':
            theta, phi, lam = op['params']
            q = op['qubits'][0]
            
            # Manual Decomposition for Tianyan Native Gates
            # Avoids 'u3 not supported' error by using standard Euler angles
            if abs(lam) > 1e-8:
                circuit.rz(q, lam)
            if abs(theta) > 1e-8:
                circuit.ry(q, theta)
            if abs(phi) > 1e-8:
                circuit.rz(q, phi)
                
    # Add measurement
    circuit.measure_all()
    return circuit

def transpile_cqlib_circuit(circuit, platform=None, target_machine=None, optimization_level=1):
    """
    Apply transpile optimization to a cqlib Circuit.
    
    Args:
        circuit: cqlib.Circuit object
        platform: TianYanPlatform instance (optional, for hardware-aware transpile)
        target_machine: Machine name (optional)
        optimization_level: Optimization level (0-3, default 1)
    
    Returns:
        Transpiled cqlib.Circuit
    """
    # Try cqlib native transpile first
    if hasattr(circuit, 'transpile'):
        try:
            if platform and target_machine:
                print(f"   [Transpile] Using cqlib native transpile (level={optimization_level})...")
                return circuit.transpile(platform=platform, machine=target_machine, 
                                       optimization_level=optimization_level)
            else:
                print(f"   [Transpile] Using cqlib native transpile (level={optimization_level})...")
                return circuit.transpile(optimization_level=optimization_level)
        except Exception as e:
            print(f"   [Warning] cqlib native transpile failed: {e}")
            print(f"   [Info] Falling back to QCIS-based transpile...")
    
    # Fallback: Try platform-level transpile
    if platform and hasattr(platform, 'transpile_circuit'):
        try:
            print(f"   [Transpile] Using platform transpile (level={optimization_level})...")
            return platform.transpile_circuit(circuit, machine=target_machine, 
                                            optimization_level=optimization_level)
        except Exception as e:
            print(f"   [Warning] Platform transpile failed: {e}")
    
    # If no native transpile available, return original circuit
    # (cqlib may handle transpile internally during submission)
    print(f"   [Info] No transpile method found, using original circuit")
    print(f"   [Info] Circuit will be optimized by platform during submission")
    return circuit

import numpy as np
import torch
import sys
from collections import Counter


def get_best_3x3_mapping():
    """
    定义一个手动映射。
    这里需要根据天衍-176 或 287 的实际拓扑图来填。
    假设我们找到了一个完美的 3x3 区域，物理编号如下（示例）：
    
    Physical Layout (Example Grid):
    12 -- 13 -- 14
    |     |     |
    22 -- 23 -- 24
    |     |     |
    32 -- 33 -- 34
    
    Logical Qubits: 0..8
    """
    # 这是一个示例映射，实际需要您查看 platform.download_config() 下来的拓扑图
    # 或者询问技术支持哪个区域最好
    mapping = {
        0: 12, 1: 7, 2: 1,
        3: 19, 4: 13, 5: 8,
        6: 25, 7: 20, 8: 14
    }
    return mapping


def calculate_fidelity_and_nll_from_cqlib_result(experiment_result, num_qubits=9):
    """
    Parse cqlib experiment result to NLL and Tensors.
    
    Args:
        experiment_result (dict): One item from the list returned by platform.query_experiment().
                                  Contains 'resultStatus', 'probability', etc.
        num_qubits (int): Total number of qubits in the circuit (default 9).
        
    Returns:
        probs_tensor, nll, raw_samples_corrected, counts
    """
    
    full_probs = np.zeros(2**num_qubits)
    raw_samples_corrected = []
    counts_for_saving = {}
    raw_data = experiment_result.get('resultStatus', [])
    
    if len(raw_data) < 2:
        raise ValueError("Invalid resultStatus: No shot data found.")

    # Header: First element is the qubit mapping, e.g., [0, 1, 2, ..., 8]
    # We assume the user measures all qubits in order for NLL calc.
    qubit_labels = raw_data[0] 
    
    # Body: The rest are shots, e.g., [[0, 1, ...], [1, 1, ...]]
    shots_data = raw_data[1:]
    total_shots = len(shots_data)
    
    # Convert list of lists [[0,1], [1,0]] to list of strings ["01", "10"]
    # Note: cqlib raw data order usually matches the 'qubit_labels' order.
    # If qubit_labels is [0, 1, ...], then result [v0, v1] means q0=v0, q1=v1.
    # This corresponds to Little Endian string "v1v0" (q0 at right) or Big Endian "v0v1".
    # To maintain consistency with 'probability' dict keys (usually Little Endian in Strings),
    # we construct the string such that q0 is at the end (Little Endian).
    
    # Let's count the occurrences
    # Convert each shot (list of ints) to a tuple for counting
    shots_as_tuples = [tuple(shot) for shot in shots_data]
    counts_obj = Counter(shots_as_tuples)
    
    for shot_tuple, count in counts_obj.items():
        # shot_tuple is (val_q0, val_q1, val_q2...) based on qubit_labels [0, 1, 2...]
        # We construct a bitstring. 
        # If we follow Qiskit standard (Little Endian string): "qn...q1q0"
        # The shot_tuple is [q0, q1, ... qn]. So we need to reverse the tuple to make the string.
        
        # Convert [1, 0, 0] (q0=1) -> "001"
        bitstring = "".join(str(bit) for bit in shot_tuple[::-1])
        
        counts_for_saving[bitstring] = count
        prob = count / total_shots
        
        # User logic: reverse bitstring again for Tensor
        # "001" [::-1] -> "100" -> Tensor [1, 0, 0]
        reversed_bs = bitstring[::-1]
        
        raw_samples_corrected.extend([reversed_bitstring_to_tensor(reversed_bs)] * count)
        
        idx = int(reversed_bs, 2)
        full_probs[idx] = prob

    # =========================================================================
    # Common: Tensor Creation & NLL
    # =========================================================================
    probs_tensor = torch.tensor(full_probs, dtype=torch.float64)
    
    # Safe Normalize
    probs_safe = probs_tensor + 1e-12
    probs_safe /= probs_safe.sum()
    
    # Calculate NLL
    nll = 0.0
    pseudo_state = torch.sqrt(probs_safe).reshape(*([2] * num_qubits))
    
    if 'nll_decomposed' in sys.modules:
        try:
            nll = nll_decomposed.calculate_nll(pseudo_state, nll_decomposed.STANDARD_INDICES)
        except Exception as e:
            # print(f"NLL Error: {e}")
            pass
            
    return probs_tensor, nll, raw_samples_corrected, counts_for_saving

def reversed_bitstring_to_tensor(bitstring):
    """Helper: '101' -> torch.tensor([1, 0, 1])"""
    return torch.tensor([int(c) for c in bitstring], dtype=torch.uint8)

def print_circuit_info(operations, circuit=None):
    """
    Print detailed circuit information including gate counts and types.
    
    Args:
        operations: List of operation dictionaries from parse_info_file
        circuit: Optional cqlib.Circuit object for additional info
    """
    print("\n" + "=" * 70)
    print("CIRCUIT INFORMATION")
    print("=" * 70)
    
    # Count gates by type
    gate_counts = {}
    for op in operations:
        gate_type = op['type']
        gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
    
    # Print gate statistics
    print(f"\nTotal operations: {len(operations)}")
    print("\nGate breakdown:")
    print("-" * 70)
    for gate_type, count in sorted(gate_counts.items()):
        percentage = (count / len(operations)) * 100
        print(f"  {gate_type.upper():8s}: {count:4d} ({percentage:5.1f}%)")
    
    # Print detailed breakdown for U3 gates (after decomposition)
    if circuit is not None:
        # Count gates in the actual circuit (after U3 decomposition)
        qcis_lines = [line.strip() for line in circuit.qcis.split('\n') if line.strip()]
        rz_count = sum(1 for line in qcis_lines if line.startswith('RZ'))
        ry_count = sum(1 for line in qcis_lines if line.startswith('RY'))
        cx_count = sum(1 for line in qcis_lines if line.startswith('CX'))
        cz_count = sum(1 for line in qcis_lines if line.startswith('CZ'))
        measure_count = sum(1 for line in qcis_lines if 'MEAS' in line.upper())
        
        print("\nAfter U3 decomposition (native gates):")
        print("-" * 70)
        if rz_count > 0:
            print(f"  RZ:      {rz_count:4d}")
        if ry_count > 0:
            print(f"  RY:      {ry_count:4d}")
        if cx_count > 0:
            print(f"  CX:      {cx_count:4d}")
        if cz_count > 0:
            print(f"  CZ:      {cz_count:4d}")
        if measure_count > 0:
            print(f"  MEASURE: {measure_count:4d}")
        
        total_native = rz_count + ry_count + cx_count + cz_count + measure_count
        print(f"\n  Total native gates: {total_native}")
    
    # Print qubit usage
    qubits_used = set()
    for op in operations:
        qubits_used.update(op['qubits'])
    print(f"\nQubits used: {sorted(qubits_used)} (total: {len(qubits_used)} qubits)")
    
    print("=" * 70 + "\n")

def main():
    print("=" * 70)
    print(f"EXECUTION ON TIANYAN (Native Cqlib)")
    print("=" * 70)
    
    # 1. Parse Info
    if not os.path.exists(FINAL_CIRCUIT_INFO):
        raise FileNotFoundError(f"Not found: {FINAL_CIRCUIT_INFO}")
    operations = parse_info_file(FINAL_CIRCUIT_INFO)
    
    # 2. Build Native Circuit
    print("   [Info] Building cqlib circuit (Decomposing U3 -> RZ, RY)...")
    cqlib_circuit = build_native_cqlib_circuit(operations)
    
    # Print circuit information
    print_circuit_info(operations, cqlib_circuit)
    
    # Optional: Print QCIS to verify it looks clean
    print(f"   [INFO] QCIS Script (before transpile):\n{cqlib_circuit.qcis[:200]}...") 

    # 3. Connect
    print("   [Info] Connecting to TianYanPlatform...")
    # 请填入您的 Key
    # TOKEN = os.getenv('TIANYAN_API_KEY', 'YOUR_LOGIN_KEY_HERE')
    TOKEN = "NqoQ5D24VQ2ky0vucVXLzCe6ozgmOdFVUSQHtt0OxhY="
    
    try:
        platform = TianYanPlatform(login_key=TOKEN)
    except Exception as e:
        print(f"   [Error] Connection failed: {e}")
        return

    # 4. Select Machine
    # 先列出可用机器
    try:
        computers = platform.query_quantum_computer_list()
        print(f"   [Info] Available computers: {[c['code'] for c in computers]}")
    except:
        pass

    # 目标机器：为了测试，先用模拟器 'tianyan_swn' 或 'tianyan_sw'
    # 如果要跑真机，改为 'tianyan-287' 或 'tianyan504'
    target_machine = "tianyan176" 
    # target_machine = "tianyan176"
    print(f"   [Target] Setting machine to: {target_machine}")
    # platform.set_machine(target_machine)
    try:
        # 设置目标机器
        platform.set_machine(target_machine)
        # [关键步骤] 下载机器的硬件配置（包含拓扑图）
        print(f"   [Info] Downloading hardware config/topology for {target_machine}...")
        # config = platform.download_config(machine=target_machine)
    except Exception as e:
        print(f"   [Error] Failed to get config for {target_machine}: {e}")
        return
    
    # 4. Apply Transpile Optimization (if available)
    # Transpile参数：optimization_level (0=无优化, 1=轻量, 2=中等, 3=激进)
    optimization_level = 1  # 可以调整为 0-3
    print(f"   [Info] Applying transpile optimization (level={optimization_level})...")
    cqlib_circuit = transpile_cqlib_circuit(
        cqlib_circuit, 
        platform=platform, 
        target_machine=target_machine,
        optimization_level=optimization_level
    )
    
    # Optional: Print QCIS after transpile
    # print(f"   [Debug] QCIS Script (after transpile):\n{cqlib_circuit.qcis[:200]}...")
    
    # 5. Apply Qubit Mapping
    compiled_circuit = apply_mapping_to_qcis(cqlib_circuit.qcis, MANUAL_MAPPING)

    # 6. Submit Experiment
    shots = 5000
    print(f"   [Info] Submitting experiment (Shots={shots})...")
    
    try:
        # submit_experiment 接受 circuit.qcis 字符串
        query_id = platform.submit_experiment(
            circuit=compiled_circuit,
            num_shots=shots
        )
        print(f"   [Query ID] {query_id}")
    except Exception as e:
        print(f"   [Error] Submission failed: {e}")
        return

    # 7. Query Results
    print("   [Info] Waiting for results (polling)...")
    try:
        # cqlib 提供了轮询等待机制
        exp_results = platform.query_experiment(query_id=query_id, max_wait_time=300, sleep_time=5)
    except Exception as e:
        print(f"   [Error] Query failed: {e}")
        return

    if not exp_results or len(exp_results) == 0:
        print("   [Error] Empty result returned.")
        return

    # 8. Process Data
    # cqlib 返回的结构是 list of dicts, key: "probability"
    result_data = exp_results[0]
    
    if 'probability' not in result_data:
        print(f"   [Error] 'probability' key not found in result. Keys: {result_data.keys()}")
        # 如果是真机，有时可能会有 'count' 或其他字段，视版本而定
        return
        
    # probs_dict = result_data['probability']
    data_dict = result_data['resultStatus']
    print(f"   [Info] Retrieved probabilities for {len(data_dict)-1} bitstrings.")
    # label_list = data_dict[0]  # First entry is qubit labels
    # samples_data = data_dict[1:]  # The rest are shots

    # 9. Metrics & Save
    # if probs_dict is None 
    probs_tensor, nll, samples_list, counts = calculate_fidelity_and_nll_from_cqlib_result(result_data, 9)
    print(f"\n   [Result] NLL on Tianyan ({target_machine}): {nll:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(BASE_DIR, "results_tianyan_native", f"{target_machine}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    metrics = {
        "backend": target_machine,
        "query_id": query_id,
        "nll": float(nll),
        "shots": shots
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    torch.save(probs_tensor, os.path.join(save_dir, "probs.pt"))
    if samples_list:
        samples_tensor = torch.stack(samples_list)
        torch.save(samples_tensor, os.path.join(save_dir, "samples.pt"))
        print(f"   [Info] Saved to {save_dir}")

if __name__ == "__main__":
    main()