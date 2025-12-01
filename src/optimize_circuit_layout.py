import os
import sys

backend = 'gpu'

import sys
import random
import numpy as np
from scipy.sparse.linalg import LinearOperator
import scipy as sp
from itertools import combinations
from functools import partial
import time, math
from pyscipopt import Model
import pyscipopt
from pyomo.environ import *

def build_tensor_bonds_dict(Lx, Ly):
    tensor_bonds = {}
    for y in range(Ly):
        for x in range(Lx):
            tensor_bonds[(x, y)] = [x + Lx*y]
            
    temp_bond_id = Lx*Ly 

    for x in range(Lx):
        tensor_bonds[(x, 0)].append(temp_bond_id)
        temp_bond_id += 1
    for y in range(1, Ly - 1):
        for x in range(Lx):
            tensor_bonds[(x, y)].append(temp_bond_id - Lx)
            tensor_bonds[(x, y)].append(temp_bond_id)
            temp_bond_id += 1
    for x in range(Lx):
        tensor_bonds[(x, Ly - 1)].append(temp_bond_id - Lx + x)


    for y in range(Ly):
        tensor_bonds[(0, y)].append(temp_bond_id)
        temp_bond_id += 1
        for x in range(1, Lx - 1):
            tensor_bonds[(x, y)].append(temp_bond_id - 1)
            tensor_bonds[(x, y)].append(temp_bond_id)
            temp_bond_id += 1
        tensor_bonds[(Lx - 1, y)].append(temp_bond_id - 1)

    return tensor_bonds


def build_tensor_channels_dict(Lx, Ly, tensor_bonds, mannul_chis):
    chis = [1]*(Lx*Ly) + [int(np.log2(chi)) for sublist in mannul_chis[0] for chi in sublist] + [int(np.log2(chi)) for sublist in mannul_chis[1] for chi in sublist]
    bond_channel_range = {}
    current = 0
    for bond_id, chi in enumerate(chis): 
        bond_channel_range[bond_id] = list(range(current, current + chi))
        current += chi
    tensor_channels = {}
    for tensor_coord, bond_list in tensor_bonds.items():
        channels = []
        for bond_id in bond_list:
            channels.extend(bond_channel_range[bond_id])
        tensor_channels[tensor_coord] = channels

    return tensor_channels, current

def build_tensor_bonds_dict_alt_direction(Lx, Ly):
    tensor_bonds_in = {}
    tensor_bonds_out = {}
    for y in range(Ly):
        for x in range(Lx):
            tensor_bonds_in[(x, y)] = [x + Lx*y]
            tensor_bonds_out[(x, y)] = []
            
    temp_bond_id = Lx*Ly 

    for x in range(Lx):
        if x % 2 == 0:
            tensor_bonds_out[(x, 0)].append(temp_bond_id)
        else:
            tensor_bonds_in[(x, 0)].append(temp_bond_id)
        temp_bond_id += 1
    for y in range(1, Ly - 1):
        for x in range(Lx):
            if x % 2 == 0:
                tensor_bonds_in[(x, y)].append(temp_bond_id - Lx)
                tensor_bonds_out[(x, y)].append(temp_bond_id)
            else:
                tensor_bonds_out[(x, y)].append(temp_bond_id - Lx)
                tensor_bonds_in[(x, y)].append(temp_bond_id)
            temp_bond_id += 1
    for x in range(Lx):
        if x % 2 == 0:
            tensor_bonds_in[(x, Ly - 1)].append(temp_bond_id - Lx + x)
        else:
            tensor_bonds_out[(x, Ly - 1)].append(temp_bond_id - Lx + x)


    for y in range(Ly):
        tensor_bonds_in[(0, y)].append(temp_bond_id)
        temp_bond_id += 1
        for x in range(1, Lx - 1):
            tensor_bonds_out[(x, y)].append(temp_bond_id - 1)
            tensor_bonds_in[(x, y)].append(temp_bond_id)
            temp_bond_id += 1
        tensor_bonds_out[(Lx - 1, y)].append(temp_bond_id - 1)

    return tensor_bonds_in, tensor_bonds_out


def build_tensor_channels_dict_alt_direction(Lx, Ly, tensor_bonds_in, tensor_bonds_out, mannul_chis):
    chis = [1]*(Lx*Ly) + [int(np.log2(chi)) for sublist in mannul_chis[0] for chi in sublist] + [int(np.log2(chi)) for sublist in mannul_chis[1] for chi in sublist]
    print("chis:")
    print(chis)
    bond_channel_range = {}
    current = 0
    for bond_id, chi in enumerate(chis): 
        bond_channel_range[bond_id] = list(range(current, current + chi))
        current += chi

    original_channel_num = current
    tensor_channels_in = {}
    tensor_channels_out = {}
    for tensor_coord, bond_list in tensor_bonds_in.items():
        channels = []
        for bond_id in bond_list:
            channels.extend(bond_channel_range[bond_id])
        tensor_channels_in[tensor_coord] = channels

    for tensor_coord, bond_list in tensor_bonds_out.items():
        channels = []
        for bond_id in bond_list:
            channels.extend(bond_channel_range[bond_id])
        tensor_channels_out[tensor_coord] = channels

    out_channel_id = 0
    for tensor_coord, bond_list in tensor_bonds_in.items():
        local_out_channels_num = len(tensor_channels_in[tensor_coord]) - len(tensor_channels_out[tensor_coord])
        if local_out_channels_num > 0:
            for i in range(local_out_channels_num):
                tensor_channels_out[tensor_coord].append(original_channel_num + out_channel_id)
                out_channel_id += 1
    out_channel_num = out_channel_id 
    return tensor_channels_in, tensor_channels_out, original_channel_num, out_channel_num
    

def build_distance_dict_on_lattice(Lx, Ly):
    N = Lx * Ly
    distance_dict = {}
    for node1 in range(N):
        for node2 in range(N):
            x1 = node1 % Lx
            y1 = node1 // Lx
            x2 = node2 % Lx
            y2 = node2 // Lx
            distance_dict[(node1,node2)] = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    return distance_dict


def maximize_localization(tensor_channels_in, tensor_channels_out, channel_num, Lx, Ly):
    N = Lx * Ly
    model = ConcreteModel()

    # Variable definition: vars[i, j], where i is the channel index and j is the qubit index
    model.I = RangeSet(0, channel_num-1)
    model.J = RangeSet(0, N-1)
    model.vars = Var(model.I, model.J, domain=Binary)

    # Constraint 1: each channel is used by exactly one qubit
    def channel_unique_rule(m, i):
        return sum(m.vars[i, j] for j in m.J) == 1
    model.channel_unique = Constraint(model.I, rule=channel_unique_rule)

    # Constraint 2: each qubit occupies exactly one branch (one channel)
    def qubit_unique_rule(m, j):
        return sum(m.vars[i, j] for i in range(N)) == 1
    model.qubit_unique = Constraint(model.J, rule=qubit_unique_rule)

    # Constraint 3: for each tensor (gate), the in_channels and out_channels
    # must correspond to the same set of qubits
    model.gate_match = ConstraintList()
    for tensor_coord in tensor_channels_in:
        in_channels = tensor_channels_in[tensor_coord]
        out_channels = tensor_channels_out[tensor_coord]
        for j in range(N):
            model.gate_match.add(
                sum(model.vars[in_channels[i], j] for i in range(len(in_channels))) ==
                sum(model.vars[out_channels[i], j] for i in range(len(out_channels)))
            )

    # Objective function
    distance_dict = build_distance_dict_on_lattice(Lx, Ly)
    expr = 0
    for tensor_coord in tensor_channels_in:
        in_channels = tensor_channels_in[tensor_coord]
        for i1 in range(len(in_channels)):
            for i2 in range(i1+1,len(in_channels)):
                for j1 in range(N):
                    for j2 in range(N):
                        expr += model.vars[in_channels[i1], j1] * model.vars[in_channels[i2], j2] * distance_dict[(j1,j2)]

    model.obj = Objective(expr=expr, sense=minimize)
    

    # Solve the optimization problem
    import os
    import subprocess
    
    # Find the actual SCIP executable path
    possible_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'scip'),
        '/data1/wangyijia/miniconda3/envs/tns/bin/scip',
        os.path.expanduser('~/miniconda3/envs/tns/bin/scip'),
    ]
    
    scip_path = None
    scip_dir = None
    for path in possible_paths:
        if path and os.path.exists(path):
            # Verify this is a real SCIP executable (not a Python wrapper script)
            try:
                result = subprocess.run([path, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if 'SCIP version' in result.stdout or 'SCIP version' in result.stderr:
                    scip_path = path
                    scip_dir = os.path.dirname(path)
                    break
            except:
                continue
    
    if not scip_path:
        raise RuntimeError("Could not find SCIP executable. Please ensure SCIP is installed in the conda environment.")
    
    # Modify PATH so that the directory containing SCIP is placed first,
    # so that Pyomo will find the real SCIP binary with highest priority
    if scip_dir:
        current_path = os.environ.get('PATH', '')
        # Remove possible paths that contain SCIP wrapper scripts
        path_parts = [p for p in current_path.split(':') if p and 'tns/tns/bin' not in p]
        # Put SCIP directory at the beginning of PATH
        os.environ['PATH'] = scip_dir + ':' + ':'.join(path_parts)
    
    solver = SolverFactory('scip')
    result = solver.solve(model, tee=True)

    for i in range(channel_num):
        for j in range(N):
            print(i," ", j, ":", int(value(model.vars[i, j])))
    
    # Build a `gates` dictionary: for each qubit j, store all channels i assigned to it
    gates = {}
    for j in range(N):
        gates[j] = []
        for i in range(channel_num):
            if int(value(model.vars[i, j])) != 0:
                gates[j].append(i)
    
    print("\ngates:")
    print(gates)
    print("Optimal value:", value(model.obj))
    print(16*6**0.5)

vertical_chis = [[2,2,2],[2,2,2]]
horizontal_chis = [[2,1],[2,2],[2,2]]
mannul_chis = [vertical_chis, horizontal_chis]

Lx = 3
Ly = 3
tensor_bonds_in, tensor_bonds_out = build_tensor_bonds_dict_alt_direction(Lx, Ly)
tensor_channels_in, tensor_channels_out, original_channel_num, out_channel_num = build_tensor_channels_dict_alt_direction(Lx, Ly, tensor_bonds_in, tensor_bonds_out, mannul_chis)
maximize_localization(tensor_channels_in, tensor_channels_out, original_channel_num + out_channel_num, Lx, Ly)
