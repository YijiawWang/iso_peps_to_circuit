from cqlib import TianYanPlatform
import os
import networkx as nx
import numpy as np

# 1. 连接
# TOKEN = os.getenv('TIANYAN_API_KEY', 'YOUR_KEY_HERE')
TOKEN = "NqoQ5D24VQ2ky0vucVXLzCe6ozgmOdFVUSQHtt0OxhY="
platform = TianYanPlatform(login_key=TOKEN)
target_machine = "tianyan176"
platform.set_machine(target_machine)

# 2. 下载配置
print(f"Downloading config for {target_machine}...")
config = platform.download_config(machine=target_machine)

couplers_map = config['overview']['coupler_map']
unused_couplers = config['disabledCouplers'].split(',')
adjacent_list = [(int(qubit_pairs[0].lstrip('Q')), int(qubit_pairs[1].lstrip('Q'))) for c, qubit_pairs in
                    couplers_map.items() if c not in unused_couplers]


ag = nx.Graph()
ag.add_edges_from(adjacent_list)
if not nx.is_connected(ag):
    # if ag is not fully connected, we use subgraph with the largest connected component.
    sub_ag_node_list = max(nx.connected_components(ag), key=len)
    ag = ag.subgraph(sub_ag_node_list)

ag.shortest_length = dict(nx.shortest_path_length(ag, source=None,
                                                    target=None,
                                                    weight=None,
                                                    method='dijkstra'))
ag.shortest_length_weight = ag.shortest_length
ag.shortest_path = nx.shortest_path(ag, source=None, target=None,
                                    weight=None, method='dijkstra')


def generate_qubit_mapping(qcis_str):
    mapping = {}
    line_pattern = re.compile(r'^([A-Z][A-Z0-9]*)\s+((?:Q[0-9]+\s*)+)(.*)')
    qcis = []
    for line in qcis_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('PLS ') or line.startswith('PULSE '):
            raise ValueError("PULSE/PLS not supported")
        match = line_pattern.match(line)
        if not match:
            raise ValueError(f'Invalid line format: {line}')
        gate, qubits_str, params_str = match.groups()
        qubits = []
        for q in qubits_str.split():
            i = int(q[1:])
            mapping.setdefault(i, len(mapping))
            qubits.append(f'Q{mapping[i]}')
        qcis.append(' '.join([gate, ' '.join(qubits), params_str]))
    return mapping, '\n'.join(qcis)


initial_layout = None  # or provide a dict like {0: 10, 1: 11, 2:12, ...}

if initial_layout is None:
    init_map = get_init_map(dg, ag, method_init_mapping)
    initial_layout = layout_list_to_dict(init_map)
else:
    initial_layout = {qubit_mapping[k]: v for k, v in initial_layout.items()}

np.savetxt("tianyan176_couplers.txt", np.array(adjacent_list), fmt='%d')