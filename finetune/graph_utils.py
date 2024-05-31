from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data
import torch
from collections import deque


def bfs_paths(data, start_node, max_depth: int):
    visited = set()
    queue = deque([(start_node, 0, [])])
    paths = {}

    while queue:
        node, depth, path = queue.popleft()
        if depth <= max_depth:  
            if depth not in paths:
                paths[depth] = {}  
            if node not in visited:
                visited.add(node)
                path = path + [node]
                paths[depth][node] = path
                for neighbor in (data.edge_index[1][data.edge_index[0] == node]).tolist():
                    queue.append((neighbor, depth + 1, path))
    return visited, paths


def node2json(graph: Data, node_id: int, hops: int = 2):
    node_json = {}
    node_json['token'] = f'node_{node_id}'
    node_json['title'] = graph.x[node_id]['title']
    node_json['abstract'] = graph.x[node_id]['abstract']

    k_hop_neighbors = bfs_paths(graph, node_id, hops)[1]
    node_json['neighbors'] = k_hop_neighbors
    node_json['label'] = graph.y[node_id]
    return node_json


def get_k_hop_neighbors(center_node: int, k: int, data: Data):
    assert k > 0, 'k must be greater than 0'
    k_hop_neighbors = {}
    seen_nodes = set()
    for i in range(1, k + 1):
        subgraph = k_hop_subgraph(center_node, i, data.edge_index)
        unseen_nodes = set(subgraph[0].tolist()) - seen_nodes
        seen_nodes.update(subgraph[0].tolist())
        unseen_nodes_list = []
        for nid in unseen_nodes:
            node = {}
            node['id'] = nid
            node['attributes'] = data.x[nid]
            unseen_nodes_list.append(node)
        k_hop_neighbors[str(i)] = unseen_nodes_list
    return k_hop_neighbors
