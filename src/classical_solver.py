"""Classical baseline: greedy Max-Cut, for comparison against the QAOA result."""
import networkx as nx

from src.qaoa_solver import cut_value


def greedy_solve(graph: nx.Graph) -> tuple[str, float]:
    n = graph.number_of_nodes()
    bits = ["0"] * n
    for node in graph.nodes:
        gain0, gain1 = 0.0, 0.0
        for neighbor in graph.neighbors(node):
            if neighbor >= node:
                continue
            w = graph[node][neighbor]["weight"]
            if bits[neighbor] == "0":
                gain1 += w
            else:
                gain0 += w
        bits[node] = "1" if gain1 > gain0 else "0"
    bitstring = "".join(bits)
    return bitstring, cut_value(graph, bitstring)
