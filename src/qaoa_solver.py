"""QAOA Max-Cut solver: splits delivery stops into two truck zones minimizing route overlap."""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import minimize


@dataclass
class QAOAResult:
    reps: int
    optimal_params: np.ndarray
    cost_history: list[float]
    best_bitstring: str
    best_cut_value: float
    optimal_cut_value: float
    partition: dict[int, int]
    shot_counts: dict[str, int] = field(default_factory=dict)
    circuit_depth: int = 0
    prob_snapshots: list[dict[str, float]] = field(default_factory=list)
    ansatz: object = None


def _cost_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    n = graph.number_of_nodes()
    pauli_list = []
    for i, j, data in graph.edges(data=True):
        w = data["weight"]
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        label[n - 1 - j] = "Z"
        pauli_list.append(("".join(label), w))
    return SparsePauliOp.from_list(pauli_list)


def cut_value(graph: nx.Graph, bits: str) -> float:
    """bits[i] is node i's assignment (0/1), same convention used everywhere here."""
    total = 0.0
    for i, j, data in graph.edges(data=True):
        if bits[i] != bits[j]:
            total += data["weight"]
    return total


def brute_force_optimum(graph: nx.Graph) -> tuple[str, float]:
    n = graph.number_of_nodes()
    best_bits, best_val = "0" * n, -1.0
    for x in range(2**n):
        bits = format(x, f"0{n}b")
        val = cut_value(graph, bits)
        if val > best_val:
            best_bits, best_val = bits, val
    return best_bits, best_val


def run_qaoa(
    graph: nx.Graph,
    reps: int = 2,
    maxiter: int = 80,
    shots: int = 2048,
    seed: int = 42,
) -> QAOAResult:
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    cost_op = _cost_hamiltonian(graph)

    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
    ansatz.measure_all()
    unmeasured = ansatz.remove_final_measurements(inplace=False)

    cost_history: list[float] = []
    prob_snapshots: list[dict[str, float]] = []
    snapshot_every = max(1, maxiter // 8)

    def objective(params: np.ndarray) -> float:
        bound = unmeasured.assign_parameters(params)
        sv = Statevector.from_instruction(bound)
        value = sv.expectation_value(cost_op).real
        cost_history.append(value)
        if len(cost_history) % snapshot_every == 0:
            raw_probs = sv.probabilities_dict()
            prob_snapshots.append({k[::-1]: v for k, v in raw_probs.items()})
        return value

    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)
    result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
    best_params = result.x

    final_probs = Statevector.from_instruction(unmeasured.assign_parameters(best_params)).probabilities_dict()
    prob_snapshots.append({k[::-1]: v for k, v in final_probs.items()})

    backend = AerSimulator()
    measured = ansatz.assign_parameters(best_params)
    transpiled = transpile(measured, backend, optimization_level=1)
    job = backend.run(transpiled, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    # qiskit bitstrings are little-endian (qubit 0 = rightmost char); flip so
    # bits[i] directly indexes node i everywhere in this codebase.
    shot_counts = {k[::-1]: v for k, v in counts.items()}

    best_bitstring = max(shot_counts, key=lambda b: cut_value(graph, b))
    best_cut = cut_value(graph, best_bitstring)
    _, optimal_cut = brute_force_optimum(graph)

    partition = {node: int(best_bitstring[node]) for node in graph.nodes}

    return QAOAResult(
        reps=reps,
        optimal_params=best_params,
        cost_history=cost_history,
        best_bitstring=best_bitstring,
        best_cut_value=best_cut,
        optimal_cut_value=optimal_cut,
        partition=partition,
        shot_counts=shot_counts,
        circuit_depth=transpiled.depth(),
        prob_snapshots=prob_snapshots,
        ansatz=ansatz,
    )
