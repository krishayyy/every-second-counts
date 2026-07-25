"""Converts a call split into a real time-to-clear estimate and a survival-odds
readout, grounded in public EMS response-time research (see README for sources)."""
from dataclasses import dataclass

import networkx as nx
import numpy as np

AVG_URBAN_RESPONSE_SPEED_KMH = 40.0
SURVIVAL_DROP_PER_MINUTE = 0.08  # conservative middle of the ~7-10%/min AHA range


@dataclass
class ImpactResult:
    crew_minutes: dict[int, float]
    time_to_clear_minutes: float


def _nearest_neighbor_tour_km(graph: nx.Graph, nodes: list[int]) -> float:
    if len(nodes) <= 1:
        return 0.0
    remaining = set(nodes[1:])
    current = nodes[0]
    total = 0.0
    while remaining:
        nxt = min(remaining, key=lambda n: graph[current][n]["weight"] if graph.has_edge(current, n) else np.inf)
        total += graph[current][nxt]["weight"] if graph.has_edge(current, nxt) else 0.0
        current = nxt
        remaining.remove(nxt)
    return total


def evaluate_split(
    graph: nx.Graph,
    partition: dict[int, int],
    avg_speed_kmh: float = AVG_URBAN_RESPONSE_SPEED_KMH,
) -> ImpactResult:
    crew_minutes: dict[int, float] = {}
    for crew in (0, 1):
        nodes = [n for n, c in partition.items() if c == crew]
        km = _nearest_neighbor_tour_km(graph, nodes)
        crew_minutes[crew] = round((km / avg_speed_kmh) * 60.0, 2)
    return ImpactResult(crew_minutes=crew_minutes, time_to_clear_minutes=max(crew_minutes.values()))


MAX_PLAUSIBLE_GAIN_PCT = 25.0  # cap so the estimate never overstates AHA's studied response-time range


def survival_odds_gain(minutes_saved: float, drop_per_minute: float = SURVIVAL_DROP_PER_MINUTE) -> float:
    raw = max(0.0, minutes_saved) * drop_per_minute * 100.0
    return round(min(raw, MAX_PLAUSIBLE_GAIN_PCT), 1)
