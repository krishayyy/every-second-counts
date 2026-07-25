"""Builds the dispatch problem on a real map: simultaneous emergency calls scattered
around a real city (default: Pleasanton, CA), to be split between two ambulance crews.
Edge weight = real ground distance (km) between two calls, via the haversine formula.

Why Max-Cut: total pairwise distance across all calls is a constant S. For any
2-way split, (within-crew-A distance) + (within-crew-B distance) + (cut distance) = S.
So minimizing each crew's total internal spread (tighter, faster-to-cover routes) is
mathematically identical to maximizing the cut — which is exactly what QAOA solves.
"""
import networkx as nx
import numpy as np

PLEASANTON_CA = (37.6624, -121.8747)
EARTH_RADIUS_KM = 6371.0


def haversine_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def generate_calls(
    n_calls: int = 8,
    center: tuple[float, float] = PLEASANTON_CA,
    radius_km: float = 6.0,
    seed: int | None = None,
) -> nx.Graph:
    rng = np.random.default_rng(seed)
    graph = nx.Graph()

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center[0]))

    coords = []
    for i in range(n_calls):
        r = radius_km * np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        lat = center[0] + (r * np.sin(theta)) / km_per_deg_lat
        lon = center[1] + (r * np.cos(theta)) / km_per_deg_lon
        coords.append((float(lat), float(lon)))
        graph.add_node(i, lat=lat, lon=lon)

    for i in range(n_calls):
        for j in range(i + 1, n_calls):
            dist_km = haversine_km(coords[i], coords[j])
            graph.add_edge(i, j, weight=round(dist_km, 2))

    return graph


def build_graph_from_points(points: list[dict]) -> nx.Graph:
    """points: [{"lat": .., "lon": .., "label": .., "severity": ..}, ...]"""
    graph = nx.Graph()
    for i, p in enumerate(points):
        graph.add_node(i, lat=p["lat"], lon=p["lon"], label=p.get("label", ""), severity=p.get("severity", 3))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist_km = haversine_km((points[i]["lat"], points[i]["lon"]), (points[j]["lat"], points[j]["lon"]))
            graph.add_edge(i, j, weight=round(dist_km, 2))

    return graph
