"""
Real road-network routing via the public OSRM demo server
(router.project-osrm.org), replacing haversine-distance + flat-mph
assumptions with actual driving distances and durations.

DISCLOSED APPROXIMATION: the public OSRM demo server is documented as a
"demo, not for production" service with an informal rate limit -- fine for
a research/prototype pipeline, not something to hit at high QPS in a real
deployment (a self-hosted OSRM instance would be the production equivalent).
We batch requests and pace them to be a good citizen of the shared server.
"""
import time
import urllib.parse
import subprocess
import json

OSRM_BASE = "http://router.project-osrm.org"
MAX_COORDS_PER_REQUEST = 90  # keep well under the public server's practical limits
REQUEST_PAUSE_SEC = 0.3


def _curl_json(url: str, timeout: int = 30):
    """Use curl (not urllib) -- this machine's Python urllib hits an SSL
    cert verification error; curl uses the system cert store correctly."""
    out = subprocess.run(["curl", "-s", "--max-time", str(timeout), url], capture_output=True, text=True)
    return json.loads(out.stdout)


def osrm_table(sources_lonlat, dest_lonlat):
    """
    sources_lonlat, dest_lonlat: lists of (lon, lat) tuples.
    Returns (durations_seconds, distances_meters) as nested lists,
    shape [len(sources)][len(dest)]. Batches destinations to stay under
    the server's coordinate-count limits.
    """
    n_src = len(sources_lonlat)
    all_durations = [[] for _ in range(n_src)]
    all_distances = [[] for _ in range(n_src)]

    max_dest_per_batch = max(1, MAX_COORDS_PER_REQUEST - n_src)
    for start in range(0, len(dest_lonlat), max_dest_per_batch):
        batch = dest_lonlat[start:start + max_dest_per_batch]
        coords = sources_lonlat + batch
        coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
        src_idx = ";".join(str(i) for i in range(n_src))
        dst_idx = ";".join(str(i) for i in range(n_src, n_src + len(batch)))
        url = (f"{OSRM_BASE}/table/v1/driving/{coord_str}"
               f"?sources={src_idx}&destinations={dst_idx}&annotations=duration,distance")
        data = _curl_json(url)
        if data.get("code") != "Ok":
            raise RuntimeError(f"OSRM table request failed: {data}")
        for i in range(n_src):
            all_durations[i].extend(data["durations"][i])
            all_distances[i].extend(data["distances"][i])
        time.sleep(REQUEST_PAUSE_SEC)

    return all_durations, all_distances
