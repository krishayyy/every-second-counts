"""
Precompute real OSRM driving durations/distances:
  1. zone-to-zone (20x20) -- used for the MEXCLP coverage standard and for
     matching idle units to compliance-table posts.
  2. zone-to-call (20 x n_calls) -- used for ambulance-to-call dispatch
     distance/time. Idle ambulances in this model always sit at one of the
     20 candidate zones (home base, or a compliance-table repositioning
     target), so this covers the large majority of dispatch lookups.

APPROXIMATION (disclosed): for the minority case where a unit is "busy and
queued" (no idle unit available, dispatch whichever frees soonest) and its
position is a raw call location rather than a zone center, we approximate
its road distance to the new call by snapping to the nearest zone center
and using that zone's row in the precomputed table, rather than querying
OSRM for every arbitrary point pair (which would be 14,070^2 queries).
This approximation is applied identically to both the static and dynamic
strategies, so it does not bias the comparison between them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from osrm_routing import osrm_table

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "seattle" / "seattle_911_raw.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
K_ZONES = 20
SIM_DAYS = 60

EMS_TYPES_CONTAINS = [
    "Aid Response", "Medic Response", "Low Acuity", "Triaged Incident",
    "Nurseline", "MVI", "Automatic Medical Alarm",
]


def load_calls() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    pattern = "|".join(EMS_TYPES_CONTAINS)
    df = df[df["type"].str.contains(pattern, case=False, na=False)].copy()
    df = df.dropna(subset=["latitude", "longitude", "datetime"])
    df = df[df["latitude"].between(47.4, 47.8) & df["longitude"].between(-122.5, -122.2)]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    end = df["datetime"].max()
    start = end - pd.Timedelta(days=SIM_DAYS)
    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)].reset_index(drop=True)
    return df


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calls = load_calls()
    print(f"Loaded {len(calls):,} real calls for routing precompute")

    zone_model = KMeans(n_clusters=K_ZONES, n_init=10, random_state=42)
    zone_model.fit(calls[["latitude", "longitude"]])
    zone_centers = zone_model.cluster_centers_  # [lat, lng]

    zone_lonlat = [(z[1], z[0]) for z in zone_centers]

    print("Fetching zone-to-zone matrix (20x20)...")
    zz_dur, zz_dist = osrm_table(zone_lonlat, zone_lonlat)
    np.save(OUT_DIR / "osrm_zone_zone_duration_sec.npy", np.array(zz_dur))
    np.save(OUT_DIR / "osrm_zone_zone_distance_m.npy", np.array(zz_dist))
    print("  done.")

    call_lonlat = list(zip(calls["longitude"].values, calls["latitude"].values))
    print(f"Fetching zone-to-call matrix (20 x {len(call_lonlat):,})... this takes a few minutes")
    zc_dur, zc_dist = osrm_table(zone_lonlat, call_lonlat)
    np.save(OUT_DIR / "osrm_zone_call_duration_sec.npy", np.array(zc_dur))
    np.save(OUT_DIR / "osrm_zone_call_distance_m.npy", np.array(zc_dist))
    print("  done.")

    meta = {
        "k_zones": K_ZONES,
        "n_calls": len(calls),
        "zone_centers": [{"lat": float(z[0]), "lng": float(z[1])} for z in zone_centers],
        "source": "OSRM public demo server (router.project-osrm.org), driving profile",
    }
    with open(OUT_DIR / "osrm_routing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nWrote:")
    print(f"  {OUT_DIR / 'osrm_zone_zone_duration_sec.npy'}  shape {np.array(zz_dur).shape}")
    print(f"  {OUT_DIR / 'osrm_zone_call_duration_sec.npy'}  shape {np.array(zc_dur).shape}")
    print(f"  {OUT_DIR / 'osrm_routing_meta.json'}")


if __name__ == "__main__":
    main()
