"""
Compliance-table (MEXCLP) relocation simulation vs. static home-base dispatch,
using REAL road-network driving times (OSRM) instead of haversine distance +
a flat mph assumption.

  MEXCLP (Maximum Expected Covering Location Problem, Daskin 1983) -- ranks
  candidate posts by expected coverage contribution, accounting for the
  probability that a unit covering a given area is already busy (the "busy
  fraction" q). The greedy add-one-at-a-time algorithm used here is the
  standard, published, near-optimal MEXCLP heuristic, not an invented one.

The output is a COMPLIANCE TABLE: a ranked list of posts (1st priority post,
2nd priority post, ...). As units go busy on calls, the remaining idle units
continuously re-fill the top of the list; lower-priority posts go uncovered
first.

REAL data: call arrival timestamps + locations (Seattle 911 dataset), and
REAL driving durations/distances between the 20 candidate posts and every
call location (precomputed via OSRM, see precompute_routing.py).

ASSUMPTION (disclosed, not data): how long a unit is busy per call --
dispatch + on-scene + transport + hospital + return-to-post. Sampled from a
lognormal centered on ~50 minutes, a commonly cited industry-average full
unit-hour cycle time.

APPROXIMATION (disclosed): idle units in this model always sit at one of the
20 candidate posts (home base, or a compliance-table repositioning target),
so the OSRM zone-to-call table covers the large majority of dispatch
lookups directly. For the minority "no idle unit, dispatch whichever frees
soonest" case, the busy unit's position (a raw call location, not a post) is
approximated by its nearest candidate post for routing-lookup purposes --
applied identically to both strategies, so it does not bias the comparison.

Both strategies run over the IDENTICAL real call stream (same arrivals, same
order, same random service-time draws) so response times are directly,
fairly paired call-by-call.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_CSV = OUT_DIR / "seattle" / "seattle_911_raw.csv"
N_AMBULANCES = 16      # sized for PEAK hourly demand (~13.8 calls/hr at peak x 50min service
                        # = ~11.5 Erlangs at peak), not just the 60-day average
K_ZONES = 20            # candidate staging posts
SIM_DAYS = 60
SERVICE_MEAN_MIN = 50   # DISCLOSED ASSUMPTION: avg minutes a unit is busy per call
RESPONSE_STANDARD_MIN = 8  # coverage standard used by the MEXCLP model (a common EMS benchmark)
RNG_SEED = 42

EMS_TYPES_CONTAINS = [
    "Aid Response", "Medic Response", "Low Acuity", "Triaged Incident",
    "Nurseline", "MVI", "Automatic Medical Alarm",
]


def haversine_miles(lat1, lng1, lat2, lng2):
    r = 3958.8
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


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


def load_routing():
    """Load precomputed OSRM matrices -- see precompute_routing.py. Must be
    run first (`python3 src/precompute_routing.py`) using the identical
    load_calls() ordering so call indices line up with the columns here."""
    meta = json.load(open(OUT_DIR / "osrm_routing_meta.json"))
    zone_centers = np.array([[z["lat"], z["lng"]] for z in meta["zone_centers"]])
    zz_dur = np.load(OUT_DIR / "osrm_zone_zone_duration_sec.npy")   # [zone, zone] seconds
    zc_dur = np.load(OUT_DIR / "osrm_zone_call_duration_sec.npy")   # [zone, call] seconds
    return zone_centers, zz_dur, zc_dur


def mexclp_compliance_table(zone_weights, q, zz_dur_sec, standard_sec, n_ranks):
    """
    Daskin's (1983) greedy MEXCLP heuristic, using REAL driving-time coverage
    (zz_dur_sec <= standard_sec) instead of a haversine-distance proxy.
    """
    n_zones = len(zone_weights)
    covers = zz_dur_sec <= standard_sec  # covers[demand_i, site_j]

    chosen = []
    coverage_count = np.zeros(n_zones)
    remaining = set(range(n_zones))

    for _ in range(min(n_ranks, n_zones)):
        best_site, best_gain = None, -np.inf
        for s in remaining:
            covered_nodes = covers[:, s]
            marginal = zone_weights[covered_nodes] * (q ** coverage_count[covered_nodes]) * (1 - q)
            gain = marginal.sum()
            if gain > best_gain:
                best_gain, best_site = gain, s
        chosen.append(best_site)
        coverage_count += covers[:, best_site]
        remaining.remove(best_site)

    return chosen


class Ambulance:
    __slots__ = ("id", "lat", "lng", "zone_id", "free_time", "home_zone_id")

    def __init__(self, id_, zone_id, lat, lng):
        self.id = id_
        self.zone_id = zone_id  # None when sitting at a raw call location, not a post
        self.lat = lat
        self.lng = lng
        self.free_time = pd.Timestamp.min
        self.home_zone_id = zone_id


def nearest_zone(lat, lng, zone_centers):
    d = haversine_miles(lat, lng, zone_centers[:, 0], zone_centers[:, 1])
    return int(np.argmin(d))


def reposition_idle_dynamic(idle_ambulances, compliance_sites, zone_centers, zz_dur_sec):
    """
    Greedy nearest-site matching (by real driving time): idle units
    continuously occupy the top-M compliance-table posts (M = number
    currently idle). Highest-priority posts filled first, matched to
    whichever idle unit's post-to-post driving time is shortest.
    """
    targets = compliance_sites[:len(idle_ambulances)]
    unassigned = list(idle_ambulances)
    for target_zone in targets:
        durs = []
        for a in unassigned:
            src_zone = a.zone_id if a.zone_id is not None else nearest_zone(a.lat, a.lng, zone_centers)
            durs.append(zz_dur_sec[src_zone][target_zone])
        nearest = unassigned.pop(int(np.argmin(durs)))
        nearest.zone_id = target_zone
        nearest.lat, nearest.lng = zone_centers[target_zone][0], zone_centers[target_zone][1]


def run_simulation(calls: pd.DataFrame, zone_centers: np.ndarray, zz_dur_sec: np.ndarray,
                    zc_dur_sec: np.ndarray, compliance_table: list, strategy: str,
                    home_zone_ids: list, rng: np.random.Generator, log_events: bool = False):
    """
    strategy: "static"  -> each ambulance has one fixed assigned post; always
                           returns there once idle.
              "dynamic" -> idle units continuously re-fill the top of the
                           MEXCLP compliance table (see reposition_idle_dynamic).
    """
    ambulances = [
        Ambulance(i, home_zone_ids[i], zone_centers[home_zone_ids[i]][0], zone_centers[home_zone_ids[i]][1])
        for i in range(N_AMBULANCES)
    ]
    response_minutes = np.empty(len(calls))
    wait_minutes = np.empty(len(calls))
    events = [] if log_events else None

    for i, row in enumerate(calls.itertuples(index=False)):
        t = row.datetime
        clat, clng = row.latitude, row.longitude

        idle = [a for a in ambulances if a.free_time <= t]

        if strategy == "static":
            for a in idle:
                a.zone_id = a.home_zone_id
                a.lat, a.lng = zone_centers[a.zone_id][0], zone_centers[a.zone_id][1]
        elif idle:
            reposition_idle_dynamic(idle, compliance_table, zone_centers, zz_dur_sec)

        if idle:
            durs = [zc_dur_sec[a.zone_id][i] for a in idle]  # every idle unit sits at a zone post now
            best = int(np.argmin(durs))
            chosen = idle[best]
            start_time = t
            wait_min = 0.0
            travel_min = durs[best] / 60.0
        else:
            chosen = min(ambulances, key=lambda a: a.free_time)
            start_time = chosen.free_time
            wait_min = (start_time - t).total_seconds() / 60.0
            src_zone = chosen.zone_id if chosen.zone_id is not None else nearest_zone(chosen.lat, chosen.lng, zone_centers)
            travel_min = zc_dur_sec[src_zone][i] / 60.0

        response_minutes[i] = wait_min + travel_min
        wait_minutes[i] = wait_min

        from_lat, from_lng = chosen.lat, chosen.lng

        service_min = max(5.0, rng.lognormal(mean=np.log(SERVICE_MEAN_MIN), sigma=0.35))
        finish_time = start_time + pd.Timedelta(minutes=travel_min + service_min)
        chosen.free_time = finish_time
        # Leave the unit at the call location on dispatch -- it gets properly
        # repositioned (home, or compliance post) the next time it's found
        # idle, at the top of a future iteration. zone_id=None marks "not
        # currently at a canonical post."
        chosen.lat, chosen.lng = clat, clng
        chosen.zone_id = None

        if log_events:
            events.append({
                "t": t.isoformat(),
                "call_lat": float(clat), "call_lng": float(clng),
                "ambulance_id": chosen.id,
                "from_lat": float(from_lat), "from_lng": float(from_lng),
                "to_lat": float(clat), "to_lng": float(clng),
                "wait_min": round(float(wait_min), 2),
                "travel_min": round(float(travel_min), 2),
                "response_min": round(float(wait_min + travel_min), 2),
                "n_idle_at_arrival": len(idle),
            })

    return response_minutes, wait_minutes, events


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calls = load_calls()
    print(f"Simulating {len(calls):,} real calls over last {SIM_DAYS} days "
          f"({calls['datetime'].min()} to {calls['datetime'].max()})")

    zone_centers, zz_dur_sec, zc_dur_sec = load_routing()
    assert zc_dur_sec.shape[1] == len(calls), (
        "OSRM routing matrix call count doesn't match load_calls() -- "
        "rerun precompute_routing.py against the current dataset first."
    )
    print(f"Loaded real OSRM driving-time matrices: zone-zone {zz_dur_sec.shape}, zone-call {zc_dur_sec.shape}")

    # Recompute zone membership counts (for MEXCLP demand weights) using the
    # SAME zone centers as the routing matrices (nearest-zone assignment).
    zone_ids_per_call = np.array([nearest_zone(r.latitude, r.longitude, zone_centers)
                                   for r in calls.itertuples(index=False)])
    zone_counts = np.bincount(zone_ids_per_call, minlength=K_ZONES)
    zone_weights = zone_counts / zone_counts.sum()

    sim_minutes = SIM_DAYS * 24 * 60
    q = (len(calls) * SERVICE_MEAN_MIN) / (N_AMBULANCES * sim_minutes)
    q = min(0.95, q)
    print(f"Empirical busy fraction q = {q:.3f}")

    standard_sec = RESPONSE_STANDARD_MIN * 60
    compliance_table = mexclp_compliance_table(zone_weights, q, zz_dur_sec, standard_sec, N_AMBULANCES)
    print(f"MEXCLP compliance table (rank 1 first, real driving-time coverage): {compliance_table}")

    home_zone_ids = compliance_table[:N_AMBULANCES]

    rng_static = np.random.default_rng(RNG_SEED)
    rng_dynamic = np.random.default_rng(RNG_SEED)

    static_resp, static_wait, static_events = run_simulation(
        calls, zone_centers, zz_dur_sec, zc_dur_sec, compliance_table, "static",
        home_zone_ids, rng_static, log_events=True)
    dynamic_resp, dynamic_wait, dynamic_events = run_simulation(
        calls, zone_centers, zz_dur_sec, zc_dur_sec, compliance_table, "dynamic",
        home_zone_ids, rng_dynamic, log_events=True)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(static_resp), size=min(5000, len(static_resp)), replace=False)
    stat, p_value = wilcoxon(static_resp[idx], dynamic_resp[idx])

    pct_improved = float((dynamic_resp < static_resp).mean() * 100)
    pct_delayed_static = float((static_wait > 0).mean() * 100)
    pct_delayed_dynamic = float((dynamic_wait > 0).mean() * 100)

    homes = zone_centers[home_zone_ids]

    summary = {
        "city": "Seattle, WA",
        "sim_days": SIM_DAYS,
        "n_calls": len(calls),
        "n_ambulances": N_AMBULANCES,
        "k_zones": K_ZONES,
        "busy_fraction_q": round(float(q), 3),
        "response_standard_min": RESPONSE_STANDARD_MIN,
        "service_time_assumption_min": SERVICE_MEAN_MIN,
        "routing_source": "OSRM public demo server, real driving durations (no flat-mph assumption)",
        "compliance_table": compliance_table,
        "avg_response_min_static": round(float(static_resp.mean()), 2),
        "avg_response_min_dynamic": round(float(dynamic_resp.mean()), 2),
        "median_response_min_static": round(float(np.median(static_resp)), 2),
        "median_response_min_dynamic": round(float(np.median(dynamic_resp)), 2),
        "p90_response_min_static": round(float(np.percentile(static_resp, 90)), 2),
        "p90_response_min_dynamic": round(float(np.percentile(dynamic_resp, 90)), 2),
        "minutes_saved_avg": round(float(static_resp.mean() - dynamic_resp.mean()), 2),
        "pct_improvement": round(float((static_resp.mean() - dynamic_resp.mean()) / static_resp.mean() * 100), 1),
        "pct_calls_improved": round(pct_improved, 1),
        "pct_calls_delayed_static": round(pct_delayed_static, 1),
        "pct_calls_delayed_dynamic": round(pct_delayed_dynamic, 1),
        "wilcoxon_p_value": float(p_value),
        "statistically_significant": bool(p_value < 0.05),
        "home_bases": [{"lat": float(h[0]), "lng": float(h[1]), "zone_id": int(z)}
                        for h, z in zip(homes, home_zone_ids)],
        "all_zone_centers": [{"lat": float(z[0]), "lng": float(z[1]), "weight": float(w)}
                              for z, w in zip(zone_centers, zone_weights)],
    }

    with open(OUT_DIR / "dynamic_sim_seattle.json", "w") as f:
        json.dump(summary, f, indent=2)

    day_counts = calls["datetime"].dt.floor("D").value_counts()
    busiest_day = day_counts.idxmax()
    day_mask = calls["datetime"].dt.floor("D") == busiest_day
    day_idx = np.where(day_mask.values)[0]
    print(f"\nBusiest day for animation trace: {busiest_day.date()} ({len(day_idx)} calls)")

    trace = {
        "date": str(busiest_day.date()),
        "n_calls": len(day_idx),
        "n_ambulances": N_AMBULANCES,
        "home_bases": summary["home_bases"],
        "all_zone_centers": summary["all_zone_centers"],
        "compliance_table": compliance_table,
        "static_events": [static_events[i] for i in day_idx],
        "dynamic_events": [dynamic_events[i] for i in day_idx],
    }
    with open(OUT_DIR / "sim_trace_seattle.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Wrote {OUT_DIR / 'sim_trace_seattle.json'}")

    print("\n=== MEXCLP + REAL ROAD ROUTING SIMULATION RESULTS (Seattle, real calls) ===")
    print(f"Static  avg response: {static_resp.mean():.2f} min (median {np.median(static_resp):.2f}, p90 {np.percentile(static_resp,90):.2f})")
    print(f"Dynamic avg response: {dynamic_resp.mean():.2f} min (median {np.median(dynamic_resp):.2f}, p90 {np.percentile(dynamic_resp,90):.2f})")
    print(f"Minutes saved (avg):  {static_resp.mean() - dynamic_resp.mean():.2f}")
    print(f"Improvement:          {summary['pct_improvement']}%")
    print(f"% calls improved:     {pct_improved:.1f}%")
    print(f"Calls delayed (no idle unit) -- static: {pct_delayed_static:.1f}%  dynamic: {pct_delayed_dynamic:.1f}%")
    print(f"Wilcoxon p-value: {p_value:.2e}  significant: {summary['statistically_significant']}")


if __name__ == "__main__":
    main()
