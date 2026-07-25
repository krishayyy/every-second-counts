"""
EMS predictive relocation pipeline.

Real ambulance relocation (System Status Management) doesn't invent new
street corners each hour -- it picks, from a fixed set of plausible staging
zones, WHICH zones to staff right now based on where demand currently is.
So the model here:

1. Load Montgomery County 911 data, keep EMS calls only, drop bad geocodes.
2. Cluster ALL EMS calls into K_ZONES candidate staging zones (fixed
   locations -- these represent realistic places crews could post).
3. Bin calls by (day-of-week bucket x hour-of-day bucket).
4. Baseline = the N_POSTS busiest zones overall, staffed all day, every day
   (models "always parked in the same historically-busy spots").
5. Recommended = the N_POSTS busiest zones IN THAT SPECIFIC TIME WINDOW
   (models "move staffed posts to wherever demand currently is").
6. Compare average distance-to-call under each, plus a paired significance
   test, and write a JSON for the map UI.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon

DATA_CSV = "/Users/krishay/.cache/kagglehub/datasets/mchirico/montcoalert/versions/32/911.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
K_ZONES = 14   # candidate staging zones across the county
N_POSTS = 5    # how many of those zones are actually staffed at once

DAY_BUCKETS = {
    **{d: "weekday" for d in ["Mon", "Tue", "Wed", "Thu"]},
    "Fri": "friday",
    "Sat": "weekend",
    "Sun": "weekend",
}


def hour_bucket(hour: int) -> str:
    starts = [0, 4, 8, 12, 16, 20]
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else 24
        if s <= hour < end:
            return f"{s:02d}-{end:02d}"
    return "20-24"


def haversine_miles(lat1, lng1, lat2, lng2):
    r = 3958.8
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_ems_calls() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, usecols=["lat", "lng", "title", "timeStamp", "twp"])
    df = df[df["title"].str.startswith("EMS")].copy()
    df = df.dropna(subset=["lat", "lng", "timeStamp"])
    # Drop bad geocodes: dataset has some rows with garbage coordinates
    # (e.g. 0,0, or points in other states/countries) that wreck clustering.
    df = df[df["lat"].between(39.9, 40.5) & df["lng"].between(-75.7, -75.0)]
    df["timeStamp"] = pd.to_datetime(df["timeStamp"], errors="coerce")
    df = df.dropna(subset=["timeStamp"])
    df["day"] = df["timeStamp"].dt.strftime("%a").map(DAY_BUCKETS)
    df["hour_bucket"] = df["timeStamp"].dt.hour.map(hour_bucket)
    df["bucket"] = df["day"] + "_" + df["hour_bucket"]
    return df


def nearest_dist(calls: pd.DataFrame, posts: np.ndarray) -> np.ndarray:
    dists = np.stack(
        [haversine_miles(calls["lat"].values, calls["lng"].values, p[0], p[1]) for p in posts]
    )
    return dists.min(axis=0)


def best_subset(dist_matrix: np.ndarray, k: int):
    """
    Exact best subset of k rows (zones) out of dist_matrix (zones x calls)
    minimizing mean of the per-column min over the chosen rows.
    Brute force over all C(K_ZONES, k) combos -- cheap at this scale.
    """
    from itertools import combinations

    best_ids, best_mean = None, np.inf
    for combo in combinations(range(dist_matrix.shape[0]), k):
        mean_dist = dist_matrix[list(combo)].min(axis=0).mean()
        if mean_dist < best_mean:
            best_mean, best_ids = mean_dist, combo
    return list(best_ids), best_mean


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calls = load_ems_calls()
    print(f"Loaded {len(calls):,} EMS calls, {calls['bucket'].nunique()} time buckets")

    # Fixed candidate staging zones, derived once from ALL historical calls.
    zone_model = KMeans(n_clusters=K_ZONES, n_init=10, random_state=42)
    calls["zone"] = zone_model.fit_predict(calls[["lat", "lng"]])
    zone_centers = zone_model.cluster_centers_

    # Baseline: the single best fixed N_POSTS-zone subset for ALL-time calls,
    # staffed unchanged all day every day ("always parked in the same spots").
    all_dist_matrix = np.stack(
        [haversine_miles(calls["lat"].values, calls["lng"].values, z[0], z[1]) for z in zone_centers]
    )
    baseline_zone_ids, _ = best_subset(all_dist_matrix, N_POSTS)
    baseline_posts = zone_centers[baseline_zone_ids]
    print(f"Baseline (fixed) zones: {baseline_zone_ids}")

    results = {}
    total_baseline_dist, total_recommended_dist = [], []
    per_call_baseline, per_call_recommended = [], []
    total_calls = 0

    for bucket, group in calls.groupby("bucket"):
        if len(group) < N_POSTS * 5:
            continue

        # Recommended: the OPTIMAL N_POSTS-zone subset for THIS window's calls.
        bucket_dist_matrix = np.stack(
            [haversine_miles(group["lat"].values, group["lng"].values, z[0], z[1]) for z in zone_centers]
        )
        rec_zone_ids, _ = best_subset(bucket_dist_matrix, N_POSTS)
        rec_posts = zone_centers[rec_zone_ids]

        base_dists = nearest_dist(group, baseline_posts)
        rec_dists = nearest_dist(group, rec_posts)

        baseline_dist = base_dists.mean()
        recommended_dist = rec_dists.mean()
        pct_improvement = (baseline_dist - recommended_dist) / baseline_dist * 100

        n = len(group)
        total_calls += n
        total_baseline_dist.append(baseline_dist * n)
        total_recommended_dist.append(recommended_dist * n)
        per_call_baseline.append(base_dists)
        per_call_recommended.append(rec_dists)

        results[bucket] = {
            "n_calls": n,
            "posts": [{"lat": float(p[0]), "lng": float(p[1])} for p in rec_posts],
            "zone_ids": [int(z) for z in rec_zone_ids],
            "avg_dist_baseline_mi": round(float(baseline_dist), 3),
            "avg_dist_recommended_mi": round(float(recommended_dist), 3),
            "pct_improvement": round(float(pct_improvement), 1),
        }
        print(f"{bucket:16s}  n={n:6d}  baseline={baseline_dist:.2f}mi  "
              f"recommended={recommended_dist:.2f}mi  ({pct_improvement:+.1f}%)  zones={rec_zone_ids}")

    weighted_baseline = sum(total_baseline_dist) / total_calls
    weighted_recommended = sum(total_recommended_dist) / total_calls

    mph = 30
    baseline_min = weighted_baseline / mph * 60
    recommended_min = weighted_recommended / mph * 60

    all_baseline = np.concatenate(per_call_baseline)
    all_recommended = np.concatenate(per_call_recommended)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_baseline), size=min(5000, len(all_baseline)), replace=False)
    stat, p_value = wilcoxon(all_baseline[sample_idx], all_recommended[sample_idx])

    pct_calls_improved = float((all_recommended < all_baseline).mean() * 100)

    summary = {
        "n_calls_used": total_calls,
        "k_zones": K_ZONES,
        "n_posts": N_POSTS,
        "all_zone_centers": [{"lat": float(p[0]), "lng": float(p[1])} for p in zone_centers],
        "static_baseline_posts": [
            {"lat": float(p[0]), "lng": float(p[1])} for p in baseline_posts
        ],
        "avg_dist_baseline_mi": round(weighted_baseline, 3),
        "avg_dist_recommended_mi": round(weighted_recommended, 3),
        "avg_response_min_baseline": round(baseline_min, 2),
        "avg_response_min_recommended": round(recommended_min, 2),
        "minutes_saved": round(baseline_min - recommended_min, 2),
        "pct_improvement_overall": round((weighted_baseline - weighted_recommended) / weighted_baseline * 100, 1),
        "pct_calls_with_shorter_distance": round(pct_calls_improved, 1),
        "wilcoxon_p_value": float(p_value),
        "statistically_significant": bool(p_value < 0.05),
        "mph_assumption": mph,
        "buckets": results,
    }

    with open(OUT_DIR / "relocation_model.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Baseline avg response:    {baseline_min:.2f} min")
    print(f"Recommended avg response: {recommended_min:.2f} min")
    print(f"Minutes saved:            {baseline_min - recommended_min:.2f} min")
    print(f"Overall improvement:      {summary['pct_improvement_overall']}%")
    print(f"% calls with shorter dist under recommended: {pct_calls_improved:.1f}%")
    print(f"Wilcoxon p-value: {p_value:.2e}  (significant: {summary['statistically_significant']})")
    print(f"\nWrote {OUT_DIR / 'relocation_model.json'}")


if __name__ == "__main__":
    main()
