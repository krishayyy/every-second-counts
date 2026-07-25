"""
Same optimal-subset relocation model as pipeline.py, run on Seattle Fire's
real-time 911 dataset instead of Montgomery County PA -- a dense urban
geography, to test whether the "no spatial shift" finding from Montgomery
County was about the MODEL or about that specific (suburban/exurban) place.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon

DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "seattle" / "seattle_911_raw.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
K_ZONES = 20
N_POSTS = 5

EMS_TYPES_CONTAINS = [
    "Aid Response", "Medic Response", "Low Acuity", "Triaged Incident",
    "Nurseline", "MVI", "Automatic Medical Alarm",
]

DAY_BUCKETS = {0: "weekday", 1: "weekday", 2: "weekday", 3: "weekday",
               4: "friday", 5: "weekend", 6: "weekend"}


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


def nearest_dist(calls, posts):
    dists = np.stack(
        [haversine_miles(calls["latitude"].values, calls["longitude"].values, p[0], p[1]) for p in posts]
    )
    return dists.min(axis=0)


def best_subset(dist_matrix: np.ndarray, k: int):
    best_ids, best_mean = None, np.inf
    for combo in combinations(range(dist_matrix.shape[0]), k):
        mean_dist = dist_matrix[list(combo)].min(axis=0).mean()
        if mean_dist < best_mean:
            best_mean, best_ids = mean_dist, combo
    return list(best_ids), best_mean


def load_calls() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    pattern = "|".join(EMS_TYPES_CONTAINS)
    df = df[df["type"].str.contains(pattern, case=False, na=False)].copy()
    df = df.dropna(subset=["latitude", "longitude", "datetime"])
    # Seattle proper bounding box -- drop any stray bad geocodes.
    df = df[df["latitude"].between(47.4, 47.8) & df["longitude"].between(-122.5, -122.2)]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["day"] = df["datetime"].dt.dayofweek.map(DAY_BUCKETS)
    df["hour_bucket"] = df["datetime"].dt.hour.map(hour_bucket)
    df["bucket"] = df["day"] + "_" + df["hour_bucket"]
    return df


def main():
    OUT_DIR.mkdir(exist_ok=True)
    calls = load_calls()
    print(f"Loaded {len(calls):,} Seattle EMS/medical calls, {calls['bucket'].nunique()} buckets")

    zone_model = KMeans(n_clusters=K_ZONES, n_init=10, random_state=42)
    calls["zone"] = zone_model.fit_predict(calls[["latitude", "longitude"]])
    zone_centers = zone_model.cluster_centers_

    all_dm = np.stack(
        [haversine_miles(calls["latitude"].values, calls["longitude"].values, z[0], z[1]) for z in zone_centers]
    )
    baseline_zone_ids, _ = best_subset(all_dm, N_POSTS)
    baseline_posts = zone_centers[baseline_zone_ids]
    print(f"Baseline (fixed) zones: {baseline_zone_ids}")

    results = {}
    total_baseline_dist, total_recommended_dist = [], []
    per_call_baseline, per_call_recommended = [], []
    total_calls = 0

    for bucket, group in calls.groupby("bucket"):
        if len(group) < N_POSTS * 5:
            continue

        dm = np.stack(
            [haversine_miles(group["latitude"].values, group["longitude"].values, z[0], z[1]) for z in zone_centers]
        )
        rec_zone_ids, _ = best_subset(dm, N_POSTS)
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
        print(f"{bucket:16s}  n={n:6d}  baseline={baseline_dist:.3f}mi  "
              f"recommended={recommended_dist:.3f}mi  ({pct_improvement:+.1f}%)  zones={rec_zone_ids}")

    weighted_baseline = sum(total_baseline_dist) / total_calls
    weighted_recommended = sum(total_recommended_dist) / total_calls

    mph = 20  # urban traffic is slower than suburban -- more realistic for Seattle
    baseline_min = weighted_baseline / mph * 60
    recommended_min = weighted_recommended / mph * 60

    all_baseline = np.concatenate(per_call_baseline)
    all_recommended = np.concatenate(per_call_recommended)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(all_baseline), size=min(5000, len(all_baseline)), replace=False)
    stat, p_value = wilcoxon(all_baseline[sample_idx], all_recommended[sample_idx])
    pct_calls_improved = float((all_recommended < all_baseline).mean() * 100)

    summary = {
        "city": "Seattle, WA",
        "n_calls_used": total_calls,
        "k_zones": K_ZONES,
        "n_posts": N_POSTS,
        "all_zone_centers": [{"lat": float(p[0]), "lng": float(p[1])} for p in zone_centers],
        "static_baseline_posts": [{"lat": float(p[0]), "lng": float(p[1])} for p in baseline_posts],
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

    with open(OUT_DIR / "relocation_model_seattle.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SEATTLE SUMMARY ===")
    print(f"Baseline avg response:    {baseline_min:.2f} min")
    print(f"Recommended avg response: {recommended_min:.2f} min")
    print(f"Minutes saved:            {baseline_min - recommended_min:.2f} min")
    print(f"Overall improvement:      {summary['pct_improvement_overall']}%")
    print(f"% calls improved: {pct_calls_improved:.1f}%   p-value: {p_value:.2e}")


if __name__ == "__main__":
    main()
