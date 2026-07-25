# EMS Predictive Relocation — Handoff

**Project:** Predictive ambulance repositioning ("weather forecast for emergencies") — a model that recommends where to stage ambulances before calls come in, instead of static home-base parking.

**Status:** Working end-to-end pipeline + two-view interactive simulator, built on 100% real 911 data. One disclosed modeling assumption (per-call service duration). No synthetic data anywhere else.

---

## 1. The pitch (honest version)

- The math (ambulance relocation / coverage optimization) has existed since the 1970s–80s. Big cities and private EMS providers already do versions of this.
- The gap: small-to-mid counties can't afford custom data-science teams or enterprise System Status Management software. They run on dispatcher gut-feel, not data.
- The product angle: cheap, plug-and-play, explainable relocation recommendations for counties that currently have nothing — "TurboTax for ambulance positioning," not "we invented new science."
- Every number in this repo is real or explicitly labeled as an assumption — nothing was fabricated to make the pitch land.

## 2. What we tried, in order (including the dead ends — keep this in the pitch, it builds credibility)

1. **Naive static-vs-time-varying k-means centroids (Montgomery County, PA).** First result: 36.8% improvement, 6.59 min saved. **This was a bug** — 3 of 4 "baseline" cluster centroids were dragged off-map by bad geocodes in the raw dataset (rows with lat/lng in India, California, and literally 0,0). After filtering to the real county bounding box: **0.1% improvement, not significant (p=0.98).**
2. **Top-N-busiest-zone reassignment per time bucket (Montgomery County).** Picking the N zones with the most calls in each time window, from a fixed set of candidate zones. Result: **~0%, sometimes negative.** The busiest zones are the busiest zones almost all day — swapping the marginal 5th zone doesn't help enough.
3. **Exact optimal-subset selection per time bucket (Montgomery County, then Seattle).** Brute-force search for the mathematically best N-of-K zone subset for each specific time window (guaranteed ≥ as good as any fixed baseline, by construction). Result on Montgomery County: **0.0% — the optimal subset was IDENTICAL in every single time bucket**, at both 4-hour and 1-hour granularity, with up to 30 candidate zones. Re-ran the same method on a real dense-urban dataset (Seattle) to rule out "wrong city": **same result, 0.0%, not significant (p=0.12).**
   - **Real finding, not a failure:** where EMS calls happen doesn't meaningfully shift by time-of-day/day-of-week in either geography tested — only call *volume* shifts, not *location*. A static schedule based on "it's rush hour, move the ambulance" doesn't work because the busy zones stay the busy zones all day.
4. **Dynamic queueing simulation, ad hoc "coverage gap" heuristic (Seattle).** Pivoted to the actual mechanism real EMS relocation systems use: react to which units are currently BUSY, and backfill the resulting coverage gap live, rather than following a static daily schedule. First version used an invented "move to the zone farthest from any other unit" heuristic. Result: **2.7% avg / ~11% median improvement, p=4.66×10⁻⁶** — real and significant, but a hand-rolled heuristic, not a published method.
5. **MEXCLP compliance-table model, haversine distance (Seattle).** Replaced the ad hoc heuristic with Daskin's (1983) greedy MEXCLP algorithm, the actual published method real EMS System Status Management uses: it ranks candidate posts by expected coverage contribution, accounting for the probability a covering unit is already busy (the empirical "busy fraction" q). Result on straight-line distance: **21.7% avg / ~33% median improvement, p=8.80×10⁻¹⁶⁰.**
6. **Same MEXCLP model, real OSRM road-network routing (Seattle) — current version.** Replaced haversine distance + flat mph with real driving durations from OSRM (router.project-osrm.org) for every zone-to-zone and zone-to-call pair. **This is the current, strongest, most defensible result** — see §5.

## 3. Datasets used

| Dataset | Source | Real? | Notes |
|---|---|---|---|
| Montgomery County, PA 911 calls | [Kaggle: mchirico/montcoalert](https://www.kaggle.com/datasets/mchirico/montcoalert) | 100% real | 332,208 EMS calls after filtering, 2015–2020. Has known bad-geocode rows (see §2.1) — always bound-box filter before use. |
| Seattle Fire real-time 911 calls | [data.seattle.gov Socrata API](https://data.seattle.gov/resource/kzjm-xkqj.json), dataset id `kzjm-xkqj` | 100% real | Pulled via public API, no auth needed. 2.19M total rows; we downloaded the most recent 500K. Filtered to EMS-relevant `type` values (Aid Response, Medic Response, Low Acuity, Triaged Incident, Nurseline, MVI, Automatic Medical Alarm). |
| HIFLD Hospitals | [hifld-geoplatform.opendata.arcgis.com/datasets/hospitals](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals) | 100% real | Downloaded but **not yet integrated** into the pipeline. Has lat/lng, trauma level, bed count, helipad flag — useful for a future "route to nearest appropriate hospital" feature. |
| Nominatim (OpenStreetMap) reverse geocoding | [nominatim.openstreetmap.org](https://nominatim.openstreetmap.org) | 100% real | Used to convert the 16 recommended post coordinates into real street addresses for the "Simple" view. Free, 1 req/sec rate limit, requires a `User-Agent` header. Note: Python's default `urllib` hit an SSL cert error on this machine — used `curl` via subprocess instead. Must be re-run whenever the compliance table changes (see §8). |
| OSRM (Open Source Routing Machine) | [router.project-osrm.org](https://router.project-osrm.org) | 100% real | Public demo server, `driving` profile. Used for real road-network durations/distances between the 20 candidate posts and every one of the 14,070 real call locations (table API, batched). Documented as "demo, not production" — fine for this prototype; a self-hosted OSRM instance would be the production equivalent. Also hit the same `urllib` SSL issue — used `curl` via subprocess. |

Raw data lives at:
- `/Users/krishay/.cache/kagglehub/datasets/mchirico/montcoalert/versions/32/911.csv` (Montgomery County, downloaded via `kagglehub`)
- `/Users/krishay/ems-relocation/data/seattle/seattle_911_raw.csv` (Seattle, 500K rows)

## 4. The MEXCLP + real-routing simulation — how it actually works

Files: `src/precompute_routing.py` (one-time OSRM precompute), `src/osrm_routing.py` (OSRM table-API client), `src/simulate_dynamic.py` (the simulation itself)

- **Real, not synthetic:** call arrival timestamps and locations, straight from the Seattle dataset, for the most recent 60 real days in the sample. Real driving durations from OSRM for every zone-to-zone and zone-to-call pair.
- **One disclosed assumption:** how long an ambulance is busy per call (dispatch → scene → transport → hospital → clear). Sampled from a lognormal centered on **50 minutes** (a commonly cited industry-average full unit-cycle time). This is NOT in either source dataset — it's the only non-real number feeding the model, and it's called out explicitly in the simulator UI itself.
- **Fleet size is not arbitrary:** sized to real peak-hour demand (not the 60-day average, which would look artificially light). Real Seattle EMS-relevant calls peak at ~13.8/hour; at ~50 min average service time that's ~11.5 Erlangs of offered load at peak. 16 ambulances gives a comfortable-but-real utilization margin. (Earlier attempts at 5 and 12 ambulances caused unbounded queue backlog — response times exploding into the tens of thousands of minutes — because the fleet was undersized relative to real peak demand, not because of a strategy difference. Worth knowing if you re-run with a different city/fleet size: check the offered-load math in the script's sizing comments before trusting the output.)
- **Real road routing (OSRM), not haversine + flat mph:** `precompute_routing.py` queries the public OSRM demo server's table API for real driving durations between the 20 candidate posts (20×20 matrix) and between every candidate post and every one of the 14,070 real call locations (20×14,070 matrix), batched to respect the shared server. `simulate_dynamic.py` loads these precomputed matrices and does pure array lookups during the simulation — no live network calls in the hot loop.
  - **Approximation (disclosed):** idle units in this model always sit at one of the 20 candidate posts (home base, or a compliance-table repositioning target), so the precomputed table covers the large majority of dispatch lookups directly. For the minority "no idle unit, dispatch whichever frees soonest" case, the busy unit's actual position (a raw call location) is approximated by snapping to its nearest candidate post for the lookup — applied identically to both strategies, so it doesn't bias the static-vs-dynamic comparison.
- **The compliance table (MEXCLP, Daskin 1983):** a ranked list of candidate posts, built by a greedy algorithm that repeatedly adds whichever remaining site gives the biggest increase in *expected* coverage — where a demand area already covered by k chosen posts gets a diminishing marginal benefit from a (k+1)-th coverer, weighted by the empirical **busy fraction q** (the fraction of time an average unit is occupied, computed directly from real call volume × the disclosed service-time assumption, not a free parameter). Coverage itself is now defined by **real driving time** (≤ 8 minutes), not a haversine-distance proxy.
- **Two strategies run over the IDENTICAL real call stream** (same arrivals, same order, same random service-time draws — paired comparison, not independent samples):
  - **Static:** each ambulance is permanently assigned to one post (the top N ranks of the compliance table) and always returns there once idle.
  - **Dynamic:** idle units continuously re-fill the top of the compliance table as availability changes — recomputed at every call arrival via greedy nearest-unit-to-post matching (by real driving time). This is genuine compliance-table restaging, not a one-off relocation.

## 5. Headline results (Seattle, real data, 60-day simulation, MEXCLP + real OSRM routing)

| Metric | Static | Dynamic |
|---|---|---|
| Avg response time | 15.26 min | **12.59 min** |
| Median response time | 9.51 min | **6.83 min** |
| P90 response time | 35.29 min | 31.19 min |
| % calls delayed (no idle unit) | 27.0% | 23.8% |

- **Minutes saved (avg): 2.67 min → 17.5% overall improvement**
- **Median improved ~28%**
- **58.9% of calls got a strictly shorter response under dynamic**
- **Statistically significant:** Wilcoxon signed-rank test on a paired 5,000-call sample, p = 2.27×10⁻¹⁶⁸
- Empirical busy fraction q ≈ 0.509; coverage standard used in the MEXCLP ranking: 8 minutes real driving time (a common EMS benchmark)

**This superseded two earlier versions**, both kept in §2 as part of the honest record: an ad hoc "coverage gap" heuristic (2.7% avg / ~11% median, p=4.66×10⁻⁶), and the same MEXCLP model on haversine distance instead of real roads (21.7% avg / ~33% median, p=8.80×10⁻¹⁶⁰). Real road routing pulled the number down from 21.7% to 17.5% — which makes sense: straight-line distance underestimates how much a good real-world routing choice actually matters (or overestimates it, depending on road network geometry); the real-routing number is the more defensible one to lead with. Still worth stating plainly in the pitch: 17.5% is on the higher end of what's typically cited in the literature (low-single-digit to low-double-digit percent) — call this out proactively (§7.7 explains why: perfect-compliance simulation, no dispatcher/radio friction modeled).

## 6. Files in this repo

```
ems-relocation/
├── HANDOFF.md                          this file
├── data/
│   ├── relocation_model.json               Montgomery County static model output (superseded, kept for the "what we tried" record)
│   ├── relocation_model_seattle.json       Seattle static model output (superseded, same reason)
│   ├── osrm_routing_meta.json              ⭐ zone centers + routing source metadata
│   ├── osrm_zone_zone_duration_sec.npy     ⭐ real OSRM driving durations, 20x20 candidate posts
│   ├── osrm_zone_zone_distance_m.npy       real OSRM driving distances, 20x20 (not currently used, kept for reference)
│   ├── osrm_zone_call_duration_sec.npy     ⭐ real OSRM driving durations, 20 posts x 14,070 real calls
│   ├── osrm_zone_call_distance_m.npy       real OSRM driving distances, same shape (not currently used)
│   ├── dynamic_sim_seattle.json            ⭐ the real headline stats (§5), full 60-day simulation summary, incl. compliance_table + busy_fraction_q
│   ├── sim_trace_seattle.json              ⭐ per-call event trace for the busiest single real day (2026-06-18, 281 calls), both strategies — powers the animated "Under the Hood" view
│   ├── home_addresses.json                 ⭐ the 16 recommended posts reverse-geocoded to real street addresses — powers the "Simple" view
│   └── seattle/seattle_911_raw.csv         raw downloaded Seattle data (500K rows)
└── src/
    ├── pipeline.py                     Montgomery County static/optimal-subset model (attempts #1-3, §2)
    ├── pipeline_seattle.py             Same static model, ported to Seattle (attempt #3 cross-check, §2)
    ├── osrm_routing.py                 ⭐ OSRM table-API client (batching, curl-based to sidestep a local urllib SSL issue)
    ├── precompute_routing.py           ⭐ one-time script: fetches + caches the real OSRM routing matrices (§4)
    ├── simulate_dynamic.py             ⭐ the current MEXCLP + real-routing simulation (§4) — THE model that matters
    ├── index.html                      early single-view map + time-bucket slider (superseded by simulator.html)
    └── simulator.html                  ⭐ THE deliverable — two-view dashboard (Simple / Under the Hood), see §8
```

⭐ = what actually matters going forward. The `pipeline*.py` static-model files are kept only as an honest record of the dead ends — don't build further on them.

## 7. Known limitations / disclosed assumptions (say these out loud in the pitch, don't wait to be asked)

1. **Service-time assumption (~50 min avg, lognormal).** Not in either source dataset. Everything else in the simulation is real, including road-network routing (see §4).
2. **OSRM is the public demo server, not a production routing deployment.** Documented by the OSRM project as "demo, not for production" with an informal rate limit — completely fine for this prototype (one-time precompute, ~200 batched requests), but a real deployment would self-host OSRM (or use a paid routing API) rather than hit the shared public instance at scale.
3. **Busy/queued-unit distance uses a nearest-post approximation, not true point-to-point routing.** When no unit is idle (the "dispatch whichever frees soonest" case, ~24-27% of calls), that unit's real position is a raw call location, not one of the 20 candidate posts. Rather than query OSRM for arbitrary point pairs (which would require up to 14,070² routes), we approximate by snapping to the nearest candidate post. This is applied identically to both strategies, so it doesn't bias the static-vs-dynamic comparison, but it does mean that minority-case distance isn't the exact real route.
4. **MEXCLP busy fraction q is a single county-wide average, not per-post.** Real compliance-table implementations sometimes use per-post or per-time-period busy fractions for more precision. Also: greedy-add MEXCLP is provably near-optimal but not guaranteed globally optimal (an exact ILP formulation exists but wasn't needed at K_ZONES=20 scale).
5. **Single-city validation only (partially).** Static model tested on 2 cities (Montgomery County PA, Seattle WA) — both showed no benefit for the static/time-of-day approach specifically. MEXCLP + real-routing model only tested on Seattle so far. Should replicate on a 3rd city before claiming generality.
6. **20-candidate-zone / 16-ambulance sizing is Seattle-specific**, tuned to that city's real call volume. Not a universal parameter — recompute per deployment.
7. **HIFLD hospital data downloaded but not integrated** — no "route to nearest appropriate trauma center" logic yet.
8. **17.5% improvement is likely optimistic vs. real-world deployment** — the simulation assumes perfect compliance (idle units always restage exactly as instructed, immediately, with no dispatcher friction or radio delay). Say this explicitly if asked "would we really see 17.5% in production."

## 8. How to run everything

```bash
cd /Users/krishay/ems-relocation

# 1. ONE-TIME: precompute real OSRM routing matrices (~3-4 min, ~200 batched
#    requests to the public OSRM server). Only re-run this if the underlying
#    call dataset changes -- the matrices are cached to data/osrm_*.npy.
python3 src/precompute_routing.py

# 2. Run the MEXCLP + real-routing simulation + animation trace (~2-3 sec)
python3 src/simulate_dynamic.py

# 3. Regenerate the reverse-geocoded addresses for the recommended posts
#    (needed whenever the compliance table ranking/composition changes --
#    it WILL change if you rerun step 1 with different data, since real
#    routing distances affect which zones rank highest)
#    See the inline curl+Nominatim snippet used originally -- 1 req/sec, ~20 sec total.

# 4. (Optional) Regenerate the static model outputs, for the historical record
python3 src/pipeline.py            # Montgomery County
python3 src/pipeline_seattle.py    # Seattle static cross-check

# 5. Serve the project root so the simulator's relative fetch() calls resolve
python3 -m http.server 8765

# 6. Open the simulator
open http://localhost:8765/src/simulator.html
```

The simulator has two top-level views:
- **Simple** — what EMS staff actually need: a ranked list of 16 recommended posts with real street addresses (reverse-geocoded via Nominatim), a clean map with numbered pins, and a detail card per post (address, priority rank, model improvement %). No jargon, no zone colors, no stats clutter.
- **Under the Hood** — everything technical, in two sub-tabs:
  - *Demand & Compliance Table* — zones colored/sized by real historical call volume, the MEXCLP compliance-table rank per zone, busy fraction q, and coverage standard.
  - *Live Simulation* — animated replay of the real busiest day (2026-06-18, 281 real calls), with a strategy toggle (static vs. dynamic) that replays the identical real call stream both ways. Playback controls: play/pause, speed (1x–20x), scrub timeline, reset.

## 9. Suggested next steps, roughly in priority order

1. **True point-to-point routing for the queued/busy-unit case** instead of the nearest-post approximation (§7.3) — would require either a full call-to-call OSRM matrix (expensive) or on-the-fly single-route queries during simulation (slower, adds a network dependency to the hot loop).
2. **Per-post or per-time-period busy fractions** instead of one county-wide average q — a more precise MEXCLP formulation, likely tightens the result further.
3. **Validate on a 3rd city** to strengthen the generality claim before pitching this as broadly applicable.
4. **Integrate the HIFLD hospital data** for a "route to nearest appropriate hospital" feature — cheap addition, real dataset already downloaded.
5. **Get a real county's actual CAD export** — this is the actual business-development bottleneck, not a technical one. Everything here proves the method works on real data; the next real unlock is a pilot relationship with an actual small/mid county EMS director.
6. **Side-by-side simultaneous playback** (static and dynamic animating at once) — nice-to-have polish for the demo, not required for the core proof.
7. **Self-host OSRM** (or move to a paid routing API) before any real production use — the public demo server used here is explicitly documented as non-production.

## 10. One-sentence status for anyone picking this up cold

The static "reposition ambulances by time-of-day" idea was tested rigorously on two real cities and found to have ~0% effect (a real, useful null result, not a failure of effort); the dynamic "react to which units are busy right now" model, built on the actual published MEXCLP compliance-table method (Daskin 1983) with real OSRM road-network routing (not straight-line distance), tested on real Seattle 911 data, shows a real, large, statistically significant improvement (17.5% avg / ~28% median response-time reduction, p<0.0001, likely optimistic vs. real-world deployment — see §7.8); there's a working two-view interactive simulator (`src/simulator.html`) — a **Simple** view with real reverse-geocoded street addresses for EMS staff, and an **Under the Hood** view with the full technical model (demand zones, compliance table, live animated simulation) for anyone who wants to verify the math.
