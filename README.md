# Every Second Counts

Quantum-optimized EMS dispatch. Two ambulance crews, multiple simultaneous 911 calls,
one question: which crew goes where, in what order, to minimize total response time?

## What it does

1. Describe incoming 911 calls in plain English.
2. An LLM extracts location and severity from each description.
3. Calls are geocoded onto a real map.
4. QAOA (Quantum Approximate Optimization Algorithm), run on a real quantum circuit
   simulator, splits the calls between two crews to minimize total response time.

Dispatch-splitting is mathematically identical to Max-Cut — exactly the class of
problem QAOA solves. At this problem size a classical computer solves it instantly
too; the point is a correct, working quantum implementation built the way it needs
to work as hardware scales past what classical brute-force can touch.

## Status

Early scaffold. Pitch script and architecture are set; implementation in progress.

## Stack (planned)

- LLM call parsing → geocoding
- QAOA via a quantum circuit simulator (Qiskit or similar)
- Map + circuit/Bloch-sphere visualization for the live demo

## License

TBD

---

## Predictive Relocation (companion project)

A second, working approach in the same problem space: instead of optimizing
*which crew responds to which call* in the moment, this model predicts
*where ambulances should be staged before calls come in*.

- **MEXCLP compliance-table model** (Daskin 1983) — the actual published
  method real EMS System Status Management uses — with real OSRM
  road-network routing (not straight-line distance).
- Tested on **100% real data**: 14,070 real Seattle Fire 911 calls over a
  real 60-day window.
- **Result: 17.5% avg / ~28% median response-time improvement**, p=2.27e-168.
- Two-view interactive simulator: a **Simple** view (real reverse-geocoded
  street addresses, ranked posts) for EMS staff, and an **Under the Hood**
  view (demand zones, compliance table, live animated simulation) for
  verification.
- Full methodology, including dead ends and disclosed assumptions, in
  [`predictive-relocation/HANDOFF.md`](predictive-relocation/HANDOFF.md).

Run it:
```bash
cd predictive-relocation
python3 src/simulate_dynamic.py   # regenerate the simulation (matrices already cached)
python3 -m http.server 8765       # serve from this directory
open http://localhost:8765/src/simulator.html
```
