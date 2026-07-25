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
