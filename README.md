# Q-Router

**One quantum engine for finance and healthcare decisions.** Q-Router turns yes/no business decisions from different industries into one shared quantum format (QUBO). It fingerprints each problem's structure and its expected noise on a real IBM chip, then learns which solving method fits which *shape* of problem. Each problem is routed to gate-based QAOA, a digitized quantum anneal, or a classical solver, and problems from different fields are solved side by side on one IBM quantum chip.

Built for the Florida Qiskit Fall Fest Hackathon 2026 (FAU).

## What's in this repo

| File | What it is |
|---|---|
| `Q_Router.ipynb` | The full pipeline. Open it in Google Colab and choose **Runtime → Run all** |
| `q_router.py` | The same code as a plain Python script (`# %%` cell markers), easier to read and diff |
| `requirements.txt` | Python packages |
| `USE-CASE.md` | The problem, who it affects, and why it matters |
| `LIMITATIONS.md` | What this project does not prove |
| `docs/PITCH_PLAN.md` | Pitch strategy, quantum theory notes, error mitigation plan |
| `docs/NEXT_STEPS.md` | Build plan, QPU budget, and how the project maps to the judging rubric |

## Pipeline (notebook sections)

1. **QUBO toolkit:** penalties, exact brute-force solver, approximation ratio (1 = optimal, 0 = random guess)
2. **Problem templates:** healthcare-stock portfolio (finance), diagnostic-test selection (healthcare), pharma R&D pipeline (finance × healthcare), Max-Cut (networks)
3. **Solvers:** gate-based QAOA, digitized anneal (linear-ramp QAOA, no optimizer loop), classical simulated annealing
4. **Profiler:** structural fingerprint plus expected circuit fidelity from the device's calibration data
5. **Pattern-recognition engine:** a multi-output random forest predicts every solver's score from the fingerprint alone
    - 5b: cross-field pattern map and leave-one-field-out transfer test
    - 5c: router vs. always-one-solver vs. oracle benchmark
    - 5d: zero-shot transfer of tuned QAOA settings between industries
6. **Router and decomposer:** spectral clustering plus block coordinate descent for large problems
7. **Error mitigation demo:** zero-noise extrapolation on a noisy simulator
8. **Roadmap:** which industries cross over to quantum first as hardware errors fall
9. **Real IBM hardware:** both modes, raw vs. mitigated (dynamical decoupling and Pauli twirling)
10. **Noise cliff:** scaling sweep from 6 to 16 qubits on hardware, batched into 2 jobs
11. **Simultaneous multi-field solving:** finance, healthcare and pharma problems packed on one chip in one job, with crosstalk measured
12. **Q-Router app:** a Gradio web demo with a public link

## Running it

- **Simulators only (free):** open the notebook in Colab and choose **Runtime → Run all**. The three hardware flags default to `False`.
- **Real IBM hardware:** set `RUN_ON_HARDWARE = True` in section 9 (then `RUN_SWEEP` and `RUN_SIMULTANEOUS_ON_HARDWARE`), and paste your IBM Quantum API key and instance CRN when prompted. **Never commit your API key to this repo.** All hardware runs together use roughly 3–4 minutes of the Open Plan's 10 minutes per 28 days.

## Status

Simulator results confirmed on Oct 17, 2026, for the pharma test problem (10 candidates). All methods found the optimal funded set, with mean-shot approximation ratios of classical SA 0.953, QAOA 0.786, and digital anneal 0.748. Hardware results are pending.
