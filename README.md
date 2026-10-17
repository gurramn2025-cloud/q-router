# Q-Router

**One quantum engine for finance and healthcare decisions.** Q-Router turns yes/no business decisions from different industries into one shared quantum format (QUBO). It fingerprints each problem's structure and its expected noise on a real IBM chip, then learns which solving method fits which *shape* of problem. Each problem is routed to gate-based QAOA, a digitized quantum anneal, or a classical solver, and problems from different fields are solved side by side on one IBM quantum chip.

Built for the Florida Qiskit Fall Fest Hackathon 2026 (FAU).

## What's in this repo

| File | What it is |
|---|---|
| `Q_Router.ipynb` | The full pipeline. Open it in Google Colab and choose **Runtime → Run all** |
| `q_router.py` | The same code as a plain Python script (`# %%` cell markers), easier to read and diff |
| `requirements.txt` | Python packages |
| `results/` | Saved simulator results (CSVs, charts, full run log) from the 60-problem run |
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

**Simulator results (Oct 17, 2026, 60 generated problems, full run log in `results/`):**

- **Flagship problem (fund 4 of 10 drug candidates):** every method found the optimal set. Mean-shot scores: classical SA 0.956, QAOA 0.786, digital anneal 0.753 (1 = optimal, 0 = random guess).
- **Router:** predicts each solver's score with a mean error of 0.02–0.03, well below the real spread between problems. When choosing between the two quantum modes, it picks the better one 88% of the time and comes within 0.003 of a perfect oracle. Classical wins every problem at this size, and the router says so.
- **Cross-industry transfer (section 5d):** QAOA settings borrowed from a structural twin in *another* industry score 0.607 with one circuit run. A random donor from another industry scores 0.580, the generic default 0.581, and fully re-optimizing (~60 runs) 0.637. So the twin match recovers about half of the optimizer's gain at 1/60 of the cost, and a random donor recovers none. Transfer does not help healthcare problems.
- **Simultaneous solving (section 11, noisy simulator of IBM Torino):** three problems from three fields packed on 18 qubits score within 0.03 of running each alone.

Real-hardware results are pending.
