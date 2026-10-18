# Q-Router

**One quantum engine for finance and healthcare decisions.** Q-Router turns yes/no business decisions from different industries into one shared quantum format (QUBO). It fingerprints each problem's structure and its expected noise on a real IBM chip, then learns which solving method fits which *shape* of problem. Each problem is routed to gate-based QAOA, a digitized quantum anneal, or a classical solver, and problems from different fields are solved side by side on one IBM quantum chip.

Built for the Florida Qiskit Fall Fest Hackathon 2026 (FAU).

**The problem it addresses:** quantum time is scarce and noisy, and there are two kinds of quantum computer (gate-based and annealing). Nobody can easily tell whether a decision should go to quantum at all, which kind, or for which part. Q-Router answers that with data, and says honestly when a classical computer is still better. See `USE-CASE.md`.

## What's in this repo

| File | What it is |
|---|---|
| `Q_Router.ipynb` | The full pipeline. Open it in Google Colab and choose **Runtime → Run all** |
| `q_router.py` | The same code as a plain Python script (`# %%` cell markers), easier to read and diff |
| `requirements.txt` | Python packages |
| `slides/Q-Router_slides.pptx` | The 10-slide presentation, with speaker notes |
| `slides/Q-Router_complete_guide.pptx` | A 58-slide guide explaining every step, decision, test and chart in plain language |
| `results/` | Saved simulator results (CSVs, charts, full run log) |
| `USE-CASE.md` | The problem, who it affects, and why it matters |
| `LIMITATIONS.md` | What this project does not prove |

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
    - 6b: **break it down, compare both:** one drug-funding + trial-site problem split into pieces, each piece solved by *both* gate-based QAOA and the digital anneal
7. **Error mitigation demo:** zero-noise extrapolation on a noisy simulator
    - 7b: **is entanglement doing any work?** entanglement entropy of the QAOA state, and the score with every entangling gate removed
8. **Roadmap:** which industries cross over to quantum first as hardware errors fall
9. **Real IBM hardware:** both modes, raw vs. mitigated (dynamical decoupling and Pauli twirling)
    - 9b: the section 6b pieces, both quantum types, on real hardware in one job
10. **Noise cliff:** scaling sweep from 6 to 16 qubits on hardware, batched into 2 jobs
11. **Simultaneous multi-field solving:** finance, healthcare and pharma problems packed on one chip in one job, with crosstalk measured
    - 11b: **one tune, one job:** group problems by shape, tune once per group, pack them all into one chip job, plus a compute ledger
12. **Q-Router app:** a Gradio web demo with a public link

## Running it

- **Simulators only (free):** open the notebook in Colab and choose **Runtime → Run all**. Every hardware flag defaults to `False`. A full run takes roughly 15–25 minutes.
- **Real IBM hardware:** set `RUN_ON_HARDWARE = True` in section 9 and paste your IBM Quantum API key and instance CRN when prompted. Then turn on, in order, `RUN_BREAKDOWN_ON_HARDWARE` (9b), `RUN_SIMULTANEOUS_ON_HARDWARE` (11), `RUN_ONE_TUNE_ON_HARDWARE` (11b) and, if budget allows, `RUN_SWEEP` (10). Every hardware cell was dry-run against a fake IBM Torino before use. **Never commit your API key to this repo.** All hardware runs together use roughly 3–4 minutes of the Open Plan's 10 minutes per 28 days.

## Status

**Simulator results (Oct 17, 2026, 60 generated problems; full run log and charts in `results/`):**

- **Flagship problem (fund 4 of 10 drug candidates):** every method found the optimal set. Mean-shot scores: classical SA 0.948, QAOA 0.782, digital anneal 0.759 (1 = optimal, 0 = random guess).
- **Router:** predicts each solver's score with a mean error of 0.01–0.03, well below the real spread between problems. Choosing between the two quantum modes, it picks the better one 90% of the time and comes within 0.002 of a perfect oracle. Classical wins every problem at this size, and the router says so.
- **When each quantum type works (sections 5 and 6b):** on dense problems, where every decision affects every other (finance, pharma), gate-based QAOA wins almost every time (the anneal wins 0 of 11 finance and 2 of 16 pharma problems). On sparser network problems the anneal catches up and wins 56% of the time. Why: the anneal needs several layers, so on dense problems it piles up too many noisy entangling gates.
- **Break it down, compare both (section 6b):** one drug-funding + trial-site problem split into 3 pieces, each solved both ways on a noisy IBM Torino model. Gate QAOA wins the dense drug-funding piece (0.56 vs. 0.45); the anneal wins both trial-site network pieces (0.44 vs. 0.37, 0.69 vs. 0.42). The router picked correctly on 2 of 3 pieces, and its mix (0.539) beats all-gate (0.450) and all-anneal (0.524).
- **Cross-industry transfer (section 5d):** QAOA settings borrowed from a structural twin in *another* industry score 0.599 with one circuit run. A random donor scores 0.577, the generic default 0.581, and fully re-optimizing (~60 runs) 0.633. So the twin match recovers about half of the optimizer's gain at 1/60 of the cost; a random donor recovers none. Transfer does not help healthcare problems.
- **One tune, one job (section 11b):** four problems from four industries, grouped into 2 shape families, tuned once per family and packed into one circuit: 2 tunings instead of 4, 55 tuning runs instead of 114, 1 chip job instead of 4, 4× fewer QPU shots. Quality: 0.504 vs. 0.524 the normal way. Packing itself costs nothing; the loss comes from sharing settings (mostly the network problem).
- **Entanglement (section 7b):** the QAOA state for the 10-drug problem carries about 1 bit of entanglement entropy. Removing every entangling gate drops the score from 0.773 to 0.308.
- **Simultaneous solving (section 11):** three problems from three fields packed on 18 qubits score within 0.03 of running each alone (noisy simulator).
- **Error mitigation (section 7):** zero-noise extrapolation recovers about a quarter of the energy lost to noise (noisy simulator).

Real-hardware results are pending. Every hardware cell has been dry-run against a fake IBM Torino.

## Prior work

Reusing tuned QAOA settings between problems is known: parameter concentration (Brandão et al., 2018), similarity-based transfer between graphs (Galda et al., 2023, arXiv:2307.05420), and machine-learned donor selection across problem types (arXiv:2504.10733, 2025). Running several circuits on one chip is known as multi-programming (e.g. arXiv:2102.05321). Q-Router's contribution is testing transfer across *real industry decisions* built from real data, with a random-donor control, using a chip-noise-aware fingerprint, and combining transfer, routing and multi-problem packing into one pipeline that measures the compute it saves.
