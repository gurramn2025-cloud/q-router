# Use case

## The problem
Quantum computing is now rented on the cloud, and there are two very different kinds: **gate-based** machines (IBM, Google) and **annealers** (D-Wave). Quantum time is scarce (IBM's free plan gives 10 minutes per 28 days, and paid time is billed by the second), and today's chips are noisy. Send the wrong problem, or the wrong *part* of a problem, to the wrong machine and the answer is no better than a random guess.

So anyone with a hard decision faces three questions nobody answers well today:
1. **Is quantum worth using at all** for this problem, on today's hardware?
2. **Which kind** of quantum computing fits it?
3. **For which part** of the problem?

**Q-Router answers them with data.** It breaks a decision into pieces, fingerprints each piece (its structure plus the real chip's expected noise), and a model trained on 60 practice problems routes each piece to gate-based QAOA, a digitized anneal, or a classical solver. It saves quantum time by reusing tuned settings from structurally similar problems in other industries and by packing several problems into one chip job.

## Who it affects
- **Companies experimenting with quantum** (pharma R&D, finance): they pay for quantum time and need to know where it helps.
- **Quantum cloud providers** (IBM, D-Wave, Amazon Braket): they offer several kinds of hardware and must decide where each job should run. This fits the 2026 theme, *a decade of quantum on the cloud*.
- **Researchers and students** with small quantum budgets.

## How it's done today
Users mostly guess, follow vendor defaults, or fill in readiness questionnaires such as the Next-Step Quantum Decision Guide. D-Wave's hybrid solvers split work between classical and quantum internally, but there is no open, data-driven way to choose between gate-based and annealing approaches piece by piece.

## Flagship decision: which drug candidates to fund
*Which drug candidates should a pharmaceutical company fund this cycle to get the most risk-adjusted value, without betting too much on drugs that tend to fail together?*

- **Stakeholders:** R&D portfolio committees, biotech investors, and ultimately patients (an unfunded candidate is a therapy that never reaches them)
- **Variables:** one yes/no per candidate (fund or not)
- **Objective:** maximize Σ (probability of approval × peak value − cost), minus a penalty for correlated risk (candidates in the same therapeutic area fail together)
- **Constraint:** fund exactly *k* candidates (capacity or budget), added as the penalty P·(Σx − k)²
- **Math:** Markowitz portfolio selection applied to healthcare assets, the same structure used to pick stock portfolios
- **Data:** simulated pipeline using approximate industry likelihood-of-approval by phase (Phase I ≈ 8%, Phase II ≈ 15%, Phase III ≈ 55%)
- **Today:** committees combine expert judgment, spreadsheet scoring and classical optimizers; at realistic sizes (50–200 candidates) classical solvers work well

The same QUBO format covers the other three fields Q-Router learns from:

| Field | Problem | Data |
|---|---|---|
| Finance | Healthcare-stock portfolio | Real daily prices of 12 healthcare stocks, 2023–2025 (Yahoo Finance) |
| Healthcare | Choose a small, non-redundant set of diagnostic tests | Real Wisconsin breast-cancer dataset (scikit-learn) |
| Networks | Split a network into two groups (Max-Cut), e.g. trial sites between two research partners | Simulated |

## Success metrics (defined before the runs)
- **Answer quality:** approximation ratio of the *mean* sampled energy, 1 = the exact optimum (brute force), 0 = a uniformly random guess
- **Routing quality:** how often the router picks the better quantum mode, and its regret against a perfect oracle
- **Compute used:** tuning runs, circuits, chip jobs and QPU shots (the compute ledger)
- **Hardware fidelity:** perfect simulator vs. noisy device model vs. real QPU, raw and mitigated

## What it would take to be real
- **Problem size:** real portfolios have 50–200 candidates; our demos use 3–18 qubits, and the exact baseline only works up to about 20 variables.
- **Hardware:** roughly 10–100× lower two-qubit error rates, or qubit-efficient encodings, before quantum methods could compete with classical solvers on these problems.
- **Data:** real pipeline data (ClinicalTrials.gov, company disclosures) instead of simulated candidates.
- **Backends:** a real annealer (D-Wave) as a third backend instead of only the digitized anneal on IBM.
- **Routing model:** a noise model that does not underestimate deep circuits, and many more training problems.

Until then, Q-Router's honest recommendation for these problems is the classical solver, and it is built to switch as hardware improves.
