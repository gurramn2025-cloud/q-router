# Limitations

- **No quantum advantage.** At 6–16 qubits, classical simulated annealing beats both quantum methods, and our results show it.
- **Small instances.** Real portfolios have 50–200 candidates. Our exact baseline (brute force) only works up to ~20 variables.
- **Partly synthetic data.** Stock prices are real (Yahoo Finance) and the diagnostic dataset is real (scikit-learn breast cancer). The pharma pipeline is simulated, using approximate industry phase-success rates.
- **The "annealer" is digitized.** IBM has no annealing hardware, so annealing runs as a fixed-schedule QAOA circuit on a gate-based chip. A real D-Wave annealer could be added as another backend.
- **The router learns from few examples.** It trains on 60 generated problems, so its predictions carry real uncertainty. Its gain over always picking gate-QAOA is small (0.436 vs. 0.423).
- **Parameter transfer is partly known.** Tuned QAOA angles are known to carry over between similar problems ("parameter concentration"). What we add is testing it *across industries* with a random-donor control: the twin beats a random donor by about 0.02 on average, on 65% of problems, and not at all for healthcare.
- **The noise model is simple.** Router labels use a global-depolarizing estimate (score ≈ fidelity × ideal score). Sections 7 and 10 test it against a noisy simulation and real hardware; it ignores crosstalk and coherent errors.
- **The best-shot metric is avoided.** With ~1,000 possible answers and 2,048 shots, the best single shot is nearly always optimal, so we report the *mean* sampled energy instead.
- **The noise model is too pessimistic for deep circuits.** The router's labels assume score ≈ fidelity × ideal score. For the many-layer anneal this badly underestimates real performance (section 6b: predicted 0.04, noisy simulation 0.45), which biases the router against the anneal. That is why it missed one of the three pieces in section 6b.
- **The break-down problem is one example.** Section 6b uses a single drug-funding + trial-site problem split into 3 pieces, so it illustrates the pattern rather than proving it. The 60-problem dataset in section 5 carries the statistical weight.
- **Sharing settings costs some quality.** In section 11b, one tuning per shape family loses about 0.02 on average, almost all of it on a network problem grouped with finance problems.
- **The entanglement test is on a perfect simulator.** It shows entanglement matters for QAOA's score, not that it gives any advantage over classical methods.
- **A simulator quirk we work around.** qiskit-aer's SamplerV2 corrupted the results of a smaller circuit batched after a larger one, which at first looked like a huge crosstalk effect. On simulators each circuit now runs as its own job; IBM Runtime handles batches correctly.
- **On real hardware, the most common answer was not the best one.** The true best drug set appeared in only about 1% of shots (2–3× random guessing); a laptop finds it every time.
- **Our device model was too optimistic for long circuits.** The noisy simulation predicted 0.67 for the anneal; the real chip gave 0.55.
- **Error mitigation did not help.** Dynamical decoupling and Pauli twirling lowered real-chip scores, most for the long annealing circuit (0.55 → 0.36). Zero-noise extrapolation was only demonstrated in simulation.
- **Limited hardware budget.** The Open Plan gives 10 QPU minutes per 28 days, so each hardware experiment was run once, without repeated trials or error bars.
