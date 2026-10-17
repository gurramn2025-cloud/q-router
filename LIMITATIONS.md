# Limitations

- **No quantum advantage.** At 6–16 qubits, classical simulated annealing beats both quantum methods, and our results show it.
- **Small instances.** Real portfolios have 50–200 candidates. Our exact baseline (brute force) only works up to ~20 variables.
- **Partly synthetic data.** Stock prices are real (Yahoo Finance) and the diagnostic dataset is real (scikit-learn breast cancer). The pharma pipeline is simulated, using approximate industry phase-success rates.
- **The "annealer" is digitized.** IBM has no annealing hardware, so annealing runs as a fixed-schedule QAOA circuit on a gate-based chip. A real D-Wave annealer could be added as another backend.
- **The router learns from few examples.** It trains on 60 generated problems, so its predictions carry real uncertainty. Its gain over always picking gate-QAOA is small (0.438 vs. 0.426).
- **Parameter transfer is partly known.** Tuned QAOA angles are known to carry over between similar problems ("parameter concentration"). What we add is testing it *across industries* with a random-donor control: the twin beats a random donor by 0.03 on average, on 73% of problems, and not at all for healthcare.
- **The noise model is simple.** Router labels use a global-depolarizing estimate (score ≈ fidelity × ideal score). Sections 7 and 10 test it against a noisy simulation and real hardware; it ignores crosstalk and coherent errors.
- **The best-shot metric is avoided.** With ~1,000 possible answers and 2,048 shots, the best single shot is nearly always optimal, so we report the *mean* sampled energy instead.
- **A simulator quirk we work around.** qiskit-aer's SamplerV2 corrupted the results of a smaller circuit batched after a larger one, which at first looked like a huge crosstalk effect. On simulators each circuit now runs as its own job; IBM Runtime handles batches correctly.
- **Limited hardware budget.** The Open Plan gives 10 QPU minutes per 28 days, so hardware experiments are single runs without repeated trials or error bars.
