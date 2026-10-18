# %% [markdown]
# # Q-Router: routing finance + healthcare optimization between a quantum annealer and gate-based QAOA
#
# **Pipeline:** real data → QUBO → structural profile → (decompose) → learned router → annealer **or** QAOA on IBM hardware (with error mitigation) → honest benchmark vs. the exact optimum.
#
# Run top to bottom in Google Colab. Everything works on simulators. The IBM hardware cell is optional and uses only a few seconds of QPU time.

# %%
# (Colab only) !pip -q install "qiskit>=1.3" qiskit-ibm-runtime qiskit-aer dwave-samplers dimod scikit-learn yfinance matplotlib

# %%
import time, json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

# %% [markdown]
# ## 1. QUBO toolkit
# A QUBO is stored as an **upper-triangular** matrix `Q`: energy = xᵀQx with x ∈ {0,1}ⁿ (the diagonal holds the linear terms).

# %%
def to_upper(M):
    """Fold any quadratic form xᵀMx into an upper-triangular QUBO."""
    U = np.triu(M + M.T)
    np.fill_diagonal(U, np.diag(M))
    return U

def energy(Q, x):
    return float(x @ Q @ x)

def default_penalty(Q):
    # Larger than the biggest change any single bit flip can cause → constraint violations never pay off
    return float(np.abs(Q + Q.T).sum(axis=1).max()) + 1e-6

def add_cardinality(Q, k, P):
    """Add P·(Σx − k)²  (choose exactly k items). Returns (Q, constant offset)."""
    n = len(Q); Q = Q.copy()
    Q[np.triu_indices(n, 1)] += 2 * P
    Q[np.diag_indices(n)] += P * (1 - 2 * k)
    return Q, P * k * k

def solve_exact(Q):
    """Brute force (n ≤ ~20). Returns best x, all energies (for approximation ratios)."""
    n = len(Q)
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    E = np.einsum("bi,ij,bj->b", X, Q, X)
    return X[E.argmin()], E

def approx_ratio(e, E_all):
    """1 = optimal, 0 = no better than a uniformly random guess (negative = worse than random).
    Measured against random rather than the worst bitstring, because huge penalty energies
    would otherwise make every answer look ~0.99."""
    lo, rnd = E_all.min(), E_all.mean()
    return float((rnd - e) / (rnd - lo)) if rnd > lo else 1.0

# %% [markdown]
# ## 2. Three problem templates: finance, healthcare, and the bridge between them

# %%
def markowitz_qubo(mu, sigma, k, risk=0.5):
    """min  risk·xᵀΣx − μᵀx   s.t. Σx = k"""
    mu = mu / np.abs(mu).max(); sigma = sigma / np.abs(sigma).max()
    Q = to_upper(risk * sigma - np.diag(mu))
    return add_cardinality(Q, k, default_penalty(Q))[0]

# --- Finance: healthcare-sector stock portfolio (real prices, synthetic fallback) ---
TICKERS = ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "CVS", "HCA", "CI", "ISRG"]
_RETURNS = {}
def load_returns(tickers=TICKERS):
    key = tuple(tickers)
    if key not in _RETURNS: _RETURNS[key] = _download_returns(tickers)
    return _RETURNS[key]

def _download_returns(tickers):
    try:
        import yfinance as yf
        px = yf.download(tickers, start="2023-01-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
        r = px.dropna(axis=1).pct_change().dropna()
        assert r.shape[1] >= 6
        return r
    except Exception as e:
        print("yfinance unavailable, using synthetic returns:", e)
        n = len(tickers); A = rng.normal(size=(n, n)) * 0.01
        return pd.DataFrame(rng.multivariate_normal(rng.normal(5e-4, 3e-4, n), A @ A.T + np.eye(n) * 1e-4, 750), columns=tickers)

def portfolio_problem(n=10, k=4):
    r = load_returns().iloc[:, :n]
    return markowitz_qubo(r.mean().values * 252, r.cov().values * 252, k), list(r.columns)

# --- Healthcare: choose a small, non-redundant set of clinical biomarkers (breast-cancer dataset) ---
def feature_selection_problem(n=10, k=None, alpha=0.6, lam=0.15, sparsify=0.0):
    data = load_breast_cancer(); X = StandardScaler().fit_transform(data.data[:, :n]); y = data.target
    rel = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n)])          # relevance to diagnosis
    red = np.abs(np.corrcoef(X, rowvar=False)); np.fill_diagonal(red, 0)      # redundancy between tests
    red[red < sparsify] = 0                                                   # drop weak links → sparser QUBO
    Q = to_upper(alpha * red / 2 - np.diag(rel) + lam * np.eye(n))
    if k is not None:
        Q = add_cardinality(Q, k, default_penalty(Q))[0]
    return Q, list(data.feature_names[:n]), (X, y)

def feature_accuracy(x, X, y):
    cols = np.flatnonzero(x)
    if len(cols) == 0: return 0.0
    return cross_val_score(LogisticRegression(max_iter=500), X[:, cols], y, cv=5).mean()

# --- Bridge: pharma R&D pipeline portfolio (finance math on healthcare assets) ---
PHASE_SUCCESS = {"Phase I": 0.08, "Phase II": 0.15, "Phase III": 0.55}   # illustrative likelihood-of-approval
def pharma_problem(n=10, k=4, seed=None):
    g = np.random.default_rng(seed)
    phase = g.choice(list(PHASE_SUCCESS), n); p = np.array([PHASE_SUCCESS[s] for s in phase])
    area = g.choice(["oncology", "cardio", "neuro", "immunology"], n)
    value = g.lognormal(mean=6.5, sigma=0.6, size=n)                  # peak value if approved ($M)
    cost = np.array([{"Phase I": 30, "Phase II": 80, "Phase III": 300}[s] for s in phase]) * g.uniform(.7, 1.3, n)
    mu = p * value - cost                                             # risk-adjusted expected value
    sd = value * np.sqrt(p * (1 - p))
    rho = np.where(area[:, None] == area[None, :], 0.6, 0.1); np.fill_diagonal(rho, 1)
    Q = markowitz_qubo(mu, rho * np.outer(sd, sd), k)
    names = [f"{a[:4]}-{s.split()[1]}-{i}" for i, (a, s) in enumerate(zip(area, phase))]
    return Q, names

# --- Sparse graph problem (Max-Cut, e.g. splitting a hospital referral or fraud network) ---
def maxcut_problem(n=8, degree=3, seed=None):
    g = np.random.default_rng(seed); Q = np.zeros((n, n))
    for i in range(n):
        for j in g.choice([v for v in range(n) if v != i], size=min(degree, n - 1) // 2 + 1, replace=False):
            a, b = sorted((i, j)); w = g.uniform(0.5, 1.5)
            Q[a, a] -= w; Q[b, b] -= w; Q[a, b] += 2 * w     # cut(i,j) = x_i + x_j − 2 x_i x_j
    return Q, [f"node{i}" for i in range(n)]

# %% [markdown]
# ## 3. Solvers
# **Annealer:** simulated annealing from D-Wave's own SDK. To use a real D-Wave QPU, replace the sampler with `EmbeddingComposite(DWaveSampler())` from `dwave-system` (needs a Leap token).
#
# **Gate model:** QAOA. QUBO → Ising via xᵢ = (1 − Zᵢ)/2, then `QAOAAnsatz`. The optimizer minimizes the **mean** energy, the same metric we report. Setting `alpha < 1` switches it to **CVaR** (the mean of the best α fraction of shots): more noise-tolerant, but it can leave the typical answer poor.

# %%
def solve_annealer(Q, num_reads=200, sampler=None):
    n = len(Q)
    bqm = dimod.BinaryQuadraticModel.from_qubo({(i, j): Q[i, j] for i in range(n) for j in range(i, n) if Q[i, j] != 0})
    ss = (sampler or SimulatedAnnealingSampler()).sample(bqm, num_reads=num_reads)
    xs = np.array([[s.get(i, 0) for i in range(n)] for s in ss.samples()], dtype=float)
    es = np.array([energy(Q, x) for x in xs])
    return xs[es.argmin()], es   # best solution, energies of every read

def qubo_to_ising(Q):
    n = len(Q); h = np.zeros(n); terms = []; offset = 0.0
    for i in range(n):
        h[i] -= Q[i, i] / 2; offset += Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                terms.append(("ZZ", [i, j], Q[i, j] / 4))
                h[i] -= Q[i, j] / 4; h[j] -= Q[i, j] / 4; offset += Q[i, j] / 4
    terms += [("Z", [i], h[i]) for i in range(n) if abs(h[i]) > 1e-12]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()
    return SparsePauliOp(H.paulis, H.coeffs / np.abs(H.coeffs).max()), offset   # normalized for stable angles

def bits_to_x(b):            # Qiskit bitstrings are little-endian: qubit 0 is the rightmost character
    return np.array([int(c) for c in b[::-1]], dtype=float)

def counts_energies(counts, Q):
    return [(energy(Q, bits_to_x(b)), c) for b, c in counts.items()]

def cvar(counts, Q, alpha=0.25):
    items = sorted(counts_energies(counts, Q)); keep = alpha * sum(c for _, c in items); acc = s = 0.0
    for e, c in items:
        take = min(c, keep - acc); s += e * take; acc += take
        if acc >= keep: break
    return s / keep

def mean_energy(counts, Q):
    items = counts_energies(counts, Q); return sum(e * c for e, c in items) / sum(c for _, c in items)

def annealing_init(reps):
    # QAOA ≈ Trotterized annealing: β ramps down, γ ramps up (parameter order: β's then γ's)
    t = (np.arange(reps) + 0.5) / reps
    # γ is NEGATIVE: QAOAAnsatz starts in |+⟩, the TOP state of its ΣX mixer, so with β, γ > 0 the anneal
    # tracks the HIGHEST-energy state. Opposite signs make it track the lowest (= best decision).
    return np.concatenate([0.6 * (1 - t), -0.6 * t])

def build_qaoa(Q, reps):
    H, _ = qubo_to_ising(Q)
    ansatz = QAOAAnsatz(H, reps=reps); ansatz.measure_all()
    return ansatz

LOCAL = AerSimulator()
def solve_qaoa(Q, reps=2, shots=2048, maxiter=120, backend=LOCAL, alpha=1.0):
    ansatz = build_qaoa(Q, reps)
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(ansatz)
    sampler = Sampler(mode=backend)
    run = lambda th: sampler.run([(isa, th)], shots=shots).result()[0].data.meas.get_counts()
    res = minimize(lambda th: cvar(run(th), Q, alpha), annealing_init(reps), method="COBYLA", options={"maxiter": maxiter})
    counts = run(res.x)
    best = min(counts, key=lambda b: energy(Q, bits_to_x(b)))
    return dict(x=bits_to_x(best), counts=counts, params=res.x, mean_energy=mean_energy(counts, Q), evals=int(res.nfev))
def is_local(backend):
    """True for simulators and fake devices, which run on Aer on this computer."""
    return isinstance(backend, AerSimulator) or "fake" in str(getattr(backend, "name", "")).lower()

def mitigated_sampler(backend, shots=4096, mitigate=True):
    s = Sampler(mode=backend); s.options.default_shots = shots
    if mitigate and not is_local(backend):
        s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True
    return s

def run_pubs(sampler, pubs, backend):
    """One batched job on hardware. Locally, one job per circuit: qiskit-aer 0.17 corrupts the results
    of a narrower circuit batched after a wider one (seen in section 11's crosstalk numbers)."""
    if is_local(backend): return [sampler.run([pub]).result()[0] for pub in pubs]
    return sampler.run(pubs).result()

# %% [markdown]
# ### Annealing mode on IBM hardware: digitized quantum annealing (linear-ramp QAOA)
# IBM has no annealer, but an anneal can be **digitized**: chop the slow sweep from the mixer Hamiltonian to the problem Hamiltonian into p small steps (Trotterization). Each step is one QAOA layer with a **fixed** schedule: β ramps down, γ ramps up. There is **no classical optimizer loop**, so it costs one hardware job.
# This is the "LR-QAOA" approach that has been run on IBM hardware at 100+ qubits.
#
# | Mode | Parameters | Layers | Hardware jobs |
# |---|---|---|---|
# | **Gate-based logic** (variational QAOA) | optimized by COBYLA on the simulator | few (p = 1–2) | 1 (parameters transferred) |
# | **Annealer** (digitized anneal) | fixed linear ramp, only the step size Δ is tuned | more (p = 3–6) | 1 |
# | **Classical baseline** (simulated annealing) | n/a | n/a | 0 |

# %%
def lr_params(p, delta=0.6):
    i = np.arange(p)
    return np.concatenate([(1 - i / p) * delta, -(i + 1) / p * delta])   # β ramps down, |γ| ramps up (γ < 0: see annealing_init)

def solve_digital_anneal(Q, p=4, delta=0.6, shots=2048, backend=LOCAL):
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(build_qaoa(Q, p))
    counts = Sampler(mode=backend).run([(isa, lr_params(p, delta))], shots=shots).result()[0].data.meas.get_counts()
    best = min(counts, key=lambda b: energy(Q, bits_to_x(b)))
    return dict(x=bits_to_x(best), counts=counts, params=lr_params(p, delta), mean_energy=mean_energy(counts, Q))

def tune_anneal(Q, ps=(2, 3, 4, 6), deltas=(0.3, 0.6, 0.9, 1.2)):
    """Small grid search on the simulator: pick the schedule (p, Δ) with the best mean energy."""
    grid = [(solve_digital_anneal(Q, p, d, shots=1024)["mean_energy"], p, d) for p in ps for d in deltas]
    _, p, d = min(grid); return p, d

# %% [markdown]
# ### Quick check on the pharma R&D problem (10 drug candidates, fund 4)

# %%
Q, names = pharma_problem(n=10, k=4, seed=1)
x_opt, E_all = solve_exact(Q)
x_sa, e_sa = solve_annealer(Q)
t0 = time.time(); q = solve_qaoa(Q, reps=2); t_q = time.time() - t0
print("Optimal funded set :", [names[i] for i in np.flatnonzero(x_opt)])
print("Annealer best      :", [names[i] for i in np.flatnonzero(x_sa)], "| mean-read ratio %.3f" % approx_ratio(e_sa.mean(), E_all))
print("QAOA best shot     :", [names[i] for i in np.flatnonzero(q['x'])], "| mean-shot ratio %.3f (%.0fs)" % (approx_ratio(q['mean_energy'], E_all), t_q))
p_a, d_a = tune_anneal(Q); da = solve_digital_anneal(Q, p_a, d_a)
print("Digital anneal     :", [names[i] for i in np.flatnonzero(da['x'])], "| mean-shot ratio %.3f (p=%d, Δ=%.1f)" % (approx_ratio(da['mean_energy'], E_all), p_a, d_a))
print("(We score the MEAN sampled energy. With 2,048 shots and only 1,024 bitstrings, 'best shot' would be cheating.)")

# %% [markdown]
# ## 4. Profiler: what a problem "looks like" to quantum hardware
# The key physics feature is the **expected circuit fidelity** after compiling QAOA to a real IBM device's heavy-hex layout, computed from the device's calibration data: F = Π(1 − εᵢ) over every gate and measurement.

# %%
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino as FakeDevice
except ImportError:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke as FakeDevice
DEVICE = FakeDevice()

def expected_fidelity(Q, reps=1, backend=DEVICE, error_scale=1.0):
    isa = generate_preset_pass_manager(optimization_level=2, backend=backend, seed_transpiler=1).run(build_qaoa(Q, reps))
    t = backend.target; logF = 0.0; twoq = 0
    for inst in isa.data:
        name = inst.operation.name; qs = tuple(isa.find_bit(b).index for b in inst.qubits)
        if name in t and qs in t[name] and t[name][qs] is not None and t[name][qs].error is not None:
            logF += np.log1p(-min(t[name][qs].error * error_scale, 0.999))
        twoq += len(qs) == 2 and name != "barrier"
    return float(np.exp(logF)), int(twoq), int(isa.depth())

def profile(Q, reps=1):
    n = len(Q); off = np.abs(np.triu(Q, 1)); A = (off + off.T) > 1e-9
    vals = np.abs(Q[np.abs(Q) > 1e-9])
    F, twoq, depth = expected_fidelity(Q, reps)
    return dict(n=n, density=A.sum() / (n * (n - 1)), max_degree=int(A.sum(1).max()), mean_degree=float(A.sum(1).mean()),
                dyn_range=float(np.log10(vals.max() / vals.min())), log_fidelity=np.log(F), two_qubit_gates=twoq, depth=depth)

print(pd.DataFrame([profile(pharma_problem(8, 3, seed=0)[0]), profile(maxcut_problem(8, seed=0)[0]),
                    profile(feature_selection_problem(8, sparsify=0.5)[0])], index=["pharma (dense)", "max-cut (sparse)", "biomarkers (sparsified)"]).round(3))

# %% [markdown]
# ## 5. The pattern-recognition engine: learn which solver fits which problem *shape*
# We generate problems from **four different fields** and run each one through all three solvers:
#
# | Field | Problem | Example decision |
# |---|---|---|
# | Finance | healthcare-stock portfolio | which 4 of 10 stocks to hold |
# | Healthcare | biomarker / diagnostic-test selection | which tests to order for a diagnosis |
# | Finance × Healthcare | pharma R&D pipeline portfolio | which drugs to fund |
# | Networks / logistics | Max-Cut | how to split a referral or fraud network |
#
# Each solver's score on hardware is estimated with the global-depolarizing model, **score ≈ F × ideal ratio** (F = expected circuit fidelity on the real chip). Section 7 checks this against a full noisy simulation, and section 10 checks it against real hardware.
# A multi-output random forest then learns **fingerprint → score of every solver**. It never sees which field a problem came from, only its shape.

# %%
FIELD = {"portfolio": "finance", "biomarkers": "healthcare", "pharma": "finance×healthcare", "maxcut": "networks"}
ANNEAL_P, ANNEAL_DELTA = 3, 0.6        # default digitized-anneal schedule for the dataset

def profile(Q):
    n = len(Q); off = np.abs(np.triu(Q, 1)); A = (off + off.T) > 1e-9
    vals = np.abs(Q[np.abs(Q) > 1e-9])
    F1, twoq, depth = expected_fidelity(Q, 1)
    Fa, _, _ = expected_fidelity(Q, ANNEAL_P)
    return dict(n=n, density=A.sum() / (n * (n - 1)), max_degree=int(A.sum(1).max()), mean_degree=float(A.sum(1).mean()),
                dyn_range=float(np.log10(vals.max() / vals.min())), log_fidelity=np.log(F1), log_fidelity_anneal=np.log(Fa),
                two_qubit_gates=twoq, depth=depth)

def random_instance(g):
    fam = g.choice(list(FIELD)); n = int(g.integers(5, 10))
    if fam == "pharma": Q = pharma_problem(n, max(2, n // 3), seed=int(g.integers(1e9)))[0]
    elif fam == "portfolio": Q = portfolio_problem(n, max(2, n // 3))[0]
    elif fam == "biomarkers": Q = feature_selection_problem(n, sparsify=float(g.uniform(0, .8)))[0]
    else: Q = maxcut_problem(n, degree=int(g.integers(2, 4)), seed=int(g.integers(1e9)))[0]
    return fam, Q

rows = []; QS = []; GATE_PARAMS = []; g = np.random.default_rng(42)   # QS / GATE_PARAMS: each problem + its tuned QAOA angles
N_INSTANCES = 60          # ~2 min on a laptop; 12 gives noisy, unreliable router statistics
for it in range(N_INSTANCES):
    fam, Q = random_instance(g)
    _, E_all = solve_exact(Q)
    feats = profile(Q)
    t0 = time.time(); qg = solve_qaoa(Q, reps=1, shots=1024, maxiter=60); t_gate = time.time() - t0
    r_gate = approx_ratio(qg["mean_energy"], E_all); QS.append(Q); GATE_PARAMS.append(qg["params"])
    t0 = time.time(); r_ann = approx_ratio(solve_digital_anneal(Q, ANNEAL_P, ANNEAL_DELTA, shots=1024)["mean_energy"], E_all); t_ann = time.time() - t0
    t0 = time.time(); _, e_sa = solve_annealer(Q, num_reads=100); t_cls = time.time() - t0
    rows.append(dict(family=fam, field=FIELD[fam], **feats, r_gate_ideal=r_gate, r_anneal_ideal=r_ann,
                     time_gate=t_gate, time_anneal=t_ann, time_classical=t_cls,
                     score_gate=np.exp(feats["log_fidelity"]) * r_gate,
                     score_anneal=np.exp(feats["log_fidelity_anneal"]) * r_ann,
                     score_classical=approx_ratio(e_sa.mean(), E_all)))
    r = rows[-1]
    print(f"{it+1:3d}/{N_INSTANCES} {r['field']:18s} n={r['n']}  gate={r['score_gate']:.2f}  anneal={r['score_anneal']:.2f}  classical={r['score_classical']:.2f}")
df = pd.DataFrame(rows)

FEATURES = ["n", "density", "max_degree", "mean_degree", "dyn_range", "log_fidelity", "log_fidelity_anneal", "two_qubit_gates", "depth"]
TARGETS = ["score_gate", "score_anneal", "score_classical"]
router_model = RandomForestRegressor(n_estimators=300, random_state=0)
from sklearn.model_selection import cross_val_predict as _cvp
_p = _cvp(router_model, df[FEATURES], df[TARGETS], cv=5)
print(pd.DataFrame({"mean abs error": np.abs(_p - df[TARGETS].values).mean(0),
                    "spread of true score (std)": df[TARGETS].std().values}, index=TARGETS).round(3).to_string())
print("(The router is useful where its error is well below the spread. Classical barely varies, so R² is meaningless for it.)")
router_model.fit(df[FEATURES], df[TARGETS])
print(pd.Series(router_model.feature_importances_, FEATURES).sort_values(ascending=False).round(3))

# %% [markdown]
# ## 5b. Cross-field pattern analysis: problems from different industries share hidden shapes
# Three tests of the core claim, *"what works for one field transfers to another"*:
# 1. **Map:** project every problem's fingerprint to 2-D. If points cluster by *shape* rather than by *field*, a finance problem and a healthcare problem can be structural twins.
# 2. **Transfer test (leave-one-field-out):** train the router on three fields and predict the fourth, unseen field. Low error means the patterns it learned transfer across industries.
# 3. **Twin finder:** for a new healthcare problem, list its closest structural relatives from *other* fields.

# %%
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
Z = StandardScaler().fit_transform(df[FEATURES]); P2 = PCA(2, random_state=0).fit_transform(Z)
for f in df["field"].unique():
    m = df["field"] == f; plt.scatter(P2[m, 0], P2[m, 1], label=f, alpha=.75)
plt.xlabel("pattern axis 1"); plt.ylabel("pattern axis 2"); plt.title("Problem fingerprints across industries"); plt.legend(); plt.show()

def best_solver(scores):
    return ["gate-QAOA", "digital-anneal", "classical"][int(np.argmax(scores))]

transfer = []
for f in df["field"].unique():
    tr, te = df[df["field"] != f], df[df["field"] == f]
    if len(te) < 2: continue
    m = RandomForestRegressor(n_estimators=300, random_state=0).fit(tr[FEATURES], tr[TARGETS])
    pred = m.predict(te[FEATURES])
    agree = np.mean([np.argmax(p[:2]) == np.argmax(t[:2]) for p, t in zip(pred, te[TARGETS].values)])   # quantum modes only
    transfer.append(dict(held_out_field=f, trained_on=", ".join(sorted(set(tr["field"]))), n_test=len(te),
                         mean_abs_error=np.abs(pred - te[TARGETS].values).mean(), same_quantum_mode_choice=agree))
print(pd.DataFrame(transfer).round(3).to_string(index=False))

nn = NearestNeighbors(n_neighbors=4).fit(Z)
i = int(df.index[df["field"] == "healthcare"][0]) if (df["field"] == "healthcare").any() else 0
_, idx = nn.kneighbors(Z[[i]])
print(f"\nStructural twins of problem #{i} ({df.loc[i, 'field']}, n={df.loc[i, 'n']}):")
print(df.loc[idx[0][1:], ["field", "n", "density", "two_qubit_gates"] + TARGETS].round(3).to_string())

# %% [markdown]
# ## 5c. Does the router actually help? Router vs. "always use one solver" vs. the oracle
# This is the test of our central claim. Every problem is scored with an **out-of-sample** router: 5-fold cross-validation, so the router never saw the problem it's choosing for. We compare:
# - **Always gate-QAOA**, **always digital anneal**, **always classical**: fixed strategies
# - **Learned router**: picks a solver per problem from its fingerprint
# - **Oracle**: picks the best solver *after* seeing all the results. It's an upper bound, not something you can deploy
#
# Two settings: **quantum only** (does the router pick the better *quantum* mode?) and **all solvers** (classical included). If classical wins everything at this size, the table shows it. That's an honest result.

# %%
from sklearn.model_selection import cross_val_predict
pred_oos = cross_val_predict(RandomForestRegressor(n_estimators=300, random_state=0), df[FEATURES], df[TARGETS], cv=5)
actual = df[TARGETS].values
times = df[["time_gate", "time_anneal", "time_classical"]].values
SOLVER_NAMES = ["gate-QAOA", "digital anneal", "classical"]

def evaluate(allowed, mask=None):
    m = np.ones(len(df), bool) if mask is None else mask
    A, P, T = actual[m][:, allowed], pred_oos[m][:, allowed], times[m][:, allowed]
    pick, best = P.argmax(1), A.argmax(1); rows_ = np.arange(len(A))
    out = {f"always {SOLVER_NAMES[a]}": A[:, j].mean() for j, a in enumerate(allowed)}
    out.update({"learned router": A[rows_, pick].mean(), "oracle (upper bound)": A.max(1).mean(),
                "router picked best": (pick == best).mean(),
                "router regret vs oracle": (A.max(1) - A[rows_, pick]).mean(),
                "router avg sim time (s)": T[rows_, pick].mean()})
    return out

bench = pd.DataFrame({"quantum only": evaluate([0, 1]), "all solvers": evaluate([0, 1, 2])}).T
print(bench.round(3).to_string())
print("\nQuantum-only routing, broken down by field:")
print(pd.DataFrame({f: evaluate([0, 1], (df["field"] == f).values) for f in df["field"].unique()}).T.round(3).to_string())
bench.to_csv("router_benchmark.csv")

# %% [markdown]
# ## 5d. Zero-shot cross-field transfer: solve a healthcare problem with quantum settings learned on finance
# Tuning QAOA's angles normally takes ~60 circuit runs per problem, which is exactly what burns QPU time. But if problems from different fields share hidden shapes, the **tuned angles** should transfer too.
# For every problem we find its **closest structural twin from a different field** (by fingerprint), copy that twin's tuned angles, and run the circuit **once, with no optimizer at all**. Three scores:
# - **own optimized**: angles tuned on this very problem (~60 runs)
# - **transferred, zero-shot**: angles borrowed from a twin in another industry (1 run)
# - **random donor**: angles borrowed from a *random* problem in another field (1 run). This is the control: if the twin beats a random donor, the fingerprint match matters; if they tie, normalized QAOA angles are simply near-universal (known as *parameter concentration*, Brandão et al. 2018; Shaydulin et al. 2023)
# - **no learning**: a generic default schedule (1 run)
#
# If "transferred" lands close to "own optimized" and well above "no learning", **quantum know-how learned in one industry solves another industry's problems for free.** Transfer works because every Hamiltonian is normalized to the same scale (section 3).

# %%
SCALER = StandardScaler().fit(df[FEATURES]); ZF = SCALER.transform(df[FEATURES])
nn_all = NearestNeighbors(n_neighbors=len(df)).fit(ZF)

def run_fixed(Q, params, reps=1, shots=1024, backend=LOCAL):
    """One circuit run with given angles: no optimizer loop."""
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(build_qaoa(Q, reps))
    counts = Sampler(mode=backend).run([(isa, params)], shots=shots).result()[0].data.meas.get_counts()
    return mean_energy(counts, Q)

def cross_field_twin(z, own_field=None):
    """Index of the nearest problem in the dataset, from a different field when own_field is given."""
    _, order = nn_all.kneighbors(z.reshape(1, -1))
    return int(next(j for j in order[0] if own_field is None or df.loc[j, "field"] != own_field))

donor_rng = np.random.default_rng(0)
transfer_rows = []
for i in range(len(df)):
    j = cross_field_twin(ZF[i], df.loc[i, "field"])
    _, E_all = solve_exact(QS[i])
    transfer_rows.append(dict(field=df.loc[i, "field"], twin_field=df.loc[j, "field"],
                              own_optimized=df.loc[i, "r_gate_ideal"],
                              transferred_zero_shot=approx_ratio(run_fixed(QS[i], GATE_PARAMS[j]), E_all),
                              random_donor=np.mean([approx_ratio(run_fixed(QS[i], GATE_PARAMS[r]), E_all) for r in
                                  donor_rng.choice(np.flatnonzero((df["field"] != df.loc[i, "field"]).values), 3)]),
                              no_learning=approx_ratio(run_fixed(QS[i], annealing_init(1)), E_all)))
transfer_df = pd.DataFrame(transfer_rows)
cols = ["own_optimized", "transferred_zero_shot", "random_donor", "no_learning"]
print(transfer_df.groupby("field")[cols].mean().round(3).to_string())
print("\nAll problems:", transfer_df[cols].mean().round(3).to_dict())
kept = transfer_df["transferred_zero_shot"].mean() / transfer_df["own_optimized"].mean()
print(f"Zero-shot transfer keeps {kept:.0%} of the fully optimized quality with ~60× fewer circuit runs.")
gap = (transfer_df["transferred_zero_shot"] - transfer_df["random_donor"])
print(f"Twin vs. random donor: {gap.mean():+.3f} (twin better on {(gap > 0).mean():.0%} of problems). "
      + ("The fingerprint match adds value." if gap.mean() > 0.02 else "No clear gain: the angles transfer because they are near-universal, not because of the match."))
transfer_df.to_csv("zero_shot_transfer.csv", index=False)

transfer_df.groupby("field")[cols].mean().plot.bar(rot=0, figsize=(9, 4))
plt.ylabel("score (1 = best, 0 = random)"); plt.title("Quantum settings borrowed from another industry (no optimizer)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Router + decomposer
# The router predicts every solver's score from the fingerprint alone. It picks the best **quantum** mode, and falls back to classical when no quantum mode is predicted to be worthwhile.
# Big problems are split by **spectral clustering** of the variable-interaction graph into qubit-sized blocks. Each block is solved with the rest of the variables held fixed (block coordinate descent, like D-Wave's qbsolv), and **each block is routed separately.**

# %%
ROUTE_THRESHOLD = 0.5   # a quantum mode is "worthwhile" if predicted to capture ≥50% of the gap between random guessing and the optimum
def route(Q, threshold=ROUTE_THRESHOLD):
    feats = profile(Q)
    s_gate, s_ann, s_cls = router_model.predict(pd.DataFrame([feats])[FEATURES])[0]
    mode, best = ("gate-QAOA", s_gate) if s_gate >= s_ann else ("digital-anneal", s_ann)
    if best < threshold: mode = "classical"
    return mode, dict(gate=round(s_gate, 3), anneal=round(s_ann, 3), classical=round(s_cls, 3)), feats

def solve_routed(Q, mode):
    if mode == "gate-QAOA": return solve_qaoa(Q, reps=1, maxiter=80)["x"]
    if mode == "digital-anneal": return solve_digital_anneal(Q, ANNEAL_P, ANNEAL_DELTA)["x"]
    return solve_annealer(Q)[0]


def sub_qubo(Q, B, x):
    """QUBO over the block B with all other variables fixed at x."""
    S = Q + Q.T - np.diag(np.diag(Q)); rest = np.setdiff1d(np.arange(len(Q)), B)
    Qb = np.triu(Q[np.ix_(B, B)]).copy()
    Qb[np.diag_indices(len(B))] += S[np.ix_(B, rest)] @ x[rest]
    return Qb

def decompose_and_solve(Q, block=8, sweeps=2, threshold=ROUTE_THRESHOLD, log=True):
    n = len(Q); A = np.abs(Q + Q.T); np.fill_diagonal(A, 0)
    k = max(1, int(np.ceil(n / block)))
    labels = SpectralClustering(k, affinity="precomputed", random_state=0).fit_predict(A + 1e-9) if k > 1 else np.zeros(n, int)
    blocks = [c[i:i + block] for c in (np.flatnonzero(labels == l) for l in range(k)) for i in range(0, len(c), block)]
    x = solve_annealer(Q, num_reads=20)[0]          # warm start
    decisions = []
    for s in range(sweeps):
        for B in blocks:
            Qb = sub_qubo(Q, B, x); target, preds, _ = route(Qb, threshold)
            xb = solve_routed(Qb, target)
            if energy(Qb, xb) <= energy(Qb, x[B]): x[B] = xb
            decisions.append(dict(sweep=s, block=list(map(int, B)), size=len(B), route=target, **preds))
    if log: print(pd.DataFrame(decisions).to_string(index=False))
    return x

Q16, names16 = pharma_problem(n=16, k=5, seed=3)
x_dec = decompose_and_solve(Q16, block=8)
x_opt16, E16 = solve_exact(Q16)
print("Decomposed solution ratio: %.4f   (feasible: %s)" % (approx_ratio(energy(Q16, x_dec), E16), x_dec.sum() == 5))

# %% [markdown]
# ## 6b. Break it down: which quantum type wins which piece?
# One realistic problem with two very different parts:
# - **Which drugs to fund** (8 candidates, fund 3): every decision affects every other one (shared budget, correlated risk), so it is **dense**.
# - **How to split clinical-trial sites between two research partners** (10 sites): each site only interacts with a few neighbors, so it is **sparse**.
#
# Small links tie them together (a funded drug's trials run at particular sites). We **break the problem into pieces** with spectral clustering, then run **both** quantum types on **every** piece: gate-based QAOA (short, tuned) and the digital anneal (a slow sweep, more layers). Each is scored three ways: perfect simulator, predicted from the chip's calibration (F × ideal), and a full noisy simulation of IBM Torino. Section 9b runs the same circuits on the real chip in one job.
#
# First, the pattern across all 60 problems from section 5: **where does each type win?**

# %%
when = df.groupby("field").agg(density=("density", "mean"), gate=("score_gate", "mean"), anneal=("score_anneal", "mean"),
                               anneal_wins=("score_anneal", lambda s: (s > df.loc[s.index, "score_gate"]).mean()))
print(when.sort_values("density", ascending=False).round(3).to_string())
plt.figure(figsize=(8, 4.5))
for f in df["field"].unique():
    m = df["field"] == f; plt.scatter(df.loc[m, "density"], df.loc[m, "score_anneal"] - df.loc[m, "score_gate"], label=f, alpha=.75)
plt.axhline(0, c="gray", lw=.8); plt.xlabel("problem density (how connected the decisions are)")
plt.ylabel("anneal score − gate score\n(above 0 = anneal wins)"); plt.title("Dense problems favor gate-based QAOA;\nsparse ones let the anneal catch up", fontsize=11)
plt.legend(loc="lower left", fontsize=9); plt.tight_layout(); plt.show()

# %%
def pharma_trials_problem(n_drugs=8, k=3, n_sites=10, link=0.1, seed=4):
    """Fund k of n_drugs (dense) + split n_sites trial sites between two partners (sparse), weakly linked."""
    Qd, dnames = pharma_problem(n_drugs, k, seed=seed); Qs, _ = maxcut_problem(n_sites, degree=2, seed=seed)
    Qd, Qs = Qd / np.abs(Qd).max(), Qs / np.abs(Qs).max()        # put both parts on the same scale
    N = n_drugs + n_sites; Q = np.zeros((N, N)); Q[:n_drugs, :n_drugs] = Qd; Q[n_drugs:, n_drugs:] = Qs
    g = np.random.default_rng(seed)
    for i in range(n_drugs):                                      # drug i's trials run at one site: small reward if both chosen
        Q[i, n_drugs + g.integers(n_sites)] -= link
    return Q, dnames + [f"site{j}" for j in range(n_sites)], n_drugs

def spectral_blocks(Q, block=8):
    A = np.abs(Q + Q.T); np.fill_diagonal(A, 0); k = max(1, int(np.ceil(len(Q) / block)))
    labels = SpectralClustering(k, affinity="precomputed", random_state=0).fit_predict(A + 1e-9) if k > 1 else np.zeros(len(Q), int)
    return [c[i:i + block] for c in (np.flatnonzero(labels == l) for l in range(k)) for i in range(0, len(c), block)]

NOISY = AerSimulator.from_backend(DEVICE)
Qbig, names_big, n_drugs = pharma_trials_problem()
x_best_big, E_big = solve_exact(Qbig)
blocks = spectral_blocks(Qbig)
x0 = np.zeros(len(Qbig))                                          # no classical warm start: pieces are judged on their own

BREAKDOWN, rows_b = [], []
for b, B in enumerate(blocks):
    Qb = sub_qubo(Qbig, B, x0); _, Eb = solve_exact(Qb); feats = profile(Qb)
    part = "drug funding" if np.mean(B < n_drugs) > .5 else "trial-site network"
    qg = solve_qaoa(Qb, reps=1, maxiter=80)                       # gate: tuned on the simulator
    p_a, d_a = tune_anneal(Qb); qa = solve_digital_anneal(Qb, p_a, d_a)   # anneal: schedule tuned on the simulator
    Fg, twoq_g, _ = expected_fidelity(Qb, 1); Fa, twoq_a, _ = expected_fidelity(Qb, p_a)
    ideal_g, ideal_a = approx_ratio(qg["mean_energy"], Eb), approx_ratio(qa["mean_energy"], Eb)
    noisy_g = approx_ratio(run_fixed(Qb, qg["params"], reps=1, shots=4096, backend=NOISY), Eb)
    noisy_a = approx_ratio(run_fixed(Qb, qa["params"], reps=p_a, shots=4096, backend=NOISY), Eb)
    pg, pa, _ = router_model.predict(pd.DataFrame([feats])[FEATURES])[0]
    pick = "gate" if pg >= pa else "anneal"; better = "gate" if noisy_g >= noisy_a else "anneal"
    BREAKDOWN.append(dict(piece=b, part=part, Q=Qb, E=Eb, gate=(1, qg["params"]), anneal=(p_a, qa["params"]),
                          noisy_gate=noisy_g, noisy_anneal=noisy_a))
    rows_b.append({"piece": b, "part": part, "qubits": len(B), "density": round(feats["density"], 2),
                   "2q gates gate / anneal": f"{twoq_g} / {twoq_a}",
                   "ideal gate": ideal_g, "ideal anneal": ideal_a,
                   "predicted gate": Fg * ideal_g, "predicted anneal": Fa * ideal_a,
                   "noisy-sim gate": noisy_g, "noisy-sim anneal": noisy_a,
                   "router picked": pick, "better on noisy chip": better, "right call": pick == better})
breakdown_df = pd.DataFrame(rows_b)
print(breakdown_df.round(3).to_string(index=False))
breakdown_df.to_csv("breakdown_comparison.csv", index=False)

# Strategy comparison on the noisy chip: average piece score
G, A = breakdown_df["noisy-sim gate"].values, breakdown_df["noisy-sim anneal"].values
pick_g = (breakdown_df["router picked"] == "gate").values
strategies = {"all gate-QAOA": G.mean(), "all digital anneal": A.mean(),
              "router's mix": np.where(pick_g, G, A).mean(), "best possible mix": np.maximum(G, A).mean()}
print("\nAverage piece score on the noisy IBM Torino model (1 = optimal, 0 = random):")
print(pd.Series(strategies).round(3).to_string())

# Reassemble the full answer from the pieces (block coordinate descent, 2 sweeps), per strategy
def assemble(choose):
    x = x0.copy()
    for _ in range(2):
        for b, B in enumerate(blocks):
            Qb = sub_qubo(Qbig, B, x); mode = choose(b)
            xb = (solve_qaoa(Qb, reps=1, maxiter=80)["x"] if mode == "gate" else
                  solve_digital_anneal(Qb, *tune_anneal(Qb))["x"] if mode == "anneal" else solve_annealer(Qb)[0])
            if energy(Qb, xb) <= energy(Qb, x[B]): x[B] = xb
    return x
full = {}
for name, choose in {"all gate-QAOA": lambda b: "gate", "all digital anneal": lambda b: "anneal",
                     "router's mix": lambda b: rows_b[b]["router picked"], "classical": lambda b: "classical"}.items():
    x = assemble(choose)
    full[name] = dict(ratio=approx_ratio(energy(Qbig, x), E_big), found_optimum=bool(energy(Qbig, x) <= E_big.min() + 1e-9),
                      drugs_funded_ok=bool(x[:n_drugs].sum() == 3))
print("\nFull 18-variable answer rebuilt from the pieces (best shot per piece, ideal simulator):")
print(pd.DataFrame(full).T.to_string())

fig, ax = plt.subplots(figsize=(9, 4)); w = .38; idx = np.arange(len(breakdown_df))
for k_, (mode, lab) in enumerate([("gate", "gate-QAOA"), ("anneal", "digital anneal")]):
    rects = ax.bar(idx + (k_ - .5) * w, breakdown_df[f"noisy-sim {mode}"], w, label=lab)
    for r, pk in zip(rects, breakdown_df["router picked"]):
        if pk == mode: r.set_edgecolor("black"); r.set_linewidth(2.5)
ax.set_xticks(idx); ax.set_xticklabels([f"piece {r.piece}\n{r.part}\n({r.qubits} qubits, density {r.density})" for r in breakdown_df.itertuples()])
ax.set_ylabel("score on noisy IBM Torino model\n(1 = optimal, 0 = random)"); ax.axhline(0, c="gray", lw=.8)
ax.set_title("Each piece solved both ways (black outline = router's choice)"); ax.legend(); fig.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Error mitigation demo: zero-noise extrapolation on a noisy simulator of a real IBM device
# We amplify the noise on purpose by **gate folding** (U → U U† U triples the noise but leaves the ideal result unchanged), measure at noise ×1, ×3, ×5, and extrapolate back to ×0. We also compare the depolarizing-model prediction from Section 5 against this full noisy simulation.

# %%
Qm, _ = maxcut_problem(6, degree=2, seed=5)
base_q = solve_qaoa(Qm, reps=1)
ans = build_qaoa(Qm, 1); ans.remove_final_measurements()
bound = ans.assign_parameters(base_q["params"])
ideal = sum(p * energy(Qm, bits_to_x(b)) for b, p in Statevector(bound.decompose(reps=3)).probabilities_dict().items())

noisy = AerSimulator.from_backend(DEVICE); sampler_noisy = Sampler(mode=noisy)
def noisy_energy(scale):
    c = bound.copy()
    for _ in range((scale - 1) // 2): c = c.compose(bound.inverse()).compose(bound)
    c.measure_all()
    isa = generate_preset_pass_manager(optimization_level=0, backend=noisy, seed_transpiler=1).run(c)
    return mean_energy(sampler_noisy.run([isa], shots=8192).result()[0].data.meas.get_counts(), Qm)

scales = np.array([1, 3, 5]); vals = np.array([noisy_energy(s) for s in scales])
zne = np.polyval(np.polyfit(scales, vals, 1), 0)
_, Em = solve_exact(Qm); F, _, _ = expected_fidelity(Qm, 1)
print(f"ideal ⟨E⟩={ideal:.3f}  raw noisy={vals[0]:.3f}  ZNE={zne:.3f}  depolarizing-model prediction={F*ideal+(1-F)*Em.mean():.3f}")
plt.plot(scales, vals, "o-", label="noisy (FakeDevice)"); plt.plot([0], [zne], "r*", ms=14, label="ZNE extrapolation")
plt.axhline(ideal, ls="--", c="g", label="ideal"); plt.xlabel("noise scale factor"); plt.ylabel("mean QUBO energy"); plt.legend(); plt.title("Zero-noise extrapolation"); plt.show()

# %% [markdown]
# ## 7b. Is entanglement doing any work?
# The two-qubit (ZZ) gates in QAOA are where qubits get **entangled**: one for every pair of decisions that affect each other (two drugs in the same disease area, two redundant tests). Two checks on the flagship 10-drug problem, on the perfect simulator:
# 1. **How entangled is the solver's quantum state?** The entanglement entropy between the first 5 and the last 5 drug qubits (0 = not entangled, 5 bits = the maximum possible).
# 2. **Switch entanglement off.** Remove every ZZ gate, keep everything else, re-tune, and compare the score. If it drops, entanglement is doing real work.
#
# This is the same trade-off the router manages: entanglement is the useful ingredient, but entangling gates are also the noisiest operations on the chip.

# %%
from qiskit.quantum_info import partial_trace, entropy

def tune_ansatz(ansatz, Q, reps, shots=2048, maxiter=120, backend=LOCAL):
    meas = ansatz.copy(); meas.measure_all()
    isa = generate_preset_pass_manager(optimization_level=1, backend=backend).run(meas); s = Sampler(mode=backend)
    run = lambda th: s.run([(isa, th)], shots=shots).result()[0].data.meas.get_counts()
    res = minimize(lambda th: mean_energy(run(th), Q), annealing_init(reps), method="COBYLA", options={"maxiter": maxiter})
    return res.x, mean_energy(run(res.x), Q)

Qf, _ = pharma_problem(n=10, k=4, seed=1); _, Ef = solve_exact(Qf); REPS = 2
H, _ = qubo_to_ising(Qf)
H_off = SparsePauliOp.from_list([(p.to_label(), c) for p, c in zip(H.paulis, H.coeffs) if p.to_label().count("Z") == 1])  # drop ZZ terms
results_ent = {}
for label, Hx in [("with entanglement (normal QAOA)", H), ("entanglement switched off", H_off)]:
    ans = QAOAAnsatz(Hx, reps=REPS); params, e = tune_ansatz(ans, Qf, REPS)
    bound = ans.assign_parameters(params)
    S_half = entropy(partial_trace(Statevector(bound.decompose(reps=3)), list(range(5))), base=2)
    ops = generate_preset_pass_manager(optimization_level=1, backend=DEVICE, seed_transpiler=1).run(bound).count_ops()
    results_ent[label] = {"score (1 = best, 0 = random)": approx_ratio(e, Ef), "entanglement entropy (bits, max 5)": S_half,
                          "two-qubit gates on IBM Torino": ops.get("cz", 0) + ops.get("ecr", 0)}
ent_df = pd.DataFrame(results_ent).T; print(ent_df.round(3).to_string()); ent_df.to_csv("entanglement_test.csv")

# %% [markdown]
# ## 8. The roadmap chart: as hardware improves, more problems from every field go to quantum hardware

# %%
scales = [1, 1/3, 1/10, 1/30, 1/100]; out = []
for s in scales:
    for _, row in df.iterrows():
        g_ = np.exp(row["log_fidelity"] * s) * row["r_gate_ideal"]        # log F ∝ total error
        a_ = np.exp(row["log_fidelity_anneal"] * s) * row["r_anneal_ideal"]
        mode = "classical" if max(g_, a_) < ROUTE_THRESHOLD else ("gate-QAOA" if g_ >= a_ else "digital-anneal")
        out.append(dict(improvement=1 / s, field=row["field"], mode=mode))
out = pd.DataFrame(out)
for f in out["field"].unique():
    d = out[out["field"] == f].groupby("improvement")["mode"].apply(lambda m: (m != "classical").mean())
    plt.plot(d.index, d.values, "o-", label=f)
plt.xscale("log"); plt.xlabel("hardware error reduction vs. today (×)"); plt.ylabel("fraction routed to a quantum mode")
plt.title("Which industries cross over to quantum first"); plt.legend(); plt.grid(alpha=.3); plt.show()

# %% [markdown]
# ## 9. Real IBM hardware: both modes, with and without error mitigation (+5 hardware bonus tier)
# This uses **4 jobs total** (gate-logic and anneal, each raw and mitigated), a few seconds of QPU time each. Every parameter was tuned on the simulator first, so there are no optimizer loops on hardware.
# Following the event handbook, **don't save your API key in Colab**: paste it at the prompt each session. Results go to `hardware_results.json`, so the demo never waits on the queue.

# %%
RUN_ON_HARDWARE = False   # flip to True when the simulator results look right
if RUN_ON_HARDWARE:
    from getpass import getpass
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=getpass("IBM API key: "),
                                   instance=getpass("Instance CRN: "))
    Qh, names_h = pharma_problem(n=8, k=3, seed=1)
    _, Eh = solve_exact(Qh)
    p_var = 1
    var_params = solve_qaoa(Qh, reps=p_var)["params"]          # gate-based logic, tuned on simulator
    p_ann, d_ann = tune_anneal(Qh)                              # annealing schedule, tuned on simulator
    modes = {"gate-logic QAOA": (p_var, var_params), "digital anneal": (p_ann, lr_params(p_ann, d_ann))}

    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=len(Qh))
    print("Using", backend.name)
    noisy_sim = AerSimulator.from_backend(backend)
    results, table = {"backend": backend.name}, []
    for mode, (p, params) in modes.items():
        isa = generate_preset_pass_manager(optimization_level=3, backend=backend).run(build_qaoa(Qh, p))
        ideal = Sampler(mode=LOCAL).run([(isa, params)], shots=4096).result()[0].data.meas.get_counts()
        sim = Sampler(mode=noisy_sim).run([(isa, params)], shots=4096).result()[0].data.meas.get_counts()
        row = {"mode": mode, "2q gates": isa.count_ops().get("cz", 0) + isa.count_ops().get("ecr", 0),
               "ideal sim": approx_ratio(mean_energy(ideal, Qh), Eh), "noisy sim": approx_ratio(mean_energy(sim, Qh), Eh)}
        for mitigated in (False, True):
            s = Sampler(mode=backend); s.options.default_shots = 4096
            if mitigated:
                s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
                s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True
            job = s.run([(isa, params)]); counts = job.result()[0].data.meas.get_counts()
            key = "QPU mitigated" if mitigated else "QPU raw"
            row[key] = approx_ratio(mean_energy(counts, Qh), Eh)
            results[f"{mode} | {key}"] = {"job_id": job.job_id(), "counts": counts}
        table.append(row)
    _, e_sa = solve_annealer(Qh)
    table.append({"mode": "classical simulated annealing", "ideal sim": approx_ratio(e_sa.mean(), Eh)})
    print(pd.DataFrame(table).round(3).to_string(index=False))
    json.dump(results, open("hardware_results.json", "w"))

# %% [markdown]
# ## 9b. The break-down on real hardware: gate vs. anneal, piece by piece (1 job)
# The section 6b pieces, each solved **both** ways on the real chip with settings tuned on the simulator. Every circuit goes into **one** batched job with error mitigation (dynamical decoupling + twirling). Run section 9 first (it creates `backend`).

# %%
RUN_BREAKDOWN_ON_HARDWARE = False   # one job, roughly 30-60 s of QPU time
if RUN_BREAKDOWN_ON_HARDWARE:
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    pubs, meta = [], []
    for pc in BREAKDOWN:
        for mode in ("gate", "anneal"):
            p, params = pc[mode]
            pubs.append((pm.run(build_qaoa(pc["Q"], p)), params)); meta.append((pc, mode))
    s = mitigated_sampler(backend, shots=4096)
    res = run_pubs(s, pubs, backend); print("Hardware job done on", backend.name)
    hw = {(pc["piece"], mode): approx_ratio(mean_energy(r.data.meas.get_counts(), pc["Q"]), pc["E"]) for (pc, mode), r in zip(meta, res)}
    breakdown_df["QPU gate"] = [hw[(b, "gate")] for b in breakdown_df["piece"]]
    breakdown_df["QPU anneal"] = [hw[(b, "anneal")] for b in breakdown_df["piece"]]
    breakdown_df["better on QPU"] = np.where(breakdown_df["QPU gate"] >= breakdown_df["QPU anneal"], "gate", "anneal")
    breakdown_df["router right on QPU"] = breakdown_df["router picked"] == breakdown_df["better on QPU"]
    print(breakdown_df[["piece", "part", "qubits", "density", "noisy-sim gate", "noisy-sim anneal", "QPU gate", "QPU anneal",
                        "router picked", "better on QPU", "router right on QPU"]].round(3).to_string(index=False))
    breakdown_df.to_csv("breakdown_comparison.csv", index=False)

# %% [markdown]
# ## 10. The noise cliff: where does each quantum mode stop helping on real hardware? (2 jobs total)
# Most teams show a single hardware run. This experiment grows the problem from 6 to 16 drug candidates and measures **where each mode falls back to random guessing** on the real chip. It also checks whether the router's noise model (predicted = F × ideal) **predicted the hardware result before we spent QPU time**.
# All 12 circuits go into **one** Sampler job (batched), run once raw and once mitigated. Run section 9 first (it creates `backend`).

# %%
RUN_SWEEP = False
if RUN_SWEEP:
    sizes = [6, 8, 10, 12, 14, 16]
    p_ann, d_ann = tune_anneal(pharma_problem(8, 3, seed=1)[0])        # schedule tuned once, transferred to every size
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    pubs, meta, Qs, Es = [], [], [], []
    for n in sizes:
        Qn, _ = pharma_problem(n, max(2, n // 3), seed=n)
        _, En = solve_exact(Qn); _, e_sa = solve_annealer(Qn)
        runs = {"gate-logic": (1, solve_qaoa(Qn, reps=1, maxiter=80)),
                "anneal": (p_ann, solve_digital_anneal(Qn, p_ann, d_ann))}
        for mode, (p, sim) in runs.items():
            r_ideal = approx_ratio(sim["mean_energy"], En)
            F, twoq, _ = expected_fidelity(Qn, p, backend=backend)
            pubs.append((pm.run(build_qaoa(Qn, p)), sim["params"])); Qs.append(Qn); Es.append(En)
            meta.append(dict(n=n, mode=mode, two_qubit_gates=twoq, ideal=r_ideal, predicted=F * r_ideal,
                             classical_SA=approx_ratio(e_sa.mean(), En)))
        print(f"prepared n={n}")
    for mitigated in (False, True):
        s = Sampler(mode=backend); s.options.default_shots = 2048
        if mitigated:
            s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
            s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True
        job = s.run(pubs); print("job", job.job_id(), "submitted"); res = job.result()
        key = "QPU mitigated" if mitigated else "QPU raw"
        for m, r, Qn, En in zip(meta, res, Qs, Es):
            m[key] = approx_ratio(mean_energy(r.data.meas.get_counts(), Qn), En)
    sweep = pd.DataFrame(meta); print(sweep.round(3).to_string(index=False))
    sweep.to_csv("noise_cliff.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ["gate-logic", "anneal"]):
        d = sweep[sweep["mode"] == mode]
        for col, style in [("ideal", "g--"), ("predicted", "k:"), ("QPU raw", "r-o"), ("QPU mitigated", "b-o"), ("classical_SA", "m-.")]:
            ax.plot(d["n"], d[col], style, label=col)
        ax.axhline(0, c="gray", lw=.8); ax.set_title(f"{mode} on {backend.name}"); ax.set_xlabel("drug candidates (= qubits)")
    axes[0].set_ylabel("approximation ratio (1 = optimal, 0 = random)"); axes[1].legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11. Solve problems from different fields **simultaneously** on one quantum chip
# A 156-qubit IBM chip is far bigger than any one of our problems. So Q-Router packs a **finance**, a **healthcare** and a **finance×healthcare** problem side by side on separate groups of qubits, in **one circuit and one job**. Each problem gets the quantum mode the router chose for it. Problems routed to classical run on the CPU at the same time.
#
# Why it matters: one job instead of three saves QPU time and queue time (this year's theme is *"a decade of quantum on the cloud"*, which is exactly cloud multi-tenancy). Running each problem **alone** as well measures **crosstalk**, how much the neighbors' noise hurts each problem. That's a real hardware noise analysis.

# %%
from qiskit import QuantumCircuit

def bound_body(Q, mode):
    """The routed circuit for one problem, parameters already bound, no measurements."""
    if mode == "gate-QAOA": p, params = 1, solve_qaoa(Q, reps=1, maxiter=80)["params"]
    else: p, params = ANNEAL_P, lr_params(ANNEAL_P, ANNEAL_DELTA)
    H, _ = qubo_to_ising(Q)
    return QAOAAnsatz(H, reps=p).assign_parameters(params)

def marginal_ratio(counts, Q, offset, E_all):
    n = len(Q); tot = sum(counts.values()); e = 0.0
    for b, c in counts.items():
        e += energy(Q, bits_to_x(b)[offset:offset + n]) * c
    return approx_ratio(e / tot, E_all)

def solve_simultaneous(problems, backend, shots=4096, force_quantum=True, mitigate=True):
    """problems: {name: Q}. Routes each problem, packs every quantum one into ONE circuit, runs ONE job."""
    plan, bodies, offset = [], [], 0
    for name, Q in problems.items():
        mode, preds, _ = route(Q)
        if force_quantum and mode == "classical":          # demo option: run it on the chip anyway
            mode = "gate-QAOA" if preds["gate"] >= preds["anneal"] else "digital-anneal"
        _, E_all = solve_exact(Q)
        item = dict(problem=name, n=len(Q), route=mode, **{f"pred_{k}": v for k, v in preds.items()}, E=E_all, Q=Q)
        if mode != "classical":
            item["offset"] = offset; bodies.append((offset, bound_body(Q, mode))); offset += len(Q)
        else:
            item["together"] = approx_ratio(solve_annealer(Q)[1].mean(), E_all)   # mean over reads, same metric as quantum
        plan.append(item)

    combined = QuantumCircuit(offset)
    for off, body in bodies: combined.compose(body, qubits=range(off, off + body.num_qubits), inplace=True)
    combined.measure_all()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    s = Sampler(mode=backend); s.options.default_shots = shots
    if mitigate and not is_local(backend):
        s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True

    # one job: the packed circuit + each quantum problem alone (for the crosstalk comparison)
    pubs = [pm.run(combined)]
    for item in plan:
        if "offset" in item:
            solo = QuantumCircuit(item["n"]); solo.compose(dict(bodies)[item["offset"]], inplace=True); solo.measure_all()
            pubs.append(pm.run(solo))
    res = run_pubs(s, pubs, backend)   # one hardware job; one job per circuit on local simulators
    together = res[0].data.meas.get_counts(); k = 1
    for item in plan:
        if "offset" in item:
            item["together"] = marginal_ratio(together, item["Q"], item["offset"], item["E"])
            item["alone"] = marginal_ratio(res[k].data.meas.get_counts(), item["Q"], 0, item["E"]); k += 1
            item["crosstalk_loss"] = item["alone"] - item["together"]
    table = pd.DataFrame([{k_: v for k_, v in it.items() if k_ not in ("E", "Q", "offset")} for it in plan])
    print(f"Packed {offset} qubits from {len(bodies)} problems into one circuit on {getattr(backend, 'name', 'simulator')}")
    print(table.round(3).to_string(index=False))
    return table

MULTI = {"finance: stock portfolio (6 stocks, hold 2)": portfolio_problem(6, 2)[0],
         "healthcare: biomarker selection (6 tests)": feature_selection_problem(6, sparsify=0.5)[0],
         "finance×healthcare: drug pipeline (6 drugs, fund 2)": pharma_problem(6, 2, seed=1)[0]}

RUN_SIMULTANEOUS_ON_HARDWARE = False   # needs section 9's `backend`; one job, ~seconds of QPU time
target_backend = backend if RUN_SIMULTANEOUS_ON_HARDWARE else AerSimulator.from_backend(DEVICE)
multi_table = solve_simultaneous(MULTI, target_backend)
multi_table.to_csv("simultaneous_results_hardware.csv" if RUN_SIMULTANEOUS_ON_HARDWARE else "simultaneous_results.csv", index=False)

# %% [markdown]
# ## 11b. One tune, one job: pattern recognition as a compute saver
# The normal way to run four problems from four industries: tune each one separately (~80 circuit runs each), then run each in its own chip job. **Q-Router's way:**
# 1. **Group** the problems into families with the same *shape* (their fingerprints), whatever industry they come from.
# 2. **Tune once per family** on the simulator, and share those settings with every family member.
# 3. **Pack every problem onto the chip in one circuit, one job.**
#
# Three versions of each problem separate the two savings: **own settings, alone** (the normal way), **shared settings, alone** (does sharing settings cost quality?), and **shared settings, packed together** (does packing cost quality?). The **compute ledger** then counts what each approach used.

# %%
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit
ONE_TUNE = {"finance: stock portfolio": portfolio_problem(4, 2)[0],
            "healthcare: diagnostic tests": feature_selection_problem(4, sparsify=0.5)[0],
            "finance×healthcare: drug pipeline": pharma_problem(4, 2, seed=11)[0],
            "networks: referral network split": maxcut_problem(4, degree=2, seed=11)[0]}

def prepare_one_tune(problems, n_families=2):
    names_ = list(problems); Qs_ = [problems[k] for k in names_]
    Z_ = SCALER.transform(pd.DataFrame([profile(Q) for Q in Qs_])[FEATURES])
    fam = KMeans(n_families, n_init=10, random_state=0).fit_predict(Z_)
    own = [solve_qaoa(Q, reps=1, maxiter=80) for Q in Qs_]                       # normal way: tune every problem
    shared = {}
    for f in sorted(set(fam)):                                                   # Q-Router: tune once per family
        members = np.flatnonzero(fam == f); centre = Z_[members].mean(0)
        rep = members[np.argmin(((Z_[members] - centre) ** 2).sum(1))]          # the most typical member
        shared[f] = own[rep]                                                     # its tuning run is the family's only one
    return dict(names=names_, Qs=Qs_, E=[solve_exact(Q)[1] for Q in Qs_], family=fam, own=own, shared=shared)

def run_one_tune(prep, backend, shots=4096):
    names_, Qs_, fam = prep["names"], prep["Qs"], prep["family"]
    circuits, bodies, offsets, offset = [], [], [], 0
    for i, Q in enumerate(Qs_):
        H, _ = qubo_to_ising(Q)
        for params in (prep["own"][i]["params"], prep["shared"][fam[i]]["params"]):  # alone: own, then shared settings
            c = QAOAAnsatz(H, reps=1).assign_parameters(params); c.measure_all(); circuits.append(c)
        bodies.append((offset, QAOAAnsatz(H, reps=1).assign_parameters(prep["shared"][fam[i]]["params"])))
        offsets.append(offset); offset += len(Q)
    packed = QuantumCircuit(offset)
    for off, body in bodies: packed.compose(body, qubits=range(off, off + body.num_qubits), inplace=True)
    packed.measure_all(); circuits.append(packed)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    res = run_pubs(mitigated_sampler(backend, shots), [pm.run(c) for c in circuits], backend)
    together = res[-1].data.meas.get_counts(); rows = []
    for i, (nm, Q) in enumerate(zip(names_, Qs_)):
        rows.append({"problem": nm, "shape family": int(fam[i]),
                     "own settings, alone": marginal_ratio(res[2 * i].data.meas.get_counts(), Q, 0, prep["E"][i]),
                     "shared settings, alone": marginal_ratio(res[2 * i + 1].data.meas.get_counts(), Q, 0, prep["E"][i]),
                     "shared settings, packed": marginal_ratio(together, Q, offsets[i], prep["E"][i])})
    return pd.DataFrame(rows)

def compute_ledger(prep, shots=4096):
    n = len(prep["Qs"]); fams = len(prep["shared"])
    runs_all = sum(o["evals"] for o in prep["own"]); runs_q = sum(s["evals"] for s in prep["shared"].values())
    return pd.DataFrame({"normal way": {"problems tuned": n, "tuning circuit runs (simulator)": runs_all,
                                         "circuits sent to the chip": n, "chip jobs": n, "QPU shots": n * shots},
                         "Q-Router (one tune, one job)": {"problems tuned": fams, "tuning circuit runs (simulator)": runs_q,
                                         "circuits sent to the chip": 1, "chip jobs": 1, "QPU shots": shots}})

ONE_TUNE_PREP = prepare_one_tune(ONE_TUNE)
one_tune_df = run_one_tune(ONE_TUNE_PREP, NOISY)
print("Noisy IBM Torino model (1 = optimal, 0 = random):"); print(one_tune_df.round(3).to_string(index=False))
print("\nAverage:", one_tune_df.iloc[:, 2:].mean().round(3).to_dict())
ledger = compute_ledger(ONE_TUNE_PREP); print("\nCompute ledger:"); print(ledger.to_string())
one_tune_df.to_csv("one_tune_one_job.csv", index=False); ledger.to_csv("compute_ledger.csv")

RUN_ONE_TUNE_ON_HARDWARE = False   # one job (9 circuits, 16 qubits packed), roughly 20-40 s of QPU time; run section 9 first
if RUN_ONE_TUNE_ON_HARDWARE:
    one_tune_hw = run_one_tune(ONE_TUNE_PREP, backend); print(f"\nReal hardware ({backend.name}):")
    print(one_tune_hw.round(3).to_string(index=False)); print("Average:", one_tune_hw.iloc[:, 2:].mean().round(3).to_dict())
    one_tune_hw.to_csv("one_tune_one_job_hardware.csv", index=False)

# %% [markdown]
# ## 12. The Q-Router app: a live demo for the judges
# Everything above is the **engine** (backend). This cell wraps it in a small **web app** (frontend) built with Gradio, and prints a public link you can open on any laptop or projector.
# In the app you pick problems from different industries and a size, then press one button. It **fingerprints** each problem, finds its **structural twin in another industry**, shows the **router's predictions and choice**, solves everything with **zero-shot borrowed quantum settings**, and charts each method against the verified best answer. Real IBM hardware results saved by sections 9–11 appear at the bottom.
# Run all sections above first. The link stays live while this Colab tab is open.

# %%
# (Colab only) !pip -q install gradio
import gradio as gr

BUILDER_FIELD = {"Finance: healthcare-stock portfolio": "finance",
                 "Healthcare: diagnostic test selection": "healthcare",
                 "Finance × Healthcare: drug pipeline": "finance×healthcare"}
BUILDERS = {
    "Finance: healthcare-stock portfolio": lambda n: portfolio_problem(n, max(2, n // 3)),
    "Healthcare: diagnostic test selection": lambda n: feature_selection_problem(n, sparsify=0.5)[:2],
    "Finance × Healthcare: drug pipeline": lambda n: pharma_problem(n, max(2, n // 3), seed=7),
}

def saved_hardware():
    """Real-hardware results only (simulator results live in other files)."""
    try:
        b = pd.read_csv("breakdown_comparison.csv")
        if "QPU gate" in b: return b[["piece", "part", "qubits", "QPU gate", "QPU anneal", "router picked", "better on QPU"]].round(3)
    except Exception: pass
    for f in ["one_tune_one_job_hardware.csv", "simultaneous_results_hardware.csv", "noise_cliff.csv"]:
        try: return pd.read_csv(f).round(3)
        except Exception: continue
    return pd.DataFrame([{"note": "No hardware results saved yet. Run sections 9-11 with the hardware flags on."}])

def run_demo(selected, n):
    n = int(n); table, bars = [], []
    for name in selected:
        Q, labels = BUILDERS[name](n)
        mode, preds, feats = route(Q)
        x_best, E_all = solve_exact(Q)
        j = cross_field_twin(SCALER.transform(pd.DataFrame([feats])[FEATURES])[0], BUILDER_FIELD[name])
        scores = {"classical": approx_ratio(solve_annealer(Q, num_reads=100)[1].mean(), E_all),
                  "gate-QAOA": approx_ratio(run_fixed(Q, GATE_PARAMS[j]), E_all),               # zero-shot borrowed angles
                  "digital-anneal": approx_ratio(solve_digital_anneal(Q, ANNEAL_P, ANNEAL_DELTA, shots=1024)["mean_energy"], E_all)}
        table.append({"Problem": name, "Qubits": n,
                      "Structural twin from": f"{df.loc[j, 'field']} (n={df.loc[j, 'n']})",
                      "Predicted gate / anneal / classical": f"{preds['gate']:.2f} / {preds['anneal']:.2f} / {preds['classical']:.2f}",
                      "Router's choice": mode, "Score of choice (simulator)": round(scores[mode], 3),
                      "Verified best decision": ", ".join(str(labels[i]) for i in np.flatnonzero(x_best))})
        bars.append((name.split(":")[0], scores, mode))
    fig, ax = plt.subplots(figsize=(9, 4)); w = 0.26
    for k, method in enumerate(["classical", "gate-QAOA", "digital-anneal"]):
        vals = [s[method] for _, s, _ in bars]
        rects = ax.bar(np.arange(len(bars)) + (k - 1) * w, vals, w, label=method)
        for r, (_, _, chosen) in zip(rects, bars):
            if chosen == method: r.set_edgecolor("black"); r.set_linewidth(2.5)
    ax.set_xticks(np.arange(len(bars))); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("score (1 = best, 0 = random)"); ax.axhline(0, c="gray", lw=.8)
    ax.set_title("Each method's score on a perfect simulator (black outline = router's choice)", fontsize=10); ax.legend(); fig.tight_layout(); plt.close(fig)
    return pd.DataFrame(table), fig

with gr.Blocks(title="Q-Router") as app:
    gr.Markdown("# Q-Router\nOne quantum engine for finance and healthcare decisions. It recognizes each problem's shape, borrows what it learned from other industries, and routes it to the best quantum or classical method.")
    with gr.Row():
        pick = gr.CheckboxGroup(list(BUILDERS), value=list(BUILDERS), label="Problems to solve together")
        size = gr.Slider(5, 9, value=6, step=1, label="Choices per problem (= qubits)")
    go = gr.Button("Fingerprint → Route → Solve", variant="primary")
    out_table = gr.Dataframe(label="Router decisions and results")
    out_plot = gr.Plot(label="Scores by method (perfect simulator; real-chip results are below)")
    gr.Markdown("### Recorded results from real IBM quantum hardware")
    gr.Dataframe(value=saved_hardware())
    go.click(run_demo, [pick, size], [out_table, out_plot])

app.launch(share=True)

