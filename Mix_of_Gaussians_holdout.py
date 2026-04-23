#!/usr/bin/env python3
"""
Mixture_of_Gaussians.py

Synthetic “data system” experiment:

  - 1D three–component Gaussian mixture with varying separation mu.
  - Data–driven generative energy E_gen(x) fitted to -log p̂(x).
  - Quadratic–program MaxEnt on the energy axis with band entropy
    (same band-based placeholder as in the double-well script).
  - Plots:
      * generator fits in x-space (with Gaussian-MaxEnt overlay),
      * energy-space histograms + QP solution + Shannon exp prior,
      * summary U_gen(mu) and entropies vs mu.

No knowledge of the mixture parameters is used in the construction
of E_gen(x) or the QP; they are only used to compute “true” Shannon
entropy on x for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

np.random.seed(123)

# ----------------------------------------------------------------------
# 1. Mixture-of-Gaussians data system
# ----------------------------------------------------------------------

def sample_mixture(mu, N=200_000,
                   weights=None,
                   sigmas=None):
    """
    Draw N samples from a 3-component Gaussian mixture:

        w1 N(-mu, sigma1^2) + w2 N(0, sigma2^2) + w3 N(mu, sigma3^2)

    Returns: samples (shape [N,])
    """
    if weights is None:
        weights = np.array([0.3, 0.4, 0.3])
    if sigmas is None:
        sigmas = np.array([0.4, 0.8, 0.4])

    weights = np.asarray(weights)
    sigmas = np.asarray(sigmas)
    assert len(weights) == 3 and len(sigmas) == 3

    means = np.array([-mu, 0.0, mu])

    comps = np.random.choice(3, size=N, p=weights / weights.sum())
    eps = np.random.randn(N) * sigmas[comps]
    samples = means[comps] + eps
    return samples


def mixture_density(x, mu,
                    weights=None,
                    sigmas=None):
    """
    Analytic PDF of the 3-component Gaussian mixture at grid x.
    Used ONLY to compute the “true” Shannon entropy in x-space.
    """
    if weights is None:
        weights = np.array([0.3, 0.4, 0.3])
    if sigmas is None:
        sigmas = np.array([0.4, 0.8, 0.4])

    weights = np.asarray(weights)
    sigmas = np.asarray(sigmas)
    means = np.array([-mu, 0.0, mu])

    pdf = np.zeros_like(x, dtype=float)
    for w, m, s in zip(weights, means, sigmas):
        pdf += w * (1.0 / (np.sqrt(2.0 * np.pi) * s)) * np.exp(
            -0.5 * ((x - m) / s) ** 2
        )
    pdf /= weights.sum()
    return pdf


# ----------------------------------------------------------------------
# 2. Generative energy fit: E_gen(x) = w0 + w2 x^2 + w4 x^4 + w6 x^6
# ----------------------------------------------------------------------

def fit_generative_energy_from_hist(samples,
                                    num_bins=300,
                                    poly_deg=6):
    """
    Fit E_gen(x) = w0 + w2 x^2 + w4 x^4 + w6 x^6
    (up to the specified even degree: 2,4,6) to -log p̂(x)
    from a histogram of the samples.

    Returns:
        w0, w2, w4, w6,
        x_fit, neglog_p_fit, Egen_fit
    """
    counts, edges = np.histogram(samples, bins=num_bins, density=True)
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0.0

    x_fit = x_centers[mask]
    neglog_p_fit = -np.log(counts[mask])

    # Design matrix: [1, x^2, x^4, x^6] (truncate if poly_deg < 6)
    cols = [np.ones_like(x_fit), x_fit**2]
    if poly_deg >= 4:
        cols.append(x_fit**4)
    if poly_deg >= 6:
        cols.append(x_fit**6)
    X = np.column_stack(cols)

    w, _, _, _ = np.linalg.lstsq(X, neglog_p_fit, rcond=None)

    # Unpack with safe defaults
    w0 = w[0]
    w2 = w[1] if len(w) > 1 else 0.0
    w4 = w[2] if len(w) > 2 else 0.0
    w6 = w[3] if len(w) > 3 else 0.0

    Egen_fit = w0 + w2 * x_fit**2 + w4 * x_fit**4 + w6 * x_fit**6
    return w0, w2, w4, w6, x_fit, neglog_p_fit, Egen_fit


def E_gen(x, w0, w2, w4, w6):
    """Evaluate the generative energy polynomial at x."""
    return w0 + w2 * x**2 + w4 * x**4 + w6 * x**6


# ----------------------------------------------------------------------
# 3. Energy histogram from E_gen(x)
# ----------------------------------------------------------------------

def energy_histogram(samples, w0, w2, w4, w6, num_bins=200):
    """
    Map microstates x -> energies E_gen(x) and build a histogram.
    Returns (E_centers, p_emp, counts, edges).
    """
    E_samples = E_gen(samples, w0, w2, w4, w6)

    # Robust range to avoid extremes dominating bins
    E_min, E_max = np.percentile(E_samples, [0.1, 99.9])
    E_edges = np.linspace(E_min, E_max, num_bins + 1)

    counts, edges = np.histogram(E_samples, bins=E_edges, density=False)
    total = counts.sum()
    if total == 0:
        raise ValueError("No counts in energy histogram.")

    p_emp = counts.astype(float) / total
    E_centers = 0.5 * (edges[:-1] + edges[1:])
    return E_centers, p_emp, counts, edges


# ----------------------------------------------------------------------
# 4. Core/shape constraints on energy bands
# ----------------------------------------------------------------------

def build_core_shape_constraints(E_centers,
                                 p_emp,
                                 q_low=0.25,
                                 q_high=0.75,
                                 include_mean=False):
    """
    Build linear constraints for band masses in energy-space.

    Bands are defined via empirical CDF quantiles q_low and q_high.
    """
    E = E_centers
    n = len(E)

    # Empirical CDF
    cdf = np.cumsum(p_emp)
    if cdf[-1] <= 0:
        raise ValueError("Empirical probabilities sum to zero.")
    cdf /= cdf[-1]

    def find_cut(q):
        idx = np.searchsorted(cdf, q)
        idx = np.clip(idx, 0, n - 1)
        return E[idx]

    E_low = find_cut(q_low)
    E_high = find_cut(q_high)

    band1 = (E <= E_low)
    band2 = (E > E_low) & (E <= E_high)
    band3 = (E > E_high)

    m1_emp = p_emp[band1].sum()
    m2_emp = p_emp[band2].sum()
    m3_emp = p_emp[band3].sum()

    rows = []
    rhs = []

    # Normalization
    rows.append(np.ones(n))
    rhs.append(1.0)

    # Band masses
    rows.append(band1.astype(float)); rhs.append(m1_emp)
    rows.append(band2.astype(float)); rhs.append(m2_emp)
    rows.append(band3.astype(float)); rhs.append(m3_emp)

    if include_mean:
        U_emp = np.dot(p_emp, E)
        rows.append(E.copy())
        rhs.append(U_emp)

    A = np.vstack(rows)
    b = np.array(rhs)
    bands = (band1, band2, band3)

    return A, b, bands


# ----------------------------------------------------------------------
# 5. Band-based quadratic form for generative entropy
# ----------------------------------------------------------------------

def build_band_quadratic_Q_c(n, bands, alphas):
    """
    Build Q, c for band-based generative entropy

        H_gen(p) = sum_k alpha_k [ m_k - m_k^2 ],
        m_k = sum_{i in band_k} p_i.

    which is a quadratic function of p.
    """
    Q = np.zeros((n, n))
    c = np.zeros(n)

    for band_mask, alpha in zip(bands, alphas):
        idx = np.where(band_mask)[0]
        # Linear terms
        for i in idx:
            c[i] -= alpha
        # Quadratic terms (2 alpha m_k^2)
        for i in idx:
            for j in idx:
                Q[i, j] += 2.0 * alpha

    return Q, c


def compute_band_entropy(p, bands, alphas):
    """
    Evaluate H_gen(p) = sum_k alpha_k [ m_k - m_k^2 ].
    """
    H = 0.0
    for band_mask, alpha in zip(bands, alphas):
        m = p[band_mask].sum()
        H += alpha * (m - m**2)
    return H


# ----------------------------------------------------------------------
# 6. Solve quadratic program in probability simplex
# ----------------------------------------------------------------------

def solve_generative_qp(E_centers,
                        p_emp,
                        q_low=0.25,
                        q_high=0.75,
                        include_mean=False,
                        alphas=None):
    """
    Solve the quadratic program that maximizes band-based H_gen
    under normalization + band-mass constraints.
    """
    n = len(E_centers)
    A, b, bands = build_core_shape_constraints(
        E_centers, p_emp, q_low=q_low, q_high=q_high,
        include_mean=include_mean
    )

    if alphas is None:
        alphas = np.array([1.0, 0.5, 0.2])
    alphas = np.asarray(alphas)

    Q, c = build_band_quadratic_Q_c(n, bands, alphas)

    def objective(p):
        return 0.5 * p @ (Q @ p) + c @ p

    def grad(p):
        return Q @ p + c

    cons = []
    # Linear equality constraints A p = b
    for k in range(A.shape[0]):
        a_row = A[k, :]
        b_k = b[k]
        cons.append({
            'type': 'eq',
            'fun': lambda p, a=a_row, b0=b_k: np.dot(a, p) - b0,
            'jac': lambda p, a=a_row: a
        })

    bounds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess: clipped empirical distribution
    p0 = p_emp.copy()
    eps = 1e-12
    p0 = np.clip(p0, eps, None)
    p0 /= p0.sum()

    res = minimize(
        objective, p0,
        method='SLSQP',
        jac=grad,
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-10, 'maxiter': 20000, 'disp': False}
    )

    if not res.success:
        print("WARNING: QP did not converge:", res.message)

    p_qp = res.x
    p_qp = np.clip(p_qp, 0.0, None)
    p_qp /= p_qp.sum()

    U_gen = np.dot(p_qp, E_centers)
    H_gen = compute_band_entropy(p_qp, bands, alphas)
    U_emp = np.dot(p_emp, E_centers)

    return p_qp, U_gen, H_gen, U_emp


# ----------------------------------------------------------------------
# 7. Shannon “exp(-lambda E)” prior in energy-space
# ----------------------------------------------------------------------

def exponential_prior_on_energy(E_centers, p_emp):
    """
    Fit log p_emp(E) ≈ a - lambda * E (least squares on bins with p_emp>0),
    then return normalized exponential prior p_sh(E) ∝ exp(a - lambda E).
    """
    mask = p_emp > 0
    if mask.sum() < 2:
        # Not enough data to fit; fall back to uniform
        p_sh = np.ones_like(p_emp) / len(p_emp)
        return p_sh, 0.0

    E = E_centers[mask]
    y = np.log(p_emp[mask])

    X = np.column_stack([np.ones_like(E), -E])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, lam = coeffs

    p_raw = np.exp(a - lam * E_centers)
    p_raw = np.clip(p_raw, 0.0, None)
    s = p_raw.sum()
    if s <= 0:
        p_sh = np.ones_like(p_emp) / len(p_emp)
    else:
        p_sh = p_raw / s
    return p_sh, lam


# ----------------------------------------------------------------------
# 8. Shannon entropy in x-space for the mixture
# ----------------------------------------------------------------------

def shannon_entropy_mixture(mu,
                            weights=None,
                            sigmas=None,
                            L=6.0,
                            Nx=4000):
    """
    Compute continuous Shannon entropy S_sh = -∫ p(x) log p(x) dx
    of the mixture via numerical integration on [-L, L].
    """
    x_grid = np.linspace(-L, L, Nx)
    p = mixture_density(x_grid, mu, weights=weights, sigmas=sigmas)
    p = np.clip(p, 1e-15, None)
    S = -np.trapz(p * np.log(p), x_grid)
    return S


# ----------------------------------------------------------------------
# 9. Main experiment over several mu values
# ----------------------------------------------------------------------

def run_mixture_demo():
    output_dir = "plots_mixture"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in '{output_dir}/'")

    mu_list = [1.0, 2.0, 3.0]
    N_samples = 200_000

    weights = np.array([0.3, 0.4, 0.3])
    sigmas = np.array([0.4, 0.8, 0.4])

    U_gen_list = []
    H_gen_list = []
    S_sh_list = []
    Ex2_list = []

    # For plotting ranges in x
    L_x = 6.0

    for mu in mu_list:
        print(f"\n=== mu = {mu:.2f} ===")

        # 1) Sample from the mixture
        samples = sample_mixture(mu, N=N_samples,
                                 weights=weights,
                                 sigmas=sigmas)

        # 2) Fit generative energy in x-space
        (w0, w2, w4, w6,
         x_fit, neglog_p_fit, Egen_fit) = fit_generative_energy_from_hist(
            samples, num_bins=300, poly_deg=6
        )

        print(f"Fitted generator coefficients:"
              f" w0={w0:.3f}, w2={w2:.3f}, w4={w4:.3f}, w6={w6:.3f}")

        # x-grid for plotting E_gen and Gaussian-MaxEnt energy
        x_plot = np.linspace(-L_x, L_x, 600)
        E_gen_plot = E_gen(x_plot, w0, w2, w4, w6)

        # Best Gaussian "MaxEnt" energy based on mean and variance of samples
        m = samples.mean()
        var = samples.var()
        Eg_gauss = (x_plot - m) ** 2 / (2.0 * var)

        # Rescale Eg_gauss to be comparable to E_gen_plot (match min & range)
        Eg_gauss_scaled = (Eg_gauss - Eg_gauss.min())
        if Eg_gauss_scaled.max() > 0:
            Eg_gauss_scaled = Eg_gauss_scaled * (
                (E_gen_plot.max() - E_gen_plot.min()) /
                Eg_gauss_scaled.max()
            ) + E_gen_plot.min()
        else:
            Eg_gauss_scaled = Eg_gauss_scaled + E_gen_plot.min()

        # Plot A: generator fit in x-space
        plt.figure(figsize=(6, 4))
        plt.plot(x_fit, neglog_p_fit, 'o', ms=3,
                 label=r'$-\log \hat{p}(x)$')
        plt.plot(x_plot, E_gen_plot, '-', lw=2,
                 label=r'$E_{\mathrm{gen}}(x)$ fit')
        plt.plot(x_plot, Eg_gauss_scaled, '--', lw=1.4,
                 label='Gaussian MaxEnt energy')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$-\log \hat{p}(x)$ / $E_{\mathrm{gen}}(x)$')
        plt.title(fr'Mixture: generator fit at $\mu={mu:.2f}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"generator_fit_mu_{mu:.2f}.png"
        ))
        plt.close()

        # 3) Energy histogram and QP in energy-space
        E_centers, p_emp, counts_E, edges_E = energy_histogram(
            samples, w0, w2, w4, w6, num_bins=150
        )

        p_qp, U_gen, H_gen, U_emp = solve_generative_qp(
            E_centers, p_emp,
            q_low=0.25, q_high=0.75,
            include_mean=False,
            alphas=np.array([1.0, 0.5, 0.2])
        )

        print(f"Empirical energy mean U_emp = {U_emp:.4f}")
        print(f"Generative internal energy U_gen (QP) = {U_gen:.4f}")
        print(f"Band-based generative entropy H_gen = {H_gen:.6f}")

        U_gen_list.append(U_gen)
        H_gen_list.append(H_gen)

        # 4) Shannon exp(-lambda E) prior fit in energy-space
        p_sh, lam = exponential_prior_on_energy(E_centers, p_emp)
        print(f"Shannon-style exp(-lambda E) prior: lambda = {lam:.4f}")

        # Plot B: Energy distributions (empirical vs QP vs Shannon prior)
        width = edges_E[1] - edges_E[0]
        plt.figure(figsize=(6, 4))
        plt.bar(E_centers, p_emp,
                width=0.8 * width,
                alpha=0.35,
                label='empirical $p_E$')
        plt.plot(E_centers, p_qp, '-r', lw=2,
                 label=r'QP solution $p_E^\star$')
        plt.plot(E_centers, p_sh, '--', lw=1.5,
                 label=r'Shannon $\exp(-\lambda E)$ prior')
        plt.xlabel(r'energy $E$')
        plt.ylabel(r'probability per bin')
        plt.title(fr'Energy distribution at $\mu={mu:.2f}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"energy_dist_mu_{mu:.2f}.png"
        ))
        plt.close()

        # 5) Shannon entropy in x-space (using analytic mixture density)
        S_sh = shannon_entropy_mixture(
            mu, weights=weights, sigmas=sigmas,
            L=L_x, Nx=4000
        )
        S_sh_list.append(S_sh)
        print(f"Shannon entropy in x-space S_sh = {S_sh:.4f}")

        # Simple second-moment macro in x-space for comparison
        Ex2 = np.mean(samples**2)
        Ex2_list.append(Ex2)

    # ------------------------------------------------------------------
    # Summary plots vs mu
    # ------------------------------------------------------------------
    mu_arr = np.array(mu_list)
    U_gen_arr = np.array(U_gen_list)
    H_gen_arr = np.array(H_gen_list)
    S_sh_arr = np.array(S_sh_list)
    Ex2_arr = np.array(Ex2_list)

    # Summary 1: internal energy vs separation mu
    plt.figure(figsize=(6, 4))
    plt.plot(mu_arr, Ex2_arr, 'o-', lw=1.5,
             label=r'$\mathbb{E}[x^2]$ (mixture)')
    plt.plot(mu_arr, U_gen_arr, 's--', lw=2,
             label=r'generative $U_{\mathrm{gen}}$ (QP)')
    plt.xlabel(r'separation $\mu$')
    plt.ylabel(r'internal energy / second moment')
    plt.title('Internal energy vs mixture separation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "summary_energy_vs_mu.png"
    ))
    plt.close()

    # Summary 2: Shannon vs generative entropy vs mu
    plt.figure(figsize=(6, 4))
    plt.plot(mu_arr, S_sh_arr, 'o-', lw=1.5,
             label=r'canonical $S_{\mathrm{Sh}}$ (mixture)')
    plt.plot(mu_arr, H_gen_arr, 's--', lw=2,
             label=r'generative $H_{\mathrm{gen}}$ (QP)')
    plt.xlabel(r'separation $\mu$')
    plt.ylabel('entropy')
    plt.title('Entropy vs mixture separation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, "summary_entropy_vs_mu.png"
    ))
    plt.close()

    print(f"\nDone. All plots saved in '{output_dir}/'.")


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 9. Holdout test: learn "generative entropy functional" on train subset,
#    then predict held-out energy histogram under held-out constraints
#    WITHOUT re-learning the functional.
# ----------------------------------------------------------------------

def energy_histogram_with_edges(samples, w0, w2, w4, w6, edges):
    """
    Same as energy_histogram, but uses fixed bin edges (e.g., train-defined)
    so that train/val histograms live on the same support.
    """
    E_samples = E_gen(samples, w0, w2, w4, w6)
    counts, _ = np.histogram(E_samples, bins=edges, density=False)
    total = counts.sum()
    if total == 0:
        raise ValueError("No counts in energy histogram (with fixed edges).")
    p_emp = counts.astype(float) / total
    E_centers = 0.5 * (edges[:-1] + edges[1:])
    return E_centers, p_emp, counts


def bands_from_reference(E_centers, p_ref, q_low=0.25, q_high=0.75):
    """
    Define band masks from a *reference* empirical distribution p_ref,
    via quantile cuts on the reference CDF.
    Returns: (band1, band2, band3), (E_low, E_high)
    """
    E = E_centers
    cdf = np.cumsum(p_ref)
    cdf = cdf / cdf[-1]

    def find_cut(q):
        idx = np.searchsorted(cdf, q)
        idx = np.clip(idx, 0, len(E) - 1)
        return E[idx]

    E_low = find_cut(q_low)
    E_high = find_cut(q_high)

    band1 = (E <= E_low)
    band2 = (E > E_low) & (E <= E_high)
    band3 = (E > E_high)
    return (band1, band2, band3), (E_low, E_high)


def constraints_from_bands(E_centers, p_emp, bands, include_mean=False):
    """
    Build A p = b constraints using *fixed* band masks.
    Here b depends on the current empirical p_emp (train or val),
    but the constraint FUNCTIONS (bands) are fixed.
    """
    band1, band2, band3 = bands
    n = len(E_centers)

    m1 = p_emp[band1].sum()
    m2 = p_emp[band2].sum()
    m3 = p_emp[band3].sum()

    rows = [np.ones(n),
            band1.astype(float),
            band2.astype(float),
            band3.astype(float)]
    rhs = [1.0, m1, m2, m3]

    if include_mean:
        rhs.append(np.dot(p_emp, E_centers))
        rows.append(E_centers.copy())

    A = np.vstack(rows)
    b = np.array(rhs)
    return A, b, (m1, m2, m3)


def solve_fixed_entropy_qp(p_init, Q, c, A, b):
    """
    Solve: maximize H_gen(p) = - (1/2 p^T Q p + c^T p)  (up to sign),
    subject to A p = b, 0<=p<=1.

    NOTE: In the original script the SLSQP 'objective' is 0.5 p^T Q p + c^T p
    and we MINIMIZE it. That is equivalent to MAXIMIZING (-0.5 p^T Q p - c^T p).
    """
    n = len(p_init)

    def objective(p):
        return 0.5 * p @ (Q @ p) + c @ p

    def grad(p):
        return Q @ p + c

    cons = []
    for k in range(A.shape[0]):
        a_row = A[k, :]
        b_k = b[k]
        cons.append({
            'type': 'eq',
            'fun': lambda p, a=a_row, b0=b_k: np.dot(a, p) - b0,
            'jac': lambda p, a=a_row: a
        })

    bounds = [(0.0, 1.0) for _ in range(n)]

    eps = 1e-12
    p0 = np.clip(p_init.copy(), eps, None)
    p0 /= p0.sum()

    res = minimize(
        objective, p0,
        method='SLSQP',
        jac=grad,
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-10, 'maxiter': 20000, 'disp': False}
    )

    if not res.success:
        print("WARNING: fixed-entropy QP did not converge:", res.message)

    p_sol = np.clip(res.x, 0.0, None)
    p_sol /= p_sol.sum()
    return p_sol


def shannon_maxent_on_bands(bands, masses):
    """
    Shannon MaxEnt under fixed band-mass constraints:
      maximize -sum p_j log p_j
      s.t. sum_{j in band k} p_j = m_k
    Solution: uniform within each band.
    """
    band1, band2, band3 = bands
    m1, m2, m3 = masses

    n = len(band1)
    p = np.zeros(n, dtype=float)

    for band, m in [(band1, m1), (band2, m2), (band3, m3)]:
        idx = np.where(band)[0]
        if len(idx) == 0:
            continue
        p[idx] = m / len(idx)

    # Numerical clean-up
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence (base e) for discrete distributions.
    """
    p = np.clip(np.asarray(p, float), eps, None)
    q = np.clip(np.asarray(q, float), eps, None)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def wasserstein_1d(x, p, q):
    """
    1D Earth Mover / Wasserstein-1 distance for discrete distributions
    supported on sorted points x (assumed increasing).
    Uses CDF integral: W1 = sum |CDF_p - CDF_q| * dx
    """
    x = np.asarray(x, float)
    p = np.asarray(p, float); p = p / p.sum()
    q = np.asarray(q, float); q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    # dx per bin center spacing (approx using edges midpoints)
    dx = np.diff(x)
    dx = np.r_[dx, dx[-1]]  # last spacing replicate
    return np.sum(np.abs(cdf_p - cdf_q) * dx)


def run_mixture_holdout_test(train_frac=0.7, seed=123,
                             mu=2.0, N_samples=200_000,
                             num_bins_x=300, num_bins_E=150,
                             q_low=0.25, q_high=0.75,
                             alphas=np.array([1.0, 0.5, 0.2]),
                             output_dir="plots_mixture_holdout"):
    """
    Within-dataset holdout test on a single mixture separation mu.

    Learn the energy fit and the entropy-QP functional on TRAIN only.
    Then, on VAL:
      - compute held-out constraints (band masses on fixed bands)
      - solve MaxEnt with *fixed functional* (same Q,c,bands)
      - compare predicted p_val^* to empirical p_val

    Also computes a Shannon MaxEnt baseline under the same band constraints.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    weights = np.array([0.3, 0.4, 0.3])
    sigmas = np.array([0.4, 0.8, 0.4])

    # 1) Sample and split
    samples = sample_mixture(mu, N=N_samples, weights=weights, sigmas=sigmas)
    idx = rng.permutation(len(samples))
    n_tr = int(train_frac * len(samples))
    tr = samples[idx[:n_tr]]
    va = samples[idx[n_tr:]]

    print(f"\n=== HOLDOUT TEST (mu={mu:.2f}) ===")
    print(f"Total N={len(samples)} | train={len(tr)} | val={len(va)} | seed={seed}")

    # 2) Fit generative energy on TRAIN only
    (w0, w2, w4, w6,
     x_fit, neglog_p_fit, Egen_fit) = fit_generative_energy_from_hist(
        tr, num_bins=num_bins_x, poly_deg=6
    )
    print(f"Energy fit (train-only): w0={w0:.3f}, w2={w2:.3f}, w4={w4:.3f}, w6={w6:.3f}")

    # 3) Train energy histogram defines fixed support/bins
    E_cent_tr, p_tr, _, edges_E = energy_histogram(tr, w0, w2, w4, w6, num_bins=num_bins_E)

    # 4) Define fixed bands from TRAIN distribution
    bands, (E_low, E_high) = bands_from_reference(E_cent_tr, p_tr, q_low=q_low, q_high=q_high)
    print(f"Band cuts from train CDF: E_low={E_low:.4f}, E_high={E_high:.4f}")

    # 5) Build the fixed entropy functional (Q,c) from bands
    Q, c = build_band_quadratic_Q_c(len(E_cent_tr), bands, np.asarray(alphas))

    # 6) Solve TRAIN MaxEnt (mainly for sanity)
    A_tr, b_tr, masses_tr = constraints_from_bands(E_cent_tr, p_tr, bands, include_mean=False)
    p_tr_star = solve_fixed_entropy_qp(p_tr, Q, c, A_tr, b_tr)

    # 7) Validation empirical histogram on the SAME edges
    E_cent_va, p_va, _ = energy_histogram_with_edges(va, w0, w2, w4, w6, edges_E)

    # 8) Validation constraints (same bands, new RHS)
    A_va, b_va, masses_va = constraints_from_bands(E_cent_va, p_va, bands, include_mean=False)

    # 9) Predict held-out distribution by MaxEnt with fixed functional
    p_va_star = solve_fixed_entropy_qp(p_va, Q, c, A_va, b_va)

    # 10) Shannon MaxEnt baseline under same band constraints
    p_va_sh = shannon_maxent_on_bands(bands, masses_va)

    # 11) Distances
    js_gen = js_divergence(p_va, p_va_star)
    w1_gen = wasserstein_1d(E_cent_va, p_va, p_va_star)

    js_sh = js_divergence(p_va, p_va_sh)
    w1_sh = wasserstein_1d(E_cent_va, p_va, p_va_sh)

    print("\nHoldout fit quality (VAL):")
    print(f"  Generative-QP vs empirical: JS={js_gen:.6g}, W1={w1_gen:.6g}")
    print(f"  Shannon-band vs empirical:  JS={js_sh:.6g}, W1={w1_sh:.6g}")

    # 12) Plot: train/val empirical + predicted
    width = edges_E[1] - edges_E[0]
    plt.figure(figsize=(7, 4))
    plt.bar(E_cent_va, p_va, width=0.85 * width, alpha=0.35, label=r'VAL empirical $\hat p_E$')
    plt.plot(E_cent_va, p_va_star, '-r', lw=2, label=r'VAL prediction (fixed functional)')
    plt.plot(E_cent_va, p_va_sh, '--', lw=1.6, label=r'VAL Shannon MaxEnt (band constraints)')
    plt.axvline(E_low, lw=1.0, alpha=0.5)
    plt.axvline(E_high, lw=1.0, alpha=0.5)
    plt.xlabel(r'energy $E$')
    plt.ylabel(r'probability per bin')
    plt.title(fr'Holdout test at $\mu={mu:.2f}$ | JS(gen)={js_gen:.3g}, JS(sh)={js_sh:.3g}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"holdout_energy_mu_{mu:.2f}.png"))
    plt.close()

    # 13) Optional: print band masses (sanity)
    print("\nBand masses (train vs val):")
    print(f"  train masses: {masses_tr}")
    print(f"  val   masses: {masses_va}")

    return {
        "mu": mu,
        "seed": seed,
        "train_frac": train_frac,
        "js_gen": js_gen, "w1_gen": w1_gen,
        "js_sh": js_sh, "w1_sh": w1_sh,
        "E_low": E_low, "E_high": E_high
    }


# ----------------------------------------------------------------------
# Main entry point with modes
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mixture of Gaussians demo + holdout test.")
    parser.add_argument("--mode", choices=["demo", "holdout"], default="demo",
                        help="Run the original demo or the within-dataset holdout test.")
    parser.add_argument("--mu", type=float, default=2.0, help="Mixture separation for holdout mode.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for holdout split.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction for holdout split.")
    parser.add_argument("--N", type=int, default=200000, help="Total number of samples.")
    args = parser.parse_args()

    if args.mode == "demo":
        run_mixture_demo()
    else:
        run_mixture_holdout_test(train_frac=args.train_frac, seed=args.seed,
                                 mu=args.mu, N_samples=args.N)
