#!/usr/bin/env python3
"""
doublewell_traceform_stageI.py

Double-well bistable potential demonstration, consistent (numerically) with the MAIN+SM method:

Pipeline:
  1) Sample microstates x from canonical p(x) ∝ exp(-beta U(x)) for a physical double-well U(x)
  2) Fit a generative energy E_gen(x) from -log p_hat(x) (least-squares on even polynomial)
  3) Pushforward to energy axis: build empirical energy histogram p̂_E on fixed bins
  4) Stage-I (core) trace-form learning on energy axis:
        learn slopes s_j = G'(p̂_{E,j}) in span(F(E_j)) with strict concavity (monotone slopes)
        implemented as alternating projections: isotonic(PAV) <-> span projection (KKT solve)
        with two anchors to fix slope gauge
  5) Trace-form MaxEnt reconstruction (KKT closed form):
        p*_E = (G')^{-1}(F θ)  on the same bins, then normalize
  6) Baseline: Shannon MaxEnt on energy axis with mean constraint only:
        q(E) ∝ exp(-λ E), choose λ so that E_q[E] = E_{p̂_E}[E]  (moment match)

Outputs:
  - generator fit plots (x-space)
  - energy distribution plots (p̂_E vs p*_E vs Shannon mean-matched exponential)
  - summary curves vs beta (U_emp, U_gen, optional S_gen)

Dependencies: numpy, matplotlib, scipy (for minimize only in baseline root search optional; we implement our own bisection)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) Physical double-well model
# -----------------------------
def U_doublewell(x, a=1.0, b=4.0):
    """Physical double-well Hamiltonian U(x)=a x^4 - b x^2."""
    return a * x**4 - b * x**2


def canonical_density(x_grid, beta, a=1.0, b=4.0):
    """Canonical density p(x) ∝ exp(-beta U(x)) on a grid."""
    U = U_doublewell(x_grid, a=a, b=b)
    p_unnorm = np.exp(-beta * U)
    Z = np.trapz(p_unnorm, x_grid)
    return p_unnorm / Z


def sample_from_canonical(beta, N=200_000, a=1.0, b=4.0, L=4.0, Nx=4000, rng=None):
    """
    Approximate canonical sampling by discretizing p(x) on a fine grid and sampling by choice.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    x_grid = np.linspace(-L, L, Nx)
    p_grid = canonical_density(x_grid, beta, a=a, b=b)
    p_discrete = p_grid / np.sum(p_grid)
    samples = rng.choice(x_grid, size=N, p=p_discrete)
    return samples, x_grid, p_grid


# -----------------------------
# 2) Fit generative energy E_gen
# -----------------------------
def fit_generative_energy_from_hist(samples, num_bins=400, deg=4):
    """
    Fit E_gen(x) = w0 + w2 x^2 + w4 x^4 (+ w6 x^6 if deg=6) to -log p̂(x) from histogram.
    Returns coefficients and the histogram centers/targets for plotting.
    """
    dens, edges = np.histogram(samples, bins=num_bins, density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])
    m = dens > 0.0
    xc = xc[m]
    y = -np.log(dens[m])

    cols = [np.ones_like(xc), xc**2]
    if deg >= 4:
        cols.append(xc**4)
    if deg >= 6:
        cols.append(xc**6)
    X = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)

    # pad to 4 coefficients for convenience
    w0 = float(w[0])
    w2 = float(w[1]) if len(w) > 1 else 0.0
    w4 = float(w[2]) if len(w) > 2 else 0.0
    w6 = float(w[3]) if len(w) > 3 else 0.0
    return (w0, w2, w4, w6), (xc, y)


def E_gen_poly(x, w0, w2, w4, w6=0.0):
    return w0 + w2 * x**2 + w4 * x**4 + w6 * x**6


# -----------------------------
# 3) Energy histogram p̂_E
# -----------------------------
def energy_histogram_from_E(E_samples, num_bins=160, qlo=0.2, qhi=99.8):
    """
    Build discrete energy histogram on robust range [qlo,qhi] percentiles.
    Returns (E_centers, p_emp, edges).
    """
    lo, hi = np.percentile(E_samples, [qlo, qhi])
    edges = np.linspace(lo, hi, num_bins + 1)
    counts, _ = np.histogram(E_samples, bins=edges, density=False)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Empty energy histogram.")
    p = counts.astype(float) / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, p, edges


# -----------------------------
# 4) Stage-I trace-form learning (slope-in-span surrogate)
# -----------------------------
def pav_isotonic_nondecreasing(y):
    """PAV for nondecreasing sequence (L2 isotonic regression)."""
    y = np.asarray(y, float)
    n = len(y)
    blocks = [(y[i], 1.0, i, i) for i in range(n)]  # (mean, weight, start, end)
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0] + 1e-15:
            i += 1
            continue
        m1, w1, s1, e1 = blocks[i]
        m2, w2, s2, e2 = blocks[i + 1]
        m = (m1 * w1 + m2 * w2) / (w1 + w2)
        blocks[i] = (m, w1 + w2, s1, e2)
        del blocks[i + 1]
        i = max(i - 1, 0)

    out = np.empty(n, float)
    for m, w, s, e in blocks:
        out[s:e + 1] = m
    return out


def project_to_span_with_anchors(F, s_target, Aeq, beq, ridge=1e-10):
    """
    Solve min ||F theta - s_target||^2 s.t. Aeq theta = beq (KKT system).
    Returns theta, s_proj = F theta.
    """
    F = np.asarray(F, float)
    s_target = np.asarray(s_target, float)
    Aeq = np.asarray(Aeq, float)
    beq = np.asarray(beq, float)

    FtF = F.T @ F + ridge * np.eye(F.shape[1])
    Fts = F.T @ s_target

    KKT = np.block([[FtF, Aeq.T],
                    [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
    rhs = np.concatenate([Fts, beq])
    sol = np.linalg.solve(KKT, rhs)
    theta = sol[:F.shape[1]]
    return theta, F @ theta


def make_energy_features_multiscale(E_centers, K=12, q_lo=0.02, q_hi=0.98,
                                    width_scale_broad=1.2, width_scale_narrow=0.45):
    """
    Feature family F(E): constant + broad RBF bank + narrow RBF bank,
    centered at quantiles of E support.
    """
    E = np.asarray(E_centers, float)
    qs = np.linspace(q_lo, q_hi, K)
    c = np.quantile(E, qs)

    dE = np.median(np.diff(np.sort(E)))
    base = max(3.0 * dE, 1e-6)
    sig_b = base * width_scale_broad
    sig_n = base * width_scale_narrow

    Phi_b = np.column_stack([np.exp(-0.5 * ((E - ck) / sig_b) ** 2) for ck in c])
    Phi_n = np.column_stack([np.exp(-0.5 * ((E - ck) / sig_n) ** 2) for ck in c])

    F = np.column_stack([np.ones(len(E)), Phi_b, Phi_n])
    return F, c, sig_b, sig_n


def learn_traceform_from_pE(p_hat, F, eps=1e-12, iters=250, tol=1e-10, anchor_scale=3.0):
    """
    Stage-I (core) trace-form learning: slope-in-span surrogate.

    We seek slopes s_j = G'(p_hat_j) such that:
      - s approximately lies in span(F): s ≈ F theta (with equality anchors)
      - strict concavity: G' monotone in p (enforced by isotonic projection in probability order)

    Returns:
      theta, gprime(p), inv_gprime(y), G_of_p(p) (generator by integrating g'),
      y_min, y_max (invertible y-range)
    """
    p_hat = np.asarray(p_hat, float)
    n = len(p_hat)

    idx = np.where(p_hat > eps)[0]
    if len(idx) < 10:
        raise RuntimeError("Too few nonzero bins. Reduce binsE or increase N.")

    # sort active bins by decreasing probability (highest first)
    order = idx[np.argsort(-p_hat[idx])]
    i_max = order[0]
    i_min = order[-1]

    # anchors to fix slope gauge
    Aeq = np.vstack([F[i_max], F[i_min]])
    beq = np.array([-anchor_scale, +anchor_scale], float)

    # initial target slopes: linear schedule across probability ranks
    t = np.linspace(-anchor_scale, +anchor_scale, len(order))
    s_target = np.zeros(n, float)
    s_target[order] = t

    theta, s = project_to_span_with_anchors(F, s_target, Aeq, beq)

    for _ in range(iters):
        s_prev = s.copy()

        # enforce monotonicity on the ordered active set (concavity proxy)
        s_ord = s[order]
        s_iso = pav_isotonic_nondecreasing(s_ord)

        s2 = s.copy()
        s2[order] = s_iso

        # project back to span with anchors
        theta, s = project_to_span_with_anchors(F, s2, Aeq, beq)

        if np.linalg.norm(s - s_prev) / (np.linalg.norm(s_prev) + 1e-12) < tol:
            break

    # final monotone slopes on active set
    s_ord = pav_isotonic_nondecreasing(s[order])

    # build p->s mapping on increasing p
    p_ord = p_hat[order]      # decreasing
    p_inc = p_ord[::-1]       # increasing
    s_inc = s_ord[::-1]

    # include p=0 endpoint
    if p_inc[0] > 0:
        p_inc = np.concatenate([[0.0], p_inc])
        s_inc = np.concatenate([[s_inc[0]], s_inc])

    def gprime(p):
        p = np.asarray(p, float)
        pp = np.clip(p, 0.0, float(p_inc[-1]))
        return np.interp(pp, p_inc, s_inc)

    # Build inverse via dense monotone grid
    p_grid = np.linspace(0.0, float(p_inc[-1]), 40000)
    gp_grid = gprime(p_grid)
    gp_u, idx_u = np.unique(gp_grid, return_index=True)
    p_u = p_grid[idx_u]
    y_min = float(gp_u[0])
    y_max = float(gp_u[-1])

    def inv_gprime(y):
        y = np.asarray(y, float)
        yy = np.clip(y, y_min, y_max)
        return np.interp(yy, gp_u, p_u)

    # Build G(p) by integrating g'(p) with G(0)=0 on the same p_grid
    # This is used to report S_gen = sum_j G(p_j).
    G_grid = np.zeros_like(p_grid)
    # cumulative trapezoid
    G_grid[1:] = np.cumsum(0.5 * (gp_grid[1:] + gp_grid[:-1]) * (p_grid[1:] - p_grid[:-1]))

    def G_of_p(p):
        p = np.asarray(p, float)
        pp = np.clip(p, 0.0, float(p_grid[-1]))
        return np.interp(pp, p_grid, G_grid)

    return theta, gprime, inv_gprime, G_of_p, y_min, y_max


# -----------------------------
# 5) Shannon baseline on E-axis: mean-matched exponential
# -----------------------------
def shannon_exp_prior_mean_match(E_centers, Ubar):
    """
    q(E) ∝ exp(-λ E), choose λ so that E_q[E] = Ubar on discrete grid.
    Uses robust bisection on λ.
    """
    E = np.asarray(E_centers, float)

    def mean_E(lam):
        a = -lam * E
        a -= np.max(a)
        w = np.exp(a)
        w /= w.sum()
        return float(w @ E), w

    # bracket
    lam_lo, lam_hi = -50.0, 50.0
    m_lo, _ = mean_E(lam_lo)
    m_hi, _ = mean_E(lam_hi)

    # expand if needed
    for _ in range(40):
        if (m_lo - Ubar) * (m_hi - Ubar) <= 0:
            break
        lam_lo *= 1.5
        lam_hi *= 1.5
        m_lo, _ = mean_E(lam_lo)
        m_hi, _ = mean_E(lam_hi)

    # if still not bracketed, pick best endpoint
    if (m_lo - Ubar) * (m_hi - Ubar) > 0:
        lam = lam_lo if abs(m_lo - Ubar) < abs(m_hi - Ubar) else lam_hi
        _, q = mean_E(lam)
        return float(lam), q

    # bisection
    for _ in range(140):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        m_mid, _ = mean_E(lam_mid)
        if (m_lo - Ubar) * (m_mid - Ubar) <= 0:
            lam_hi, m_hi = lam_mid, m_mid
        else:
            lam_lo, m_lo = lam_mid, m_mid

    lam = 0.5 * (lam_lo + lam_hi)
    _, q = mean_E(lam)
    return float(lam), q


# -----------------------------
# 6) Canonical (Shannon) thermo in x-space (for reference)
# -----------------------------
def canonical_internal_energy_and_entropy(x_grid, beta, a=1.0, b=4.0):
    p_x = canonical_density(x_grid, beta, a=a, b=b)
    Ux = U_doublewell(x_grid, a=a, b=b)
    U_sh = np.trapz(p_x * Ux, x_grid)
    S_sh = -np.trapz(p_x * np.log(p_x + 1e-15), x_grid)
    return float(U_sh), float(S_sh)


# -----------------------------
# 7) Demo runner
# -----------------------------
def run_doublewell_demo():
    # output dir
    outdir = "plots_doublewell_traceform"
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(123)

    # beta sweep
    beta_list = [0.5, 1.0, 2.0]
    N_samples = 200_000

    # storage
    U_emp_list, U_gen_list = [], []
    S_gen_list = []
    U_sh_list, S_sh_list = [], []
    lam_list = []

    for beta in beta_list:
        print(f"\n=== beta = {beta:.3f} (T_bath = {1.0/beta:.3f}) ===")

        # 1) sample canonical x
        samples, x_grid, _ = sample_from_canonical(
            beta, N=N_samples, a=1.0, b=4.0, L=4.0, Nx=4000, rng=rng
        )

        # 2) fit E_gen(x) from -log p_hat(x)
        (w0, w2, w4, w6), (xc, y) = fit_generative_energy_from_hist(
            samples, num_bins=450, deg=4  # deg=4 is enough to represent ax^4 - bx^2 up to gauge
        )
        print(f"Fitted E_gen coeffs: w0={w0:.3f}, w2={w2:.3f}, w4={w4:.3f}, w6={w6:.3f}")

        # plot generator fit
        x_plot = np.linspace(-4.0, 4.0, 600)
        E_fit_plot = E_gen_poly(x_plot, w0, w2, w4, w6)
        U_true = U_doublewell(x_plot)

        # rescale physical U(x) to compare shape (plot-only)
        U_scaled = (U_true - np.min(U_true))
        U_scaled *= (np.max(E_fit_plot) - np.min(E_fit_plot)) / (np.max(U_scaled) - np.min(U_scaled) + 1e-15)
        U_scaled += np.min(E_fit_plot)

        plt.figure(figsize=(6, 4))
        plt.scatter(xc, y, s=10, alpha=0.65, label=r"$-\log \hat p(x)$")
        plt.plot(x_plot, E_fit_plot, lw=2.2, label=r"$E_{\rm gen}(x)$ fit")
        plt.plot(x_plot, U_scaled, ls="--", lw=1.6, label=r"rescaled physical $U(x)$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$-\log \hat p(x)$ / $E_{\rm gen}(x)$")
        plt.title(fr"Double-well: generator fit ($\beta={beta:.2f}$)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"generator_fit_beta_{beta:.2f}.png"), dpi=250)
        plt.close()

        # 3) pushforward to energy axis p̂_E
        E_samples = E_gen_poly(samples, w0, w2, w4, w6)
        E_centers, p_emp, edges = energy_histogram_from_E(E_samples, num_bins=160, qlo=0.2, qhi=99.8)

        U_emp = float(p_emp @ E_centers)
        U_emp_list.append(U_emp)

        # 4) Stage-I trace-form learning (slope-in-span surrogate)
        F, rbf_centers, sig_b, sig_n = make_energy_features_multiscale(E_centers, K=12)
        theta, gprime, inv_gprime, G_of_p, y_min, y_max = learn_traceform_from_pE(
            p_emp, F, anchor_scale=3.0
        )

        # 5) KKT reconstruction: p*_E = (G')^{-1}(F theta)
        y_star = F @ theta
        y_star = np.clip(y_star, y_min + 1e-10, y_max - 1e-10)
        p_star = inv_gprime(y_star)
        p_star = np.clip(p_star, 0.0, None)
        p_star /= p_star.sum()

        U_gen = float(p_star @ E_centers)
        U_gen_list.append(U_gen)

        # Report a trace-form entropy value (Stage-I generator integrated from G')
        S_gen = float(np.sum(G_of_p(p_star)))
        S_gen_list.append(S_gen)

        # 6) Shannon baseline on E-axis: mean match
        lam, q_sh = shannon_exp_prior_mean_match(E_centers, U_emp)
        lam_list.append(lam)

        # plot energy distributions
        width = edges[1] - edges[0]
        plt.figure(figsize=(6, 4))
        plt.bar(E_centers, p_emp, width=0.85 * width, alpha=0.35, label=r"empirical $\hat p_E$")
        plt.plot(E_centers, p_star, lw=2.2, color="red", label=r"trace-form MaxEnt $p_E^\star$")
        plt.plot(E_centers, q_sh, lw=1.8, ls="--", color="#1f77b4",
                 label=rf"Shannon $e^{{-\lambda E}}$ (mean-match), $\lambda={lam:.3g}$")
        plt.xlabel(r"energy $E$")
        plt.ylabel("probability per bin")
        plt.title(fr"Energy distribution ($\beta={beta:.2f}$)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"energy_dist_beta_{beta:.2f}.png"), dpi=250)
        plt.close()

        # Canonical thermodynamics in x-space (reference only)
        U_sh, S_sh = canonical_internal_energy_and_entropy(x_grid, beta, a=1.0, b=4.0)
        U_sh_list.append(U_sh)
        S_sh_list.append(S_sh)

        print(f"U_emp(E_gen)={U_emp:.6f} | U_gen(trace-form)={U_gen:.6f} | S_gen={S_gen:.6f} | lambda_meanmatch={lam:.6g}")
        print(f"(reference physical) U_Sh={U_sh:.6f} | S_Sh={S_sh:.6f}")

    # Summary plot: energies vs beta
    beta_arr = np.array(beta_list, float)

    plt.figure(figsize=(6, 4))
    plt.plot(beta_arr, np.array(U_emp_list), "o-", label=r"$U_{\rm emp}$ on $E_{\rm gen}$ axis")
    plt.plot(beta_arr, np.array(U_gen_list), "s--", label=r"$U_{\rm gen}$ (trace-form MaxEnt)")
    plt.plot(beta_arr, np.array(U_sh_list), "d-", label=r"$U_{\rm Sh}$ (physical canonical)")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"internal energy $U$")
    plt.title("Double-well: internal energy vs inverse temperature")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_energy_vs_beta.png"), dpi=250)
    plt.close()

    # Summary plot: entropies vs beta (optional)
    plt.figure(figsize=(6, 4))
    plt.plot(beta_arr, np.array(S_gen_list), "s--", label=r"$S_{\rm gen}$ (trace-form, energy-axis)")
    plt.plot(beta_arr, np.array(S_sh_list), "o-", label=r"$S_{\rm Sh}$ (physical canonical)")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"entropy (arb. units)")
    plt.title("Double-well: entropy vs inverse temperature")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_entropy_vs_beta.png"), dpi=250)
    plt.close()

    print(f"\nDone. Plots saved in: {outdir}/")


if __name__ == "__main__":
    run_doublewell_demo()
