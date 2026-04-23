#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Single_well.py  (HARMONIC / SINGLE-WELL PROTOCOL — METHOD-CONSISTENT)
#
# Goal (conceptually aligned with MAIN+SM):
#  1) Generate a 1D harmonic “data system” at several bath temperatures T.
#  2) Fit a generative energy E_gen(x) from -log p̂(x) at a reference temperature T_ref.
#  3) Push-forward to energy space using the *train-fitted* E_gen, and build a *fixed*
#     energy discretisation from training only (no leakage).
#  4) Stage-I: infer a trace-form entropy derivative g'(p) from the training energy
#     histogram p_E,tr on that fixed grid (monotone + span-smoothing on E).
#  5) Protocol: hold g' fixed, update only the macrostate (mean energy U(T)) for each
#     bath T, and compute the trace-form MaxEnt prediction on the energy grid via KKT:
#         p_i(T) = (g')^{-1}(λ0(T) + λ1(T) E_i),
#     where (λ0,λ1) are set so that sum_i p_i = 1 and sum_i p_i E_i = U(T).
#  6) Compare to Shannon MaxEnt on energy axis: exp(-λE) with mean-energy matching.
#  7) Compute protocol curve S_gen(U) using the inferred trace-form generator G
#     (numerical integral of g'), and operational temperature:
#         T_gen = 1 / (dS_gen/dU).
#
# Outputs:
#  - single_well_protocol.png : 2x2 panel figure (calibration, energy-axis prediction,
#                              S_gen(U), and T_gen vs bath T)
#  - single_well_w2_vs_beta.png : diagnostic linearity of fitted quadratic coefficient vs β
# ---------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Utilities ------------------------------------

def js_divergence(p, q, eps=1e-15):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pav_isotonic_nondecreasing(y):
    """Pool-Adjacent-Violators for nondecreasing sequence."""
    y = np.asarray(y, float)
    n = len(y)
    blocks = [(y[i], 1.0, i, i) for i in range(n)]
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


def project_to_span_with_anchors(X, s, Aeq, beq, ridge=1e-10):
    """
    Solve min_theta ||X theta - s||^2 + ridge||theta||^2  s.t. Aeq theta = beq
    via KKT.
    """
    X = np.asarray(X, float)
    s = np.asarray(s, float)
    Aeq = np.asarray(Aeq, float)
    beq = np.asarray(beq, float)

    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    Xts = X.T @ s

    KKT = np.block([
        [XtX, Aeq.T],
        [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]
    ])
    rhs = np.concatenate([Xts, beq])
    sol = np.linalg.solve(KKT, rhs)
    theta = sol[:X.shape[1]]
    s_proj = X @ theta
    return theta, s_proj


def make_energy_features_multiscale(E_centers, K=12, q_lo=0.02, q_hi=0.98,
                                    width_scale_broad=1.2, width_scale_narrow=0.45):
    """
    Smooth basis on the energy axis for slope-span regularisation:
      F(E) = [1, broad RBFs at quantiles, narrow RBFs at quantiles]
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


def learn_traceform_from_pE(p_hat, F, eps=1e-12, iters=220, tol=1e-10, anchor_scale=3.0):
    """
    Stage-I (simplified but method-consistent):
      - define a target monotone slope schedule on nonzero bins
      - alternate: project to span(F) with anchors, then isotonic enforce monotonicity
    Returns:
      theta, gprime(p), inv_gprime(y), and slope range [y_min,y_max]
    """
    n = len(p_hat)
    idx = np.where(p_hat > eps)[0]
    if len(idx) < 10:
        raise RuntimeError("Too few nonzero energy bins. Reduce binsE or increase N.")

    # Sort by decreasing probability (so slope should be nondecreasing along this order)
    order = idx[np.argsort(-p_hat[idx])]
    i_max = order[0]
    i_min = order[-1]

    # Two anchor equalities on extreme-probability bins (keeps y-range controlled)
    Aeq = np.vstack([F[i_max], F[i_min]])
    beq = np.array([-anchor_scale, +anchor_scale], float)

    # Initial target slope schedule on active set
    t = np.linspace(-anchor_scale, +anchor_scale, len(order))
    s_target = np.zeros(n, float)
    s_target[order] = t

    theta, s = project_to_span_with_anchors(F, s_target, Aeq, beq)

    for _ in range(iters):
        s_prev = s.copy()
        s_ord = s[order]
        s_iso = pav_isotonic_nondecreasing(s_ord)
        s2 = s.copy()
        s2[order] = s_iso
        theta, s = project_to_span_with_anchors(F, s2, Aeq, beq)
        if np.linalg.norm(s - s_prev) / (np.linalg.norm(s_prev) + 1e-12) < tol:
            break

    # Final isotonic enforcement on the active set
    s_ord = pav_isotonic_nondecreasing(s[order])

    # Build mapping p -> s on increasing-p axis
    p_ord = p_hat[order]          # decreasing
    p_inc = p_ord[::-1]           # increasing
    s_inc = s_ord[::-1]

    # include p=0 to stabilise inverse near 0
    if p_inc[0] > 0:
        p_inc = np.concatenate([[0.0], p_inc])
        s_inc = np.concatenate([[s_inc[0]], s_inc])

    def gprime(p):
        p = np.asarray(p, float)
        pp = np.clip(p, 0.0, float(p_inc[-1]))
        return np.interp(pp, p_inc, s_inc)

    # dense grid for inversion
    p_grid = np.linspace(0.0, float(p_inc[-1]), 50000)
    gp_grid = gprime(p_grid)

    # enforce unique monotone mapping for inversion
    gp_u, idx_u = np.unique(gp_grid, return_index=True)
    p_u = p_grid[idx_u]
    y_min = float(gp_u[0])
    y_max = float(gp_u[-1])

    def inv_gprime(y):
        y = np.asarray(y, float)
        yy = np.clip(y, y_min, y_max)
        return np.interp(yy, gp_u, p_u)

    return theta, gprime, inv_gprime, y_min, y_max


def build_G_from_gprime(gprime, p_max, ngrid=60000):
    """
    Numerically integrate g' to obtain generator G with G(0)=0:
      G(p) = ∫_0^p g'(u) du
    """
    p = np.linspace(0.0, float(p_max), ngrid)
    gp = gprime(p)
    G = np.cumsum((gp[:-1] + gp[1:]) * 0.5 * (p[1:] - p[:-1]))
    G = np.concatenate([[0.0], G])

    def Gfun(pp):
        pp = np.asarray(pp, float)
        qq = np.clip(pp, 0.0, float(p_max))
        return np.interp(qq, p, G)

    return Gfun


# --------------------------- Harmonic system --------------------------------

def sample_harmonic(T, N, rng, k=1.0):
    """
    Canonical harmonic oscillator: p(x) is Gaussian N(0, T/k).
    Sampling directly is a valid shortcut (same equilibrium).
    """
    sigma = np.sqrt(T / k)
    return rng.standard_normal(N) * sigma


def fit_quadratic_energy_from_hist(x, bins=450):
    """
    Fit -log p_hat(x) ~ w0 + w2 x^2  using histogram density (discard empty bins).
    """
    dens, edges = np.histogram(x, bins=bins, density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])
    m = dens > 0
    xc = xc[m]
    y = -np.log(dens[m])

    X = np.column_stack([np.ones_like(xc), xc**2])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w0, w2 = float(w[0]), float(w[1])
    return (w0, w2), (xc, y)


def E_gen(x, w0, w2):
    return w0 + w2 * x**2


def hist_prob(E, edges):
    c, _ = np.histogram(E, bins=edges, density=False)
    p = c.astype(float)
    s = p.sum()
    if s <= 0:
        raise ValueError("Empty histogram on energy axis.")
    return p / s


# --------------------- Trace-form MaxEnt (fixed g') --------------------------

def _solve_lambda0_for_norm(inv_gprime, E, lam1, y_min, y_max, tol=1e-12):
    """
    For a fixed λ1, solve for λ0 s.t. sum_i p_i = 1, where
      p_i = inv_gprime(λ0 + λ1 E_i).
    Uses bisection on λ0 with bracket expansion.
    """

    def S(lam0):
        y = lam0 + lam1 * E
        y = np.clip(y, y_min, y_max)
        return float(np.sum(inv_gprime(y)))

    # Start bracket around 0; expand until it brackets 1
    a, b = -5.0, 5.0
    Sa, Sb = S(a), S(b)

    # Want Sa <= 1 <= Sb or Sb <= 1 <= Sa
    for _ in range(80):
        if (Sa - 1.0) * (Sb - 1.0) <= 0:
            break
        a *= 1.4
        b *= 1.4
        Sa, Sb = S(a), S(b)

    # If still no bracket, just pick closer end (should be rare with clipping)
    if (Sa - 1.0) * (Sb - 1.0) > 0:
        return a if abs(Sa - 1.0) < abs(Sb - 1.0) else b

    # Bisection
    for _ in range(120):
        m = 0.5 * (a + b)
        Sm = S(m)
        if abs(Sm - 1.0) < tol:
            return m
        if (Sa - 1.0) * (Sm - 1.0) <= 0:
            b, Sb = m, Sm
        else:
            a, Sa = m, Sm
    return 0.5 * (a + b)


def traceform_maxent_mean_energy(inv_gprime, E, U_target, y_min, y_max):
    """
    Solve trace-form MaxEnt with constraints:
      sum p = 1,  sum p E = U_target
    using the KKT form:
      p_i = (g')^{-1}(λ0 + λ1 E_i).
    Two-level bisection:
      - for each λ1, solve λ0 by norm constraint
      - then bisection on λ1 to match U_target
    """
    E = np.asarray(E, float)

    def mean_for_lam1(lam1):
        lam0 = _solve_lambda0_for_norm(inv_gprime, E, lam1, y_min, y_max)
        y = lam0 + lam1 * E
        y = np.clip(y, y_min, y_max)
        p = inv_gprime(y)
        # norm is already enforced numerically, but guard:
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / s
        U = float(p @ E)
        return U, p

    # Find a bracket for λ1 by expanding until U crosses target
    lam1_lo, lam1_hi = -2.0, 2.0
    U_lo, _ = mean_for_lam1(lam1_lo)
    U_hi, _ = mean_for_lam1(lam1_hi)

    # Expand bracket (direction depends on monotonicity; we just expand both ends)
    for _ in range(80):
        if (U_lo - U_target) * (U_hi - U_target) <= 0:
            break
        lam1_lo *= 1.3
        lam1_hi *= 1.3
        U_lo, _ = mean_for_lam1(lam1_lo)
        U_hi, _ = mean_for_lam1(lam1_hi)

    # If still no bracket (pathological with heavy clipping), pick best endpoint
    if (U_lo - U_target) * (U_hi - U_target) > 0:
        if abs(U_lo - U_target) < abs(U_hi - U_target):
            _, p_best = mean_for_lam1(lam1_lo)
        else:
            _, p_best = mean_for_lam1(lam1_hi)
        return p_best

    # Bisection on λ1
    for _ in range(120):
        lam1_mid = 0.5 * (lam1_lo + lam1_hi)
        U_mid, p_mid = mean_for_lam1(lam1_mid)
        if abs(U_mid - U_target) < 1e-12:
            return p_mid
        if (U_lo - U_target) * (U_mid - U_target) <= 0:
            lam1_hi, U_hi = lam1_mid, U_mid
        else:
            lam1_lo, U_lo = lam1_mid, U_mid

    _, p_mid = mean_for_lam1(0.5 * (lam1_lo + lam1_hi))
    return p_mid


def shannon_exp_prior_on_E(E, U_target):
    """
    Shannon MaxEnt on energy axis with mean-energy constraint:
      p(E) ∝ exp(-λ E)
    Choose λ by bisection to match mean energy.
    """
    E = np.asarray(E, float)

    def mean_E(lam):
        a = -lam * E
        a = a - np.max(a)
        w = np.exp(a)
        w /= w.sum()
        return float(w @ E), w

    lam_lo, lam_hi = -50.0, 50.0
    m_lo, _ = mean_E(lam_lo)
    m_hi, _ = mean_E(lam_hi)

    for _ in range(60):
        if (m_lo - U_target) * (m_hi - U_target) <= 0:
            break
        lam_lo *= 1.5
        lam_hi *= 1.5
        m_lo, _ = mean_E(lam_lo)
        m_hi, _ = mean_E(lam_hi)

    if (m_lo - U_target) * (m_hi - U_target) > 0:
        lam = lam_lo if abs(m_lo - U_target) < abs(m_hi - U_target) else lam_hi
        _, p = mean_E(lam)
        return lam, p

    for _ in range(140):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        m_mid, _ = mean_E(lam_mid)
        if (m_lo - U_target) * (m_mid - U_target) <= 0:
            lam_hi, m_hi = lam_mid, m_mid
        else:
            lam_lo, m_lo = lam_mid, m_mid

    lam = 0.5 * (lam_lo + lam_hi)
    _, p = mean_E(lam)
    return lam, p


# --------------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--T_ref", type=float, default=1.0)
    ap.add_argument("--T_list", type=str, default="0.5,0.6667,1.0,1.5,2.0")
    ap.add_argument("--xhist_bins", type=int, default=500)
    ap.add_argument("--binsE", type=int, default=160)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--anchor_scale", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="single_well_protocol.png")
    ap.add_argument("--out_w2", type=str, default="single_well_w2_vs_beta.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    T_list = [float(s.strip()) for s in args.T_list.split(",") if s.strip() != ""]
    beta_list = [1.0 / T for T in T_list]

    # ------------------- Fit E_gen(x) at T_ref (train) -------------------
    x_ref = sample_harmonic(args.T_ref, args.N, rng, k=args.k)
    (w0_ref, w2_ref), (xc_ref, y_ref) = fit_quadratic_energy_from_hist(x_ref, bins=args.xhist_bins)

    # theory: for p(x) ∝ exp(-β k x^2 / 2), -log p = const + (β k /2) x^2
    beta_ref = 1.0 / args.T_ref
    w2_theory_ref = 0.5 * beta_ref * args.k

    # ------------------- Fixed energy grid from training only -------------
    E_ref = E_gen(x_ref, w0_ref, w2_ref)
    lo, hi = np.percentile(E_ref, [0.2, 99.8])
    edgesE = np.linspace(lo, hi, args.binsE + 1)
    E_centers = 0.5 * (edgesE[:-1] + edgesE[1:])

    pE_ref = hist_prob(E_ref, edgesE)
    U_ref = float(pE_ref @ E_centers)

    # ------------------- Stage-I: learn g'(p) from pE_ref -----------------
    F, _, sig_b, sig_n = make_energy_features_multiscale(E_centers, K=args.K)
    theta, gprime, inv_gprime, y_min, y_max = learn_traceform_from_pE(
        pE_ref, F, anchor_scale=args.anchor_scale
    )

    # Build generator G for entropy evaluation
    p_max = float(np.max(pE_ref))
    Gfun = build_G_from_gprime(gprime, p_max=max(p_max, 1e-6))

    # ------------------- Protocol: for each T, predict p_E^*(T) -----------
    U_emp_list = []
    pE_emp_list = []
    pE_trace_list = []
    S_gen_list = []

    # Also fit w2 at each beta for diagnostic linearity plot
    w2_fit_list = []

    for T in T_list:
        xT = sample_harmonic(T, args.N, rng, k=args.k)

        # diagnostic fit at this T (not used for the protocol mapping)
        (_, w2_T), _ = fit_quadratic_energy_from_hist(xT, bins=args.xhist_bins)
        w2_fit_list.append(float(w2_T))

        # empirical energy histogram on *fixed* bins using train-fitted E_gen
        ET = E_gen(xT, w0_ref, w2_ref)
        pE_emp = hist_prob(ET, edgesE)
        U_emp = float(pE_emp @ E_centers)

        # trace-form MaxEnt prediction with fixed functional, updated mean energy
        pE_star = traceform_maxent_mean_energy(inv_gprime, E_centers, U_emp, y_min, y_max)

        # entropy on energy axis from trace-form generator
        S_gen = float(np.sum(Gfun(pE_star)))

        U_emp_list.append(U_emp)
        pE_emp_list.append(pE_emp)
        pE_trace_list.append(pE_star)
        S_gen_list.append(S_gen)

    U_emp_arr = np.array(U_emp_list, float)
    S_gen_arr = np.array(S_gen_list, float)
    T_arr = np.array(T_list, float)

    # Operational temperature from protocol derivative (central differences)
    # (endpoints have one-sided derivative; we plot only interior for cleanliness)
    if len(T_arr) >= 3:
        dS = (S_gen_arr[2:] - S_gen_arr[:-2])
        dU = (U_emp_arr[2:] - U_emp_arr[:-2])
        with np.errstate(divide="ignore", invalid="ignore"):
            T_gen_mid = 1.0 / (dS / dU)
        T_mid = T_arr[1:-1]
    else:
        T_gen_mid = np.array([], float)
        T_mid = np.array([], float)

    # Choose one temperature to show energy-axis panel (matches your earlier: T=0.5)
    # Pick the closest available.
    T_show = 0.5
    j_show = int(np.argmin(np.abs(T_arr - T_show)))
    pE_emp_show = pE_emp_list[j_show]
    pE_trace_show = pE_trace_list[j_show]
    U_show = float(pE_emp_show @ E_centers)
    lam_sh, pE_sh_show = shannon_exp_prior_on_E(E_centers, U_show)

    js_trace = js_divergence(pE_trace_show, pE_emp_show)
    js_sh = js_divergence(pE_sh_show, pE_emp_show)

    # ------------------- Plot: 2x2 protocol panel -------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (Top-left) generator calibration at T_ref
    ax = axs[0, 0]
    ax.scatter(xc_ref, y_ref, s=10, alpha=0.70, color="#1f77b4", label=r"$-\log \hat p(x)$ (hist)")
    xg = np.linspace(np.percentile(x_ref, 0.2), np.percentile(x_ref, 99.8), 800)
    ax.plot(xg, E_gen(xg, w0_ref, w2_ref), lw=2.5, color="#ff7f0e", label=r"$E_{\rm gen}(x)$ (fit)")
    ax.set_title(rf"Harmonic well: generator calibration at $T_{{\rm ref}}={args.T_ref:.1f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$-\log \hat p(x)$ / $E_{\rm gen}(x)$")
    ax.legend(frameon=True, loc="lower left")
    ax.text(0.03, 0.97,
            rf"$w_2$ fit = {w2_ref:.4f}" "\n" rf"$w_2$ theory = {w2_theory_ref:.4f}",
            transform=ax.transAxes, va="top")

    # (Top-right) energy-axis predictions at T_show
    ax = axs[0, 1]
    width = edgesE[1] - edgesE[0]
    ax.bar(E_centers, pE_emp_show, width=0.90 * width, alpha=0.25, color="#1f77b4",
           label="empirical $p_E$ (fixed bins)")
    ax.plot(E_centers, pE_trace_show, lw=2.4, color="#1f77b4",
            label=rf"trace-form $p_E^\star$ (T={T_arr[j_show]:.2f})")
    ax.plot(E_centers, pE_sh_show, lw=2.0, ls="--", color="#ff7f0e",
            label=rf"Shannon $e^{{-\lambda E}}$ (T={T_arr[j_show]:.2f})")
    ax.set_title("Energy axis: empirical vs fixed-functional predictions")
    ax.set_xlabel(r"energy $E$")
    ax.set_ylabel("probability per bin")
    ax.legend(frameon=True)
    ax.text(0.03, 0.97,
            rf"JS(trace)={js_trace:.4f}" "\n" rf"JS(Sh)={js_sh:.4f}",
            transform=ax.transAxes, va="top")

    # (Bottom-left) protocol curve S_gen(U)
    ax = axs[1, 0]
    ax.plot(U_emp_arr, S_gen_arr, marker="o", lw=2.0)
    ax.set_title(r"Protocol curve on energy state space: $S_{\rm gen}(U)$")
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$S_{\rm gen}=\sum_i G(p_i^\star)$")

    # (Bottom-right) operational temperature check
    ax = axs[1, 1]
    # ideal line
    ax.plot(T_arr, T_arr, ls="--", lw=2.0, color="#ff7f0e", label=r"ideal $T_{\rm gen}=T$")
    # computed points (interior)
    if len(T_gen_mid) > 0:
        m = np.isfinite(T_gen_mid)
        ax.plot(T_mid[m], T_gen_mid[m], marker="o", lw=2.0, color="#1f77b4",
                label=r"$T_{\rm gen}=1/(dS/dU)$")
    ax.set_title("Operational temperature check along bath protocol")
    ax.set_xlabel(r"bath temperature $T$")
    ax.set_ylabel(r"$T_{\rm gen}$")
    ax.legend(frameon=True, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=250)
    plt.close(fig)

    # ------------------- Plot: w2 vs beta (diagnostic) ---------------------
    beta_arr = np.array(beta_list, float)
    w2_fit_arr = np.array(w2_fit_list, float)
    w2_theory_arr = 0.5 * beta_arr * args.k

    fig2, axs2 = plt.subplots(1, 2, figsize=(13, 4.5))

    # (Left) representative calibration panel at T_ref (cleaner version)
    ax = axs2[0]
    ax.scatter(xc_ref, y_ref, s=10, alpha=0.65, color="#1f77b4", label=r"$-\log \hat p(x)$")
    ax.plot(xg, E_gen(xg, w0_ref, w2_ref), lw=2.5, color="#ff7f0e", label=r"$E_{\rm gen}(x)$ fit")
    ax.set_title(rf"Harmonic well calibration at $T={args.T_ref:.1f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$-\log \hat p(x)$ / $E_{\rm gen}(x)$")
    ax.legend(frameon=True, loc="lower left")

    # (Right) w2 vs beta (this is the “beautiful linear” one)
    ax = axs2[1]
    ax.plot(beta_arr, w2_fit_arr, marker="o", lw=2.0, label=r"fit $w_2$")
    ax.plot(beta_arr, w2_theory_arr, lw=2.0, label=r"theory $\frac{1}{2}\beta k$")
    ax.set_title(r"Generator quadratic coefficient vs $\beta$")
    ax.set_xlabel(r"$\beta=1/T$")
    ax.set_ylabel(r"quadratic coefficient $w_2$")
    ax.legend(frameon=True)

    fig2.tight_layout()
    fig2.savefig(args.out_w2, dpi=250)
    plt.close(fig2)

    print(f"Saved: {args.out}")
    print(f"Saved: {args.out_w2}")
    print(f"Train (T_ref={args.T_ref}): w2_fit={w2_ref:.6f}, w2_theory={w2_theory_ref:.6f}")
    print(f"Energy-basis widths: sig_b={sig_b:.6g}, sig_n={sig_n:.6g}")
    print(f"Protocol U range: [{U_emp_arr.min():.6g}, {U_emp_arr.max():.6g}]")


if __name__ == "__main__":
    main()
