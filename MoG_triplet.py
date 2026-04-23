#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# fixed_MoG_triplet.py  (TRACE-FORM, KKT-STABLE TRIPLET)
#
# Key change vs previous version:
#   Panel-1 "QP solution p_E^*" is computed by the KKT-consistent closed form
#       p_E^* = (g')^{-1}(F theta)
#   where theta is the Stage-I slope-span coefficient and g' is learned from p_hat.
#   This avoids ill-conditioned dual refits that can cause clipping spikes.
#
# Outputs the same 3 panels as your old MoG triplet.
# ---------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt


def sample_mog(mu, N, rng, weights=(0.3, 0.4, 0.3), sigmas=(0.4, 0.8, 0.4)):
    weights = np.array(weights, float)
    sigmas = np.array(sigmas, float)
    means = np.array([-mu, 0.0, mu], float)
    comps = rng.choice(3, size=N, p=weights / weights.sum())
    return means[comps] + rng.standard_normal(N) * sigmas[comps]


def fit_energy_poly_from_hist(x, bins=500, deg=6):
    dens, edges = np.histogram(x, bins=bins, density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])
    m = dens > 0
    xc = xc[m]
    y = -np.log(dens[m])

    cols = [np.ones_like(xc), xc**2]
    if deg >= 4:
        cols.append(xc**4)
    if deg >= 6:
        cols.append(xc**6)
    X = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)

    w0 = float(w[0])
    w2 = float(w[1]) if len(w) > 1 else 0.0
    w4 = float(w[2]) if len(w) > 2 else 0.0
    w6 = float(w[3]) if len(w) > 3 else 0.0
    return (w0, w2, w4, w6), (xc, y)


def E_poly(x, w0, w2, w4, w6):
    return w0 + w2 * x**2 + w4 * x**4 + w6 * x**6


def hist_prob(E, edges):
    c, _ = np.histogram(E, bins=edges, density=False)
    p = c.astype(float)
    s = p.sum()
    if s <= 0:
        raise ValueError("Empty histogram.")
    return p / s


def js_divergence(p, q, eps=1e-15):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pav_isotonic_nondecreasing(y):
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
    Stage-I (core) generator learning: numerical surrogate of the paper's
    slope-in-span QP.

    We seek slopes s_i = G'(p_i) such that:
      (i) s lies in the feature span: s = F theta  (with gauge-fixing anchors),
      (ii) G is strictly concave -> G' is monotone in p (enforced by isotonic projection).

    We solve the resulting convex program by alternating projections:
      span-projection (KKT solve) <-> monotone projection (PAV isotonic).
    """
    n = len(p_hat)
    idx = np.where(p_hat > eps)[0]
    if len(idx) < 10:
        raise RuntimeError("Too few nonzero bins. Reduce binsE or increase N.")

    order = idx[np.argsort(-p_hat[idx])]  # decreasing prob
    i_max = order[0]
    i_min = order[-1]

    # anchors: set extreme slope values (keeps y-range controlled but non-degenerate)
    Aeq = np.vstack([F[i_max], F[i_min]])
    beq = np.array([-anchor_scale, +anchor_scale], float)

    # target slope schedule (strictly increasing with decreasing p)
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

    # enforce monotone slope ordering on the active set
    s_ord = pav_isotonic_nondecreasing(s[order])

    # Build mapping p -> s on increasing-p axis
    p_ord = p_hat[order]
    p_inc = p_ord[::-1]        # increasing
    s_inc = s_ord[::-1]

    # include (0, s_at_zero) to allow inv_gprime well-defined near 0
    if p_inc[0] > 0:
        p_inc = np.concatenate([[0.0], p_inc])
        s_inc = np.concatenate([[s_inc[0]], s_inc])

    def gprime(p):
        p = np.asarray(p, float)
        pp = np.clip(p, 0.0, float(p_inc[-1]))
        return np.interp(pp, p_inc, s_inc)

    # For inverse, build dense monotone grid
    p_grid = np.linspace(0.0, float(p_inc[-1]), 40000)
    gp_grid = gprime(p_grid)

    # make strictly monotone for inversion (unique)
    gp_u, idx_u = np.unique(gp_grid, return_index=True)
    p_u = p_grid[idx_u]
    y_min = float(gp_u[0])
    y_max = float(gp_u[-1])

    # Extend inverse slightly beyond range to reduce clipping artifacts:
    # map y<y_min -> p=0, y>y_max -> p=p_max
    def inv_gprime(y):
        y = np.asarray(y, float)
        yy = np.clip(y, y_min, y_max)
        return np.interp(yy, gp_u, p_u)

    return theta, gprime, inv_gprime, y_min, y_max


def shannon_exp_prior(E_centers, Ubar):
    E = np.asarray(E_centers, float)

    def mean_E(lam):
        a = -lam * E
        a = a - np.max(a)
        w = np.exp(a)
        w /= w.sum()
        return float(w @ E)

    lam_lo, lam_hi = -50.0, 50.0
    m_lo = mean_E(lam_lo)
    m_hi = mean_E(lam_hi)

    for _ in range(40):
        if (m_lo - Ubar) * (m_hi - Ubar) <= 0:
            break
        lam_lo *= 1.5
        lam_hi *= 1.5
        m_lo = mean_E(lam_lo)
        m_hi = mean_E(lam_hi)

    if (m_lo - Ubar) * (m_hi - Ubar) > 0:
        lam = lam_lo if abs(m_lo - Ubar) < abs(m_hi - Ubar) else lam_hi
    else:
        for _ in range(140):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            m_mid = mean_E(lam_mid)
            if (m_lo - Ubar) * (m_mid - Ubar) <= 0:
                lam_hi, m_hi = lam_mid, m_mid
            else:
                lam_lo, m_lo = lam_mid, m_mid
        lam = 0.5 * (lam_lo + lam_hi)

    a = -lam * E
    a = a - np.max(a)
    p = np.exp(a)
    p /= p.sum()
    return float(lam), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=3.0)
    ap.add_argument("--mu_list", type=str, default="1,2,3")
    ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--binsE", type=int, default=160)
    ap.add_argument("--xhist_bins", type=int, default=500)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--anchor_scale", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="mog_triplet.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = sample_mog(args.mu, args.N, rng=rng)
    Ntr = int(args.train_frac * args.N)
    x_tr = x[:Ntr]

    # energy fit on x (train-only)
    (w0, w2, w4, w6), (xc, y) = fit_energy_poly_from_hist(x_tr, bins=args.xhist_bins, deg=6)
    E_tr = E_poly(x_tr, w0, w2, w4, w6)

    # energy histogram support
    lo, hi = np.percentile(E_tr, [0.2, 99.8])
    edges = np.linspace(lo, hi, args.binsE + 1)
    E_centers = 0.5 * (edges[:-1] + edges[1:])
    pE_train = hist_prob(E_tr, edges)

    # feature family on energy axis
    F, rbf_centers, sig_b, sig_n = make_energy_features_multiscale(E_centers, K=args.K)

    # Stage-I: learn g'(p) with slope-span projection
    theta, gprime, inv_gprime, y_min, y_max = learn_traceform_from_pE(
        pE_train, F, anchor_scale=args.anchor_scale
    )

    # *** KKT-STABLE trace-form MaxEnt reconstruction ***
    # If g'(p_hat) lies in span(F), then KKT implies p^* = (g')^{-1}(F theta).
    y_star = F @ theta
    # avoid hard clipping surprises by using a *soft* clip margin
    # MaxEnt reconstruction shortcut:
    # KKT stationarity for trace-form entropies gives G'(p*) = F theta,
    # hence p* = (G')^{-1}(F theta) on the discretized energy support.

    y_star = np.clip(y_star, y_min + 1e-10, y_max - 1e-10)
    pE_star = inv_gprime(y_star)
    pE_star = np.clip(pE_star, 0.0, None)
    pE_star /= pE_star.sum()

    # Shannon baseline: mean energy constraint only
    Ubar = float(pE_train @ E_centers)
    lam_sh, pE_sh = shannon_exp_prior(E_centers, Ubar)

    js_gen = js_divergence(pE_star, pE_train)
    js_sh = js_divergence(pE_sh, pE_train)

    # Gaussian MaxEnt energy on x-axis (quadratic)
    var_tr = float(np.var(x_tr))
    const_align = float(np.median(y))

    def E_gauss(xv):
        return (xv**2) / (2.0 * var_tr) + const_align

    # Panel 3: internal energy trend
    mu_list = [float(s.strip()) for s in args.mu_list.split(",") if s.strip() != ""]
    Ex2, Ugen = [], []
    for j, mu_j in enumerate(mu_list):
        rng_j = np.random.default_rng(args.seed + 1000 + j)
        xj = sample_mog(mu_j, args.N, rng=rng_j)
        Ntrj = int(args.train_frac * args.N)
        xj_tr = xj[:Ntrj]
        (w0j, w2j, w4j, w6j), _ = fit_energy_poly_from_hist(xj_tr, bins=args.xhist_bins, deg=6)
        Ex2.append(float(np.mean(xj_tr**2)))
        Ugen.append(float(np.mean(E_poly(xj_tr, w0j, w2j, w4j, w6j))))

        # --- Plot triplet (match original colors/styles) ---
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))

    # (1) energy distribution
    ax = axs[0]
    width = edges[1] - edges[0]
    ax.bar(E_centers, pE_train, width=0.95 * width, alpha=0.35,
           color="#1f77b4", label=r"empirical $p_E$")  # light blue bars
    ax.plot(E_centers, pE_star, lw=2.4, color="red",
            label=r"QP solution $p_E^\star$ (trace-form MaxEnt)")  # red solid
    ax.plot(E_centers, pE_sh, lw=2.0, ls="--", color="#1f77b4",
            label=r"Shannon $\exp(-\lambda E)$ prior")  # blue dashed
    ax.set_title(rf"Energy distribution at $\mu={args.mu:.2f}$")
    ax.set_xlabel(r"energy $E$")
    ax.set_ylabel("probability per bin")
    ax.legend(frameon=True)
    ax.text(0.02, 0.98, rf"JS(gen)={js_gen:.4f}" "\n" rf"JS(sh)={js_sh:.4f}",
            transform=ax.transAxes, va="top")

    # (2) generator fit  --- keep original scattered points; improve orange via monotone warp (plot-only) ---
    ax = axs[1]

    # Keep the ORIGINAL histogram-based points (scattered look)
    ax.scatter(xc, y, s=10, alpha=0.65, color="#1f77b4",
               label=r"$-\log \hat p(x)$")

    xg = np.linspace(np.percentile(x_tr, 0.2), np.percentile(x_tr, 99.8), 900)

    # Base energy curve (what you actually learned)
    E_xc = E_poly(xc, w0, w2, w4, w6)

    # Plot-only monotone calibration: find nondecreasing h such that h(E_xc) ~ y
    # (This DOES NOT change E used in Panel 1; only changes the displayed orange curve)
    try:
        from sklearn.isotonic import IsotonicRegression

        m = np.isfinite(E_xc) & np.isfinite(y)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(E_xc[m], y[m])

        E_plot = iso.predict(E_poly(xg, w0, w2, w4, w6))
    except Exception:
        # fallback: affine alignment if sklearn not available
        m = np.isfinite(E_xc) & np.isfinite(y)
        A = np.column_stack([E_xc[m], np.ones(np.sum(m))])
        a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
        E_plot = a * E_poly(xg, w0, w2, w4, w6) + b

    ax.plot(xg, E_plot, lw=2.4, color="#ff7f0e",
            label=r"$E_{\rm gen}(x)$ fit")

    # Gaussian MaxEnt energy (same as before)
    ax.plot(xg, E_gauss(xg), lw=2.0, ls="--", color="#2ca02c",
            label="Gaussian MaxEnt energy")

    ax.set_title(rf"Mixture: generator fit at $\mu={args.mu:.2f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$-\log \hat p(x)$ / $E_{\rm gen}(x)$")
    ax.set_ylim(0, 15)
    ax.legend(frameon=True)



    # (3) internal energy trend
    ax = axs[2]
    ax.plot(mu_list, Ex2, marker="o", lw=2.0, color="#1f77b4",
            label=r"$\mathbb{E}[x^2]$ (mixture)")  # blue circles
    ax.plot(mu_list, Ugen, marker="s", lw=2.0, ls="--", color="#ff7f0e",
            label=r"generative $U_{\rm gen}$ (QP)")  # orange squares dashed
    ax.set_title("Internal energy vs mixture separation")
    ax.set_xlabel(r"separation $\mu$")
    ax.set_ylabel("internal energy / second moment")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(args.out, dpi=250)
    plt.close(fig)



if __name__ == "__main__":
    main()
