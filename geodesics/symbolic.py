"""
Symbolic tools: Christoffel symbols and LaTeX pretty-printing.

These are for looking at the geometry in a notebook. The numerical
integrator in ``geodesics.integrate`` builds its own lambdified RHS and
does not go through these functions.
"""

import sympy as sp
from IPython.display import Math


def compute_christoffel(metric, simplify=True):
    """
    Gamma^s_{mu nu} = 1/2 g^{s rho} (d_mu g_{nu rho} + d_nu g_{mu rho} - d_rho g_{mu nu})

    Returns
    -------
    Gamma   : nested list Gamma[s][mu][nu] (symmetric in mu, nu)
    nonzero : dict {(s, mu, nu): expr} with mu <= nu, only non-zero entries

    simplify=True calls sp.simplify (slow but pretty); False uses sp.cancel,
    which is much faster on messy metrics such as Majumdar-Papapetrou.
    """
    n = metric.dim
    coords, g, g_inv = metric.coords, metric.g, metric.g_inv

    Gamma = [[[sp.S.Zero]*n for _ in range(n)] for _ in range(n)]
    nonzero = {}
    for s in range(n):
        for m in range(n):
            for nu in range(m, n):
                val = sum(
                    sp.Rational(1, 2) * g_inv[s, rho] * (
                        sp.diff(g[nu, rho], coords[m])
                        + sp.diff(g[m, rho], coords[nu])
                        - sp.diff(g[m, nu], coords[rho]))
                    for rho in range(n))
                val = sp.simplify(val) if simplify else sp.cancel(val)
                Gamma[s][m][nu] = Gamma[s][nu][m] = val
                if val != 0:
                    nonzero[(s, m, nu)] = val
    return Gamma, nonzero


def print_metric(metric):
    """Line element ds^2 = ... as a LaTeX Math object."""
    terms = []
    for i in range(metric.dim):
        for j in range(i, metric.dim):
            gij = metric.g[i, j]
            if gij == 0:
                continue
            coeff = sp.latex(gij)
            if gij.is_Add:
                coeff = f"\\left({coeff}\\right)"
            di, dj = sp.latex(metric.coords[i]), sp.latex(metric.coords[j])
            terms.append(f"{coeff} \\, d{di}^2" if i == j
                         else f"2 {coeff} \\, d{di} \\, d{dj}")
    return Math("ds^2 = " + " + ".join(terms))


def print_christoffel(metric, nonzero):
    """Non-zero Christoffel symbols (upper triangle only) as LaTeX."""
    lines = []
    for (s, m, nu), expr in nonzero.items():
        cs, cm, cn = (sp.latex(metric.coords[k]) for k in (s, m, nu))
        lines.append(f"\\Gamma^{{{cs}}}_{{{cm}{cn}}} = {sp.latex(expr)}")
    return Math(" \\\\ ".join(lines))


def print_geodesic_eqs(metric, nonzero):
    """Geodesic equations  x''^s = -Gamma^s_{mu nu} x'^mu x'^nu  as LaTeX."""
    n = metric.dim
    v = [sp.Symbol(f'\\dot{{{sp.latex(c)}}}') for c in metric.coords]
    lines = []
    for s in range(n):
        acc = sp.S.Zero
        for (si, m, nu), expr in nonzero.items():
            if si == s:
                acc -= expr * v[m] * v[nu]
                if m != nu:
                    acc -= expr * v[nu] * v[m]
        lines.append(f"\\ddot{{{sp.latex(metric.coords[s])}}} = {sp.latex(acc)}")
    return Math(" \\\\ ".join(lines))
