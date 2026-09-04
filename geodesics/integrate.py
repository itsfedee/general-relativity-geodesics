"""
Numerical integration of the geodesic equation:

    x'^mu = v^mu,   v'^s = -Gamma^s_{mu nu} v^mu v^nu

The acceleration is built symbolically once per metric, lambdified into a
single numpy function (with common-subexpression elimination) and cached
on the metric object. After the first call, each RHS evaluation is one
compiled function call: no sympy, no dict loops, no matrix inversion.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


# ============================================================================
# Right-hand side
# ============================================================================

def geodesic_rhs(metric):
    """
    Build rhs(tau, y) for ``solve_ivp``, with y = [x^0..x^{n-1}, v^0..v^{n-1}].
    Metric parameters (M, a, ...) are substituted with their numerical values.
    """
    n = metric.dim
    coords = list(metric.coords)
    g = metric.g.subs(metric.params)
    ginv = metric.g_inv.subs(metric.params)
    vs = list(sp.symbols(f'vv0:{n}'))

    dg = [[[sp.diff(g[i, j], coords[k]) for j in range(n)] for i in range(n)]
          for k in range(n)]

    accel = []
    for s in range(n):
        expr = sp.Integer(0)
        for mu in range(n):
            for nu in range(n):
                gamma = sum(ginv[s, rho] * (dg[mu][rho][nu] + dg[nu][rho][mu] - dg[rho][mu][nu])
                            for rho in range(n))
                expr += -sp.Rational(1, 2) * gamma * vs[mu] * vs[nu]
        accel.append(expr)

    f_accel = sp.lambdify(coords + vs, accel, 'numpy', cse=True)

    def rhs(tau, y):
        a = np.asarray(f_accel(*y[:n], *y[n:]), dtype=float)
        return np.concatenate([y[n:], a])

    return rhs


def get_rhs(metric):
    """Cached geodesic_rhs for this metric instance."""
    if getattr(metric, '_rhs_cache', None) is None:
        metric._rhs_cache = geodesic_rhs(metric)
    return metric._rhs_cache


# ============================================================================
# Events
# ============================================================================

def cross_equator(tau, y):
    """Non-terminal event: theta = pi/2 (both crossing directions)."""
    return y[2] - np.pi/2
cross_equator.terminal = False
cross_equator.direction = 0


def escape_event(r_max=250.0):
    """Terminal event: r grows past r_max (the ray escaped to infinity)."""
    def escaping(tau, y):
        return y[1] - r_max
    escaping.terminal = True
    escaping.direction = 1
    return escaping


escaping = escape_event(250.0)   # module-level default, picklable for workers


# ============================================================================
# Solver
# ============================================================================

def solve_geodesic(metric, x0, v0, tau_span, rhs=None, extra_events=None,
                   rtol=1e-10, atol=1e-12, dense_output=True, **kwargs):
    """
    Integrate a geodesic with DOP853.

    Parameters
    ----------
    metric        : Metric instance
    x0, v0        : initial position / velocity in the metric's coordinates
                    (use ``metric.from_cartesian`` to build them)
    tau_span      : (tau_0, tau_1) affine-parameter interval
    rhs           : precomputed RHS (optional; cached on the metric otherwise)
    extra_events  : list of solve_ivp event functions appended after the
                    metric's own ``stop_events()`` (horizon crossing for black
                    holes). Event index 0 is the stop event, extras follow.
    kwargs        : forwarded to solve_ivp (t_eval, max_step, ...)

    Returns the solve_ivp result. If the geodesic hits the horizon and dense
    output is available, the exact stopping point is appended to sol.t / sol.y.
    """
    if rhs is None:
        rhs = get_rhs(metric)
    y0 = np.concatenate([x0, v0])

    events = list(metric.stop_events())
    n_stop = len(events)
    if extra_events:
        events.extend(extra_events)

    opts = dict(method='DOP853', rtol=rtol, atol=atol,
                dense_output=dense_output, events=events)
    opts.update(kwargs)
    sol = solve_ivp(rhs, tau_span, y0, **opts)

    if n_stop and sol.t_events[0].size > 0 and sol.sol is not None:
        tau_stop = sol.t_events[0][0]
        sol.y = np.column_stack([sol.y, sol.sol(tau_stop).reshape(-1, 1)])
        sol.t = np.append(sol.t, tau_stop)

    return sol


def resample(sol, n=1000, power=1.0):
    """
    Resample a dense solution on n points, tau = tau_end * u**power.
    power < 1 concentrates points near the end (useful for infall animations).
    Modifies and returns sol.
    """
    assert sol.sol is not None, "need dense_output=True"
    t_new = sol.t[-1] * np.linspace(0, 1, n)**power
    sol.t = t_new
    sol.y = sol.sol(t_new)
    return sol
