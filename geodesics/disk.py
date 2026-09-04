"""
Thin accretion disk physics: Novikov-Thorne temperature profile and
Doppler / gravitational redshift factors for gas on circular equatorial orbits.
"""

import numpy as np
import sympy as sp
from scipy.interpolate import interp1d


def disk_flux(metric, r_array, r_isco=None):
    """
    Numerical Novikov-Thorne flux for any stationary, axisymmetric metric
    with coordinates (t, r, theta, phi).

        F(r) = -Omega_,r / (E - Omega L)^2 * int_{r_isco}^{r} (E - Omega L) L_,r dr

    Returns the normalised temperature T = F^{1/4} / max, on r_array.
    """
    if r_isco is None:
        r_isco = metric.r_isco
    g_expr = metric.g.subs(metric.params)
    coords = metric.coords
    r_sym = coords[1]
    eq = {coords[2]: sp.pi/2}

    comps = {}
    for name, (i, j) in {'tt': (0, 0), 'pp': (3, 3), 'tp': (0, 3)}.items():
        e = g_expr[i, j].subs(eq)
        comps[name] = sp.lambdify(r_sym, e, 'numpy')
        comps['d' + name] = sp.lambdify(r_sym, sp.diff(e, r_sym), 'numpy')

    def circular_quantities(r):
        gtt, gpp, gtp = comps['tt'](r), comps['pp'](r), comps['tp'](r)
        dgtt, dgpp, dgtp = comps['dtt'](r), comps['dpp'](r), comps['dtp'](r)
        # circular-orbit condition: dgtt + 2 Omega dgtp + Omega^2 dgpp = 0 (prograde root)
        A, B, C = dgpp, 2 * dgtp, dgtt
        Omega = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
        denom = np.sqrt(-(gtt + 2*gtp*Omega + gpp*Omega**2))
        E = -(gtt + gtp*Omega) / denom
        L = (gtp + gpp*Omega) / denom
        return E, L, Omega

    E = np.zeros_like(r_array)
    L = np.zeros_like(r_array)
    Om = np.zeros_like(r_array)
    for i, r in enumerate(r_array):
        E[i], L[i], Om[i] = circular_quantities(r)

    dOm = np.gradient(Om, r_array)
    dL = np.gradient(L, r_array)
    integrand = (E - Om * L) * dL

    flux = np.zeros_like(r_array)
    for i in range(len(r_array)):
        if r_array[i] <= r_isco:
            continue
        mask = (r_array >= r_isco) & (r_array <= r_array[i])
        flux[i] = -dOm[i] / (E[i] - Om[i]*L[i])**2 * np.trapezoid(integrand[mask], r_array[mask])

    flux = np.maximum(flux, 0)
    T = flux**0.25
    return T / T.max()


def temperature_interpolator(metric, r_max=30.0, n_pts=3000):
    """
    Precompute T(r) on a grid and return an interpolant that is 0 outside
    [r_isco, r_max] (so a ray missing the disk maps to black).
    """
    r_array = np.linspace(metric.r_isco + 0.01, r_max, n_pts)
    T = disk_flux(metric, r_array, metric.r_isco)
    return interp1d(r_array, T, fill_value=0.0, bounds_error=False)


def doppler_factor(r_hit, phi_hit, M):
    """
    Doppler factor g = 1 / (gamma (1 + v_los)) for gas on a circular
    Schwarzschild orbit, observer along -x. Intensity scales as g**4.
    """
    Omega = np.sqrt(M / r_hit**3)
    ut = 1 / np.sqrt(1 - 3*M/r_hit)
    v_orb = r_hit * Omega / (ut * (1 - 2*M/r_hit))
    v_los = v_orb * np.sin(phi_hit)
    gamma = 1 / np.sqrt(1 - v_orb**2)
    return 1 / (gamma * (1 + v_los))


def redshift_factor(r, phi, M=1.0, theta_obs_deg=12.0):
    """
    Combined gravitational + Doppler redshift factor for a Schwarzschild disk
    seen at elevation theta_obs_deg above the disk plane. Vectorised and
    safe (returns 0 where undefined).
    """
    r = np.asarray(r, float)
    phi = np.asarray(phi, float)
    incl = np.radians(90.0 - theta_obs_deg)
    with np.errstate(all='ignore'):
        beta = np.sqrt(M / r) / np.sqrt(1 - 2*M/r)
        gamma = 1 / np.sqrt(1 - beta**2)
        cos_xi = np.sin(incl) * np.cos(phi)
        delta = 1.0 / (gamma * (1 - beta * cos_xi))
        g = np.sqrt(1 - 2*M/r) * delta
    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
