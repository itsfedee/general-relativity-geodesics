"""
Backward ray tracing of a thin equatorial disk.

A camera at distance 200 M, elevated by theta_obs above the disk plane,
shoots one null geodesic per pixel (impact parameters by, bz). Each ray is
integrated until it is captured, escapes, or has crossed the equatorial
plane k_max times; the crossing radii r_n and azimuths phi_n are recorded
(n = 0 direct image, n = 1 first lensed image, ...).
"""

import time
import numpy as np

from .integrate import solve_geodesic, get_rhs, cross_equator, escaping
from .disk import doppler_factor

CLS_CAPTURED = 0
CLS_ESCAPED = 1
CAMERA_DISTANCE = 200.0


def camera_ray(metric, by, bz, theta_obs, distance=CAMERA_DISTANCE):
    """Initial conditions for the pixel (by, bz) of a camera at elevation theta_obs."""
    ct, st = np.cos(theta_obs), np.sin(theta_obs)
    return metric.from_cartesian(
        x=distance*ct - bz*st, y=by, z=distance*st + bz*ct,
        vx=-ct, vy=0.0, vz=-st, massive=False)


def trace_ray(metric, by, bz, theta_obs, r_max=20.0, k_max=3,
              rtol=1e-8, atol=1e-8, rhs=None):
    """
    Trace one ray and return a flat row:
        [by, bz, theta_deg, cls, r_0, phi_0, hit_0, r_1, phi_1, hit_1, ...]
    with ncols = 4 + 3*k_max.
    """
    row = np.zeros(4 + 3 * k_max)
    row[0], row[1], row[2] = by, bz, np.degrees(theta_obs)

    if np.hypot(by, bz) <= metric.b_crit:
        row[3] = CLS_CAPTURED
        return row

    x0, v0 = camera_ray(metric, by, bz, theta_obs)
    sol = solve_geodesic(metric, x0, v0, (0, 1000), rhs=rhs or get_rhs(metric),
                         extra_events=[cross_equator, escaping],
                         rtol=rtol, atol=atol, dense_output=False)

    captured = sol.t_events[0].size > 0
    row[3] = CLS_CAPTURED if captured else CLS_ESCAPED

    n = 0
    for y_cross in sol.y_events[1]:
        if n >= k_max:
            break
        r_c = y_cross[1]
        if metric.r_isco <= r_c <= r_max and r_c > metric.r_horizon + 2:
            base = 4 + 3 * n
            row[base], row[base + 1], row[base + 2] = r_c, y_cross[3], 1.0
            n += 1
    return row


def render_image(metric, T_interp, theta_obs_deg, N=200, b_max=30.0, r_max=20.0,
                 rtol=1e-8, atol=1e-8, verbose=True):
    """
    Brute-force render: one geodesic per pixel, direct image only (n = 0),
    intensity = T(r_hit) * g**4 with g the Doppler factor.

    Returns (img, r_hit_map, phi_hit_map), all N x N.
    """
    theta_obs = np.radians(theta_obs_deg)
    b_range = np.linspace(-b_max, b_max, N)
    img = np.zeros((N, N))
    r_map = np.zeros((N, N))
    phi_map = np.zeros((N, N))
    rhs = get_rhs(metric)
    t0 = time.time()

    for i, by in enumerate(b_range):
        for j, bz in enumerate(b_range):
            if abs(bz) < 0.01:
                continue
            row = trace_ray(metric, by, bz, theta_obs, r_max=r_max, k_max=1,
                            rtol=rtol, atol=atol, rhs=rhs)
            if row[6] > 0:
                r_hit, phi_hit = row[4], row[5]
                r_map[j, i], phi_map[j, i] = r_hit, phi_hit
                img[j, i] = T_interp(r_hit) * doppler_factor(r_hit, phi_hit, metric.M_val)**4
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"row {i+1}/{N}: {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
    return img, r_map, phi_map


def tone_map(img, mode='gamma', gamma=0.3):
    """
    Map raw intensities to [0, 1] for display. Background (img == 0) stays 0.
    mode='log'   : log10 stretch between min and max of the lit pixels
    mode='gamma' : (img / max)**gamma
    """
    out = np.zeros_like(img, dtype=float)
    mask = img > 0
    if not mask.any():
        return out
    if mode == 'log':
        lg = np.log10(img[mask])
        out[mask] = (lg - lg.min()) / (lg.max() - lg.min())
    else:
        out[mask] = (img[mask] / img.max())**gamma
    return out
