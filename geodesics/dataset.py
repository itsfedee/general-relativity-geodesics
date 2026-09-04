"""
Parallel generation of the ray-tracing dataset used to train the neural surrogate.

Each row is produced by ``raytrace.trace_ray``:
    0: by   1: bz   2: theta_deg   3: cls (0 = captured, 1 = escaped)
    for each image order n in [0, k_max):  4+3n: r_n,  4+3n+1: phi_n,  4+3n+2: hit_n

Sampling: a uniform grid in (by, bz) per viewing angle, plus an optional
"ring boost" of extra samples just outside the critical impact parameter,
where the lensed images live and the map is steepest.

Multiprocessing is Windows-safe: each worker builds its own metric + RHS
once at start-up, and tasks are plain float tuples.
"""

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from . import metrics as _metrics
from .integrate import get_rhs
from .raytrace import trace_ray, CLS_CAPTURED, CLS_ESCAPED

_W = {}


def _init_worker(metric_name, metric_kwargs, r_max, k_max, rtol, atol):
    metric = getattr(_metrics, metric_name)(**metric_kwargs)
    _W.update(metric=metric, rhs=get_rhs(metric),
              r_max=r_max, k_max=k_max, rtol=rtol, atol=atol)


def _work(task):
    by, bz, theta_obs = task
    return trace_ray(_W['metric'], by, bz, theta_obs, r_max=_W['r_max'],
                     k_max=_W['k_max'], rtol=_W['rtol'], atol=_W['atol'],
                     rhs=_W['rhs'])


def build_tasks(theta_obs, b_range, b_crit, ring_boost=True, n_ring_r=40,
                n_ring_ang=180, ring_frac=0.3):
    tasks = [(by, bz, theta_obs) for by in b_range for bz in b_range if abs(bz) >= 0.01]
    if ring_boost:
        b_vals = b_crit + np.geomspace(1e-3, ring_frac * b_crit, n_ring_r)
        psi = np.linspace(0, 2*np.pi, n_ring_ang, endpoint=False)
        for bb in b_vals:
            for pp in psi:
                by, bz = bb*np.cos(pp), bb*np.sin(pp)
                if abs(bz) >= 0.01:
                    tasks.append((by, bz, theta_obs))
    return tasks


def create_raytracing_dataset(metric_name='Schwarzschild', metric_kwargs=None,
                              n_angles=20, n_per_angle=100, b_max=12, r_max=20,
                              k_max=3, ring_boost=True, n_ring_r=40, n_ring_ang=180,
                              ring_frac=0.3, rtol=1e-8, atol=1e-8,
                              n_workers=None, filename=None, partial_file=None):
    """
    Generate and save the dataset. Returns the (n_rows, 4 + 3*k_max) array.

    metric_name / metric_kwargs : any class in ``geodesics.metrics`` with
        r_horizon, r_isco, b_crit attributes (Schwarzschild, Kerr).
    """
    metric_kwargs = dict(metric_kwargs or {})
    metric = getattr(_metrics, metric_name)(**metric_kwargs)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    if filename is None:
        filename = f'raytracing_dataset_{metric_name}.npy'

    angles = np.linspace(1, 89, n_angles)
    b_range = np.linspace(-b_max, b_max, n_per_angle)
    print(f"workers={n_workers}, rtol={rtol}, atol={atol}")

    data = []
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(metric_name, metric_kwargs, r_max, k_max, rtol, atol)) as ex:
        for k, theta_deg in enumerate(angles):
            tasks = build_tasks(np.radians(theta_deg), b_range, metric.b_crit,
                                ring_boost, n_ring_r, n_ring_ang, ring_frac)
            data.extend(ex.map(_work, tasks, chunksize=64))
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(angles) - k - 1)
            print(f"angle {theta_deg:.0f} deg ({k+1}/{n_angles}): "
                  f"{elapsed:.0f}s, ~{eta:.0f}s left, rows={len(data)}")
            if partial_file:
                np.save(partial_file, np.array(data))

    data = np.array(data)
    np.save(filename, data)
    cls = data[:, 3]
    print(f"\ndataset: {len(data)} rows -> {filename}")
    print(f"  captured={int((cls == CLS_CAPTURED).sum())}, escaped={int((cls == CLS_ESCAPED).sum())}")
    for n in range(k_max):
        print(f"  hit n={n}: {int(data[:, 4 + 3*n + 2].sum())}")
    return data


def render_truth(theta_deg, N=200, k_max=3, b_max=12, metric_name='Schwarzschild',
                 metric_kwargs=None, r_max=20, rtol=1e-8, atol=1e-8, n_workers=None,
                 orders=None):
    """
    Parallel ground-truth hit mask at a given angle: pixel value = number of
    recorded crossings among the selected image ``orders`` (default: all).
    Useful for checking that the lensed ring is captured by k_max > 1.
    """
    metric_kwargs = dict(metric_kwargs or {})
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    orders = range(k_max) if orders is None else orders
    theta_obs = np.radians(theta_deg)
    b = np.linspace(-b_max, b_max, N)
    tasks, idx = [], []
    for i, by in enumerate(b):
        for j, bz in enumerate(b):
            if abs(bz) >= 0.01:
                tasks.append((by, bz, theta_obs))
                idx.append((i, j))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(metric_name, metric_kwargs, r_max, k_max, rtol, atol)) as ex:
        rows = list(ex.map(_work, tasks, chunksize=64))
    img = np.zeros((N, N))
    for (i, j), row in zip(idx, rows):
        img[j, i] = sum(row[4 + 3*n + 2] > 0 for n in orders)
    return img
