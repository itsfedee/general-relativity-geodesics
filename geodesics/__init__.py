"""
geodesics: symbolic metrics, numerical geodesics, black-hole ray tracing.

Quick start
-----------
>>> from geodesics import Schwarzschild, solve_geodesic, animate_geodesic
>>> bh = Schwarzschild(M_val=1.0)
>>> x0, v0 = bh.from_cartesian(x=10, y=4, z=0, vx=-0.5, vy=0.1, vz=0, massive=True)
>>> sol = solve_geodesic(bh, x0, v0, (0, 80))
>>> animate_geodesic(sol, bh).show()
"""

from .metrics import (Metric, Sphere, Schwarzschild, ReissnerNordstrom,
                      MajumdarPapapetrou, Kerr)
from .symbolic import (compute_christoffel, print_metric, print_christoffel,
                       print_geodesic_eqs)
from .integrate import (geodesic_rhs, get_rhs, solve_geodesic, resample,
                        cross_equator, escape_event, escaping)
from .disk import disk_flux, temperature_interpolator, doppler_factor, redshift_factor
from .raytrace import trace_ray, render_image, tone_map, camera_ray
from .plotting import animate_geodesic, plot_radial_velocity, show_image

__version__ = "0.1.0"
