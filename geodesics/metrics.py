"""
Spacetime metrics as symbolic sympy matrices.

Every metric knows:
  - its coordinates and metric tensor g (sympy),
  - how to convert Cartesian initial conditions (position + velocity) into
    its own coordinates (``from_cartesian``),
  - how to map a trajectory back to Cartesian space for plotting
    (``to_cartesian``),
  - how to draw its own background (horizon, ergosphere, embedding funnel)
    on a plotly figure (``draw_background``).

The metrics defined here are:
  - Sphere
  - Schwarzschild
  - Reissner-Nordstrom
  - Majumdar-Papapetrou
  - Kerr


The base class assumes spherical-like coordinates (t, r, theta, phi);
subclasses override the conversion helpers where needed (Kerr, MP).
"""

import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.interpolate import interp1d


# ============================================================================
# Base class
# ============================================================================

class Metric:
    """
    Generic metric.

    Parameters
    ----------
    coords : list of sympy Symbols, e.g. [t, r, theta, phi]
    g      : n x n sympy.Matrix
    params : dict {Symbol: float} with the numerical values of the free
             parameters (M, a, Q, ...). Substituted before integration.
    """

    def __init__(self, coords, g, params=None):
        assert len(coords) == g.shape[0] == g.shape[1], \
            f"inconsistent dimensions: {len(coords)} coords, metric {g.shape}"
        self.coords = list(coords)
        self.dim = len(coords)
        self.g = g
        self.params = dict(params or {})
        self._g_inv = None
        self._rhs_cache = None   # filled lazily by geodesics.integrate

    @property
    def g_inv(self):
        if self._g_inv is None:
            if self.g.is_diagonal():
                self._g_inv = sp.diag(*[1 / self.g[i, i] for i in range(self.dim)])
            else:
                self._g_inv = self.g.inv()
        return self._g_inv

    def stop_events(self):
        """
        Terminal solve_ivp events that end the integration (e.g. horizon
        crossing). Default: a single event on r = r_horizon if the metric
        defines ``r_horizon``; otherwise none. Event index 0 in the solver
        output is always the first of these.
        """
        if getattr(self, 'r_horizon', None) is None:
            return []

        def hit_horizon(tau, y):
            return y[1] - self.r_horizon - 0.01
        hit_horizon.terminal = True
        hit_horizon.direction = -1
        return [hit_horizon]

    @property
    def signature(self):
        return [sp.sign(self.g[i, i]) for i in range(self.dim)]

    # ------------------------------------------------------------------ plotting

    def to_cartesian(self, sol, **kwargs):
        """Map a trajectory (solve_ivp result) to (x, y, z) arrays."""
        n = self.dim
        if n == 4:
            r, theta, phi = sol.y[1], sol.y[2], sol.y[3]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
        elif n == 3:
            r, phi = sol.y[1], sol.y[2]
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = np.zeros_like(x)
        elif n == 2:
            theta, phi = sol.y[0], sol.y[1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
        else:
            raise ValueError(f"cannot plot dim={n}")
        return x, y, z

    def draw_background(self, fig, sol=None, **kwargs):
        """Add surfaces / wireframes to a plotly figure. Default: nothing."""
        pass

    # ------------------------------------------------------ initial conditions

    def _pos_from_cartesian(self, x, y, z):
        """(x, y, z) -> spatial curvilinear coordinates. Default: spherical."""
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / r, -1, 1)) if r > 1e-14 else 0.0
        phi = np.arctan2(y, x)
        return np.array([r, theta, phi])

    def _jac_cart_to_curv(self, x, y, z):
        """3x3 Jacobian J_ij = d(x,y,z)_i / d(spatial coord)_j at (x, y, z)."""
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / r, -1, 1)) if r > 1e-14 else 0.0
        phi = np.arctan2(y, x)
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp_ = np.cos(phi), np.sin(phi)
        return np.array([
            [st*cp,   r*ct*cp,  -r*st*sp_],
            [st*sp_,  r*ct*sp_,  r*st*cp ],
            [ct,     -r*st,      0       ]
        ])

    def from_cartesian(self, x, y, z, vx, vy, vz, massive=True):
        """
        Cartesian initial conditions -> (x0, v0) in the metric's coordinates,
        ready for ``solve_geodesic``.

        x0 = [t=0, spatial coords...],  v0 = [v^t, spatial velocities...]

        v^t is fixed by the normalisation g_{mu nu} v^mu v^nu = -1
        (massive=True) or 0 (massive=False, null geodesic).
        """
        spatial = self._pos_from_cartesian(x, y, z)
        x0 = np.concatenate([[0.0], spatial])

        J = self._jac_cart_to_curv(x, y, z)
        v_spatial = np.linalg.solve(J, [vx, vy, vz])

        epsilon = -1.0 if massive else 0.0
        g_num = np.array(self.g.subs(self.params).subs(
            list(zip(self.coords, x0))
        )).astype(float)

        n = self.dim
        A = g_num[0, 0]
        B = 2 * sum(g_num[0, i] * v_spatial[i-1] for i in range(1, n))
        C = sum(g_num[i, j] * v_spatial[i-1] * v_spatial[j-1]
                for i in range(1, n) for j in range(1, n)) - epsilon

        disc = B**2 - 4*A*C
        assert disc >= 0, f"negative discriminant: {disc:.4e} (unphysical initial conditions)"

        vt1 = (-B + np.sqrt(disc)) / (2*A)
        vt2 = (-B - np.sqrt(disc)) / (2*A)
        vt = vt1 if vt1 > 0 else vt2
        assert vt > 0, "no future-directed solution for v^t"

        return x0, np.concatenate([[vt], v_spatial])


# ============================================================================
# Helpers for drawing
# ============================================================================

def _black_sphere(fig, R, center=(0, 0, 0), n=30, name=None, **kw):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u, v)
    fig.add_trace(go.Surface(
        x=R*np.sin(V)*np.cos(U) + center[0],
        y=R*np.sin(V)*np.sin(U) + center[1],
        z=R*np.cos(V) + center[2],
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False, name=name, **kw))


def _wireframe(fig, X, Y, Z, step_i=4, step_j=4, color='gray'):
    for i in range(0, X.shape[1], step_i):
        fig.add_trace(go.Scatter3d(x=X[:, i], y=Y[:, i], z=Z[:, i], mode='lines',
                                   line=dict(color=color, width=1), showlegend=False))
    for j in range(0, X.shape[0], step_j):
        fig.add_trace(go.Scatter3d(x=X[j, :], y=Y[j, :], z=Z[j, :], mode='lines',
                                   line=dict(color=color, width=1), showlegend=False))


# ============================================================================
# Sphere
# ============================================================================

class Sphere(Metric):
    """
    Round n-sphere of radius R:
    ds^2 = R^2 dtheta_0^2 + R^2 sin^2(theta_0) dtheta_1^2 + ...
    """

    def __init__(self, n, R_val=1.0):
        assert n >= 1
        R = sp.Symbol('R', positive=True)
        theta = [sp.Symbol(f'theta_{i}') for i in range(n)]
        diag = []
        for i in range(n):
            entry = R**2
            for j in range(i):
                entry *= sp.sin(theta[j])**2
            diag.append(entry)
        super().__init__(theta, sp.diag(*diag), params={R: R_val})

    def draw_background(self, fig, sol=None, **kwargs):
        if self.dim != 2:
            return
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        U, V = np.meshgrid(u, v)
        _wireframe(fig, np.sin(V)*np.cos(U), np.sin(V)*np.sin(U), np.cos(V),
                   step_i=2, step_j=1, color='lightblue')


# ============================================================================
# Schwarzschild (D dimensions)
# ============================================================================

class Schwarzschild(Metric):
    """Schwarzschild(-Tangherlini) metric in D = dim spacetime dimensions."""

    def __init__(self, dim=4, M_val=1.0, embedding=False):
        t, r = sp.symbols('t r')
        M = sp.Symbol('M', positive=True)
        f = 1 - (2*M/r)**(dim - 3)

        sphere = Sphere(dim - 2)
        sphere_g = sphere.g.subs(sp.Symbol('R', positive=True), r)
        coords = [t, r] + sphere.coords
        g = sp.diag(-f, 1/f, *[sphere_g[i, i] for i in range(sphere.dim)])

        super().__init__(coords, g, params={M: M_val})
        self.M_val = M_val
        self.r_horizon = 2 * M_val
        self.r_isco = 6 * M_val
        self.b_crit = 3 * np.sqrt(3) * M_val   # photon-sphere critical impact parameter
        self.embedding = embedding             # Flamm paraboloid view

    def to_cartesian(self, sol, **kwargs):
        if self.embedding:
            r_orb = sol.y[1]
            phi_orb = sol.y[2] if self.dim == 3 else sol.y[3]
            z = 2 * np.sqrt(2 * self.M_val * np.maximum(r_orb - self.r_horizon, 0))
            return r_orb * np.cos(phi_orb), r_orb * np.sin(phi_orb), z
        return super().to_cartesian(sol)

    def draw_background(self, fig, sol=None, **kwargs):
        r_s = self.r_horizon
        if self.embedding:
            r_max = sol.y[1].max() * 1.1 if sol is not None else 20.0
            r_surf = np.linspace(r_s * 1.01, r_max, 40)
            phi_surf = np.linspace(0, 2*np.pi, 40)
            R_g, P_g = np.meshgrid(r_surf, phi_surf)
            Z = 2 * np.sqrt(2 * self.M_val * (R_g - r_s))
            _wireframe(fig, R_g*np.cos(P_g), R_g*np.sin(P_g), Z)
            z_bottom = 2 * np.sqrt(2 * self.M_val * (r_s * 0.01))
            _black_sphere(fig, r_s, center=(0, 0, z_bottom))
        else:
            _black_sphere(fig, r_s)


# ============================================================================
# Reissner-Nordstrom
# ============================================================================

class ReissnerNordstrom(Metric):
    """Charged, non-rotating black hole."""

    def __init__(self, M_val=1.0, Q_val=0.5):
        t, r, th, ph = sp.symbols('t r θ φ')
        M = sp.Symbol('M', positive=True)
        Q = sp.Symbol('Q', real=True)
        f = 1 - 2*M/r + Q**2/r**2
        g = sp.diag(-f, 1/f, r**2, r**2 * sp.sin(th)**2)
        super().__init__([t, r, th, ph], g, params={M: M_val, Q: Q_val})
        self.M_val, self.Q_val = M_val, Q_val

        disc = M_val**2 - Q_val**2
        if disc > 0:
            self.r_horizon = M_val + np.sqrt(disc)
            self.r_inner = M_val - np.sqrt(disc)
        elif disc == 0:
            self.r_horizon = self.r_inner = M_val
        else:                                   # naked singularity
            self.r_horizon = self.r_inner = None

    def draw_background(self, fig, sol=None, **kwargs):
        if self.r_horizon is not None:
            _black_sphere(fig, self.r_horizon, name='outer horizon')
        if self.r_inner is not None and self.r_inner != self.r_horizon:
            u = np.linspace(0, 2*np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            U, V = np.meshgrid(u, v)
            r_i = self.r_inner
            fig.add_trace(go.Surface(
                x=r_i*np.sin(V)*np.cos(U), y=r_i*np.sin(V)*np.sin(U), z=r_i*np.cos(V),
                colorscale=[[0, 'rgba(0,0,100,0.5)'], [1, 'rgba(0,0,100,0.5)']],
                showscale=False, opacity=0.4, name='inner horizon'))


# ============================================================================
# Majumdar-Papapetrou
# ============================================================================

class MajumdarPapapetrou(Metric):
    """Static configuration of extremal charged black holes in equilibrium."""

    def __init__(self, masses, positions):
        t, x, y, z = sp.symbols('t x y z')
        U = sp.S.One
        for M_val, pos in zip(masses, positions):
            r_i = sp.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
            U += M_val / r_i
        super().__init__([t, x, y, z], sp.diag(-1/U**2, U**2, U**2, U**2))
        self.masses = list(masses)
        self.positions = [np.asarray(p, float) for p in positions]

    def _pos_from_cartesian(self, x, y, z):
        return np.array([x, y, z])

    def _jac_cart_to_curv(self, x, y, z):
        return np.eye(3)

    def to_cartesian(self, sol, **kwargs):
        return sol.y[1], sol.y[2], sol.y[3]

    def stop_events(self, margin=0.05):
        """Stop when the particle gets within ``margin`` * M_i of any centre
        (the horizons are at r_i = 0 in isotropic coordinates and the
        integration becomes stiff there)."""
        def near_center(tau, y):
            d = [np.linalg.norm(y[1:4] - pos) - margin * M
                 for M, pos in zip(self.masses, self.positions)]
            return min(d)
        near_center.terminal = True
        near_center.direction = -1
        return [near_center]

    def draw_background(self, fig, sol=None, **kwargs):
        for M_val, pos in zip(self.masses, self.positions):
            _black_sphere(fig, M_val, center=pos)   # horizon at r = M in these coords


# ============================================================================
# Kerr (Boyer-Lindquist)
# ============================================================================

class Kerr(Metric):
    """Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi)."""

    def __init__(self, M_val=1.0, a_val=0.5, embedding=False):
        t, r, th, ph = sp.symbols('t r θ φ')
        M = sp.Symbol('M', positive=True)
        a = sp.Symbol('a', real=True)

        Sigma = r**2 + a**2 * sp.cos(th)**2
        Delta = r**2 - 2*M*r + a**2
        g_tt = -(1 - (2*M*r)/Sigma)
        g_rr = Sigma / Delta
        g_thth = Sigma
        g_phph = (r**2 + a**2 + (2*M*r*a**2 * sp.sin(th)**2)/Sigma) * sp.sin(th)**2
        g_tphi = -(2*M*r*a * sp.sin(th)**2) / Sigma
        g = sp.Matrix([
            [g_tt,   0,    0,      g_tphi],
            [0,      g_rr, 0,      0     ],
            [0,      0,    g_thth, 0     ],
            [g_tphi, 0,    0,      g_phph],
        ])
        super().__init__([t, r, th, ph], g, params={M: M_val, a: a_val})

        self.M_val, self.a_val = M_val, a_val
        self.embedding = embedding
        self._embedding_data = None

        self.r_horizon = M_val + np.sqrt(M_val**2 - a_val**2)
        self.omega_h = a_val / (self.r_horizon**2 + a_val**2)   # horizon angular velocity
        z1 = 1 + (1 - a_val**2)**(1/3) * ((1 + a_val)**(1/3) + (1 - a_val)**(1/3))
        z2 = np.sqrt(3 * a_val**2 + z1**2)
        self.r_isco = M_val * (3 + z2 - np.sqrt((3 - z1)*(3 + z1 + 2*z2)))   # prograde ISCO
        self.b_crit = 3 * np.sqrt(3) * M_val   # Schwarzschild value, used only as a capture cutoff

    @property
    def g_inv(self):
        """Closed-form inverse (much faster than sympy's generic inv())."""
        t, r, th, ph = self.coords
        M = sp.Symbol('M', positive=True)
        a = sp.Symbol('a', real=True)
        Sigma = r**2 + a**2 * sp.cos(th)**2
        Delta = r**2 - 2*M*r + a**2
        return sp.Matrix([
            [-(r**2 + a**2 + (2*M*r*a**2*sp.sin(th)**2)/Sigma)/Delta, 0, 0, -(2*M*r*a)/(Sigma*Delta)],
            [0, Delta/Sigma, 0, 0],
            [0, 0, 1/Sigma, 0],
            [-(2*M*r*a)/(Sigma*Delta), 0, 0, (1 - (2*M*r)/Sigma)/(Delta*sp.sin(th)**2)],
        ])

    # ------------------------------------------------------------ coordinates

    @staticmethod
    def bl_to_cartesian(r, theta, phi, a):
        rho = np.sqrt(r**2 + a**2)
        return (rho*np.sin(theta)*np.cos(phi),
                rho*np.sin(theta)*np.sin(phi),
                r*np.cos(theta))

    @staticmethod
    def cartesian_to_bl(x, y, z, a):
        w2 = x**2 + y**2 + z**2
        diff = w2 - a**2
        r2 = 0.5 * (diff + np.sqrt(diff**2 + 4 * a**2 * z**2))
        r = np.sqrt(np.maximum(r2, 0.0))
        cos_theta = np.where(r > 1e-14, z / r, 0.0)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def _pos_from_cartesian(self, x, y, z):
        r, theta, phi = Kerr.cartesian_to_bl(x, y, z, self.a_val)
        return np.array([float(r), float(theta), float(phi)])

    def _jac_cart_to_curv(self, x, y, z):
        r_bl, theta, phi = (float(v) for v in Kerr.cartesian_to_bl(x, y, z, self.a_val))
        a = self.a_val
        rho = np.sqrt(r_bl**2 + a**2)
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp_ = np.cos(phi), np.sin(phi)
        dr_drho = r_bl / rho
        return np.array([
            [dr_drho*st*cp,  rho*ct*cp,  -rho*st*sp_],
            [dr_drho*st*sp_, rho*ct*sp_,  rho*st*cp ],
            [ct,            -r_bl*st,     0          ]
        ])

    # -------------------------------------------------------------- embedding

    @property
    def embedding_data(self):
        if self._embedding_data is None:
            self.compute_embedding()
        return self._embedding_data

    def compute_embedding(self, r_max=20.0, n_pts=2000):
        """Equatorial embedding surface (r, rho, z) of the Kerr geometry; cached."""
        M, a = self.M_val, self.a_val
        r = np.linspace(self.r_horizon * 1.01, r_max, n_pts)
        Delta = r**2 - 2*M*r + a**2
        rho = np.sqrt(r**2 + a**2 + 2*M*a**2 / r)
        drho = (r**3 - M*a**2) / (r**2 * rho)
        z_prime_sq = r**2 / Delta - drho**2
        valid = z_prime_sq >= 0
        r, rho, z_prime_sq = r[valid], rho[valid], z_prime_sq[valid]
        z = np.cumsum(np.sqrt(z_prime_sq) * np.gradient(r))
        self._embedding_data = (r, rho, z)
        return r, rho, z

    def to_cartesian(self, sol, **kwargs):
        r_traj, theta, phi = sol.y[1], sol.y[2], sol.y[3]
        if self.embedding:
            r_emb, rho_emb, z_emb = self.embedding_data
            rho_traj = interp1d(r_emb, rho_emb, fill_value='extrapolate')(r_traj)
            z_traj = interp1d(r_emb, z_emb, fill_value='extrapolate')(r_traj)
            return rho_traj * np.cos(phi), rho_traj * np.sin(phi), z_traj
        return Kerr.bl_to_cartesian(r_traj, theta, phi, self.a_val)

    def rotation_markers(self, t, n_dots=4):
        """Dots on the horizon rotating with omega_h, to visualise frame dragging."""
        phi_now = np.linspace(0, 2*np.pi, n_dots, endpoint=False) + self.omega_h * t
        if self.embedding:
            r_emb, rho_emb, z_emb = self.embedding_data
            return rho_emb[0]*np.cos(phi_now), rho_emb[0]*np.sin(phi_now), np.full(n_dots, z_emb[0])
        rho_h = np.sqrt(self.r_horizon**2 + self.a_val**2)
        return rho_h*np.cos(phi_now), rho_h*np.sin(phi_now), np.zeros(n_dots)

    def draw_background(self, fig, sol=None, **kwargs):
        ergo_color = [[0, 'rgba(80,0,120,0.3)'], [1, 'rgba(80,0,120,0.3)']]
        if self.embedding:
            r_emb, rho_emb, z_emb = self.embedding_data
            phi_surf = np.linspace(0, 2*np.pi, 100)
            step = max(1, len(r_emb) // 40)
            R_g, P_g = np.meshgrid(rho_emb[::step], phi_surf)
            Z_g = np.tile(z_emb[::step], (len(phi_surf), 1))
            _wireframe(fig, R_g*np.cos(P_g), R_g*np.sin(P_g), Z_g, step_i=8, step_j=12)

            mask = r_emb <= 2 * self.M_val      # equatorial ergosphere
            if np.any(mask):
                phi_e = np.linspace(0, 2*np.pi, 60)
                R_e, P_e = np.meshgrid(rho_emb[mask], phi_e)
                Z_e = np.tile(z_emb[mask], (len(phi_e), 1))
                fig.add_trace(go.Surface(x=R_e*np.cos(P_e), y=R_e*np.sin(P_e), z=Z_e,
                                         colorscale=ergo_color, showscale=False,
                                         opacity=0.3, name='ergosphere'))
            _black_sphere(fig, rho_emb[0]*0.3, center=(0, 0, z_emb[0]))
        else:
            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            U, V = np.meshgrid(u, v)
            rho_h = np.sqrt(self.r_horizon**2 + self.a_val**2)
            fig.add_trace(go.Surface(
                x=rho_h*np.sin(V)*np.cos(U), y=rho_h*np.sin(V)*np.sin(U), z=self.r_horizon*np.cos(V),
                colorscale=[[0, 'black'], [1, 'black']], showscale=False, name='horizon'))
            r_e = self.M_val + np.sqrt(np.maximum(self.M_val**2 - self.a_val**2*np.cos(V)**2, 0))
            rho_e = np.sqrt(r_e**2 + self.a_val**2)
            fig.add_trace(go.Surface(
                x=rho_e*np.sin(V)*np.cos(U), y=rho_e*np.sin(V)*np.sin(U), z=r_e*np.cos(V),
                colorscale=ergo_color, showscale=False, opacity=0.25, name='ergosphere'))

        xd, yd, zd = self.rotation_markers(0.0)
        fig.add_trace(go.Scatter3d(x=xd, y=yd, z=zd, mode='markers',
                                   marker=dict(color='yellow', size=3), name='rotation'))
