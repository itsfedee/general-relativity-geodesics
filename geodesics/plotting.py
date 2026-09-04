"""
Plotly animations of trajectories and matplotlib helpers.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from .raytrace import tone_map


def animate_geodesic(sol, metric, n_frames=200, duration=30,
                     coordinate_time=False, title='Geodesic', **kwargs):
    """
    Animated 3D trajectory on top of ``metric.draw_background``.

    coordinate_time=True samples frames uniformly in coordinate time t
    (so infall toward a horizon visibly slows down); otherwise uniformly
    in the affine parameter tau.

    If the metric defines ``rotation_markers(t)`` (Kerr), the markers are
    animated too. Works for embedding views as well, since the embedding
    is handled inside ``metric.to_cartesian``.
    """
    x, y, z = metric.to_cartesian(sol, **kwargs)
    if coordinate_time and metric.dim > 2:
        clock = sol.y[0]
    else:
        clock = sol.t
    targets = np.linspace(clock[0], clock[-1], n_frames)
    frame_idx = [int(np.argmin(np.abs(clock - c))) for c in targets]

    fig = go.Figure(data=[
        go.Scatter3d(x=x[:1], y=y[:1], z=z[:1], mode='lines',
                     line=dict(color='red', width=4), name='trajectory'),
        go.Scatter3d(x=x[:1], y=y[:1], z=z[:1], mode='markers',
                     marker=dict(color='red', size=2), name='particle'),
    ])
    metric.draw_background(fig, sol=sol, **kwargs)

    has_rotation = hasattr(metric, 'rotation_markers')
    dot_trace_idx = len(fig.data) - 1 if has_rotation else None

    frames = []
    for i in frame_idx:
        data = [go.Scatter3d(x=x[:i+1], y=y[:i+1], z=z[:i+1]),
                go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]])]
        traces = [0, 1]
        if has_rotation:
            xd, yd, zd = metric.rotation_markers(sol.y[0][i])
            data.append(go.Scatter3d(x=xd, y=yd, z=zd))
            traces.append(dot_trace_idx)
        frames.append(go.Frame(data=data, traces=traces))
    fig.frames = frames

    hidden = dict(showgrid=False, showbackground=False, visible=False)
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis=hidden, yaxis=hidden, zaxis=hidden),
        updatemenus=[dict(type='buttons', buttons=[dict(
            label='Play', method='animate',
            args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True)])])],
        title=title)
    return fig


def plot_radial_velocity(sol, r_index=1):
    """|dr/dtau| and |dr/dt| vs r: the first grows monotonically during infall,
    the second goes to zero at the horizon (coordinate-time freezing)."""
    tau, t_coord, r = sol.t, sol.y[0], sol.y[r_index]
    dr_dtau = np.diff(r) / np.diff(tau)
    dr_dt = np.diff(r) / np.diff(t_coord)
    r_mid = 0.5 * (r[:-1] + r[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(r_mid, np.abs(dr_dtau))
    ax1.set(xlabel='r', ylabel='|dr/dτ|', title='proper-time velocity')
    ax2.plot(r_mid, np.abs(dr_dt))
    ax2.set(xlabel='r', ylabel='|dr/dt|', title='coordinate-time velocity')
    plt.tight_layout()
    return fig


def show_image(img, b_max, title='', mode='gamma', gamma=0.3, figsize=(8, 8),
               interpolation=None):
    """Display a rendered disk image with inferno colormap."""
    plt.figure(figsize=figsize)
    plt.imshow(tone_map(img, mode=mode, gamma=gamma), cmap='inferno', origin='lower',
               extent=[-b_max, b_max, -b_max, b_max], interpolation=interpolation)
    plt.axis('off')
    plt.title(title)
    plt.show()
