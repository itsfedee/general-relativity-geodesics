"""
Neural surrogate for the ray-tracing map  (by, bz, theta) -> (r_hit, phi_hit).

Once trained, an image at any viewing angle is a single batched forward
pass instead of N*N geodesic integrations, which makes interactive
re-rendering possible.

Requires torch (optional dependency: ``pip install geodesics[nn]``).
"""

import numpy as np

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:   # pragma: no cover
    raise ImportError("geodesics.surrogate needs torch: pip install torch") from e

from .disk import doppler_factor


class Normalizer:
    """Per-feature standardisation fitted on the training set."""

    def __init__(self, X, Y):
        self.X_mean, self.X_std = X.mean(axis=0), X.std(axis=0)
        self.Y_mean, self.Y_std = Y.mean(axis=0), Y.std(axis=0)
        self.X_std[self.X_std < 1e-8] = 1.0
        self.Y_std[self.Y_std < 1e-8] = 1.0

    def x(self, X):
        return torch.tensor((X - self.X_mean) / self.X_std, dtype=torch.float32)

    def y(self, Y):
        return torch.tensor((Y - self.Y_mean) / self.Y_std, dtype=torch.float32)

    def y_inv(self, Yn):
        return Yn * self.Y_std + self.Y_mean


def load_dataset(path, hits_only=True, order=0, test_frac=0.2, seed=42, batch_size=512):
    """
    Load a dataset produced by ``geodesics.dataset`` and split it.

    Inputs  X = (by, bz, theta_deg); targets Y = (r_n, phi_n) for image order n.
    hits_only=True keeps only rays that actually hit the disk at that order.

    Returns (loader, X_test, Y_test_raw, norm).
    """
    data = np.load(path)
    base = 4 + 3 * order
    if hits_only:
        data = data[data[:, base + 2] > 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    split = int((1 - test_frac) * len(data))
    train, test = data[:split], data[split:]
    X_raw, Y_raw = train[:, :3], train[:, base:base + 2]
    norm = Normalizer(X_raw, Y_raw)

    loader = DataLoader(TensorDataset(norm.x(X_raw), norm.y(Y_raw)),
                        batch_size=batch_size, shuffle=True)
    return loader, norm.x(test[:, :3]), test[:, base:base + 2], norm


def build_model(hidden=(256, 256, 128), n_in=3, n_out=2):
    layers, d = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.SiLU()]
        d = h
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers)


def train(model, loader, X_test, Y_test_raw, norm, epochs=1000, lr=1e-3,
          log_every=50, optimizer=None):
    """MSE training loop in normalised space; reports test MSE in physical units."""
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % log_every == 0:
            with torch.no_grad():
                pred = norm.y_inv(model(X_test).numpy())
                test_mse = ((pred - Y_test_raw)**2).mean()
            history.append((epoch + 1, total / len(loader), test_mse))
            print(f"epoch {epoch+1}: train_loss={total/len(loader):.6f}  test_MSE={test_mse:.4f}")
    return history


def evaluate(model, X_test, Y_test_raw, norm):
    """MSE, RMSE, MAE, R^2 and MAPE on r_hit, in physical units."""
    with torch.no_grad():
        pred = norm.y_inv(model(X_test).numpy())
    real = Y_test_raw
    mse = ((pred - real)**2).mean()
    ss_res = ((pred - real)**2).sum()
    ss_tot = ((real - real.mean(axis=0))**2).sum()
    mask = real[:, 0] > 0
    return dict(
        mse=mse, rmse=np.sqrt(mse), mae=np.abs(pred - real).mean(),
        r2=1 - ss_res / ss_tot,
        mape_r=(np.abs(pred[mask, 0] - real[mask, 0]) / real[mask, 0]).mean() * 100)


def render_nn(model, norm, metric, T_interp, theta_obs_deg, N=300, b_max=12.0, r_max=20.0):
    """Render the disk at angle theta_obs_deg with one forward pass."""
    b = np.linspace(-b_max, b_max, N)
    by, bz = np.meshgrid(b, b)
    b_mag = np.hypot(by, bz)
    inputs = norm.x(np.column_stack([by.ravel(), bz.ravel(), np.full(N*N, theta_obs_deg)]))
    with torch.no_grad():
        out = norm.y_inv(model(inputs).numpy())
    r_hit = out[:, 0].reshape(N, N)
    phi_hit = out[:, 1].reshape(N, N)
    mask = (b_mag > metric.b_crit * 1.03) & (r_hit >= metric.r_isco) & (r_hit <= r_max)
    with np.errstate(all='ignore'):
        img = T_interp(r_hit) * doppler_factor(r_hit, phi_hit, metric.M_val)**4
    return np.where(mask, img, 0.0)
