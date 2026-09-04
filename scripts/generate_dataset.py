"""Generate the Schwarzschild ray-tracing dataset (see geodesics.dataset)."""
import argparse
from geodesics.dataset import create_raytracing_dataset

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metric', default='Schwarzschild')
    p.add_argument('--n-angles', type=int, default=20)
    p.add_argument('--n-per-angle', type=int, default=100)
    p.add_argument('--k-max', type=int, default=3)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    create_raytracing_dataset(metric_name=a.metric, n_angles=a.n_angles,
                              n_per_angle=a.n_per_angle, k_max=a.k_max,
                              n_workers=a.workers, filename=a.out,
                              partial_file='raytracing_partial.npy')
