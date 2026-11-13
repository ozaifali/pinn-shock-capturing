"""Physics-informed neural networks for shock-dominated flows.

Benchmarks: the Sod shock tube (1D Euler) and viscous Burgers' equation.
The exact/reference solvers in :mod:`pinn_shocks.exact` depend only on NumPy;
the networks and trainers need PyTorch.
"""

__version__ = "0.1.0"
