| Method | RMSE ρ | RMSE u | RMSE p | rel. L2 p | Shock position | Shock width | Train time |
|---|---|---|---|---|---|---|---|
| Exact | – | – | – | – | 0.8504 | 0 | – |
| Vanilla PINN | 0.0412 | 0.1124 | 0.0363 | 5.85% | 0.8261 | n/a | 28 s |
| PINN + artificial viscosity | 0.0462 | 0.1261 | 0.0440 | 7.09% | 0.8205 | 0.0897 | 71 s |
| Improved PINN (Adam only) | 0.0296 | 0.0995 | 0.0316 | 5.09% | 0.8390 | 0.1037 | 38 s |
| Improved PINN + L-BFGS | 0.0219 | 0.1038 | 0.0254 | 4.08% | 0.8637 | 0.0191 | 98 s |
| Fixed-interface XPINN | 0.0317 | 0.1401 | 0.0355 | 5.71% | 0.8115 | 0.0753 | 39 s |
| **Shock-fitting PINN (learned s)** | 0.0104 | 0.0299 | 0.0075 | 1.21% | 0.8493 | 0.0004 | 160 s |

| Burgers | rel. L2 t=0.25 | t=0.5 | t=0.75 | t=1.0 | Train time |
|---|---|---|---|---|---|
| ν = 0.01/π | 0.20% | 0.33% | 0.76% | 1.00% | 125 s |
| ν = 0.002/π | 2.57% | 21.95% | 8.87% | 16.58% | 109 s |
