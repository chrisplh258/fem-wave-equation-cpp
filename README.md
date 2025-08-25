

## How this branch differs from `main`

| Aspect             | `main` branch                                | `feature/mixed-bc` (this branch)       |
|--------------------|-----------------------------------------------|----------------------------------------|
| Boundary conditions| Dirichlet on all boundaries                  | Dirichlet on $x=0,1$, Neumann on $y=0,1$ |
| Initial displacement| $\sin(\pi x)\sin(\pi y)$                     | $\sin(\pi x)$                          |
| Initial velocity   | $0$                                          | $0$                                    |
| Analytic check     | $\sin(\pi x)\sin(\pi y)\cos(\pi \sqrt{2} c t)$ | $\sin(\pi x)\cos(\pi c t)$            |

---