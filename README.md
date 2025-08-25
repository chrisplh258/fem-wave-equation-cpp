# 2D Wave Equation (FEM, deal.II)

This project implements a minimal finite element solver for the **2D wave equation**

$u_{tt} = c^2 (u_{xx} + u_{yy})$

on the unit square with:

* **Homogeneous Dirichlet boundary conditions**: $u = 0$ on the boundary.
* **Initial displacement**: $u(x,y,0) = \sin(\pi x) \sin(\pi y)$.
* **Initial velocity**: $u_t(x,y,0) = 0$.

The solution is advanced in time using a two–step scheme with assembled mass (**M**) and stiffness (**K**) matrices. At the final time, the solver outputs a ParaView‑ready `.vtu` file and computes the L2 error against the exact analytical solution.

The project also ships with a small smoke‑test harness. Everything is containerized so **no local deal.II install is required**.

---

## Run with Docker (recommended)

> Only Docker is required. The image build step compiles the code inside the container.

```bash
# 1) Clone
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2) Build the image (this compiles the C++ code inside Docker)
docker build -t fem-wave:1.0 .

# 3) Choose a host folder for outputs (ParaView will open files from here)
mkdir -p outputs

# 4) Run the solver and write all files into ./outputs on your machine
docker run --rm -v "$PWD/outputs:/outputs" -w /outputs fem-wave:1.0 \
  /app/build/wave_solver --h 0.04 --dt 2.5e-4 --c 1.0

# (Optional) Run the tests, logs go to ./outputs
docker run --rm -v "$PWD/outputs:/outputs" -w /outputs fem-wave:1.0 \
  /app/build/run_tests

# (Optional) Or via CTest for verbose failure messages
docker run --rm -v "$PWD/outputs:/outputs" -w /outputs fem-wave:1.0 \
  ctest --test-dir /app/build --output-on-failure
```

**Windows PowerShell:** replace `$PWD` with `${PWD}` or use `%cd%` in CMD.

### What gets produced

* `final_solution.vtu` — final-time displacement/velocity (open in ParaView)
* `out*.txt` — textual logs written by the test harness


---

## Command‑line arguments

The solver accepts a few simple flags (all optional with sensible defaults):

* `--h <0<h<1>` : target mesh size (default `0.025`)
* `--dt <>0>`   : time step (default `5e-4`)
* `--c  <>0>`   : wave speed (default `1.0`)

Examples:

```bash
# coarser mesh, larger dt
/app/build/wave_solver --h 0.05 --dt 1e-3 --c 1.0

# finer mesh
/app/build/wave_solver --h 0.02 --dt 1.25e-4 --c 1.0
```

The program validates inputs and exits non‑zero if a value is invalid.

---

## Tests

We ship a tiny executable `run_tests` that:

* checks baseline accuracy via a measured L2 error parsed from solver output,
* checks that refinement improves the error,
* verifies invalid inputs are rejected (non‑zero exit),
* asserts a small run finishes under a few seconds (in `Release`).

Run inside Docker (writes `out*.txt` to `./outputs` on the host):

```bash
docker run --rm -v "$PWD/outputs:/outputs" -w /outputs fem-wave:1.0 /app/build/run_tests
```

> Internally, tests call the solver in the container at `/app/build/wave_solver`. In CMake, we set
> `BIN="${CMAKE_BINARY_DIR}/wave_solver"` so the test can locate the solver regardless of cwd.

---

## License

MIT (see `LICENSE`).

---

## Citation / Acknowledgements

* Built with [deal.II](https://www.dealii.org/).
* Sparse direct solves via `SparseDirectUMFPACK`.
