# Thermo-Mechanical Digital Twin
### POD-ROM + Kalman Filter Data Assimilation for Coupled Thermo-Mechanical Systems

> Real-time full-field reconstruction of temperature and displacement in a 2D fin from sparse sensor measurements, with simultaneous online identification of unknown thermal conductivity — implemented as a two-container Docker pipeline with an interactive Streamlit dashboard.

---

## Overview

This project implements a complete **data-driven digital twin** for a 2D coupled thermo-mechanical system (aluminium fin geometry). The pipeline combines:

- A **Full-Order Model (FOM)** using FEniCSx finite element analysis to generate high-fidelity simulation data
- A **Proper Orthogonal Decomposition Reduced Order Model (POD-ROM)** with non-intrusive Operator Inference for ~100× speedup
- A **Linear Kalman Filter (LKF)** for real-time full-field state reconstruction from 8 sparse sensors
- An **Ensemble Kalman Filter (EnKF)** for simultaneous state estimation and online thermal conductivity identification
- An interactive **Streamlit dashboard** for visualisation of all results

---

## Physics

The system solves two coupled PDEs in a staggered scheme.

**Thermal — transient heat conduction (parabolic PDE):**

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q_0 \quad \text{in } \Omega$$

with a Dirichlet condition $T = T_{\rm hot}$ on the left boundary and a Robin convection condition $k \nabla T \cdot \mathbf{n} = -h(T - T_\infty)$ on the top surface.

**Mechanical — quasi-static linear thermoelasticity (elliptic PDE):**

$$\nabla \cdot \boldsymbol{\sigma} = 0, \quad \boldsymbol{\sigma} = \mathbb{C} : \left(\boldsymbol{\varepsilon} - \alpha(T - T_{\rm ref})\mathbf{I}\right)$$

with the bottom edge clamped ($\mathbf{u} = \mathbf{0}$) and all other boundaries traction-free.

**Staggered coupling:** At each time step, the thermal PDE is solved first to obtain $T^{n+1}$, which is then passed as a known field to the mechanical solver.

---

## Methodology

```
Full-Order Model (FEniCSx)
  └── Latin Hypercube Sampling over (k, h, Q₀) parameter space
  └── Staggered thermo-mechanical time-stepping (backward Euler)
  └── Outputs: T_snapshots (N_T × N_steps), u_snapshots (N_u × N_steps)

POD-ROM (NumPy / Operator Inference)
  └── Truncated SVD independently on thermal and mechanical snapshot matrices
  └── Non-intrusive Operator Inference → identifies reduced operators A_r, b_r from data
  └── Coupling matrix K_cu maps thermal reduced coords to mechanical reduced coords
  └── ~100× speedup over FOM with <5% reconstruction error

Data Assimilation (Kalman Filtering)
  └── LKF  : reconstructs full T and u fields from 5 temperature + 3 strain sensors
  └── EnKF : augments state with log(k) for simultaneous field reconstruction
             and online thermal conductivity identification
  └── Uncertainty quantification via posterior covariance (LKF) and ensemble spread (EnKF)

Dashboard (Streamlit + Plotly)
  └── Interactive 2D heatmaps: FOM vs ROM+KF side by side
  └── Time-resolved reconstruction error plots (log scale)
  └── Online k-identification convergence
  └── POD singular value decay showing model compressibility
```

---

## Key Results

| Metric | Value |
|---|---|
| Thermal POD modes required (99.99% energy) | 2 |
| Mechanical POD modes required | 3 |
| ROM speedup over FOM | ~100× |
| Temperature reconstruction error (LKF) | ~3% |
| Displacement reconstruction error | ~5% |
| Sensors used | 5 temperature + 3 strain |
| Uncertainty quantification | Posterior covariance (LKF) + Ensemble spread (EnKF) |

---

## Architecture

Two Docker containers share a single named volume. The simulation container runs once to generate snapshots, exits cleanly, and the data-driven container reads from the shared volume to train the ROM and launch the dashboard.

```
┌──────────────────────────────┐     shared volume        ┌─────────────────────────────────┐
│   Container 1: FEniCSx       │ ────────────────────────► │   Container 2: Data-driven      │
│                              │   /workspace/data/        │                                 │
│   fom_solver.py              │   snapshots/              │   pod_rom.py                    │
│   run_parametric.py          │   ──────────────────      │   kalman_filter.py              │
│   config.py                  │   T_snapshots_train.npy   │   train_rom.py                  │
│                              │   u_snapshots_train.npy   │   dashboard.py                  │
│   Base: dolfinx/dolfinx:v0.7 │   params.npy              │   config.py                     │
│   Generates FEM snapshots    │   metadata.json           │                                 │
│   → exits after completing   │                           │   Base: python:3.11-slim        │
└──────────────────────────────┘                           │   Trains ROM + runs Streamlit   │
                                                           └─────────────────────────────────┘
```

---

## Project Structure

```
thermo-mechanical-data-assimilation/
├── docker-compose.yml
├── simulation/
│   ├── Dockerfile               # FEniCSx container definition
│   ├── config.py                # shared physical parameters
│   ├── fom_solver.py            # coupled FEM solver (thermal + mechanical)
│   └── run_parametric.py        # LHS parametric sweep orchestrator
└── data_driven/
    ├── Dockerfile               # Python slim container definition
    ├── requirements.txt
    ├── config.py                # shared physical parameters
    ├── pod_rom.py               # dual-field POD basis + Operator Inference
    ├── kalman_filter.py         # LKF + augmented EnKF
    ├── train_rom.py             # training orchestrator
    └── dashboard.py             # Streamlit interactive dashboard
```

---

## Setup and Run

### Prerequisites
- Windows with WSL2 enabled
- Docker Desktop installed and running
- At least 8 GB RAM allocated to Docker
- At least 10 GB free disk space

### Step 1 — Clone the repository
```bash
git clone https://github.com/anish_hilary/thermo-mechanical-data-assimilation
cd thermo-mechanical-data-assimilation
```

### Step 2 — Build both Docker images
```bash
docker compose build
```
Downloads `dolfinx/dolfinx:v0.7.3` (~5 GB) and `python:3.11-slim`. Takes 10–20 minutes on first run.

### Step 3 — Run the simulation container
```bash
docker compose run simulation
```
Runs parametric FEM simulations over the (k, h, Q₀) parameter space using Latin Hypercube Sampling. Writes snapshot matrices to the shared Docker volume. To run a quick test with fewer samples, edit `config.py`:
```python
N_SAMPLES_TRAIN = 5    # default
N_SAMPLES_TEST  = 3
```

### Step 4 — Run the data-driven pipeline and launch dashboard
```bash
docker compose up data_driven
```
This sequentially builds the POD-ROM, runs both Kalman filters, and launches the Streamlit dashboard.

### Step 5 — Open the dashboard
```
http://localhost:8501
```

---

## Parameter Space

| Parameter | Symbol | Range | Units |
|---|---|---|---|
| Thermal conductivity | k | 50 – 200 | W/mK |
| Convection coefficient | h | 5 – 80 | W/m²K |
| Heat source amplitude | Q₀ | 10⁴ – 5×10⁵ | W/m³ |

---

## Uncertainty Quantification

The project includes two levels of UQ:

**State UQ (LKF):** The posterior covariance matrix $\mathbf{P}$ is propagated through the POD basis to give pointwise standard deviations on the reconstructed temperature field at every node and every time step.

**Parameter UQ (EnKF):** The ensemble spread over the augmented state $[\mathbf{q}_T; \log k]$ provides a mean and standard deviation on the identified thermal conductivity at every assimilation step.

---

## References

- Peherstorfer & Willcox (2016) — *Data-driven operator inference for nonintrusive projection-based model reduction* — Computer Methods in Applied Mechanics and Engineering
- Benner, Gugercin & Willcox (2015) — *A survey of projection-based model reduction methods for parametric dynamical systems* — SIAM Review
- Evensen (2003) — *The Ensemble Kalman Filter: theoretical formulation and practical implementation* — Ocean Dynamics
- Hesthaven, Rozza & Stamm (2016) — *Certified Reduced Basis Methods for Parametrized Partial Differential Equations* — Springer

---

## Author

**Anish Hilary Ignatius**  
M.Sc. Systems Engineering and Engineering Management  
Research Assistant, Institute of System Dynamics (ISD), University of Stuttgart  

