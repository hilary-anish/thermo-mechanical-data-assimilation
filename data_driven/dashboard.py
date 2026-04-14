"""
dashboard.py
============
Streamlit interactive dashboard for the thermo-mechanical digital twin.

Panels
------
1. Sidebar        : parameter sliders + filter selector
2. Temperature    : 2D heatmap of reconstructed T field (LKF or EnKF)
3. Displacement   : 2D heatmap of |u| field
4. Kalman metrics : error over time + uncertainty bands
5. k-identification: EnKF convergence of thermal conductivity

Run locally:
    streamlit run dashboard.py

Deploy to HuggingFace Spaces:
    - Create a new Space, SDK = Streamlit
    - Upload all files in data_driven/ and requirements.txt
    - The data/ folder must be present (commit snapshot + rom outputs)
"""

import numpy as np
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────
SNAP_DIR = "/workspace/data/snapshots"
ROM_DIR  = "/workspace/data/rom"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermo-Mechanical Digital Twin",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Load all data (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_all():
    data = {}
    try:
        with open(f"{SNAP_DIR}/metadata.json") as f:
            data["meta"] = json.load(f)
        with open(f"{ROM_DIR}/rom_metadata.json") as f:
            data["rom_meta"] = json.load(f)
        with open(f"{ROM_DIR}/results_meta.json") as f:
            data["res_meta"] = json.load(f)

        for key, path in [
            ("coords",     f"{SNAP_DIR}/dof_coords.npy"),
            ("T_true",     f"{ROM_DIR}/T_true.npy"),
            ("u_true",     f"{ROM_DIR}/u_true.npy"),
            ("T_lkf",      f"{ROM_DIR}/T_lkf.npy"),
            ("u_lkf",      f"{ROM_DIR}/u_lkf.npy"),
            ("T_enkf",     f"{ROM_DIR}/T_enkf.npy"),
            ("u_enkf",     f"{ROM_DIR}/u_enkf.npy"),
            ("err_T_lkf",  f"{ROM_DIR}/err_T_lkf.npy"),
            ("err_u_lkf",  f"{ROM_DIR}/err_u_lkf.npy"),
            ("err_T_enkf", f"{ROM_DIR}/err_T_enkf.npy"),
            ("k_hist",     f"{ROM_DIR}/k_hist.npy"),
            ("t_axis",     f"{ROM_DIR}/t_axis.npy"),
            ("T_sensor",   f"{ROM_DIR}/T_sensor_nodes.npy"),
            ("sigma_T",    f"{ROM_DIR}/sigma_T.npy"),
            ("sigma_u",    f"{ROM_DIR}/sigma_u.npy"),
        ]:
            data[key] = np.load(path)

        data["loaded"] = True
    except FileNotFoundError as e:
        data["loaded"]  = False
        data["error"]   = str(e)
    return data


D = load_all()


# ── Helpers ────────────────────────────────────────────────────────────────

def field_to_grid(field_flat, nx, ny):
    """Reshape flat DOF array to 2D grid for plotting."""
    n_nodes = (nx + 1) * (ny + 1)
    return field_flat[:n_nodes].reshape((ny + 1, nx + 1))


def u_magnitude_grid(u_flat, nx, ny):
    n_nodes = (nx + 1) * (ny + 1)
    ux = u_flat[0::2][:n_nodes].reshape((ny + 1, nx + 1))
    uy = u_flat[1::2][:n_nodes].reshape((ny + 1, nx + 1))
    return np.sqrt(ux**2 + uy**2) * 1e6   # µm


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌡️ Digital Twin Controls")
    st.markdown("---")

    if D["loaded"]:
        meta     = D["meta"]
        rom_meta = D["rom_meta"]
        res_meta = D["res_meta"]

        st.subheader("Time step")
        t_axis   = D["t_axis"]
        n_steps  = len(t_axis)
        step_idx = st.slider(
            "Select time step",
            min_value=0, max_value=n_steps - 1,
            value=n_steps - 1, step=1,
            format="t = %d"
        )
        st.caption(f"t = {t_axis[step_idx]:.1f} s")

        st.markdown("---")
        st.subheader("Filter")
        filter_choice = st.radio(
            "Data assimilation method",
            ["Linear Kalman Filter (LKF)",
             "Ensemble Kalman Filter (EnKF)"],
            index=1
        )

        st.markdown("---")
        st.subheader("ROM info")
        st.metric("Thermal modes r_T",  rom_meta["r_T"])
        st.metric("Mechanical modes r_u", rom_meta["r_u"])
        st.metric("T sensor nodes",  res_meta["n_T_sensors"])
        st.metric("Strain sensor nodes", res_meta["n_u_sensors"])

        st.markdown("---")
        st.subheader("Key results")
        st.metric(
            "Max T reconstruction error",
            f"{res_meta['max_err_T_lkf_pct']:.2f}%"
        )
        st.metric(
            "True thermal conductivity k",
            f"{res_meta['k_true']:.1f} W/mK"
        )
        st.metric(
            "Identified k (EnKF final)",
            f"{res_meta['k_final_estimate']:.2f} W/mK"
        )
    else:
        st.error("Data not loaded.")
        st.code(D.get("error", "Unknown error"))


# ── Main page ──────────────────────────────────────────────────────────────

st.title("Thermo-Mechanical Digital Twin")
st.caption(
    "POD-ROM + Kalman Filter data assimilation | "
    "Coupled heat conduction & thermoelasticity | "
    "Anish Hilary Ignatius"
)

if not D["loaded"]:
    st.error(
        "Snapshot and ROM data not found. "
        "Run `python3 train_rom.py` inside the ML container first."
    )
    st.stop()

meta  = D["meta"]
NX    = meta["geometry"]["NX"]
NY    = meta["geometry"]["NY"]
L_geo = meta["geometry"]["L"]
H_geo = meta["geometry"]["H"]

# choose fields based on filter selection
if "LKF" in filter_choice:
    T_est = D["T_lkf"]
    u_est = D["u_lkf"]
    err_T = D["err_T_lkf"]
    err_u = D["err_u_lkf"]
    tag   = "LKF"
else:
    T_est = D["T_enkf"]
    u_est = D["u_enkf"]
    err_T = D["err_T_enkf"]
    err_u = D["err_u_lkf"]
    tag   = "EnKF"

x_1d = np.linspace(0, L_geo, NX + 1)
y_1d = np.linspace(0, H_geo, NY + 1)

# ── Row 1: Field heatmaps ──────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Temperature field ({tag}) — t = {t_axis[step_idx]:.1f}s")

    T_grid_true = field_to_grid(D["T_true"][:, step_idx], NX, NY)
    T_grid_est  = field_to_grid(T_est[:, step_idx],        NX, NY)

    fig_T = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FOM (ground truth)", f"ROM + {tag} reconstruction"]
    )
    fig_T.add_trace(
        go.Heatmap(
            z=T_grid_true, x=x_1d, y=y_1d,
            colorscale="Hot",
            zmin=280, zmax=620,
            colorbar=dict(title="T [K]", len=0.5, x=0.45),
            showscale=True
        ), row=1, col=1
    )
    fig_T.add_trace(
        go.Heatmap(
            z=T_grid_est, x=x_1d, y=y_1d,
            colorscale="Hot",
            zmin=280, zmax=620,
            colorbar=dict(title="T [K]", len=0.5, x=1.0),
            showscale=True
        ), row=1, col=2
    )

    # overlay sensor markers
    coords = D["coords"]
    T_nodes = D["T_sensor"].astype(int)
    sensor_x = coords[T_nodes, 0]
    sensor_y = coords[T_nodes, 1]
    for col_idx in [1, 2]:
        fig_T.add_trace(
            go.Scatter(
                x=sensor_x, y=sensor_y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10,
                            color="cyan", line=dict(width=1, color="black")),
                name="T sensors",
                showlegend=(col_idx == 1)
            ), row=1, col=col_idx
        )

    fig_T.update_layout(
        height=240, margin=dict(t=40, b=10, l=10, r=10),
        yaxis_scaleanchor="x", yaxis2_scaleanchor="x2"
    )
    st.plotly_chart(fig_T, use_container_width=True)

with col2:
    st.subheader(f"Displacement magnitude ({tag}) — t = {t_axis[step_idx]:.1f}s")

    u_grid_true = u_magnitude_grid(D["u_true"][:, step_idx], NX, NY)
    u_grid_est  = u_magnitude_grid(u_est[:, step_idx],        NX, NY)
    u_max       = max(u_grid_true.max(), 1e-12)

    fig_u = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FOM (ground truth)", f"ROM + {tag} reconstruction"]
    )
    for i, (grid, show) in enumerate(
        [(u_grid_true, True), (u_grid_est, False)], start=1
    ):
        fig_u.add_trace(
            go.Heatmap(
                z=grid, x=x_1d, y=y_1d,
                colorscale="Viridis",
                zmin=0, zmax=u_max,
                colorbar=dict(
                    title="|u| [µm]", len=0.5,
                    x=0.45 if i == 1 else 1.0
                ),
                showscale=show
            ), row=1, col=i
        )

    fig_u.update_layout(
        height=240, margin=dict(t=40, b=10, l=10, r=10),
        yaxis_scaleanchor="x", yaxis2_scaleanchor="x2"
    )
    st.plotly_chart(fig_u, use_container_width=True)

# ── Row 2: Error and k-identification ─────────────────────────────────────

st.markdown("---")
col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("Temperature reconstruction error")
    fig_eT = go.Figure()
    fig_eT.add_trace(go.Scatter(
        x=t_axis, y=err_T * 100,
        mode="lines", name=f"T error ({tag})",
        line=dict(color="tomato", width=2)
    ))
    fig_eT.add_vline(
        x=t_axis[step_idx], line_dash="dash",
        line_color="gray", annotation_text="current t"
    )
    fig_eT.update_layout(
        height=280,
        xaxis_title="t [s]",
        yaxis_title="Relative L2 error [%]",
        yaxis_type="log",
        margin=dict(t=20, b=30, l=40, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_eT, use_container_width=True)
    st.metric(
        "Error at selected step",
        f"{err_T[step_idx]*100:.3f}%"
    )

with col4:
    st.subheader("Displacement reconstruction error")
    fig_eu = go.Figure()
    fig_eu.add_trace(go.Scatter(
        x=t_axis, y=err_u * 100,
        mode="lines", name=f"u error ({tag})",
        line=dict(color="steelblue", width=2)
    ))
    fig_eu.add_vline(
        x=t_axis[step_idx], line_dash="dash",
        line_color="gray"
    )
    fig_eu.update_layout(
        height=280,
        xaxis_title="t [s]",
        yaxis_title="Relative L2 error [%]",
        yaxis_type="log",
        margin=dict(t=20, b=30, l=40, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_eu, use_container_width=True)
    st.metric(
        "Error at selected step",
        f"{err_u[step_idx]*100:.3f}%"
    )

with col5:
    st.subheader("Online k identification (EnKF)")
    k_hist  = D["k_hist"]
    k_true  = D["res_meta"]["k_true"]
    fig_k   = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=t_axis, y=k_hist,
        mode="lines+markers", name="EnKF estimate",
        line=dict(color="seagreen", width=2),
        marker=dict(size=4)
    ))
    fig_k.add_hline(
        y=k_true, line_dash="dash",
        line_color="black", line_width=1.5,
        annotation_text=f"True k = {k_true:.1f}",
        annotation_position="bottom right"
    )
    fig_k.add_vline(
        x=t_axis[step_idx], line_dash="dash",
        line_color="gray"
    )
    fig_k.update_layout(
        height=280,
        xaxis_title="t [s]",
        yaxis_title="k [W/mK]",
        margin=dict(t=20, b=30, l=40, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_k, use_container_width=True)
    st.metric(
        "k error at final step",
        f"{abs(k_hist[-1]-k_true)/k_true*100:.2f}%"
    )

# ── Row 3: SVD decay ──────────────────────────────────────────────────────

st.markdown("---")
st.subheader("POD singular value decay — model compressibility")
col6, col7 = st.columns(2)

for col_obj, sigma, name, col_color in [
    (col6, D["sigma_T"], "Thermal field (T)",      "crimson"),
    (col7, D["sigma_u"], "Mechanical field (u)",   "royalblue"),
]:
    with col_obj:
        r      = np.arange(1, len(sigma) + 1)
        energy = np.cumsum(sigma**2) / np.sum(sigma**2) * 100

        fig_sv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sv.add_trace(
            go.Scatter(
                x=r, y=sigma / sigma[0],
                mode="lines", name="σᵢ/σ₁",
                line=dict(color=col_color, width=2)
            ), secondary_y=False
        )
        fig_sv.add_trace(
            go.Scatter(
                x=r, y=energy,
                mode="lines", name="Cumulative energy [%]",
                line=dict(color="gray", width=1.5, dash="dash")
            ), secondary_y=True
        )
        fig_sv.update_yaxes(
            title_text="Normalised σᵢ/σ₁",
            type="log", secondary_y=False
        )
        fig_sv.update_yaxes(
            title_text="Cumulative energy [%]",
            secondary_y=True
        )
        fig_sv.update_layout(
            title=name,
            xaxis_title="Mode index r",
            height=300,
            margin=dict(t=40, b=30, l=50, r=50),
            legend=dict(x=0.5, y=0.5)
        )
        st.plotly_chart(fig_sv, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Thermo-Mechanical Digital Twin — "
    "POD-ROM + Kalman Filter Data Assimilation | "
    "FEniCSx · NumPy · Streamlit · Plotly | "
    "Anish Hilary Ignatius · Stuttgart</small>",
    unsafe_allow_html=True
)
