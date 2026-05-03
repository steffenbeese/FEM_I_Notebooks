"""JupyterDash app for fem1d.

Features:
- Load a matrix-style TOML file via file chooser (dcc.Upload)
- Visualize mesh + boundary conditions (no re-parsing needed)
- Solve (assemble + solve)
- Visualize solution (deformed mesh + result curve)

Run in a Jupyter notebook cell:

    from dashboard.jupyter_dash_app import app
    app.run_server(mode="inline", debug=True)

Or from terminal (opens external browser tab):

    pip install -e ".[dashboard]"
    python -c "from dashboard.jupyter_dash_app import app; app.run_server(mode='external', debug=True)"

"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, dash_table
from jupyter_dash import JupyterDash

from fem1d import Fem1dInputError, read_problem_toml_str, read_model_toml_str


# ----------------------------
# Helpers (upload / conversion)
# ----------------------------

def _decode_upload(contents: str) -> str:
    """Decode `dcc.Upload.contents` to UTF-8 text."""

    b64 = contents.split(",", 1)[1]
    data = base64.b64decode(b64)
    return data.decode("utf-8")


def _to_store_problem(problem) -> Dict[str, Any]:
    """Convert Fem1dProblem -> JSON-serializable dict for dcc.Store."""

    return {
        "model_type": problem.model_type,
        "coords": problem.coords.tolist(),
        "elements": problem.elements.tolist(),
        "dirichlet": {
            "nodes": list(problem.dirichlet_nodes),
            "dofs": list(problem.dirichlet_dofs),
            "vals": list(problem.dirichlet_vals),
        },
        "neumann": {
            "nodes": list(problem.neumann_nodes),
            "dofs": list(problem.neumann_dofs),
            "vals": list(problem.neumann_vals),
        },
    }


def _solve_to_store_solution(toml_text: str) -> Dict[str, Any]:
    fem = read_model_toml_str(toml_text)
    fem.assembleGlobalMatrix()
    fem.solveSystem()

    model_type = "stab" if fem.__class__.__name__.lower().startswith("stab") else "balken"

    out: Dict[str, Any] = {
        "model_type": model_type,
        "dof": np.asarray(fem.dof, dtype=float).tolist(),
        # Global nodal force vector (incl. reactions after solveSystem)
        "Fges": np.asarray(fem.Fges, dtype=float).tolist(),
    }

    if model_type == "stab":
        x, N = fem.computeNormalkraft(n=150)
        out["result"] = {
            "key": "N",
            "title": "Normalkraft N(x)",
            "x": np.asarray(x, dtype=float).tolist(),
            "y": np.asarray(N, dtype=float).tolist(),
            "y_label": "N",
        }
    else:
        xM, M = fem.computeMoment(n=250)
        out["result"] = {
            "key": "M",
            "title": "Moment M(x)",
            "x": np.asarray(xM, dtype=float).tolist(),
            "y": np.asarray(M, dtype=float).tolist(),
            "y_label": "M",
        }

    return out


# ----------------------------
# Plot helpers
# ----------------------------

def _fig_empty(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _extent(coords: np.ndarray) -> Tuple[float, float, float, float, float]:
    """(xmin, xmax, ymin, ymax, size)"""

    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))
    ymin = float(np.min(coords[:, 1]))
    ymax = float(np.max(coords[:, 1]))
    size = float(max(xmax - xmin, ymax - ymin, 1.0))
    return xmin, xmax, ymin, ymax, size


def _add_mesh_traces(fig: go.Figure, coords: np.ndarray, elements: np.ndarray, *, color: str, dash: str):
    for (n1, n2) in elements:
        fig.add_trace(
            go.Scatter(
                x=[coords[n1, 0], coords[n2, 0]],
                y=[coords[n1, 1], coords[n2, 1]],
                mode="lines",
                line=dict(color=color, width=3, dash=dash),
                showlegend=False,
            )
        )


def _add_nodes(
    fig: go.Figure,
    coords: np.ndarray,
    *,
    color: str = "black",
    symbol: str = "circle",
    size: int = 9,
    show_text: bool = True,
    name: Optional[str] = None,
    showlegend: bool = False,
):
    mode = "markers+text" if show_text else "markers"
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode=mode,
            text=[str(i) for i in range(coords.shape[0])] if show_text else None,
            textposition="top center",
            marker=dict(size=size, color=color, symbol=symbol),
            name=name,
            showlegend=showlegend,
        )
    )


def _fit_axes(fig: go.Figure, coords: np.ndarray, *, pad_frac: float = 0.15):
    """Set explicit axis ranges so the whole structure is visible."""

    xmin, xmax, ymin, ymax, size = _extent(coords)
    pad = float(pad_frac) * size

    fig.update_xaxes(range=[xmin - pad, xmax + pad], zeroline=False)
    fig.update_yaxes(range=[ymin - pad, ymax + pad], zeroline=False)


def _add_dirichlet_markers(fig: go.Figure, model_type: str, coords: np.ndarray, dirichlet: Dict[str, Any]):
    """Visual cue for Dirichlet BCs.

    - If only one dof fixed: triangle in direction
    - If both dofs fixed at same node: square
    """

    nodes = np.asarray(dirichlet.get("nodes", []), dtype=int)
    dofs = np.asarray(dirichlet.get("dofs", []), dtype=int)
    if nodes.size == 0:
        return

    # group constraints per node
    per_node: Dict[int, set] = {}
    for n, d in zip(nodes.tolist(), dofs.tolist()):
        per_node.setdefault(int(n), set()).add(int(d))

    # build 3 groups: dof0-only, dof1-only, both
    g0, g1, g01 = [], [], []
    for n, ds in per_node.items():
        if ds == {0}:
            g0.append(n)
        elif ds == {1}:
            g1.append(n)
        else:
            g01.append(n)

    def _scatter(ns: List[int], symbol: str, name: str):
        if not ns:
            return
        fig.add_trace(
            go.Scatter(
                x=coords[ns, 0],
                y=coords[ns, 1],
                mode="markers",
                marker=dict(size=14, color="red", symbol=symbol),
                name=name,
                showlegend=True,
            )
        )

    if model_type == "stab":
        _scatter(g0, "triangle-left", "Dirichlet ux")
        _scatter(g1, "triangle-down", "Dirichlet uy")
        _scatter(g01, "square", "Dirichlet ux,uy")
    else:
        # beam: dof0 = uy, dof1 = rotation
        _scatter(g0, "triangle-down", "Dirichlet uy")
        _scatter(g1, "circle-open", "Dirichlet phi")
        _scatter(g01, "square", "Dirichlet uy,phi")


def _add_neumann_annotations(fig: go.Figure, model_type: str, coords: np.ndarray, neumann: Dict[str, Any]):
    """Add arrow-like annotations for nodal loads.

    For beams, dof==0 is shown as vertical force arrow.
    Beam moments (dof==1) are shown as an 'M' label.
    """

    nodes = np.asarray(neumann.get("nodes", []), dtype=int)
    dofs = np.asarray(neumann.get("dofs", []), dtype=int)
    vals = np.asarray(neumann.get("vals", []), dtype=float)

    if nodes.size == 0:
        return

    *_bbox, size = _extent(coords)
    base = 0.15 * size

    # scale arrows by max magnitude (avoid huge arrows)
    maxabs = float(np.max(np.abs(vals))) if vals.size else 1.0
    maxabs = max(maxabs, 1e-12)

    for node, dof, val in zip(nodes.tolist(), dofs.tolist(), vals.tolist()):
        x0, y0 = float(coords[node, 0]), float(coords[node, 1])
        s = 1.0 if val >= 0 else -1.0
        rel = min(abs(val) / maxabs, 1.0)
        L = base * (0.4 + 0.6 * rel)

        if model_type == "stab":
            if int(dof) == 0:  # ux
                x1, y1 = x0 + s * L, y0
            else:  # uy
                x1, y1 = x0, y0 + s * L

            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor="green",
            )
        else:
            # beam
            if int(dof) == 0:  # transverse force
                x1, y1 = x0, y0 + s * L
                fig.add_annotation(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=3,
                    arrowcolor="green",
                )
            else:
                # moment/rotation as label
                fig.add_annotation(
                    x=x0,
                    y=y0,
                    text=f"M={val:g}",
                    showarrow=False,
                    font=dict(color="green"),
                    yshift=18,
                )


def fig_mesh_problem(problem_store: Dict[str, Any], *, title: str = "Netz + Randbedingungen") -> go.Figure:
    coords = np.asarray(problem_store["coords"], dtype=float)
    elements = np.asarray(problem_store["elements"], dtype=int)
    model_type = str(problem_store["model_type"])

    fig = go.Figure()
    _add_mesh_traces(fig, coords, elements, color="black", dash="solid")
    _add_nodes(fig, coords)
    _add_dirichlet_markers(fig, model_type, coords, problem_store.get("dirichlet", {}))
    _add_neumann_annotations(fig, model_type, coords, problem_store.get("neumann", {}))

    # Neumann arrows/text can extend beyond the node coordinates a bit.
    _fit_axes(fig, coords, pad_frac=0.25)

    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def fig_mesh_solution(
    problem_store: Dict[str, Any],
    solution_store: Dict[str, Any],
    *,
    scale: float = 1.0,
    title: str = "Netz (undeformed + deformed)",
) -> go.Figure:
    coords = np.asarray(problem_store["coords"], dtype=float)
    elements = np.asarray(problem_store["elements"], dtype=int)
    dof = np.asarray(solution_store["dof"], dtype=float)
    coords_def = coords + float(scale) * dof

    fig = go.Figure()
    _add_mesh_traces(fig, coords, elements, color="black", dash="solid")
    _add_mesh_traces(fig, coords_def, elements, color="#1f77b4", dash="dash")

    # Visualize nodes for both meshes
    _add_nodes(fig, coords, color="black", symbol="circle", size=8, show_text=True)
    _add_nodes(
        fig,
        coords_def,
        color="#1f77b4",
        symbol="circle-open",
        size=8,
        show_text=False,
        name="Knoten (deformiert)",
        showlegend=True,
    )

    # Always zoom out to include undeformed + deformed structure
    coords_all = np.vstack([coords, coords_def])
    _fit_axes(fig, coords_all, pad_frac=0.15)

    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def fig_result(solution_store: Dict[str, Any]) -> go.Figure:
    r = solution_store.get("result")
    if not r:
        return _fig_empty("Ergebnis")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["x"], y=r["y"], mode="lines", name=r.get("key", "result")))
    fig.update_layout(
        title=r.get("title", "Ergebnis"),
        xaxis=dict(title="x"),
        yaxis=dict(title=r.get("y_label", "")),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def div_dof_table(problem_store: Dict[str, Any], solution_store: Dict[str, Any]) -> html.Div:
    """Render a small table with nodal DOFs (displacements / rotations)."""

    coords = np.asarray(problem_store["coords"], dtype=float)
    dof = np.asarray(solution_store["dof"], dtype=float)

    # Forces may be missing in older cached store-solution values.
    Fges = np.asarray(solution_store.get("Fges", np.zeros_like(dof)), dtype=float)

    model_type = str(solution_store.get("model_type") or problem_store.get("model_type") or "")

    if model_type == "stab":
        dof_names = ("ux", "uy")
        force_names = ("Fx", "Fy")
    else:
        # beam: dof0=uy, dof1=phi
        dof_names = ("uy", "phi")
        force_names = ("F", "M")

    rows: List[Dict[str, Any]] = []
    n = int(coords.shape[0])
    for i in range(n):
        rows.append(
            {
                "node": i,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                dof_names[0]: float(dof[i, 0]),
                dof_names[1]: float(dof[i, 1]),
                force_names[0]: float(Fges[i, 0]),
                force_names[1]: float(Fges[i, 1]),
            }
        )

    columns = [
        {"name": "Knoten", "id": "node", "type": "numeric"},
        {"name": "x", "id": "x", "type": "numeric"},
        {"name": "y", "id": "y", "type": "numeric"},
        {"name": dof_names[0], "id": dof_names[0], "type": "numeric"},
        {"name": dof_names[1], "id": dof_names[1], "type": "numeric"},
        {"name": force_names[0], "id": force_names[0], "type": "numeric"},
        {"name": force_names[1], "id": force_names[1], "type": "numeric"},
    ]

    return html.Div(
        [
            html.H4("Knotenverschiebungen / Knotenkräfte"),
            dash_table.DataTable(
                data=rows,
                columns=columns,
                page_size=min(n, 30),
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "padding": "0.25rem 0.5rem",
                    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
                    "fontSize": "0.9rem",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "node"}, "textAlign": "right"},
                    {"if": {"column_id": "x"}, "textAlign": "right"},
                    {"if": {"column_id": "y"}, "textAlign": "right"},
                    {"if": {"column_id": dof_names[0]}, "textAlign": "right"},
                    {"if": {"column_id": dof_names[1]}, "textAlign": "right"},
                    {"if": {"column_id": force_names[0]}, "textAlign": "right"},
                    {"if": {"column_id": force_names[1]}, "textAlign": "right"},
                ],
            ),
        ],
        style={"marginTop": "0.5rem"},
    )


# ----------------------------
# Dash/JupyterDash app
# ----------------------------

app = JupyterDash(__name__)

app.layout = html.Div(
    [
        html.H3("1D FEM – Einführung in die FEM"),
        html.Div(
            [
                # Left column: controls
                html.Div(
                    [
                        dcc.Upload(
                            id="upload-toml",
                            children=html.Button("TOML laden (Dateiauswahl)", style={"width": "100%"}),
                            multiple=False,
                        ),
                        html.Div(id="status", style={"marginTop": "0.5rem"}),
                        html.Hr(),
                        html.Div(
                            [
                                html.Button("Netz + RB", id="btn-mesh", style={"width": "100%"}),
                                html.Button("Lösen", id="btn-solve", style={"width": "100%", "marginTop": "0.5rem"}),
                                html.Button(
                                    "Lösung darstellen",
                                    id="btn-solution",
                                    style={"width": "100%", "marginTop": "0.5rem"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Deformationsskala"),
                                dcc.Slider(id="scale", min=0.0, max=1000.0, step=10, value=1.0),
                            ],
                            style={"marginTop": "0.75rem"},
                        ),
                        # Stores can live anywhere in the layout
                        dcc.Store(id="store-toml"),
                        dcc.Store(id="store-problem"),
                        dcc.Store(id="store-solution"),
                        dcc.Store(id="store-view", data="problem"),  # "problem" | "solution"
                    ],
                    style={
                        "flex": "0 0 260px",
                        "paddingRight": "1rem",
                    },
                ),
                # Right column: plots
                html.Div(
                    [
                        dcc.Graph(id="fig-mesh"),
                        dcc.Graph(id="fig-result"),
                    ],
                    style={
                        "flex": "1 1 auto",
                        "minWidth": 0,
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "1rem",
                "alignItems": "flex-start",
            },
        ),
        html.Hr(),
        # Under everything: table
        html.Div(id="div-dof-table"),
    ],
    style={"padding": "0.75rem"},
)


@callback(
    Output("store-toml", "data"),
    Output("store-problem", "data"),
    Output("store-solution", "data"),
    Output("store-view", "data"),
    Output("status", "children"),
    Input("upload-toml", "contents"),
    State("upload-toml", "filename"),
)
def on_upload(contents: Optional[str], filename: Optional[str]):
    if not contents:
        return None, None, None, "problem", "Noch keine Datei geladen."

    try:
        toml_text = _decode_upload(contents)
        problem = read_problem_toml_str(toml_text)
    except Exception as e:
        return None, None, None, "problem", f"Fehler beim Laden/Parsen: {e}"

    name = filename or "<upload>"
    return (
        toml_text,
        _to_store_problem(problem),
        None,  # reset solution
        "problem",
        f"Geladen: {name} (model={problem.model_type})",
    )


@callback(
    Output("store-solution", "data", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("btn-solve", "n_clicks"),
    State("store-toml", "data"),
    prevent_initial_call=True,
)
def on_solve(_n, toml_text: Optional[str]):
    if not toml_text:
        return None, "Bitte erst TOML laden."

    try:
        sol = _solve_to_store_solution(toml_text)
    except Fem1dInputError as e:
        return None, f"Inputfehler: {e}"
    except Exception as e:
        return None, f"Solve-Fehler: {e}"

    return sol, "Gelöst. (Jetzt: Button 'Lösung')"


@callback(
    Output("store-view", "data", allow_duplicate=True),
    Input("btn-mesh", "n_clicks"),
    Input("btn-solution", "n_clicks"),
    prevent_initial_call=True,
)
def on_set_view(_n_mesh, _n_solution):
    if ctx.triggered_id == "btn-solution":
        return "solution"
    return "problem"


@callback(
    Output("fig-mesh", "figure"),
    Output("fig-result", "figure"),
    Output("div-dof-table", "children"),
    Output("status", "children", allow_duplicate=True),
    Input("store-problem", "data"),
    Input("store-solution", "data"),
    Input("store-view", "data"),
    Input("scale", "value"),
    State("status", "children"),
    prevent_initial_call=True,
)
def on_update_figures(
    problem_store: Optional[Dict[str, Any]],
    solution_store: Optional[Dict[str, Any]],
    view: str,
    scale: float,
    status: str,
):
    if not problem_store:
        return _fig_empty(""), _fig_empty(""), "", status

    if view == "solution":
        if not solution_store:
            return fig_mesh_problem(problem_store), _fig_empty(""), "", "Bitte erst lösen."
        return (
            fig_mesh_solution(problem_store, solution_store, scale=float(scale)),
            fig_result(solution_store),
            div_dof_table(problem_store, solution_store),
            status,
        )

    # default: show problem mesh + BCs
    return fig_mesh_problem(problem_store), _fig_empty(""), "", status
