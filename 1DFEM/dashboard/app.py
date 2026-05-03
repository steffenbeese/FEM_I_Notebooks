"""Dash dashboard for fem1d.

Run (from repository root):

    pip install -e ".[dashboard]"
    python -m dashboard.app

Then open the printed local URL in your browser.

The dashboard supports:
- Upload of a matrix-style TOML problem
- Solve
- Plot mesh (undeformed + deformed)
- Plot a selected result quantity

"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dash import Dash, Input, Output, State, callback, dcc, html

from fem1d import Fem1dInputError, read_model_toml_str, read_problem_toml_str
from fem1d.viz_plotly import line_figure, mesh_figure


def _decode_upload(contents: str) -> str:
    """Decode `dcc.Upload.contents` to UTF-8 text."""

    # contents: "data:<mime>;base64,<payload>"
    b64 = contents.split(",", 1)[1]
    data = base64.b64decode(b64)
    return data.decode("utf-8")


def _as_store_problem(problem) -> Dict[str, Any]:
    """Convert Fem1dProblem to JSON-serializable dict."""

    return {
        "model_type": problem.model_type,
        "coords": problem.coords.tolist(),
        "elements": problem.elements.tolist(),
        "element_data": {k: v.tolist() for k, v in problem.element_data.items()},
        "dirichlet": [problem.dirichlet_nodes, problem.dirichlet_dofs, problem.dirichlet_vals],
        "neumann": [problem.neumann_nodes, problem.neumann_dofs, problem.neumann_vals],
        "distributed_nodal": None if problem.distributed_nodal is None else problem.distributed_nodal.tolist(),
    }


def _solve_to_store(toml_text: str) -> Dict[str, Any]:
    fem = read_model_toml_str(toml_text)

    fem.assembleGlobalMatrix()
    fem.solveSystem()

    out: Dict[str, Any] = {
        "model_type": "stab" if fem.__class__.__name__.lower().startswith("stab") else "balken",
        "coords": np.asarray(fem.coords, dtype=float).tolist(),
        "elements": np.asarray(fem.elements, dtype=int).tolist(),
        "dof": np.asarray(fem.dof, dtype=float).tolist(),
    }

    if out["model_type"] == "stab":
        x, N = fem.computeNormalkraft(n=100)
        out["results"] = {
            "N": {"x": np.asarray(x, dtype=float).tolist(), "y": np.asarray(N, dtype=float).tolist()},
        }
    else:
        xM, M = fem.computeMoment(n=200)
        xQ, Q = fem.computeQuerkraft(n=200)
        out["results"] = {
            "M": {"x": np.asarray(xM, dtype=float).tolist(), "y": np.asarray(M, dtype=float).tolist()},
            "Q": {"x": np.asarray(xQ, dtype=float).tolist(), "y": np.asarray(Q, dtype=float).tolist()},
        }

    return out


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("fem1d Dashboard"),
        html.Div(
            [
                dcc.Upload(
                    id="upload-toml",
                    children=html.Button("TOML Problem hochladen"),
                    multiple=False,
                ),
                html.Div(id="upload-status", style={"marginTop": "0.5rem"}),
                html.Hr(),
                html.Label("Deformationsskala"),
                dcc.Slider(id="scale", min=0.0, max=50.0, step=0.5, value=10.0),
                html.Br(),
                html.Button("Solve", id="btn-solve"),
                html.Div(id="solve-status", style={"marginTop": "0.5rem"}),
                html.Hr(),
                html.Label("Ergebnis"),
                dcc.Dropdown(id="quantity", options=[], value=None, clearable=False),
                dcc.Store(id="store-toml"),
                dcc.Store(id="store-problem"),
                dcc.Store(id="store-solution"),
            ],
            style={"width": "25%", "display": "inline-block", "verticalAlign": "top"},
        ),
        html.Div(
            [
                dcc.Graph(id="fig-mesh"),
                dcc.Graph(id="fig-result"),
            ],
            style={"width": "72%", "display": "inline-block", "paddingLeft": "1rem"},
        ),
    ]
)


@callback(
    Output("store-toml", "data"),
    Output("store-problem", "data"),
    Output("upload-status", "children"),
    Input("upload-toml", "contents"),
    State("upload-toml", "filename"),
)
def on_upload(contents: Optional[str], filename: Optional[str]):
    if not contents:
        return None, None, ""

    try:
        text = _decode_upload(contents)
        problem = read_problem_toml_str(text)
    except Exception as e:
        return None, None, f"Upload/Parse Fehler: {e}"

    name = filename or "<upload>"
    return text, _as_store_problem(problem), f"Geladen: {name} (model={problem.model_type})"


@callback(
    Output("store-solution", "data"),
    Output("solve-status", "children"),
    Input("btn-solve", "n_clicks"),
    State("store-toml", "data"),
    prevent_initial_call=True,
)
def on_solve(_, toml_text: Optional[str]):
    if not toml_text:
        return None, "Kein Problem geladen."

    try:
        sol = _solve_to_store(toml_text)
    except Fem1dInputError as e:
        return None, f"Inputfehler: {e}"
    except Exception as e:
        return None, f"Solve Fehler: {e}"

    return sol, "Solve OK"


@callback(
    Output("quantity", "options"),
    Output("quantity", "value"),
    Input("store-problem", "data"),
)
def update_quantity_options(problem: Optional[Dict[str, Any]]):
    if not problem:
        return [], None

    if problem.get("model_type") == "stab":
        opts = [
            {"label": "Normalkraft N(x)", "value": "N"},
        ]
        return opts, "N"

    opts = [
        {"label": "Moment M(x)", "value": "M"},
        {"label": "Querkraft Q(x)", "value": "Q"},
    ]
    return opts, "M"


@callback(
    Output("fig-mesh", "figure"),
    Input("store-problem", "data"),
    Input("store-solution", "data"),
    Input("scale", "value"),
)
def update_mesh(problem: Optional[Dict[str, Any]], sol: Optional[Dict[str, Any]], scale: float):
    if not problem:
        # empty figure
        return mesh_figure(np.zeros((0, 2)), np.zeros((0, 2), dtype=int), title="Mesh")

    coords = np.asarray(problem["coords"], dtype=float)
    elements = np.asarray(problem["elements"], dtype=int)

    coords_def = None
    if sol and sol.get("dof") is not None:
        dof = np.asarray(sol["dof"], dtype=float)
        coords_def = coords + float(scale) * dof

    return mesh_figure(coords, elements, coords_deformed=coords_def, title="Mesh")


@callback(
    Output("fig-result", "figure"),
    Input("store-solution", "data"),
    Input("quantity", "value"),
)
def update_result(sol: Optional[Dict[str, Any]], quantity: Optional[str]):
    if not sol or not quantity:
        return line_figure(np.zeros(0), np.zeros(0), title="Result", y_label="")

    r = sol.get("results", {}).get(quantity)
    if not r:
        return line_figure(np.zeros(0), np.zeros(0), title="Result", y_label="")

    x = np.asarray(r["x"], dtype=float)
    y = np.asarray(r["y"], dtype=float)

    if quantity == "N":
        return line_figure(x, y, title="Normalkraft N(x)", y_label="N")
    if quantity == "M":
        return line_figure(x, y, title="Moment M(x)", y_label="M")
    if quantity == "Q":
        return line_figure(x, y, title="Querkraft Q(x)", y_label="Q")

    return line_figure(x, y, title=f"Result {quantity}", y_label=quantity)


def main() -> None:
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
