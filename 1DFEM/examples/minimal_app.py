from __future__ import annotations

from dash import Dash, Input, Output, dcc, html
import numpy as np

from fem1d import read_model_toml_str
from fem1d.viz_plotly import mesh_figure, line_figure

# Minimal problem (TOML as string)
PROBLEM_TOML = """
[model]
type = "stab"

[mesh]
coords = [
  [0.0, 0.0],
  [10.0, 0.0],
]
elements = [
  [0, 1],
]

[element_data]
area = [100.0]
youngs_modulus = [210000.0]

[bc]
dirichlet = [
  [0, "ux", 0.0],
  [0, "uy", 0.0],
  [1,"uy",1.0],
]

[loads]
nodal = [
  [1, "ux", 1000.0],
]
"""

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H3("fem1d + Dash (Minimal)"),
        html.Button("Solve", id="btn-solve"),
        dcc.Slider(id="scale", min=0, max=50, step=1, value=10),
        dcc.Graph(id="fig-mesh"),
        dcc.Graph(id="fig-N"),
    ]
)


@app.callback(
    Output("fig-mesh", "figure"),
    Output("fig-N", "figure"),
    Input("btn-solve", "n_clicks"),
    Input("scale", "value"),
    prevent_initial_call=False,
)
def solve_and_plot(_n_clicks, scale):
    fem = read_model_toml_str(PROBLEM_TOML)
    fem.assembleGlobalMatrix()
    fem.solveSystem()

    coords = np.asarray(fem.coords, dtype=float)
    elements = np.asarray(fem.elements, dtype=int)
    coords_def = coords + float(scale) * np.asarray(fem.dof, dtype=float)

    fig_mesh = mesh_figure(coords, elements, coords_deformed=coords_def, title="Mesh (deformed)")
    x, N = fem.computeNormalkraft(n=50)
    fig_N = line_figure(x, N, title="Normalkraft N(x)", y_label="N")

    return fig_mesh, fig_N


if __name__ == "__main__":
    app.run(debug=True)