"""fem1d.viz_plotly

Plotly-based visualization helpers for use in Dash.

The goal is to keep plotting separate from the FEM core and to avoid a hard
runtime dependency on Plotly for users that only need the solver.

To use these helpers install the optional dependencies:

    pip install -e ".[dashboard]"

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore

        return go
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotly is required for fem1d.viz_plotly. Install optional deps: pip install -e '.[dashboard]'"
        ) from e


def mesh_figure(
    coords: np.ndarray,
    elements: np.ndarray,
    *,
    coords_deformed: Optional[np.ndarray] = None,
    title: str = "Mesh",
) -> "object":
    """Create a Plotly figure for the undeformed/deformed mesh.

    Args:
        coords: (numnp, 2)
        elements: (numel, 2) integer node indices
        coords_deformed: optional (numnp, 2)
    """

    go = _require_plotly()
    coords = np.asarray(coords, dtype=float)
    elements = np.asarray(elements, dtype=int)

    fig = go.Figure()

    def _add_lines(c: np.ndarray, *, name: str, color: str, dash: str = "solid") -> None:
        # Each element is a polyline with 2 points.
        for (n1, n2) in elements:
            fig.add_trace(
                go.Scatter(
                    x=[c[n1, 0], c[n2, 0]],
                    y=[c[n1, 1], c[n2, 1]],
                    mode="lines+markers",
                    line=dict(color=color, dash=dash, width=3),
                    marker=dict(size=8, color=color),
                    name=name,
                    showlegend=False,
                )
            )

    _add_lines(coords, name="undeformed", color="black", dash="solid")

    if coords_deformed is not None:
        coords_deformed = np.asarray(coords_deformed, dtype=float)
        _add_lines(coords_deformed, name="deformed", color="#1f77b4", dash="dash")

    fig.update_layout(
        title=title,
        xaxis=dict(title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def mesh_figure_from_model(
    fem,
    *,
    deformed: bool = True,
    scale: float = 1.0,
    title: str = "Mesh",
) -> "object":
    """Create a mesh figure directly from a FEM model instance."""

    coords_def = None
    if deformed:
        coords_def = np.asarray(fem.coords, dtype=float) + float(scale) * np.asarray(fem.dof, dtype=float)
    return mesh_figure(np.asarray(fem.coords), np.asarray(fem.elements), coords_deformed=coords_def, title=title)


def line_figure(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    y_label: str,
    x_label: str = "x",
) -> "object":
    """Create a simple x-y line plot."""

    go = _require_plotly()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=y_label))
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
