"""fem1d.reader

Reader utilities for fem1d problem definitions.

The primary goal of this module is to load a complete FEM1D problem from a
human-editable TOML file and build a ready-to-solve FEM model.

TOML parsing:
- Python >= 3.11: uses `tomllib` (stdlib)
- Python <  3.11: falls back to `tomli` (optional dependency)

Example TOML (matrix-style):

    [model]
    type = "stab"  # or "balken"

    [mesh]
    coords = [[0.0, 0.0], [10.0, 0.0]]
    elements = [[0, 1]]

    [element_data]
    area = [100.0]
    youngs_modulus = [210000.0]

    [bc]
    dirichlet = [[0, "ux", 0.0], [0, "uy", 0.0]]

    [loads]
    nodal = [[1, "ux", 1000.0]]

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union, Optional

import numpy as np

from .balkenfem import BalkenFEM
from .stabfem import StabFEM


PathLike = Union[str, Path]


class Fem1dInputError(ValueError):
    """Raised when a TOML input file is missing required keys or contains invalid data."""


def _toml_loads(text: str) -> Dict[str, Any]:
    """Parse TOML from a string (Python >=3.11: tomllib, else tomli)."""

    try:
        import tomllib  # Python 3.11+

        return tomllib.loads(text)
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Reading TOML requires Python >= 3.11 (tomllib) or the optional dependency 'tomli'"
            ) from e

        return tomli.loads(text)


def _load_toml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Read as text and parse via `loads()` so we can reuse the same parsing
    # code path for Dash uploads (string/bytes inputs).
    return _toml_loads(path.read_text(encoding="utf-8"))


def _as_2d_float_array(name: str, x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise Fem1dInputError(f"'{name}' must be a numeric 2D array-like") from e

    if arr.ndim != 2:
        raise Fem1dInputError(f"'{name}' must be 2-dimensional (got shape {arr.shape})")

    return arr


def _as_2d_int_array(name: str, x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=int)
    except Exception as e:
        raise Fem1dInputError(f"'{name}' must be an integer 2D array-like") from e

    if arr.ndim != 2:
        raise Fem1dInputError(f"'{name}' must be 2-dimensional (got shape {arr.shape})")

    return arr


def _as_1d_float_array(name: str, x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise Fem1dInputError(f"'{name}' must be a numeric 1D array-like") from e

    if arr.ndim != 1:
        raise Fem1dInputError(f"'{name}' must be 1-dimensional (got shape {arr.shape})")

    return arr


def _get(dct: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first existing key from `keys` in `dct`, else `default`."""

    for k in keys:
        if k in dct:
            return dct[k]
    return default


def _normalize_model_type(model_type: str) -> str:
    mt = str(model_type).strip().lower()
    aliases = {
        "stab": "stab",
        "bar": "stab",
        "truss": "stab",
        "balken": "balken",
        "beam": "balken",
    }
    if mt not in aliases:
        raise Fem1dInputError(f"Unsupported model.type '{model_type}'. Use 'stab' or 'balken'.")
    return aliases[mt]


def _dof_to_index(model_type: str, dof: Union[int, str]) -> int:
    if isinstance(dof, (int, np.integer)):
        dof_i = int(dof)
        if dof_i not in (0, 1):
            raise Fem1dInputError(f"DOF index must be 0 or 1 (got {dof_i}).")
        return dof_i

    d = str(dof).strip().lower()

    if model_type == "stab":
        mapping = {
            "ux": 0,
            "x": 0,
            "uy": 1,
            "y": 1,
        }
    else:  # "balken"
        # In this codebase a beam node has 2 dofs: (transversal displacement, rotation).
        mapping = {
            "uy": 0,
            "w": 0,
            "phi": 1,
            "rot": 1,
            "rotation": 1,
        }

    if d not in mapping:
        raise Fem1dInputError(
            f"Unknown DOF '{dof}' for model '{model_type}'. Allowed: {sorted(mapping.keys())} or 0/1."
        )

    return mapping[d]


def _parse_triplets(
    name: str,
    rows: Any,
    *,
    model_type: str,
) -> Tuple[List[int], List[int], List[float]]:
    """Parse [[node, dof, value], ...] into 3 lists."""

    if rows is None:
        return [], [], []

    try:
        seq = list(rows)
    except TypeError as e:
        raise Fem1dInputError(f"'{name}' must be a list of [node, dof, value] rows") from e

    nodes: List[int] = []
    dofs: List[int] = []
    vals: List[float] = []

    for i, r in enumerate(seq):
        if not isinstance(r, (list, tuple)) or len(r) != 3:
            raise Fem1dInputError(f"'{name}[{i}]' must be [node, dof, value]")
        node, dof, val = r
        try:
            node_i = int(node)
        except Exception as e:
            raise Fem1dInputError(f"'{name}[{i}][0]' must be an integer node id") from e
        dof_i = _dof_to_index(model_type, dof)
        try:
            val_f = float(val)
        except Exception as e:
            raise Fem1dInputError(f"'{name}[{i}][2]' must be a float") from e

        nodes.append(node_i)
        dofs.append(dof_i)
        vals.append(val_f)

    return nodes, dofs, vals


@dataclass(frozen=True)
class Fem1dProblem:
    """Parsed fem1d problem data (independent of FEM classes)."""

    model_type: str
    coords: np.ndarray  # (numnp, 2)
    elements: np.ndarray  # (numel, 2)
    element_data: Dict[str, np.ndarray]  # each is (numel,)

    dirichlet_nodes: List[int]
    dirichlet_dofs: List[int]
    dirichlet_vals: List[float]

    neumann_nodes: List[int]
    neumann_dofs: List[int]
    neumann_vals: List[float]

    distributed_nodal: Optional[np.ndarray] = None  # (numnp,)


def _parse_problem_dict(raw: Mapping[str, Any]) -> Fem1dProblem:
    """Parse and validate an already-loaded TOML dict into a :class:`Fem1dProblem`."""

    model = raw.get("model", {})
    if not isinstance(model, Mapping):
        raise Fem1dInputError("[model] must be a table")

    model_type = _normalize_model_type(_get(model, "type", "model", default=None))

    mesh = raw.get("mesh", {})
    if not isinstance(mesh, Mapping):
        raise Fem1dInputError("[mesh] must be a table")

    coords = _as_2d_float_array("mesh.coords", mesh.get("coords"))
    if coords.shape[1] != 2:
        raise Fem1dInputError(f"mesh.coords must have 2 columns (x,y). Got shape {coords.shape}.")

    elements = _as_2d_int_array("mesh.elements", mesh.get("elements"))
    if elements.shape[1] != 2:
        raise Fem1dInputError(f"mesh.elements must have 2 columns (node1,node2). Got {elements.shape}.")

    numnp = coords.shape[0]
    numel = elements.shape[0]

    if np.any(elements < 0) or np.any(elements >= numnp):
        raise Fem1dInputError(f"mesh.elements contains node indices outside [0, {numnp-1}].")

    # Element data
    e = raw.get("element_data", {})
    if not isinstance(e, Mapping):
        raise Fem1dInputError("[element_data] must be a table")

    element_data: Dict[str, np.ndarray] = {}

    def _require_numel_vector(key: str, arr: np.ndarray) -> None:
        if arr.shape[0] != numel:
            raise Fem1dInputError(
                f"element_data.{key} must have length numel={numel} (got {arr.shape[0]})."
            )

    if model_type == "stab":
        area = _get(e, "area", "areas")
        E = _get(e, "youngs_modulus", "young_modulus", "youngsmodulus", "youngsmoduli", "E")
        if area is None or E is None:
            raise Fem1dInputError(
                "For model.type='stab' you must provide element_data.area and element_data.youngs_modulus"
            )
        area_v = _as_1d_float_array("element_data.area", area)
        E_v = _as_1d_float_array("element_data.youngs_modulus", E)
        _require_numel_vector("area", area_v)
        _require_numel_vector("youngs_modulus", E_v)
        element_data["area"] = area_v
        element_data["youngs_modulus"] = E_v

    elif model_type == "balken":
        E = _get(e, "youngs_modulus", "young_modulus", "youngsmodulus", "youngsmoduli", "E")
        I = _get(e, "second_moment_area", "sma", "I", "inertia")
        if E is None or I is None:
            raise Fem1dInputError(
                "For model.type='balken' you must provide element_data.youngs_modulus and "
                "element_data.second_moment_area"
            )
        E_v = _as_1d_float_array("element_data.youngs_modulus", E)
        I_v = _as_1d_float_array("element_data.second_moment_area", I)
        _require_numel_vector("youngs_modulus", E_v)
        _require_numel_vector("second_moment_area", I_v)
        element_data["youngs_modulus"] = E_v
        element_data["second_moment_area"] = I_v

    # BCs and loads: allow either [bc] or [boundary_conditions]
    bc = raw.get("bc", raw.get("boundary_conditions", {}))
    if not isinstance(bc, Mapping):
        raise Fem1dInputError("[bc] must be a table")

    dir_rows = _get(bc, "dirichlet", "dir")
    dir_nodes, dir_dofs, dir_vals = _parse_triplets("bc.dirichlet", dir_rows, model_type=model_type)
    if len(dir_nodes) == 0:
        raise Fem1dInputError("At least one Dirichlet boundary condition is required (bc.dirichlet).")

    for node in dir_nodes:
        if node < 0 or node >= numnp:
            raise Fem1dInputError(f"Dirichlet node index {node} out of range [0, {numnp-1}].")

    loads = raw.get("loads", raw.get("load", {}))
    if not isinstance(loads, Mapping):
        raise Fem1dInputError("[loads] must be a table")

    neu_rows = _get(loads, "nodal", "neumann")
    neu_nodes, neu_dofs, neu_vals = _parse_triplets("loads.nodal", neu_rows, model_type=model_type)

    for node in neu_nodes:
        if node < 0 or node >= numnp:
            raise Fem1dInputError(f"Load node index {node} out of range [0, {numnp-1}].")

    # Distributed nodal loads (optional): length numnp
    dist = _get(loads, "distributed_nodal", "distributed")
    distributed_nodal: Optional[np.ndarray]
    if dist is None:
        distributed_nodal = None
    else:
        q = _as_1d_float_array("loads.distributed_nodal", dist)
        if q.shape[0] != numnp:
            raise Fem1dInputError(
                f"loads.distributed_nodal must have length numnp={numnp} (got {q.shape[0]})."
            )
        distributed_nodal = q

    return Fem1dProblem(
        model_type=model_type,
        coords=coords,
        elements=elements,
        element_data=element_data,
        dirichlet_nodes=dir_nodes,
        dirichlet_dofs=dir_dofs,
        dirichlet_vals=dir_vals,
        neumann_nodes=neu_nodes,
        neumann_dofs=neu_dofs,
        neumann_vals=neu_vals,
        distributed_nodal=distributed_nodal,
    )


def read_problem_toml(path: PathLike) -> Fem1dProblem:
    """Read and validate a fem1d TOML problem file."""

    return _parse_problem_dict(_load_toml(path))


def read_problem_toml_str(text: str) -> Fem1dProblem:
    """Read and validate a fem1d TOML problem from a TOML string."""

    return _parse_problem_dict(_toml_loads(text))


def read_problem_toml_bytes(data: bytes, *, encoding: str = "utf-8") -> Fem1dProblem:
    """Read and validate a fem1d TOML problem from bytes (e.g. Dash upload)."""

    return read_problem_toml_str(data.decode(encoding))


def build_model(problem: Fem1dProblem) -> Union[StabFEM, BalkenFEM]:
    """Build a :class:`~fem1d.stabfem.StabFEM` or :class:`~fem1d.balkenfem.BalkenFEM` from a Fem1dProblem."""

    numnp = int(problem.coords.shape[0])
    numel = int(problem.elements.shape[0])

    if problem.model_type == "stab":
        fem: Union[StabFEM, BalkenFEM] = StabFEM(numnp=numnp, numel=numel)
        fem.setNodalCoordinates(problem.coords)
        fem.setElementConnectivity(problem.elements)
        fem.setElementData(
            areas=problem.element_data["area"].tolist(),
            youngsmoduli=problem.element_data["youngs_modulus"].tolist(),
        )

    elif problem.model_type == "balken":
        fem = BalkenFEM(numnp=numnp, numel=numel)
        fem.setNodalCoordinates(problem.coords)
        fem.setElementConnectivity(problem.elements)
        fem.setElementData(
            youngsmoduli=problem.element_data["youngs_modulus"].tolist(),
            sma=problem.element_data["second_moment_area"].tolist(),
        )

    else:  # pragma: no cover
        raise Fem1dInputError(f"Unsupported model_type '{problem.model_type}'")

    fem.setDirichletBoundaryCondition(
        problem.dirichlet_nodes,
        problem.dirichlet_dofs,
        problem.dirichlet_vals,
    )

    if len(problem.neumann_nodes) > 0:
        fem.setExternalForces(problem.neumann_nodes, problem.neumann_dofs, problem.neumann_vals)

    if problem.distributed_nodal is not None:
        fem.setDistributedLoads(problem.distributed_nodal.reshape((numnp, 1)))

    return fem


def read_model_toml(path: PathLike) -> Union[StabFEM, BalkenFEM]:
    """Convenience function: read TOML file and directly return a ready FEM model."""

    return build_model(read_problem_toml(path))


def read_model_toml_str(text: str) -> Union[StabFEM, BalkenFEM]:
    """Convenience function: read TOML string and directly return a ready FEM model."""

    return build_model(read_problem_toml_str(text))


def read_model_toml_bytes(data: bytes, *, encoding: str = "utf-8") -> Union[StabFEM, BalkenFEM]:
    """Convenience function: read TOML bytes and directly return a ready FEM model."""

    return build_model(read_problem_toml_bytes(data, encoding=encoding))
