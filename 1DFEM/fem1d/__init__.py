"""
FEM1D - Finite-Element-Methode für 1D-Strukturen (Stäbe und Balken)

Dieses Paket implementiert die Finite-Elemente-Analyse für lineare 1D-Strukturen.
Es bietet zwei Hauptklassen:
- StabFEM: Für die Analyse von Stäben (Stabwerke)
- BalkenFEM: Für die Analyse von Balken (Balkenwerke)

Beispiel:
    >>> from fem1d import StabFEM, BalkenFEM
    >>> 
    >>> # Erstelle ein Stabwerk mit 2 Knoten und 1 Element
    >>> stab = StabFEM(numnp=2, numel=1)
"""

from .element1d import FEM1D
from .stabfem import StabFEM
from .balkenfem import BalkenFEM
from .reader import (
    Fem1dInputError,
    Fem1dProblem,
    build_model,
    read_model_toml,
    read_model_toml_bytes,
    read_model_toml_str,
    read_problem_toml,
    read_problem_toml_bytes,
    read_problem_toml_str,
)

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    "FEM1D",
    "StabFEM",
    "BalkenFEM",
    "Fem1dInputError",
    "Fem1dProblem",
    "build_model",
    "read_problem_toml",
    "read_problem_toml_str",
    "read_problem_toml_bytes",
    "read_model_toml",
    "read_model_toml_str",
    "read_model_toml_bytes",
]
