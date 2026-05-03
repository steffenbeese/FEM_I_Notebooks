# FEM1D - Finite-Element-Methode für 1D-Strukturen

Ein Python-Paket zur Finite-Elemente-Analyse (FEM) von eindimensionalen Strukturen wie Stäben und Balken im 2D-Raum.

## Features

- **StabFEM**: Analyse von Stabwerken mit linearen Elementen
  - Berechnung von Normalkräften
  - Verformungsberechnung
  - 2D-Raumrahmen-Unterstützung

- **BalkenFEM**: Analyse von Balkenwerken mit kubischen Elementen
  - Berechnung von Biegemoment und Querkraft
  - Verformungsberechnung
  - Unterstützung für verteilte Lasten

- **Kernfunktionalität**:
  - Aufbau der globalen Steifigkeitsmatrix
  - Lösung von Gleichungssystemen mit Randbedingungen
  - Berechnung von Reaktionskräften
  - Template-Methoden für beliebige Berechnungen entlang der Struktur

## Installation

### Über pip (lokal)

```bash
# Im Projektverzeichnis
pip install -e .
```

### Direktes Installieren von Abhängigkeiten

```bash
pip install -r requirements.txt
```

### Mit Entwicklungs-Tools

```bash
pip install -r requirements-dev.txt
```

## Abhängigkeiten

- **numpy** >= 1.20.0 - Numerische Berechnungen
- **matplotlib** >= 3.4.0 - Visualisierung (nur für StabFEM)

### Optionale Entwicklungs-Abhängigkeiten

- pytest - Unit-Tests
- black - Code-Formatierung
- flake8 - Code-Qualitätsprüfung

## Schnellstart

### StabFEM - Stabwerk-Beispiel

```python
import numpy as np
from fem1d import StabFEM

# Erstelle ein Stabwerk mit 2 Knoten und 1 Element
stab = StabFEM(numnp=2, numel=1)

# Definiere Knotenkoordinaten
coords = np.array([[0.0, 0.0],
                   [10.0, 0.0]])
stab.setNodalCoordinates(coords)

# Definiere Elementverbindung
elements = np.array([[0, 1]], dtype=int)
stab.setElementConnectivity(elements)

# Setze Materialeigenschaften
areas = [100.0]  # Querschnittsfläche in mm²
youngsmoduli = [210000.0]  # Elastizitätsmodul in N/mm²
stab.setElementData(areas=areas, youngsmoduli=youngsmoduli)

# Definiere Randbedingungen (Knoten 0 eingespannt)
dirichletNodes = [0]
dirichletDir = [0, 1]  # beide Richtungen
dirichletVal = [0.0, 0.0]
stab.setDirichletBoundaryCondition(dirichletNodes, dirichletDir, dirichletVal)

# Definiere externe Lasten
neumannNodes = [1]
neumannDir = [0]  # horizontale Kraft
neumannVal = [1000.0]  # 1000 N
stab.setExternalForces(neumannNodes, neumannDir, neumannVal)

# Assembliere und löse
stab.assembleGlobalMatrix()
stab.solveSystem()

# Berechne Ergebnisse
x, N = stab.computeNormalkraft(n=20)

# Visualisiere
fig, ax = stab.plotMesh(deformed=True, scale=10.0)
```

### BalkenFEM - Balken-Beispiel

```python
import numpy as np
from fem1d import BalkenFEM

# Erstelle einen Balken mit 3 Knoten und 2 Elementen
balken = BalkenFEM(numnp=3, numel=2)

# Definiere Knotenkoordinaten
coords = np.array([[0.0, 0.0],
                   [5.0, 0.0],
                   [10.0, 0.0]])
balken.setNodalCoordinates(coords)

# Definiere Elementverbindung
elements = np.array([[0, 1],
                     [1, 2]], dtype=int)
balken.setElementConnectivity(elements)

# Setze Materialeigenschaften
youngsmoduli = [210000.0, 210000.0]  # E in N/mm²
sma = [1000.0, 1000.0]  # Flächenträgheitsmoment in mm⁴
balken.setElementData(youngsmoduli=youngsmoduli, sma=sma)

# Definiere Randbedingungen (beidseitig eingespannt)
dirichletNodes = [0, 2]
dirichletDir = [0, 1, 0, 1]
dirichletVal = [0.0, 0.0, 0.0, 0.0]
balken.setDirichletBoundaryCondition(dirichletNodes, dirichletDir, dirichletVal)

# Definiere externe Last in Knoten 1
neumannNodes = [1]
neumannDir = [0]  # Kraft in y-Richtung
neumannVal = [-1000.0]  # 1000 N nach unten
balken.setExternalForces(neumannNodes, neumannDir, neumannVal)

# Assembliere und löse
balken.assembleGlobalMatrix()
balken.solveSystem()

# Berechne Ergebnisse
x_M, M = balken.computeMoment(n=50)
x_Q, Q = balken.computeQuerkraft(n=50)

# Verformung an Position x=5.0
u = balken.getDisplacement(5.0)
```

## Paketstruktur

```
fem1d/
├── __init__.py           # Paket-Initialisierung
├── element1d.py          # Basisklasse FEM1D
├── stabfem.py            # StabFEM-Klasse
└── balkenfem.py          # BalkenFEM-Klasse

setup.py                  # Setup-Skript für Installation
requirements.txt          # Produktions-Abhängigkeiten
requirements-dev.txt      # Entwicklungs-Abhängigkeiten
README.md                 # Diese Datei
```

## Dokumentation der Hauptklassen

### FEM1D (Basisklasse)

**Methoden:**
- `setNodalCoordinates(coords)` - Setzt Knotenkoordinaten
- `setElementConnectivity(elements)` - Setzt Elementverbindungen
- `setDirichletBoundaryCondition(nodes, dir, val)` - Setzt Verschiebungs-RB
- `setExternalForces(nodes, dir, val)` - Setzt externe Kräfte
- `setElementData(**data)` - Setzt Elementdaten
- `setDistributedLoads(dloads)` - Setzt verteilte Lasten
- `assembleGlobalMatrix()` - Assembliert globale Steifigkeitsmatrix
- `solveSystem()` - Löst das Gleichungssystem
- `computeDisplacement(n)` - Berechnet Verformungen entlang der Struktur
- `getDisplacement(x)` - Holt Verformung an Position x

### StabFEM

**Zusätzliche Methoden:**
- `computeNormalkraft(n)` - Berechnet Normalkräfte entlang des Stabes
- `plotMesh(deformed, scale, ax, fig)` - Visualisiert das Netz

### BalkenFEM

**Zusätzliche Methoden:**
- `computeMoment(n)` - Berechnet Biegemomente
- `computeQuerkraft(n)` - Berechnet Querkräfte

## Randbedingungen

### Dirichlet-Randbedingungen (Verschiebungen)

Werden verwendet, um Verschiebungen und Neigungen an Knoten vorzuschreiben:

```python
dirichletNodes = [0, 2]      # Knoten-Indizes
dirichletDir = [0, 1, 0, 1]  # Richtungen (0=vertikal, 1=horizontal/Rotation)
dirichletVal = [0, 0, 0, 0]  # Werte der Verschiebungen
fem.setDirichletBoundaryCondition(dirichletNodes, dirichletDir, dirichletVal)
```

### Neumann-Randbedingungen (Kräfte)

Werden verwendet, um externe Kräfte und Momente vorzuschreiben:

```python
neumannNodes = [1]        # Knoten-Indizes
neumannDir = [0]          # Richtungen
neumannVal = [1000.0]     # Kraftwerte
fem.setExternalForces(neumannNodes, neumannDir, neumannVal)
```

## Beispiele

Siehe Verzeichnis `examples/` für vollständige Beispiele:
- `example_stab.py` - Einfaches Stabwerk
- `example_balken.py` - Einfacher Balken

## Lizenz

MIT License - Siehe LICENSE Datei für Details

## Kontakt

Für Fragen und Feedback: your.email@example.com

## Changelog

### Version 1.0.0 (2024)
- Initial Release
- StabFEM für Stabwerke
- BalkenFEM für Balkenwerke
- Grundlegende FEM-Funktionalität
