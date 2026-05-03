import numpy as np


class FEM1D:
    """
    Abstrakte Basisklasse für 1D-Finite-Elemente (Stab- und Balkenelemente).

    Diese Klasse implementiert die gemeinsamen Methoden und Attribute für die
    Finite-Elemente-Analyse von 1D-Strukturen. Sie stellt die Grundfunktionalität
    für Knotenkoordinaten, Elementkonnektivität, Randbedingungen, externe Kräfte,
    den Aufbau der globalen Steifigkeitsmatrix und die Lösung des Gleichungssystems bereit.

    Attribute:
        numnp (int): Anzahl der Knoten.
        numel (int): Anzahl der Elemente.
        dim (int): Dimension des Problems (standardmäßig 2 für 2D-Probleme).
        nel (int): Anzahl der Knoten pro Element (standardmäßig 2).
        numdof (int): Gesamtanzahl der Freiheitsgrade.
        elements (np.ndarray): Array zur Speicherung der Elementverbindungen.
        eData (list): Liste zur Speicherung der Materialeigenschaften der Elemente.
        coords (np.ndarray): Array zur Speicherung der Knotenkoordinaten.
        dof (np.ndarray): Array zur Speicherung der Freiheitsgrade der Knoten.
        eqind (np.ndarray): Array zur Speicherung der Gleichungsindizes der Freiheitsgrade.
        Kges (np.ndarray): Globale Steifigkeitsmatrix.
        Fges (np.ndarray): Globaler Kraftvektor.
        dirichletBC (any): Dirichlet-Randbedingungen (Verschiebungen).
        Kuu, Kud, Kdu, Kdd (np.ndarray): Partitionierte Steifigkeitsmatrizen.
    """

    def __init__(self, numnp=2, numel=1):
        """
        Initialisiert die FEM1D-Basisklasse.

        Args:
            numnp (int): Anzahl der Knoten.
            numel (int): Anzahl der Elemente.
        """
        self.numnp = numnp
        self.numel = numel
        self.dim = 2
        self.nel = 2
        self.numdof = self.numnp * self.dim
        self.elements = np.zeros((self.numel, 2), dtype=int)
        self.eData = []
        self.coords = np.zeros((self.numnp, 2))
        self.dof = np.zeros((self.numnp, self.dim))
        self.eqind = np.zeros((self.numnp, self.dim), dtype=int)
        self.Kges = np.zeros((self.numdof, self.numdof))
        self.Fges = np.zeros((self.numnp, self.dim))
        self.dL = np.zeros((self.numnp, 1))  # Verteilte Lasten
        self.dirichletBC = None
        self.Kuu = self.Kud = self.Kdu = self.Kdd = None

    def setNodalCoordinates(self, coords):
        """
        Setzt die Knotenkoordinaten.

        Args:
            coords (np.ndarray): Array der Knotenkoordinaten.
        """
        self.coords[:, :] = coords[:, :]

    def setElementConnectivity(self, elements):
        """
        Setzt die Elementkonnektivität.

        Args:
            elements (np.ndarray): Array der Elementkonnektivität.
        """
        for i in range(self.numel):
            for j in range(self.nel):
                self.elements[i, j] = int(elements[i, j])

    def setDirichletBoundaryCondition(self, dirichletNodes, dirichletDir, dirichletVal):
        """
        Setzt die Dirichlet-Randbedingungen (Verschiebungen und Neigungen).

        Args:
            dirichletNodes (list): Liste der Knoten mit Dirichlet-Randbedingungen.
            dirichletDir (list): Liste der Dirichlet-Randbedingungen: 0 für Verschiebung, 1 für Neigung.
            dirichletVal (list): Liste der Werte der Dirichlet-Randbedingungen.
        """
        eqID = 1
        for nodeID in range(self.numnp):
            for dirID in range(self.dim):
                self.eqind[nodeID, dirID] = eqID
                eqID += 1
        # Knoten an denen Dirichlet Randbedingungen vorgegeben werden
        # müssen im Gleichungssystem nicht berücksichtigt werden.
        # Deshalb erhalten sie negative Gleichungsnummern zur Identifikation

        self.dirichletBC = np.zeros((len(dirichletNodes), 3))

        for i, (nodeID, dirId, dVal) in enumerate(zip(dirichletNodes, dirichletDir, dirichletVal)):
            self.eqind[nodeID, dirId] = -self.eqind[nodeID, dirId]
            self.dirichletBC[i, 0] = nodeID
            self.dirichletBC[i, 1] = dirId
            self.dirichletBC[i, 2] = dVal
            self.dof[nodeID, dirId] = dVal

    def setExternalForces(self, neumannNodes, neumannDir, neumannVal):
        """
        Setzt die Neumann-Randbedingungen (externe Kräfte).

        Args:
            neumannNodes (list): Liste der Knoten mit Neumann-Randbedingungen.
            neumannDir (list): Liste der Neumann-Randbedingungen: 0 = Kraft, 1 = Moment.
            neumannVal (list): Liste der Werte der Neumann-Randbedingungen.
        """
        for nodeID, nDir, nVal in zip(neumannNodes, neumannDir, neumannVal):
            self.Fges[nodeID, nDir] = nVal

    def resetFEM(self):
        """
        Setzt alle FEM-Matrizen und Freiheitsgrade zurück auf null.
        """
        self.dof = np.zeros((self.numnp, self.dim))
        self.eqind = np.zeros((self.numnp, self.dim), dtype=int)
        self.Kges = np.zeros((self.numdof, self.numdof))
        self.Fges = np.zeros((self.numnp, self.dim))

    def _getElementDirector(self, elID):
        """
        Berechnet den Richtungsvektor eines Elements.

        Args:
            elID (int): Element-ID.

        Returns:
            np.ndarray: Richtungsvektor des Elements.
        """
        nodeID1 = self.elements[elID, 0]
        nodeID2 = self.elements[elID, 1]
        director = self.coords[nodeID2, :] - self.coords[nodeID1, :]
        return director

    def setElementData(self, **data):
        """
        Setzt die Elementdaten. Subklassen können beliebige Properties übergeben.
        
        Parameter:
        **data: Beliebige Schlüssel-Wert-Paare für Elementdaten.
                Jeder Wert sollte eine Liste mit Länge numel sein.
        
        Beispiele:
        - StabFEM: setElementData(areas=[...], youngsmoduli=[...], lineLoads=[...])
        - BalkenFEM: setElementData(youngsmoduli=[...], sma=[...])
        """
        for i in range(self.numel):
            element_dict = {}
            for key, values in data.items():
                element_dict[key] = values[i]
            self.eData.append(element_dict)
            
    def setDistributedLoads(self, dloads):
        """
        Setzt die verteilten Lasten an den Knoten. So definiert sind keine Sprünge über Elementgrenzen hinweg möglich.

        Args:
            dloads (np.ndarray): Array der verteilten Lasten an den Knoten.
        """
        self.dL[:] = dloads[:]

    def _getElementstiffness_(self, elID):
        """
        Berechnet die Steifigkeitsmatrix eines Elements. Muss in Unterklassen implementiert werden.

        Args:
            elID (int): Element-ID.

        Raises:
            NotImplementedError: Muss in Unterklassen implementiert werden.
        """
        raise NotImplementedError("Muss in Unterklassen implementiert werden.")

    def assembleGlobalMatrix(self):
        """
        Baut die globale Steifigkeitsmatrix und den globalen Kraftvektor auf.
        Muss in Unterklassen implementiert werden.
        """
        self._assembleGlobalMatrix2D()

    def solveSystem(self):
        """
        Löst das Gleichungssystem.

        Partitioniert die globale Steifigkeitsmatrix in freie und vorgeschriebene
        Freiheitsgrade, löst nach den unbekannten Verschiebungen und berechnet
        die Reaktionskräfte.
        """
        # Erzeuge Hilfsmatrizen um die Freiheitsgrade zu sortieren
        numcdof = self.dirichletBC.shape[0]
        numfdof = self.numdof - numcdof
        self.Kuu = np.zeros((numfdof, numfdof))
        self.Kud = np.zeros((numfdof, numcdof))
        self.Kdd = np.zeros((numcdof, numcdof))
        Fu = np.zeros((numfdof, 1))
        Fc = np.zeros((numcdof, 1))
        uc = np.zeros((numcdof, 1))
        eqf_inv = np.zeros((numfdof, 2), dtype=int)
        eqc_inv = np.zeros((numcdof, 2), dtype=int)

        # Sortiere Freiheitsgrade
        icon = -1
        ifree = -1
        iIsFree = False
        for nodeI in range(self.numnp):
            for dirI in range(self.dim):
                jcon = -1
                jfree = -1
                jIsFree = False
                if self.eqind[nodeI, dirI] >= 0:
                    iIsFree = True
                    ifree += 1
                else:
                    iIsFree = False
                    icon += 1
                eqI = int(np.sign(self.eqind[nodeI, dirI]) * self.eqind[nodeI, dirI] - 1)
                if iIsFree:
                    eqf_inv[ifree, 0] = nodeI
                    eqf_inv[ifree, 1] = dirI
                    Fu[ifree] = self.Fges[nodeI, dirI]
                else:
                    eqc_inv[icon, 0] = nodeI
                    eqc_inv[icon, 1] = dirI
                    Fc[icon] = self.Fges[nodeI, dirI]
                    uc[icon] = self.dof[nodeI, dirI]
                for nodeJ in range(self.numnp):
                    for dirJ in range(self.dim):
                        if self.eqind[nodeJ, dirJ] >= 0:
                            jIsFree = True
                            jfree += 1
                        else:
                            jIsFree = False
                            jcon += 1

                        eqJ = int(np.sign(self.eqind[nodeJ, dirJ]) * self.eqind[nodeJ, dirJ] - 1)
                        if iIsFree and jIsFree:
                            self.Kuu[ifree, jfree] = self.Kges[eqI, eqJ]
                        if iIsFree and not jIsFree:
                            self.Kud[ifree, jcon] = self.Kges[eqI, eqJ]
                        if not iIsFree and not jIsFree:
                            self.Kdd[icon, jcon] = self.Kges[eqI, eqJ]

        # Löse Gleichungssystem
        # K_uu*u = F - K_ud*u_d
        rhs = Fu - self.Kud @ uc
        u = np.linalg.solve(self.Kuu, rhs)

        # Kopiere zu globalen Freiheitsgraden
        ieq = 0
        for nodeI in range(numfdof):
            self.dof[eqf_inv[nodeI, 0], eqf_inv[nodeI, 1]] = u[ieq, 0]
            ieq += 1

        # Bestimme Reaktionskräfte
        # R = K_du * u + K_dd*u_c - F_c
        rF = self.Kud.T @ u + self.Kdd @ uc - Fc

        # Kopiere zu den globalen Kräften
        ieq = 0
        for nodeI in range(numcdof):
            self.Fges[eqc_inv[nodeI, 0], eqc_inv[nodeI, 1]] = rF[ieq, 0]
            ieq += 1

    def _computeAlongStructure(self, callback, n=10):
        """
        Template-Methode zum Berechnen von Größen entlang der Struktur.
        
        Args:
            callback: Funktion(elID, xi, le, dofe) → Ergebniswert
            n: Anzahl der Stützpunkte pro Element
        
        Returns:
            X: Koordinaten
            Result: Ergebnisse
        """
        X = np.zeros(n * self.numel)
        Result = np.zeros(n * self.numel)
        
        for elID in range(self.numel):
            director = self._getElementDirector(elID)
            le = np.linalg.norm(director)
            node1 = self.elements[elID, 0]
            node2 = self.elements[elID, 1]
            x1 = self.coords[node1, 0]
            x2 = self.coords[node2, 0]
            X[elID*n:(elID+1)*n] = np.linspace(x1, x2, n)
            xilin = np.linspace(0, 1, n)
            dofe = np.array([ [self.dof[node1, 0], self.dof[node1, 1]],
                              [self.dof[node2, 0], self.dof[node2, 1]]
                               ])
            
            for i, xi in enumerate(xilin):
                Result[elID*n+i] = callback(elID, xi, le, dofe)
        
        return X, Result
    
    def _computeAtMaterialPoint(self, x, callback):
        """
        Berechnet einen Wert an einer bestimmten Position entlang der Struktur.
        
        Args:
            x (float): Position entlang der Struktur.
            callback: Funktion(elID, xi, le, dofe) → Ergebniswert
            
        Returns:
            float: Berechneter Wert an Position x.
        """
        for elID in range(self.numel):
            node1 = self.elements[elID, 0]
            node2 = self.elements[elID, 1]
            x1 = self.coords[node1, 0]
            x2 = self.coords[node2, 0]

            if (x >= x1 - 1.e-12) and (x < x2 + 1.e-12):
                director = self._getElementDirector(elID) 
                le = np.linalg.norm(director)
                dofe = np.array([self.dof[node1, 0], self.dof[node2, 0],
                               self.dof[node1, 1], self.dof[node2, 1]])
                xi = (x - x1) / (x2 - x1)
                
                return callback(elID, xi, le, dofe)
            
    def computeDisplacement(self, n=10):
        """
        Berechnet die Verschiebungen entlang der Struktur.
        
        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Verschiebungen (np.ndarray) entlang der Struktur.
        """
        def callback(elID, xi, le, dofe):
            N = self.shapeFunction(xi)
            return (N @ dofe.T) [1]
        
        return self._computeAlongStructure(callback, n)
            
    def getDisplacement(self, x):
        """
        Berechnet die Verschiebung an einer bestimmten Position x entlang der Struktur.

        Args:
            x (float): Position entlang der Struktur.

        Returns:
            float: Verschiebung an der Position x.
        """
        def callback(elID, xi, le, dofe):
            N = self.shapeFunction(xi)
            return N @ dofe.T
        
        return self._computeAtMaterialPoint(x, callback)

    def shapeFunction(self, xi):
        """
        Berechnet die Formfunktionen. Muss in Unterklassen implementiert werden.

        Args:
            xi (float): Normalisierte Koordinate.

        Raises:
            NotImplementedError: Muss in Unterklassen implementiert werden.
        """
        raise NotImplementedError("Muss in Unterklassen implementiert werden.")
