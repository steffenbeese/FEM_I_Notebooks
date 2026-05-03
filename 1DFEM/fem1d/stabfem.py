import numpy as np
import matplotlib.pyplot as plt
from .element1d import FEM1D


class StabFEM(FEM1D):
    """
    Klasse zur Durchführung von Finite-Elemente-Analysen für lineare Stabtragwerke im 2D-Raum.
    
    Diese Klasse implementiert die FEM für Stabwerke unter Verwendung von linearen
    Formfunktionen und kann Normalkräfte sowie Verformungen berechnen.
    """
    
    def __init__(self, numnp=2, numel=1):
        """
        Initialisiert eine neue Instanz der StabFEM-Klasse.

        Parameter:
        numnp (int): Anzahl der Knotenpunkte im Modell.
        numel (int): Anzahl der Elemente im Modell.
        """
        super().__init__(numnp, numel)
        
    def shapeFunction(self, xi):
        """
        Berechnet die Formfunktionen (Shape Functions) für ein Stabelement.

        Args:
            xi (float): Normalisierte Koordinate (0 <= xi <= 1).

        Returns:
            np.ndarray: Array der Formfunktionen an der Position xi.
        """
        return np.array([1 - xi, xi])

    def setElementData(self, areas, youngsmoduli):
        """
        Setzt die Elementdaten wie Querschnittsflächen, Elastizitätsmoduln.

        Parameter:
        areas (list): Liste der Querschnittsflächen für jedes Element.
        youngsmoduli (list): Liste der Elastizitätsmodul-Werte für jedes Element.
        
        Elementdata-Felder:
        - area: Querschnittsfläche des Elements
        - youngsmodulus: Elastizitätsmodul des Elements
        """
        super().setElementData(area=areas, youngsmodulus=youngsmoduli)
            
    def _getElementstiffness_(self, elID):
        """
        Berechnet die Steifigkeitsmatrix und den Elementvektor der verteilten Last eines Elements.

        Parameter:
        elID (int): Die ID des Elements.

        Rückgabewert:
        tuple: Steifigkeitsmatrix (np.ndarray) und Kraftvektor Fq (np.ndarray) des Elements.
        """
        director = self._getElementDirector(elID) 
        le = np.linalg.norm(director) 
        Ke = self.eData[elID]["area"] * self.eData[elID]["youngsmodulus"] / le * np.ones((2, 2))
        Ke[0, 1] *= -1
        Ke[1, 0] *= -1
        
        # Äquivalente Knotenkräfte
        Fq = np.zeros((2, 1))
        node1 = self.elements[elID, 0]
        node2 = self.elements[elID, 1]
        n1 = self.dL[node1]
        n2 = self.dL[node2]

        Fq[0] = le * (2*n1 + n2) / 6
        Fq[1] = le * (n1 + 2*n2) / 6
        
        return Ke, Fq

    def _assembleGlobalMatrix2D(self):
        """
        Assemblierung der globalen Steifigkeitsmatrix im 2D-Raum.
        """
        for elID in range(self.numel):
            Ke, F_Q = self._getElementstiffness_(elID)
            Ke_enhanced = np.zeros((4, 4))
            Ke_enhanced[0, 0] = Ke[0, 0]
            Ke_enhanced[0, 2] = Ke[0, 1]
            Ke_enhanced[2, 0] = Ke[1, 0]
            Ke_enhanced[2, 2] = Ke[1, 1]
            R = self._getTransFormationMatrix(elID)
            Ke = R.T @ Ke_enhanced @ R
            Fe = np.zeros((4, 1))
            Fe[0, 0] = F_Q[0,0]
            Fe[2, 0] = F_Q[1,0]
            Fe = (R.T @ Fe)
            for nodeI in range(self.nel):
                globalNodeI = self.elements[elID, nodeI]
                for dimI in range(self.dim):
                    rowID = int(np.sign(self.eqind[globalNodeI, dimI]) * self.eqind[globalNodeI, dimI] - 1)
                    self.Fges[globalNodeI, dimI] += Fe[(nodeI * self.dim) + dimI, 0]
                    for nodeJ in range(self.nel):
                        globalNodeJ = self.elements[elID, nodeJ]
                        for dimJ in range(self.dim):
                            colID = int(np.sign(self.eqind[globalNodeJ, dimJ]) * self.eqind[globalNodeJ, dimJ] - 1)
                            self.Kges[rowID, colID] += Ke[(nodeI * self.dim) + dimI, (nodeJ * self.dim) + dimJ]

    def _getTransFormationMatrix(self, elID):
        """
        Berechnet die Transformationsmatrix eines Elements.

        Parameter:
        elID (int): Die ID des Elements.

        Rückgabewert:
        np.ndarray: Transformationsmatrix des Elements.
        """
        director = self._getElementDirector(elID)
        le = np.linalg.norm(director)
        director = director / le
        normal = np.array([[-director[1]], [director[0]]])
        E1 = np.zeros((2, 2))
        E1[0, 0] = 1.0
        E1[1, 1] = 1.0
        Euser = np.zeros((2, 2))
        Euser[0, 0] = director[0]
        Euser[1, 0] = director[1]
        Euser[0, 1] = normal[0]
        Euser[1, 1] = normal[1]
        Rot = Euser.T @ E1.T
        
        R = np.array([[Rot[0, 0], Rot[0, 1], 0, 0], 
                      [Rot[1, 0], Rot[1, 1], 0, 0], 
                      [0, 0, Rot[0, 0], Rot[0, 1]], 
                      [0, 0, Rot[1, 0], Rot[1, 1]]])
        return R

    def computeNormalkraft(self, n=10):
        """
        Berechnet die Normalkräfte entlang der Stabstruktur.

        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Normalkräfte (np.ndarray) entlang der Stabstruktur.
        """
        def callback(elID, xi, le, dofe):
            R = self._getTransFormationMatrix(elID)
            EA = self.eData[elID]["area"] * self.eData[elID]["youngsmodulus"]
            dofl = (R @ dofe.reshape(-1, 1))
            B = 1.0 / le * np.array([-1, 0, 1, 0])
            return EA * B @ dofl
       
        return self._computeAlongStructure(callback, n)
    
    def computeDisplacement(self,n=10):
        
        def callback(elID,xi,le,dofe):
            N = self.shapeFunction(xi)
            u1 = N[0] * dofe[0,0] + N[1] * dofe[1,0]
            u2 = N[0] * dofe[0,1] + N[1] * dofe[1,1]
            return np.sign(u1)*np.sqrt(u1**2+u2**2)
        
        return self._computeAlongStructure(callback, n)
    
    def getDisplacement(self,X):
        
        def callback(elID,xi,le,dofe):
            N = self.shapeFunction(xi)
            u1 = N[0] * dofe[0,0] + N[1] * dofe[1,0]
            u2 = N[0] * dofe[0,1] + N[1] * dofe[1,1]
            return np.sign(u1)*np.sqrt(u1**2+u2**2)
        
        return self._computeAtMaterialPoint(X,callback)

    def plotMesh(self, deformed=False, scale=1.0, ax=None, fig=None):
        """
        Plottet das Netz des Modells.

        Parameter:
        deformed (bool): Gibt an, ob das deformierte Netz geplottet werden soll.
        scale (float): Skalierungsfaktor für die Darstellung.
        ax (matplotlib.axes.Axes): Achsenobjekt für die Darstellung.
        fig (matplotlib.figure.Figure): Figur-Objekt für die Darstellung.

        Rückgabewert:
        tuple: Figur und Achsenobjekt (matplotlib.figure.Figure, matplotlib.axes.Axes).
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        X = self.coords
        Fnp = self.Fges
        F = np.zeros((self.numnp, 2))
        for n in range(self.numnp):
            for d in range(self.dim):
                F[n, d] = Fnp[n, d]
        
        maxscale = float(np.max(np.max(X)))
        minscale = float(np.min(np.min(X)))
        ax.scatter(X[:, 0], X[:, 1], color="black", s=100)
        for i in range(self.numel):
            x1 = X[self.elements[i, 0], :]
            x2 = X[self.elements[i, 1], :]
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], color="black", linewidth=5)
        
        if deformed:
            u = self.dof
            Xd = np.zeros((self.numnp, 2))
            for dimI in range(self.dim):
                Xd[:, dimI] = X[:, dimI] + scale * u[:, dimI]
            ax.scatter(Xd[:, 0], Xd[:, 1], color="blue", s=100)
            for i in range(self.numel):
                x1 = Xd[self.elements[i, 0], :]
                x2 = Xd[self.elements[i, 1], :]
                ax.plot([x1[0], x2[0]], [x1[1], x2[1]], color="blue", linewidth=5, linestyle="--")
            maxscale = float(np.max([maxscale, np.max(np.max(Xd))]))
            minscale = float(np.min([minscale, np.min(np.min(Xd))]))

        ax.quiver(X[:, 0], X[:, 1], F[:, 0], F[:, 1], color="red")
        ax.grid(True)
        dx = (maxscale - minscale) * 0.1
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(minscale - dx, maxscale + dx)
        ax.set_ylim(minscale - dx, maxscale + dx)

        return fig, ax
