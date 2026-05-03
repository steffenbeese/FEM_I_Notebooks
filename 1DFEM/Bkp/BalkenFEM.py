import numpy as np
from Element1D import FEM1D


class BalkenFEM(FEM1D):
    """
    Eine Klasse zur Finite-Elemente-Methode (FEM) für Balkenstrukturen.

    Diese Klasse erweitert Element1D und implementiert die Finite-Elemente-Methode
    für die Analyse von Balkenstrukturen. Sie unterstützt die Definition von Knoten,
    Elementen, Materialeigenschaften, Randbedingungen und externen Kräften.
    Die Klasse kann die globale Steifigkeitsmatrix aufbauen, das Gleichungssystem
    lösen und die Resultate wie Verformungen, Momente und Querkräfte berechnen.

    Zusätzliche Attribute:
        qL (np.ndarray): Verteilte Lasten an den Knoten.
        u (np.ndarray): Verformungen der Knoten.
        neumannBC (any): Neumann-Randbedingungen (externe Kräfte).
    """

    def __init__(self, numnp=2, numel=1):
        """
        Initialisiert die BalkenFEM-Klasse.

        Args:
            numnp (int): Anzahl der Knoten.
            numel (int): Anzahl der Elemente.
        """
        super().__init__(numnp, numel)
        self.dL = np.zeros((self.numnp, 1))
        self.u = np.zeros((self.numnp, self.dim))
        self.neumannBC = None

    # Backward-compatible aliases for legacy API
    def setNodalCoords(self, coords):
        """Alias für setNodalCoordinates (Abwärtskompatibilität)."""
        self.setNodalCoordinates(coords)

    def setElements(self, elements):
        """Alias für setElementConnectivity (Abwärtskompatibilität)."""
        self.setElementConnectivity(elements)

    def setElementData(self, youngsmoduli, sma):
        """
        Setzt die Materialeigenschaften der Elemente.

        Args:
            youngsmoduli (list): Liste der Elastizitätsmodule der Elemente.
            sma (list): Liste der Querschnittsflächenmomente (Flächenträgheitsmomente) der Elemente.
        """
        super().setElementData(sma=sma, youngsmodulus=youngsmoduli)
        

    def _getElementstiffness_(self,elID):
        """
        Berechnet die Steifigkeitsmatrix und die Kraftvektoren aus der verteilten Last eines Elements.

        Args:
            elID (int): Element-ID.

        Returns:
            tuple: Steifigkeitsmatrix (np.ndarray) und Kraftvektor F_Q (np.ndarray) des Elements.
        """
        director = self._getElementDirector(elID) 
        le = np.linalg.norm(director)
        Ke = self.eData[elID]["youngsmodulus"]*self.eData[elID]["sma"]/(le**3) * np.ones((4,4))
        Ke[0,0] *= 12
        Ke[0,1] *= 6
        Ke[0,2] *= -12
        Ke[0,3] *= 6

        Ke[1,0] *= 6
        Ke[1,1] *= 4
        Ke[1,2] *= -6
        Ke[1,3] *= 2

        Ke[2,0] *= -12
        Ke[2,1] *= -6
        Ke[2,2] *= 12
        Ke[2,3] *= -6

        Ke[3,0] *= 6
        Ke[3,1] *= 2
        Ke[3,2] *= -6
        Ke[3,3] *= 4

        # Äquivalente Knotenkräfte
        Fq =  np.zeros((4,1))
        node1 = self.elements[elID,0]
        node2 = self.elements[elID,1]
        q1 = self.dL[node1]
        q2 = self.dL[node2]
        
        Fq[0] = le/20 * (7*q1+3*q2)
        Fq[1] = le * (q1/20+q2/30)
        Fq[2] = le/20 * (3*q1+7*q2)
        Fq[3] = le * (-q1/30-q2/20)
        
        return Ke,Fq

    def shapeFunction(self, xi):
        """
        Berechnet die Formfunktionen (Shape Functions) für ein Balkenelement.

        Args:
            xi (float): Normalisierte Koordinate (0 <= xi <= 1).

        Returns:
            np.ndarray: Array der Formfunktionen an der Position xi.
        """
        return np.array([2*xi**3 - 3*xi**2 + 1, (xi**3 - 2*xi**2 + xi), -2*xi**3 + 3*xi**2, (xi**3 - xi**2)])

    def secondDerivative(self,xi):
        """
        Berechnet die zweite Ableitung der Formfunktionen nach der normalisierten Koordinate xi.

        Args:
            xi (float): Normalisierte Koordinate.

        Returns:
            np.ndarray: Zweite Ableitung der Formfunktionen.
        """
        return np.array([6*(2*xi-1),2*(3*xi-2),6*(1-2*xi),2*(3*xi-1) ])

    def thirdDerivative(self,xi):
        """
        Berechnet die dritte Ableitung der Formfunktionen.

        Args:
            xi (float): Normalisierte Koordinate.

        Returns:
            np.ndarray: Dritte Ableitung der Formfunktionen.
        """
        return np.array([6*(2),2*(3),6*(-2),2*(3) ])

    def computeMoment(self,n=10):
        """
        Berechnet die Momente entlang der Balkenstruktur.

        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Momente (np.ndarray) entlang der Balkenstruktur.
        """
        def callback(elID, xi, le, dofe):
            B = 1.0/(le**2) * self.secondDerivative(xi)
            return -(B @ dofe.T)
        
        return self._computeAlongStructure(callback, n)
    
    

    def computeQuerkraft(self,n=10):
        """
        Berechnet die Querkräfte entlang der Balkenstruktur.

        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Querkräfte (np.ndarray) entlang der Balkenstruktur.
        """
        def callback(elID, xi, le, dofe):
            B = 1.0/(le**3) * self.thirdDerivative(xi)
            return -(B @ dofe.T)
        
        return self._computeAlongStructure(callback, n)


    def _assembleGlobalMatrix2D(self):
        """
        Baut die globale Steifigkeitsmatrix und den globalen Kraftvektor auf (interne Methode).
        """
        for elID in range(self.numel):
            Ke,Fq = self._getElementstiffness_(elID)
            for nodeI in range(self.nel):
                globalNodeI = self.elements[elID,nodeI]
                for dimI in range(self.dim):
                    rowID = int(np.sign(self.eqind[globalNodeI,dimI])*self.eqind[globalNodeI,dimI]-1)
                    self.Fges[globalNodeI,dimI] += Fq[(nodeI*self.dim)+dimI,0]
                    for nodeJ in range(self.nel):
                        globalNodeJ = self.elements[elID,nodeJ]
                        for dimJ in range(self.dim):
                            colID = int(np.sign(self.eqind[globalNodeJ,dimJ])*self.eqind[globalNodeJ,dimJ]-1)
                            self.Kges[rowID,colID] += Ke[(nodeI*self.dim)+dimI,(nodeJ*self.dim)+dimJ]

