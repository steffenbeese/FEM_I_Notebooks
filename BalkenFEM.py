import numpy as np
class BalkenFEM:
    """
    Eine Klasse zur Finite-Elemente-Methode (FEM) für Balkenstrukturen.

    Diese Klasse implementiert die Finite-Elemente-Methode für die Analyse von Balkenstrukturen.
    Sie unterstützt die Definition von Knoten, Elementen, Materialeigenschaften, Randbedingungen
    und externen Kräften. Die Klasse kann die globale Steifigkeitsmatrix aufbauen, das Gleichungssystem
    lösen und die Resultate wie Verformungen, Momente und Querkräfte berechnen.

    Attribute:
        numnp (int): Anzahl der Knoten.
        numel (int): Anzahl der Elemente.
        dim (int): Dimension des Problems (standardmäßig 2 für 2D-Probleme).
        nel (int): Anzahl der Knoten pro Element (standardmäßig 2 für Balkenelemente).
        numdof (int): Gesamtanzahl der Freiheitsgrade.
        elements (np.ndarray): Array zur Speicherung der Elementverbindungen.
        eData (list): Liste zur Speicherung der Materialeigenschaften der Elemente.
        coords (np.ndarray): Array zur Speicherung der Knotenkoordinaten.
        dof (np.ndarray): Array zur Speicherung der Freiheitsgrade der Knoten.
        eqind (np.ndarray): Array zur Speicherung der Gleichungsindizes der Freiheitsgrade.
        Kges (np.ndarray): Globale Steifigkeitsmatrix.
        Fges (np.ndarray): Globale Kraftvektor.
        qL (np.ndarray): Verteilte Lasten an den Knoten.
        u (np.ndarray): Verformungen der Knoten.
        neumannBC (any): Neumann-Randbedingungen (externe Kräfte).
        dirichletBC (any): Dirichlet-Randbedingungen (Verschiebungen).
        Kuu, Kud, Kdu, Kdd (np.ndarray): Partitionierte Steifigkeitsmatrizen.
    """
    def __init__(self,numnp=2,numel=1):
        """
        Initialisiert die BalkenFEM-Klasse.

        Args:
            numnp (int): Anzahl der Knoten.
            numel (int): Anzahl der Elemente.
        """        
        self.numnp = numnp
        self.numel = numel
        self.dim = 2
        self.nel = 2
        self.numdof = self.numnp * self.dim
        self.elements = np.zeros((self.numel,2),dtype=int)
        self.eData = []
        self.coords = np.zeros((self.numnp,2))
        self.dof = np.zeros((self.numnp,self.dim))
        self.eqind = np.zeros((self.numnp,self.dim))
        self.Kges = np.zeros((self.numdof,self.numdof))
        self.Fges = np.zeros((self.numnp,self.dim))
        self.qL = np.zeros((self.numnp,1))
        self.u = np.zeros((self.numnp,self.dim))
        self.neumannBC = None
        self.dirichletBC = None
        self.Kuu = self.Kud = self.Kdu = self.Kdd = None

    def setNodalCoords(self,coords):
        """
        Setzt die Knotenkoordinaten.

        Args:
            coords (np.ndarray): Array der Knotenkoordinaten.
        """        
        self.coords[:,:] = coords[:,:]
    
    def setElements(self,elements):
        """
        Setzt die ElementKonnektivität.

        Args:
            elements (np.ndarray): Array der ElementKonnektivität.
        """        
        for i in range(self.numel):
            for j in range(self.nel):
                self.elements[i,j] = int(elements[i,j])
                

    def setElementData(self,youngsmoduli,sma):
        """
        Setzt die Materialeigenschaften der Elemente.

        Args:
            youngsmoduli (list): Liste der Elastizitätsmodule der Elemente.
            sma (list): Liste der Querschnittsflächenmomente (Flächentägheitsmomente) der Elemente.
        """        
        for i in range(self.numel):
            self.eData.append({"sma":sma[i],"youngsmodulus":youngsmoduli[i]})

    def setDirichletBoundaryCondition(self,dirichletNodes,dirichletDir,dirichletVal):
        """
        Setzt die Dirichlet-Randbedingungen (Verschiebungen und Neigungen).

        Args:
            dirichletNodes (list): Liste der Knoten mit Dirichlet-Randbedingungen.
            dirichletDir (list): Liste der  Dirichlet-Randbedingungen: 0 für Verschiebung, 1 für Neigung.
            dirichletVal (list): Liste der Werte der Dirichlet-Randbedingungen.
        """        
        eqID = 1
        for nodeID in range(self.numnp):
            for dirID in range(self.dim):
                self.eqind[nodeID,dirID] = eqID
                eqID +=1
        # Knoten an denen Dirichlet Randbedingungen vorgegeben werden
        # müssen im Gleichungssystem nicht berücksichtigt werden.
        # Deshalb erhalten sie negative Gleichungsnummern zur Identifikation
        
        self.dirichletBC = np.zeros((len(dirichletNodes),3))
        
        for i,(nodeID,dirId,dVal) in enumerate(zip(dirichletNodes,dirichletDir,dirichletVal)):
            self.eqind[nodeID,dirId] = -self.eqind[nodeID,dirId]
            self.dirichletBC[i,0] = nodeID
            self.dirichletBC[i,1] = dirID
            self.dirichletBC[i,2] = dVal
            self.dof[nodeID,dirId] = dVal
    
    def setExternalForces(self,neumannNodes,neumannDir,neumannVal):
        """
        Setzt die Neumann-Randbedingungen (externe Kräfte).

        Args:
            neumannNodes (list): Liste der Knoten mit Neumann-Randbedingungen.
            neumannDir (list): Liste  Neumann-Randbedingungen: 0 = Kraft, 1 = Moment.
            neumannVal (list): Liste der Werte der Neumann-Randbedingungen.
        """        
        for nodeID,nDir,nVal in zip(neumannNodes,neumannDir,neumannVal):
            self.Fges[nodeID,nDir] = nVal

    def setDistributedLoads(self,qloads):
        """
        Setzt die verteilten Lasten an den Knoten. So definiert sind keine Sprünge über Elementgrenzen hinweg möglich.

        Args:
            qloads (np.ndarray): Array der verteilten Lasten an den Knoten.
        """
        self.qL[:] = qloads[:]

    def _getElementDirector(self,elID):
        """
        Berechnet den Richtungsvektor eines Elements.

        Args:
            elID (int): Element-ID.

        Returns:
            np.ndarray: Richtungsvektor des Elements.
        """
        nodeID1 = self.elements[elID,0]
        nodeID2 = self.elements[elID,1]
        director = self.coords[nodeID2,:] - self.coords[nodeID1,:]
        return director

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

        Fq =  np.zeros((4,1))
        node1 = self.elements[elID,0]
        node2 = self.elements[elID,1]
        q1 = self.qL[node1]
        q2 = self.qL[node2]
        
        Fq[0] = le/20 * (7*q1+3*q2)
        Fq[1] = le * (q1/20+q2/30)
        Fq[2] = le/20 * (3*q1+7*q2)
        Fq[3] = le * (-q1/30-q2/20)
        
        return Ke,Fq

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
        X = np.zeros(n*self.numel)
        Mges = np.zeros(n*self.numel)
        
        for elID in range(self.numel):
            director = self._getElementDirector(elID) 
            le = np.linalg.norm(director)
            node1 = self.elements[elID,0]
            node2 =self.elements[elID,1]
            x1 = self.coords[node1,0]
            x2 = self.coords[node2,0]
            X[elID*n:(elID+1)*n] = xlin =  np.linspace(x1,x2,n)
            xilin = np.linspace(0,1,n)
            dofe = np.array([self.dof[node1,0],self.dof[node1,1],self.dof[node2,0],self.dof[node2,1]])
            for i,xi in enumerate(xilin):
                B=1.0/(le**2) * self.secondDerivative(xi)
                # M = self.eData[elID]["youngsmodulus"]*self.eData[elID]["sma"] * B @ dofe.T
                M = B @ dofe.T
                Mges[elID*n+i] = -M
        return X,Mges

    def computeQuerkraft(self,n=10):
        """
        Berechnet die Querkräfte entlang der Balkenstruktur.

        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Querkräfte (np.ndarray) entlang der Balkenstruktur.
        """
        X = np.zeros(n*self.numel)
        Qges = np.zeros(n*self.numel)
        
        for elID in range(self.numel):
            director = self._getElementDirector(elID) 
            le = np.linalg.norm(director)
            node1 = self.elements[elID,0]
            node2 =self.elements[elID,1]
            x1 = self.coords[node1,0]
            x2 = self.coords[node2,0]
            X[elID*n:(elID+1)*n] = xlin =  np.linspace(x1,x2,n)
            xilin = np.linspace(0,1,n)
            dofe = np.array([self.dof[node1,0],self.dof[node1,1],self.dof[node2,0],self.dof[node2,1]])
            for i,xi in enumerate(xilin):
                B=1.0/(le**3) * self.thirdDerivative(xi)
                # M = self.eData[elID]["youngsmodulus"]*self.eData[elID]["sma"] * B @ dofe.T
                Q = B @ dofe.T
                Qges[elID*n+i] = -Q
        return X,Qges

    def getDisplacement(self,x):
        """
        Berechnet die Durchbiegung an einer bestimmten Position x entlang der Struktur.

        Args:
            x (float): Position entlang der Struktur.

        Returns:
            float: Verschiebung an der Position x.
        """
        for elID in range(self.numel):
            node1 = self.elements[elID,0]
            node2 =self.elements[elID,1]
            x1 = self.coords[node1,0]
            x2 = self.coords[node2,0]

            if (x >= x1-1.e-12) and (x < x2):
                director = self._getElementDirector(elID) 
                le = np.linalg.norm(director)
                dofe = np.array([self.dof[node1,0],self.dof[node1,1],self.dof[node2,0],self.dof[node2,1]])
                xi = (x-x1)/(x2-x1)
                N = np.array([2*xi**3 - 3*xi**2 + 1, (xi**3 - 2*xi**2 + xi), -2*xi**3 + 3*xi**2, (xi**3 - xi**2)])
                return N @ dofe.T

    def assembleGlobalMatrix2D(self):
        """
        Baut die globale Steifigkeitsmatrix und den globalen Kraftvektor auf.
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

    def solveSystem(self):
        """
        Löse das Gleichungssystem.
        """
        # Erzeuge Hiflsmatrizen um die Freiheitsgrade zu sortieren
        numcdof = self.dirichletBC.shape[0]
        numfdof = self.numdof - numcdof
        self.Kuu = np.zeros((numfdof,numfdof)) 
        self.Kud = np.zeros((numfdof,numcdof))
        self.Kdd = np.zeros((numcdof,numcdof))
        Fu = np.zeros((numfdof,1))
        Fc = np.zeros((numcdof,1))
        uc = np.zeros((numcdof,1))
        eqf_inv = np.zeros((numfdof,2),dtype=int)
        eqc_inv = np.zeros((numcdof,2),dtype=int)
        Fvec = np.zeros((2*self.numnp,1))

        # Sortiere Freiheitsgrade
        icon  = -1
        ifree = -1
        iIsFree = False
        for nodeI in range(self.numnp):
            for dirI in range(self.dim):
                jcon  = -1
                jfree = -1
                jIsFree = False
                if self.eqind[nodeI,dirI]>=0:
                    iIsFree = True
                    ifree+=1
                else:
                    iIsFree = False
                    icon +=1
                eqI = int(np.sign(self.eqind[nodeI,dirI])*self.eqind[nodeI,dirI]-1)
                Fvec[eqI] = self.Fges[nodeI,dirI]
                if iIsFree:
                    eqf_inv[ifree,0] = nodeI
                    eqf_inv[ifree,1] = dirI
                    Fu[ifree]=self.Fges[nodeI,dirI]
                else:
                    eqc_inv[icon,0] = nodeI
                    eqc_inv[icon,1] = dirI
                    Fc[icon] = self.Fges[nodeI,dirI]
                    uc[icon] = self.dof[nodeI,dirI]
                for nodeJ in range(self.numnp):
                    for dirJ in range(self.dim):
                        if self.eqind[nodeJ,dirJ] >= 0 :
                            jIsFree = True
                            jfree +=1
                        else:
                            jIsFree = False
                            jcon += 1 
                        
                        eqJ = int(np.sign(self.eqind[nodeJ,dirJ])*self.eqind[nodeJ,dirJ]-1)
                        if iIsFree and jIsFree:
                            self.Kuu[ifree,jfree] = self.Kges[eqI,eqJ]
                        if iIsFree and not jIsFree:
                            self.Kud[ifree,jcon] = self.Kges[eqI,eqJ]
                        if not iIsFree and not jIsFree:
                            self.Kdd[icon,jcon] = self.Kges[eqI,eqJ]
        # display("Fvec: ",Fvec)
        # Löse Gleichungssystem
        # K_uu*u = F-K_ud*u_d
        rhs = Fu-self.Kud @ uc
        # display("RHS: ",rhs)
        # display("Kuu:",self.Kuu)
        u = np.linalg.solve(self.Kuu,rhs)
        ieq = 0
        # Kopiere zu globalen Freiheitsgraden
        for nodeI in range(numfdof):
            self.dof[eqf_inv[nodeI,0],eqf_inv[nodeI,1]] = u[ieq,0]
            ieq += 1
        # Bestimme Reaktionskräfte
        # R = K_du * u + K_dd*u_c - F_c
        rF = self.Kud.T@u+self.Kdd@uc -Fc
        # Kopiere zu den globalen Kräften
        ieq = 0
        for nodeI in range(numcdof):
            self.Fges[eqc_inv[nodeI,0],eqc_inv[nodeI,1]] = rF[ieq,0]
            ieq += 1 