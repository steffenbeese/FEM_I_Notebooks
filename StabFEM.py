import numpy as np
import matplotlib.pyplot as plt

class StabFEM:
    """
    Klasse zur Durchführung von Finite-Elemente-Analysen für lineare Stabtragwerke im 2D-Raum.
    """
    
    def __init__(self, numnp=2, numel=1):
        """
        Initialisiert eine neue Instanz der StabFEM-Klasse.

        Parameter:
        numnp (int): Anzahl der Knotenpunkte im Modell.
        numel (int): Anzahl der Elemente im Modell.
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
        self.dirichletBC = None
        self.Kuu = self.Kud = self.Kdu = self.Kdd = None
        
    def setNodalCoordinates(self, X):
        """
        Setzt die Koordinaten der Knotenpunkte.

        Parameter:
        X (np.ndarray): 2D-Array mit den Koordinaten der Knotenpunkte.
        """
        self.coords[:, :] = X[:, :]
                
    def setElementConnectivity(self, IX):
        """
        Definiert die Element-Knoten-Konnektivität.

        Parameter:
        IX (np.ndarray): Array mit den Indizes der Knoten für jedes Element.
        """
        for i in range(self.numel):
            for j in range(self.nel):
                self.elements[i, j] = int(IX[i, j])
        
    def setElementData(self, areas, youngsmoduli, lineLoads):
        """
        Setzt die Elementdaten wie Querschnittsflächen, Elastizitätsmoduln und Längslasten.

        Parameter:
        areas (list): Liste der Querschnittsflächen für jedes Element.
        youngsmoduli (list): Liste der Elastizitätsmodul-Werte für jedes Element.
        lineLoads (list): Liste der Längslasten für jedes Element.
        """
        for i in range(self.numel):
            self.eData.append({
                "area": areas[i],
                "youngsmodulus": youngsmoduli[i],
                "lineLoad": lineLoads[i]
            })
            
    def setDirichletBoundaryCondition(self, dirichletNodes, dirichletDir, dirichletVal):
        """
        Setzt die Dirichlet-Randbedingungen.

        Parameter:
        dirichletNodes (list): Liste der Knoten, für die die Dirichlet-Randbedingung gilt.
        dirichletDir (list): Liste der Richtungen (0:x, 1:y) der Randbedingungen.
        dirichletVal (list): Liste der Werte der Dirichlet-Randbedingungen.
        """
        eqID = 1
        for nodeID in range(self.numnp):
            for dirID in range(self.dim):
                self.eqind[nodeID, dirID] = eqID
                eqID += 1
        self.dirichletBC = np.zeros((len(dirichletNodes), 3))
        
        for i, (nodeID, dirId, dVal) in enumerate(zip(dirichletNodes, dirichletDir, dirichletVal)):
            self.eqind[nodeID, dirId] = -self.eqind[nodeID, dirId]
            self.dirichletBC[i, 0] = nodeID
            self.dirichletBC[i, 1] = dirId
            self.dirichletBC[i, 2] = dVal
            self.dof[nodeID, dirId] = dVal

    def setExternalForces(self, neumannNodes, neumannDir, neumannVal):
        """
        Setzt die äußeren Kräfte auf die Knoten.

        Parameter:
        neumannNodes (list): Liste der Knoten, auf die Kräfte wirken.
        neumannDir (list): Liste der Richtungen (0:x, 1:y) der Kräfte.
        neumannVal (list): Liste der Kraftwerte für jeden Knoten.
        """
        for nodeID, nDir, nVal in zip(neumannNodes, neumannDir, neumannVal):
            self.Fges[nodeID, nDir] = nVal

    def _getElementstiffness_(self, elID):
        """
        Berechnet die Steifigkeitsmatrix eines Elements.

        Parameter:
        elID (int): Die ID des Elements.

        Rückgabewert:
        np.ndarray: Steifigkeitsmatrix des Elements.
        """
        director = self._getElementDirector(elID) 
        le = np.linalg.norm(director) 
        Ke = self.eData[elID]["area"] * self.eData[elID]["youngsmodulus"] / le * np.ones((2, 2))
        Ke[0, 1] *= -1
        Ke[1, 0] *= -1
        return Ke

    def assembleGlobalMatrix(self):
        """
        Assemblierung der globalen Steifigkeitsmatrix.
        """
        self._assembleGlobalMatrix2D()

    def _assembleGlobalMatrix2D(self):
        """
        Assemblierung der globalen Steifigkeitsmatrix im 2D-Raum.
        """
        for elID in range(self.numel):
            Ke = self._getElementstiffness_(elID)
            Ke_enhanced = np.zeros((4, 4))
            Ke_enhanced[0, 0] = Ke[0, 0]
            Ke_enhanced[0, 2] = Ke[0, 1]
            Ke_enhanced[2, 0] = Ke[1, 0]
            Ke_enhanced[2, 2] = Ke[1, 1]
            R = self._getTransFormationMatrix(elID)
            Ke = R.T @ Ke_enhanced @ R
            for nodeI in range(self.nel):
                globalNodeI = self.elements[elID, nodeI]
                for dimI in range(self.dim):
                    rowID = int(np.sign(self.eqind[globalNodeI, dimI]) * self.eqind[globalNodeI, dimI] - 1)
                    for nodeJ in range(self.nel):
                        globalNodeJ = self.elements[elID, nodeJ]
                        for dimJ in range(self.dim):
                            colID = int(np.sign(self.eqind[globalNodeJ, dimJ]) * self.eqind[globalNodeJ, dimJ] - 1)
                            self.Kges[rowID, colID] += Ke[(nodeI * self.dim) + dimI, (nodeJ * self.dim) + dimJ]

    def _getElementDirector(self, elID):
        """
        Bestimmt den Richtungsvektor eines Elements.

        Parameter:
        elID (int): Die ID des Elements.

        Rückgabewert:
        np.ndarray: Richtungsvektor des Elements.
        """
        nodeID1 = self.elements[elID, 0]
        nodeID2 = self.elements[elID, 1]
        director = self.coords[nodeID2, :] - self.coords[nodeID1, :]
        return director

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
        Rot = np.zeros((2, 2))
        E1 = np.zeros((2, 2))
        E1[0, 0] = 1.0
        E1[1, 1] = 1.0
        Euser = np.zeros((2, 2))
        Euser[0, 0] = director[0]
        Euser[1, 0] = director[1]
        Euser[0, 1] = normal[0]
        Euser[1, 1] = normal[1]
        Rot = Euser.T @ E1.T
        
        R = np.array([[Rot[0, 0], Rot[0, 1], 0, 0], [Rot[1, 0], Rot[1, 1], 0, 0], [0, 0, Rot[0, 0], Rot[0, 1]], [0, 0, Rot[1, 0], Rot[1, 1]]])
        return R

    def assembleRightHandSide(self):
        """
        Assemblierung der rechten Seite des Gleichungssystems.
        """
        self._assembleRightHandSide2D()

    
    def _assembleRightHandSide2D(self):
        """
        Assemblierung der rechten Seite im 2D-Raum.
        """
        for elID in range(self.numel):
            director = self._getElementDirector(elID) 
            le = np.linalg.norm(director)
            R = self._getTransFormationMatrix(elID)
            Feval = self.eData[elID]["area"] * le * self.eData[elID]["lineLoad"] * 0.5
            Fe = np.zeros((4, 1))
            Fe[0, 0] = 1
            Fe[2, 0] = 1
            Fe = Feval * (R.T @ Fe)
            for nodeI in range(self.nel):
                for dirI in range(self.dim):
                    globalNodeI = self.elements[elID, nodeI]
                    self.Fges[globalNodeI, dirI] += Fe[(nodeI * self.dim) + dirI]

    def solveSystem(self):
        """
        Lösung des Gleichungssystems.
        """
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
                        
        rhs = Fu - self.Kud @ uc
        u = np.linalg.solve(self.Kuu, rhs)
        ieq = 0
        for nodeI in range(numfdof):
            self.dof[eqf_inv[nodeI, 0], eqf_inv[nodeI, 1]] = u[ieq]
            ieq += 1
        rF = self.Kud.T @ u + self.Kdd @ uc - Fc
        ieq = 0
        for nodeI in range(numcdof):
            self.Fges[eqc_inv[nodeI, 0], eqc_inv[nodeI, 1]] = rF[ieq]
            ieq += 1 
            
    def computeNormalkraft(self,n=10):
        """
        Berechnet die Normalkräfte entlang der Stabstruktur.

        Args:
            n (int): Anzahl der Stützpunkte je Element für die Berechnung.

        Returns:
            tuple: Koordinaten (np.ndarray) und Normalkräfte (np.ndarray) entlang der Stabstruktur.
        """
        X = np.zeros(n*self.numel)
        Nges = np.zeros(n*self.numel)
        
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
            R = self._getTransFormationMatrix(elID)
            EA = self.eData[elID]["area"] * self.eData[elID]["youngsmodulus"]
            dofl = R @ dofe.T
            for i,xi in enumerate(xilin):
                B=1.0/(le) * np.array([-1,0,1,0])
                N = EA * B @ dofl.T
                Nges[elID*n+i] = N
        return X,Nges
    
    def computeDisplacement(self,n=10):
        X = np.zeros(n*self.numel)
        uges = np.zeros(n*self.numel)
        
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
            R = self._getTransFormationMatrix(elID)
            Rl = R[0:2,0:2]
            dofl = R @ dofe.T
            for i,xi in enumerate(xilin):
                N = np.array([1-xi,0,xi,0])
                ul = N @ dofl.T
                uges[elID*n+i] = ul
                
        return X,uges
            
    def getDisplacement(self, x):
        """
        Berechnet die Verschiebung an einer bestimmten Position x entlang der Struktur.

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
            
            if (x >= x1-1.e-12) and (x <= x2+1.e-12):
                director = self._getElementDirector(elID) 
                le = np.linalg.norm(director)
                dofe = np.array([self.dof[node1,0],self.dof[node1,1],self.dof[node2,0],self.dof[node2,1]])
                R = self._getTransFormationMatrix(elID)
                dofl = R @ dofe.T
                xi = (x-x1)/(x2-x1)
                N = np.array([1-xi,0,xi,0])
                Rl = R[0:2,0:2]
                ul = np.array([N @ dofl.T,0])
                
                return Rl.T @ ul.T

    def plotMesh(self, deformed=False, scale=1.0, ax=None, fig=None):
        """
        Plottet das Netz des Modells.

        Parameter:
        deformed (bool): Gibt an, ob das deformierte Netz geplottet werden soll.
        scale (float): Skalierungsfaktor für die Darstellung.
        ax (matplotlib.axes.Axes): Achsenobjekt für die Darstellung.
        fig (matplotlib.figure.Figure): Figur-Objekt für die Darstellung.

        Rückgabewert:
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Figur und Achsenobjekt.
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
                Xd[:, dimI] = X[:, dimI] + scale*u[:, dimI]
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
        # K_uu*u = F-K_ud*u_d
        rhs = Fu-self.Kud @ uc
        # display("RHS: ",rhs)
        # display("Kuu:",self.Kuu)
        u = np.linalg.solve(self.Kuu,rhs)
        rhs = Fu - self.Kud @ uc
        u = np.linalg.solve(self.Kuu, rhs)
        # Kopiere zu globalen Freiheitsgraden
        for nodeI in range(numfdof):
        # Kopiere zu globalen Freiheitsgraden
            ieq += 1
            self.dof[eqf_inv[nodeI,0],eqf_inv[nodeI,1]] = u[ieq]
            self.dof[eqf_inv[nodeI, 0], eqf_inv[nodeI, 1]] = u[ieq]
        # R = K_du * u + K_dd*u_c - F_c
        # Bestimme Reaktionskräfte
        # R = K_du * u + K_dd*u_c - F_c
        rF = self.Kud.T@u+self.Kdd@uc -Fc
        # Kopiere zu den globalen Kräften
        rF = self.Kud.T @ u + self.Kdd @ uc - Fc
        self.Fges[eqc_inv[nodeI,0],eqc_inv[nodeI,1]] = rF[ieq]
        ieq += 1 
        self.Fges[eqc_inv[nodeI,0],eqc_inv[nodeI,1]] = rF[ieq]
        self.Fges[eqc_inv[nodeI, 0], eqc_inv[nodeI, 1]] = rF[ieq]
        
        return fig,ax

