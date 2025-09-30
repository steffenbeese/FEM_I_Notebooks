import numpy as np

class TimeIntegratorExplicit:
    def __init__(self):
        pass

    def integrate(self, ode, y0, ts, arguments):
        raise NotImplementedError("Subclasses should implement this method")


class EulerExplicit(TimeIntegratorExplicit):
    def __init__(self):
        super().__init__()

    def integrate(self, ode, y0, ts, arguments):
        """Euler Vorwärts Methode zur Lösung von System von DGL erster Ordnung.

        Args:
            ode: Funktion der DGL.
            y0: Anfangswert.
            ts: Zeitpunkte.
            arguments: Zusätzliche Argumente für die DGL.

        Returns:
            y: Numerische Lösung
        """
        y0t = np.array([y0]) if np.isscalar(y0) else y0
        y = np.zeros((len(ts), len(y0t)))
        
        y[0, :] = y0t
        tn = ts[0]

        for i, t in enumerate(ts[1:]):
            dt = t - tn
            y[i + 1, :] = y[i, :] + dt * ode(tn, y[i, :], *arguments)
            tn = t

        return y
    

class HeunExplicit(TimeIntegratorExplicit):
    def __init__(self):
        super().__init__()
    def integrate(self, ode, y0, ts, arguments):
        """Heun's Method zur Lösung eines Systems von DGL erster Ordnung.
    
        Args:
            ode: Funktion der DGL (t, y, ...).
            y0: Anfangswert.
            ts: Zeitpunkte.
            arguments: Zusätzliche Argumente für die DGL.
        
        Returns:
            y: Numerische Lösung
        """
    y0t = np.array([y0]) if np.isscalar(y0) else y0
    y = np.zeros((len(ts), len(y0t)))
    y[0, :] = y0t
    tn = ts[0]
    
    for i, t in enumerate(ts[1:]):
        dt = t - tn
        
        # Berechne die Steigung k1 am Anfang des Intervalls
        k1 = ode(tn, y[i, :], *arguments)
        
        # Berechne den vorläufigen Wert für y am Ende des Intervalls
        y_tilde = y[i, :] + dt * k1
        
        # Berechne die Steigung k2 am vorläufigen Endpunkt des Intervalls
        k2 = ode(t, y_tilde, *arguments)
        
        # Berechne den endgültigen Wert für y
        y[i + 1, :] = y[i, :] + (dt / 2) * (k1 + k2)
        tn = t
    
    return y

class CentralDifference(TimeIntegratorExplicit):
    def __init__(self):
        super().__init__()

    def integrate(self, ode, y0, ts, arguments):
        """Central Difference Method for solving a system of first-order ODEs.

        Args:
            ode: Function representing the ODE(s) (t, y, ...).
            y0: Initial value.
            ts: Time points.
            arguments: Additional arguments for the ODE(s).

        Returns:
            y: Numerical solution
        """
        y0t = np.array([y0]) if np.isscalar(y0) else y0
        y = np.zeros((len(ts), len(y0t)))
        y[0, :] = y0t

        # Use Euler's method to get the first step
        dt = ts[1] - ts[0]
        y[1, :] = y[0, :] + dt * ode(ts[0], y[0, :], *arguments)

        for i in range(1, len(ts) - 1):
            dt = ts[i+1] - ts[i]
            tn = ts[i]
            y[i+1, :] = y[i-1, :] + 2 * dt * ode(tn, y[i, :], *arguments)

        return y



class TimeIntegratorImplicit:
    def __init__(self):
        pass

    def integrate(self, ode,J , y0, ts, arguments):
        raise NotImplementedError("Subclasses should implement this method")
    
    def newton_raphson_multivariate(f, J, x0, tol=1e-6, max_iter=100):
        """
        Newton-Raphson method for finding roots of multivariate functions.

        Parameters:
        f (function): The multivariate function to find roots of.
        J (function): The Jacobian matrix of f.
        x0 (array): Initial guess for the root.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

        Returns:
        array: Approximate root of the function.
        """
        x = np.array(x0, dtype=float)
        # print('----------------------')
        for i in range(max_iter):
            fx = f(x)
            # print(f'it: {i}, res: {fx}')
            Jx = J(x)
            delta = np.linalg.solve(Jx, -fx)
            x_new = x + delta

            if np.linalg.norm(delta) < tol:
                return x_new

            x = x_new

        raise RuntimeError("Newton-Raphson did not converge within maximum iterations")

class EulerImplicit(TimeIntegratorImplicit):
    def __init__(self):
        super().__init__()
    def integrate(self, f, J, y0, ts, arguments):
        """
    Euler implicit method for solving ODEs.

    Parameters:
    f (function): The right-hand side of the ODE dy/dt = f(y, t).
    J (function): The Jacobian matrix of f with respect to y.
    y0 (array): Initial condition.
    ts (array): Time points at which to solve the ODE.
        arguments (tuple): Additional arguments for f and J.

    Returns:
    y - y is the solution array.
    """
    y = np.zeros((len(ts), len(y0)))
    y[0, :] = y0

    tn = ts[0]
    for i, t in enumerate(ts[1:]):
        dt = t - tn

        def residual(y_new):
                return y_new - (y[i, :] + dt * f(t, y_new, *arguments))

        def Jac(y_new):
                return np.eye(len(y0)) - dt * J(t, y_new, *arguments)
        
        y[i + 1, :] = self.newton_raphson_multivariate(residual, Jac, y[i, :])
        
        tn = t

    return y



class ThetaMethodImplicit(TimeIntegratorImplicit):
    def __init__(self, theta=0.5):
        super().__init__()
        self.theta = theta

    def integrate(self, f, J, y0, ts, arguments):
        """
    Theta method for solving ODEs.

    Parameters:
    f (function): The right-hand side of the ODE dy/dt = f(y, t).
    J (function): The Jacobian matrix of f with respect to y.
    y0 (array): Initial condition.
    ts (array): Time points at which to solve the ODE.
        arguments (tuple): Additional arguments for f and J.
    Returns:
    y - y is the solution array.
    """
    
    y = np.zeros((len(ts), len(y0)))
    y[0, :] = y0

    tn = ts[0]
    for i, t in enumerate(ts[1:]):
        dt = t - tn
        
        def residual(y_new):
                return y_new - y[i, :] - self.theta * dt * f(t, y_new, *arguments) \
                       - (1-self.theta) * dt * f(tn, y[i, :], *arguments)
                   
        def Jac(y_new):
                return np.eye(len(y0)) - self.theta * dt * J(t, y_new, *arguments)
        
        y[i + 1, :] = self.newton_raphson_multivariate(residual, Jac, y[i, :])
        tn = t

    return y