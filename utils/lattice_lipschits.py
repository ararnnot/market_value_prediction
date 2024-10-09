
import  numpy   as  np
from    tqdm    import tqdm

class lattice_Lipschitz:

    # Start with a dataset and compute its extension. R^n -> R^n
    # See arXiv:2307.00927 or "lattice Lipschitz on C(K) spaces"

    def __init__(self, X, Y):
        
        if (len(X.shape) != 2) or (len(Y.shape) != 2) or (X.shape != Y.shape):
            raise Exception("X and Y must be 2-D arrays of the same shape \n" + \
                            f"Shapes: X: {X.shape}, Y: {Y.shape}")
        self.X = np.array(X)
        self.Y = np.array(Y)

    def compute_phi(self):

        # Faster method, but need a lot of memory

        n, m = self.X.shape
        double_X = np.abs(self.X.reshape((1,n,m)) - self.X.reshape((n,1,m)))
        double_Y = np.abs(self.Y.reshape((1,n,m)) - self.Y.reshape((n,1,m)))

        # Check 0s !
        double_X[double_X == 0] = 1

        phi = np.max(double_Y / double_X, axis = (0,1))
        if any(np.isinf(phi)):
            raise Exception("Repeated values on X must have the same on Y" + \
                            f"Errors index: {np.where(np.isinf(phi))}")
        self.phi = phi
        return phi
    
    def compute_phi_for(self):

        # Slower method than compute_K(), but less memory requirements

        n, m = self.X.shape
        phi = np.zeros(shape = m)

        index = [[i, j] for i in range(n) for j in range(i)]
        for i, j in tqdm(index):
            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.abs(self.Y[i] - self.Y[j])/np.abs(self.X[i] - self.X[j])
            if any(np.isinf(val)):
                infi = np.where(np.isinf(val))
                raise Exception("Repeated values on X must have the same on Y \n" + \
                                f"Errors index: {i}, {j},{infi}")
            val = np.nan_to_num(val, nan = 0)
            phi = np.maximum(phi, val)

        self.phi = phi
        return phi
    
    def McShane_Whitney(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to one element

        if (len(new_X.shape) != 1) or (new_X.shape[0] != self.X.shape[1]):
            raise Exception("X_new must be a vector of length as X \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")

        # shapes: [n,m] - [m] * [m] -> [n,m] -> [m]
        ext_M = np.max(self.Y - \
                       np.array([self.phi]) * \
                       np.abs(self.X - np.array([new_X])),
                       axis = 0)
        ext_W = np.min(self.Y + \
                       np.array([self.phi]) * \
                       np.abs(self.X - np.array([new_X])),
                       axis = 0)

        if both:
            return ext_M, ext_W
        else:
            return alpha * ext_M + (1-alpha) * ext_W
    
    def McShane_Whitney_multiple(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to multiple new element
        # Much memory is needed and (seems to be) slower

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")

        n, m = self.X.shape
        n0   = new_X.shape[0]

        # shapes: [1,n,m] - [m] * [n0,1,m] -> [n0,n,m] -> [n0,m]
        ext_M = np.max(self.Y.reshape(1, n, m) - \
                       self.phi.reshape(1, 1, m) * \
                       np.abs(self.X.reshape(1, n, m) - new_X.reshape(n0, 1, m)),
                       axis = 1)
        ext_W = np.min(self.Y.reshape(1, n, m) + \
                       self.phi.reshape(1, 1, m) * \
                       np.abs(self.X.reshape(1, n, m) - new_X.reshape(n0, 1, m)),
                       axis = 1)

        if both:
            return ext_M, ext_W
        else:
            return alpha * ext_M + (1-alpha) * ext_W
    
    def McShane_Whitney_multiple_for(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to multiple new element, faster

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        return np.array([self.McShane_Whitney(x, alpha, both) for x in tqdm(new_X)])
    
    def Oberman_Milman(self, new_X):

        # Oberman-Milman extension to one point. Works VERY slow

        if (len(new_X.shape) != 1) or (new_X.shape[0] != self.X.shape[1]):
            raise Exception("X_new must be a vector \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        n, m = self.X.shape
        with np.errstate(divide='ignore', invalid='ignore'):
            mat = np.abs(self.Y.reshape(1, n, m) - self.Y.reshape(n,1,m)) / \
                (np.abs(self.X.reshape(1,n,m) - new_X.reshape(1,1,m)) + \
                np.abs(self.X.reshape(n,1,m) - new_X.reshape(1,1,m)))

        x_p, x_m = np.unravel_index(np.argmax(mat.reshape(-1,m), axis = 0), [n,n])

        d_p = np.abs(self.X[x_p, range(m)] - new_X)
        d_m = np.abs(self.X[x_m, range(m)] - new_X)
        f_p = self.Y[x_p, range(m)]
        f_m = self.Y[x_m, range(m)]

        # Find repeated numbers and correct
        with np.errstate(divide='ignore', invalid='ignore'):
            ext = (d_m*f_p + d_p*f_m) / (d_m + d_p)
        rep = new_X == self.X
        rep_f, rep_c = np.where(rep)
        ext[rep_c] = self.Y[rep_f, rep_c]

        return ext

    def Oberman_Milman_multiple_for(self, new_X):

        # Oberman-Milman extenison to multiple new element

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        return np.array([self.Oberman_Milman(x) for x in tqdm(new_X)])
    

