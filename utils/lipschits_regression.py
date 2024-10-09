
import  numpy   as  np
from    tqdm    import tqdm

class Lipschitz_regression:

    # Classical extension methods. R^n -> R

    def __init__(self, X, Y, dist_value = 2):
        
        if (len(X.shape) != 2) or (len(Y.shape) != 1) or (X.shape[0] != Y.shape[0]):
            raise Exception("X and Y must be 2-D and 1-D arrays respectively" + \
                            "with the same number of rows. \n" + \
                            f"Shapes: X: {X.shape}, Y: {Y.shape}")
        
        self.X = np.array(X)
        self.Y = np.array(Y)

        self.normalized_m_sd_computed = False
        self.dist_value = dist_value

    def normalize_compute_m_sd(self):

        if self.normalized_m_sd_computed:
            print(" (!!!) Means and sd computed by second time." + \
                  "Probably already 0s and 1s.")
        self.normalized_m_sd_computed = True

        # Computes the mean and sd X (by columns) 
        self.means  =   np.mean(self.X, axis = 0) 
        self.stds   =   np.std(self.X, axis = 0)
        # contant columns (will not 7 std)
        self.const_col = self.stds == 0 

    def normalize(self, means = None, stds = None):

        # Normalizes X
        if means == None: means = self.means
        if stds  == None: stds  = self.stds

        cols_divide = [not z for z in self.const_col]
        self.X = self.X - means
        self.X[:, cols_divide] = self.X[:, cols_divide] / stds[cols_divide]

    def normalize_inverse(self, means = None, stds = None):

        # De-normalizes X
        if means == None: means = self.means
        if stds  == None: stds  = self.stds

        self.X = self.X * stds + means

    def normalize_new_data(self, X_new, means = None, stds = None):

        # Normalizes new data with previous means and stds X
        if (len(X_new.shape) != 2) or (X_new.shape[1] != self.X.shape[1]):
            raise Exception("X_ne must be 2-D array with the same columns as X. \n" + \
                            f"Shapes: X_new: {X_new.shape}, X: {self.X.shape}")

        if means == None: means = self.means
        if stds  == None: stds  = self.stds

        cols_divide = [not z for z in self.const_col]
        X_new = X_new - means
        X_new[:, cols_divide] = X_new[:, cols_divide] / stds[cols_divide]
        return X_new

    def normalize_new_data_inverse(self, X_new, means = None, stds = None):

        if (len(X_new.shape) != 2) or (X_new.shape[1] != self.X.shape[1]):
            raise Exception("X_ne must be 2-D array with the same columns as X. \n" + \
                            f"Shapes: X_new: {X_new.shape}, X: {self.X.shape}")
        # De-normalizes X
        if means == None: means = self.means
        if stds  == None: stds  = self.stds

        X_new = X_new * stds + means
        return X_new


    def dist(self, x, y):

        # Compute the distance betwen vectors x and y
        # Modify this function to obtain diferent distances and extensions

        return np.linalg.norm(x - y, ord = self.dist_value)


    def compute_K_2(self):

        # Fastest method, only valid for l2 norm
        # TODO: copute_K vectorized and faster for OTHER dist

        n, m = self.X.shape
        double_X_pre = np.abs(self.X.reshape((1,n,m)) - self.X.reshape((n,1,m)))
        double_X = np.sqrt(np.sum(double_X_pre**2, axis = 2))
        double_Y = np.abs(self.Y.reshape((1,n)) - self.Y.reshape((n,1)))

        # Check 0s !
        double_X[double_X == 0] = 1

        K = np.max(double_Y / double_X)
        if np.isinf(K):
            raise Exception("Repeated values on X must have the same on Y" + \
                            f"Errors index: {np.where(np.isinf(double_Y/double_X))}")
        self.K = K
        return K

    def compute_K_for(self):

        # Slower method than compute_K(), but less memory requirements

        n, m = self.X.shape
        K = 0

        index = [[i, j] for i in range(n) for j in range(i)]
        for i, j in tqdm(index):
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                d = self.dist(self.X[i], self.X[j])
                val  = np.abs(self.Y[i] - self.Y[j]) / d
            if np.isinf(val):
                raise Exception("Repeated values on X must have the same on Y" + \
                                f"Errors index: {i}, {j}")
            val = np.nan_to_num(val, nan = 0)
            K = np.maximum(K, val)

        self.K = K
        return K
    
    def McShane_Whitney(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to one element

        if (len(new_X.shape) != 1) or (new_X.shape[0] != self.X.shape[1]):
            raise Exception("X_new must be a vector of length as X \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")

        d     = np.array([self.dist(z, new_X) for z in self.X])
        ext_M = np.max(self.Y - self.K * d)
        ext_W = np.min(self.Y + self.K * d)

        if both:
            return ext_M, ext_W
        else:
            return alpha * ext_M + (1-alpha) * ext_W
    
    def McShane_Whitney_2(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to one new element
        # Much memory is needed and faster

        if (len(new_X.shape) != 1) or (new_X.shape[0] != self.X.shape[1]):
            raise Exception("X_new must be a vector of length as X \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")

        n, m = self.X.shape
        n0   = new_X.shape[0]

        # shapes: [1,n,m] - [m] * [n0,1,m] -> [n0,n,m] -> [n0,m]
        d     = np.sqrt(np.sum((self.X - np.array([new_X]))**2, axis = 1))
        ext_M = np.max(self.Y - self.K * d)
        ext_W = np.min(self.Y + self.K * d)

        if both:
            return ext_M, ext_W
        else:
            return alpha * ext_M + (1-alpha) * ext_W

    def McShane_Whitney_multiple_for(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to multiple new element, slow

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        return np.array([self.McShane_Whitney(x, alpha, both) for x in tqdm(new_X)])

    def McShane_Whitney_multiple_2(self, new_X, alpha = 0.5, both = False):

        # McShane-Whitney extenison to multiple new element, faster

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        return np.array([self.McShane_Whitney_2(x, alpha, both) for x in tqdm(new_X)])
    
    def Oberman_Milman(self, new_X):

        # Oberman-Milman extension to one point. Works VERY slow

        if (len(new_X.shape) != 1) or (new_X.shape[0] != self.X.shape[1]):
            raise Exception("X_new must be a vector \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        rep = [all(new_X == z) for z in self.X]
        if any(rep):
            rep_i = np.where(rep)
            print(rep_i)
            ext = self.Y[rep_i]
            return ext
        
        n = self.X.shape[0]
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            d    = np.array([self.dist(z, new_X) for z in self.X])
            mat  = np.abs(self.Y.reshape(1, n) - self.Y.reshape(n, 1)) / \
                    (d.reshape(1,n) + d.reshape(n, 1))

        x_p, x_m = np.unravel_index(np.argmax(mat), [n,n])

        d_p = d[x_p]
        d_m = d[x_m]
        f_p = self.Y[x_p]
        f_m = self.Y[x_m]

        # Find repeated numbers and correct
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            ext = (d_m*f_p + d_p*f_m) / (d_m + d_p)
            
        return ext

    def Oberman_Milman_multiple_for(self, new_X):

        # Oberman-Milman extenison to multiple new element

        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a 2-D array \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
        
        return np.array([self.Oberman_Milman(x) for x in tqdm(new_X)])
    
    def Oberman_Milman_2_all(self, new_X):
        
        # Oberman-Milman extension to multiple new element, faster
        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            dist2 = ((np.sum((new_X[:, np.newaxis, :] - self.X[np.newaxis, :, :])**2, axis = 2))**0.5).astype(np.float32)
            fdist = (np.abs(self.Y[np.newaxis, np.newaxis, :] - self.Y[np.newaxis, :, np.newaxis]) / \
                        (dist2[:, np.newaxis, :] + dist2[:, :, np.newaxis])).astype(np.float32) 
            
        argmax_indices = np.argmax(fdist.reshape(fdist.shape[0], -1), axis=1)
        arg_plus, arg_minus = np.unravel_index(argmax_indices, (fdist.shape[1], fdist.shape[2]))
        
        indices = np.arange(len(arg_minus))        
        ext = (dist2[indices, arg_minus] * self.Y[arg_plus] + dist2[indices, arg_plus] * self.Y[arg_minus]) / \
                (dist2[indices, arg_minus] + dist2[indices, arg_plus])
        
        return ext
    
    def slope_2_average_all(self, new_X):
        
        # 2-average slope minimizing extension (paper Axioms) 2-norm metric
        
        if (len(new_X.shape) != 2) or (new_X.shape[1] != self.X.shape[1]):
            raise Exception("X_new must be a vector \n" + \
                            f"Shapes: X: {self.X.shape}, new_X: {new_X.shape}")
            
        dist2 = np.sum((new_X[:, np.newaxis, :] - self.X[np.newaxis, :, :])**2, axis = 2)
        dist2 = np.maximum(dist2, 1e-10)
        
        new_Y = np.mean(self.Y[np.newaxis, :] / dist2, axis = 1) / np.mean(1 / dist2, axis = 1)
        
        return new_Y
        
        
    

