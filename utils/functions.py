#%%

import  numpy   as      np
from    tqdm    import  tqdm

def duplicate_mean(X, Y, show_progess = True):

    # Given two datasets, for applying a llattice Lipschitz extension,
    # tho repeated values in a column on X must have the same value
    # on Y, so we compute the mean on that case.

    n, m = X.shape

    for i in tqdm(range(m), desc = "Columns: ", disable = not show_progess):

        Xcol = X[:,i]

        unique_elements, counts = np.unique(Xcol, return_counts = True)
        duplicated_elements = unique_elements[counts > 1]
        
        if show_progess:
            print(f'Column {i} has {len(duplicated_elements)} of {len(Xcol)} duplicated elements')

        for v in duplicated_elements:

            index = Xcol == v
            Y[index, i] = Y[index, i].mean()

    return Y

def gaussian_kernel(x, x_vals, sigma):
    return np.exp(-((x - x_vals)**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def kernel_smoothing_one(x_vals, y_vals, sigma):
    
    norm2 = np.sum((x_vals[:, np.newaxis, :] - x_vals[np.newaxis, :, :])**2, axis = 2)
    weights = np.exp(- norm2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    
    smoothed_y = np.sum(weights * y_vals, axis=1) / np.sum(weights, axis=1)
    
    return smoothed_y

def kernel_smoothing(x_vals, y_vals, sigma):
    
    diff = x_vals[:, np.newaxis, :] - x_vals[np.newaxis, :, :]
    weights = np.exp(-(diff**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    
    smoothed_y = np.sum(weights * y_vals, axis=1) / np.sum(weights, axis=1)
    
    return smoothed_y

def filter_points_axis(x, y, proportion = 0.1):
    
    eps = [-1] * len(x)
    
    for i, (xx, yy) in enumerate(zip(x, y)):
        
        # How far is x, y = T(x) from being an eigenvector
        
        eps[i] = np.linalg.norm(yy)**2 / np.linalg.norm(xx)**2 - \
                    np.dot(xx, yy)**2 / np.linalg.norm(xx)**4
                    
    filter = eps < np.quantile(eps, proportion)
    print(eps)
    
    return filter

def NRMSE(G, P):

    # Given two numpy 2D-arrays computes the normalized RMSE:
    # result = RMSE of the NRMSE each column
    # each NRME = sqrt( sum( (G-P)^2 )/N )/( G: max - min )

    if (G.shape != P.shape) or (len(G.shape) != 2):
        raise ValueError(f'G and P must be 2D-arrays with the same shape' + \
                         f'Shapes: {G.shape}, {P.shape}')

    columns = np.sqrt(np.sum((G - P)**2, axis = 0) / G.shape[0])
    columns = columns / (np.max(G, axis = 0) - np.min(G, axis = 0))
    nrmse   = np.sqrt(np.sum(columns**2) / G.shape[1]) 

    return nrmse

def NRMSE_1D(G, P):

    # Given two numpy 1D-arrays computes the normalized RMSE:
    # result = RMSE of the NRMSE each column
    # each NRME = sqrt( sum( (G-P)^2 )/N )/( G: max - min )

    if (G.shape != P.shape) or (len(G.shape) != 1):
        raise ValueError(f'G and P must be 1D-arrays with the same shape' + \
                         f'Shapes: {G.shape}, {P.shape}')

    nrmse = np.sqrt(np.sum((G - P)**2) / G.shape[0]) / (np.max(G) - np.min(G))

    return nrmse