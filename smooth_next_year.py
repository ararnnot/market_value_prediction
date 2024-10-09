#%%

import  numpy           as      np
import  pandas          as      pd
import  matplotlib.pyplot as    plt
import  os
import  time
import  h5py

from    importlib               import import_module
from    utils.functions         import duplicate_mean, kernel_smoothing, filter_points_axis
from    utils.lattice_lipschits import lattice_Lipschitz
from    sklearn.preprocessing   import StandardScaler
from    sklearn.decomposition   import PCA
from    tqdm                    import tqdm

# Import predefined configuration
config_num = 12
if config_num is None :
    config_file = import_module("configs.next_year.config")
else :
    config_file = import_module(f"configs.next_year.config_{config_num}")
config = config_file.configuration
del config_file

#%%

print(f'Starting training of next year statistics predictions. \n' + \
      f'Model: {config_num}. \n' + \
      f'Ages division: {config.ages_division if config.divide_ages else "No"} \n'
      f'Type of extension: {config.ext_type}. \n'
      f'Using PCA decomposition: {config.PCA}.')

# Import data

current_directory = os.getcwd()
data_path       =   os.path.join(current_directory, 'data', 'data_XY.h5')
players_info    =   pd.read_hdf(data_path, 'players_info')
X_p             =   pd.read_hdf(data_path, 'X')
Y_p             =   pd.read_hdf(data_path, 'Y')
X               =   np.array(X_p)
Y               =   np.array(Y_p)

players_info['age'] = X_p['age']
del X_p, Y_p, data_path
print(f"Statistical data imported. {X.shape[0]} players and {X.shape[1]} columns.")

if False:
    subset = np.random.choice(np.arange(0, len(players_info)),
                              size = 500,
                              replace = False)
    players_info = players_info.loc[subset].reset_index()
    X = X[subset]
    Y = Y[subset]
    del subset

seasons_train = [42]
seasons_test  = [90]

#%%
    
index_train  = players_info \
                    .query(f'season_id in {seasons_train}') \
                    .index
index_test   = players_info \
                    .query(f'season_id in {seasons_test}') \
                    .index

X_train      = X[index_train]
Y_train      = Y[index_train]

#%%

# Compute predictions

PCA_use = True
    
if PCA_use: 
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_pca = scaler.transform(X_train.copy())
    Y_train_pca = scaler.transform(Y_train.copy())    
    
    pca = PCA(n_components = X_train_pca.shape[1])
    pca.fit(X_train_pca)
    
    X_train_pca  = pca.transform(X_train_pca)
    Y_train_pca  = pca.transform(Y_train_pca)
    
    Y_train_pca  = duplicate_mean(X_train_pca, Y_train_pca, show_progess = False)

list_sigma = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]   
list_K     = [-1 for i in list_sigma]    

LL = lattice_Lipschitz(X_train_pca, Y_train_pca)
list_K[0] = np.mean(LL.compute_phi())

for i, sigma in tqdm(enumerate(list_sigma)):
    
    if i == 0:
        smoothed_Y = Y_train_pca
    else:
        smoothed_Y = kernel_smoothing(X_train_pca, Y_train_pca, sigma)
        
    LL = lattice_Lipschitz(X_train_pca, smoothed_Y)
    list_K[i]  = np.mean(LL.compute_phi())
    
    del LL

for k in list_K:
    print(k)

#%%

plt.plot(list_sigma, list_K)
plt.xscale('log')
plt.yscale('log')
plt.show()


#%%

# Compute predictions following
# Approximation of Almost Diagonal Non-linear Maps 
# by Lattice Lipschitz Operators

PCA_use = True
    
if PCA_use: 
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_pca = scaler.transform(X_train.copy())
    Y_train_pca = scaler.transform(Y_train.copy())    
    
    filter = filter_points_axis(X_train_pca, Y_train_pca, proportion = 0.1)
    
    pca = PCA(n_components = X_train_pca.shape[1])
    pca.fit(X_train_pca[filter])
    
    X_train_pca  = pca.transform(X_train_pca)
    Y_train_pca  = pca.transform(Y_train_pca)
    
    Y_train_pca  = duplicate_mean(X_train_pca, Y_train_pca, show_progess = False)

list_sigma = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]   
list_K     = [-1 for i in list_sigma]    

LL = lattice_Lipschitz(X_train_pca, Y_train_pca)
list_K[0] = np.mean(LL.compute_phi())

for i, sigma in tqdm(enumerate(list_sigma)):
    
    if i == 0:
        smoothed_Y = Y_train_pca
    else:
        smoothed_Y = kernel_smoothing(X_train_pca, Y_train_pca, sigma)
        
    LL = lattice_Lipschitz(X_train_pca, smoothed_Y)
    list_K[i]  = np.mean(LL.compute_phi())
    
    del LL

for k in list_K:
    print(k)

#%%

plt.plot(list_sigma, list_K)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
