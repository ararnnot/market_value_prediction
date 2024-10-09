#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  h5py

from    importlib               import import_module
from    utils.functions         import duplicate_mean, kernel_smoothing, NRMSE, filter_points_axis
from    utils.lattice_lipschits import lattice_Lipschitz
from    sklearn.preprocessing   import StandardScaler
from    sklearn.decomposition   import PCA

# Import predefined configuration
config_num = 43
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

# For checking model, fast computing on a subset
if config.fast:
    subset = np.random.choice(np.arange(0, len(players_info)),
                              size = 500,
                              replace = False)
    players_info = players_info.loc[subset].reset_index()
    X = X[subset]
    Y = Y[subset]
    del subset

#%%

# Divide by ages and create datasets train-test X-Y

if config.divide_ages:
    ages_intervals = [float('-inf')] + config.ages_division + [float('inf')]
else:
    ages_intervals = [float('-inf'), float('inf')]

N_intervals = (len(ages_intervals) - 1)
index_train = [None] * N_intervals
index_test  = [None] * N_intervals
X_train     = [None] * N_intervals
X_test      = [None] * N_intervals
Y_train     = [None] * N_intervals
Y_test      = [None] * N_intervals

for i in range(N_intervals):
    
    condition = f'(age > {ages_intervals[i]}) and (age <= {ages_intervals[i+1]})'
    index_train[i]  = players_info \
                        .query(condition) \
                        .query(f'season_id in {config.seasons_train}') \
                        .index
    index_test[i]   = players_info \
                        .query(condition) \
                        .query(f'season_id in {config.seasons_test}') \
                        .index
    
    X_train[i]      = X[index_train[i]]
    Y_train[i]      = Y[index_train[i]]
    
    # Avoid duplicate for verify the lattice Lipschitz condition
    if config.PCA not in ['first', 'first0', 'all']:
        Y_train[i]      = duplicate_mean(X_train[i], Y_train[i], show_progess = False)

    X_test[i]       = X[index_test[i]]
    Y_test[i]       = Y[index_test[i]]

del condition, i, ages_intervals

print(f'Data processed. Created {N_intervals} datasets')
for i in range(N_intervals):
    print(f'{i+1} -> Train {len(index_train[i])} rows, Test {len(index_test[i])} rows')


#%%

# Compute predictions

LL          = [None] * N_intervals
Y_test_pred = [None] * N_intervals

results = {}
time_0  = time.time()

for i in range(N_intervals):

    print(f'Computing prediction {i+1} of {N_intervals}')
    
    if config.PCA in ['first', 'first0',  'first_R','all']:
        
        # By the moment ! using all PCA all components
        
        scaler = StandardScaler()
        scaler.fit(X_train[i])
        
        X_train_pca = scaler.transform(X_train[i].copy())
        Y_train_pca = scaler.transform(Y_train[i].copy())
        X_test_pca  = scaler.transform(X_test[i].copy())        
        
        #MAKE 0 the less important variables !!!!!!!!!!
        pca = PCA(n_components = X_train_pca.shape[1])
        
        if config.PCA == 'first_B':
            filter = filter_points_axis(X_train_pca, Y_train_pca, proportion = config.PCA_proportion)
            pca.fit(X_train_pca[filter])
        else:
            pca.fit(X_train_pca)
        
        X_train_pca  = pca.transform(X_train_pca)
        Y_train_pca  = pca.transform(Y_train_pca)
        X_test_pca   = pca.transform(X_test_pca)
        
        if config.PCA == 'first0':
            
            # Drop less important components
            PC_used = pca.explained_variance_ratio_ > 0.005
            print(f'Using {np.sum(PC_used)} components of {len(PC_used)}')
            
            X_train_pca[:,~PC_used] = 0
            Y_train_pca[:,~PC_used] = 0
            X_test_pca[:,~PC_used]  = 0
        
        if config.ext_type in ['MW_sm', 'OM_sm']:
            Y_train_pca = kernel_smoothing(X_train_pca, Y_train_pca, config.sigma)
        else:
            Y_train_pca  = duplicate_mean(X_train_pca, Y_train_pca, show_progess = False)
        
        LL[i] = lattice_Lipschitz(X_train_pca, Y_train_pca)
        
        if config.ext_type in ["MW", "MW_sm"]:
            LL[i].compute_phi_for()
            #print(LL[i].phi)
            Y_test_pred_pca = LL[i].McShane_Whitney_multiple_for(X_test_pca)
        elif config.ext_type in ["OM", "OM_sm"]:
            Y_test_pred_pca = LL[i].Oberman_Milman_multiple_for(X_test_pca)
        else:
            print("Unknow method to compute the extension.")
            
        Y_test_pred_pca = pca.inverse_transform(Y_test_pred_pca)
        Y_test_pred[i]  = scaler.inverse_transform(Y_test_pred_pca).copy()
                
    else:
        
        if config.ext_type in ['MW_sm', 'OM_sm']:
            Y_train_ = kernel_smoothing(X_train[i], Y_train[i], config.sigma)
        else:
            Y_train_  = duplicate_mean(X_train[i], Y_train[i], show_progess = False)
    
        LL[i] = lattice_Lipschitz(X_train[i], Y_train_)

        if config.ext_type in ["MW", "MW_sm"]:
            LL[i].compute_phi_for()
            Y_test_pred[i] = LL[i].McShane_Whitney_multiple_for(X_test[i])
        elif config.ext_type in ["OM", "OM_sm"]:
            Y_test_pred[i] = LL[i].Oberman_Milman_multiple_for(X_test[i])
        else:
            print("Unknow method to compute the extension.")

results['time'] = time.time() - time_0          
results['error_parts'] = [NRMSE(a,b) for a, b in zip(Y_test, Y_test_pred)]

# The sets (particularly) Y_test are divided into diferent ages intervals
# We join them now in a new dataframe

X_train     = np.concatenate(X_train,     axis = 0)
Y_train     = np.concatenate(Y_train,     axis = 0)
X_test      = np.concatenate(X_test,      axis = 0)
Y_test      = np.concatenate(Y_test,      axis = 0)
Y_test_pred = np.concatenate(Y_test_pred, axis = 0)
index_train = np.concatenate(index_train, axis = 0)
index_test  = np.concatenate(index_test,  axis = 0)

results['error'] = NRMSE(Y_test, Y_test_pred)

print(f"Error: {np.round(results['error']*100,2)} %. " + \
      f"Computing time: {np.round(results['time'])} s.")

#%%

# Save

# numpy arrays
save_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_{config_num}_np.h5')
with h5py.File(save_path, 'w') as file:
    file.create_dataset('X_train',      data = X_train)
    file.create_dataset('Y_train',      data = Y_train)
    file.create_dataset('X_test',       data = X_test)
    file.create_dataset('Y_test',       data = Y_test)
    file.create_dataset('Y_test_pred',  data = Y_test_pred)
    file.create_dataset('index_train',  data = index_train)
    file.create_dataset('index_test',   data = index_test)

# Dict (results info)
save_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_{config_num}_dict.h5')
with h5py.File(save_path, 'w') as file:
    for key, value in results.items():
        file.create_dataset(key, data = value)

# Conversion to avoid errors
players_info['player_name'] = players_info['player_name'].astype(str)
players_info['player_known_name'] = players_info['player_known_name'].astype(str)

# pandas datasets (saved in a diferent )
save_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_{config_num}_pd.h5')
with pd.HDFStore(save_path) as store:
    store['players_info']       = players_info