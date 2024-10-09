#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  h5py
import  pickle

from    importlib               import import_module
from    utils.functions         import NRMSE_1D, kernel_smoothing_one
from    tqdm                    import tqdm
from    utils.lipschits_regression import Lipschitz_regression
from sklearn.preprocessing import StandardScaler

# Import predefined configuration
config_num = 4
if config_num is None :
    config_file = import_module("configs.all_direct.config")
else :
    config_file = import_module(f"configs.all_direct.config_{config_num}")
config = config_file.configuration
del config_file
#%%

print(f'Starting training the Lipschitz benchmark model. \n' + \
      f'Model: {config_num}. \n')

# Import results from predict_next_year

current_directory   =   os.getcwd()

# Load predict_next_year data from config 1

file_path = os.path.join(current_directory, 'data_pred_next_year',
               f'predictions_next_year_config_1_np.h5')
with h5py.File(file_path, 'r') as file:
    X_train           = np.array(file['X_train'])
    Y_train           = np.array(file['Y_train'])
    X_test            = np.array(file['X_test'])
    Y_test            = np.array(file['Y_test'])
    Y_test_pred       = np.array(file['Y_test_pred'])
    index_train_ny    = np.array(file['index_train'])
    index_test_ny     = np.array(file['index_test'])
# those index refer to players_info

file_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_1_pd.h5')
with pd.HDFStore(file_path) as store:
    players_info        = store['players_info']

# Load Lipschitz_regression object of predict_market_value from config 1

file_path = os.path.join(current_directory, 'data_pred_mvalue',
                 f'predictions_mvalue_1_pd.h5')
with pd.HDFStore(file_path) as store:
    players_values  = store['players_values']
    index_train_mv  = store['index_train']
    index_test_mv   = store['index_test']
# those index refer to players_values

# %%

# Players to predict value

# Players to predict value
players_value_train = players_values[['player_id', 'season_id', 'price']] \
                        .query(f'season_id in {config.seasons_train}') \
                        .dropna(subset = 'price')
players_value_test  = players_values[['player_id', 'season_id', 'price']] \
                        .query(f'season_id in {config.seasons_test}') \
                        .dropna(subset = 'price')

# Players that can be used for the final model
# Data prediction_next_year AND data of market value
players_train = pd.merge(
    players_info.loc[index_train_ny, ['player_id', 'season_id', 'next_season_id']],
    players_value_train,
    left_on  = ['player_id', 'next_season_id'],
    right_on = ['player_id', 'season_id'],
    how = 'left'
)
players_test = pd.merge(
    players_info.loc[index_test_ny, ['player_id', 'season_id', 'next_season_id']],
    players_value_test,
    left_on  = ['player_id', 'next_season_id'],
    right_on = ['player_id', 'season_id'],
    how = 'left'
)
# Players test and (X,Y)_test sorted in the same way (index)
players_train['season_id'] = players_train['season_id_x']
players_train.drop(columns = ['season_id_x', 'season_id_y', 'next_season_id'],
                  inplace = True)
players_train.dropna(subset = 'price', inplace = True)
index_train = players_train.index
players_test['season_id'] = players_test['season_id_x']
players_test.drop(columns = ['season_id_x', 'season_id_y', 'next_season_id'],
                  inplace = True)
players_test.dropna(subset = 'price', inplace = True)
index_test = players_test.index

print(f'Loaded data.')
print(f'Train {len(players_train)} from {len(X_train)} and {len(players_value_train)}.')
print(f'Test {len(players_test)} from {len(X_test)} and {len(players_value_test)}.')

X_train    = X_train[index_train]
Y_train    = Y_train[index_train]
Z_train = np.array(players_train['price'])

X_test    = X_test[index_test]
Y_test    = Y_test[index_test]
Z_test = np.array(players_test['price'])


#%% Train: X -> Z

time_0 = time.time()

if config.ext_type in ['MW_sm']:
    scaler  = StandardScaler()
    Z_train = kernel_smoothing_one(
        scaler.fit_transform(X_train),
        Z_train,
        sigma = 0.01*(150)**0.5)

LR = Lipschitz_regression(X_train, Z_train)

times = pd.DataFrame()

# Normalize
if config.normalize == 'sd':
    LR.normalize_compute_m_sd()
    LR.normalize()    
    X_test_new = LR.normalize_new_data(X_test)
else:
    X_test_new = X_test

results = {}

if config.ext_type in ['MW', 'MW_sm']:
    LR.compute_K_2()
    Z_test_pred = LR.McShane_Whitney_multiple_2(X_test_new)
elif config.ext_type == 'OM':
    Z_test_pred = LR.Oberman_Milman_multiple_for(X_test_new)
elif config.ext_type == 'ASM':
    Z_test_pred = LR.slope_2_average_all(X_test_new)
else:
    print('Unknown extension type')

results['time'] = time.time() - time_0
results['error'] = NRMSE_1D(Z_test, Z_test_pred)

print(f"Error: {np.round(results['error']*100,2)} %. " + \
      f"Computing time: {np.round(results['time'],2)} s.")

# %%

# SAVE

save_path = os.path.join(current_directory, 'data_predict_all_direct',
                         f'predictions_all_{config_num}_pd.h5')
with pd.HDFStore(save_path) as store:
    store['index']          = index_test.to_series()

save_path = os.path.join(current_directory, 'data_predict_all_direct',
                f'predictions_all_{config_num}_dict.h5')
with h5py.File(save_path, 'w') as file:
    for key, value in results.items():
        file.create_dataset(key, data = value)

save_path = os.path.join(current_directory, 'data_predict_all_direct',
                         f'predictions_all_{config_num}_.h5')
with h5py.File(save_path, 'w') as file:
    file.create_dataset('X_test',      data = X_test)
    file.create_dataset('Y_test',      data = Y_test)
    file.create_dataset('Z_test',      data = Z_test)
    file.create_dataset('Z_test_pred', data = Z_test_pred)
    
# Save the results dictionary
save_path = os.path.join(current_directory, 'data_predict_all_direct',
                         f'errors_all_{config_num}.pkl')
with open(save_path, 'wb') as file:
    pickle.dump(
        {'config_num': config_num,
         'total_error': results['error'],
         'total_time': results['time']
        },
        file)
    