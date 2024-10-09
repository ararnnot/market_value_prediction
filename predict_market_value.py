#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  pickle
import  h5py

from    importlib                   import import_module
from    utils.functions             import  NRMSE_1D
from    utils.lipschits_regression  import Lipschitz_regression

# Import predefined configuration
config_num = 2
if config_num is None :
    config_file = import_module("configs.market_value.config")
else :
    config_file = import_module(f"configs.market_value.config_{config_num}")
config = config_file.configuration
del config_file

#%%

print(f'Starting training of market value predictions. \n' + \
      f'Model: {config_num}. \n' + \
      f'Type of extension: {config.ext_type}.')

# Import data

current_directory   =   os.getcwd()

file_path           =   os.path.join(current_directory, 'data', 'players_value.h5')
players_values      =   pd.read_hdf(file_path, 'players_value')

# INFO: players_values  has statistics of THIS season
#                       and values of THIS and NEXT season

if config.fast:
    subset = np.random.choice(np.arange(0, len(players_values)),
                              size = 500,
                              replace = False)
    players_values = players_values.loc[subset].reset_index()
    del subset

N = len(players_values)
print('Data loaded')
print('Players with statisticas and price: ' + \
      f'{N - np.sum(np.isnan(players_values.price))} of {N}')
print('Players with statisticas and next season price: ' + \
      f'{N - np.sum(np.isnan(players_values.price_next_season))} of {N}')
del file_path, N

#%%

# Ceate datasets for Lipschitz (all 150 colums withs statistics)

players_train = players_values.dropna(subset = 'price') \
                    .query(f'season_id in {config.seasons_train}')
players_test  = players_values.dropna(subset = 'price') \
                    .query(f'season_id in {config.seasons_test}')
index_train   = players_train.index.to_series()
index_test    = players_test.index.to_series()
Y_train = np.array(players_train.dropna(subset = 'price').loc[:, 'player_weight':'age'])
Z_train = np.array(players_train.dropna(subset = 'price').loc[:, 'price'])
Y_test  = np.array(players_test.dropna(subset = 'price').loc[:, 'player_weight':'age'])
Z_test  = np.array(players_test.dropna(subset = 'price').loc[:, 'price'])

print(f'Created dataframe. Data: train {len(Y_train)}, test {len(Y_test)}, ' + \
      f'variables {Y_test.shape[1]}')

#%%

# Train: Y -> Z

LR = Lipschitz_regression(Y_train, Z_train)

times = pd.DataFrame()

# Normalize
if config.normalize == 'sd':
    LR.normalize_compute_m_sd()
    LR.normalize()    
    Y_test_new = LR.normalize_new_data(Y_test)
else:
    Y_test_new = Y_test

results = {}
time_0 = time.time()

if config.ext_type == 'MW':
    LR.compute_K_2()
    Z_test_pred = LR.McShane_Whitney_multiple_2(Y_test_new)
elif config.ext_type == 'OM':
    Z_test_pred = LR.Oberman_Milman_multiple_for(Y_test_new)
    #Z_test_pred = LR.Oberman_Milman_2_all(Y_test_new)
elif config.ext_type == 'ASM':
    Z_test_pred = LR.slope_2_average_all(Y_test_new)
else:
    print('Unknown extension type')

results['time'] = time.time() - time_0
results['error'] = NRMSE_1D(Z_test, Z_test_pred)

print(f"Error: {np.round(results['error']*100,2)} %. " + \
      f"Computing time: {np.round(results['time'])} s.")

# %%

# SAVE

save_path = os.path.join(current_directory, 'data_pred_mvalue',
                         f'predictions_mvalue_{config_num}_pd.h5')
with pd.HDFStore(save_path) as store:
    store['players_values'] = players_values
    store['index_train']    = index_train
    store['index_test']     = index_test

# Dict (results info)
save_path = os.path.join(current_directory, 'data_pred_mvalue',
                f'info_mvalue_{config_num}_dict.h5')
with h5py.File(save_path, 'w') as file:
    for key, value in results.items():
        file.create_dataset(key, data = value)

save_path = os.path.join(current_directory, 'data_pred_mvalue',
                         f'predictor_mvalue_{config_num}.pkl')
with open(save_path, 'wb') as file:
    pickle.dump(LR, file)