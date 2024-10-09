#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  h5py
import  pickle

from    importlib               import import_module
from    utils.functions         import NRMSE_1D
from    tqdm                    import tqdm
from    utils.lipschits_regression import Lipschitz_regression

# Import predefined configuration
config_num = 46
if config_num is None :
    config_file = import_module("configs.all.config")
else :
    config_file = import_module(f"configs.all.config_{config_num}")
config = config_file.configuration
del config_file

if config.config_market_value is None :
    config_file = import_module("configs.market_value.config")
else :
    config_file = import_module("configs.market_value.config_" + \
                                str(config.config_market_value))
config_mv = config_file.configuration
del config_file

#%%

print(f'Starting training the final model. \n' + \
      f'Model: {config_num}. \n' + \
      f'Model_next_year: {config.config_next_year}. ' + \
      f'Model_market_value: {config.config_market_value}.')

# Import results from predict_next_year

current_directory   =   os.getcwd()

# Load predict_next_year data

file_path = os.path.join(current_directory, 'data_pred_next_year',
               f'predictions_next_year_config_{config.config_next_year}_np.h5')
with h5py.File(file_path, 'r') as file:
    X_train           = np.array(file['X_train'])
    Y_train           = np.array(file['Y_train'])
    X_test            = np.array(file['X_test'])
    Y_test            = np.array(file['Y_test'])
    Y_test_pred       = np.array(file['Y_test_pred'])
    index_train_ny    = np.array(file['index_train'])
    index_test_ny     = np.array(file['index_test'])
# those index refer to players_info

results_ny = {}
file_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_{config.config_next_year}_dict.h5')
with h5py.File(file_path, 'r') as file:
    for key in file.keys():
        results_ny[key] = file[key]

file_path = os.path.join(current_directory, 'data_pred_next_year',
                f'predictions_next_year_config_{config.config_next_year}_pd.h5')
with pd.HDFStore(file_path) as store:
    players_info        = store['players_info']


# Load Lipschitz_regression object of predict_market_value

file_path = os.path.join(current_directory, 'data_pred_mvalue',
                 f'predictions_mvalue_{config.config_market_value}_pd.h5')
with pd.HDFStore(file_path) as store:
    players_values  = store['players_values']
    index_train_mv  = store['index_train']
    index_test_mv   = store['index_test']
# those index refer to players_values

results_mv = {}
file_path = os.path.join(current_directory, 'data_pred_mvalue',
                f'info_mvalue_{config.config_market_value}_dict.h5')
with h5py.File(file_path, 'r') as file:
    for key in file.keys():
        results_mv[key] = file[key]

file_path = os.path.join(current_directory, 'data_pred_mvalue',
                f'predictor_mvalue_{config.config_market_value}.pkl')
with open(file_path, 'rb') as file:
    LR = pickle.load(file)

# %%

# Players to predict value
players_value_test = players_values[['player_id', 'season_id', 'price']] \
                        .query(f'season_id in {config.seasons_test}') \
                        .dropna(subset = 'price')

# Players that can be used for the final model
# Data prediction_next_year AND data of market value
players_test = pd.merge(
    players_info.loc[index_test_ny, ['player_id', 'season_id', 'next_season_id']],
    players_value_test,
    left_on  = ['player_id', 'next_season_id'],
    right_on = ['player_id', 'season_id'],
    how = 'left'
)
# Players test and (X,Y)_test sorted in the same way (index)

players_test['season_id'] = players_test['season_id_x']
players_test.drop(columns = ['season_id_x', 'season_id_y', 'next_season_id'],
                  inplace = True)
players_test.dropna(subset = 'price', inplace = True)

index = players_test.index
print(f'Loaded data: {len(players_test)} aviabale common data from ' + \
      f'{len(X_test)} and {len(players_value_test)}.')

X_test    = X_test[index]
Y_test    = Y_test[index]
Y_test_pred = Y_test_pred[index]

# New years price
Y_test_pred = LR.normalize_new_data(Y_test_pred)
Z_test = np.array(players_test['price'])

results = {}

time_0 = time.time()

if config_mv.ext_type == "MW":
    # Recompute K with time objectives
    LR.compute_K_2()
    Z_test_pred = LR.McShane_Whitney_multiple_2(Y_test_pred)
elif config_mv.ext_type == "OM":
    Z_test_pred = LR.Oberman_Milman_multiple_for(Y_test_pred)
elif config_mv.ext_type == 'ASM':
    Z_test_pred = LR.slope_2_average_all(Y_test_pred)
else:
    print("Unknown extension type.")

results['time']  = time.time() - time_0
results['error'] = NRMSE_1D(Z_test, Z_test_pred)

print(f"Error: {np.round(results['error']*100,2)} %. " + \
      f"Computing time: {np.round(results['time'])} s.")

# %%

# SAVE

save_path = os.path.join(current_directory, 'data_predict_all',
                         f'predictions_all_{config_num}_pd.h5')
with pd.HDFStore(save_path) as store:
    store['index']          = index.to_series()

save_path = os.path.join(current_directory, 'data_predict_all',
                f'predictions_all_{config_num}_dict.h5')
with h5py.File(save_path, 'w') as file:
    for key, value in results.items():
        file.create_dataset(key, data = value)

save_path = os.path.join(current_directory, 'data_predict_all',
                         f'predictions_all_{config_num}_.h5')
with h5py.File(save_path, 'w') as file:
    file.create_dataset('X_test',      data = X_test)
    file.create_dataset('Y_test',      data = Y_test)
    file.create_dataset('Z_test',      data = Z_test)
    file.create_dataset('Z_test_pred', data = Z_test_pred)
    
# Save the results dictionary
save_path = os.path.join(current_directory, 'data_predict_all',
                         f'errors_all_{config_num}.pkl')
with open(save_path, 'wb') as file:
    pickle.dump(
        {'config_num': config_num,
         'config_ny':  config.config_next_year,
         'config_mv':  config.config_market_value,
         'total_error': results['error'],
         'total_time': results['time']
        },
        file)
    