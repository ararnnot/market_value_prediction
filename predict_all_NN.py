#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  h5py
import  pickle
import  random

from    importlib               import import_module
from    utils.functions         import NRMSE_1D

import  tensorflow as tf

# Import predefined configuration
config_num = 2
if config_num is None :
    config_file = import_module("configs.all_NN.config")
else :
    config_file = import_module(f"configs.all_NN.config_{config_num}")
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

print(f'Starting training the final model with Neural Netwrok. \n' + \
      f'Model: {config_num}. \n' + \
      f'Model_next_year: {config.config_next_year}. ' + \
      f'Model_market_value: {config.config_market_value}.')

# Import results from predict_next_year

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

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

if False:
    results_mv = {}
    file_path = os.path.join(current_directory, 'data_pred_mvalue',
                    f'info_mvalue_{config_num}_dict.h5')
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            results_mv[key] = file[key]

file_path = os.path.join(current_directory, 'data_pred_mvalue',
                f'predictor_mvalue_{config.config_market_value}.pkl')
with open(file_path, 'rb') as file:
    LR = pickle.load(file)

# %%

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

#X_train    = X_test[index_train]
#Y_train    = Y_test[index_train]
#Z_train = np.array(players_train['price'])

X_train    = X_train[index_train]
Y_train    = Y_train[index_train]
Z_train = np.array(players_train['price'])

X_test    = X_test[index_test]
Y_test    = Y_test[index_test]
Z_test = np.array(players_test['price'])


# Normalize

means  =   np.mean(X_train, axis = 0) 
stds   =   np.std(X_train, axis = 0)
const_col = stds == 0 
cols_divide = [not z for z in const_col]

X_train = X_train - means
X_train[:, cols_divide] = X_train[:, cols_divide] / stds[cols_divide]

X_test = X_test - means
X_test[:, cols_divide] = X_test[:, cols_divide] / stds[cols_divide]

# Neural Network

h_layers = config.h_layers + [1]
fun_act = [config.fun_act] * len(h_layers)
fun_act[-1] = 'linear'
model = tf.keras.models.Sequential(
  [tf.keras.layers.Dense(n, activation = a) for n, a in zip(h_layers, fun_act)] 
)

#%%

results = {}
time_0 = time.time()

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['mean_squared_error'])
history = model.fit(X_train, Z_train, epochs = 100, batch_size=32, verbose=0)

results['time']  = time.time() - time_0
print(model.summary())

Z_test_pred = model.predict(X_test).flatten()
results['error'] = NRMSE_1D(Z_test, Z_test_pred)

print(f"Error: {np.round(results['error']*100,2)} %. " + \
      f"Computing time: {np.round(results['time'],2)} s.")

# %%

# SAVE

save_path = os.path.join(current_directory, 'data_predict_all_NN',
                         f'predictions_all_{config_num}_pd.h5')
with pd.HDFStore(save_path) as store:
    store['index_train']  = index_train.to_series()
    store['index_test']   = index_test.to_series()

save_path = os.path.join(current_directory, 'data_predict_all_NN',
                f'predictions_all_{config_num}_dict.h5')
with h5py.File(save_path, 'w') as file:
    for key, value in results.items():
        file.create_dataset(key, data = value)

save_path = os.path.join(current_directory, 'data_predict_all_NN',
                         f'predictions_all_{config_num}_.h5')
with h5py.File(save_path, 'w') as file:
    file.create_dataset('X_train',     data = X_train)
    file.create_dataset('Y_train',     data = Y_train)
    file.create_dataset('Z_train',     data = Z_train)
    file.create_dataset('X_test',      data = X_test)
    file.create_dataset('Y_test',      data = Y_test)
    file.create_dataset('Z_test',      data = Z_test)
    file.create_dataset('Z_test_pred', data = Z_test_pred)