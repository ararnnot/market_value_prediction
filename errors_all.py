#%%

import  numpy           as      np
import  pandas          as      pd
import  os
import  time
import  pickle
import  h5py

from    importlib                   import import_module

current_directory   =   os.getcwd()

configs_all = [1, 2, 3, 5, 6]

for config_num in configs_all:

    config_file = import_module(f"configs.all.config_{config_num}")
    config = config_file.configuration
    del config_file

    results_ny = {}
    file_path = os.path.join(current_directory, 'data_pred_next_year',
                    f'predictions_next_year_config_{config.config_next_year}_dict.h5')
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            results_ny[key] = file[key][()]

    results_mv = {}
    file_path = os.path.join(current_directory, 'data_pred_mvalue',
                    f'info_mvalue_{config.config_market_value}_dict.h5')
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            results_mv[key] = file[key][()]

    file_path = os.path.join(current_directory, 'data_predict_all',
                f'predictions_all_{config_num}_dict.h5')
    results = {}
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            results[key] = file[key][()]

    print(f"=================")
    print(f"Config {config_num}")
    print(f"Config next_year {config.config_next_year}")
    for key in results_ny.keys():
        print(f" {key}: {results_ny[key]}")
    print(f"Config market_value {config.config_market_value}")
    for key in results_mv.keys():
        print(f" {key}: {results_mv[key]}")
    print(f"All")
    for key in results.keys():
        print(f" {key}: {results[key]}")
