#%%

import  pandas  as  pd
import  numpy   as  np
import  os
import  re
from    tqdm    import  tqdm

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def season_next(season):
    if season == 42  : return 90
    if season == 90  : return 108
    if season == 108 : return 235
    return np.nan

#%%

current_directory = os.getcwd()
csv_file_path = os.path.join(current_directory,
                             'data', 'prices_raw.csv')
csv_info_path = os.path.join(current_directory,
                             'data', 'data_processed.h5')
csv_play_path = os.path.join(current_directory,
                             'data', 'players_filtered.h5')

prices              =   pd.read_csv(csv_file_path, encoding = 'unicode_escape')
competition_name    =   pd.read_hdf(csv_info_path, 'competition_name')
player_name         =   pd.read_hdf(csv_info_path, 'player_name')
players             =   pd.read_hdf(csv_play_path, 'players')

#%%

# From TM to DataFrame

values = pd.DataFrame(columns = ['player_name', 'country',
                                 'year', 'season_id', 'price'])

for i in tqdm(range(len(prices) - 1)):

    name    =   prices.at[i, 'table.player_name']
    price   =   prices.at[i+1, 'table.player_name']
    country =   prices.at[i, 'country']
    year    =   prices.at[i, 'year']

    if has_numbers(name):
        continue
    if not has_numbers(price):
        continue

    try:
        if price[-1] == 'k':
            value = int(float(price[1:-1])*1e3)
        elif price[-1] == 'm':
            value = int(float(price[1:-1])*1e6)
        else:
            print(f"Non convertible price: {price}")
            continue
    except ValueError as e:
        print(f"Error {e}: with string {price}")

    if   year == 2019 : season_id = 42
    elif year == 2020 : season_id = 90
    elif year == 2021 : season_id = 108
    elif year == 2022 : season_id = 235
    else : season_id = np.nan

    values = pd.concat([values,
                        pd.DataFrame({
                            'player_name'  : [name],
                            'country'      : [country],
                            'year'         : [year],
                            'season_id'    : [season_id],
                            'price'        : [value]
                        })], ignore_index = True)

print(values.shape)

# %%
    
# Drop players with the same name and join with players
    
values.drop_duplicates(subset = ['player_name', 'season_id'],
                       keep = False,
                       inplace = True)
    
values = pd.merge(values,
                  player_name.loc[:, ['player_id', 'player_name']],
                  on = 'player_name',
                  how = 'left')

values = pd.merge(values,
                  player_name.loc[:, ['player_id', 'player_known_name']].dropna(),
                  left_on = 'player_name',
                  right_on = 'player_known_name',
                  how = 'left')

values['player_id'] = np.where(pd.notna(values['player_id_x']), 
                               values['player_id_x'],
                               values['player_id_y'])

values.drop(columns = ['player_id_x', 'player_id_y', 'player_known_name'],
            inplace = True)
values = values[['player_id', 'season_id', 'price']].copy()


### (!) Some players have the same name, so we discard it (about 20 percent)

#%%

players = pd.merge(players, values,
                   on = ['player_id', 'season_id'],
                   how = 'left')


players['season_next_id'] = players['season_id'].apply(season_next)
players = pd.merge(players, values,
                   left_on  = ['player_id', 'season_next_id'],
                   right_on = ['player_id', 'season_id'],
                   how = 'left')

players.drop(columns = ['season_next_id', 'season_id_y'], inplace = True)
players.rename(columns = {'season_id_x' : 'season_id',
                          'price_x'     : 'price',
                          'price_y'     : 'price_next_season'},
               inplace = True)

players = players.astype(float)
values  = values.astype(float)

 #%%   

# Save

save_path = os.path.join(current_directory,
                         'data', 'players_value.h5')
with pd.HDFStore(save_path) as store:
    store['players_value']  =   players
    store['values']    =   values

# %%
