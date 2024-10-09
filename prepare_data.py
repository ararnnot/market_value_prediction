# A. Roger Arnau
# 21 dic 23

#%%

import  pandas  as  pd
import  numpy   as  np
import  os
from    tqdm    import  tqdm

#%%

current_directory = os.getcwd()
csv_save_path   = os.path.join(current_directory, 'data','data_processed.h5')
players         =   pd.read_hdf(csv_save_path, 'players')
player_name     =   pd.read_hdf(csv_save_path, 'player_name')
team_name       =   pd.read_hdf(csv_save_path, 'team_name')
competition_name =  pd.read_hdf(csv_save_path, 'competition_name')
season_name     =   pd.read_hdf(csv_save_path, 'season_name')

player_name     =   pd.merge(player_name,
                             players[['player_id', 'country_id']],
                             on = 'player_id', how = 'left')
player_name.drop_duplicates(inplace = True)
players.drop(columns = 'country_id', inplace = True)

# %%

# Only european leagues
competition_european_id = [9, 2, 3, 6, 12, 84, 46, 11, 42, 13, 7]
players.query("competition_id in @competition_european_id", inplace = True)
# No use data of 2022/2023, incomplete by the moment
seasons_complete = [42, 90, 108]
players.query("season_id in @seasons_complete", inplace = True)
# Discard players with less than 500 minuts played
players.query("player_season_minutes >= 500", inplace = True)
players.drop(columns = 'player_season_minutes', inplace = True)


#%%

# We prepare now a dataframe in wich duplicate player-season
# (for example, change of a team in the middle of the season)
# form only one row as the mean of boths

remove_columns = ['team_id', 'competition_id']
columns = ['team_id_A', 'team_id_B',
           'competition_id_A', 'competition_id_B'] + \
          [col for col in players.columns.to_list() if col not in remove_columns]
players_filtered = pd.DataFrame(columns = columns)

index = players.index
repeated = [False] * (index.max()+1)

for i in tqdm(index, desc = 'Checking duplicates'):

    # Check if this player-season has been copied before
    if repeated[i]:
        continue

    player_id       =   players.at[i, 'player_id']
    season_id       =   players.at[i, 'season_id']

    full = players.query('(player_id == @player_id) and (season_id == @season_id)')
    if len(full) == 2:

        # Players that are repeated
        index_full = full.index
        repeated[index_full[0]] = True
        repeated[index_full[1]] = True

        remove_columns = ['player_id', 'team_id',
                          'competition_id', 'season_id']
        add_player = pd.DataFrame([full.drop(columns = remove_columns).mean()],
                                  index = [i])
        previous_columns = add_player.columns.to_list()

        add_player['player_id']         =   player_id
        add_player['season_id']         =   season_id
        add_player['team_id_A']         =   full.at[index_full[0], 'team_id']
        add_player['team_id_B']         =   full.at[index_full[1], 'team_id']
        add_player['competition_id_A']  =   full.at[index_full[0], 'competition_id']
        add_player['competition_id_B']  =   full.at[index_full[1], 'competition_id']

        add_player = add_player[['player_id', 'season_id',
                                 'team_id_A', 'team_id_B',
                                 'competition_id_A', 'competition_id_B'] + \
                                previous_columns]
        
    else :

        # Players only once
        remove_columns = ['player_id', 'team_id',
                          'competition_id', 'season_id']
        add_player          =   full.drop(columns = remove_columns).copy()
        previous_columns    =   add_player.columns.to_list()
        [team]              =   full['team_id']
        [competition]       =   full['competition_id']

        add_player['player_id']         =   player_id
        add_player['season_id']         =   season_id
        add_player['team_id_A']         =   team
        add_player['team_id_B']         =   team
        add_player['competition_id_A']  =   competition
        add_player['competition_id_B']  =   competition

        add_player = add_player[['player_id', 'season_id',
                                 'team_id_A', 'team_id_B',
                                 'competition_id_A', 'competition_id_B'] + \
                                previous_columns]
        
    players_filtered = players_filtered.append(add_player.astype(float))

save_path = os.path.join(current_directory, 'data', 'players_filtered.h5')
with pd.HDFStore(save_path) as store:
    store['players']        =   players_filtered

# %%

# Create the X (year) and Y (next year) for playes we have data of

columns = players_filtered.columns
X = pd.DataFrame(columns = columns)
Y = pd.DataFrame(columns = columns)
remove_columns = ['player_id', 'season_id',
                  'team_id_A', 'team_id_B',
                  'competition_id_A', 'competition_id_B']
X = X.drop(columns = remove_columns)
Y = Y.drop(columns = remove_columns)

# 36813	Janio Bikel Figueiredo Silva has change of country
player_name.query('(player_id != 36813) or (country_id != 96)', inplace = True)

index = players_filtered.index
players_info = pd.merge(players_filtered[remove_columns], player_name,
                        on = 'player_id', how = 'left' ).set_index(index)


not_matched = 0
for i in tqdm(index, desc = 'Creating X and Y') :
    
    act_s   =   players_filtered.at[i, 'season_id']
    player  =   players_filtered.at[i, 'player_id']
    
    # Seasons: 42, 90, 108, 235 (incomplete)
    if act_s == 42:
        next_s = 90
    elif act_s == 90:
        next_s = 108
    elif act_s == 108:
        next_s = 235
        not_matched += 1
        continue

    full = players_filtered.query('player_id == @player').copy()
    if not full['season_id'].isin([next_s]).any():
        not_matched += 1
        continue

    data_act    =   full.query('season_id == @act_s').copy()
    data_act.drop(columns = remove_columns, inplace = True)
    data_next   =   full.query('season_id == @next_s').copy()
    data_next.drop(columns = remove_columns, inplace = True)

    if len(data_act) != 1 or len(data_next) != 1:
        raise ValueError("Not find exactly one row for each season.")
    
    X = X.append(data_act)
    Y = Y.append(data_next)

print(f"Players not matched with the corresponding season: {not_matched}")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# %%

# Add varibale new_season_id to later merge.

players_info.loc[players_info.season_id == 42,  'next_season_id'] = 90
players_info.loc[players_info.season_id == 90,  'next_season_id'] = 108
players_info.loc[players_info.season_id == 108, 'next_season_id'] = 235

players_next = players_info.loc[:, ['season_id',
                                    'team_id_A', 'team_id_B',
                                    'competition_id_A', 'competition_id_B']]
players_next = players_next.add_prefix('next_')
players_next['player_id'] = players_info['player_id']

players_info = players_info.loc[X.index]
players_info = pd.merge(players_info, players_next,
                        how = 'left',
                        left_on  = ['player_id', 'next_season_id'],
                        right_on = ['player_id', 'next_season_id'])

players_info.reset_index(inplace = True, drop = True)
X.reset_index(inplace = True, drop = True)
X = X.astype(float)
Y.reset_index(inplace = True, drop = True)
Y = Y.astype(float)

#%%

# Save

save_path = os.path.join(current_directory, 'data', 'data_XY.h5')
with pd.HDFStore(save_path) as store:
    store['X']              =   X
    store['Y']              =   Y
    store['players_info']   =   players_info

# %%
