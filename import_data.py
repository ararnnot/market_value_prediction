# A. Roger Arnau
# 19 dic 23

#%%

import  pandas  as  pd
import  numpy   as  np
import  os

#%%

current_directory = os.getcwd()
csv_file_path = os.path.join(current_directory,
                             r'data/all_players_per_season.csv')
players = pd.read_csv(csv_file_path, encoding='unicode_escape')


players = players.drop('Unnamed: 0', axis = 1)
# remove columns with all NA
players = players.dropna(axis=1, how='all')
# remove players withot birth date, weight, height
players = players.dropna(subset=['birth_date',
                                 'player_weight',
                                 'player_height',
                                 'country_id'])

# calculate (aproximatelly) age at 31-dec
players['age']  =   players['season_name'].str[0:4].astype(int) - \
                        players['birth_date'].str[0:4].astype(int)

player_name     =   players.loc[:, ['player_id', 'player_name',
                                    'player_known_name']]
player_name     =   player_name.drop_duplicates()
team_name       =   players.loc[:, ['team_id', 'team_name']]
team_name       =   team_name.drop_duplicates()
competition_name =  players.loc[:, ['competition_id', 'competition_name']]
competition_name =  competition_name.drop_duplicates()
season_name     =   players.loc[:, ['season_id', 'season_name']]
season_name     =   season_name.drop_duplicates()

# remove columns of text and others
players =   players.drop(['player_name', 'team_name',
                          'competition_name', 'season_name',
                          'birth_date', 'player_female',
                          'player_first_name', 'player_last_name',
                          'player_known_name',
                          'primary_position', 'secondary_position',
                          'player_season_most_recent_match',
                          'player_season_360_minutes'],
                         axis=1)

# all other NA are ratios, proportions... set to 0
players = players.fillna(0)

# %%

# SAVE DATA

csv_save_path = os.path.join(current_directory,
                             r'data\data_processed.h5')
with pd.HDFStore(csv_save_path) as store:
    store['players']        =   players
    store['player_name']    =   player_name
    store['team_name']      =   team_name
    store['competition_name'] = competition_name
    store['season_name']    =   season_name
