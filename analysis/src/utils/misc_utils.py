import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Dict, Set, List

"""This module provides miscellaneous utility constants and functions, whether for working with raw data
or creating visualizations."""

# Mapping from each team ID to their full name, including active years (duplicate full names otherwise)

teams = pd.read_csv('../data/teams.csv')

# Merge city and nickname into one 'fullname' column
teams['fullname'] = teams['CITY'] + ' ' + teams['NICKNAME'] + ' (' + teams['FIRST'].astype(str) + ' - ' + teams['LAST'].astype(str) + ')'

TEAM_FULLNAME_MAP = {team: full for team, full in zip(teams['TEAM'], teams['fullname'])}

def get_prev_date_midnight(dt: pd.Timestamp) -> pd.Timestamp:
    """For the given timestamp, gets the timestamp for the previous day at midnight."""
    return dt.normalize() + pd.Timedelta(days=-1)

def load_all_games_csv(filename: str, preprocess=False) -> pd.DataFrame:
    """Prodcuces filename as a Dataframe, doing any
    necessary operations on it such as getting the correct dtypes and setting
    the index to the game id. If preprocess=True, this will take max 3 rest days,
    cube root of distance traveled, and square root of margin of victory, adding
    'adjusted' before the original column names."""
    all_games = pd.read_csv(filename)
    
    # Initially string, must be made timestamp
    all_games['timestamp'] = pd.to_datetime(all_games['timestamp'])
    all_games = all_games.set_index('gid')
    
    if preprocess:
        # Max rest days
        all_games['adjustedvisrestdays'] = all_games['visrestdays'].apply(lambda x: min(3,x))
        all_games['adjustedhomerestdays'] = all_games['homerestdays'].apply(lambda x: min(3,x))

        # Take cube root of distance traveled
        all_games['adjustedhomedistancetraveled'] = all_games['homedistancetraveled']**(1/3)
        all_games['adjustedvisdistancetraveled'] = all_games['visdistancetraveled']**(1/3)

        # Take square root of margin of victory
        all_games['adjustedmarginofvictory'] = np.sqrt(all_games['marginofvictory'])
        
    return all_games

def get_teams(game_df: pd.DataFrame) -> Set[str]:
    """Returns the set of all teams in game_df"""
    teams = set(game_df['hometeam'].unique()) | set(game_df['visteam'].unique())
    return teams

def plot_elo_ratings_over_time(team: str, elos_df: pd.DataFrame) -> None:
    """Plots the Elo ratings from elos_dict for the given team over time.
    
    Args:
        team (str): The team whose Elo ratings will be plotted.
        elos_df (pd.DataFrame): DataFrame whose rows are chronologically ordered
            game box scores, including columns 'homewon', 'hometeam', 'visteam',
            'homeelo', and 'viselo', where 'homeelo' and 'viselo' are the home
            and visitor Elo ratings before the game.
    """
    elos_df_filtered = elos_df[(elos_df['hometeam'] == team) | (elos_df['visteam'] == team)]
    last_row_season = -1
    
    dates = []
    elos = []
    
    for _, row in elos_df_filtered.iterrows():
        
        if row['hometeam'] == team:
            elo = row['homeeloafter']
        else:
            elo = row['viseloafter']
            
        season = row['season']
        date = row['timestamp']
        
        if season != last_row_season:
            last_row_season = season
            
            dates.append(get_prev_date_midnight(date))
            if row['hometeam'] == team:
                before_elo = row['homeelobefore']
            else:
                before_elo = row['viselobefore']
            elos.append(before_elo)
            
        dates.append(date)
        elos.append(elo)
    
    data = {'Date':dates, 'Elos':elos}
    
    plt.grid()
    sns.lineplot(data=data, x='Date', y='Elos')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Elo')
    plt.title(f'Elo Over Time for {team}')
    plt.show()
    
def get_team_lineup_mapping_first_game(season: int, teams: Set[str]) -> Dict[str, List[str]]:
    """Given a season, returns a mapping from each team in teams to their starting lineup
    for the first game of the season.
    
    Args:
        season (int): The season year.
        teams (Set[str]): The set of teams to get lineups for.
        
    Returns:
        Dict[str, List[str]]: Mapping from each team to their starting lineup
        for the first game of the season. Each player is represented by their
        retrosheet ID.
    """
    team_stats = pd.read_csv(f'../data/teamstats_clean.csv')
    team_stats_season = team_stats[team_stats['season'] == season]
    mapping = {}
    
    for team in teams:
        first_game = team_stats_season[(team_stats_season['team'] == team)].iloc[0]
        
        lineup = [first_game[f'start_l{i}'] for i in range(1,10)]
        mapping[team] = lineup
        
    return mapping