import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Set, Dict, List, Tuple

"""This module provides miscellaneous utility functions, whether for working with raw data
or creating visualizations."""

def get_prev_date_midnight(dt: pd.Timestamp) -> pd.Timestamp:
    """For the given timestamp, gets the timestamp for the previous day at midnight."""
    return dt.normalize() + pd.Timedelta(days=-1)

def load_all_games_csv(filename: str) -> pd.DataFrame:
    """Prodcuces filename as a Dataframe, doing any
    necessary operations on it such as getting the correct dtypes and setting
    the index to the game id."""
    all_games = pd.read_csv(filename)
    
    # Initially string, must be made timestamp
    all_games['timestamp'] = pd.to_datetime(all_games['timestamp'])
    all_games = all_games.set_index('gid')
    return all_games

def get_teams(game_df: pd.DataFrame) -> Set[str]:
    """Returns the set of all teams in game_df"""
    teams = set(game_df['hometeam'].unique()) | set(game_df['visteam'].unique())
    return teams

def plot_elo_ratings_over_time(team: str, elos_map: Dict[str, List[Tuple[pd.Timestamp, float, int, int]]]) -> None:
    """Plots the Elo ratings from elos_dict for the given team over time.
    
    Args:
        team (str): The team whose Elo ratings will be plotted.
        elos_map (Dict[str, List[Tuple[pd.Timestamp, float, int, int]]]): Mapping from each team to a 
            non-empty, chronologically ordered list of tuples containing:
            (1) the game id,
            (2) the date/time their Elo updated,
            (3) their Elo before that update occurred,
            (4) their Elo after that update occurred,
            (5) 1 if they won or 0 if they lost,
            (6) their number of wins after that update occurred,
            (7) their number of losses after that update occurred,
            (8) the current season.
    """
    dates = [get_prev_date_midnight(elos_map[team][0][1])] + [elos_map[team][i][1] for i in range(len(elos_map[team]))]
    elos = [elos_map[team][i][2] for i in range(len(elos_map[team]))] + [elos_map[team][-1][3]]
    
    data = {'Date':dates, 'Elos':elos}
    
    plt.grid()
    sns.lineplot(data=data, x='Date', y='Elos')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Elo')
    plt.title(f'Elo Over Time for {team}')
    plt.show()
    
def plot_elos_distribution(teams: Set[str], elos_map: Dict[str, List[Tuple[pd.Timestamp, float, int, int]]]) -> Tuple[float, float]:
    """Plots the distribution of the latest elos for each team in elos_map, returning the mean and std.
    
    Args:
        elos_map (Dict[str, List[Tuple[pd.Timestamp, float, int, int]]]): Mapping from each team to a 
            non-empty, chronologically ordered list of tuples containing:
            (1) the game id,
            (2) the date/time their Elo updated,
            (3) their Elo before that update occurred,
            (4) their Elo after that update occurred,
            (5) 1 if they won or 0 if they lost,
            (6) their number of wins after that update occurred,
            (7) their number of losses after that update occurred,
            (8) the current season.
    """

    latest_elos = np.array([elos_map[team][-1][3] for team in teams])
    
    plt.grid()
    plt.hist(latest_elos)
    plt.xlabel('Elo Rating')
    plt.ylabel('Count')
    plt.title('Elo Ratings Counts')
    plt.show()
    
    return  np.mean(latest_elos), np.std(latest_elos)