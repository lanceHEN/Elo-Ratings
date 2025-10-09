import pandas as pd
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
    necessary operations on it such as getting the correct dtypes."""
    all_games = pd.read_csv(filename)
    
    # Initially string, must be made timestamp
    all_games['timestamp'] = pd.to_datetime(all_games['timestamp'])
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
            (1) the date/time their Elo updated,
            (2) their Elo after that update occurred,
            (3) their number of wins after that update occurred,
            (4) their number of losses after that update occurred.
    """
    dates = [elos_map[team][i][0] for i in range(len(elos_map[team]))]
    elos = [elos_map[team][i][1] for i in range(len(elos_map[team]))]
    
    data = {'Date':dates, 'Elos':elos}
    
    sns.lineplot(data=data, x='Date', y='Elos')
    plt.grid()
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Elo')
    plt.title(f'Elo Over Time for {team}')
    plt.show()