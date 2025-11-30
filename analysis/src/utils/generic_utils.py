import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Set, Dict, List, Tuple
from scipy.special import expit

"""This module provides miscellaneous utility constants and functions, whether for working with raw data
or creating visualizations.

NOTE: Any helpers using the EloTracker class cannot be in this module, to prevent circular imports."""

# Mapping from each team ID to their full name, including active years (duplicate full names otherwise)

teams = pd.read_csv('../data/teams.csv')

# Merge city and nickname into one 'fullname' column
teams['fullname'] = teams['CITY'] + ' ' + teams['NICKNAME'] + ' (' + teams['FIRST'].astype(str) + ' - ' + teams['LAST'].astype(str) + ')'

TEAM_FULLNAME_MAP = {team: full for team, full in zip(teams['TEAM'], teams['fullname'])}

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
        elos_map (Dict[str, List[Tuple[str, pd.Timestamp, float, float, bool, int, int, int, bool]]]): Mapping from each team to a 
            chronologically ordered list of tuples containing:
            (1) the game id,
            (2) the date/time their Elo updated,
            (3) their Elo before that update occurred,
            (4) their Elo after that update occurred,
            (5) True if they won or False if they lost,
            (6) their number of wins after that update occurred,
            (7) their number of losses after that update occurred,
            (8) the current season,
            (9) True if it's the first game of that season (or ever) and False otherwise.
            This is the centerpoint of this class and may be referenced at any time
            to observe a team's Elo history.
    """
    dates = []
    elos = []
    for i in range(len(elos_map[team])):
        if elos_map[team][i][8]: # If it's the first game of the season, we need an additional entry for before it starts
            dates.append(get_prev_date_midnight(elos_map[team][i][1]))
            elos.append(elos_map[team][i][2])
        
        # Always get date and Elo after update
        dates.append(elos_map[team][i][1])
        elos.append(elos_map[team][i][3])
    
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
        elos_map (Dict[str, List[Tuple[str, pd.Timestamp, float, float, bool, int, int, int, bool]]]): Mapping from each team to a 
            chronologically ordered list of tuples containing:
            (1) the game id,
            (2) the date/time their Elo updated,
            (3) their Elo before that update occurred,
            (4) their Elo after that update occurred,
            (5) True if they won or False if they lost,
            (6) their number of wins after that update occurred,
            (7) their number of losses after that update occurred,
            (8) the current season,
            (9) True if it's the first game of that season (or ever) and False otherwise.
            This is the centerpoint of this class and may be referenced at any time
            to observe a team's Elo history.
    """

    latest_elos = np.array([elos_map[team][-1][3] for team in teams])
    
    plt.grid()
    plt.hist(latest_elos)
    plt.xlabel('Elo Rating')
    plt.ylabel('Count')
    plt.title('Elo Ratings Counts')
    plt.show()
    
    return np.mean(latest_elos), np.std(latest_elos)

def basic_win_prob(home_elo: float, away_elo: float) -> float:
    """Fetches the basic Elo probability the home team wins, given each team's Elo, along
    with game_info, storing additional game info.
    
    The basic Elo probability is given by 1 / (1+10^((away_elo - home_elo) / 400).
    
    Args:
        home_elo (float): Home team Elo.
        away_elo (float): Away team Elo.
        game_info (pd.Series): Row of a game info DataFrame storing additional information.
        
    Returns:
        float: The basic probability the home team wins.
    """
    return 1 / (1+10**((away_elo - home_elo) / 400))

def basic_win_prob_for_et(home_elo: float, away_elo: float, game_info: pd.Series) -> float:
    """Wrapper around basic_win_prob with game_info as an additional game_info arg to be compatible
    for use in an EloTracker object."""
    return basic_win_prob(home_elo, away_elo)
    
def elo_update(home_elo: float, away_elo: float, home_won: int,
                    K: float=3, elo_prob_func=basic_win_prob_for_et, game_info: pd.Series = None, use_margin_of_victory=False) -> Tuple[float, float]:
    """Returns updated home and away team Elos, given a result.
    
    Args:
        home_elo (float): Initial home Elo.
        away_elo (float): Initial away Elo.
        home_won (int): 1 if home team won, else 0.
        K: The K factor, determining how large the update should be.
        elo_prob_func (function): Function that takes in a home elo, away elo and optionally game information
            (i.e. game_info) and produces the probability of the home team winning.
        game_info (pd.Series): Row of a game info DataFrame storing additional information.
        use_margin_of_victory (bool): If True, incorporates margin of victory in the update, where
            higher margins result in larger updates. It is included as an additional variable multiplied by K.
    """
    home_win_prob = elo_prob_func(home_elo, away_elo, game_info)
        
    away_win_prob = 1 - home_win_prob
    
    away_won = 1 - home_won
        
    c = K
    if use_margin_of_victory and game_info is not None:
        c *= game_info['marginofvictory']
    
    # Update elos
    home_elo = home_elo + c*(home_won - home_win_prob)
    away_elo = away_elo + c*(away_won - away_win_prob)
    
    return home_elo, away_elo

def p(X,w):
    """Vector form Elo pdf for a tabular input."""
    z = X @ w
    return expit((-np.log(10) / 400) * z)


def predict_lr(home_elo, away_elo, game, w):
    """Predicts probability of home team winning via sigmoid function with given weight vector."""
    elo_diff = away_elo - home_elo
    rest_day_diff = game['visrestdays'] - game['homerestdays']
    rest_day_diff = rest_day_diff if not np.isnan(rest_day_diff) else 0
    
    travel_diff = game['visdistancetraveled'] - game['homedistancetraveled']
    travel_diff = travel_diff if not np.isnan(travel_diff) else 0
    
    home_adv_diff = 0 - 1
    
    pitcher_diff = game['vispitcherminusteamrgs'] - game['homepitcherminusteamrgs']
    pitcher_diff = pitcher_diff if not np.isnan(pitcher_diff) else 0
    
    x = np.array([elo_diff, home_adv_diff, rest_day_diff, travel_diff, pitcher_diff]).reshape(-1,1)
     
    return p(x.T, w).item()

