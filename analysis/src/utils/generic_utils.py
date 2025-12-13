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

def load_all_games_csv(filename: str, preprocess=False) -> pd.DataFrame:
    """Prodcuces filename as a Dataframe, doing any
    necessary operations on it such as getting the correct dtypes and setting
    the index to the game id. If preprocess=True, this will take max 3 rest days,
    cube root of distance traveled, and square root of margin of victory."""
    all_games = pd.read_csv(filename)
    
    # Initially string, must be made timestamp
    all_games['timestamp'] = pd.to_datetime(all_games['timestamp'])
    all_games = all_games.set_index('gid')
    
    if preprocess:
        # Max rest days
        all_games['visrestdays'] = all_games['visrestdays'].apply(lambda x: min(3,x))
        all_games['homerestdays'] = all_games['homerestdays'].apply(lambda x: min(3,x))

        # Take cube root of distance traveled
        all_games['homedistancetraveled'] = all_games['homedistancetraveled']**(1/3)
        all_games['visdistancetraveled'] = all_games['visdistancetraveled']**(1/3)

        # Take square root of margin of victory
        all_games['marginofvictory'] = np.sqrt(all_games['marginofvictory'])
        
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
    
def plot_elos_distribution(teams: Set[str], elos_map: Dict[str, Tuple[float, int, int, int]]) -> Tuple[float, float]:
    """Plots the distribution of the latest elos for each team in elos_map, returning the mean and std.
    
    Args:
        teams(Set[str]): The teams to get elos for and plot.
        elos_map (Dict[str, Tuple[float, int, int, int]]): Mapping from each
            team to their latest Elo rating, wins, losses, and season played.
            
    Returns:
        Tuple[float, float]: The mean and standard deviation of the latest elos for each team.
    """

    latest_elos = np.array([elos_map[team][0] for team in teams])
    
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

    Returns:
        float: The basic probability the home team wins.
    """
    return 1 / (1+10**((away_elo - home_elo) / 400))

def basic_win_prob_for_et(home_elo: float, away_elo: float, game_info: pd.Series) -> float:
    """Wrapper around basic_win_prob with game_info as an additional game_info arg to be compatible
    for use in an EloTracker object."""
    return basic_win_prob(home_elo, away_elo)
    
def elo_update(home_elo: float, away_elo: float, home_won: int, home_win_prob: float,
                    K: float=3, margin_of_victory: int=None) -> Tuple[float, float]:
    """Returns updated home and away team Elos, given a result.
    
    Args:
        home_elo (float): Initial home Elo.
        away_elo (float): Initial away Elo.
        home_won (int): 1 if home team won, else 0.
        home_win_prob (float): Probability of home team winning.
        K: The K factor, determining how large the update should be.
        margin_of_victory (int): The margin of victory; if specified, this will be factored
            into the Elo update.
            
    Returns:
        Tuple[float, float]: Updated home and away Elos.
    """
    away_win_prob = 1 - home_win_prob
    
    away_won = 1 - home_won
        
    c = K
    if margin_of_victory is not None:
        c *= margin_of_victory
    
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

