from typing import Set, Dict, List, Tuple
from scipy.special import expit
import numpy as np
import pandas as pd

"""This module provides mathematical utility constants and functions used throughout the analysis codebase."""

# Learned in src/win_prob.py
w = np.array([[ 1.        ],
       [27.1440666 ],
       [ 4.81476328],
       [-0.30523283],
       [ 1.58004319]])

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


def predict_lr(home_elo, away_elo, game, w=w):
    """Predicts probability of home team winning via sigmoid function with given weight vector."""
    elo_diff = away_elo - home_elo
    rest_day_diff = game['adjustedvisrestdays'] - game['adjustedhomerestdays']
    rest_day_diff = rest_day_diff if not np.isnan(rest_day_diff) else 0
    
    travel_diff = game['adjustedvisdistancetraveled'] - game['adjustedhomedistancetraveled']
    travel_diff = travel_diff if not np.isnan(travel_diff) else 0
    
    home_adv_diff = 0 - 1
    
    pitcher_diff = game['vispitcherminusteamrgs'] - game['homepitcherminusteamrgs']
    pitcher_diff = pitcher_diff if not np.isnan(pitcher_diff) else 0
    
    x = np.array([elo_diff, home_adv_diff, rest_day_diff, travel_diff, pitcher_diff]).reshape(-1,1)
     
    return p(x.T, w).item()
    
# Given a season and player names, return a player name -> transition matrix mapping
def get_player_transition_matrices(season: int, player_names: Set[str]) -> Dict[str, np.array]:
    """Given a season and player ids, produces a mapping from each player id to their transition matrix.
    
    Each transition matrix is 24x25, where each of the initial 24 is some combination of # outs and bases occupied,
    and there is one additional column for the terminal state of 3 outs.
    
    Note that in this implementation, we only use the player out, reach first, reach second, reach third, and home run events
    to have enough data for each state.
    
    Args:
        season (int): The season year to get player data for.
        player_names (Set[str]): The set of player retrosheet IDs to get transition matrices for.
        
    Returns:
        Dict[str, np.array]: Mapping from each player retrosheet ID to their transition matrix.
    """