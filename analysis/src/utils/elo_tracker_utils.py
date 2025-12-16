from utils.math_utils import basic_win_prob_for_et
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
from typing import Tuple, Set
from elos.elo_tracker import EloTracker

"""This module provides helper functions that work with the EloTracker class explicitly.
A separate module was necessary to prevent circular imports between the utils and elo_tracker
modules."""

def evaluate_elo_prob_func(games_df: pd.DataFrame, elo_prob_func=basic_win_prob_for_et, K: float = 3, margin_of_victory_column: str = None, skip_first_n: int=0, years:Set[int]=None) -> Tuple[float, float]:
    """Evaluates how well the given function to calculate Elo probabilties does on games_df,
    producing binary cross entropy and accuracy.
    
    Args:
        games_df (pd.DataFrame): Table whose rows are chronologically ordered game box scores,
                including columns 'hometeam' for the home team, 'visteam' for the away team, and
                'homewon' which is True if home won and False otherwise. Each game in game_df must take
                place after the games that have already been logged for the given teams it includes.
                Must be indexed by a game id column 'gid'.
        elo_prob_func (function): Function that takes in a home elo, away elo, and game information
            (i.e. row of box scores dataframe) and produces the probability of the home team winning.
        K (float): The K factor, controlling how sensitive each Elo update should be.
        margin_of_victory_column (str): If given, incorporates the margin of victory column into the Elo update, where
            higher margins result in larger updates. It is included as an additional variable multiplied by K.
        skip_first_n (int): The first skip_first_n games for each team will not be considered when computing the accuracy or
            cross entropy metrics, to allow time for the ratings to adjust to performance.
        years (Set[int]): If specified, gets metrics for the df filtered for the given years.
    """
    games_df = games_df.copy()
    
    # Add elos - one should do this from the very start so Elos are accurate
    et = EloTracker(K=K, elo_prob_func=elo_prob_func, margin_of_victory_column=margin_of_victory_column)
    et.add_history(games_df, add_win_probs_to_df=True)
    
    games_df = games_df.dropna(subset=['adjustedhomedistancetraveled', 'adjustedvisdistancetraveled', 'adjustedhomerestdays', 'adjustedvisrestdays', 'homepitcherminusteamrgs', 'vispitcherminusteamrgs'])
    
    # Filter out games where the team hasn't played more than skip_first_n
    games_df = games_df[(games_df['hometeamgamecount'] > skip_first_n) & (games_df['visteamgamecount'] > skip_first_n)]
    
    if years is not None:
        games_df = games_df[games_df['season'].isin(years)]

    #print(games_df['homewon'])
    #print(games_df['homewinprob'])
    bce = log_loss(games_df['homewon'], games_df['homewinprob'])
    accuracy = accuracy_score(games_df['homewon'], round(games_df['homewinprob']))
    
    return bce, accuracy
