from utils.generic_utils import basic_win_prob_for_et, get_teams
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
from typing import Tuple
from elos.elo_tracker import EloTracker

"""This module provides helper functions that work with the EloTracker class explicitly.
A separate module was necessary to prevent circular imports between the utils and elo_tracker
modules."""


def add_elos_to_games_df(games_df: pd.DataFrame, elo_prob_func=basic_win_prob_for_et, K: float = 3, use_margin_of_victory: bool = False) -> pd.DataFrame:
    """Returns a version of games_df with columns 'homeelo' and 'viselo' added, which
    are calculated in part with the Elo probability function, elo_prob_func.
    
    Removes any na rows for important features at the end.
    
    Args:
        games_df (pd.DataFrame): Table whose rows are chronologically ordered game box scores,
                including columns 'hometeam' for the home team, 'visteam' for the away team, and
                'homewon' which is True if home won and False otherwise. Each game in game_df must take
                place after the games that have already been logged for the given teams it includes.
                Must be indexed by a game id column 'gid'.
        elo_prob_func (function): Function that takes in a home elo, away elo, and game information
            (i.e. row of box scores dataframe) and produces the probability of the home team winning.
        K (float): The K factor, controlling how sensitive each Elo update should be.
        use_margin_of_victory (bool): If True, incorporates margin of victory in the Elo update, where
                higher margins result in larger updates. It is included as an additional variable multiplied by K.
    """
    games_df = games_df.copy() # Don't modify original
    
    teams = get_teams(games_df)
    
    # First, get all Elo ratings
    et = EloTracker(teams, elo_prob_func=elo_prob_func, K=K, use_margin_of_victory=use_margin_of_victory)
    
    et.add_history(games_df)
    
    # Add raw pre-game Elo Ratings
    games_df['homeelo'] = [0.0] * len(games_df)
    games_df['viselo'] = [0.0] * len(games_df)
    
    home_elos = {}
    vis_elos = {}

    for team in teams:
        #print(len(et.elos_map[team]))
        for game in et.elos_map[team]:
            gid = game[0]
            elo = game[2] # Before update
            #print(elo)
        
            if games_df.loc[gid,'hometeam'] == team:
                home_elos[gid] = elo
            else:
                vis_elos[gid] = elo
                
    games_df['homeelo'] = games_df.index.map(home_elos)
    games_df['viselo'] = games_df.index.map(vis_elos)
                
    # Drop rows with na travel distance, rest or pitcher info
    games_df = games_df.dropna(subset=['homedistancetraveled', 'visdistancetraveled', 'homerestdays', 'visrestdays', 'homepitcherminusteamrgs', 'vispitcherminusteamrgs', 'homemomentum', 'vismomentum']).copy()
    
    #print(games_df[games_df['homeelo'].isna() | games_df['viselo'].isna()])
                
    return games_df

def evaluate_elo_prob_func(games_df: pd.DataFrame, elo_prob_func=basic_win_prob_for_et, K: float = 3, use_margin_of_victory: bool = False, skip_first_n: int=0) -> Tuple[float, float]:
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
        use_margin_of_victory (bool): If True, incorporates margin of victory in the Elo update, where
                higher margins result in larger updates. It is included as an additional variable multiplied by K.
        skip_first_n (int): The first skip_first_n games for each team will not be considered when computing the accuracy or
            cross entropy metrics, to allow time for the ratings to adjust to performance.
    """
    
    # Add elos
    games_df = add_elos_to_games_df(games_df, elo_prob_func, K=K, use_margin_of_victory=use_margin_of_victory)
    
    # Filter out games where the team hasn't more than played skip_first_n
    games_df = games_df[(games_df['hometeamgamecount'] > skip_first_n) & (games_df['visteamgamecount'] > skip_first_n)]
        
    games_df['homewinprob'] = games_df.apply(lambda game: elo_prob_func(game['homeelo'], game['viselo'], game), axis=1)
    
    bce = log_loss(games_df['homewon'], games_df['homewinprob'])
    accuracy = accuracy_score(games_df['homewon'], round(games_df['homewinprob']))
    
    return bce, accuracy
