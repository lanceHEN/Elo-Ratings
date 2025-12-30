from typing import Set, Dict, List, Tuple
from scipy.special import expit
import numpy as np
import pandas as pd
from tqdm import tqdm
from .misc_utils import load_batting_csv
from sklearn.metrics import log_loss, accuracy_score

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

def predict_lr_use_pitchers_if_first_games(home_elo: float, away_elo: float, game: pd.Series):
    """Uses logistic regression with pitcher rating included if it is each pitcher's first game, and
    the game is not None.
    
    We allow the game to be None in case it is a playoff game and we don't have explicit information for it.
    In this case, we just use home advantage.
    
    Args:
        home_elo (float): Elo of home team.
        away_elo (float): Elo of away team.
        game (pd.Series): Row of game dataframe containing game info including rest days, travel, etc. If none,
            this will default the calculation to adjust for home advantage only.
    
    """
    
    if game is None: # For playoffs we don't have rows in the dataframe
        elo_diff = away_elo - home_elo
    
        home_adv_diff = 0 - 1
    
        x = np.array([elo_diff, home_adv_diff])
        return p(x.T, w[:2]).item()
    
    else:
        game = game.copy()
        
        if not (game['homefirstpitchergameofseason'] and game['homefirstpitchergameofseason']):
            game['homepitcherminusteamrgs'] = 0
            game['vispitcherminusteamrgs'] = 0
            
        return predict_lr(home_elo, away_elo, game)

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
    from elos import EloTracker
    
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


def get_player_transition_matrices(season: int, players: Set[str]) -> Dict[str, np.array]:
    """Given a season and player ids, produces a mapping from each player id to their transition matrix.
    
    Each transition matrix is 24x25, where each of the initial 24 is some combination of # outs and bases occupied,
    and there is one additional column for the terminal state of 3 outs.
    
    They are ordered first by outs (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).
    Note bases occupied are represented and ordered as 3-bit binary numbers, for convenience with calculating
    transitions for singles and doubles.
    
    Note that in this implementation, we only use the player out, reach first, reach second, reach third, and home run events
    to have enough data for each state. We also don't have information for advancing runners currently.
    
    Args:
        season (int): The season year to get player data for.
        player_names (Set[str]): The set of player retrosheet IDs to get transition matrices for.
        
    Returns:
        Dict[str, np.array]: Mapping from each player retrosheet ID to their transition matrix.
    """
    batting = load_batting_csv(f'../data/batting_clean.csv')
    batting = batting[batting['season'] == season]
    
    mapping = {}
    
    # Get averages in case a player has no data
    
    all_plate_apps = sum(batting[col].sum() for col in ['b_ab', 'b_iw', 'b_w', 'b_hbp', 'b_sf', 'b_sh', 'b_xi'])
    
    all_firsts = (batting['b_h'].sum() - sum(batting[col].sum() for col in ['b_d', 'b_t', 'b_hr'])) + sum(batting[col].sum() for col in ['b_iw', 'b_w', 'b_hbp', 'b_xi'])
    
    all_seconds = batting['b_d'].sum()
    
    all_thirds = batting['b_t'].sum()
    
    all_homers = batting['b_hr'].sum()
    
    all_outs = all_plate_apps - (all_firsts + all_seconds + all_thirds + all_homers) # Not correct but close enough
    
    all_first_pct = all_firsts / all_plate_apps
    
    all_second_pct = all_seconds / all_plate_apps
    
    all_third_pct = all_thirds / all_plate_apps
    
    all_homer_pct = all_homers / all_plate_apps
    
    all_out_pct = all_outs / all_plate_apps
    
    for player in tqdm(players):
        player_batting = batting[batting['id'] == player]
        
        plate_apps = sum(player_batting[col].sum() for col in ['b_ab', 'b_iw', 'b_w', 'b_hbp', 'b_sf', 'b_sh', 'b_xi'])
        
        firsts = (player_batting['b_h'].sum() - sum(player_batting[col].sum() for col in ['b_d', 'b_t', 'b_hr'])) + sum(player_batting[col].sum() for col in ['b_iw', 'b_w', 'b_hbp', 'b_xi'])
    
        seconds = player_batting['b_d'].sum()
    
        thirds = player_batting['b_t'].sum()
    
        homers = player_batting['b_hr'].sum()
    
        outs = plate_apps - (firsts + seconds + thirds + homers) # Not correct but close enough
    
        first_pct = firsts / plate_apps if plate_apps > 0 else all_first_pct
    
        second_pct = seconds / plate_apps if plate_apps > 0 else all_second_pct
    
        third_pct = thirds / plate_apps if plate_apps > 0 else all_third_pct
    
        homer_pct = homers / plate_apps if plate_apps > 0 else all_homer_pct
        
        out_pct = outs / plate_apps if plate_apps > 0 else all_out_pct
        
        T_third = np.zeros((8, 25)) # Make zeros so we only have to worry about possible transitions
        # We split this into thirds because the calculations for singles, doubles, triples, homers will
        # be the same whether 0, 1, or 2 outs
        
        terminal_state = 24
            
        # Singles, doubles, triples, homers
        # In each of them, we assume the number of outs won't change
        for bases in range(8):
            initial_state = bases # row index
            
            for bases_hit, pct in zip(range(1,5), [first_pct, second_pct, third_pct, homer_pct]): # Singles thru homers
                next_state = int(bin((bases << bases_hit) + 2**(bases_hit-1))[2:].zfill(3)[-3:], 2)
                T_third[initial_state, next_state] = pct
        
        # Stack for 0, 1, 2 outs
        T = np.vstack((T_third, T_third, T_third))
                
        # End with outs - if < 2 outs, transition to same state but with 1 more out
        # if 2 outs, transition to terminal state
        
        for outs in range(2): # < 2 outs
            for bases in range(8):
                initial_state = outs * 8 + bases # row index
                next_state = initial_state + 8
        
                T[initial_state, next_state] = out_pct

        # 0 and 1 initial outs - go from current state to same state, but with 1 more out
        
        # 0 outs -> 1 out
        np.fill_diagonal(T[:1*8, 1*8:2*8], out_pct)
        
        # 1 out -> 2 outs
        np.fill_diagonal(T[1*8:2*8, 2*8:3*8], out_pct)
                        
        # 2 outs -> 3 outs (1 terminal state no matter the initial state)
        T[2*8:, terminal_state] = out_pct
                
        mapping[player] = T
        
    return mapping

def get_runs_for_transition_matrix() -> np.array:
    """Produces a 24x25 matrix R where entry R[i,j] contains the number of runs produced
    by transitioning from state i to state j.
    
    The states are ordered exactly the same as for the transition matrix, first by outs
    (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).
    
    Returns:
        np.array: Matrix of runs for each transition.
    """
    R_third = np.zeros((8, 25)) # Transitions will be the same for 0, 1, 2 outs so calculate once and stack 3x later
    
    count_runs = lambda x: sum(int(l) for l in x) # Count number of ones in binary string
    
    # We can find the runs by going over each initial state and considering singles, doubles, triples, and homers
    # We look at the overflow on the left side after shifting by different amounts, counting the '1's
    for bases in range(8):
        initial_state = bases
        
        for bases_hit in range(1,5): # Singles thru homers
            full = bin((bases << bases_hit) + 2**(bases_hit-1))[2:].zfill(3)
            next_state = int(full[-3:], 2)
            overflow = full[:-3]
            runs = count_runs(overflow)
            
            R_third[initial_state, next_state] = runs
            
    R = np.vstack((R_third, R_third, R_third))    
            
    return R