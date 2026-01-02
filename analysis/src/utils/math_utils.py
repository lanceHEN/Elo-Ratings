from typing import Set, Dict, List, Tuple, Iterable
from pathlib import Path

from scipy.special import expit
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score

from .misc_utils import load_clean_csv, get_teams, get_team_lineup_mapping_first_game

"""This module provides mathematical utility constants and functions used throughout the analysis codebase."""

# Learned in src/win_prob.py
# w = np.array([[1.0], [27.1440666], [4.81476328], [-0.30523283], [1.58004319]])
w = np.array([[1.0], [27.76849962], [5.9476822], [-0.31031039], [1.59619219]])

root = Path(__file__).parent.parent.parent
data_dir = root / "data"
raw_data_dir = data_dir / "raw"
clean_data_dir = data_dir / "clean"
batting_name = clean_data_dir / "batting_clean.csv"
pitching_name = clean_data_dir / "pitching_clean.csv"


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
    return 1 / (1 + 10 ** ((away_elo - home_elo) / 400))


def basic_win_prob_for_et(
    home_elo: float, away_elo: float, game_info: pd.Series
) -> float:
    """Wrapper around basic_win_prob with game_info as an additional game_info arg to be compatible
    for use in an EloTracker object."""
    return basic_win_prob(home_elo, away_elo)


def elo_update(
    home_elo: float,
    away_elo: float,
    home_won: int,
    home_win_prob: float,
    K: float = 3,
    margin_of_victory: int = None,
) -> Tuple[float, float]:
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
    home_elo = home_elo + c * (home_won - home_win_prob)
    away_elo = away_elo + c * (away_won - away_win_prob)

    return home_elo, away_elo


def p(X, w):
    """Vector form Elo pdf for a tabular input."""
    z = X @ w
    return expit((-np.log(10) / 400) * z)


def predict_lr(home_elo, away_elo, game, w=w):
    """Predicts probability of home team winning via sigmoid function with given weight vector."""
    elo_diff = away_elo - home_elo
    rest_day_diff = game["adjustedvisrestdays"] - game["adjustedhomerestdays"]
    rest_day_diff = rest_day_diff if not np.isnan(rest_day_diff) else 0

    travel_diff = (
        game["adjustedvisdistancetraveled"] - game["adjustedhomedistancetraveled"]
    )
    travel_diff = travel_diff if not np.isnan(travel_diff) else 0

    home_adv_diff = 0 - 1

    pitcher_diff = game["vispitcherminusteamrgs"] - game["homepitcherminusteamrgs"]
    pitcher_diff = pitcher_diff if not np.isnan(pitcher_diff) else 0

    x = np.array(
        [elo_diff, home_adv_diff, rest_day_diff, travel_diff, pitcher_diff]
    ).reshape(-1, 1)

    return p(x.T, w).item()


def predict_lr_use_pitchers_if_first_games(
    home_elo: float, away_elo: float, game: pd.Series
):
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

    if game is None:  # For playoffs we don't have rows in the dataframe
        elo_diff = away_elo - home_elo

        home_adv_diff = 0 - 1

        x = np.array([elo_diff, home_adv_diff])
        return p(x.T, w[:2]).item()

    else:
        game = game.copy()

        if not (
            game["homefirstpitchergameofseason"] and game["visfirstpitchergameofseason"]
        ):
            game["homepitcherminusteamrgs"] = 0
            game["vispitcherminusteamrgs"] = 0

        return predict_lr(home_elo, away_elo, game)


def evaluate_elo_prob_func(
    games_df: pd.DataFrame,
    elo_prob_func=basic_win_prob_for_et,
    K: float = 3,
    margin_of_victory_column: str = None,
    skip_first_n: int = 0,
    years: Set[int] = None,
) -> Tuple[float, float]:
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
    et = EloTracker(
        K=K,
        elo_prob_func=elo_prob_func,
        margin_of_victory_column=margin_of_victory_column,
    )
    et.add_history(games_df, add_win_probs_to_df=True)

    games_df = games_df.dropna(
        subset=[
            "adjustedhomedistancetraveled",
            "adjustedvisdistancetraveled",
            "adjustedhomerestdays",
            "adjustedvisrestdays",
            "homepitcherminusteamrgs",
            "vispitcherminusteamrgs",
        ]
    )

    # Filter out games where the team hasn't played more than skip_first_n
    games_df = games_df[
        (games_df["hometeamgamecount"] > skip_first_n)
        & (games_df["visteamgamecount"] > skip_first_n)
    ]

    if years is not None:
        games_df = games_df[games_df["season"].isin(years)]

    # print(games_df['homewon'])
    # print(games_df['homewinprob'])
    bce = log_loss(games_df["homewon"], games_df["homewinprob"])
    accuracy = accuracy_score(games_df["homewon"], round(games_df["homewinprob"]))

    return bce, accuracy


def _get_transition_percents_from_batting_df(batting_df: pd.DataFrame) -> np.array:
    """
    Produces numpy array percentages for getting to first, second, third, home, or out from the given dataframe.
    """
    plate_apps = sum(
        batting_df[col].sum()
        for col in ["b_ab", "b_iw", "b_w", "b_hbp", "b_sf", "b_sh", "b_xi"]
    )

    firsts = (
        batting_df["b_h"].sum()
        - sum(batting_df[col].sum() for col in ["b_d", "b_t", "b_hr"])
    ) + sum(batting_df[col].sum() for col in ["b_iw", "b_w", "b_hbp", "b_xi"])
    seconds = batting_df["b_d"].sum()
    thirds = batting_df["b_t"].sum()
    homers = batting_df["b_hr"].sum()

    first_pct = firsts / plate_apps
    second_pct = seconds / plate_apps
    third_pct = thirds / plate_apps
    homer_pct = homers / plate_apps
    out_pct = 1 - (first_pct + second_pct + third_pct + homer_pct)

    return np.array([first_pct, second_pct, third_pct, homer_pct, out_pct])


def _get_transition_percents_from_pitching_df(pitching_df: pd.DataFrame) -> np.array:
    """
    Produces numpy array of percentages for hitters getting to first, second, third, home, and out from the given pitching dataframe.
    """
    plate_apps = pitching_df["p_bfp"].sum()

    firsts = (
        pitching_df["p_h"].sum()
        - sum(pitching_df[col].sum() for col in ["p_d", "p_t", "p_hr"])
    ) + sum(
        pitching_df[col].sum() for col in ["p_iw", "p_w", "p_hbp"]
    )  # No catcher interference but close enough
    seconds = pitching_df["p_d"].sum()
    thirds = pitching_df["p_t"].sum()
    homers = pitching_df["p_hr"].sum()

    first_pct = firsts / plate_apps
    second_pct = seconds / plate_apps
    third_pct = thirds / plate_apps
    homer_pct = homers / plate_apps
    out_pct = 1 - (first_pct + second_pct + third_pct + homer_pct)

    return np.array([first_pct, second_pct, third_pct, homer_pct, out_pct])


def _season_transition_percentages(season: int) -> np.array:
    """Produces numpy array of overall percentages for getting to first, second, third, home,
    or out for the given season."""
    batting = load_clean_csv(batting_name)
    batting = batting[batting["season"] == season]

    return _get_transition_percents_from_batting_df(batting)


def _hitter_transition_percentages(
    season: int, hitter_id: str, global_transition_percentages: np.array
) -> np.array:
    """
    Produces numpy array of percentages for getting to first, second, third, home, or out
    for the given hitter retrosheet id, in the given season; defaults to the
    global percentages if the hitter had < 30 plate appearances.
    """
    batting = load_clean_csv(batting_name)
    batting = batting[(batting["season"] == season) & (batting["id"] == hitter_id)]

    if len(batting) < 30:
        return global_transition_percentages
    else:
        return _get_transition_percents_from_batting_df(batting)


def _team_pitcher_transition_percentages(season: int, team_id: str) -> np.array:
    """
    Produces numpy array of percentages for opponent hitters getting to first, second, third, and home
    for the given pitching team retrosheet id, in the given season; defaults to the
    global percentages if the pitcher had < 30 plate appearances.
    """
    pitching = load_clean_csv(pitching_name)
    pitching = pitching[(pitching["season"] == season) & (pitching["team"] == team_id)]

    return _get_transition_percents_from_pitching_df(pitching)


def _get_transition_matrix(transition_values: np.array) -> np.array:
    """
    Given an array of transition values, produces the corresponding transition matrix.

    Args:
        transition_values (np.array): Array of transition values for reaching
            first, second, third, home, or getting out. These may be ratios
            or percentages

    Returns:
        np.array: The corresponding transition matrix.
    """

    first, second, third, home, out = tuple(transition_values)

    T_third = np.zeros((8, 25))
    # Make zeros so we only have to worry about possible transitions
    # We split this into thirds because the calculations for singles, doubles, triples, homers will
    # be the same whether 0, 1, or 2 outs

    terminal_state = 24

    # Singles, doubles, triples, homers
    # In each of them, we assume the number of outs won't change
    for bases in range(8):
        initial_state = bases  # row index

        for bases_hit, val in zip(
            range(1, 5), [first, second, third, home]
        ):  # Singles thru homers
            next_state = int(
                bin((bases << bases_hit) + 2 ** (bases_hit - 1))[2:].zfill(3)[-3:],
                2,
            )
            T_third[initial_state, next_state] = val

    # Stack for 0, 1, 2 outs
    T = np.vstack((T_third, T_third, T_third))

    # 0 and 1 initial outs - go from current state to same state, but with 1 more out

    # 0 outs -> 1 out
    np.fill_diagonal(T[: 1 * 8, 1 * 8 : 2 * 8], out)

    # 1 out -> 2 outs
    np.fill_diagonal(T[1 * 8 : 2 * 8, 2 * 8 : 3 * 8], out)

    # 2 outs -> 3 outs (1 terminal state no matter the initial state)
    T[2 * 8 :, terminal_state] = out

    return T


def _get_hitter_global_ratio_transition_matrix(
    season: int, hitter_id: str, global_transition_percentages: np.array
) -> np.array:
    """
    Produces transition matrix using stats for the given hitter in the given season, normalized by
    global percentages. In other words, each nonzero matrix element is the batter's percentage
    divided by the global percentage, e.g. for 1 base transitions, the value is the percentage
    of time the batter hits for 1 base over the global percentage of times that season
    any batter hits for 1 base.

    Each matrix is 24x25, where each of the initial 24 is some combination of # outs and bases occupied,
    and there is one additional column for the terminal state of 3 outs.

    They are ordered first by outs (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).
    Note bases occupied are represented and ordered as 3-bit binary numbers, for convenience with calculating
    transitions for singles and doubles.

    Note that in this implementation, we only use the player out, reach first, reach second, reach third, and home run events
    to have enough data for each state. We also don't have information for advancing runners currently.

    Args:
        season (int): The season year to get player data for.
        hitter_id (str): The retrosheet id of the hitter to get stats for.
        global_transition_percentages np.array: Overall percentages for
            reaching first, second, third, home, or out. If player has < 30 at bats, the batter
            percentages will default to these

    Returns:
        np.array: 24x25 matrix of batter transition percentages divided by global transition
            percentages.
    """
    batter_pcts = _hitter_transition_percentages(
        season, hitter_id, global_transition_percentages
    )

    ratios = batter_pcts / global_transition_percentages

    return _get_transition_matrix(ratios)


def _get_team_pitching_transition_matrix(season: int, team_id: str) -> np.array:
    """
    Produces transition matrix using pitching stats for the given team in the given season.

    Each transition matrix is 24x25, where each of the initial 24 is some combination of # outs and bases occupied,
    and there is one additional column for the terminal state of 3 outs.

    They are ordered first by outs (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).
    Note bases occupied are represented and ordered as 3-bit binary numbers, for convenience with calculating
    transitions for singles and doubles.

    Note that in this implementation, we only use the player out, reach first, reach second, reach third, and home run events
    to have enough data for each state. We also don't have information for advancing runners currently.

    Args:
        season (int): The season year to get player data for.
        team_id (str): The retrosheet id of the team to get stats for.

    Returns:
        np.array: 24x25 transition matrix.
    """
    pitching_pcts = _team_pitcher_transition_percentages(season, team_id)

    return _get_transition_matrix(pitching_pcts)


def get_team_hitter_transition_matrices(
    season: int, teams: Iterable[str]
) -> Dict[str, np.array]:
    """
    Given a season, produces a mapping from each team to a numpy array of each player's transition matrices,
    normalized by global percentages, e.g. for 1 base transitions, the value is the percentage
    of time the batter hits for 1 base over the global percentage of times that season
    any batter hits for 1 base.

    The player transition matrices are from the previous season.

    The lineup for each team is simply the lineup for their first game for the season.

    Args:
        season (int): The season year to get player/team data for.
        teams (Iterable[str]): The teams to get data for.

    Returns:
        Dict[str, np.array]: Mapping from each team to their lineup transition matrices.
    """
    default_probs = _season_transition_percentages(season - 1)
    team_lineups = get_team_lineup_mapping_first_game(season, teams)
    transition_matrices = {}
    for team, lineup in team_lineups.items():
        lineup_matrices = []
        for player in lineup:
            lineup_matrices.append(
                _get_hitter_global_ratio_transition_matrix(
                    season - 1, player, default_probs
                )
            )

        transition_matrices[team] = np.array(lineup_matrices)

    return transition_matrices


def get_team_pitcher_transition_matrices(
    season: int, teams: Iterable[str]
) -> Dict[str, np.array]:
    """
    Given a season, produces a mapping from each team to their overall
    transition matrix derived from combined pitching for the previous season.

    We do the previous season here for consistency because get_team_hitter_transition_matrices
    similarly gets data for the previous season.

    Args:
        season (int): The season year to get team data for.
        teams (Iterable[str]): The teams to get data for.

    Returns:
        Dict[str, np.array]: Mapping from each team to their pitcher transition matrix.
    """
    transition_matrices = {}

    for team in teams:
        transition_matrices[team] = _get_team_pitching_transition_matrix(
            season - 1, team
        )

    return transition_matrices


def get_runs_for_transition_matrix() -> np.array:
    """Produces a 24x25 matrix R where entry R[i,j] contains the number of runs produced
    by transitioning from state i to state j.

    The states are ordered exactly the same as for the transition matrix, first by outs
    (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).

    Returns:
        np.array: Matrix of runs for each transition.
    """
    R_third = np.zeros(
        (8, 25)
    )  # Transitions will be the same for 0, 1, 2 outs so calculate once and stack 3x later

    count_runs = lambda x: sum(
        int(l) for l in x
    )  # Count number of ones in binary string

    # We can find the runs by going over each initial state and considering singles, doubles, triples, and homers
    # We look at the overflow on the left side after shifting by different amounts, counting the '1's
    for bases in range(8):
        initial_state = bases

        for bases_hit in range(1, 5):  # Singles thru homers
            full = bin((bases << bases_hit) + 2 ** (bases_hit - 1))[2:].zfill(3)
            next_state = int(full[-3:], 2)
            overflow = full[:-3]
            runs = count_runs(overflow)

            R_third[initial_state, next_state] = runs

    R = np.vstack((R_third, R_third, R_third))

    return R


def get_outs_for_transition_matrix() -> np.array:
    """Produces a 24x25 matrix O where entry O[i,j] contains the number of outs produced
    by transitioning from state i to state j (0 or 1).

    The states are ordered exactly the same as for the transition matrix, first by outs
    (0, 1, 2) then by bases occupied (000, 001, 010, 011, 100, 101, 110, 111).

    Returns:
        np.array: Matrix of outs for each transition.
    """
    O = np.zeros((24, 25))

    # For 0 -> 1 or 1 -> 2: Same state but 1 more out, so add 8 to current
    # state idx to get new state idx - this should have a diagonal shape!
    np.fill_diagonal(O[0:16, 8:24], 1)

    # For 2->3 outs, just 1 state
    O[16:, 24] = 1

    return O
