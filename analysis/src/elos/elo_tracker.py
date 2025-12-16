import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Set
from utils.math_utils import basic_win_prob_for_et, elo_update

class EloTracker():
    """This class provides an interface to store and add to team
    Elo ratings over time.
    
    Attributes:
        elos_map (Dict[str, Tuple[float, int, int, int]]): Mapping from each team to their latest Elo
            rating, wins, losses, and season played.
        initial_elo (float): The initial Elo rating for each team. This will be used for the elo in the
            first entry in elos_map[team], the day before the first
            game they eventually play.
        K (float): The K factor, controlling how sensitive each Elo update should be.
        elo_prob_func (function): Function that takes in a home elo, away elo, and game information
            (i.e. row of box scores dataframe) and produces the probability of the home team winning.
        margin_of_victory_column (str): If given, incorporates the margin of victory column into the Elo update, where
            higher margins result in larger updates. It is included as an additional variable multiplied by K.
    """
    
    def __init__(self, initial_elo: float=1500, K: float=3, elo_prob_func=basic_win_prob_for_et, margin_of_victory_column: str=None):
        """Constructs an EloTracker from scratch with the given initial elo, probability function, and whether
        to use margin of victory.
        
        Args:
            initial_elo (float): The initial Elo rating for each team.
            K (float): The K factor, controlling how sensitive each Elo update should be.
            elo_prob_func (function): Function that takes in a home elo, away elo, and game information
                (i.e. row of box scores dataframe) and produces the probability of the home team winning.
            margin_of_victory_column (str): If given, incorporates the margin of victory column into the Elo update, where
                higher margins result in larger updates. It is included as an additional variable multiplied by K.
        """
        self.elos_map = {}
        self.initial_elo = initial_elo
        self.K = K
        self.elo_prob_func = elo_prob_func
        self.margin_of_victory_column = margin_of_victory_column
            
    def _get_initial_team_stats(self, team: str, season: int) -> Tuple[float, int, int, bool]:
        """Fetches the initial Elo, wins and losses for the team.
        
        If elos_map[team] is empty, it will produce initial_elo,
        along with 0 wins and 0 losses.
        
        If the season is a new season, it will be the team's previous elo reverted to initial_elo by 1/3,
        along with 0 wins and 0 losses.
        
        Otherwise, it will just be the previous Elo, and the last recorded wins and losses.
         
        Args:
            team (str): The team to check.
            season (int): The possibly new season to check.
            
        Returns:
            Tuple[float, int, int, bool]: A tuple containing:
                (1) The initial Elo,
                (2) the initial wins,
                (3) the initial losses.
        """
        
        if team not in self.elos_map:
            return self.initial_elo, 0, 0

        elif self.elos_map[team][3] < season:
            old_elo = self.elos_map[team][0]
            new_elo = old_elo + (self.initial_elo - old_elo) / 3
            return new_elo, 0, 0
        
        else:
            wins = self.elos_map[team][1]
            losses = self.elos_map[team][2]
            return self.elos_map[team][0], wins, losses
    
    def add_history(self, game_df: pd.DataFrame, add_elos_to_df=False, add_win_probs_to_df=False) -> None:
        """Updates the values in elos_map with the results from game_df, optionally adding elo and win prob
        columns to the df.
        
        Args:
            game_df (pd.DataFrame): Table whose rows are chronologically ordered game box scores,
                including columns 'hometeam' for the home team, 'visteam' for the away team, and
                'homewon' which is True if home won and False otherwise. Each game in game_df must take
                place after the games that have already been logged for the given teams it includes.
                Must be indexed by a game id column 'gid'.
            add_elos_to_df (bool): If True, adds 'homeelobefore' and 'viselobefore' columns to game_df, with the home
                and visitor elos before the games took place, as well as 'homeeloafter' and 'viseloafter'
                columns for elos after the game.
            add_elos_to_df (bool): If True, adds column 'homewinprob' to game_df, with the probability the home
                team won--the same probability used in the elo updates.
        """
        
        if add_elos_to_df:
        
            home_elos_before = []
            vis_elos_before = []
        
            home_elos_after= []
            vis_elos_after = []
            
        if add_win_probs_to_df:
            home_win_probs = []
        
        for _, game in game_df.iterrows():
            home_team = game['hometeam']
            away_team = game['visteam']
            
            season = game['season']
            
            # Get initial elos, home wins and losses
            initial_home_elo, home_wins, home_losses = self._get_initial_team_stats(home_team, season)
            initial_away_elo, away_wins, away_losses = self._get_initial_team_stats(away_team, season)
            
            # Final result
            home_won = int(game['homewon'])
            away_won = 1 - home_won
            
            home_win_prob = self.elo_prob_func(initial_home_elo, initial_away_elo, game)
            
            mov = game[self.margin_of_victory_column] if self.margin_of_victory_column else None
            
            updated_home_elo, updated_away_elo = elo_update(initial_home_elo, initial_away_elo,
                                                                           home_won, home_win_prob, self.K,
                                                                           mov)
            
            if add_elos_to_df:
                home_elos_before.append(initial_home_elo)
                vis_elos_before.append(initial_away_elo)
                
                home_elos_after.append(updated_home_elo)
                vis_elos_after.append(updated_away_elo)
                
            if add_win_probs_to_df:
                home_win_probs.append(home_win_prob)
            
            # Update records
    
            home_wins += home_won
            home_losses += away_won
            
            away_wins += away_won
            away_losses += home_won
        
            # Add to elos_map
            h_tuple = (updated_home_elo, home_wins, home_losses, season)
            a_tuple = (updated_away_elo, away_wins, away_losses, season)
            self.elos_map[home_team] = h_tuple
            self.elos_map[away_team] = a_tuple
            
        if add_elos_to_df:
            game_df['homeelobefore'] = home_elos_before
            game_df['viselobefore'] = vis_elos_before
        
            game_df['homeeloafter'] = home_elos_after
            game_df['viseloafter'] = vis_elos_after
            
        if add_win_probs_to_df:
            game_df['homewinprob'] = home_win_probs
            
    def plot_elos_distribution(self, teams: Set[str]) -> Tuple[float, float]:
        """Plots the distribution of the latest elos for the given teams, returning the mean and standard deviation.
    
        Args:
            teams(Set[str]): The teams to get elos for and plot.
            
        Returns:
            Tuple[float, float]: The mean and standard deviation of the latest elos for each team.
        """

        latest_elos = np.array([self.elos_map[team][0] for team in teams])
    
        plt.grid()
        plt.hist(latest_elos)
        plt.xlabel('Elo Rating')
        plt.ylabel('Count')
        plt.title('Elo Ratings Counts')
        plt.show()

        return np.mean(latest_elos), np.std(latest_elos)