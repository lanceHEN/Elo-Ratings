import pandas as pd
from typing import Tuple, Set
from utils.utils import get_prev_date_midnight

class EloTracker(object):
    """This class provides an interface to store and add to team
    Elo ratings over time.
    
    Attributes:
        elos_map (Dict[str, List[Tuple[pd.Timestamp, float, int, int, int]]]): Mapping from each team to a 
            chronologically ordered list of tuples containing:
            (1) the date/time their Elo updated,
            (2) their Elo after that update occurred,
            (3) their number of wins after that update occurred,
            (4) their number of losses after that update occurred,
            (5) the current season.
            This is the centerpoint of this class and may be referenced at any time
            to observe a team's Elo history.
        initial_elo (float): The initial Elo rating for each team. This will be used for the
            first entry in elos_map[team] once it is created, the day before the first
            game they eventually play.
        K (float): The K factor, controlling how sensitive each Elo update should be.
    """
    
    def __init__(self, teams: Set[str], initial_elo: float=1500, K: float=25):
        """Constructs an EloTracker from scratch with empty listings for each team.
        
        Args:
            teams (Set[str]): Set of teams to collect Elos for.
            initial_elo (float): The initial Elo rating for each team. This will be used for the
                first entry in elos_map[team] once it is created, the day before the first
                game they eventually play.
            K (float): The K factor, controlling how sensitive each Elo update should be.
        """
        self.elos_map = {team: [] for team in teams}
        self.initial_elo = initial_elo
        self.K = K
        
    @staticmethod
    def _prob_home_wins(home_elo: float, away_elo: float) -> float:
        """Fetches the probability the home team wins, given each team's Elo.
    
        The probability is given by 1 / (1+10^((away_elo - home_elo) / 400).
    
        Args:
            home_elo (float): Home team Elo.
            away_elo (float): Away team Elo.
        
        Returns:
            float: The probability the home team wins.
        """
        return 1 / (1+10**((away_elo - home_elo) / 400))
    
    @staticmethod
    def _elo_update(home_elo: float, away_elo: float, home_won: int, K: float=25) -> Tuple[float, float]:
        """Returns updated home and away team Elos, given a result.
    
        Args:
            home_elo (float): Initial home Elo.
            away_elo (float): Initial away Elo.
            home_won (int): 1 if home team won, else 0.
            K: The K factor, determining how large the update should be.
        """
        home_win_prob = EloTracker._prob_home_wins(home_elo, away_elo)
        away_win_prob = 1 - home_win_prob
    
        away_won = 1 - home_won
    
        # Update elos
        home_elo = home_elo + int(K*(home_won - home_win_prob))
        away_elo = away_elo + int(K*(away_won - away_win_prob))
    
        return home_elo, away_elo
    
    def _initialize_history_if_necessary(self, team: str, dt: pd.Timestamp, season: int) -> None:
        """If elos_map[team] == [], then an initial entry will created in elos_map
        for the given team for 00:00:00 the day before the given timestamp,
        with their Elo being initial_elo, and having 0 wins and 0 losses for the given season.
        
        Args:
            team (str): The team to check.
            dt (pd.Timestamp): The game timestamp.
            season (int): The season the game takes place in.
        """
        if self.elos_map[team] == []:
            t = (get_prev_date_midnight(dt), self.initial_elo, 0, 0, season)
            self.elos_map[team].append(t)
            
    def _initialize_new_season_entry_if_necessary(self, team: str, dt: pd.Timestamp, season: int) -> None:
        """If the last recorded season in elos_map[team] < season, this will add another
        entry in elos_map for the given team for 00:00:00 the day before the given timestamp,
        with their Elo being their previous elo reverted to initial_elo by 1/3, and having
        0 wins and 0 losses.
        
        Args:
            team (str): The team to check.
            dt (pd.Timestamp): The game timestamp.
            season (int): The possibly new season to check.
        """
        if self.elos_map[team][-1][4] < season:
            old_elo = self.elos_map[team][-1][1]
            new_elo = old_elo + int((self.initial_elo - old_elo) / 3)
            
            t = (get_prev_date_midnight(dt), new_elo, 0, 0, season)
            self.elos_map[team].append(t)
    
    def add_history(self, game_df: pd.DataFrame) -> None:
        """Adds the result and updated Elo for every game in game_df to self.elos_map.
        
        If a team in a game has never played before (i.e. self.elos_map[team] == []),
        then an initial entry will created for 00:00:00 the day before the game,
        with their Elo being initial_elo, and having 0 wins and 0 losses.
        
        If at any point a game takes place in a season beyond the one last logged in
        elos_map, there will be an additional entry added, before the one for that game,
        containing the team's previous elo reverted to initial_elo by 1/3, 0 wins, and
        0 losses.
        
        Args:
            game_df (pd.DataFrame): Table whose rows are chronologically ordered game box scores,
                including columns 'hometeam' for the home team, 'visteam' for the away team, and
                'homewon' which is 1 if home won and 0 otherwise. Each game in game_df must take
                place after the games that have already been logged for the given teams it includes.
        """
        
        for _, game in game_df.iterrows():
            home_team = game['hometeam']
            away_team = game['visteam']
            
            # Get timestamp of game
            timestamp = game['timestamp']
            
            season = game['season']
            
            # Adds initial entry to elos_map for the team if it is empty
            self._initialize_history_if_necessary(home_team, timestamp, season)
            self._initialize_history_if_necessary(away_team, timestamp, season)
            
            # Checks if current season > last recorded season, if so reverting to initial_elo by 1/3.
            self._initialize_new_season_entry_if_necessary(home_team, timestamp, season)
            self._initialize_new_season_entry_if_necessary(away_team, timestamp, season)
            
            # Final result
            home_won = game['homewon']
            away_won = 1 - home_won
            
            home_elo = self.elos_map[home_team][-1][1]
            away_elo = self.elos_map[away_team][-1][1]
            
            home_elo, away_elo = EloTracker._elo_update(home_elo, away_elo, home_won, self.K)
            
            # Update records
        
            home_wins = self.elos_map[home_team][-1][2]
            home_losses = self.elos_map[home_team][-1][3]
        
            home_wins += home_won
            home_losses += away_won
            
            away_wins = self.elos_map[away_team][-1][2]
            away_losses = self.elos_map[away_team][-1][3]
        
            away_wins += away_won
            away_losses += home_won
        
            # Add to elos_map
            h_tuple = (timestamp, home_elo, home_wins, home_losses, season)
            a_tuple = (timestamp, away_elo, away_wins, away_losses, season)
            self.elos_map[home_team].append(h_tuple)
            self.elos_map[away_team].append(a_tuple)
        