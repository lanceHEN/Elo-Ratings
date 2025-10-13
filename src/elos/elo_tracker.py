import pandas as pd
from typing import Tuple, Set
from utils.utils import get_prev_date_midnight

class EloTracker(object):
    """This class provides an interface to store and add to team
    Elo ratings over time.
    
    Attributes:
        elos_map (Dict[str, List[Tuple[pd.Timestamp, float, int, int, int]]]): Mapping from each team to a 
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
            
    def _get_initial_team_stats(self, team: str, season: int) -> Tuple[float, int, int, bool]:
        """Fetches the initial Elo, wins and losses for the team.
        
        If elos_map[team] is empty, it will produce initial_elo,
        along with 0 wins and 0 losses.
        
        If the season is a new season, it will be the team's previous elo reverted to initial_elo by 1/3,
        along with 0 wins and 0 losses.
        
        Otherwise, it will just be the previous Elo, and the last recorded wins and losses.
        
        Lastly, this returns true if it is the first game for the team,
        whether ever or for the season, and False otherwise.
        
        Args:
            team (str): The team to check.
            season (int): The possibly new season to check.
            
        Returns:
            Tuple[float, int, int, bool]: A tuple containing:
                (1) The initial Elo,
                (2) the initial wins,
                (3) the initial losses,
                (4) boolean flag for whether it's the first game for the team either ever or for the season.
        """
        
        if self.elos_map[team] == []:
            return self.initial_elo, 0, 0, True

        elif self.elos_map[team][-1][7] < season:
            old_elo = self.elos_map[team][-1][3]
            new_elo = old_elo + int((self.initial_elo - old_elo) / 3)
            return new_elo, 0, 0, True
        
        else:
            wins = self.elos_map[team][-1][5]
            losses = self.elos_map[team][-1][6]
            return self.elos_map[team][-1][3], wins, losses, False
    
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
                'homewon' which is True if home won and False otherwise. Each game in game_df must take
                place after the games that have already been logged for the given teams it includes.
                Must be indexed by a game id column 'gid'.
        """
        
        for game_id, game in game_df.iterrows():
            home_team = game['hometeam']
            away_team = game['visteam']
            
            # Get timestamp of game
            timestamp = game['timestamp']
            
            season = game['season']
            
            # Get initial elos, home wins and losses, and first game flag
            initial_home_elo, home_wins, home_losses, home_first_game = self._get_initial_team_stats(home_team, season)
            initial_away_elo, away_wins, away_losses, away_first_game = self._get_initial_team_stats(away_team, season)
            
            # Final result
            home_won = int(game['homewon'])
            away_won = 1 - home_won
            
            updated_home_elo, updated_away_elo = EloTracker._elo_update(initial_home_elo, initial_away_elo, home_won, self.K)
            
            # Update records
    
            home_wins += home_won
            home_losses += away_won
            
            away_wins += away_won
            away_losses += home_won
        
            # Add to elos_map
            h_tuple = (game_id, timestamp, initial_home_elo, updated_home_elo, bool(home_won), home_wins, home_losses, season, home_first_game)
            a_tuple = (game_id, timestamp, initial_away_elo, updated_away_elo, bool(away_won), away_wins, away_losses, season, away_first_game)
            self.elos_map[home_team].append(h_tuple)
            self.elos_map[away_team].append(a_tuple)