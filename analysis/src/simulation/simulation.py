import numpy as np
import pandas as pd
from utils import basic_win_prob_for_et, elo_update, get_player_transition_matrices, get_runs_for_transition_matrix
from typing import Tuple, Dict, List, Set
import heapq
import math
from tqdm import tqdm
from scipy.stats import uniform, poisson

"""This provides functions useful to simulate an MLB season with the help of Elo ratings."""

def uniform_simulation(home_win_prob: float, home_team: str, away_team: str) -> int:
    """Simulates a game simply according to the home win probability, returning 1 if home team wins, 0 otherwise.
    
    This function does not simulate any in-game events. home_team and away_team arguments are unused and
    given only to match the signature of simulation functions that do vary by the given teams
    
    Args:
        home_win_prob (float): Probability of home team winning.
        home_team (str): Name of home team (unused).
        away_team (str): Name of away team (unused).
        
    Returns:
        int: 1 if home team wins, 0 otherwise.
    """
    return int(uniform.rvs() <= home_win_prob)

def pa_simulation(home_win_prob: float, home_team: str, away_team: str) -> int:
    """Simulates the game including each plate appearance, returning 1 if home team wins, 0 otherwise.
    
    It will leverage home_win_prob as an initial win probability, which will change over time through
    each PA.
    
    Args:
        home_win_prob (float): Probability of home team winning.
        home_team (str): Name of home team (unused).
        away_team (str): Name of away team (unused).
        
    Returns:
        int: 1 if home team wins, 0 otherwise.
    """
    

class MLBSimulator():
    
    """The MLBSimulator class, given initial Elos, enables simulation of a given schedule DataFrame, allowing
    one to see how often each team makes each round of the playoffs and wins the WS.
    
    Attributes:
        schedule (pd.DataFrame): Chronologically ordered DataFrame of matchups.
        american_league (Set[str]): American league teams.
        national_league (Set[str]): National league teams.
        initial_elos (Dict[str, float]): Mapping of each team to initial Elo rating.
        simulation_func (function): Function that takes in the initial home win probability and team names, and
            simulates the game result, returning 1 if home team wins, 0 otherwise.
        K: The K factor, determining how large the update should be.
        elo_prob_func (function): Function that takes in a home elo and away elo and optional game info,
            and produces the probability of the home team winning.
        simulate_mov (bool): Whether to simulate the margin of victory.
        team_results (Dict[str, np.array]): A mapping from each team to a numpy array containing (1) team average wins
            and percentages for how often they (2) make the playoffs, (3) make the divisional, (4) make the league championship,
            (5) make the WS, and (6) win the WS. This will be None until simulate_seasons is ran, which will update it with
            the results of the simulation.
    """
    
    def __init__(self, schedule: pd.DataFrame, american_league: Set[str], national_league: Set[str], initial_elos: Dict[str, float],
                 simulation_func=uniform_simulation, K: float=3, elo_prob_func=basic_win_prob_for_et, simulate_mov: bool=False):
        """Initializes an MLBSimulator object with the given parameters.
        
        Args:
        schedule (pd.DataFrame): Chronologically ordered DataFrame of matchups.
        american_league (Set[str]): American league teams.
        national_league (Set[str]): National league teams.
        initial_elos (Dict[str, float]): Mapping of each team to initial Elo rating.
        simulation_func (function): Function that takes in the initial home win probability and team names, and
            simulates the game result, returning 1 if home team wins, 0 otherwise.
        K: The K factor, determining how large the update should be.
        elo_prob_func (function): Function that takes in a home elo and away elo and optional game info,
            and produces the probability of the home team winning.
        simulate_mov (bool): Whether to simulate the margin of victory.
        """
        self.schedule = schedule
        self.american_league = american_league
        self.national_league = national_league
        self.initial_elos = initial_elos
        self.teams = self.initial_elos.keys()
        self.simulation_func = simulation_func
        self.K = K
        self.elo_prob_func = elo_prob_func
        self.simulate_mov = simulate_mov
        self.simulation_results = None
        
        if simulation_func == pa_simulation:
            season_players = set()
            self.player_transition_matrices = get_player_transition_matrices(season_players)
            self.R = get_runs_for_transition_matrix()

    def _simulate_game(self, home_elo: float, away_elo: float, home_team: str, away_team: str, game_info: pd.Series = None) -> Tuple[int, float, float]:
        """Returns simulated result and the updated home and away team Elos.
    
        Args:
            home_elo (float): Initial home Elo.
            away_elo (float): Initial away Elo.
            home_team (str): Name of home team.
            away_team (str): Name of away team.
        """
        home_win_prob = self.elo_prob_func(home_elo, away_elo, game_info)
    
        # Simulate result
        home_won = self.simulation_func(home_win_prob, home_team, away_team)
    
        sqrt_mov = None
        if self.simulate_mov:
            winning_elo_difference = home_elo - away_elo if home_won else away_elo - home_elo
            lambda_poisson = 1.702 + .001*winning_elo_difference
            sqrt_mov = poisson.rvs(mu=lambda_poisson - 1, loc=1)
    
        home_elo, away_elo = elo_update(home_elo, away_elo, home_won, home_win_prob, self.K, margin_of_victory=sqrt_mov)
    
        return home_won, home_elo, away_elo

    def _sim_regular_season(self) -> Dict[str, np.array]:
        """Simulates the regular season with the given schedule, updating the elos in initial_elos with the given parameters.
    
        Args:
            schedule (pd.DataFrame): Chronologically ordered DataFrame of matchups.
            initial_elos (Dict[str, float]): Mapping of each team to their initial Elo before the season starts.
            simulation_func (function): Function that takes in the initial home win probability and team names, and
                simulates the game result, returning 1 if home team wins, 0 otherwise.
            K: The K factor, determining how large the update should be.
            elo_prob_func (function): Function that takes in a home elo and away elo and optional game info, 
                and produces the probability of the home team winning.
            simulate_mov (bool): Whether to simulate the margin of victory.
            
        Returns:
            Dict[str, np.array]: Mapping from each team to their final Elo and number of wins.
        """
        season_history = {team: np.array([self.initial_elos[team], 0]) for team in self.initial_elos.keys()}
    
        for _, game in self.schedule.iterrows():
            home = game['hometeam']
            away = game['visteam']
    
            home_elo, home_wins = season_history[home][0], season_history[home][1]
            away_elo, away_wins = season_history[away][0], season_history[away][1]
    
            home_won, home_elo, away_elo = self._simulate_game(home_elo, away_elo, home, away)
            away_won = 1 - home_won
    
            # Add result to history
            season_history[home][0] = home_elo
            season_history[home][1] = home_wins + home_won
        
            season_history[away][0] = away_elo
            season_history[away][1] = away_wins + away_won
        
        return season_history

    def _get_playoff_teams(self, season_history: Dict[str, np.array]) -> Tuple[List[str], List[str]]:
        """Given the regular season results, finds the playoff teams for the american league and national league, ordered by seed.
    
        Args:
            season_history (Dict[str, np.array]): Mapping from each team to their final Elo and number of wins after the regular season.
    
        Returns:
            Tuple[List[str], List[str]]: Playoff teams for the american league and national league respectively, ordered by seed.
        """
        playoff_teams = []
    
        for league in [self.american_league, self.national_league]:
            div_playoff_teams = []
            wildcard_teams = []
            for div in league:
                best_div_teams = []
                for team in div:
                    heapq.heappush(best_div_teams, (-season_history[team][1], team)) # Make negative because PQ is negative, but we want to maximize wins
        
                # Add best team in conf to playoffs
                div_playoff_teams.append(heapq.heappop(best_div_teams))
        
                # Rest are potential wildcard teams
                for _ in best_div_teams:
                    heapq.heappush(wildcard_teams, heapq.heappop(best_div_teams))
                
            div_playoff_teams.sort(key=lambda x: x[0]) # Sort top 3 by negative wins (most wins will be most negative)
            div_playoff_teams = [team for _, team in div_playoff_teams]
            
            # Add 3 best wildcards to playoffs
            for _ in range(3):
                div_playoff_teams.append(heapq.heappop(wildcard_teams)[1])

            playoff_teams.append(div_playoff_teams)
        
        return playoff_teams[0], playoff_teams[1] # Ordered by seed

    def _sim_playoff_round(self, playoff_games: List[Tuple[str, str]], game_locations: List[bool], elos_map: Dict[str, float]) -> List[str]:
        """Simulates one round of the playoffs on one side of a bracket, updating the elos in elos_map.
    
        Args:
            playoff_games (List[Tuple[str, str]]): List of (start_home, start_away) team tuples representing matchups
                for the round.
            game_locations (List[bool]): List indicating whether each game is at the start_home team's location.
                Length should be equal to number of games in series.
            elos_map (Dict[str, float]): Mapping of each team to current Elo rating.
        """
        next_round_teams = []
    
        num_games = len(game_locations)
    
        games_to_win = math.ceil(num_games / 2)
    
        for matchup in playoff_games:
            start_home = matchup[0]
            start_away = matchup[1]
            
            # Simulate each game in series
            start_home_wins = 0
            start_away_wins = 0
        
            for start_home_is_home in game_locations: # Over each game
                if start_home_is_home:
                    home_team = start_home
                    away_team = start_away
                else:
                    home_team = start_away
                    away_team = start_home
                
                home_elo = elos_map[home_team]
                away_elo = elos_map[away_team]
        
                home_won, home_elo, away_elo = self._simulate_game(home_elo, away_elo, home_team, away_team)
                
                # Update elos
                elos_map[home_team] = home_elo
                elos_map[away_team] = away_elo
            
                start_home_won = home_won if start_home_is_home else 1 - home_won
            
                # Update wins
                start_home_wins += start_home_won
                start_away_wins += 1 - start_home_won
            
                if start_home_wins == games_to_win:
                    next_round_teams.append(start_home)
                    break
                elif start_away_wins == games_to_win:
                    next_round_teams.append(start_away)
                    break
            
        return next_round_teams

    @staticmethod
    def _get_matchups(teams: List[str], seeds: Dict[str, int]) -> List[Tuple[str, str]]:
        """Given a list of teams and their seeds, returns the playoff matchups where the team with the lower
        seed starts home.
    
        Args:
            teams (List[str]): List of playoff teams.
            seeds (Dict[str, int]): Mapping of each team to their seed.
        
        Returns:
            List[Tuple[str, str]]: List of (start_home, start_away) team tuples representing matchups for the playoff round.
        """
        matchups = []
    
        for i in range(0, len(teams), 2):
            first = teams[i]
            second = teams[i + 1]
        
            if seeds[first] <= seeds[second]:
                matchups.append((first, second))
            else:
                matchups.append((second, first))
        
        return matchups

    def _sim_playoffs(self, al_playoff_teams: List[str], nl_playoff_teams: List[str], elos_map: Dict[str, float]):
        """Simulates playoffs using the AL and NL teams, where the first 2 in each list are the 1 and 2 seed with byes and the other 4 are WC teams.
    
        Returns a tuple containing a 2d list of teams that made the divisional, 2d list of the teams that made the league championships,
        1d list of the WS teams, and the WS winner.
    
        Args:
            al_playoff_teams (List[str]): AL playoff teams ordered by seed.
            nl_playoff_teams (List[str]): NL playoff teams ordered by seed.
            elos_map (Dict[str, float]): Mapping of each team to current Elo rating.
        """
    
        elos_map = elos_map.copy()
    
        # Constant time seed lookup, plus we reorder the teams anyway.
        al_seeds = {al_playoff_teams[i]: i+1 for i in range(len(al_playoff_teams))}
        nl_seeds = {nl_playoff_teams[i]: i+1 for i in range(len(nl_playoff_teams))}
    
        # Reorder for wildcard - [2, 3, 6, 1, 4, 5]
        # Current index - 0,1,2,3,4,5 -> new index
        wc_order = [0, 3, 4, 1, 2, 5]
    
        #print(al_playoff_teams)
    
        al_playoff_teams = [al_playoff_teams[wc_order[i]] for i in range(len(al_playoff_teams))]
        nl_playoff_teams = [nl_playoff_teams[wc_order[i]] for i in range(len(nl_playoff_teams))]
    
        #print(al_playoff_teams)
        #print(nl_playoff_teams)
    
        al_wc_teams = al_playoff_teams[1:3] + al_playoff_teams[4:]
        nl_wc_teams = nl_playoff_teams[1:3] + nl_playoff_teams[4:]
    
        al_wc_matchups = self._get_matchups(al_wc_teams, al_seeds)
        nl_wc_matchups = self._get_matchups(nl_wc_teams, nl_seeds)
    
        # Get wildcard results
        wc_home_locations = [True, False, True]
        al_divisional_teams = self._sim_playoff_round(al_wc_matchups, wc_home_locations, elos_map)
        nl_divisional_teams = self._sim_playoff_round(nl_wc_matchups, wc_home_locations, elos_map)
    
        # Insert 1 and 2 seeds that had byes
        al_divisional_teams.insert(0, al_playoff_teams[0])
        al_divisional_teams.insert(2, al_playoff_teams[3])
    
        nl_divisional_teams.insert(0, nl_playoff_teams[0])
        nl_divisional_teams.insert(2, nl_playoff_teams[3])
    
        #print(al_divisional_teams)
        #print(nl_divisional_teams)
    
        al_divisional_matchups = self._get_matchups(al_divisional_teams, al_seeds)
        nl_divisional_matchups = self._get_matchups(nl_divisional_teams, nl_seeds)
    
        div_home_locations = [True, True, False, False, True]
        # Simulate divisional games - 1 vs 4/5, 2 vs 3/6
        al_championship_teams = self._sim_playoff_round(al_divisional_matchups, div_home_locations, elos_map)
        nl_championship_teams = self._sim_playoff_round(nl_divisional_matchups, div_home_locations, elos_map)
    
        #print(al_championship_teams)
        #print(nl_championship_teams)
    
        al_championship_matchup = self._get_matchups(al_championship_teams, al_seeds)
        nl_championship_matchup = self._get_matchups(nl_championship_teams, nl_seeds)
    
        # Simulate al and nl championships
        champ_ws_home_locations = [True, True, False, False, False, True, True]
        al_ws_team = self._sim_playoff_round(al_championship_matchup, champ_ws_home_locations, elos_map)[0]
        nl_ws_team = self._sim_playoff_round(nl_championship_matchup, champ_ws_home_locations, elos_map)[0]
    
        #print(al_ws_team)
        #print(nl_ws_team)
    
        ws_matchup = self._get_matchups([al_ws_team, nl_ws_team], {al_ws_team:al_seeds[al_ws_team], nl_ws_team:nl_seeds[nl_ws_team]})
    
        # Simulate WS
        ws_winner = self._sim_playoff_round(ws_matchup, champ_ws_home_locations, elos_map)[0]
    
        #print(ws_winner)
    
        return [al_divisional_teams, nl_divisional_teams], [al_championship_teams, nl_championship_teams], [al_ws_team, nl_ws_team], ws_winner

    def simulate_seasons(self, iterations: int) -> None:
        """Simulates the given number of MLB seasons with the given schedule, recording for each team average wins and percentages for how often
        they make the playoffs, make the divisional, make the league championship, make the WS, and win the WS into self.simulation_results.
    
        Args:
            iterations (int): Number of seasons to simulate.
        """
    
        team_results = {team: np.array([0,0,0,0,0,0]) for team in self.teams} # wins, make playoffs, make divisional, make champ., make ws, win ws
    
        for _ in tqdm(range(iterations)):
            season_history = self._sim_regular_season()
            # Add # RS wins to team_results
            for team in self.teams:
                team_results[team][0] += season_history[team][1]
            
            # Get playoff teams
            al_playoff_teams, nl_playoff_teams = self._get_playoff_teams(season_history)
        
            for team in al_playoff_teams + nl_playoff_teams:
                team_results[team][1] += 1
                
            elos_map = {team: season_history[team][0] for team in self.teams}
        
            # Get results for playoffs
            divisional_teams, champ_teams, ws_teams, ws_winner = self._sim_playoffs(al_playoff_teams, nl_playoff_teams, elos_map)
        
            for team in divisional_teams[0] + divisional_teams[1]:
                team_results[team][2] += 1
            
            for team in champ_teams[0] + champ_teams[1]:
                team_results[team][3] += 1
            
            for team in ws_teams:
                team_results[team][4] += 1
            
            team_results[ws_winner][5] += 1
        
        #print(team_results)
        
        team_results = {team:team_results[team] / iterations for team in self.teams} # divide by iteratoins to get average or percents
    
        # convert percents to be between 0 and 100
        team_results = {team:team_results[team]*np.array([1,100,100,100,100,100]) for team in self.teams}
    
        self.team_results = team_results