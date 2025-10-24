import numpy as np

"""This provides functions useful to simulate an MLB season with the help of Elo ratings."""

def prob_home_wins(home_elo, away_elo):
    """Given the Elos for the home and away teams, fetches the probability the home team wins."""
    return 1 / (1+10**((away_elo - home_elo) / 400))

def simulate_game(home_elo, away_elo, K=25):
    # given home and away elos, simulates the game and returns 1 if home won else 0, the updated home elo, and updated away elo, using the given K factor
    home_win_prob = prob_home_wins(home_elo, away_elo)
    away_win_prob = 1 - home_win_prob
    
    # Simulate result
    home_won = int(np.random.uniform() <= home_win_prob)
        
    away_won = 1 - home_won
    
    # Update elos
    home_elo = home_elo + int(K*(home_won - home_win_prob))
    away_elo = away_elo + int(K*(away_won - away_win_prob))
    
    return home_won, home_elo, away_elo

def sim_regular_season(initial_elos, K=25):
    # Sims the regular season, using the initial team elos dict, producing a dict mapping each team
    # to np.arrays storing their final elos, # wins, and # losses.
    # Uses K-factor K for Elo update sensitivity
    season_history = {team: np.array([initial_elos[team], 0, 0]) for team in TEAMS}
    
    for game in schedule:
        home = game[0]
        away = game[1]
    
        home_elo, home_wins, home_losses = season_history[home][0], season_history[home][1], season_history[home][2]
        away_elo, away_wins, away_losses = season_history[away][0], season_history[away][1], season_history[away][2]
    
        home_won, home_elo, away_elo = simulate_game(home_elo, away_elo, K)
        away_won = 1 - home_won
    
        # Add result to history
        season_history[home] = np.array([home_elo, home_wins + home_won, home_losses + (1 - home_won)])
        season_history[away] = np.array([away_elo, away_wins + away_won, away_losses + (1 - away_won)])
        
    return season_history

def sim_playoff_round(playoff_teams, season_history, K=25):
    # Simulates one round of the playoffs on one side of a bracket
    # Updates the elos in the given dictionary as the games are simulated
    
    next_round_teams = []
    
    num_teams = len(playoff_teams)
    
    home_idx = 0 # 0-based indices for current home and away teams
    away_idx = num_teams - 1
    # 2 pointer method
    for _ in range(num_teams // 2): # 3 wildcard games per bracket side
        home = playoff_teams[home_idx]
        away = playoff_teams[away_idx]
            
        home_elo = season_history[home][0]
        away_elo = season_history[away][0]
            
        home_won, home_elo, away_elo = simulate_game(home_elo, away_elo, K)
        if home_won == 1:
            next_round_teams.append(home)
        else:
            next_round_teams.append(away)
                
        # Update elos
        season_history[home][0] = home_elo
        season_history[away][0] = away_elo
            
        home_idx += 1
        away_idx -= 1
            
    return next_round_teams

def get_playoff_teams(season_history):
    # Given the regular season results, finds the playoff NFC and AFC teams
    playoff_teams = []
    
    for div in [NFC_CONFS, AFC_CONFS]:
        div_playoff_teams = []
        wildcard_teams = []
        for conf in div:
            best_conf_teams = []
            for team in conf:
                heapq.heappush(best_conf_teams, (-season_history[team][1], team)) # Make negative because PQ is negative, but we want to maximize
        
            # Add best team in conf to playoffs
            heapq.heappush(div_playoff_teams, heapq.heappop(best_conf_teams))
        
            # Rest are potential wildcard teams
            for _ in best_conf_teams:
                heapq.heappush(wildcard_teams, heapq.heappop(best_conf_teams))
            
        # Add 3 best wildcards to playoffs
        for _ in range(3):
            div_playoff_teams.append(heapq.heappop(wildcard_teams))

        # Only get teams, not wins
        div_playoff_teams = list(map(lambda x: x[1], div_playoff_teams))
        
        playoff_teams.append(div_playoff_teams)
        
    return playoff_teams[0], playoff_teams[1]

def sim_playoffs(nfc_playoff_teams, afc_playoff_teams, season_history, K=25):
    # Simulates playoffs using the NFC and AFC playoff teams, where the first 4 in each list are the division winners ordered by wins, and the remaining 3 are the wilcard teams ordered by wins
    # Returns a tuple containing a 2d list of teams that made the divisoinal, 2d list of the teams that made the conference championships, 1d list of the superbowl teams, and the superbowl winner
    # Updates elos in season_history after simulating each game
    
    seeds = {team:idx+1 for idx, team in enumerate(nfc_playoff_teams)}
    seeds.update({team:idx+1 for idx, team in enumerate(afc_playoff_teams)})
    
    # Get wildcard results
    nfc_divisional_teams = sim_playoff_round(nfc_playoff_teams, season_history, K=K)
    afc_divisional_teams = sim_playoff_round(afc_playoff_teams, season_history, K=K)
    # Insert 1 seeds that had byes
    nfc_divisional_teams.insert(0, nfc_playoff_teams[0])
    afc_divisional_teams.insert(0, afc_playoff_teams[0])
    
    # Divisional - reorganize teams - sort by seeding
    nfc_divisional_teams.sort(key=lambda x: seeds[x])
    afc_divisional_teams.sort(key=lambda x: seeds[x])
    
    # Simulate divisional games - 1 vs worst, other 2 against each other
    nfc_championship_teams = sim_playoff_round(nfc_divisional_teams, season_history, K=K)
    afc_championship_teams = sim_playoff_round(afc_divisional_teams, season_history, K=K)
    
    # Simulate NFC and AFC championships
    nfc_sb_team = sim_playoff_round(nfc_divisional_teams, season_history, K=K)[0]
    afc_sb_team = sim_playoff_round(afc_divisional_teams, season_history, K=K)[0]
    
    # Simulate superbowl
    sb_winner = sim_playoff_round([nfc_sb_team, afc_sb_team], season_history, K=K)[0]
    
    return [nfc_divisional_teams, afc_divisional_teams], [nfc_championship_teams, afc_championship_teams], [nfc_sb_team, afc_sb_team], sb_winner

def simulate_seasons(iterations, initial_elos, K=25):
    # Simulates the given number of NFL seasons, recording for each team average wins and percentages for how often
    # they make the playoffs, make the divisional, make the div championship, make the SB, and win the SB
    team_results = {team: np.array([0,0,0,0,0,0]) for team in TEAMS} # wins, make playoffs, make divisional, make champ., make sb, win sb
    
    for _ in tqdm(range(iterations)):
        season_history = sim_regular_season(initial_elos, K=25)
        # Add # RS wins to team_results
        for team in TEAMS:
            team_results[team][0] += season_history[team][1]
            
        # Get playoff teams
        nfc_playoff_teams, afc_playoff_teams = get_playoff_teams(season_history)
        
        for team in nfc_playoff_teams + afc_playoff_teams:
            team_results[team][1] += 1
            
        # Get results for playoffs
        divisional_teams, champ_teams, sb_teams, sb_winner = sim_playoffs(nfc_playoff_teams, afc_playoff_teams, season_history, K=K)
        
        for team in divisional_teams[0] + divisional_teams[1]:
            team_results[team][2] += 1
            
        for team in champ_teams[0] + champ_teams[1]:
            team_results[team][3] += 1
            
        for team in sb_teams:
            team_results[team][4] += 1
            
        team_results[sb_winner][5] += 1
        
    #print(team_results)
        
    team_results = {team:team_results[team] / iterations for team in TEAMS} # divide by iteratoins to get average or percents
    
    # convert percents to be between 0 and 100
    team_results = {team:team_results[team]*np.array([1,100,100,100,100,100]) for team in TEAMS}
    
    return team_results

team_results = simulate_seasons(1000, RATINGS_MAP)
# Make DF for team forecasts
avg_wins = [team_results[team][0] for team in TEAMS_LIST]
playoff_pcts = [team_results[team][1] for team in TEAMS_LIST]
div_pcts = [team_results[team][2] for team in TEAMS_LIST]
champ_pcts = [team_results[team][3] for team in TEAMS_LIST]
sb_pcts = [team_results[team][4] for team in TEAMS_LIST]
sb_win_pcts = [team_results[team][5] for team in TEAMS_LIST]
print(pd.DataFrame({'Team':TEAMS_LIST, 'Avg. Wins':avg_wins, 'Make Playoff':playoff_pcts, 'Make Divisional':div_pcts, 'Make Champ.':champ_pcts, 'Make SB':sb_pcts, 'Win SB':sb_win_pcts}).sort_values(by='Win SB', ascending=False).reset_index(drop=True))