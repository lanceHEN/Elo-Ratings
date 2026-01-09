from typing import Dict, Iterable, Tuple
from pathlib import Path
import math

import pandas as pd
from tqdm import tqdm
import numpy as np

from .misc_utils import TEAM_RENAME_MAP

class DFPreprocessor:
    """
    The DFPreprocessor class, given a pandas dataframe, allows one to perform
    common preprocessing operations on it in-place.
    
    Attributes:
        df (pd.DataFrame): The Dataframe to preprocess.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes a DFPreprocessor with the given pandas dataframe.
        """
        self.df = df

    def rename_teams(self, team_cols: Iterable[str]) -> None:
        """
        For each column in team_cols in the dataframe, renames any key found in TEAM_RENAME_MAP to its value.
        """
        for team_col in team_cols:
            self.df[team_col] = self.df[team_col].apply(lambda x: TEAM_RENAME_MAP[x] if x in TEAM_RENAME_MAP else x)
            
    def add_season_col(self, date_col: str = 'date') -> None:
        """
        Adds a 'season' column to the dataframe, according to a date string
        of form 'YYYYMMDD', e.g. '20250729', given by column date_col.
        """
        self.df['season'] = self.df[date_col].apply(lambda x: int(str(x)[:4]))
        
    def add_distance_traveled_cols(self, home_team_col: str, vis_team_col: str, location_col: str, over_seasons: bool = True) -> None:
        """
        Adds 'homedistancetraveled' and 'visdistancetraveled' columns to the dataframe,
        using the given team names and location column. Assumes df has a 'timestamp' column.
        If over_seasons, this will use the 'season' column to keep track of the
        season each game occurred in--if a game is the first of the season, each
        team will be assumed to travel from their home stadium location.
        """
        df = self.df.copy()
        # Add temporary latitude and longitude columns
        parks = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "raw" / "Parks.csv")
        df = pd.merge(df, parks[['PARKID', 'Latitude', 'Longitude']], how='left', left_on=location_col, right_on='PARKID').reset_index(drop=True)
        df = df.drop('PARKID', axis=1)
        
        def haversine(lat1, lon1, lat2, lon2):
            """Returns Haversine distance between two pairs of latitudes and longitudes."""
            R = 3958.8  # Earth radius in miles
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
            return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        def get_closest_home_game_coords(team: str, idx: int) -> Tuple[float, float]:
            """For the given team, returns the latitude and longitude of the closest home game they played
            to wherever they are in schedule, given by idx. If no home game is found, it returns None, None."""
    
            n_games = len(df)
    
            before = idx - 1
            after = idx + 1
            while before >= 0 or after <= n_games - 1:
                if before >= 0:
                    before_gm = df.iloc[before]
                    if before_gm[home_team_col] == team:
                        return before_gm['Latitude'], before_gm['Longitude']
            
                    before -= 1

                if after <= n_games - 1:
                    after_gm = df.iloc[after]
                    if after_gm[home_team_col] == team:
                        return after_gm['Latitude'], after_gm['Longitude']
            
                    after += 1
            
            return None, None
        
        home_dists = []
        away_dists = []

        last_games = {} # Mapping of team id to (lat, lon, season) tuple of previous game - if empty, just use np.nan

        for i, game in tqdm(df.iterrows()):
    
            cur_lat = game['Latitude']
            cur_lon = game['Longitude']
    
            home_team = game[home_team_col]
            away_team = game[vis_team_col]
            
            if over_seasons:
                season = game['season']
    
            for team, dists in zip([home_team_col, vis_team_col], [home_dists, away_dists]):
                if team in last_games:
                    last_lat, last_lon, last_season = last_games[team]
            
                    if over_seasons and season != last_season: # If last game was a season ago - not always traveling from the game from the past season
                        last_lat, last_lon = get_closest_home_game_coords(team, i)
                       
            
                else: # Never played a game before
                    last_lat, last_lon = get_closest_home_game_coords(team, i)
            
                if last_lat is not None:
                    dist = haversine(last_lat, last_lon, cur_lat, cur_lon)
                    dists.append(dist)
                else:
                    dists.append(np.nan)
            
                last_games[team] = (cur_lat, cur_lon, season)

        # Finally add columns in place
        self.df['homedistancetraveled'] = home_dists    
        self.df['visdistancetraveled'] = away_dists
        
    def add_rest_days_cols(self, home_team_col: str, vis_team_col: str) -> None:
        """
        Adds 'homerestdays' and 'visrestdays' columns to the dataframe,
        using the given team names. Assumes df has a 'timestamp' column.
        """
        home_rest_days = []
        away_rest_days = []

        last_played = {}

        for _, game in tqdm(self.df.iterrows()):
            home_team = game[home_team_col]
            away_team = game[vis_team_col]
            timestamp = game['timestamp']
    
            prev_home_t = last_played.get(home_team)
            prev_away_t = last_played.get(away_team)
    
            home_rest_days.append((timestamp.floor('D') - prev_home_t.floor('D')).days if prev_home_t is not None else np.nan)
            away_rest_days.append((timestamp.floor('D') - prev_away_t.floor('D')).days if prev_away_t is not None else np.nan)

            last_played[home_team] = timestamp
            last_played[away_team] = timestamp
 
        self.df['homerestdays'] = home_rest_days
        self.df['visrestdays'] = away_rest_days