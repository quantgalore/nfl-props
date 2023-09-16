# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import time
import meteostat
from datetime import datetime, timedelta

import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

stadiums = pd.read_excel("nfl_stadiums.xlsx")

def celsius_to_fahrenheit(celsius):
    
    return ( celsius * (9/5) ) + 32
    
def km_to_mph(km):
    
    return round(km * 0.621371, 2)

receiver_positions = ["Tight End", "Wide Receiver", "Running Back", "Fullback"]

active_months = [1, 2, 8, 9, 10, 11, 12]

# the code logic assumes that the today = the date of game you want to bet on
today = datetime.today()# + timedelta(days = 1)

prop_odds_api_key = "your prop-odds.com API KEY"

reception_yards_market = "player_receiving_yds_over_under"
receptions_market = "player_receptions_over_under"
longest_reception_market = "player_longest_reception"

todays_games = pd.json_normalize(requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={today.strftime('%Y%m%d')}").json()["events"])
todays_games_from_prop_odds = pd.json_normalize(requests.get(f"https://api.prop-odds.com/beta/games/nfl?date={today.strftime('%Y-%m-%d')}&tz=America/New_York&api_key={prop_odds_api_key}").json()["games"]).set_index("start_timestamp")

all_dates = pd.date_range(start = "2015-09-07", end = today)
dates = all_dates[all_dates.month.isin(active_months)].strftime("%Y%m%d")

# start

reception_records_by_game = []
start_time = datetime.now()
  
for date in dates:
    
    try:

        available_games = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}").json()["events"]
        
    except Exception:
        continue
    
    if len(available_games) < 1:
        continue

    for game in available_games:
        
        if "AFC" in game["name"]:
            continue
        
        game_id = int(game["id"])
        game_date = pd.to_datetime(game["date"]).tz_convert("US/Eastern")
        
        play_by_play_data = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}").json()
        
        if len(play_by_play_data) < 4:
            continue

        game_venue = play_by_play_data["gameInfo"]["venue"]["fullName"]
        
        if date == dates[-1]:
            
            reception_dataframes = []
            teams = play_by_play_data["boxscore"]["teams"]
            
            for team in teams:
                
                team_name = team["team"]["displayName"]
                team_id = team["team"]["id"]
            
                team_roster = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster").json()
                
                offense = pd.json_normalize(team_roster["athletes"][0]["items"])
                active_offense = offense[offense["status.type"] == "active"]
                
                quarterbacks = active_offense[active_offense["position.name"] == "Quarterback"]
                receivers = active_offense[active_offense["position.name"].isin(receiver_positions)]
                
                quarterback_names = quarterbacks["displayName"].values
                receiver_names = receivers["displayName"].values
                receiver_ids = receivers["id"].values
                
                receiver_dataframes = []
                
                for quarterback in range(0, len(quarterback_names)):
                
                    receiver_reception_dataframe = pd.DataFrame({"game_date":game_date,"team_quarterback":quarterback_names[quarterback],
                                                                 "athlete_name": receiver_names, "athlete_id": receiver_ids,
                                                                 "game_id":game_id, "game_venue":game_venue,
                                                                 "athlete_receptions": np.nan,
                                                                 "athlete_reception_yards": np.nan,
                                                                 "athlete_average_yards_per_reception": np.nan,
                                                                 "athlete_touchdowns": np.nan,
                                                                 "athlete_longest_reception": np.nan,
                                                                 "athlete_reception_attempts": np.nan})
                    
                    receiver_dataframes.append(receiver_reception_dataframe)
                    
                athlete_reception_dataframe = pd.concat(receiver_dataframes)
                reception_dataframes.append(athlete_reception_dataframe)
                
            complete_game_reception_dataframe = pd.concat(reception_dataframes)
            #### weather calc
            
            stadium_of_game = stadiums[stadiums["stadium_name"] == game_venue]
            
            if len(stadium_of_game) < 1:
                continue
            
            game_start_time = game_date.strftime("%Y-%m-%d")
            # create point object consisting of latitude and longitude
            meteostat_point_object = meteostat.Point(lat = stadium_of_game["lat"].iloc[0], lon = stadium_of_game["long"].iloc[0])
            # get the hourly data
            meteostat_hourly_data = meteostat.Hourly(loc = meteostat_point_object, start = pd.to_datetime(game_start_time), end = pd.to_datetime(game_start_time) + timedelta(days = 1), timezone = "US/Eastern").fetch()
            # get the data for 1 hour into the game
            
            hour_into_the_game = meteostat_hourly_data[meteostat_hourly_data.index <= game_date + timedelta(hours = 1)].tail(1)
            
            game_hour_weather = celsius_to_fahrenheit(hour_into_the_game["temp"].iloc[0])
            game_wind_speed = km_to_mph(hour_into_the_game["wspd"].iloc[0])
            game_wind_direction = hour_into_the_game["wdir"].iloc[0]
            game_relative_humidity = hour_into_the_game["rhum"].iloc[0]
            game_precipitation = hour_into_the_game["prcp"].iloc[0]
            
            #### weather calc
            complete_game_reception_dataframe["temp"] = game_hour_weather
            complete_game_reception_dataframe["wind_speed"] = game_wind_speed
            complete_game_reception_dataframe["wind_direction"] = game_wind_direction
            complete_game_reception_dataframe["relative_humidity"] = game_relative_humidity
            complete_game_reception_dataframe["precipitation"] = game_precipitation
            
            reception_records_by_game.append(complete_game_reception_dataframe)
            
            continue
                    
        game_stats = play_by_play_data["boxscore"]["players"]
        
        reception_dataframes = []
        
        for team in game_stats:
            
            team_name = team["team"]["displayName"]
            
            if len(team["statistics"]) < 1:
                continue
            
            try:
            
                team_quarterback = team["statistics"][0]["athletes"][0]["athlete"]["displayName"]
                
            except:
                break
            
            receiving_stats = team["statistics"][2]
            
            if receiving_stats["name"] != "receiving":
                continue
            
            receiving_athletes = receiving_stats["athletes"]
            
            athlete_stat_list = []
            
            for athlete in receiving_athletes:
                
                athlete_id = int(athlete["athlete"]["id"])
                athlete_name = athlete["athlete"]["displayName"]
                
                # labels are receptions[0], receiving yards[1], yards per reception[2], touchdowns[3], long receptions[4], and targets[5]
                athlete_receptions = float(athlete["stats"][0])
                athlete_reception_yards = float(athlete["stats"][1])
                athlete_average_yards_per_reception = float(athlete["stats"][2])
                athlete_touchdowns = float(athlete["stats"][3])
                athlete_longest_reception = float(athlete["stats"][4])
                athlete_reception_attempts = float(athlete["stats"][5])
                
                athlete_reception_dataframe = pd.DataFrame([{"game_date":game_date,"team_quarterback":team_quarterback,
                                                             "athlete_name": athlete_name, "athlete_id": athlete_id,
                                                             "game_id":game_id, "game_venue":game_venue,
                                                             "athlete_receptions": athlete_receptions,
                                                             "athlete_reception_yards": athlete_reception_yards,
                                                             "athlete_average_yards_per_reception": athlete_average_yards_per_reception,
                                                             "athlete_touchdowns": athlete_touchdowns,
                                                             "athlete_longest_reception": athlete_longest_reception,
                                                             "athlete_reception_attempts": athlete_reception_attempts}])
            
                athlete_stat_list.append(athlete_reception_dataframe)
                
            team_stat_dataframe = pd.concat(athlete_stat_list)
            reception_dataframes.append(team_stat_dataframe)
            
        if len(reception_dataframes) < 1:
            continue
            
        complete_game_reception_dataframe = pd.concat(reception_dataframes)
        
        #### weather calc
        
        stadium_of_game = stadiums[stadiums["stadium_name"] == game_venue]
        
        if len(stadium_of_game) < 1:
            continue
        
        game_start_time = game_date.strftime("%Y-%m-%d")
        # create point object consisting of latitude and longitude
        meteostat_point_object = meteostat.Point(lat = stadium_of_game["lat"].iloc[0], lon = stadium_of_game["long"].iloc[0])
        # get the hourly data
        meteostat_hourly_data = meteostat.Hourly(loc = meteostat_point_object, start = pd.to_datetime(game_start_time), end = pd.to_datetime(game_start_time) + timedelta(days = 1), timezone = "US/Eastern").fetch()
        # get the data for 1 hour into the game
        
        hour_into_the_game = meteostat_hourly_data[meteostat_hourly_data.index <= game_date + timedelta(hours = 1)].tail(1)
        
        if len(hour_into_the_game) <1:
            continue
        
        game_hour_weather = celsius_to_fahrenheit(hour_into_the_game["temp"].iloc[0])
        game_wind_speed = km_to_mph(hour_into_the_game["wspd"].iloc[0])
        game_wind_direction = hour_into_the_game["wdir"].iloc[0]
        game_relative_humidity = hour_into_the_game["rhum"].iloc[0]
        game_precipitation = hour_into_the_game["prcp"].iloc[0]
        
        #### weather calc
        complete_game_reception_dataframe["temp"] = game_hour_weather
        complete_game_reception_dataframe["wind_speed"] = game_wind_speed
        complete_game_reception_dataframe["wind_direction"] = game_wind_direction
        complete_game_reception_dataframe["relative_humidity"] = game_relative_humidity
        complete_game_reception_dataframe["precipitation"] = game_precipitation
        
        reception_records_by_game.append(complete_game_reception_dataframe)
        
end_time = datetime.now()
print(f"Elapsed Time: {end_time-start_time}")
# end

    
full_reception_dataframe = pd.concat(reception_records_by_game).reset_index(drop=True).copy().set_index("game_date")

columns_to_average = ['athlete_receptions', 'athlete_reception_yards','athlete_touchdowns',
                      'athlete_longest_reception', 'athlete_reception_attempts']

averaged_columns = ['avg_athlete_receptions', 'avg_athlete_reception_yards','avg_athlete_touchdowns',
                      'avg_athlete_longest_reception', 'avg_athlete_reception_attempts']

# each team plays 17 games per season, so try to take the average of the last 17 of the pair, if there aren't that many played of that pair, then settle for at least the last 5 games
rolling_averaged_stats = full_reception_dataframe.groupby(['team_quarterback', 'athlete_name'])[columns_to_average].rolling(window = 17, min_periods = 5, closed = "left").mean().reset_index()
rolling_averaged_stats = rolling_averaged_stats.join(rolling_averaged_stats[columns_to_average].add_prefix("avg_"))
rolling_averaged_stats = rolling_averaged_stats.drop(columns_to_average, axis = 1)

full_dataset_averaged = pd.merge(full_reception_dataframe, rolling_averaged_stats, on=['team_quarterback', 'athlete_name', 'game_date']).set_index("game_date")
full_dataset_averaged["year"] = full_dataset_averaged.index.year
full_dataset_averaged["month"] = full_dataset_averaged.index.month
full_dataset_averaged["day"] = full_dataset_averaged.index.day

full_dataset_averaged = full_dataset_averaged[["year","month","day",'team_quarterback', 'athlete_name',
                                               'game_venue', 'avg_athlete_receptions', 'avg_athlete_reception_yards',
                                               'avg_athlete_touchdowns','avg_athlete_longest_reception',
                                               'avg_athlete_reception_attempts', 'temp','wind_speed', 'wind_direction',
                                               'relative_humidity', 'precipitation','athlete_receptions',
                                               'athlete_reception_yards', 'athlete_touchdowns',
                                               'athlete_longest_reception', 'athlete_reception_attempts']]

transformed_dataset = pd.get_dummies(full_dataset_averaged)

historical_records = transformed_dataset[transformed_dataset.index.date < today.date()].dropna().reset_index(drop=True)
todays_records = transformed_dataset[transformed_dataset.index.date >= today.date()].dropna(subset = averaged_columns).reset_index(drop=True)
todays_original_records = full_dataset_averaged[full_dataset_averaged.index.date >= today.date()].dropna(subset = averaged_columns)

reception_features_to_drop = ['athlete_receptions','athlete_touchdowns',"athlete_reception_yards",
                                    'athlete_longest_reception','athlete_reception_attempts']

production_reception_data = todays_records.drop(reception_features_to_drop, axis = 1)

### reception yards prediction

x_reception_yards = historical_records.drop(reception_features_to_drop, axis = 1)
y_reception_yards = historical_records["athlete_reception_yards"].values

y_reception_yards_int = historical_records["athlete_reception_yards"].astype(int).values

Reception_Yards_XGBoost_Model = xgboost.XGBRegressor()
Reception_Yards_RandomForest_Model = RandomForestRegressor()
Reception_Yards_MLP_Model = MLPRegressor(max_iter = 2000)

Fitted_Reception_Yards_XGBoost_Model = Reception_Yards_XGBoost_Model.fit(X = x_reception_yards, y = y_reception_yards)
Fitted_Reception_Yards_RandomForest_Model = Reception_Yards_RandomForest_Model.fit(X = x_reception_yards, y = y_reception_yards)
Fitted_Reception_Yards_MLP_Model = Reception_Yards_MLP_Model.fit(X = x_reception_yards, y = y_reception_yards)

Reception_Yards_XGBoost_Prediction = Fitted_Reception_Yards_XGBoost_Model.predict(X = production_reception_data)
Reception_Yards_RandomForest_Prediction = Fitted_Reception_Yards_RandomForest_Model.predict(X = production_reception_data)
Reception_Yards_MLP_Prediction = Fitted_Reception_Yards_MLP_Model.predict(X = production_reception_data)

reception_yards_handicaps = []

for prop_game in todays_games_from_prop_odds["game_id"]:
    
    prop_odds = requests.get(f"https://api.prop-odds.com/beta/odds/{prop_game}/{reception_yards_market}?api_key={prop_odds_api_key}").json()
    prop_books = prop_odds["sportsbooks"]
    
    desired_book_odds = []
    
    for book in prop_books:
        
        if book["bookie_key"] == "draftkings":
            draftkings = book
            desired_book_odds.append(draftkings)
            break
        
    if len(desired_book_odds) < 1:
        
        available_book = prop_books[0]
        desired_book_odds.append(draftkings)
        
    book_odds = pd.json_normalize(desired_book_odds[0]["market"]["outcomes"]).sort_values(by = "timestamp", ascending = True)
    book_odds["name"] = book_odds["name"].str.strip("Over - ")
    book_odds["name"] = book_odds["name"].str.strip("Under - ")
    
    player_names = book_odds["name"].drop_duplicates().values
    
    for player in player_names:
        
        player_odds = book_odds[book_odds["name"] == player].sort_values(by = "timestamp", ascending = True).tail(1)
        player_name = player_odds["name"].iloc[0]
        player_handicap = player_odds["handicap"].iloc[0]
        
        handicap_dataframe = pd.DataFrame([{"athlete_name":player_name, "handicap":player_handicap}])
        reception_yards_handicaps.append(handicap_dataframe)
        
reception_yards_handicaps = pd.concat(reception_yards_handicaps)

Todays_Reception_Yards_Predictions = todays_original_records.copy()
Todays_Reception_Yards_Predictions["xgboost_reg_prediction"] = Reception_Yards_XGBoost_Prediction.astype(np.float64)
Todays_Reception_Yards_Predictions["random_forest_reg_prediction"] = Reception_Yards_RandomForest_Prediction
Todays_Reception_Yards_Predictions["mlp_reg_prediction"] = Reception_Yards_MLP_Prediction
Todays_Reception_Yards_Predictions["averaged_prediction"] = (Todays_Reception_Yards_Predictions["xgboost_reg_prediction"] + Todays_Reception_Yards_Predictions["random_forest_reg_prediction"] + Todays_Reception_Yards_Predictions["mlp_reg_prediction"]) / 3
Todays_Reception_Yards_Predictions = Todays_Reception_Yards_Predictions[["team_quarterback", "athlete_name", "xgboost_reg_prediction", "random_forest_reg_prediction","mlp_reg_prediction", "averaged_prediction"]]

Todays_Reception_Yards_Predictions = pd.merge(Todays_Reception_Yards_Predictions, reception_yards_handicaps, on = "athlete_name")
Todays_Reception_Yards_Predictions["edge"] = abs(Todays_Reception_Yards_Predictions["averaged_prediction"] - Todays_Reception_Yards_Predictions["handicap"])
Todays_Reception_Yards_Predictions["xg_edge"] = abs(Todays_Reception_Yards_Predictions["xgboost_reg_prediction"] - Todays_Reception_Yards_Predictions["handicap"])

###

### receptions prediction

x_receptions = historical_records.drop(reception_features_to_drop, axis = 1)
y_receptions = historical_records["athlete_receptions"].values

Receptions_XGBoost_Model = xgboost.XGBRegressor()
Receptions_RandomForest_Model = RandomForestRegressor()
Receptions_MLP_Model = MLPRegressor(max_iter = 2000)

Fitted_Receptions_XGBoost_Model = Receptions_XGBoost_Model.fit(X = x_receptions, y = y_receptions)
Fitted_Receptions_RandomForest_Model = Receptions_RandomForest_Model.fit(X = x_receptions, y = y_receptions)
Fitted_Receptions_MLP_Model = Receptions_MLP_Model.fit(X = x_receptions, y = y_receptions)

Receptions_XGBoost_Prediction = Fitted_Receptions_XGBoost_Model.predict(X = production_reception_data)
Receptions_RandomForest_Prediction = Fitted_Receptions_RandomForest_Model.predict(X = production_reception_data)
Receptions_MLP_Prediction = Fitted_Receptions_MLP_Model.predict(X = production_reception_data)

receptions_handicaps = []

for prop_game in todays_games_from_prop_odds["game_id"]:
    
    prop_odds = requests.get(f"https://api.prop-odds.com/beta/odds/{prop_game}/{receptions_market}?api_key={prop_odds_api_key}").json()
    prop_books = prop_odds["sportsbooks"]
    
    desired_book_odds = []
    
    for book in prop_books:
        
        if book["bookie_key"] == "draftkings":
            draftkings = book
            desired_book_odds.append(draftkings)
            break
        
    if len(desired_book_odds) < 1:
        
        available_book = prop_books[0]
        desired_book_odds.append(draftkings)
        
    book_odds = pd.json_normalize(desired_book_odds[0]["market"]["outcomes"]).sort_values(by = "timestamp", ascending = True)
    book_odds["name"] = book_odds["name"].str.strip("Over - ")
    book_odds["name"] = book_odds["name"].str.strip("Under - ")
    
    player_names = book_odds["name"].drop_duplicates().values
    
    for player in player_names:
        
        player_odds = book_odds[book_odds["name"] == player].sort_values(by = "timestamp", ascending = True).tail(1)
        player_name = player_odds["name"].iloc[0]
        player_handicap = player_odds["handicap"].iloc[0]
        
        handicap_dataframe = pd.DataFrame([{"athlete_name":player_name, "handicap":player_handicap}])
        receptions_handicaps.append(handicap_dataframe)
        
receptions_handicaps = pd.concat(receptions_handicaps)

Todays_Receptions_Predictions = todays_original_records.copy()
Todays_Receptions_Predictions["xgboost_reg_prediction"] = Receptions_XGBoost_Prediction.astype(np.float64)
Todays_Receptions_Predictions["random_forest_reg_prediction"] = Receptions_RandomForest_Prediction
Todays_Receptions_Predictions["mlp_reg_prediction"] = Receptions_MLP_Prediction
Todays_Receptions_Predictions["averaged_prediction"] = (Todays_Receptions_Predictions["xgboost_reg_prediction"] + Todays_Receptions_Predictions["random_forest_reg_prediction"] + Todays_Receptions_Predictions["mlp_reg_prediction"]) / 3
Todays_Receptions_Predictions = Todays_Receptions_Predictions[["team_quarterback", "athlete_name", "xgboost_reg_prediction", "random_forest_reg_prediction","mlp_reg_prediction", "averaged_prediction"]]

Todays_Receptions_Predictions = pd.merge(Todays_Receptions_Predictions, receptions_handicaps, on = "athlete_name")
Todays_Receptions_Predictions["edge"] = abs(Todays_Receptions_Predictions["averaged_prediction"] - Todays_Receptions_Predictions["handicap"])
Todays_Receptions_Predictions["xg_edge"] = abs(Todays_Receptions_Predictions["xgboost_reg_prediction"] - Todays_Receptions_Predictions["handicap"])

###

### receptions touchdown prediction

x_reception_touchdowns = historical_records.drop(reception_features_to_drop, axis = 1)
y_reception_touchdowns = historical_records["athlete_touchdowns"].values

Receptions_Touchdowns_XGBoost_Model = xgboost.XGBRegressor()
Receptions_Touchdowns_RandomForest_Model = RandomForestRegressor()
Receptions_Touchdowns_MLP_Model = MLPRegressor(max_iter = 2000)

Fitted_Receptions_Touchdowns_XGBoost_Model = Receptions_Touchdowns_XGBoost_Model.fit(X = x_reception_touchdowns, y = y_reception_touchdowns)
Fitted_Receptions_Touchdowns_RandomForest_Model = Receptions_Touchdowns_RandomForest_Model.fit(X = x_reception_touchdowns, y = y_reception_touchdowns)
Fitted_Receptions_Touchdowns_MLP_Model = Receptions_Touchdowns_MLP_Model.fit(X = x_reception_touchdowns, y = y_reception_touchdowns)

Receptions_Touchdowns_XGBoost_Prediction = Fitted_Receptions_Touchdowns_XGBoost_Model.predict(X = production_reception_data)
Receptions_Touchdowns_RandomForest_Prediction = Fitted_Receptions_Touchdowns_RandomForest_Model.predict(X = production_reception_data)
Receptions_Touchdowns_MLP_Prediction = Fitted_Receptions_Touchdowns_MLP_Model.predict(X = production_reception_data)

Todays_Receptions_Touchdowns_Predictions = todays_original_records.copy()
Todays_Receptions_Touchdowns_Predictions["xgboost_reg_prediction"] = Receptions_Touchdowns_XGBoost_Prediction.astype(np.float64)
Todays_Receptions_Touchdowns_Predictions["random_forest_reg_prediction"] = Receptions_Touchdowns_RandomForest_Prediction
Todays_Receptions_Touchdowns_Predictions["mlp_reg_prediction"] = Receptions_Touchdowns_MLP_Prediction
Todays_Receptions_Touchdowns_Predictions["averaged_prediction"] = (Todays_Receptions_Touchdowns_Predictions["xgboost_reg_prediction"] + Todays_Receptions_Touchdowns_Predictions["random_forest_reg_prediction"] + Todays_Receptions_Touchdowns_Predictions["mlp_reg_prediction"]) / 3

Todays_Receptions_Touchdowns_Predictions = Todays_Receptions_Touchdowns_Predictions[["team_quarterback", "athlete_name", "xgboost_reg_prediction", "random_forest_reg_prediction","mlp_reg_prediction", "averaged_prediction"]]

###

### longest reception prediction

x_reception_longest = historical_records.drop(reception_features_to_drop, axis = 1)
y_reception_longest = historical_records["athlete_longest_reception"].values

Receptions_Longest_XGBoost_Model = xgboost.XGBRegressor()
Receptions_Longest_RandomForest_Model = RandomForestRegressor()
Receptions_Longest_MLP_Model = MLPRegressor(max_iter = 2000)

Fitted_Receptions_Longest_XGBoost_Model = Receptions_Longest_XGBoost_Model.fit(X = x_reception_longest, y = y_reception_longest)
Fitted_Receptions_Longest_RandomForest_Model = Receptions_Longest_RandomForest_Model.fit(X = x_reception_longest, y = y_reception_longest)
Fitted_Receptions_Longest_MLP_Model = Receptions_Longest_MLP_Model.fit(X = x_reception_longest, y = y_reception_longest)

Receptions_Longest_XGBoost_Prediction = Fitted_Receptions_Longest_XGBoost_Model.predict(X = production_reception_data)
Receptions_Longest_RandomForest_Prediction = Fitted_Receptions_Longest_RandomForest_Model.predict(X = production_reception_data)
Receptions_Longest_MLP_Prediction = Fitted_Receptions_Longest_MLP_Model.predict(X = production_reception_data)

longest_reception_handicaps = []

for prop_game in todays_games_from_prop_odds["game_id"]:
    
    prop_odds = requests.get(f"https://api.prop-odds.com/beta/odds/{prop_game}/{longest_reception_market}?api_key={prop_odds_api_key}").json()
    prop_books = prop_odds["sportsbooks"]
    
    desired_book_odds = []
    
    for book in prop_books:
        
        if book["bookie_key"] == "draftkings":
            draftkings = book
            desired_book_odds.append(draftkings)
            break
        
    if len(desired_book_odds) < 1:
        
        available_book = prop_books[0]
        desired_book_odds.append(draftkings)
        
    book_odds = pd.json_normalize(desired_book_odds[0]["market"]["outcomes"]).sort_values(by = "timestamp", ascending = True)
    book_odds["name"] = book_odds["name"].str.strip("Over - ")
    book_odds["name"] = book_odds["name"].str.strip("Under - ")
    
    player_names = book_odds["name"].drop_duplicates().values
    
    for player in player_names:
        
        player_odds = book_odds[book_odds["name"] == player].sort_values(by = "timestamp", ascending = True).tail(1)
        player_name = player_odds["name"].iloc[0]
        player_handicap = player_odds["handicap"].iloc[0]
        
        handicap_dataframe = pd.DataFrame([{"athlete_name":player_name, "handicap":player_handicap}])
        longest_reception_handicaps.append(handicap_dataframe)
        
longest_reception_handicaps = pd.concat(longest_reception_handicaps)


Todays_Receptions_Longest_Predictions = todays_original_records.copy()
Todays_Receptions_Longest_Predictions["xgboost_reg_prediction"] = Receptions_Longest_XGBoost_Prediction.astype(np.float64)
Todays_Receptions_Longest_Predictions["random_forest_reg_prediction"] = Receptions_Longest_RandomForest_Prediction
Todays_Receptions_Longest_Predictions["mlp_reg_prediction"] = Receptions_Longest_MLP_Prediction
Todays_Receptions_Longest_Predictions["averaged_prediction"] = (Todays_Receptions_Longest_Predictions["xgboost_reg_prediction"] + Todays_Receptions_Longest_Predictions["random_forest_reg_prediction"] + Todays_Receptions_Longest_Predictions["mlp_reg_prediction"]) / 3

Todays_Receptions_Longest_Predictions = Todays_Receptions_Longest_Predictions[["team_quarterback", "athlete_name", "xgboost_reg_prediction", "random_forest_reg_prediction","mlp_reg_prediction", "averaged_prediction"]]

Todays_Receptions_Longest_Predictions = pd.merge(Todays_Receptions_Longest_Predictions, longest_reception_handicaps, on = "athlete_name")
Todays_Receptions_Longest_Predictions["edge"] = abs(Todays_Receptions_Longest_Predictions["averaged_prediction"] - Todays_Receptions_Longest_Predictions["handicap"])
Todays_Receptions_Longest_Predictions["xg_edge"] = abs(Todays_Receptions_Longest_Predictions["xgboost_reg_prediction"] - Todays_Receptions_Longest_Predictions["handicap"])

###
Todays_Reception_Yards_Predictions["market"] = "reception_yards"
Todays_Receptions_Predictions["market"] = "receptions"
Todays_Receptions_Touchdowns_Predictions["market"] = "reception_touchdowns"
Todays_Receptions_Longest_Predictions["market"] = "reception_longest"

All_Reception_Predictions = pd.concat([Todays_Reception_Yards_Predictions, Todays_Receptions_Predictions, Todays_Receptions_Touchdowns_Predictions, Todays_Receptions_Longest_Predictions], axis = 0).reset_index(drop=True)
