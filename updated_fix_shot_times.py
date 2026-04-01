import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from scipy.spatial.distance import euclidean


def sg_filter(x, m, k=0):
    mid = len(x) // 2
    a = x - x[mid]
    expa = [a**i for i in range(0, m + 1)]
    A = np.array(expa).T
    Ai = np.linalg.pinv(A)
    return Ai[k]


def smooth(x, y, size=5, order=2, deriv=0):
    if deriv > order:
        raise Exception("deriv must be <= order")
    n = len(x)
    m = size

    result = np.zeros(n)

    for i in range(m, n - m):
        start, end = i - m, i + m + 1
        f = sg_filter(x[start:end], order, deriv)
        result[i] = np.dot(f, y[start:end])

    if deriv > 1:
        result *= math.factorial(deriv)

    return result

##LEGACY VERSION that finds local max of ball height to determine shot time and location without constricting time window around shot
def correct_shots_legacy(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            original_time = shot["MINUTES_REMAINING"] * 60 + shot["SECONDS_REMAINING"]
            x_loc_original = shot["LOC_X"]
            y_loc_original = shot["LOC_Y"]
            movement_around_shot = movement.loc[
                movement["event_id"].isin([event_id, event_id - 1])
            ].drop_duplicates(subset=["game_clock"])

            game_clock_time = movement_around_shot.query("team_id == -1")[
                "game_clock"
            ].values
            ball_height = movement_around_shot.query("team_id == -1")["radius"].values

            size = 10
            order = 3

            params = (game_clock_time, ball_height, size, order)

            position_smoothed = smooth(*params, deriv=0)
            acceleration_smoothed = smooth(*params, deriv=2)
            max_ind = np.argmax(position_smoothed)

            shot_window = acceleration_smoothed[max(0, max_ind - 25) : max_ind]
            shot_min_ind = np.argmin(shot_window)
            shot_ind = max_ind - shot_min_ind
            shot_time = game_clock_time[shot_ind]

            quarter = movement_around_shot["quarter"].values[0]
            movement_around_shot = movement_around_shot.query(
                "game_clock == @shot_time"
            )

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = shot_time
            shot["LOC_X"] = movement_around_shot.query("team_id == -1")["x_loc"].values[
                0
            ]
            shot["LOC_Y"] = movement_around_shot.query("team_id == -1")["y_loc"].values[
                0
            ]
            shot["ORIGINAL_TIME"] = original_time
            shot["LOC_X_ORIGINAL"] = x_loc_original
            shot["LOC_Y_ORIGINAL"] = y_loc_original
            shot["DIST_FROM_ORIGINAL"] = euclidean(
                (shot["LOC_X"], shot["LOC_Y"]), (x_loc_original, y_loc_original)
            )

        except Exception:
            print(f"Legacy Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            continue
        

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots


##Version that constricts time window around shot and finds local max of ball height to determine shot time and location
def correct_shotsv1(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            original_time = shot["MINUTES_REMAINING"] * 60 + shot["SECONDS_REMAINING"]
            x_loc_original = shot["LOC_X"]
            y_loc_original = shot["LOC_Y"]
            movement_around_shot = movement.loc[
                (movement["event_id"].isin([event_id, event_id - 1]))
                & (movement["game_clock"] <= original_time + 6)
                & (movement["game_clock"] >= original_time)
            ].drop_duplicates(subset=["game_clock"])

            game_clock_time = movement_around_shot.query("team_id == -1")[
                "game_clock"
            ].values
            ball_height = movement_around_shot.query("team_id == -1")["radius"].values

            size = 10
            order = 3

            params = (game_clock_time, ball_height, size, order)

            position_smoothed = smooth(*params, deriv=0)
            acceleration_smoothed = smooth(*params, deriv=2)
            max_ind = np.argmax(position_smoothed)

            shot_window = acceleration_smoothed[max(0, max_ind - 25) : max_ind]
            shot_min_ind = np.argmin(shot_window)
            shot_ind = max_ind - shot_min_ind
            shot_time = game_clock_time[shot_ind]

            quarter = movement_around_shot["quarter"].values[0]
            movement_around_shot = movement_around_shot.query(
                "game_clock == @shot_time"
            )

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = shot_time
            shot["LOC_X"] = movement_around_shot.query("team_id == -1")["x_loc"].values[
                0
            ]
            shot["LOC_Y"] = movement_around_shot.query("team_id == -1")["y_loc"].values[
                0
            ]
            shot["ORIGINAL_TIME"] = original_time
            shot["LOC_X_ORIGINAL"] = x_loc_original
            shot["LOC_Y_ORIGINAL"] = y_loc_original
            shot["DIST_FROM_ORIGINAL"] = euclidean(
                (shot["LOC_X"], shot["LOC_Y"]), (x_loc_original, y_loc_original)
            )

        except Exception:
            print(f"V1 Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            continue

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots


##Version that constrict time window based on distance from ball to original shot location and finds local max of ball height to determine shot time and location
def correct_shotsv2(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            original_time = shot["MINUTES_REMAINING"] * 60 + shot["SECONDS_REMAINING"]
            x_loc_original = shot["LOC_X"]
            y_loc_original = shot["LOC_Y"]
            movement_around_shot = movement.loc[
                movement["event_id"].isin([event_id, event_id - 1])
            ].drop_duplicates(subset=["game_clock"])

            ##Calculate distance from ball to original shot location
            movement_around_shot["ball_dist_from_shot"] = movement_around_shot.apply(
                lambda row: euclidean(
                    (row["x_loc"], row["y_loc"]),
                    (x_loc_original, y_loc_original),
                )
                if row["team_id"] == -1
                else np.nan,
                axis=1,
            )
            min_dist_ind = movement_around_shot["ball_dist_from_shot"].idxmin()
            min_dist_time = movement_around_shot.loc[min_dist_ind, "game_clock"]
            movement_around_shot = movement_around_shot.loc[
                (movement_around_shot["game_clock"] <= min_dist_time + 3)
                & (movement_around_shot["game_clock"] >= min_dist_time - 1)
            ].drop_duplicates(subset=["game_clock"])

            quarter = movement_around_shot["quarter"].values[0]

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = min_dist_time
            shot["LOC_X"] = movement_around_shot.loc[min_dist_ind, "x_loc"]
            shot["LOC_Y"] = movement_around_shot.loc[min_dist_ind, "y_loc"]

            shot["ORIGINAL_TIME"] = original_time
            shot["LOC_X_ORIGINAL"] = x_loc_original
            shot["LOC_Y_ORIGINAL"] = y_loc_original
            shot["DIST_FROM_ORIGINAL"] = euclidean(
                (shot["LOC_X"], shot["LOC_Y"]), (x_loc_original, y_loc_original)
            )

        except Exception:
            print(f"V2 Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            continue
            continue

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots

##Combined version that constrict time window based on distance from ball to original shot location and 
#  finds local max of ball height to determine shot time and location, but also includes original time and location for comparison
def correct_shotsv3(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            original_time = shot["MINUTES_REMAINING"] * 60 + shot["SECONDS_REMAINING"]
            x_loc_original = shot["LOC_X"]
            y_loc_original = shot["LOC_Y"]
            movement_around_shot = movement.loc[
                (movement["event_id"].isin([event_id, event_id - 1]))
                & (movement["game_clock"] <= original_time + 6)
                & (movement["game_clock"] >= original_time - 2)
            ].drop_duplicates(subset=["game_clock"])

            ##Calculate distance from ball to original shot location
            movement_around_shot["ball_dist_from_shot"] = movement_around_shot.apply(
                lambda row: euclidean(
                    (row["x_loc"], row["y_loc"]),
                    (x_loc_original, y_loc_original),
                )
                if row["team_id"] == -1
                else np.nan,
                axis=1,
            )
            min_dist_ind = movement_around_shot.loc[movement_around_shot["game_clock"] <= original_time+6]["ball_dist_from_shot"].idxmin()
            min_dist_time = movement_around_shot.loc[min_dist_ind, "game_clock"]
            movement_around_shot = movement_around_shot.loc[
                (movement_around_shot["game_clock"] <= min_dist_time + 3)
                & (movement_around_shot["game_clock"] >= min_dist_time - 1)
            ].drop_duplicates(subset=["game_clock"])

            game_clock_time = movement_around_shot.query("team_id == -1")[
                "game_clock"
            ].values
            ball_height = movement_around_shot.query("team_id == -1")["radius"].values

            size = 10
            order = 3

            params = (game_clock_time, ball_height, size, order)

            position_smoothed = smooth(*params, deriv=0)
            acceleration_smoothed = smooth(*params, deriv=2)
            max_ind = np.argmax(position_smoothed)

            shot_window = acceleration_smoothed[max(0, max_ind - 25) : max_ind]
            shot_min_ind = np.argmin(shot_window)
            shot_ind = max_ind - shot_min_ind
            shot_time = game_clock_time[shot_ind]

            quarter = movement_around_shot["quarter"].values[0]
            movement_around_shot = movement_around_shot.query(
                "game_clock == @shot_time"
            )

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = shot_time
            shot["LOC_X"] = movement_around_shot.query("team_id == -1")["x_loc"].values[
                0
            ]
            shot["LOC_Y"] = movement_around_shot.query("team_id == -1")["y_loc"].values[
                0
            ]

            shot["ORIGINAL_TIME"] = original_time
            shot["LOC_X_ORIGINAL"] = x_loc_original
            shot["LOC_Y_ORIGINAL"] = y_loc_original
            shot["DIST_FROM_ORIGINAL"] = euclidean(
                (shot["LOC_X"], shot["LOC_Y"]), (x_loc_original, y_loc_original)
            )

        except Exception:
            print(f"V3 Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            continue

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots

def correct_shotsv4(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            original_time = shot["MINUTES_REMAINING"] * 60 + shot["SECONDS_REMAINING"]
            x_loc_original = shot["LOC_X"]
            y_loc_original = shot["LOC_Y"]
            movement_around_shot = movement.loc[
                (movement["event_id"].isin([event_id, event_id - 1]))
                & (movement["game_clock"] <= original_time + 6)
                & (movement["game_clock"] >= original_time - 2)
            ].drop_duplicates(subset=["game_clock"])

            ##Calculate distance from ball to original shot location
            movement_around_shot["ball_dist_from_shot"] = movement_around_shot.apply(
                lambda row: euclidean(
                    (row["x_loc"], row["y_loc"]),
                    (x_loc_original, y_loc_original),
                )
                if row["team_id"] == -1
                else np.nan,
                axis=1,
            )
            min_dist_ind = movement_around_shot.loc[movement_around_shot["game_clock"] <= original_time+6]["ball_dist_from_shot"].idxmin()
            min_dist_time = movement_around_shot.loc[min_dist_ind, "game_clock"]
            movement_around_shot = movement_around_shot.loc[
                (movement_around_shot["game_clock"] <= min_dist_time + 3)
                & (movement_around_shot["game_clock"] >= min_dist_time - 1)
            ].drop_duplicates(subset=["game_clock"])

            game_clock_time = movement_around_shot.query("team_id == -1")[
                "game_clock"
            ].values
            ball_height = movement_around_shot.query("team_id == -1")["radius"].values

            size = 10
            order = 3

            params = (game_clock_time, ball_height, size, order)

            position_smoothed = smooth(*params, deriv=0)
            acceleration_smoothed = smooth(*params, deriv=2)
            max_ind = np.argmax(position_smoothed)

            shot_window = acceleration_smoothed[max(0, max_ind - 25) : max_ind]
            shot_min_ind = np.argmin(shot_window)
            shot_ind = max_ind - shot_min_ind
            shot_time = game_clock_time[shot_ind]

            quarter = movement_around_shot["quarter"].values[0]
            movement_around_shot = movement_around_shot.query(
                "game_clock == @shot_time"
            )

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = shot_time
            shot["LOC_X"] = movement_around_shot.query("team_id == -1")["x_loc"].values[
                0
            ]
            shot["LOC_Y"] = movement_around_shot.query("team_id == -1")["y_loc"].values[
                0
            ]

            shot["ORIGINAL_TIME"] = original_time
            shot["LOC_X_ORIGINAL"] = x_loc_original
            shot["LOC_Y_ORIGINAL"] = y_loc_original
            shot["DIST_FROM_ORIGINAL"] = euclidean(
                (shot["LOC_X"], shot["LOC_Y"]), (x_loc_original, y_loc_original)
            )

        except Exception:
            print(f"V4 Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            try:
                correct_shots_legacy(game_shots, movement, events)
            except Exception:
                print(f"Legacy Error processing shot with event_id {shot['GAME_EVENT_ID']}")
            continue

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots
