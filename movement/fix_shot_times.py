import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from movement import utils
import movement.config as CONFIG


def load_shots():
    shots = pd.read_csv(f"{CONFIG.data.shots.dir}/shots.csv")
    shots["EVENTTIME"] = utils.convert_time(
        minutes=shots["MINUTES_REMAINING"].values, seconds=shots["SECONDS_REMAINING"].values
    )
    shots["GAME_ID"] = "00" + shots["GAME_ID"].astype(int).astype(str)
    return shots


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


def correct_shots(game_shots, movement, events):
    fixed_shots = pd.DataFrame(columns=game_shots.columns)

    for ind, shot in game_shots.iterrows():
        try:
            event_id = shot["GAME_EVENT_ID"]
            movement_around_shot = movement.loc[movement["event_id"].isin([event_id, event_id - 1])].drop_duplicates(
                subset=["game_clock"]
            )

            game_clock_time = movement_around_shot.query("team_id == -1")["game_clock"].values
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
            movement_around_shot = movement_around_shot.query("game_clock == @shot_time")

            shot["QUARTER"] = quarter
            shot["SHOT_TIME"] = shot_time
            shot["LOC_X"] = movement_around_shot.query("team_id == -1")["x_loc"].values[0]
            shot["LOC_Y"] = movement_around_shot.query("team_id == -1")["y_loc"].values[0]
        except Exception:
            continue

        # fixed_shots = fixed_shots.append(shot)
        fixed_shots = pd.concat([fixed_shots, pd.DataFrame([shot])], ignore_index=True)

    return fixed_shots


if __name__ == "__main__":
    games = utils.get_games()
    events = utils.get_events(CONFIG.data.events.dir, games)
    shots = load_shots()
    fixed_shots = pd.DataFrame(columns=shots.columns)

    for game in tqdm(games):
        try:
            game_movement = pd.read_csv(f"{CONFIG.data.movement.converted.dir}/{game}_converted.csv")
            game_shots = shots.query("GAME_ID == @game")
            game_events = events.query("GAME_ID == @game")
        except FileNotFoundError:
            continue

        fixed_shots = pd.concat([fixed_shots, correct_shots(game_shots, game_movement, game_events)], ignore_index=True)
        # fixed_shots = fixed_shots.append(correct_shots(game_shots, game_movement, game_events))

    fixed_shots.to_csv(f"{CONFIG.data.shots.dir}/shots_fixed.csv", index=False)
