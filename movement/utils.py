import pandas as pd
from os import listdir
from os.path import isfile, join, exists
import re

import movement.config as CONFIG


def convert_time(minutes, seconds):
    return (pd.Series(minutes).astype(int) * 60 + pd.Series(seconds).astype(int)).tolist()


def get_events(event_dir, games):
    all_events = []

    for game in games:
        game_events = pd.read_csv(f"{event_dir}/{game}.csv")
        all_events.append(game_events)

    events = pd.concat(all_events, ignore_index=True)
    events["GAME_ID"] = "00" + events["GAME_ID"].astype(int).astype(str)

    return events


def get_games():
    potential_game_dirs = [CONFIG.data.movement.dir]
    return _get_game_names(potential_game_dirs)


def get_converted_games():
    potential_game_dirs = [CONFIG.data.movement.converted.dir]
    return _get_game_names(potential_game_dirs)


def _get_game_names(potential_game_dirs):
    names = []
    for path in potential_game_dirs:
        if exists(path):
            for f in listdir(path):
                if isfile(join(path, f)):
                    m = re.match(r"\d+", f)
                    if m:
                        names.append(m.string[m.start() : m.end()])

    if names:
        return names

    print("No games found.")
    return names
