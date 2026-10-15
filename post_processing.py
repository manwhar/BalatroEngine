"""
Methods used to continue modifying and evaluating data after the
initial first-pass producing the json mapping frames to button
presses.
"""

import json
from pprint import pp as pprint

from config import configure_logging, get_config


def remove_duplicates(d: dict[str, int]) -> dict:
    new_dict: dict[str, int] = {}
    prev_action = None
    for frame, action in reversed(d.items()):
        if prev_action != action:
            new_dict[frame] = action
        prev_action = action

    return dict(reversed(new_dict.items()))

def get_relevant_frames(d: dict[str, dict[str, int]]) -> tuple[list[str], list[str]]:
    key = d["key"]
    frames = d["frames"]

if __name__ == "__main__":
    configure_logging()

    config = get_config()

    filename = "bu_full_1791436070.json"
    path = rf"{config.output_dir}\{filename}"

    with open(path) as f:
        data = json.load(f)
        f.close()

    cleaned = remove_duplicates(data["frames"])

    pprint(cleaned)
