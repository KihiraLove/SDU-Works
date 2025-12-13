#!/usr/bin/env python3
"""
TempleOSRS batch fetch + CSV formatter in one script.

Usage:
    python temple_combined.py usernames.txt output.csv

Defaults:
    input:  usernames.txt
    output: temple_stats_formatted.csv

Input file: one username per line (names may contain spaces).

For each username:
  - Calls add_datapoint.php to update data.
  - Calls player_info.php?player=<name> to get Game mode & GIM.
  - Calls player_stats.php?player=<name>&duration=alltime to get EHP/EHB stats.

Output CSV columns:
  Username,
  Gamemode,
  EHP,
  IM EHP,
  LVL3 EHP,
  UIM EHP,
  1 DEF EHP,
  GIM EHP,
  EHB,
  IM EHB,
  UIM EHB,
  1 DEF EHB

Gamemode rules:
- If GIM != 0:
    * First digit: 1 = Regular, 2 = Hardcore
    * Second digit: team size (2..5)
    * Example: 12 -> "Regular GIM 2-player", 25 -> "Hardcore GIM 5-player"
- If GIM == 0:
    * game_mode: 0 = Main, 1 = IM, 2 = UIM, 3 = HCIM
"""

import csv
import sys
import time
from typing import Optional, Tuple, Dict, Any

import requests
UPDATE = False

ADD_DATAPOINT_URL = "https://templeosrs.com/php/add_datapoint.php"
PLAYER_INFO_URL = "https://templeosrs.com/api/player_info.php"
PLAYER_STATS_URL = "https://templeosrs.com/api/player_stats.php"

REQUEST_TIMEOUT = 10  # seconds
DELAY_BETWEEN_USERS = 1.0  # seconds
RATE_LIMIT_SLEEP = 60  # seconds when HTTP 429 is returned
MAX_429_RETRIES = 5    # maximum number of retries on HTTP 429


def read_usernames(path: str) -> list[str]:
    usernames: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            usernames.append(name)
    return usernames


def get_with_429_retry(url: str, params: Dict[str, Any], timeout: int) -> requests.Response:
    """
    Perform a GET request, and if HTTP 429 is returned, wait RATE_LIMIT_SLEEP
    seconds and retry, up to MAX_429_RETRIES times.
    """
    attempts = 0
    while True:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and attempts < MAX_429_RETRIES:
            attempts += 1
            print(
                f"[WARN] HTTP 429 (rate limited) from {url} for params {params}. "
                f"Sleeping {RATE_LIMIT_SLEEP} seconds before retry {attempts}/{MAX_429_RETRIES}...",
                file=sys.stderr,
            )
            time.sleep(RATE_LIMIT_SLEEP)
            continue
        return resp


def fetch_player_info(username: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Fetch Game mode and GIM from the Player Information endpoint.
    Game mode: 0 = Main, 1 = IM, 2 = UIM, 3 = HCIM
    GIM: 0, 12, 13, 14, 15, 22, 23, 24, 25
    Includes bosses=1 param as requested.
    """
    try:
        resp = get_with_429_retry(
            PLAYER_INFO_URL,
            params={"player": username},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[INFO] Failed to fetch player_info for '{username}': {e}", file=sys.stderr)
        return None, None

    if isinstance(raw, dict):
        info = raw.get("data", raw)
    else:
        return None, None

    game_mode = info.get("Game mode")
    gim = info.get("GIM")

    try:
        if game_mode is not None:
            game_mode = int(game_mode)
    except (TypeError, ValueError):
        game_mode = None

    try:
        if gim is not None:
            gim = int(gim)
    except (TypeError, ValueError):
        gim = None

    return game_mode, gim


def fetch_player_stats(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stats from Player Stats endpoint, with duration=alltime.
    Expected keys for EHP:
        Ehp, Im_ehp, Lvl3_ehp, Uim_ehp, 1def_ehp, Gim_ehp
    Expected keys for EHB:
        Ehb, Im_ehb, Uim_ehb, 1def_ehb
    """
    try:
        resp = get_with_429_retry(
            PLAYER_STATS_URL,
            params={"player": username, "duration": "alltime", "bosses": "1"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[INFO] Failed to fetch player_stats for '{username}': {e}", file=sys.stderr)
        return None

    if not isinstance(raw, dict):
        return None

    data = raw.get("data")
    if not isinstance(data, dict):
        return None

    return data


def round_metric(stats: Optional[Dict[str, Any]], key: str) -> str:
    """
    Extract a numeric metric from stats and round to the nearest integer.
    Returns an empty string if missing or invalid.
    """
    if stats is None:
        return ""

    value = stats.get(key)
    if value is None:
        return ""

    try:
        rounded = int(round(float(value)))
        return str(rounded)
    except (TypeError, ValueError):
        return ""


def trigger_datapoint_update(username: str) -> None:
    """
    Call the add_datapoint endpoint to update the player's data.
    Errors are logged but do not stop processing.
    """
    try:
        resp = get_with_429_retry(
            ADD_DATAPOINT_URL,
            params={"player": username},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            print(
                f"[INFO] add_datapoint for '{username}' "
                f"returned HTTP {resp.status_code}",
                file=sys.stderr,
            )
    except requests.RequestException as e:
        print(f"[INFO] add_datapoint failed for '{username}': {e}", file=sys.stderr)


def derive_gamemode(game_mode_raw: Optional[int], gim_raw: Optional[int]) -> str:
    """
    Return a human-readable gamemode string based on gim and game_mode.
    """
    gim_val = gim_raw or 0

    # Prefer GIM if non-zero
    if gim_val != 0:
        code = str(gim_val)
        kind = "GIM"
        team_desc: Optional[str] = None

        if len(code) >= 2:
            type_digit = code[0]
            size_digit = code[1]

            if type_digit == "1":
                kind = "Regular GIM"
            elif type_digit == "2":
                kind = "Hardcore GIM"
            else:
                kind = "GIM"

            if size_digit.isdigit():
                team_desc = f"{int(size_digit)}-player"

        if team_desc:
            return f"{kind} {team_desc}"
        return kind

    # Fall back to non-GIM game_mode
    gm_val = game_mode_raw if isinstance(game_mode_raw, int) else -1

    mapping = {
        0: "Main",
        1: "IM",
        2: "UIM",
        3: "HCIM",
    }
    return mapping.get(gm_val, "Unknown")


def calculate_special_ehp(im_ehp, lvl3_ehp, uim_ehp, one_def_ehp, gim_ehp):
    if lvl3_ehp != "0":
        return lvl3_ehp
    if one_def_ehp != "0":
        return one_def_ehp
    if uim_ehp != "0":
        return uim_ehp
    if gim_ehp != "0":
        return gim_ehp
    if im_ehp != "0":
        return im_ehp
    return "0"


def calculate_special_ehb(im_ehb, uim_ehb, one_def_ehb):
    if one_def_ehb != "0":
        return one_def_ehb
    if uim_ehb != "0":
        return uim_ehb
    if im_ehb != "0":
        return im_ehb
    return "0"


def main(input_path: str, output_path: str) -> None:
    usernames = read_usernames(input_path)
    if not usernames:
        print("No usernames found in input file.", file=sys.stderr)
        return

    print(f"Processing {len(usernames)} usernames...", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)

        # Final CSV header
        writer.writerow(
            [
                "Username",
                "Gamemode",
                "EHP",
                "Special EHP",
                "EHB",
                "Special EHB",
            ]
        )

        for idx, username in enumerate(usernames, start=1):
            print(f"[{idx}/{len(usernames)}] {username}", file=sys.stderr)

            # Update datapoint
            if UPDATE:
                trigger_datapoint_update(username)

            # Fetch player info (game_mode, gim)
            game_mode, gim = fetch_player_info(username)

            # Fetch stats (EHP/EHB)
            stats = fetch_player_stats(username)

            # EHP metrics
            ehp = round_metric(stats, "Ehp")
            im_ehp = round_metric(stats, "Im_ehp")
            lvl3_ehp = round_metric(stats, "Lvl3_ehp")
            uim_ehp = round_metric(stats, "Uim_ehp")
            one_def_ehp = round_metric(stats, "1def_ehp")
            gim_ehp = round_metric(stats, "Gim_ehp")
            special_ehp = calculate_special_ehp(im_ehp, lvl3_ehp, uim_ehp, one_def_ehp, gim_ehp)
            print(f"ehp: {ehp}, im_ehp: {im_ehp}, lvl3_ehp: {lvl3_ehp}, uim_ehp: {uim_ehp}, one_def_ehp: {one_def_ehp}, gim_ehp: {gim_ehp}, special_ehp: {special_ehp}")

            # EHB metrics
            ehb = round_metric(stats, "Ehb")
            im_ehb = round_metric(stats, "Im_ehb")
            uim_ehb = round_metric(stats, "Uim_ehb")
            one_def_ehb = round_metric(stats, "1def_ehb")
            special_ehb = calculate_special_ehb(im_ehb, uim_ehb, one_def_ehb)
            print(f"ehb: {ehb}, im_ehb: {im_ehb}, uim_ehb: {uim_ehb}, one_def_ehb: {one_def_ehb}, special_ehb: {special_ehb}")

            gamemode_str = derive_gamemode(game_mode, gim)

            writer.writerow(
                [
                    username,
                    gamemode_str,
                    ehp,
                    special_ehp,
                    ehb,
                    special_ehb,
                ]
            )

            # Respect delay between users
            time.sleep(DELAY_BETWEEN_USERS)

    print(f"Done. Results written to '{output_path}'.", file=sys.stderr)


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "usernames.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "temple_stats_formatted.csv"
    main(in_path, out_path)
