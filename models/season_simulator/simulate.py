"""
Season Simulator — Monte Carlo con distribución de Poisson.

Carga ratings base de club_ratings.json, aplica ajustes por fichajes
hipotéticos y simula 1000 temporadas para calcular tabla proyectada
y delta de probabilidad de título.
"""

import os
import json
import copy
import numpy as np
from typing import Any

RATINGS_PATH = os.path.join("models", "season_simulator", "club_ratings.json")
ITERATIONS = 1000
POSITION_ATTACK_WEIGHT = {
    "Forward": 0.06,
    "Midfielder": 0.03,
    "Defender": 0.01,
    "Goalkeeper": 0.0,
}
POSITION_DEFENSE_WEIGHT = {
    "Forward": 0.005,
    "Midfielder": 0.015,
    "Defender": 0.04,
    "Goalkeeper": 0.05,
}


def _load_ratings() -> dict:
    with open(RATINGS_PATH) as f:
        return json.load(f)


def _club_name_by_id(ratings: dict, club_id: int) -> str | None:
    for name, data in ratings["clubs"].items():
        if data["id"] == club_id:
            return name
    return None


def _apply_transfers(ratings: dict, transfers: list[dict]) -> dict:
    """
    Ajusta ratings por cada fichaje hipotético.
    El club que pierde al jugador baja, el que lo gana sube.
    """
    adjusted = copy.deepcopy(ratings)

    for t in transfers:
        from_name = _club_name_by_id(adjusted, t["from_club_id"])
        to_name = _club_name_by_id(adjusted, t["to_club_id"])
        position = t["player_stats"].get("position", "Midfielder")
        goals_per90 = t["player_stats"].get("goals_per90", 0)
        assists_per90 = t["player_stats"].get("assists_per90", 0)

        atk_boost = POSITION_ATTACK_WEIGHT.get(position, 0.02) + goals_per90 * 0.04 + assists_per90 * 0.02
        def_boost = POSITION_DEFENSE_WEIGHT.get(position, 0.01)

        if to_name and to_name in adjusted["clubs"]:
            adjusted["clubs"][to_name]["attack"] = round(adjusted["clubs"][to_name]["attack"] + atk_boost, 4)
            adjusted["clubs"][to_name]["defense"] = round(adjusted["clubs"][to_name]["defense"] - def_boost, 4)

        if from_name and from_name in adjusted["clubs"]:
            adjusted["clubs"][from_name]["attack"] = round(adjusted["clubs"][from_name]["attack"] - atk_boost * 0.7, 4)
            adjusted["clubs"][from_name]["defense"] = round(adjusted["clubs"][from_name]["defense"] + def_boost * 0.7, 4)

    return adjusted


def _simulate_season(ratings: dict, rng: np.random.Generator) -> dict[str, int]:
    """Simula una temporada completa y devuelve puntos por club."""
    clubs = list(ratings["clubs"].keys())
    home_adv = ratings["home_advantage"]
    points: dict[str, int] = {c: 0 for c in clubs}

    for i, home in enumerate(clubs):
        for j, away in enumerate(clubs):
            if i == j:
                continue
            h_atk = ratings["clubs"][home]["attack"]
            h_def = ratings["clubs"][home]["defense"]
            a_atk = ratings["clubs"][away]["attack"]
            a_def = ratings["clubs"][away]["defense"]

            exp_home = max(h_atk * a_def * home_adv, 0.1)
            exp_away = max(a_atk * h_def, 0.1)

            home_goals = rng.poisson(exp_home)
            away_goals = rng.poisson(exp_away)

            if home_goals > away_goals:
                points[home] += 3
            elif away_goals > home_goals:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

    return points


def simulate(transfers: list[dict]) -> dict[str, Any]:
    base_ratings = _load_ratings()
    adjusted_ratings = _apply_transfers(base_ratings, transfers)

    clubs = list(base_ratings["clubs"].keys())
    rng = np.random.default_rng(seed=None)

    base_titles: dict[str, int] = {c: 0 for c in clubs}
    adj_titles: dict[str, int] = {c: 0 for c in clubs}
    base_pts_acc: dict[str, list] = {c: [] for c in clubs}
    adj_pts_acc: dict[str, list] = {c: [] for c in clubs}

    for _ in range(ITERATIONS):
        base_pts = _simulate_season(base_ratings, rng)
        adj_pts = _simulate_season(adjusted_ratings, rng)

        base_winner = max(base_pts, key=lambda c: base_pts[c])
        adj_winner = max(adj_pts, key=lambda c: adj_pts[c])
        base_titles[base_winner] += 1
        adj_titles[adj_winner] += 1

        for c in clubs:
            base_pts_acc[c].append(base_pts[c])
            adj_pts_acc[c].append(adj_pts[c])

    avg_adj_pts = {c: float(np.mean(adj_pts_acc[c])) for c in clubs}
    sorted_clubs = sorted(clubs, key=lambda c: avg_adj_pts[c], reverse=True)

    table = []
    for pos, club in enumerate(sorted_clubs, start=1):
        base_odds = base_titles[club] / ITERATIONS * 100
        adj_odds = adj_titles[club] / ITERATIONS * 100
        table.append({
            "position": pos,
            "club": club,
            "club_id": base_ratings["clubs"][club]["id"],
            "title_odds_delta": round(adj_odds - base_odds, 2),
        })

    return {"table": table}
