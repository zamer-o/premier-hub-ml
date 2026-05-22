"""
Season Simulator — Monte Carlo vectorizado con NumPy + distribución de Poisson.

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
    adjusted = copy.deepcopy(ratings)
    for t in transfers:
        from_name = _club_name_by_id(adjusted, t["from_club_id"])
        to_name   = _club_name_by_id(adjusted, t["to_club_id"])
        position      = t["player_stats"].get("position", "Midfielder")
        goals_per90   = t["player_stats"].get("goals_per90", 0)
        assists_per90 = t["player_stats"].get("assists_per90", 0)

        atk_boost = POSITION_ATTACK_WEIGHT.get(position, 0.02) + goals_per90 * 0.04 + assists_per90 * 0.02
        def_boost = POSITION_DEFENSE_WEIGHT.get(position, 0.01)

        if to_name and to_name in adjusted["clubs"]:
            adjusted["clubs"][to_name]["attack"]  += atk_boost
            adjusted["clubs"][to_name]["defense"] -= def_boost

        if from_name and from_name in adjusted["clubs"]:
            adjusted["clubs"][from_name]["attack"]  -= atk_boost * 0.7
            adjusted["clubs"][from_name]["defense"] += def_boost * 0.7

    return adjusted


def _build_lambdas(
    ratings: dict, clubs: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute expected goals for every non-self match pair."""
    n        = len(clubs)
    home_adv = ratings["home_advantage"]
    mask     = ~np.eye(n, dtype=bool)
    h_idx, a_idx = np.where(mask)
    n_matches    = len(h_idx)

    exp_h = np.empty(n_matches)
    exp_a = np.empty(n_matches)
    for k in range(n_matches):
        i, j    = int(h_idx[k]), int(a_idx[k])
        h_atk   = ratings["clubs"][clubs[i]]["attack"]
        h_def   = ratings["clubs"][clubs[i]]["defense"]
        a_atk   = ratings["clubs"][clubs[j]]["attack"]
        a_def   = ratings["clubs"][clubs[j]]["defense"]
        exp_h[k] = max(h_atk * a_def * home_adv, 0.1)
        exp_a[k] = max(a_atk * h_def, 0.1)

    return exp_h, exp_a, h_idx, a_idx


def _run_simulations(
    exp_h: np.ndarray, exp_a: np.ndarray,
    h_idx: np.ndarray, a_idx: np.ndarray,
    n_clubs: int, rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate ITERATIONS full seasons in one vectorized pass.
    Returns pts array of shape (ITERATIONS, n_clubs).
    """
    # Single draw for all iterations × all matches
    hg = rng.poisson(exp_h, size=(ITERATIONS, len(exp_h)))  # (N, M)
    ag = rng.poisson(exp_a, size=(ITERATIONS, len(exp_a)))

    h_pts = (hg > ag).astype(np.int32) * 3 + (hg == ag).astype(np.int32)
    a_pts = (ag > hg).astype(np.int32) * 3 + (hg == ag).astype(np.int32)

    pts = np.zeros((ITERATIONS, n_clubs), dtype=np.int32)
    for i in range(n_clubs):
        hm = h_idx == i
        am = a_idx == i
        pts[:, i] = h_pts[:, hm].sum(axis=1) + a_pts[:, am].sum(axis=1)

    return pts


def simulate(transfers: list[dict]) -> dict[str, Any]:
    base_ratings = _load_ratings()
    adj_ratings  = _apply_transfers(base_ratings, transfers)
    clubs        = list(base_ratings["clubs"].keys())
    n_clubs      = len(clubs)
    rng          = np.random.default_rng()

    exp_h_base, exp_a_base, h_idx, a_idx = _build_lambdas(base_ratings, clubs)
    exp_h_adj,  exp_a_adj,  _,    _     = _build_lambdas(adj_ratings,  clubs)

    base_pts = _run_simulations(exp_h_base, exp_a_base, h_idx, a_idx, n_clubs, rng)
    adj_pts  = _run_simulations(exp_h_adj,  exp_a_adj,  h_idx, a_idx, n_clubs, rng)

    avg_base = base_pts.mean(axis=0)
    avg_adj  = adj_pts.mean(axis=0)

    base_title_counts = np.bincount(base_pts.argmax(axis=1), minlength=n_clubs)
    adj_title_counts  = np.bincount(adj_pts.argmax(axis=1),  minlength=n_clubs)

    base_ranks = np.argsort(-base_pts, axis=1)
    adj_ranks  = np.argsort(-adj_pts,  axis=1)

    base_top4 = np.zeros(n_clubs, dtype=np.int32)
    adj_top4  = np.zeros(n_clubs, dtype=np.int32)
    base_rel  = np.zeros(n_clubs, dtype=np.int32)
    adj_rel   = np.zeros(n_clubs, dtype=np.int32)

    for pos in range(4):
        np.add.at(base_top4, base_ranks[:, pos], 1)
        np.add.at(adj_top4,  adj_ranks[:, pos],  1)
    for pos in range(n_clubs - 3, n_clubs):
        np.add.at(base_rel, base_ranks[:, pos], 1)
        np.add.at(adj_rel,  adj_ranks[:, pos],  1)

    sorted_idx = np.argsort(-avg_adj)
    table = []
    for pos, idx in enumerate(sorted_idx, start=1):
        club = clubs[int(idx)]
        b_tp = float(base_title_counts[idx]) / ITERATIONS * 100
        a_tp = float(adj_title_counts[idx])  / ITERATIONS * 100
        table.append({
            "position":               pos,
            "club":                   club,
            "club_id":                base_ratings["clubs"][club]["id"],
            "avg_pts":                round(float(avg_adj[idx]),  1),
            "avg_pts_base":           round(float(avg_base[idx]), 1),
            "title_probability":      round(a_tp, 1),
            "title_odds_delta":       round(a_tp - b_tp, 2),
            "top4_probability":       round(float(adj_top4[idx])  / ITERATIONS * 100, 1),
            "top4_delta":             round(float(adj_top4[idx]  - base_top4[idx]) / ITERATIONS * 100, 2),
            "relegation_probability": round(float(adj_rel[idx])   / ITERATIONS * 100, 1),
            "relegation_delta":       round(float(adj_rel[idx]   - base_rel[idx])  / ITERATIONS * 100, 2),
        })

    return {"table": table}
