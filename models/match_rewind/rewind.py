"""
Classic Match Rewind — xG proxy + simulación de Poisson.

No requiere modelo entrenado: calcula xG a partir de estadísticas
del partido disponibles en API-Football (tiros, tiros a puerta,
ataques peligrosos) y ajusta según la modificación recibida.
"""

import numpy as np
from typing import Any

ITERATIONS = 5000

# Pesos para calcular xG proxy desde estadísticas del partido
XG_WEIGHTS = {
    "shots_on_target": 0.28,
    "shots_total": 0.04,
    "dangerous_attacks_ratio": 0.18,
}

# Impacto de quitar un jugador por posición (reducción de xG del equipo)
REMOVE_PLAYER_IMPACT = {
    "Forward": 0.22,
    "Midfielder": 0.12,
    "Defender": 0.06,
    "Goalkeeper": 0.0,
}

# Impacto de cambiar el minuto de una sustitución
SUBSTITUTION_DELTA_PER_MINUTE = 0.002


def _calc_xg(stats: dict, team: str) -> float:
    shots_on = stats.get(f"{team}_shots_on_target", 0)
    shots_total = stats.get(f"{team}_shots", 0)
    attacks = stats.get(f"{team}_dangerous_attacks", 0)
    total_attacks = stats.get("total_attacks", max(attacks, 1))
    attack_ratio = attacks / total_attacks if total_attacks > 0 else 0

    xg = (
        shots_on * XG_WEIGHTS["shots_on_target"]
        + shots_total * XG_WEIGHTS["shots_total"]
        + attack_ratio * XG_WEIGHTS["dangerous_attacks_ratio"]
    )
    return max(xg, 0.05)


def _apply_modifications(
    xg_home: float,
    xg_away: float,
    modifications: list[dict],
    match_data: dict,
) -> tuple[float, float, list[dict]]:
    key_changes = []
    mod_xg_home = xg_home
    mod_xg_away = xg_away

    for mod in modifications:
        mod_type = mod["type"]
        team = mod["team"]
        player_id = mod["player_id"]

        player_position = _find_player_position(match_data, player_id, team)

        if mod_type == "remove_player":
            impact = REMOVE_PLAYER_IMPACT.get(player_position, 0.10)
            if team == "home":
                mod_xg_home = max(mod_xg_home - impact, 0.05)
                key_changes.append({
                    "description": f"Quitar al {player_position} del equipo local reduce su xG en {impact:.2f}",
                    "xg_delta": -impact,
                })
            else:
                mod_xg_away = max(mod_xg_away - impact, 0.05)
                key_changes.append({
                    "description": f"Quitar al {player_position} del equipo visitante reduce su xG en {impact:.2f}",
                    "xg_delta": -impact,
                })

        elif mod_type == "change_substitution":
            original_minute = mod.get("original_minute", 60)
            new_minute = mod.get("minute", 60)
            delta_minutes = original_minute - new_minute
            xg_delta = delta_minutes * SUBSTITUTION_DELTA_PER_MINUTE

            if team == "home":
                mod_xg_home = max(mod_xg_home + xg_delta, 0.05)
            else:
                mod_xg_away = max(mod_xg_away + xg_delta, 0.05)

            direction = "antes" if new_minute < original_minute else "después"
            key_changes.append({
                "description": f"Sustitución del {team} en minuto {new_minute} en vez de {original_minute} ({direction})",
                "xg_delta": round(xg_delta, 3),
            })

    return mod_xg_home, mod_xg_away, key_changes


def _find_player_position(match_data: dict, player_id: int, team: str) -> str:
    lineups = match_data.get("lineups", {})
    team_lineup = lineups.get(team, {})
    for player in team_lineup.get("startXI", []) + team_lineup.get("substitutes", []):
        if player.get("player", {}).get("id") == player_id:
            pos = player.get("player", {}).get("pos", "")
            if pos in ("G",):
                return "Goalkeeper"
            if pos in ("D", "CB", "LB", "RB", "LWB", "RWB"):
                return "Defender"
            if pos in ("M", "CM", "DM", "AM", "LM", "RM"):
                return "Midfielder"
            if pos in ("F", "CF", "LW", "RW", "SS"):
                return "Forward"
    return "Midfielder"


def _poisson_simulate(xg_home: float, xg_away: float, rng: np.random.Generator) -> dict:
    results = {"home": {}, "away": {}}
    home_scores = rng.poisson(xg_home, ITERATIONS)
    away_scores = rng.poisson(xg_away, ITERATIONS)

    home_wins = int((home_scores > away_scores).sum())
    away_wins = int((away_scores > home_scores).sum())
    draws = ITERATIONS - home_wins - away_wins

    if home_wins >= away_wins and home_wins >= draws:
        mode_h = int(np.bincount(home_scores[home_scores > away_scores]).argmax())
        mode_a = int(np.bincount(away_scores[home_scores > away_scores]).argmax())
    elif away_wins >= home_wins and away_wins >= draws:
        mode_h = int(np.bincount(home_scores[away_scores > home_scores]).argmax())
        mode_a = int(np.bincount(away_scores[away_scores > home_scores]).argmax())
    else:
        equal_mask = home_scores == away_scores
        mode_h = int(np.bincount(home_scores[equal_mask]).argmax()) if equal_mask.sum() > 0 else 0
        mode_a = mode_h

    return {"home": mode_h, "away": mode_a}


def rewind(match_data: dict, modifications: list[dict]) -> dict[str, Any]:
    stats = match_data.get("stats", {})
    original_score = {
        "home": match_data.get("score", {}).get("home", 0),
        "away": match_data.get("score", {}).get("away", 0),
    }

    xg_home = _calc_xg(stats, "home")
    xg_away = _calc_xg(stats, "away")

    mod_xg_home, mod_xg_away, key_changes = _apply_modifications(
        xg_home, xg_away, modifications, match_data
    )

    rng = np.random.default_rng()
    predicted_score = _poisson_simulate(mod_xg_home, mod_xg_away, rng)

    no_change = (
        predicted_score["home"] == original_score["home"]
        and predicted_score["away"] == original_score["away"]
    )

    return {
        "original_score": original_score,
        "predicted_score": predicted_score,
        "key_changes": key_changes,
        "no_change": no_change,
    }
