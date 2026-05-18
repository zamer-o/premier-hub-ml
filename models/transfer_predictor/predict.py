"""
Inferencia del Transfer Predictor.

Si model.pkl existe: usa XGBoost.
Si no: usa scoring heurístico de respaldo para desarrollo.
"""

import os
import numpy as np
import joblib
from typing import Any

MODEL_PATH = os.path.join("models", "transfer_predictor", "model.pkl")
ENCODER_PATH = os.path.join("models", "transfer_predictor", "encoder.pkl")

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

FEATURE_REASONS = {
    "market_value_eur": ("alto valor de mercado", "bajo valor de mercado"),
    "goals_per90": ("anotador prolífico", "aportación ofensiva limitada"),
    "assists_per90": ("creador de juego clave", "poca participación en asistencias"),
    "player_age": ("edad ideal para el fichaje", "edad fuera del rango objetivo del club"),
    "years_left": ("contrato próximo a vencer", "contrato largo, fichaje costoso"),
    "position_needed": ("posición que necesita el club", "posición bien cubierta en el club"),
    "minutes_played": ("minutos sólidos la temporada pasada", "falta de minutos recientes"),
}


def _fit_score(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def _heuristic_predict(player_stats: dict, target_club_stats: dict) -> dict:
    """Scoring de respaldo cuando no hay modelo entrenado."""
    score = 0.5

    age = player_stats.get("player_age", 25)
    if 21 <= age <= 28:
        score += 0.08
    elif age > 32:
        score -= 0.12

    market_value = player_stats.get("market_value_eur", 0)
    if market_value > 30_000_000:
        score += 0.06
    elif market_value < 5_000_000:
        score -= 0.05

    years_left = player_stats.get("years_left", 2)
    if years_left <= 1:
        score += 0.10
    elif years_left >= 4:
        score -= 0.08

    position_needed = target_club_stats.get("position_needed", False)
    if position_needed:
        score += 0.12

    goals = player_stats.get("goals_per90", 0)
    assists = player_stats.get("assists_per90", 0)
    if goals + assists > 0.8:
        score += 0.07

    probability = float(np.clip(score, 0.01, 0.99))
    return {"probability": round(probability * 100, 1), "fit_score": _fit_score(probability)}


def _build_reasons(player_stats: dict, target_club_stats: dict, model=None, feature_vector=None) -> list[str]:
    reasons = []

    years_left = player_stats.get("years_left", 2)
    age = player_stats.get("player_age", 25)
    market_value = player_stats.get("market_value_eur", 0)
    goals = player_stats.get("goals_per90", 0)
    assists = player_stats.get("assists_per90", 0)
    position_needed = target_club_stats.get("position_needed", False)
    league_pos = target_club_stats.get("target_league_position", 10)

    if years_left <= 1:
        reasons.append("Contrato próximo a vencer — el club puede negociar a precio reducido")
    elif years_left >= 4:
        reasons.append("Contrato largo — el fichaje requeriría una cláusula de rescisión elevada")

    if 21 <= age <= 27:
        reasons.append(f"Edad de {age} años: en su pico de rendimiento y con valor de reventa")
    elif age > 31:
        reasons.append(f"Con {age} años, el impacto en el valor de la plantilla es limitado")

    if market_value > 50_000_000:
        reasons.append(f"Valor de mercado alto (€{market_value/1e6:.0f}M) — solo clubes top pueden permitírselo")
    elif market_value < 5_000_000:
        reasons.append("Valor de mercado bajo — poca resistencia del club vendedor")

    if position_needed:
        reasons.append("El club destino tiene necesidad real en la posición del jugador")
    else:
        reasons.append("La posición del jugador ya está bien cubierta en el club destino")

    if goals + assists > 0.8:
        reasons.append(f"Estadísticas ofensivas sólidas ({goals:.2f} goles + {assists:.2f} asist. por 90 min)")
    elif goals + assists < 0.3 and player_stats.get("position", "") not in ["Goalkeeper", "Defender"]:
        reasons.append("Producción ofensiva baja para su posición")

    if league_pos <= 6:
        reasons.append("El club destino juega competición europea — atractivo para el jugador")

    return reasons[:4]


def predict(player_stats: dict, target_club_stats: dict) -> dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        result = _heuristic_predict(player_stats, target_club_stats)
        reasons = _build_reasons(player_stats, target_club_stats)
        return {**result, "reasons": reasons}

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    position_raw = player_stats.get("position", "Midfielder")
    position_enc = le.transform([position_raw if position_raw in POSITIONS else "Midfielder"])[0]

    feature_vector = np.array([[
        player_stats.get("player_age", 25),
        position_enc,
        player_stats.get("market_value_eur", 0),
        player_stats.get("years_left", 2),
        player_stats.get("goals_per90", 0),
        player_stats.get("assists_per90", 0),
        player_stats.get("minutes_played", 1800),
        target_club_stats.get("target_league_position", 10),
        int(target_club_stats.get("position_needed", False)),
    ]])

    probability = float(model.predict_proba(feature_vector)[0][1])
    reasons = _build_reasons(player_stats, target_club_stats, model, feature_vector)

    return {
        "probability": round(probability * 100, 1),
        "fit_score": _fit_score(probability),
        "reasons": reasons,
    }
