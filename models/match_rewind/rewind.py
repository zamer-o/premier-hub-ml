"""
Classic Match Rewind v2 — modelo determinista por evento + momentum.

Cada tipo de evento se modela con su mecánica real:

- Gol eliminado: resta directa al marcador. Los demás goles del partido
  permanecen — quitar el gol del minuto 90 no borra los 5 anteriores.

- Momentum (heurística): quitar un gol, sobre todo temprano, deja al equipo
  que lo recibió jugar con más libertad el resto del partido. Se le estima un
  pequeño boost de xG ponderado por el minuto. ES UNA ESTIMACIÓN, no un dato
  del partido, y está topada para que nunca fabrique una remontada absurda.

- Tarjeta roja eliminada: ponderada por el minuto. El equipo expulsado jugó
  W = L - minuto con un hombre menos. Quitar la roja revierte parte de la
  ventaja numérica del rival (proporcional a W) y devuelve algo de producción
  ofensiva al expulsado.

- Resultado final: marcador real ajustado por todos los deltas y redondeado
  al entero más probable. Determinista — el mismo escenario da siempre lo mismo.
"""

from typing import Any

# Parte de los goles del rival en la ventana post-roja atribuible a la
# superioridad numérica (se revierte al quitar la roja).
RED_CARD_OPP_SHARE = 0.42
# Producción ofensiva que recupera el equipo expulsado al volver a ser 11.
RED_CARD_TEAM_RECLAIM = 0.30
# Tope: una roja nunca explica más del 60 % del marcador del rival.
RED_CARD_MAX_SHARE = 0.60

# Heurística de momentum: xG estimado que gana el equipo por cada gol temprano
# evitado, ponderado por minutos restantes. Topado para no fabricar remontadas.
MOMENTUM_FACTOR = 0.45
MOMENTUM_CAP    = 1.2
# Boost mínimo para que valga la pena mostrarlo como razón.
MOMENTUM_MIN_SHOWN = 0.15


def _calc_xg(stats: dict, team: str) -> float:
    """xG proxy desde stats del partido (tiros, tiros a puerta, ataques)."""
    shots_on    = stats.get(f"{team}_shots_on_target", 0)
    shots_total = stats.get(f"{team}_shots", 0)
    attacks     = stats.get(f"{team}_dangerous_attacks", 0)
    total       = stats.get("total_attacks", max(attacks, 1))
    ratio       = attacks / total if total > 0 else 0
    xg = shots_on * 0.28 + shots_total * 0.04 + ratio * 0.18
    return max(xg, 0.05)


def _performance_label(goals: int, xg: float) -> str:
    if xg <= 0.1:
        return "sin apenas ocasiones de peligro"
    ratio = goals / xg
    if ratio >= 1.8:
        return "muy eficaz de cara al gol"
    if ratio >= 1.2:
        return "bien aprovechando sus ocasiones"
    if ratio >= 0.8:
        return "convirtiendo un porcentaje normal de sus llegadas"
    return "desperdiciando ocasiones claras de gol"


def rewind(
    match_data: dict,
    removed_goals: list | None = None,
    removed_red_cards: list | None = None,
) -> dict[str, Any]:

    score = match_data.get("score", {})
    stats = match_data.get("stats", {})
    L = match_data.get("match_minutes") or 95

    original_score = {"home": score.get("home", 0), "away": score.get("away", 0)}
    xg_home = _calc_xg(stats, "home")
    xg_away = _calc_xg(stats, "away")

    removed_goals     = removed_goals or []
    removed_red_cards = removed_red_cards or []

    key_changes: list[dict] = []

    # ── Contexto del partido real ─────────────────────────────────────────────
    key_changes.append({
        "description": (
            f"Partido real: el local marcó {original_score['home']} "
            f"({_performance_label(original_score['home'], xg_home)}); "
            f"el visitante marcó {original_score['away']} "
            f"({_performance_label(original_score['away'], xg_away)})."
        ),
        "xg_delta": 0.0,
    })

    exp_home = float(original_score["home"])
    exp_away = float(original_score["away"])

    # ── Goles eliminados — resta directa, determinista ────────────────────────
    g_home = [g for g in removed_goals if g.get("team") == "home"]
    g_away = [g for g in removed_goals if g.get("team") == "away"]

    if g_home:
        exp_home -= len(g_home)
        mins = ", ".join(f"min {g.get('minute', 0)}" for g in g_home)
        key_changes.append({
            "description": (
                f"Se eliminan {len(g_home)} gol(es) del local ({mins}). "
                f"Los demás goles del partido permanecen: el local pasa de "
                f"{original_score['home']} a {max(int(exp_home), 0)}."
            ),
            "xg_delta": float(-len(g_home)),
        })

    if g_away:
        exp_away -= len(g_away)
        mins = ", ".join(f"min {g.get('minute', 0)}" for g in g_away)
        key_changes.append({
            "description": (
                f"Se eliminan {len(g_away)} gol(es) del visitante ({mins}). "
                f"El visitante pasa de {original_score['away']} a "
                f"{max(int(exp_away), 0)}."
            ),
            "xg_delta": float(-len(g_away)),
        })

    # ── Momentum — efecto de game-state de los goles eliminados ───────────────
    # El equipo que RECIBÍA cada gol eliminado gana un boost de xG: sin ir
    # perdiendo (sobre todo temprano) habría jugado con más libertad.
    momentum = {"home": 0.0, "away": 0.0}
    for g in removed_goals:
        scorer   = g.get("team")
        conceder = "away" if scorer == "home" else "home"
        minute   = max(0, min(int(g.get("minute", 45)), L))
        weight   = (L - minute) / L if L > 0 else 0   # gol temprano pesa más
        momentum[conceder] += MOMENTUM_FACTOR * weight

    momentum["home"] = min(momentum["home"], MOMENTUM_CAP)
    momentum["away"] = min(momentum["away"], MOMENTUM_CAP)

    for side, side_label in (("home", "local"), ("away", "visitante")):
        if momentum[side] >= MOMENTUM_MIN_SHOWN:
            if side == "home":
                exp_home += momentum[side]
            else:
                exp_away += momentum[side]
            key_changes.append({
                "description": (
                    f"Sin esos goles en contra, el {side_label} no habría jugado "
                    f"a la defensiva. Habría llegado más al área rival y generado "
                    f"más ocasiones de gol a lo largo del partido."
                ),
                "xg_delta": round(momentum[side], 2),
            })

    # ── Tarjetas rojas eliminadas — ponderadas por minuto ─────────────────────
    for rc in removed_red_cards:
        carded = rc.get("team", "home")          # equipo que recibió la roja
        opp    = "away" if carded == "home" else "home"
        minute = max(0, min(int(rc.get("minute", 45)), L))
        window = L - minute                      # minutos jugados en inferioridad
        frac   = window / L if L > 0 else 0      # peso temporal 0..1

        opp_goals = original_score[opp]
        opp_loss = min(
            opp_goals * frac * RED_CARD_OPP_SHARE,
            opp_goals * RED_CARD_MAX_SHARE,
        )
        carded_xg = xg_home if carded == "home" else xg_away
        team_gain = max(carded_xg, 0.3) * frac * RED_CARD_TEAM_RECLAIM

        if opp == "home":
            exp_home -= opp_loss
        else:
            exp_away -= opp_loss
        if carded == "home":
            exp_home += team_gain
        else:
            exp_away += team_gain

        carded_label = "local" if carded == "home" else "visitante"
        opp_label    = "local" if opp == "home" else "visitante"
        if window <= 6:
            key_changes.append({
                "description": (
                    f"Sin la expulsión del {carded_label} (min {minute}): solo "
                    f"quedaban {window} minutos, por lo que el impacto en el "
                    f"marcador habría sido mínimo."
                ),
                "xg_delta": 0.0,
            })
        else:
            key_changes.append({
                "description": (
                    f"Sin la expulsión del {carded_label} (min {minute}): el "
                    f"partido habría seguido 11 contra 11 durante {window} "
                    f"minutos más. El {opp_label} habría marcado menos goles al "
                    f"perder la ventaja numérica, y el {carded_label} habría "
                    f"tenido más presencia en ataque."
                ),
                "xg_delta": -round(opp_loss, 2),
            })

    exp_home = max(exp_home, 0.0)
    exp_away = max(exp_away, 0.0)

    # ── Resultado final — redondeo determinista ───────────────────────────────
    ph = int(exp_home + 0.5)
    pa = int(exp_away + 0.5)
    predicted = {"home": ph, "away": pa}

    if ph > pa:
        verdict = f"el local habría ganado {ph}–{pa}"
    elif pa > ph:
        verdict = f"el visitante habría ganado {pa}–{ph}"
    else:
        verdict = f"el partido habría terminado en empate {ph}–{pa}"

    key_changes.append({
        "description": (
            f"Con todos los cambios aplicados, el resultado alternativo "
            f"más probable habría sido: {verdict}."
        ),
        "xg_delta": 0.0,
    })

    no_change = (
        ph == original_score["home"] and pa == original_score["away"]
    )

    return {
        "original_score": original_score,
        "predicted_score": predicted,
        "key_changes": key_changes,
        "no_change": no_change,
    }
