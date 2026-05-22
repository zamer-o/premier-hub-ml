"""
Calibra club_ratings.json usando resultados reales de API-Football.

Modelo: Poisson multiplicativo
  E(home_goals) = attack_home * defense_away * home_advantage
  E(away_goals) = attack_away * defense_home

Calibración directa (no Dixon-Coles MLE):
  attack_i  = (goals_scored_i  / games_i) / league_avg_per_team
  defense_i = (goals_conceded_i / games_i) / league_avg_per_team
  home_advantage = total_home_goals / total_away_goals

Corre desde la raíz del repo:
  python scripts/calibrate_ratings.py
  python scripts/calibrate_ratings.py --season 2024
"""

import os
import sys
import json
import argparse
import httpx

API_KEY = os.environ.get("APIFOOTBALL_KEY", "45379e002ce9894ab347104d24165229")
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 39  # Premier League
RATINGS_PATH = os.path.join("models", "season_simulator", "club_ratings.json")

HEADERS = {"x-apisports-key": API_KEY}


def fetch_fixtures(season: int) -> list[dict]:
    """Devuelve todos los partidos finalizados (FT) de la temporada."""
    fixtures = []
    page = 1
    while True:
        resp = httpx.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": LEAGUE_ID, "season": season, "status": "FT"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("response", [])
        fixtures.extend(results)
        paging = data.get("paging", {})
        if page >= paging.get("total", 1):
            break
        page += 1
    return fixtures


def fetch_teams(season: int) -> dict[int, str]:
    """Devuelve {team_id: team_name} para la temporada."""
    resp = httpx.get(
        f"{BASE_URL}/teams",
        headers=HEADERS,
        params={"league": LEAGUE_ID, "season": season},
        timeout=30,
    )
    resp.raise_for_status()
    teams = {}
    for entry in resp.json().get("response", []):
        t = entry.get("team", {})
        if t.get("id") and t.get("name"):
            teams[t["id"]] = t["name"]
    return teams


def calibrate(season: int) -> dict:
    print(f"[calibrate] Fetching teams for season {season}...")
    teams = fetch_teams(season)
    print(f"[calibrate] {len(teams)} equipos encontrados")

    print(f"[calibrate] Fetching fixtures (status=FT) para season {season}...")
    fixtures = fetch_fixtures(season)
    print(f"[calibrate] {len(fixtures)} partidos finalizados")

    if not fixtures:
        print("[calibrate] Sin datos de partidos. Saliendo.")
        sys.exit(1)

    # Acumula stats por equipo
    stats: dict[int, dict] = {
        tid: {"scored": 0, "conceded": 0, "games": 0}
        for tid in teams
    }
    total_home_goals = 0
    total_away_goals = 0

    for f in fixtures:
        goals = f.get("goals", {})
        home_g = goals.get("home")
        away_g = goals.get("away")
        if home_g is None or away_g is None:
            continue

        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        if home_id in stats:
            stats[home_id]["scored"] += home_g
            stats[home_id]["conceded"] += away_g
            stats[home_id]["games"] += 1
        if away_id in stats:
            stats[away_id]["scored"] += away_g
            stats[away_id]["conceded"] += home_g
            stats[away_id]["games"] += 1

        total_home_goals += home_g
        total_away_goals += away_g

    total_goals = total_home_goals + total_away_goals
    total_games = len(fixtures)
    league_avg_per_team = (total_goals / total_games) / 2  # goals per team per game

    print(f"[calibrate] Goles totales: {total_goals} en {total_games} partidos")
    print(f"[calibrate] Liga avg goles/equipo/partido: {league_avg_per_team:.3f}")
    print(f"[calibrate] Home/Away ratio: {total_home_goals}/{total_away_goals} = {total_home_goals/max(total_away_goals,1):.3f}")

    home_advantage = round(total_home_goals / max(total_away_goals, 1), 4)

    clubs_out = {}
    for tid, name in sorted(teams.items(), key=lambda x: x[1]):
        s = stats.get(tid, {})
        games = s.get("games", 0)
        if games == 0:
            print(f"  [WARN] {name} sin partidos registrados, usando defaults")
            clubs_out[name] = {"id": tid, "attack": 1.0, "defense": 1.0}
            continue

        scored_per_game = s["scored"] / games
        conceded_per_game = s["conceded"] / games
        attack = round(scored_per_game / league_avg_per_team, 4)
        defense = round(conceded_per_game / league_avg_per_team, 4)

        clubs_out[name] = {"id": tid, "attack": attack, "defense": defense}
        print(f"  {name:25s} games={games:3d}  atk={attack:.4f}  def={defense:.4f}")

    result = {"home_advantage": home_advantage, "clubs": clubs_out}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    calibrated = calibrate(args.season)

    with open(RATINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(calibrated, f, indent=2, ensure_ascii=False)

    print(f"\n[calibrate] Guardado en {RATINGS_PATH}")
    print(f"[calibrate] home_advantage = {calibrated['home_advantage']}")
    print(f"[calibrate] {len(calibrated['clubs'])} clubes calibrados")


if __name__ == "__main__":
    main()
