"""
Genera data/iconic_matches.json con los 50 partidos más memorables de la PL.

Estrategia:
- Descarga todas las temporadas 2015-2024 de API-Football (1 req/temporada)
- Filtra: total_goals >= 5 OR margin >= 4
- Calcula score de "iconicidad" = total_goals*2 + margin + derby_bonus
- Toma los top 50 y genera descripciones automáticas

Corre desde la raíz del repo:
  python scripts/find_iconic_fixtures.py
"""

import os
import json
import time
import httpx

API_KEY = os.environ.get("APIFOOTBALL_KEY", "45379e002ce9894ab347104d24165229")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
OUT_PATH = os.path.join("data", "iconic_matches.json")

SEASONS = list(range(2015, 2025))  # 2015/16 hasta 2024/25

DERBIES = {
    frozenset({50, 33}): "derbi de Manchester",
    frozenset({40, 33}): "clásico Liverpool-United",
    frozenset({42, 47}): "derbi del norte de Londres",
    frozenset({40, 42}): "duelo de titanes",
    frozenset({49, 42}): "duelo de Londres",
    frozenset({49, 47}): "derbi de Londres",
    frozenset({45, 40}): "derbi de Merseyside",
    frozenset({48, 47}): "derbi del este de Londres",
}

CLUB_NAMES = {
    33: "Manchester United", 34: "Newcastle", 35: "Bournemouth", 36: "Fulham",
    38: "Watford", 39: "Wolves", 40: "Liverpool", 41: "Southampton",
    42: "Arsenal", 45: "Everton", 46: "Leicester", 47: "Tottenham",
    48: "West Ham", 49: "Chelsea", 50: "Manchester City", 51: "Brighton",
    52: "Crystal Palace", 55: "Brentford", 57: "Ipswich", 63: "Leeds",
    65: "Nottingham Forest", 66: "Aston Villa", 71: "Norwich",
}


def fetch_season(season: int) -> list[dict]:
    resp = httpx.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": 39, "season": season, "status": "FT"},
        timeout=40,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("response", [])
    # Paginacion si aplica
    paging = data.get("paging", {})
    page = 1
    while page < paging.get("total", 1):
        page += 1
        r2 = httpx.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": 39, "season": season, "status": "FT", "page": page},
            timeout=40,
        )
        r2.raise_for_status()
        results.extend(r2.json().get("response", []))
    return results


def iconicity_score(home_g: int, away_g: int, home_id: int, away_id: int) -> float:
    total = home_g + away_g
    margin = abs(home_g - away_g)
    score = total * 2 + margin
    if frozenset({home_id, away_id}) in DERBIES:
        score += 6
    return score


def make_description(home_name: str, away_name: str, home_g: int, away_g: int,
                     home_id: int, away_id: int, season: int) -> str:
    total = home_g + away_g
    margin = abs(home_g - away_g)
    winner = home_name if home_g > away_g else away_name
    loser = away_name if home_g > away_g else home_name
    derby_label = DERBIES.get(frozenset({home_id, away_id}))
    season_str = f"{season}/{str(season+1)[2:]}"

    if total >= 9:
        return f"Récord histórico en la Premier League. {winner} logró una goleada monumental ante {loser} en la temporada {season_str}."
    if total >= 7:
        return f"Una tarde de ensueño para {winner}. {total} goles en un partido que quedó grabado en la memoria del fútbol inglés ({season_str})."
    if total >= 5 and margin >= 4:
        if derby_label:
            return f"El {derby_label} más desequilibrado en años. {winner} dominó de principio a fin con {home_g}-{away_g} en {season_str}."
        return f"{winner} aplastó a {loser} con una actuación dominante. Una de las goleadas más recordadas de la temporada {season_str}."
    if derby_label and margin >= 3:
        return f"El {derby_label} se resolvió con contundencia. {winner} ganó {max(home_g,away_g)}-{min(home_g,away_g)} en un resultado que sorprendió a todos en {season_str}."
    if total >= 6:
        return f"Un partido de infarto con {total} goles. {home_name} {home_g}-{away_g} {away_name}, temporada {season_str}."
    if derby_label:
        return f"Partido memorable del {derby_label} en {season_str}. {home_name} {home_g}-{away_g} {away_name}."
    return f"{winner} se impuso ante {loser} en uno de los partidos más emotivos de la temporada {season_str} con {total} goles."


def main():
    candidates = []

    for season in SEASONS:
        print(f"[iconic] Fetching season {season}...")
        try:
            fixtures = fetch_season(season)
        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(2)
            continue
        print(f"  {len(fixtures)} partidos finalizados")

        for f in fixtures:
            goals = f.get("goals", {})
            home_g = goals.get("home")
            away_g = goals.get("away")
            if home_g is None or away_g is None:
                continue

            total = home_g + away_g
            margin = abs(home_g - away_g)

            if total < 5 and margin < 4:
                continue

            home_id = f["teams"]["home"]["id"]
            away_id = f["teams"]["away"]["id"]
            home_name = f["teams"]["home"]["name"]
            away_name = f["teams"]["away"]["name"]
            fixture_id = f["fixture"]["id"]
            date = f["fixture"]["date"][:10]

            score = iconicity_score(home_g, away_g, home_id, away_id)

            candidates.append({
                "fixture_id": fixture_id,
                "title": f"{home_name} {home_g}-{away_g} {away_name}",
                "description": make_description(home_name, away_name, home_g, away_g, home_id, away_id, season),
                "date": date,
                "season": season,
                "home_team": {"id": home_id, "name": home_name},
                "away_team": {"id": away_id, "name": away_name},
                "score": {"home": home_g, "away": away_g},
                "_score": score,
            })

        time.sleep(0.3)

    # Deduplicar por fixture_id, ordenar y tomar top 50
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: -x["_score"]):
        if c["fixture_id"] not in seen:
            seen.add(c["fixture_id"])
            del c["_score"]
            unique.append(c)

    top50 = unique[:50]

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(top50, f, indent=2, ensure_ascii=False)

    print(f"\n[iconic] {len(top50)} partidos guardados en {OUT_PATH}")
    for m in top50[:10]:
        print(f"  {m['title']} ({m['date']})")


if __name__ == "__main__":
    main()
