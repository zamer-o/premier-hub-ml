"""
Descarga el dataset transfermarkt de Kaggle y genera data/transfers_enriched.csv
listo para entrenar el Transfer Predictor.

Requiere credenciales de Kaggle configuradas (~/.kaggle/kaggle.json).

Corre desde la raíz del repo:
  python scripts/prepare_dataset.py
"""

import os
import numpy as np
import pandas as pd
import kagglehub

DATASET = "davidcariboo/player-scores"
OUT_PATH = os.path.join("data", "transfers_enriched.csv")
MIN_DATE = "2014-01-01"

POS_MAP = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender",
    "Left-Back": "Defender",
    "Right-Back": "Defender",
    "Left Wing-Back": "Defender",
    "Right Wing-Back": "Defender",
    "Defensive Midfield": "Midfielder",
    "Central Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Left Midfield": "Midfielder",
    "Right Midfield": "Midfielder",
    "Left Winger": "Forward",
    "Right Winger": "Forward",
    "Centre-Forward": "Forward",
    "Second Striker": "Forward",
}


def map_pos(pos):
    if pd.isna(pos):
        return "Midfielder"
    return POS_MAP.get(pos, "Midfielder")


def parse_market_value(series: pd.Series) -> pd.Series:
    """Convierte strings como '€1.23bn' o '€45.00m' a float en EUR."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.replace("€", "", regex=False).str.strip()
    multiplier = s.str.endswith("bn").map({True: 1e9, False: 1e6})
    values = s.str.rstrip("bm").str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce") * multiplier


def build_positives(transfers, players, appearances, clubs):
    df = transfers.copy()
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df = df[df["transfer_date"] >= MIN_DATE].dropna(subset=["transfer_date"])

    # Edad en el momento del traspaso
    players["dob"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["contract_expiry"] = pd.to_datetime(
        players["contract_expiration_date"], errors="coerce"
    )
    players["position_mapped"] = players["position"].apply(map_pos)
    players["market_value_eur"] = parse_market_value(
        players["market_value_in_eur"]
    ).fillna(1_000_000)

    df = df.merge(
        players[
            [
                "player_id",
                "position_mapped",
                "dob",
                "contract_expiry",
                "market_value_eur",
            ]
        ],
        on="player_id",
        how="inner",
    )

    df["player_age"] = ((df["transfer_date"] - df["dob"]).dt.days / 365.25).round(1)
    df["years_left"] = (
        (df["contract_expiry"] - df["transfer_date"]).dt.days / 365.25
    ).clip(0, 6)
    df["years_left"] = df["years_left"].fillna(1.5)

    # Stats ofensivas de la temporada previa
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["season"] = np.where(
        appearances["date"].dt.month >= 7,
        appearances["date"].dt.year,
        appearances["date"].dt.year - 1,
    )
    stats = (
        appearances.groupby(["player_id", "season"])
        .agg(goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum"))
        .reset_index()
    )
    stats["goals_per90"] = (stats["goals"] / stats["minutes"].clip(lower=1) * 90).clip(0, 3)
    stats["assists_per90"] = (stats["assists"] / stats["minutes"].clip(lower=1) * 90).clip(0, 2)

    df["prev_season"] = np.where(
        df["transfer_date"].dt.month >= 7,
        df["transfer_date"].dt.year - 1,
        df["transfer_date"].dt.year - 2,
    )
    df = df.merge(
        stats[["player_id", "season", "goals_per90", "assists_per90", "minutes"]],
        left_on=["player_id", "prev_season"],
        right_on=["player_id", "season"],
        how="left",
    )
    df["goals_per90"] = df["goals_per90"].fillna(0.0)
    df["assists_per90"] = df["assists_per90"].fillna(0.0)
    df["minutes_played"] = df["minutes"].fillna(900)

    # Rango del club destino dentro de su liga
    clubs = clubs.copy()
    clubs["mv_num"] = parse_market_value(clubs["total_market_value"])
    clubs["target_league_position"] = (
        clubs.groupby("domestic_competition_id")["mv_num"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    df = df.merge(
        clubs[["club_id", "target_league_position"]].rename(
            columns={"club_id": "to_club_id"}
        ),
        on="to_club_id",
        how="left",
    )
    df["target_league_position"] = df["target_league_position"].fillna(10).astype(int)

    # Necesidad posicional: ¿el club destino tiene < 3 jugadores en esa posición?
    squad_counts = (
        players.groupby(["current_club_id", "position_mapped"])
        .size()
        .reset_index(name="pos_count")
    )
    df = df.merge(
        squad_counts.rename(
            columns={
                "current_club_id": "to_club_id",
                "position_mapped": "position_match",
            }
        ),
        left_on=["to_club_id", "position_mapped"],
        right_on=["to_club_id", "position_match"],
        how="left",
    )
    df["position_needed"] = (df["pos_count"].fillna(0) < 3).astype(int)

    df["position"] = df["position_mapped"]
    df["transferred"] = 1

    cols = [
        "player_age", "position", "market_value_eur", "years_left",
        "goals_per90", "assists_per90", "minutes_played",
        "target_league_position", "position_needed", "transferred",
    ]
    return df[cols].dropna(subset=["player_age", "market_value_eur"])


def build_negatives(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "player_age": rng.normal(26, 4, n).clip(17, 38),
            "position": rng.choice(
                ["Goalkeeper", "Defender", "Midfielder", "Forward"], n
            ),
            "market_value_eur": rng.exponential(10_000_000, n).clip(100_000, 150_000_000),
            "years_left": rng.uniform(1, 5, n),
            "goals_per90": rng.exponential(0.15, n).clip(0, 1.5),
            "assists_per90": rng.exponential(0.10, n).clip(0, 1.0),
            "minutes_played": rng.normal(1500, 600, n).clip(0, 3420),
            "target_league_position": rng.integers(1, 21, n),
            "position_needed": rng.choice([0, 1], n, p=[0.6, 0.4]),
            "transferred": 0,
        }
    )


def main():
    print("[prepare] Descargando dataset de Kaggle…")
    path = kagglehub.dataset_download(DATASET)
    print(f"[prepare] Dataset en: {path}")

    transfers = pd.read_csv(f"{path}/transfers.csv")
    players = pd.read_csv(f"{path}/players.csv")
    appearances = pd.read_csv(f"{path}/appearances.csv")
    clubs = pd.read_csv(f"{path}/clubs.csv")

    print(f"[prepare] Transfers: {len(transfers):,} | Players: {len(players):,} | Appearances: {len(appearances):,}")

    print("[prepare] Construyendo ejemplos positivos…")
    positives = build_positives(transfers, players, appearances, clubs)
    print(f"[prepare] Positivos: {len(positives):,}")

    print("[prepare] Generando negativos sintéticos…")
    negatives = build_negatives(len(positives))

    combined = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    os.makedirs("data", exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"[prepare] Guardado en {OUT_PATH}: {len(combined):,} filas")
    print("[prepare] Siguiente paso: python -m models.transfer_predictor.train")


if __name__ == "__main__":
    main()
