"""
Descarga el dataset transfermarkt de Kaggle y genera data/transfers_enriched.csv
listo para entrenar el Transfer Predictor.

Requiere credenciales de Kaggle configuradas (~/.kaggle/access_token o kaggle.json).

Corre desde la raíz del repo:
  python scripts/prepare_dataset.py

Fixes aplicados respecto a la versión anterior:
  1. POS_MAP corregido — Kaggle usa "Defender"/"Midfield"/"Attack", no strings granulares.
  2. Club ratings — total_market_value es NaN en clubs.csv; se calcula sumando market_value_in_eur
     de los jugadores actuales de cada club desde players.csv.
  3. Stats join — stats de carrera agregadas por jugador en lugar de join por temporada frágil.
"""

import os
import numpy as np
import pandas as pd
import kagglehub

DATASET = "davidcariboo/player-scores"
OUT_PATH = os.path.join("data", "transfers_enriched.csv")
MIN_DATE = "2014-01-01"
PL_COMPETITION = "GB1"

POS_MAP = {
    "Goalkeeper": "Goalkeeper",
    "Defender":   "Defender",
    "Midfield":   "Midfielder",
    "Attack":     "Forward",
    "Missing":    "Midfielder",
}

FINAL_COLS = [
    "player_age", "position", "market_value_eur", "years_left",
    "goals_per90", "assists_per90", "minutes_played",
    "target_league_position", "position_needed", "transferred",
]


def map_pos(pos):
    if pd.isna(pos):
        return "Midfielder"
    return POS_MAP.get(pos, "Midfielder")


# Fix 2: calcula el MV del club sumando market_value_in_eur de sus jugadores actuales
def enrich_clubs(clubs: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    club_mv = (
        players.dropna(subset=["market_value_in_eur", "current_club_id"])
        .groupby("current_club_id")["market_value_in_eur"]
        .sum()
        .reset_index(name="total_mv")
    )
    clubs = clubs.merge(club_mv, left_on="club_id", right_on="current_club_id", how="left")
    clubs["target_league_position"] = (
        clubs.groupby("domestic_competition_id")["total_mv"]
        .rank(ascending=False, method="first", na_option="bottom")
        .fillna(999)
        .astype(int)
    )
    return clubs


def build_squad_counts(players: pd.DataFrame) -> pd.DataFrame:
    return (
        players.groupby(["current_club_id", "position_mapped"])
        .size()
        .reset_index(name="pos_count")
    )


def build_career_stats(appearances: pd.DataFrame) -> pd.DataFrame:
    stats = (
        appearances.groupby("player_id")
        .agg(goals=("goals", "sum"), assists=("assists", "sum"), minutes=("minutes_played", "sum"))
        .reset_index()
    )
    stats["goals_per90"] = (stats["goals"] / stats["minutes"].clip(lower=1) * 90).clip(0, 3)
    stats["assists_per90"] = (stats["assists"] / stats["minutes"].clip(lower=1) * 90).clip(0, 2)
    stats["minutes_played"] = stats["minutes"].clip(0, 3420)
    return stats[["player_id", "goals_per90", "assists_per90", "minutes_played"]]


def build_positives(transfers, players, career_stats, clubs_enriched, squad_counts):
    df = transfers.copy()
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df = df[df["transfer_date"] >= MIN_DATE].dropna(subset=["transfer_date"])

    players["dob"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["contract_expiry"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
    players["position_mapped"] = players["position"].apply(map_pos)
    players["market_value_eur"] = players["market_value_in_eur"].fillna(500_000)

    df = df.merge(
        players[["player_id", "position_mapped", "dob", "contract_expiry", "market_value_eur"]],
        on="player_id",
        how="inner",
    )

    df["player_age"] = ((df["transfer_date"] - df["dob"]).dt.days / 365.25).round(1)
    df["years_left"] = (
        (df["contract_expiry"] - df["transfer_date"]).dt.days / 365.25
    ).clip(0, 6).fillna(1.5)

    df = df.merge(career_stats, on="player_id", how="left")
    df["goals_per90"] = df["goals_per90"].fillna(0.0)
    df["assists_per90"] = df["assists_per90"].fillna(0.0)
    df["minutes_played"] = df["minutes_played"].fillna(900.0)

    pl_club_ids = set(
        clubs_enriched.loc[clubs_enriched["domestic_competition_id"] == PL_COMPETITION, "club_id"]
    )
    df = df[df["to_club_id"].isin(pl_club_ids)]

    df = df.merge(
        clubs_enriched[["club_id", "target_league_position"]].rename(columns={"club_id": "to_club_id"}),
        on="to_club_id",
        how="left",
    )
    df["target_league_position"] = df["target_league_position"].fillna(10).astype(int)

    df = df.merge(
        squad_counts.rename(columns={"current_club_id": "to_club_id", "position_mapped": "position_match"}),
        left_on=["to_club_id", "position_mapped"],
        right_on=["to_club_id", "position_match"],
        how="left",
    )
    df["position_needed"] = (df["pos_count"].fillna(0) < 3).astype(int)
    df["position"] = df["position_mapped"]
    df["transferred"] = 1

    return df.dropna(subset=["player_age", "market_value_eur"])


OTHER_TOP_COMPS = {"ES1", "L1", "IT1", "FR1"}


def build_negatives_unsigned(
    positives_df: pd.DataFrame,
    all_transfers: pd.DataFrame,
    players: pd.DataFrame,
    career_stats: pd.DataFrame,
    clubs_enriched: pd.DataFrame,
    squad_counts: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Negatives: jugadores que se traspasaron a otras ligas top (La Liga, Bundesliga,
    Serie A, Ligue 1) en el mismo periodo pero NO fueron fichados por la PL.
    Esto hace que los features de jugador sean discriminativos, a diferencia de
    usar el mismo jugador con un club alternativo.
    """
    pl_player_ids = set(positives_df["player_id"])
    other_top_club_ids = set(
        clubs_enriched.loc[
            clubs_enriched["domestic_competition_id"].isin(OTHER_TOP_COMPS), "club_id"
        ]
    )

    non_pl = all_transfers[
        all_transfers["to_club_id"].isin(other_top_club_ids)
        & ~all_transfers["player_id"].isin(pl_player_ids)
    ].copy()
    non_pl["transfer_date"] = pd.to_datetime(non_pl["transfer_date"], errors="coerce")
    non_pl = non_pl[non_pl["transfer_date"] >= MIN_DATE].dropna(subset=["transfer_date"])

    players_copy = players.copy()
    players_copy["dob"] = pd.to_datetime(players_copy["date_of_birth"], errors="coerce")
    players_copy["contract_expiry"] = pd.to_datetime(players_copy["contract_expiration_date"], errors="coerce")
    players_copy["position_mapped"] = players_copy["position"].apply(map_pos)
    players_copy["market_value_eur"] = players_copy["market_value_in_eur"].fillna(500_000)

    df = non_pl.merge(
        players_copy[["player_id", "position_mapped", "dob", "contract_expiry", "market_value_eur"]],
        on="player_id",
        how="inner",
    )
    df["player_age"] = ((df["transfer_date"] - df["dob"]).dt.days / 365.25).round(1)
    df["years_left"] = (
        (df["contract_expiry"] - df["transfer_date"]).dt.days / 365.25
    ).clip(0, 6).fillna(1.5)

    df = df.merge(career_stats, on="player_id", how="left")
    df["goals_per90"] = df["goals_per90"].fillna(0.0)
    df["assists_per90"] = df["assists_per90"].fillna(0.0)
    df["minutes_played"] = df["minutes_played"].fillna(900.0)

    n_needed = len(positives_df)
    if len(df) > n_needed:
        df = df.sample(n_needed, random_state=42).reset_index(drop=True)

    pl_clubs = clubs_enriched[clubs_enriched["domestic_competition_id"] == PL_COMPETITION].copy()
    sampled_clubs = pl_clubs.sample(len(df), replace=True, random_state=42).reset_index(drop=True)
    squad_lookup = (
        squad_counts
        .set_index(["current_club_id", "position_mapped"])["pos_count"]
        .to_dict()
    )

    df["target_league_position"] = sampled_clubs["target_league_position"].values
    df["_club_id"] = sampled_clubs["club_id"].values
    df["position_needed"] = df.apply(
        lambda r: int(squad_lookup.get((r["_club_id"], r["position_mapped"]), 0) < 3),
        axis=1,
    )
    df["position"] = df["position_mapped"]
    df["transferred"] = 0

    return df.dropna(subset=["player_age", "market_value_eur"])[FINAL_COLS]


def main():
    print("[prepare] Cargando dataset desde cache de Kaggle…")
    path = kagglehub.dataset_download(DATASET)
    print(f"[prepare] Dataset en: {path}")

    transfers   = pd.read_csv(f"{path}/transfers.csv")
    players     = pd.read_csv(f"{path}/players.csv")
    appearances = pd.read_csv(f"{path}/appearances.csv")
    clubs       = pd.read_csv(f"{path}/clubs.csv")

    print(f"[prepare] Transfers: {len(transfers):,} | Players: {len(players):,} | Appearances: {len(appearances):,}")

    players["position_mapped"] = players["position"].apply(map_pos)

    print("[prepare] Calculando ratings de clubes desde market values de jugadores…")
    clubs_enriched = enrich_clubs(clubs, players)
    squad_counts = build_squad_counts(players)

    print("[prepare] Calculando stats de carrera por jugador…")
    career_stats = build_career_stats(appearances)
    print(f"[prepare] Stats disponibles para {len(career_stats):,} jugadores")

    print("[prepare] Construyendo ejemplos positivos…")
    positives_df = build_positives(transfers, players, career_stats, clubs_enriched, squad_counts)
    positives = positives_df[FINAL_COLS]
    print(f"[prepare] Positivos: {len(positives):,}")

    print("[prepare] Generando negativos (jugadores de otras ligas top que NO fueron fichados por PL)…")
    negatives = build_negatives_unsigned(
        positives_df, transfers, players, career_stats, clubs_enriched, squad_counts, np.random.default_rng(42)
    )
    print(f"[prepare] Negativos: {len(negatives):,}")

    combined = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    os.makedirs("data", exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\n[prepare] Guardado en {OUT_PATH}: {len(combined):,} filas")

    print("\n[prepare] Calidad del dataset:")
    print(f"  Target balance: {combined['transferred'].value_counts().to_dict()}")
    print(f"  Posiciones:     {combined['position'].value_counts().to_dict()}")
    print(f"  goals_per90==0:       {(combined['goals_per90']==0).mean()*100:.1f}%")
    print(f"  minutes==900:         {(combined['minutes_played']==900).mean()*100:.1f}%")
    print(f"  league_pos==10 (def): {(combined['target_league_position']==10).mean()*100:.1f}%")
    print("\n[prepare] Siguiente paso: python -m models.transfer_predictor.train")


if __name__ == "__main__":
    main()
