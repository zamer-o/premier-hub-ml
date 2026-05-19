"""
Entrena el modelo de Transfer Predictor.

Dataset esperado en data/transfers_enriched.csv con columnas:
  player_age, position_encoded, market_value_eur, years_left,
  goals_per90, assists_per90, minutes_played,
  target_league_position, position_needed,
  transferred (1 = ocurrió, 0 = negativo sintético)

Corre desde la raíz del repo:
  python -m models.transfer_predictor.train
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
DATA_PATH = os.path.join("data", "transfers_enriched.csv")
MODEL_PATH = os.path.join("models", "transfer_predictor", "model.pkl")
ENCODER_PATH = os.path.join("models", "transfer_predictor", "encoder.pkl")

FEATURES = [
    "player_age",
    "position_encoded",
    "market_value_eur",
    "years_left",
    "goals_per90",
    "assists_per90",
    "minutes_played",
    "target_league_position",
    "position_needed",
]

TARGET = "transferred"

RAW_FEATURES = [f if f != "position_encoded" else "position" for f in FEATURES]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=RAW_FEATURES + [TARGET])
    df["market_value_eur"] = df["market_value_eur"].clip(upper=200_000_000)
    df["goals_per90"] = df["goals_per90"].clip(0, 3)
    df["assists_per90"] = df["assists_per90"].clip(0, 2)
    df["minutes_played"] = df["minutes_played"].clip(0, 3420)
    df["years_left"] = df["years_left"].clip(0, 6)
    return df


def encode_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    le.fit(POSITIONS)
    df = df.copy()
    df["position_encoded"] = le.transform(
        df["position"].apply(lambda p: p if p in POSITIONS else "Midfielder")
    )
    return df, le


def train():
    if not os.path.exists(DATA_PATH):
        print(f"[train] Dataset no encontrado en {DATA_PATH}. Ver README para instrucciones.")
        sys.exit(1)

    print("[train] Cargando dataset...")
    df = load_and_clean(DATA_PATH)
    df, le = encode_positions(df)

    X = df[FEATURES].values
    y = df[TARGET].values

    print(f"[train] Muestras totales: {len(y)} | Positivas: {y.sum()} | Negativas: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"[train] Distribución train — neg={neg}, pos={pos}")

    print("[train] Entrenando XGBoost con early stopping…")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    print("\n[train] Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n[train] Modelo guardado en {MODEL_PATH}")
    print(f"[train] Encoder guardado en {ENCODER_PATH}")


if __name__ == "__main__":
    train()
