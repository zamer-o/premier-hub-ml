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
from imblearn.over_sampling import SMOTE

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


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
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

    print("[train] Aplicando SMOTE para balancear clases...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("[train] Entrenando XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = model.predict(X_test)
    print("\n[train] Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n[train] Modelo guardado en {MODEL_PATH}")
    print(f"[train] Encoder guardado en {ENCODER_PATH}")


if __name__ == "__main__":
    train()
