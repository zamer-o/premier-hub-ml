from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import uvicorn

app = FastAPI(title="Premier Hub ML Service", version="1.0.0")


# ── Transfer Predictor ────────────────────────────────────────────────────────

class TransferRequest(BaseModel):
    player_id: int
    target_club_id: int
    player_stats: dict  # {age, position, market_value_eur, goals_per90, assists_per90, minutes, years_left}
    target_club_stats: dict  # {league_position, squad_needs: [positions]}

class TransferResponse(BaseModel):
    probability: float
    fit_score: str  # "Low" | "Medium" | "High"
    reasons: List[str]


@app.post("/ml/transfer", response_model=TransferResponse)
async def predict_transfer(req: TransferRequest):
    try:
        from models.transfer_predictor.predict import predict
        return predict(req.player_stats, req.target_club_stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Season Simulator ──────────────────────────────────────────────────────────

class HypotheticalTransfer(BaseModel):
    player_id: int
    from_club_id: int
    to_club_id: int
    player_stats: dict  # {goals_per90, assists_per90, position}

class SimulateRequest(BaseModel):
    transfers: List[HypotheticalTransfer]

class ClubResult(BaseModel):
    position: int
    club: str
    club_id: int
    avg_pts: float
    avg_pts_base: float
    title_probability: float
    title_odds_delta: float
    top4_probability: float
    top4_delta: float
    relegation_probability: float
    relegation_delta: float

class SimulateResponse(BaseModel):
    table: List[ClubResult]


@app.post("/ml/simulate", response_model=SimulateResponse)
async def simulate_season(req: SimulateRequest):
    try:
        from models.season_simulator.simulate import simulate
        return simulate([t.model_dump() for t in req.transfers])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Classic Match Rewind ──────────────────────────────────────────────────────

class Modification(BaseModel):
    type: str  # "remove_player" | "change_substitution"
    player_id: int
    team: str  # "home" | "away"
    minute: Optional[int] = None

class RewindRequest(BaseModel):
    match_id: int
    match_data: dict  # {home_team, away_team, stats: {shots, shots_on_target, ...}, events, lineups}
    modifications: List[Modification]

class KeyChange(BaseModel):
    description: str
    xg_delta: float

class RewindResponse(BaseModel):
    original_score: dict  # {home: int, away: int}
    predicted_score: dict  # {home: int, away: int}
    key_changes: List[KeyChange]
    no_change: bool


@app.post("/ml/rewind", response_model=RewindResponse)
async def rewind_match(req: RewindRequest):
    try:
        from models.match_rewind.rewind import rewind
        return rewind(req.match_data, [m.model_dump() for m in req.modifications])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/iconic-matches")
async def iconic_matches():
    path = os.path.join("data", "iconic_matches.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="iconic_matches.json not found")
    with open(path, encoding="utf-8") as f:
        return {"matches": json.load(f)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
