# Premier Hub ML Service

Microservicio de Machine Learning de **Premier Hub**. Expone, mediante una API
HTTP construida con **FastAPI**, tres modelos analíticos sobre la Premier League
y un catálogo de partidos icónicos:

| Modelo | Endpoint | Qué hace |
| --- | --- | --- |
| **Transfer Predictor** | `POST /ml/transfer` | Estima la probabilidad de que un jugador fiche por un club destino y un *fit score* (Low / Medium / High) con sus razones. |
| **Season Simulator** | `POST /ml/simulate` | Simula 1000 temporadas (Monte Carlo + Poisson) aplicando fichajes hipotéticos y devuelve la tabla proyectada con deltas de título, top-4 y descenso. |
| **Classic Match Rewind** | `POST /ml/rewind` | Recalcula el marcador de un partido real eliminando goles o tarjetas rojas (modelo determinista por evento + momentum). |
| **Iconic Matches** | `GET /ml/iconic-matches` | Devuelve el catálogo de partidos memorables de la Premier League. |

El servicio escucha en el puerto **8080** y forma parte del stack desplegado en
**K3s** (namespaces `prod` / `preprod`), consumido por el backend `premier-hub-backend`.

---

## Tabla de contenidos

- [Arquitectura del repositorio](#arquitectura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución local](#ejecución-local)
- [API](#api)
- [Entrenamiento y generación de datos](#entrenamiento-y-generación-de-datos)
- [Variables de entorno](#variables-de-entorno)
- [Docker](#docker)
- [Despliegue (CI/CD)](#despliegue-cicd)
- [Notas de seguridad](#notas-de-seguridad)

---

## Arquitectura del repositorio

```
premier-hub-ml/
├── api.py                          # Entrypoint FastAPI: define los endpoints y modelos Pydantic
├── requirements.txt                # Dependencias de Python
├── Dockerfile                      # Imagen de producción (python:3.11-slim)
├── .github/workflows/
│   ├── deploy-prod.yml             # CI/CD → namespace prod   (push a main)
│   └── deploy-preprod.yml          # CI/CD → namespace preprod (push a preprod)
├── data/
│   ├── .gitkeep
│   └── iconic_matches.json         # Catálogo de partidos icónicos (servido por /ml/iconic-matches)
├── models/
│   ├── transfer_predictor/
│   │   ├── train.py                # Entrena el clasificador XGBoost
│   │   ├── predict.py              # Inferencia (XGBoost o heurística de respaldo)
│   │   ├── model.pkl               # Modelo entrenado
│   │   └── encoder.pkl             # LabelEncoder de posiciones
│   ├── season_simulator/
│   │   ├── simulate.py             # Monte Carlo vectorizado con Poisson
│   │   └── club_ratings.json       # Ratings de ataque/defensa por club
│   └── match_rewind/
│       └── rewind.py               # Modelo determinista por evento + momentum
└── scripts/
    ├── prepare_dataset.py          # Genera data/transfers_enriched.csv desde Kaggle
    ├── calibrate_ratings.py        # Calibra club_ratings.json con API-Football
    └── find_iconic_fixtures.py     # Genera data/iconic_matches.json con API-Football
```

**Diseño clave:** los modelos degradan con elegancia. `predict.py` usa el
`model.pkl` entrenado si existe y, si no, cae a un scoring heurístico; el
simulador y el rewind no requieren modelo entrenado (operan sobre
`club_ratings.json` y las estadísticas del propio partido). Esto permite
levantar el servicio sin necesidad de reentrenar nada.

---

## Requisitos

- **Python 3.11** (la imagen Docker usa `python:3.11-slim`).
- `pip` y, recomendado, un entorno virtual (`venv`).
- Solo para regenerar datos/modelos:
  - Credenciales de **Kaggle** (`prepare_dataset.py`).
  - Una **API key de API-Football** (`calibrate_ratings.py`, `find_iconic_fixtures.py`).

Dependencias principales (ver `requirements.txt` para versiones exactas):
`fastapi`, `uvicorn[standard]`, `pydantic`, `pandas`, `numpy`, `scikit-learn`,
`xgboost`, `imbalanced-learn`, `joblib`, `kagglehub`, `httpx`.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/zamer-o/premier-hub-ml.git
cd premier-hub-ml

# 2. Crear y activar un entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ejecución local

El servicio funciona de inmediato con los modelos y datos ya versionados en el
repo (`model.pkl`, `club_ratings.json`, `iconic_matches.json`).

```bash
# Opción A — directamente
python api.py

# Opción B — con uvicorn y recarga en caliente (desarrollo)
uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

Una vez arriba:

- API: <http://localhost:8080>
- Documentación interactiva (Swagger UI): <http://localhost:8080/docs>
- Health check: <http://localhost:8080/health> → `{"status": "ok"}`

> **Importante:** ejecuta siempre desde la **raíz del repositorio**. Los modelos
> y scripts resuelven rutas relativas (`models/...`, `data/...`).

---

## API

Base URL local: `http://localhost:8080`

### `GET /health`
Devuelve `{"status": "ok"}`. Usado por el liveness/readiness de K3s.

### `POST /ml/transfer`
Predice la probabilidad de un fichaje.

```bash
curl -X POST http://localhost:8080/ml/transfer \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": 1100,
    "target_club_id": 40,
    "player_stats": {
      "player_age": 24, "position": "Forward", "market_value_eur": 60000000,
      "goals_per90": 0.7, "assists_per90": 0.3, "minutes_played": 2700, "years_left": 1
    },
    "target_club_stats": { "target_league_position": 3, "position_needed": true }
  }'
```

Respuesta:
```json
{ "probability": 72.4, "fit_score": "High", "reasons": ["..."] }
```

### `POST /ml/simulate`
Simula la temporada con fichajes hipotéticos.

```bash
curl -X POST http://localhost:8080/ml/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "transfers": [{
      "player_id": 1100, "from_club_id": 50, "to_club_id": 40,
      "player_stats": { "goals_per90": 0.8, "assists_per90": 0.2, "position": "Forward" }
    }]
  }'
```

Respuesta: `{ "table": [ { "position", "club", "club_id", "avg_pts",
"avg_pts_base", "title_probability", "title_odds_delta", "top4_probability",
"top4_delta", "relegation_probability", "relegation_delta" }, ... ] }`.

### `POST /ml/rewind`
Recalcula el marcador eliminando goles o rojas.

```bash
curl -X POST http://localhost:8080/ml/rewind \
  -H "Content-Type: application/json" \
  -d '{
    "match_id": 868001,
    "match_data": { "score": {"home": 3, "away": 2}, "stats": {}, "match_minutes": 90 },
    "removed_goals": [{ "team": "home", "minute": 90 }],
    "removed_red_cards": []
  }'
```

Respuesta: `{ "original_score", "predicted_score", "key_changes": [{ "description",
"xg_delta" }], "no_change" }`.

### `GET /ml/iconic-matches`
Devuelve `{ "matches": [...] }` desde `data/iconic_matches.json`.

---

## Entrenamiento y generación de datos

Los artefactos entrenados (`model.pkl`, `encoder.pkl`, `club_ratings.json`,
`iconic_matches.json`) ya vienen en el repo. Solo necesitas estos pasos si
quieres **regenerarlos**. Todos los scripts se ejecutan desde la raíz.

### 1. Transfer Predictor

```bash
# Genera data/transfers_enriched.csv desde el dataset de Kaggle
# (davidcariboo/player-scores). Requiere credenciales de Kaggle.
python scripts/prepare_dataset.py

# Entrena el clasificador XGBoost y guarda model.pkl + encoder.pkl
python -m models.transfer_predictor.train
```

> `data/transfers_enriched.csv` está en `.gitignore` (no se versiona); se
> regenera con el script. Si falta y no hay `model.pkl`, `predict.py` usa la
> heurística de respaldo automáticamente.

### 2. Season Simulator

```bash
# Recalibra los ratings de ataque/defensa con resultados reales de API-Football
python scripts/calibrate_ratings.py                 # temporada por defecto
python scripts/calibrate_ratings.py --season 2024   # temporada concreta
```

### 3. Iconic Matches

```bash
# Genera data/iconic_matches.json con los partidos más memorables (2015–2024)
python scripts/find_iconic_fixtures.py
```

---

## Variables de entorno

| Variable | Usada por | Descripción |
| --- | --- | --- |
| `APIFOOTBALL_KEY` | `calibrate_ratings.py`, `find_iconic_fixtures.py` | API key de API-Football (v3). |
| Credenciales Kaggle | `prepare_dataset.py` | `~/.kaggle/kaggle.json` o token, requeridos por `kagglehub`. |

El servicio en runtime (`api.py`) **no** requiere variables de entorno: sirve
los artefactos ya versionados.

---

## Docker

```bash
# Construir la imagen
docker build -t premier-ml:latest .

# Ejecutar el contenedor
docker run --rm -p 8080:8080 premier-ml:latest
```

La imagen parte de `python:3.11-slim`, instala `requirements.txt`, copia el
código y arranca `uvicorn api:app` en el puerto **8080**.

---

## Despliegue (CI/CD)

El despliegue es automático mediante **GitHub Actions** sobre un **runner
self-hosted** (la propia máquina con K3s). No hay registry externo: la imagen se
construye, se importa a `containerd` de K3s y se actualiza el `Deployment`.

| Rama | Workflow | Destino |
| --- | --- | --- |
| `main` | `deploy-prod.yml` | namespace **`prod`** (`premier-ml:latest`) |
| `preprod` | `deploy-preprod.yml` | namespace **`preprod`** (`premier-ml:preprod`) |

Pasos que ejecuta cada workflow al hacer push:

```bash
docker build -t premier-ml:<sha> -t premier-ml:latest .
docker save premier-ml:<sha> | sudo k3s ctr images import -
kubectl set image deployment/ml ml=premier-ml:<sha> -n <prod|preprod>
kubectl rollout status deployment/ml -n <prod|preprod>
```

> **Atención:** cualquier push a `main` (incluido un cambio de documentación)
> dispara una reconstrucción de imagen y un redeploy en `prod`. Los manifiestos
> base del Deployment viven en el repo de infraestructura `premier-hub-infra`
> (`k8s/prod/ml-deployment.yaml`, `k8s/preprod/ml-deployment.yaml`) y usan
> `imagePullPolicy: Never` con imágenes locales de K3s.

---

## Notas de seguridad

Los scripts `calibrate_ratings.py` y `find_iconic_fixtures.py` traen una API key
de API-Football **hardcodeada como valor por defecto**. Se recomienda eliminarla
del código y exigir siempre `APIFOOTBALL_KEY` por entorno, además de rotar esa
clave. Mientras tanto, define la variable de entorno para sobreescribir el valor
embebido:

```bash
export APIFOOTBALL_KEY="tu_api_key"
```
