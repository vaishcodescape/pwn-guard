# Pwn-Guard

**Spam & fraud detection REST API** — production-ready service for message analysis, scam classification, threat intelligence extraction, and risk scoring. Runs on CPU-only infrastructure (Docker, [Render](https://render.com)).

---

## At a glance

| | |
|--|--|
| **Domain** | NLP, ML, fraud/spam detection, threat intelligence |
| **Backend** | Python 3.10+, FastAPI, Pydantic |
| **ML/NLP** | scikit-learn (TF-IDF, Logistic Regression), spaCy NER, custom feature pipelines |
| **Infra** | Docker, docker-compose, Render; optional Ollama/LLM locally |
| **Implementation** | REST API, ML training & serving, text preprocessing, NER, rule-based escalation, containerization |

---

## Features

- **Spam detection** — TF-IDF + Logistic Regression; returns `is_spam` and confidence.
- **Scam type classification** — `bank_phishing`, `otp_scam`, `job_fraud`, `crypto_scam`, `lottery_scam`, `loan_scam`, `other`.
- **Threat intelligence** — URLs, phone numbers, money amounts, org names (spaCy NER), action phrases and indicators.
- **Risk score (0–100)** and **recommended action** with detailed guidance.
- **Rule-based escalation simulation** — Offline simulation of N conversation turns (no LLM).
- **Scam baiting chat** — Optional LLM-powered victim persona for engaging scammers and extracting intel (Ollama).

---

## Architecture

<p align="center">
  <img src="pwn-guard architecture.png" alt="Pwn-Guard architecture diagram" width="900" />
</p>

- **API layer:** FastAPI routes for health, analyze, simulation, reports, reference, and chat.
- **Core:** Model loading, risk engine, rule-based escalation simulator.
- **Extraction:** Preprocessing, features, threat intel (spaCy NER + custom logic).
- **Optional:** Ollama for scam-baiting chat; Docker/Render for deployment.

---

## Quick start

### Local run

```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train models (from project root)
python -m training.train_spam
python -m training.train_scam_type

# Run API
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

| Link | URL |
|------|-----|
| **API** | http://127.0.0.1:8000 |
| **Docs (Swagger)** | http://127.0.0.1:8000/docs |
| **Health** | http://127.0.0.1:8000/health |
| **Ready** | http://127.0.0.1:8000/ready |

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/` | GET | Service info and model status |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness (models loaded) |
| `/analyze_message` | POST | Full analysis: spam, scam type, intel, risk score, recommended action |
| `/simulate_escalation` | POST | Rule-based scam escalation (offline, no LLM) |
| `/risk_report` | GET | Last analysis risk report |
| `/personas`, `/scam_types` | GET | Reference data |
| `/chat/start`, `/chat/respond`, `/chat/summary`, `/chat/end/{id}` | POST/GET/DELETE | Scam baiting chat (optional Ollama) |

### Example: analyze a message

**POST /analyze_message**

```json
{
  "text": "URGENT! Your SBI account is blocked. Verify at http://sbi-fake.com. Call +91-xxxxxxxxxx",
  "sender": "+91-xxxxxxxxxx",
  "channel": "sms"
}
```

Response includes: `is_spam`, `spam_confidence`, `scam_type`, `scam_confidence`, `risk_score`, `risk_level`, `recommended_action`, `action_details`, `threat_intelligence`, `features`, `processing_time_ms`.

### Example: simulate escalation

**POST /simulate_escalation**

```json
{
  "scam_type": "bank_phishing",
  "num_turns": 5,
  "initial_message": "Your SBI account is blocked. Verify at http://sbi-fake.com"
}
```

Returns a list of escalation turns with `phase` and `scammer_message` (no external service calls).

---

## Docker

### Build and run (API only)

```bash
docker build -t pwn-guard-api .
docker run -p 8000:8000 pwn-guard-api
```

### With Ollama (scam baiting chat)

```bash
docker-compose up --build
```

- Starts Ollama on port 11434, imports Sarah (3B) from [Hugging Face](https://huggingface.co/Het0456/sarah), runs FastAPI on 8000.
- **Memory:** allow Docker at least 8GB (Sarah needs ~4–6GB).

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Ollama | http://localhost:11434 |

First run may download `sarah.gguf` (~2–3GB); later runs use cache.

---

## Deployment

- **Render:** Use `/health` and `/ready` for health checks. Prefer `Dockerfile.cloudrun` (no Ollama) for Web Services.
- **Other:** Set env vars below; CPU-only, no GPU required.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `API_VERSION` | `1.0.0` | API version string |
| `MODEL_DIR` | `./models` | Path to model files |

---

## Project structure

```
├── config.py                 # Paths and env config
├── api/
│   ├── main.py               # FastAPI app, lifespan, routers
│   ├── schemas.py            # Pydantic request/response models
│   └── routes/               # health, analyze, simulation, reports, chat, reference
├── core/
│   ├── model_loader.py       # Load/cache spam & scam models
│   ├── risk_engine.py        # Risk score & recommendations
│   └── escalation_simulator.py  # Rule-based escalation (offline)
├── extraction/
│   ├── preprocess.py         # Text normalization
│   ├── features.py           # Feature extraction
│   └── extract_intel.py      # Threat intel (spaCy NER + rules)
├── bots/
│   └── scam_baiting_bot.py   # LLM-powered scam baiting (Ollama)
├── training/
│   ├── train_spam.py
│   └── train_scam_type.py
├── data/                     # Training data (CSV)
├── models/                   # Trained artifacts (.joblib)
├── Dockerfile
├── Dockerfile.cloudrun
├── docker-compose.yml
└── requirements.txt
```

---

## Requirements

- **Python** 3.10+
- **CPU only** (no GPU required)

---

## Contributing  

We welcome contributions. Here’s how to get started.

### 1. Fork and clone

- Fork the repository on GitHub.
- Clone your fork locally:
  ```bash
  git clone https://github.com/YOUR_USERNAME/pwn-guard.git
  cd pwn-guard
  ```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m training.train_spam
python -m training.train_scam_type
```

### 3. Create a branch

- Do not commit directly to `main`.
- Create a branch for your change:
  ```bash
  git checkout -b feature/your-feature-name
  # or: fix/issue-description
  ```

### 4. Make your changes

- Follow existing code style (e.g. type hints, docstrings, clear names).
- Prefer small, focused commits.
- If you add dependencies, add them to `requirements.txt` with versions where appropriate.

### 5. Test locally

- Run the API and hit the endpoints you changed:
  ```bash
  uvicorn api.main:app --reload
  ```
- Manually test or add/run tests if the project has a test suite.

### 6. Submit a pull request

- Push your branch to your fork:
  ```bash
  git push origin feature/your-feature-name
  ```
- Open a **Pull Request** against the upstream `main` branch.
- In the PR description, explain what changed and why; reference any issues (e.g. “Fixes #123”).
- Be responsive to review feedback.

### 7. Code of conduct

- Be respectful and constructive. This project aims to be inclusive and harassment-free.

---

## License

See [LICENSE](LICENSE) in this repository.
