# ⬡ CryptoMind

**AI-Powered Cryptocurrency Price Direction Predictor**

LSTM-based up/down classification with a full AI intelligence layer (Ollama LLM), real-time drift monitoring, and a cyberpunk Streamlit dashboard.

---

## What Makes It Special

| Typical Project | CryptoMind |
|-----------------|------------|
| Train in Jupyter → Done | Live pipeline → API → AI Layer → Drift Detection → Docker → Render.com |
| 50% random baseline | 55–65% directional accuracy + risk-adjusted Sharpe |
| Static predictions | Real-time AI explanations grounded in RSI/MACD/Bollinger |
| No monitoring | PSI/KS drift detection with automated retraining alerts |
| No deployment | One-click Render.com deploy with Ollama sidecar |

---

## Architecture
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CoinGecko  │────▶│  Scheduler  │────▶│   SQLite    │
│   (hourly)  │     │  (APSched)  │     │   (data)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
│
┌────────────────────────┘
▼
┌─────────────────┐
│  Technical      │
│  Indicators     │  RSI, MACD, Bollinger, Volatility
│  (pandas_ta)    │
└────────┬────────┘
│
┌─────────────┼─────────────┐
▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────────┐
│  LSTM   │  │  Drift  │  │  AI Layer   │
│  Model  │  │Detector │  │  (Ollama)   │
│(TF/Keras│  │(PSI/KS) │  │  llama3.1   │
└────┬────┘  └────┬────┘  └──────┬──────┘
│            │              │
└────────────┴──────────────┘
│
┌────┴────┐
▼         ▼
┌────────┐ ┌──────────┐
│ FastAPI│ │ Streamlit│
│  /docs │ │Dashboard │
└────────┘ └──────────┘
plain

---

## Quick Start

### Local Development

```bash
# 1. Clone & setup
git clone <repo>
cd cryptomind
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama (for AI layer)
ollama serve &
ollama pull llama3.1

# 3. Seed demo data (2000+ synthetic rows)
python seed_demo_data.py

# 4. Train model
python train.py --epochs 50 --batch-size 32

# 5. Start API
uvicorn src.api:app --reload --port 8000

# 6. Start Dashboard (new terminal)
streamlit run dashboard.py