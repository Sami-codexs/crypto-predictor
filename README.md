# Crypto Price Predictor

Predicts Bitcoin price direction (up/down) using AI.

## What Makes It Special
- Most projects: Train model in Jupyter notebook → Done
- This project: Live data → API → Docker → Testing → Monitoring

## Quick Start
1. pip install -r requirements.txt
2. python collect_500.py (collect data)
3. python train.py --quick (train model)
4. streamlit run dashboard.py (see dashboard)

## Technologies
Python, TensorFlow, FastAPI, Docker, Prometheus, Streamlit

## Features
- Live Bitcoin data every hour
- LSTM neural network predictions
- REST API for predictions
- Docker containerization
- Automated testing (15 tests)
- Real-time dashboard
- Data drift monitoring

## Result
55-65% accuracy predicting Bitcoin price direction
(Better than random guessing = 50%)