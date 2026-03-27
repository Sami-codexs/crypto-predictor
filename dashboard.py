"""
Day 23: Streamlit Dashboard

Interactive web dashboard for crypto prediction visualization.
Run with: streamlit run dashboard.py
"""

import sys
sys.path.insert(0, 'src')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from database import CryptoDatabase
from indicators import TechnicalIndicators
from predictor import CryptoPredictor
from model_manager import ModelManager

# Page config
st.set_page_config(
    page_title="Crypto Predictor Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #1E88E5;
    }
    .prediction-up {
        color: #00C853;
        font-weight: bold;
    }
    .prediction-down {
        color: #FF1744;
        font-weight: bold;
    }
    .prediction-neutral {
        color: #FFC400;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load recent price data."""
    db = CryptoDatabase()
    try:
        df = db.get_recent_prices("bitcoin", hours=168)  # Last 7 days
        return df
    except:
        return pd.DataFrame()


def get_prediction():
    """Get current prediction."""
    try:
        manager = ModelManager()
        best_model = manager.get_best_model('accuracy')
        
        if best_model:
            predictor = CryptoPredictor(model_path=best_model)
            result = predictor.predict_with_threshold("bitcoin", confidence_threshold=0.6)
            return result
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main():
    """Main dashboard."""
    
    # Header
    st.markdown('<p class="main-header">📈 Crypto Price Predictor Dashboard</p>', unsafe_allow_html=True)
    st.markdown("*Real-time Bitcoin price prediction using LSTM neural networks*")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    coin = st.sidebar.selectbox("Select Coin", ["bitcoin", "ethereum"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
    hours_to_show = st.sidebar.slider("Hours to Display", 24, 168, 72, 24)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About:**
    - LSTM Neural Network
    - Technical Indicators: RSI, MACD, Bollinger Bands
    - Real-time predictions
    """)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()
    
    # Get prediction
    st.subheader("🔮 Current Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    
    prediction_data = get_prediction()
    
    if prediction_data and 'error' not in prediction_data:
        pred = prediction_data.get('prediction', 'unknown')
        conf = prediction_data.get('confidence', 0)
        price = prediction_data.get('current_price', 0)
        prob = prediction_data.get('probability', 0)
        
        # Determine color class
        pred_class = f"prediction-{pred}"
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction</h3>
                <p class="{pred_class}" style="font-size: 2rem;">{pred.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <p style="font-size: 2rem; color: #1E88E5;">{conf:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <p style="font-size: 2rem; color: #1E88E5;">${price:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Probability</h3>
                <p style="font-size: 2rem; color: #1E88E5;">{prob:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Indicators
        if 'indicators' in prediction_data:
            st.subheader("📊 Technical Indicators")
            ind = prediction_data['indicators']
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("RSI", f"{ind.get('rsi', 0):.2f}")
            with c2:
                st.metric("MACD Signal", ind.get('macd_signal', 'neutral'))
            with c3:
                st.metric("BB Position", f"{ind.get('bb_position', 0):.2f}")
            with c4:
                st.metric("Volatility", f"{ind.get('volatility', 0):.2f}" if ind.get('volatility') else "N/A")
    else:
        st.warning("⚠️ No prediction available. Train a model first.")
    
    # Price Chart
    st.subheader("📈 Price History")
    
    df = load_data()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create candlestick-like chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price_usd'],
            mode='lines',
            name='Price',
            line=dict(color='#1E88E5', width=2),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.1)'
        ))
        
        fig.update_layout(
            title=f"{coin.upper()} Price (Last {hours_to_show} hours)",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📉 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current", f"${df['price_usd'].iloc[-1]:,.2f}")
        with col2:
            change = ((df['price_usd'].iloc[-1] / df['price_usd'].iloc[0]) - 1) * 100
            st.metric("Change", f"{change:+.2f}%")
        with col3:
            st.metric("High", f"${df['price_usd'].max():,.2f}")
        with col4:
            st.metric("Low", f"${df['price_usd'].min():,.2f}")
    else:
        st.warning("No data available. Run data collection.")
    
    # Model Performance
    st.subheader("🤖 Model Performance")
    
    try:
        manager = ModelManager()
        models = manager.list_models()
        
        if models:
            model_data = []
            for m in models[:5]:
                model_data.append({
                    'Model': m['name'],
                    'Created': m.get('created', 'Unknown')[:10] if m.get('created') else 'Unknown',
                    'Accuracy': f"{m.get('accuracy', 0):.3f}" if m.get('accuracy') else 'N/A'
                })
            
            st.table(pd.DataFrame(model_data))
        else:
            st.info("No models found. Run training first.")
    except Exception as e:
        st.error(f"Could not load models: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit | LSTM Crypto Predictor</p>
        <p>Day 23: Interactive Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()