import streamlit as st






import sys






import os






import time






import json






import numpy as np






import pandas as pd






from datetime import datetime, timedelta













# Page config MUST be first Streamlit command






st.set_page_config(






    page_title="CryptoMind",






    page_icon="⬡",






    layout="wide",






    initial_sidebar_state="expanded"






)













# Add src to path






sys.path.insert(0, 'src')













# ─── AI Layer (lazy import) ───






_AI_AVAILABLE = False






_ai_explainer = None






_ai_drift_narrator = None






_ai_reporter = None






_ai_chat = None













def _ensure_ai():






    global _AI_AVAILABLE, _ai_explainer, _ai_drift_narrator, _ai_reporter






    if _AI_AVAILABLE and _ai_reporter is not None:






        return






    try:






        from ai.engine import AIEngine






        from ai.explainer import PredictionExplainer






        from ai.drift_narrator import DriftNarrator






        from ai.reporter import MarketReporter






        engine = AIEngine()






        _ai_explainer = PredictionExplainer(engine)






        _ai_drift_narrator = DriftNarrator(engine)






        _ai_reporter = MarketReporter(engine)






        _AI_AVAILABLE = True






        print("AI layer initialized successfully with Groq")






    except Exception as e:






        print(f"AI init error: {e}")






        _AI_AVAILABLE = False






        













# Try importing backend modules






try:






    from database import CryptoDatabase






    from indicators import TechnicalIndicators






    from predictor import CryptoPredictor






    from model_manager import ModelManager






    from backtester import Backtester






    from model import CryptoLSTM






    from config import settings






    from drift_detector import DriftDetector






    BACKEND_AVAILABLE = True






except ImportError:






    BACKEND_AVAILABLE = False













import plotly.graph_objects as go






from plotly.subplots import make_subplots













# =============================================================================






# DESIGN SYSTEM CONSTANTS






# =============================================================================






BG_PAGE = "#060C10"






BG_CARD = "#0B1319"






BG_INNER = "#111C24"






BG_DEEP = "#172230"













GREEN = "#00FF88"






RED = "#FF4466"






AMBER = "#FFB700"






BLUE = "#4DC3FF"






MUTED = "#3A5060"













TEXT_PRIMARY = "#C8DDE8"






TEXT_SECONDARY = "#7A9BAD"






TEXT_MUTED = "#3A5060"













BORDER_DEFAULT = "rgba(0,255,136,0.12)"






BORDER_HOVER = "rgba(0,255,136,0.25)"






BORDER_ACTIVE = "rgba(0,255,136,0.45)"













# =============================================================================






# HIDE DEFAULT STREAMLIT CHROME






# =============================================================================






st.components.v1.html("""






<style>






@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;700;800&display=swap');













#MainMenu {visibility: hidden;}






footer {visibility: hidden;}






header {visibility: hidden;}













.stApp {






    background-color: #060C10;






    font-family: 'JetBrains Mono', monospace;






}













[data-testid="stSidebar"] {






    background-color: #0B1319 !important;






    border-right: 0.5px solid rgba(0,255,136,0.12) !important;






}













[data-testid="stSidebar"] .stMarkdown {






    color: #C8DDE8;






}













.stTabs [data-baseweb="tab-list"] {






    gap: 0px;






    background-color: #0B1319;






    border-radius: 12px;






    border: 0.5px solid rgba(0,255,136,0.12);






    padding: 4px;






}













.stTabs [data-baseweb="tab"] {






    font-family: 'JetBrains Mono', monospace;






    font-size: 12px;






    font-weight: 600;






    color: #3A5060;






    background: transparent;






    border: none;






    border-radius: 8px;






    padding: 8px 16px;






}













.stTabs [aria-selected="true"] {






    background-color: rgba(0,255,136,0.12) !important;






    color: #00FF88 !important;






    border: 0.5px solid rgba(0,255,136,0.3) !important;






}













.stButton>button {






    font-family: 'JetBrains Mono', monospace;






    background-color: #0B1319;






    color: #00FF88;






    border: 0.5px solid rgba(0,255,136,0.3);






    border-radius: 8px;






}













.stButton>button:hover {






    border-color: rgba(0,255,136,0.6);






    box-shadow: 0 0 20px rgba(0,255,136,0.1);






}













.stSlider>div>div>div {






    color: #00FF88;






}













.stSelectbox>div>div {






    background-color: #111C24;






    border: 0.5px solid rgba(0,255,136,0.12);






    border-radius: 8px;






    color: #C8DDE8;






}













.stMetric {






    background-color: #0B1319;






    border: 0.5px solid rgba(0,255,136,0.12);






    border-radius: 12px;






    padding: 16px;






}













.stMetric label {






    font-family: 'JetBrains Mono', monospace;






    font-size: 10px;






    color: #3A5060;






    letter-spacing: .12em;






    text-transform: uppercase;






}













.stMetric .css-1wivap2 {






    font-family: 'Syne', sans-serif;






    font-weight: 800;






    font-size: 28px;






}













.stDataFrame {






    background-color: #0B1319;






    border: 0.5px solid rgba(0,255,136,0.12);






    border-radius: 12px;






}













.stWarning {






    background-color: rgba(255,183,0,0.1);






    border: 0.5px solid rgba(255,183,0,0.3);






    color: #FFB700;






}













.stInfo {






    background-color: rgba(77,195,255,0.1);






    border: 0.5px solid rgba(77,195,255,0.3);






    color: #4DC3FF;






}













.section-label {






    font-size: 10px;






    font-weight: 600;






    letter-spacing: .12em;






    text-transform: uppercase;






    color: #3A5060;






    border-bottom: 0.5px solid rgba(0,255,136,0.07);






    padding-bottom: 8px;






    margin-bottom: 12px;






}













.badge-up {






    background: rgba(0,255,136,0.12);






    color: #00FF88;






    border: 0.5px solid rgba(0,255,136,0.3);






    border-radius: 6px;






    font-size: 11px;






    font-weight: 700;






    padding: 3px 9px;






}













.badge-down {






    background: rgba(255,68,102,0.12);






    color: #FF4466;






    border: 0.5px solid rgba(255,68,102,0.3);






    border-radius: 6px;






    font-size: 11px;






    font-weight: 700;






    padding: 3px 9px;






}













.badge-neutral {






    background: rgba(255,183,0,0.12);






    color: #FFB700;






    border: 0.5px solid rgba(255,183,0,0.3);






    border-radius: 6px;






    font-size: 11px;






    font-weight: 700;






    padding: 3px 9px;






}













.indicator-row {






    background-color: #111C24;






    border: 0.5px solid rgba(0,255,136,0.08);






    border-radius: 10px;






    padding: 12px 16px;






    margin-bottom: 8px;






    display: flex;






    align-items: center;






    justify-content: space-between;






}













.signal-history-row {






    background-color: #111C24;






    border: 0.5px solid rgba(0,255,136,0.08);






    border-radius: 8px;






    padding: 8px 12px;






    margin-bottom: 6px;






    display: flex;






    align-items: center;






    justify-content: space-between;






    font-size: 12px;






}













.trade-row-pos {






    background-color: rgba(0,255,136,0.05);






    border: 0.5px solid rgba(0,255,136,0.12);






    border-radius: 8px;






    padding: 10px 14px;






    margin-bottom: 6px;






}













.trade-row-neg {






    background-color: rgba(255,68,102,0.05);






    border: 0.5px solid rgba(255,68,102,0.12);






    border-radius: 8px;






    padding: 10px 14px;






    margin-bottom: 6px;






}













.status-dot {






    width: 8px;






    height: 8px;






    border-radius: 50%;






    display: inline-block;






    margin-right: 8px;






}













.status-ok { background-color: #00FF88; box-shadow: 0 0 8px #00FF88; }






.status-warn { background-color: #FFB700; box-shadow: 0 0 8px #FFB700; }






.status-err { background-color: #FF4466; box-shadow: 0 0 8px #FF4466; }













.ai-card {






    background: #111C24;






    border: 0.5px solid rgba(77,195,255,0.15);






    border-radius: 12px;






    padding: 16px;






    margin-bottom: 12px;






}













.ai-card-title {






    font-size: 11px;






    color: #4DC3FF;






    letter-spacing: .12em;






    text-transform: uppercase;






    margin-bottom: 8px;






}













.ai-card-body {






    font-size: 13px;






    color: #C8DDE8;






    line-height: 1.6;






}













.drift-alert-critical {






    background: rgba(255,68,102,0.08);






    border: 0.5px solid rgba(255,68,102,0.3);






    border-radius: 10px;






    padding: 12px 16px;






    margin-bottom: 8px;






}













.drift-alert-warning {






    background: rgba(255,183,0,0.08);






    border: 0.5px solid rgba(255,183,0,0.3);






    border-radius: 10px;






    padding: 12px 16px;






    margin-bottom: 8px;






}













.drift-alert-ok {






    background: rgba(0,255,136,0.05);






    border: 0.5px solid rgba(0,255,136,0.2);






    border-radius: 10px;






    padding: 12px 16px;






    margin-bottom: 8px;






}













.chat-bubble-user {






    background: #172230;






    border: 0.5px solid rgba(0,255,136,0.15);






    border-radius: 12px 12px 2px 12px;






    padding: 10px 14px;






    margin-bottom: 8px;






    font-size: 13px;






    color: #C8DDE8;






}













.chat-bubble-ai {






    background: #111C24;






    border: 0.5px solid rgba(77,195,255,0.15);






    border-radius: 12px 12px 12px 2px;






    padding: 10px 14px;






    margin-bottom: 8px;






    font-size: 13px;






    color: #C8DDE8;






    line-height: 1.5;






}






</style>






""", height=150, scrolling=False)













# =============================================================================






# SESSION STATE






# =============================================================================






if 'signal_history' not in st.session_state:






    st.session_state.signal_history = []






if 'last_coin' not in st.session_state:






    st.session_state.last_coin = None






if 'chat_history' not in st.session_state:






    st.session_state.chat_history = []






if 'briefing_cache' not in st.session_state:






    st.session_state.briefing_cache = {}






if 'drift_cache' not in st.session_state:






    st.session_state.drift_cache = {}













# =============================================================================






# DATA FUNCTIONS






# =============================================================================






@st.cache_data(ttl=300)






def load_price_data(coin, hours=3744):






    if not BACKEND_AVAILABLE:






        return None






    try:






        db = CryptoDatabase()






        df = db.get_recent_prices(coin, hours=hours)






        return df






    except Exception:






        return None













@st.cache_data(ttl=300)






def load_indicators(coin, hours=3744):






    if not BACKEND_AVAILABLE:






        return None






    try:






        df = load_price_data(coin, hours)






        if df is None or df.empty:






            return None






        ti = TechnicalIndicators()






        df = ti.engineer_features(coin, hours=hours)






        return df






    except Exception:






        return None













@st.cache_data(ttl=300)






def get_prediction(coin):






    if not BACKEND_AVAILABLE:






        return None






    try:






        predictor = CryptoPredictor()






        pred = predictor.predict_next(coin)






        return pred






    except Exception:






        return None













@st.cache_data(ttl=300)






def get_backtest_results(coin, threshold=0.6):






    if not BACKEND_AVAILABLE:






        return None






    try:






        predictor = CryptoPredictor()






        hours = 3744






        df = predictor.indicators.engineer_features(coin, hours=hours)






        if len(df) < 48:






            return None






        backtester = Backtester(initial_capital=10000.0)






        results = backtester.run_backtest(df, predictor.model, threshold=threshold, fee_pct=0.001)






        return results






    except Exception:






        return None













@st.cache_data(ttl=300)






def get_model_info():






    if not BACKEND_AVAILABLE:






        return None






    try:






        mm = ModelManager()






        models = mm.list_models()






        return models






    except Exception:






        return None













@st.cache_data(ttl=300)






def get_drift_data(coin):
    if not BACKEND_AVAILABLE:
        return None
    try:
        from drift_detector import DriftDetector
        from indicators import TechnicalIndicators
        import pandas as pd
        
        ti = TechnicalIndicators()
        df = ti.engineer_features(coin, hours=3744)
        if len(df) < 100:
            return None
        
        split_idx = int(len(df) * 0.8)
        reference = df.iloc[:split_idx]
        current = df.iloc[split_idx:]
        
        detector = DriftDetector()
        detector.set_reference(reference)
        results = detector.detect_drift(current)
        summary = detector.get_drift_summary()
        
        return {'results': results, 'summary': summary}
    except Exception as e:
        print(f"Drift error: {e}")
        return None

# =============================================================================






# COMPONENT 1 — WAVES HEADER






# =============================================================================






def waves_header_html(coin, price, prediction):






    pred_word = prediction.get('prediction', 'NEUTRAL').upper() if prediction else 'NEUTRAL'






    pred_color = GREEN if pred_word == 'UP' else RED if pred_word == 'DOWN' else AMBER






    pred_arrow = '↑' if pred_word == 'UP' else '↓' if pred_word == 'DOWN' else '◈'






    price_str = f"${price:,.2f}" if price else "$--"






    coin_upper = coin.upper() if coin else 'BTC'













    return f"""






<!DOCTYPE html>






<html>






<head>






<meta charset="UTF-8">






<style>






@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;700;800&display=swap');






* {{ margin: 0; padding: 0; box-sizing: border-box; }}






body {{ background: #060C10; overflow: hidden; font-family: 'JetBrains Mono', monospace; }}






#canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}






.overlay {{ 






    position: absolute; top: 0; left: 0; width: 100%; height: 100%; 






    background: linear-gradient(90deg, rgba(6,12,16,0.85) 0%, rgba(6,12,16,0.4) 50%, rgba(6,12,16,0.85) 100%);






    z-index: 2; pointer-events: none;






}}






.content {{ 






    position: absolute; top: 0; left: 0; width: 100%; height: 100%; 






    z-index: 10; display: flex; align-items: center; justify-content: space-between;






    padding: 0 32px; pointer-events: none;






}}






.left-section {{ display: flex; flex-direction: column; gap: 6px; }}






.logo-container {{ display: flex; align-items: center; gap: 12px; }}






.pulse-dot {{ 






    width: 10px; height: 10px; background: #00FF88; border-radius: 50%;






    box-shadow: 0 0 14px #00FF88;






    animation: pulse 2s ease-in-out infinite;






}}






@keyframes pulse {{






    0%, 100% {{ opacity: 1; transform: scale(1); }}






    50% {{ opacity: 0.6; transform: scale(0.8); }}






}}






.logo-text {{ 






    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 36px; 






    color: white; letter-spacing: -0.02em;






}}






.subtitle {{ 






    font-family: 'JetBrains Mono', monospace; font-size: 11px; 






    color: #3A5060; letter-spacing: .1em; text-transform: uppercase;






}}






.right-section {{ display: flex; gap: 16px; }}






.glass-card {{






    background: rgba(11,19,25,0.7); backdrop-filter: blur(4px);






    border: 0.5px solid rgba(0,255,136,0.15); border-radius: 10px;






    padding: 14px 20px; display: flex; flex-direction: column; gap: 4px;






    min-width: 140px;






}}






.glass-label {{ font-size: 10px; color: #3A5060; letter-spacing: .12em; text-transform: uppercase; }}






.glass-value {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px; color: #C8DDE8; }}






.glass-pred {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px; color: {pred_color}; }}






</style>






</head>






<body>






<canvas id="canvas"></canvas>






<div class="overlay"></div>






<div class="content">






    <div class="left-section">






        <div class="logo-container">






            <div class="pulse-dot"></div>






            <div class="logo-text" id="logoText">⬡ CryptoMind</div>






        </div>






        <div class="subtitle">LSTM PREDICTION TERMINAL · {coin_upper} / USD</div>






    </div>






    <div class="right-section">






        <div class="glass-card">






            <div class="glass-label">Live Price</div>






            <div class="glass-value" id="priceText">{price_str}</div>






        </div>






        <div class="glass-card">






            <div class="glass-label">Prediction</div>






            <div class="glass-pred" id="predText">{pred_arrow} {pred_word}</div>






        </div>






    </div>






</div>






<script>






(function() {{






    const canvas = document.getElementById('canvas');






    const ctx = canvas.getContext('2d');






    let width, height, points = [];






    let mouseX = -9999, mouseY = -9999;






    const xGap = 12, yGap = 36;













    function resize() {{






        width = canvas.width = window.innerWidth;






        height = canvas.height = window.innerHeight;






        initPoints();






    }}













    function initPoints() {{






        points = [];






        const cols = Math.floor(width / xGap) + 2;






        const rows = Math.floor(height / yGap) + 2;






        for (let r = 0; r < rows; r++) {{






            for (let c = 0; c < cols; c++) {{






                points.push({{






                    baseX: c * xGap, baseY: r * yGap,






                    x: c * xGap, y: r * yGap,






                    vx: 0, vy: 0, col: c, row: r






                }});






            }}






        }}






    }}













    canvas.addEventListener('mousemove', e => {{






        const rect = canvas.getBoundingClientRect();






        mouseX = e.clientX - rect.left;






        mouseY = e.clientY - rect.top;






    }});






    canvas.addEventListener('mouseleave', () => {{






        mouseX = -9999; mouseY = -9999;






    }});













    let t = 0;






    function animate() {{






        t += 1;






        ctx.clearRect(0, 0, width, height);






        const cols = Math.floor(width / xGap) + 2;






        const rows = Math.floor(height / yGap) + 2;













        for (let p of points) {{






            let tx = p.baseX + Math.sin(p.col * 0.4 + t * 0.0125) * 40;






            let ty = p.baseY + Math.sin(p.row * 0.5 + t * 0.01) * 20;






            if (mouseX > -1000) {{






                const dx = mouseX - p.x;






                const dy = mouseY - p.y;






                const dist = Math.sqrt(dx*dx + dy*dy);






                if (dist < 120 && dist > 0) {{






                    const force = (120 - dist) / 120;






                    p.vx += dx / dist * force * 0.5;






                    p.vy += dy / dist * force * 0.5;






                }}






            }}






            p.vx *= 0.9;






            p.vy *= 0.9;






            p.x += (tx - p.x) * 0.01 + p.vx;






            p.y += (ty - p.y) * 0.01 + p.vy;






        }}













        ctx.strokeStyle = 'rgba(0,255,136,0.12)';






        ctx.lineWidth = 1;






        for (let r = 0; r < rows; r++) {{






            for (let c = 0; c < cols - 1; c++) {{






                const i = r * cols + c;






                const j = r * cols + c + 1;






                if (points[i] && points[j]) {{






                    ctx.beginPath();






                    ctx.moveTo(points[i].x, points[i].y);






                    ctx.lineTo(points[j].x, points[j].y);






                    ctx.stroke();






                }}






            }}






        }}






        for (let r = 0; r < rows - 1; r++) {{






            for (let c = 0; c < cols; c++) {{






                const i = r * cols + c;






                const j = (r + 1) * cols + c;






                if (points[i] && points[j]) {{






                    ctx.beginPath();






                    ctx.moveTo(points[i].x, points[i].y);






                    ctx.lineTo(points[j].x, points[j].y);






                    ctx.stroke();






                }}






            }}






        }}






        requestAnimationFrame(animate);






    }}













    window.addEventListener('resize', resize);






    resize();






    animate();













    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$&*⬡◈';






    function shuffleText(element, finalText, delay) {{






        setTimeout(() => {{






            const original = finalText;






            const charArray = original.split('');






            const totalChars = charArray.length;






            let startTime = performance.now();






            const duration = 700;






            const staggerMs = 35;






            function update(now) {{






                const elapsed = now - startTime;






                let allDone = true;






                let result = '';






                for (let i = 0; i < totalChars; i++) {{






                    const charDelay = i * staggerMs;






                    const charElapsed = elapsed - charDelay;






                    if (charArray[i] === ' ') {{






                        result += ' ';






                        continue;






                    }}






                    if (charElapsed < 0) {{






                        result += chars[Math.floor(Math.random() * chars.length)];






                        allDone = false;






                    }} else if (charElapsed < duration) {{






                        const progress = charElapsed / duration;






                        if (Math.random() > progress) {{






                            result += chars[Math.floor(Math.random() * chars.length)];






                        }} else {{






                            result += charArray[i];






                        }}






                        allDone = false;






                    }} else {{






                        result += charArray[i];






                    }}






                }}






                element.textContent = result;






                if (!allDone) {{






                    requestAnimationFrame(update);






                }} else {{






                    element.textContent = original;






                }}






            }}






            requestAnimationFrame(update);






        }}, delay);






    }}






    shuffleText(document.getElementById('logoText'), '⬡ CryptoMind', 100);






    shuffleText(document.getElementById('predText'), '{pred_arrow} {pred_word}', 500);






}})();






</script>






</body>






</html>






"""













# =============================================================================






# COMPONENT 2 — LETTER GLITCH SIGNAL CARD






# =============================================================================






def letter_glitch_signal_html(prediction):






    pred_word = prediction.get('prediction', 'NEUTRAL').upper() if prediction else 'NEUTRAL'






    pred_color = GREEN if pred_word == 'UP' else RED if pred_word == 'DOWN' else AMBER






    pred_arrow = '↑' if pred_word == 'UP' else '↓' if pred_word == 'DOWN' else '◈'






    confidence = prediction.get('confidence', 0.5) if prediction else 0.5






    probability = prediction.get('probability', 0.5) if prediction else 0.5






    price = prediction.get('current_price', 0) if prediction else 0













    bg_color = f"rgba(0,255,136,0.05)" if pred_word == 'UP' else f"rgba(255,68,102,0.05)" if pred_word == 'DOWN' else f"rgba(255,183,0,0.05)"






    katakana = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'













    return f"""






<!DOCTYPE html>






<html>






<head>






<meta charset="UTF-8">






<style>






@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;700;800&display=swap');






* {{ margin: 0; padding: 0; box-sizing: border-box; }}






body {{ background: #060C10; overflow: hidden; font-family: 'JetBrains Mono', monospace; }}






#glitchCanvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}






.signal-content {{






    position: absolute; top: 0; left: 0; width: 100%; height: 100%;






    z-index: 10; display: flex; flex-direction: column;






    align-items: center; justify-content: center; gap: 12px;






    background: {bg_color};






    border-top: 1px solid;






    border-image: linear-gradient(90deg, transparent, {pred_color}, transparent) 1;






}}






.arrow {{ 






    font-size: 64px; color: {pred_color}; 






    filter: drop-shadow(0 0 24px {pred_color});






    animation: float 3s ease-in-out infinite;






}}






@keyframes float {{






    0%, 100% {{ transform: translateY(0); }}






    50% {{ transform: translateY(-8px); }}






}}






.pred-word {{ 






    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 52px; 






    color: {pred_color}; text-shadow: 0 0 40px {pred_color}66;






}}






.subtitle {{ 






    font-size: 11px; color: #3A5060; letter-spacing: .12em; text-transform: uppercase;






}}






.metrics-row {{ display: flex; gap: 32px; margin-top: 8px; }}






.metric {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}






.metric-label {{ font-size: 9px; color: #3A5060; letter-spacing: .12em; text-transform: uppercase; }}






.metric-value {{ font-family: 'Syne', sans-serif; font-weight: 800; font-size: 18px; color: {pred_color}; }}






.confidence-bar {{ 






    width: 120px; height: 5px; background: rgba(0,255,136,0.1); 






    border-radius: 3px; overflow: hidden; margin-top: 2px;






}}






.confidence-fill {{ 






    height: 100%; width: {confidence*100}%; background: {pred_color};






    box-shadow: 0 0 10px {pred_color}; border-radius: 3px;






    transition: width 1s ease;






}}






.live-price {{ font-size: 13px; color: #7A9BAD; margin-top: 4px; }}






</style>






</head>






<body>






<canvas id="glitchCanvas"></canvas>






<div class="signal-content">






    <div class="arrow">{pred_arrow}</div>






    <div class="pred-word">{pred_word}</div>






    <div class="subtitle">LSTM NEURAL NETWORK PREDICTION</div>






    <div class="metrics-row">






        <div class="metric">






            <div class="metric-label">Confidence</div>






            <div class="metric-value">{confidence:.1%}</div>






            <div class="confidence-bar"><div class="confidence-fill"></div></div>






        </div>






        <div class="metric">






            <div class="metric-label">Probability</div>






            <div class="metric-value" style="color:#C8DDE8">{probability:.1%}</div>






        </div>






    </div>






    <div class="live-price">Current Price: ${price:,.2f}</div>






</div>






<script>






(function() {{






    const canvas = document.getElementById('glitchCanvas');






    const ctx = canvas.getContext('2d');






    let width, height, cells = [];






    const fontSize = 13;






    const katakana = '{katakana}';






    const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*⬡◈↑↓';






    const allChars = katakana + latin;






    const color = '{pred_color}';






    let lastGlitch = 0;






    const glitchSpeed = 50;













    function resize() {{






        width = canvas.width = window.innerWidth;






        height = canvas.height = window.innerHeight;






        initCells();






    }}













    function initCells() {{






        cells = [];






        const cols = Math.floor(width / fontSize);






        const rows = Math.floor(height / fontSize);






        const cx = cols / 2, cy = rows / 2;






        for (let r = 0; r < rows; r++) {{






            for (let c = 0; c < cols; c++) {{






                const dx = (c - cx) / cx;






                const dy = (r - cy) / cy;






                const dist = Math.sqrt(dx*dx + dy*dy);






                cells.push({{






                    x: c * fontSize, y: r * fontSize,






                    char: allChars[Math.floor(Math.random() * allChars.length)],






                    opacity: Math.random() * 0.58 + 0.02,






                    targetOpacity: Math.random() * 0.58 + 0.02,






                    speed: Math.random() * 0.03 + 0.01,






                    changing: Math.random() < 0.7,






                    dist: dist






                }});






            }}






        }}






    }}













    function animate(now) {{






        requestAnimationFrame(animate);






        ctx.clearRect(0, 0, width, height);






        ctx.font = fontSize + 'px "JetBrains Mono", monospace';






        ctx.textBaseline = 'top';






        const doGlitch = now - lastGlitch > glitchSpeed;






        if (doGlitch) lastGlitch = now;






        for (let cell of cells) {{






            cell.opacity += (cell.targetOpacity - cell.opacity) * cell.speed;






            if (Math.abs(cell.opacity - cell.targetOpacity) < 0.01) {{






                cell.targetOpacity = Math.random() * 0.58 + 0.02;






            }}






            let finalOpacity = cell.opacity * Math.max(0, 1 - cell.dist * 1.2);






            if (doGlitch && cell.changing && Math.random() < 0.15) {{






                cell.char = allChars[Math.floor(Math.random() * allChars.length)];






            }}






            ctx.fillStyle = color;






            ctx.globalAlpha = finalOpacity;






            ctx.fillText(cell.char, cell.x, cell.y);






        }}






        ctx.globalAlpha = 1;






    }}






    window.addEventListener('resize', resize);






    resize();






    requestAnimationFrame(animate);






}})();






</script>






</body>






</html>






"""













# =============================================================================






# COMPONENT 3 — GOOEY NAV COIN SELECTOR






# =============================================================================






def gooey_nav_html(active_coin):






    is_btc = active_coin.lower() == 'bitcoin' or active_coin.lower() == 'btc'






    active_index = 0 if is_btc else 1













    return f"""






<!DOCTYPE html>






<html>






<head>






<meta charset="UTF-8">






<style>






@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;700;800&display=swap');






* {{ margin: 0; padding: 0; box-sizing: border-box; }}






body {{ 






    background: #060C10; display: flex; align-items: center; justify-content: center;






    height: 88px; overflow: hidden; font-family: 'JetBrains Mono', monospace;






}}






.nav-container {{






    position: relative; display: flex; align-items: center;






    background: #0B1319; border: 0.5px solid rgba(0,255,136,0.12);






    border-radius: 14px; padding: 6px; filter: url('#goo');






}}






.nav-item {{






    position: relative; z-index: 2; padding: 14px 36px;






    cursor: pointer; text-align: center; min-width: 140px;






    transition: color 0.3s ease;






}}






.nav-item .ticker {{






    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 20px;






    color: #3A5060; transition: color 0.3s ease;






}}






.nav-item .label {{






    font-size: 9px; color: #3A5060; opacity: 0.6;






    margin-top: 2px; transition: color 0.3s ease;






}}






.nav-item.active .ticker {{ color: #000; }}






.nav-item.active .label {{ color: rgba(0,0,0,0.6); }}






.nav-item:not(.active):hover .ticker {{ color: #C8DDE8; }}






.divider {{






    width: 0.5px; height: 32px; background: rgba(0,255,136,0.12);






    position: relative; z-index: 2;






}}






.blob {{






    position: absolute; top: 6px; bottom: 6px;






    background: #00FF88; border-radius: 10px;






    transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);






    z-index: 1;






}}






.particle {{






    position: absolute; width: 6px; height: 6px; border-radius: 50%;






    pointer-events: none; z-index: 3;






}}






</style>






</head>






<body>






<svg style="position:absolute;width:0;height:0;">






    <defs>






        <filter id="goo">






            <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur"/>






            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -10" result="goo"/>






            <feComposite in="SourceGraphic" in2="goo" operator="atop"/>






        </filter>






    </defs>






</svg>






<div class="nav-container" id="navContainer">






    <div class="blob" id="blob"></div>






    <div class="nav-item {'active' if is_btc else ''}" data-index="0" onclick="selectItem(0)">






        <div class="ticker">BTC</div>






        <div class="label">Bitcoin</div>






    </div>






    <div class="divider"></div>






    <div class="nav-item {'active' if not is_btc else ''}" data-index="1" onclick="selectItem(1)">






        <div class="ticker">ETH</div>






        <div class="label">Ethereum</div>






    </div>






</div>






<script>






(function() {{






    const blob = document.getElementById('blob');






    const container = document.getElementById('navContainer');






    const items = document.querySelectorAll('.nav-item');






    const colors = ['#00FF88', '#4DC3FF', '#FFB700', '#00CC6A'];






    let currentIndex = {active_index};













    function updateBlob(index) {{






        const item = items[index];






        const rect = item.getBoundingClientRect();






        const containerRect = container.getBoundingClientRect();






        blob.style.left = (rect.left - containerRect.left) + 'px';






        blob.style.width = rect.width + 'px';






    }}













    function spawnParticles(targetIndex) {{






        const item = items[targetIndex];






        const rect = item.getBoundingClientRect();






        const containerRect = container.getBoundingClientRect();






        const cx = rect.left - containerRect.left + rect.width / 2;






        const cy = rect.top - containerRect.top + rect.height / 2;






        for (let i = 0; i < 15; i++) {{






            const p = document.createElement('div');






            p.className = 'particle';






            p.style.background = colors[Math.floor(Math.random() * colors.length)];






            p.style.left = cx + 'px';






            p.style.top = cy + 'px';






            container.appendChild(p);






            const angle = Math.random() * Math.PI * 2;






            const dist = 20 + Math.random() * 70;






            const tx = Math.cos(angle) * dist;






            const ty = Math.sin(angle) * dist;






            const duration = 400 + Math.random() * 300;






            p.animate([






                {{ transform: 'translate(0,0) scale(1)', opacity: 1 }},






                {{ transform: 'translate(' + tx + 'px,' + ty + 'px) scale(0)', opacity: 0 }}






            ], {{






                duration: duration,






                easing: 'cubic-bezier(0.34,1.56,0.64,1)'






            }}).onfinish = () => p.remove();






        }}






    }}













    function selectItem(index) {{






        if (index === currentIndex) return;






        items[currentIndex].classList.remove('active');






        items[index].classList.add('active');






        currentIndex = index;






        updateBlob(index);






        spawnParticles(index);






    }}













    updateBlob(currentIndex);






}})();






</script>






</body>






</html>






"""













# =============================================================================






# COMPONENT 4 — SHUFFLE METRICS ROW






# =============================================================================






def shuffle_metrics_html(metrics):






    metrics_json = json.dumps(metrics)






    return f"""






<!DOCTYPE html>






<html>






<head>






<meta charset="UTF-8">






<style>






@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Syne:wght@400;700;800&display=swap');






* {{ margin: 0; padding: 0; box-sizing: border-box; }}






body {{ 






    background: #060C10; display: flex; align-items: center;






    min-height: 72px; padding: 0 4px; font-family: 'JetBrains Mono', monospace;






}}






.metrics-container {{






    display: flex; width: 100%; gap: 1px;






    background: rgba(0,255,136,0.12); border-radius: 12px;






    overflow: hidden; border: 0.5px solid rgba(0,255,136,0.12);






}}






.metric-cell {{






    flex: 1; background: #0B1319; padding: 12px 16px;






    display: flex; flex-direction: column; gap: 4px;






    position: relative; overflow: hidden;






}}






.metric-cell::before {{






    content: ''; position: absolute; top: 0; left: 0; right: 0;






    height: 1px;






    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), transparent);






}}






.metric-label {{






    font-size: 9px; font-weight: 600; letter-spacing: .12em;






    text-transform: uppercase; color: #3A5060;






}}






.metric-value {{






    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px;






}}






</style>






</head>






<body>






<div class="metrics-container" id="metricsContainer"></div>






<script>






(function() {{






    const metrics = {metrics_json};






    const container = document.getElementById('metricsContainer');






    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%.+−';













    metrics.forEach((m, idx) => {{






        const cell = document.createElement('div');






        cell.className = 'metric-cell';






        cell.innerHTML = '<div class="metric-label">' + m[0] + '</div><div class="metric-value" style="color:' + m[2] + '" data-final="' + m[1] + '" data-idx="' + idx + '"></div>';






        container.appendChild(cell);






    }});













    const valueEls = document.querySelectorAll('.metric-value');






    function shuffleElement(el, finalText, delay) {{






        setTimeout(() => {{






            const original = finalText;






            const charArray = original.split('');






            const totalChars = charArray.length;






            let startTime = performance.now();






            const duration = 700;






            const staggerMs = 35;






            function update(now) {{






                const elapsed = now - startTime;






                let result = '';






                let allDone = true;






                for (let i = 0; i < totalChars; i++) {{






                    const charDelay = i * staggerMs;






                    const charElapsed = elapsed - charDelay;






                    if (charArray[i] === ' ' || charArray[i] === '$' || charArray[i] === '%' || charArray[i] === '.' || charArray[i] === ',') {{






                        result += charArray[i];






                        continue;






                    }}






                    if (charElapsed < 0) {{






                        result += chars[Math.floor(Math.random() * chars.length)];






                        allDone = false;






                    }} else if (charElapsed < duration) {{






                        const progress = charElapsed / duration;






                        if (Math.random() > progress) {{






                            result += chars[Math.floor(Math.random() * chars.length)];






                        }} else {{






                            result += charArray[i];






                        }}






                        allDone = false;






                    }} else {{






                        result += charArray[i];






                    }}






                }}






                el.textContent = result;






                if (!allDone) {{






                    requestAnimationFrame(update);






                }} else {{






                    el.textContent = original;






                }}






            }}






            requestAnimationFrame(update);






        }}, delay);






    }}






    valueEls.forEach((el, idx) => {{






        shuffleElement(el, el.dataset.final, idx * 80);






    }});






}})();






</script>






</body>






</html>






"""













# =============================================================================






# SIDEBAR






# =============================================================================






with st.sidebar:






    st.components.v1.html("""






    <div style="margin-bottom: 24px;">






        <div style="font-family: 'Syne', sans-serif; font-weight: 800; font-size: 24px; color: white;">






            ⬡ CryptoMind






        </div>






        <div style="font-size: 10px; color: #3A5060; letter-spacing: .1em; text-transform: uppercase; margin-top: 4px;">






            LSTM Prediction Terminal






        </div>






    </div>






    """, height=100, scrolling=False)













    coin = st.selectbox(






        "Select Asset",






        ["bitcoin", "ethereum"],






        format_func=lambda x: "₿ Bitcoin (BTC)" if x == "bitcoin" else "Ξ Ethereum (ETH)",






        index=0






    )













    if st.session_state.last_coin != coin:






        st.session_state.signal_history = []






        st.session_state.last_coin = coin













    confidence_threshold = st.slider(






        "Confidence Threshold", 






        min_value=0.50, max_value=0.90, 






        value=0.60, step=0.05,






        format="%.2f"






    )













    data_window = st.slider(






        "Data Window (hours)", 






        min_value=24, max_value=3744, 






        value=3744, step=12






    )













    auto_refresh = st.toggle("Auto Refresh", value=True)






    refresh_interval = st.slider(






        "Refresh Interval (s)", 






        min_value=15, max_value=120, 






        value=30, step=15






    )













    st.components.v1.html('<div class="section-label">Pipeline Status</div>')













    if BACKEND_AVAILABLE:






        try:






            db = CryptoDatabase()






            recent = db.get_recent_prices(coin, hours=1)






            db_count = len(recent) if recent is not None else 0






            db_status = "ok" if db_count > 0 else "warn"






            db_color = GREEN if db_count > 0 else AMBER






        except Exception:






            db_count = 0






            db_status = "err"






            db_color = RED






    else:






        db_count = 0






        db_status = "err"






        db_color = RED













    st.components.v1.html(f"""






    <div style="display:flex; align-items:center; margin-bottom:8px;">






        <span class="status-dot status-{db_status}"></span>






        <span style="font-size:11px; color:{db_color};">Database: {db_count} recent records</span>






    </div>






    """, height=100, scrolling=False)













    if BACKEND_AVAILABLE:






        try:






            mm = ModelManager()






            models = mm.list_models()






            model_count = len(models) if models else 0






            model_status = "ok" if model_count > 0 else "warn"






            model_color = GREEN if model_count > 0 else AMBER






        except Exception:






            model_count = 0






            model_status = "err"






            model_color = RED






    else:






        model_count = 0






        model_status = "err"






        model_color = RED













    st.components.v1.html(f"""






    <div style="display:flex; align-items:center; margin-bottom:8px;">






        <span class="status-dot status-{model_status}"></span>






        <span style="font-size:11px; color:{model_color};">Models: {model_count} available</span>






    </div>






    """, height=100, scrolling=False)













    # AI Status






    _ensure_ai()






    ai_status = "ok" if _AI_AVAILABLE and _ai_explainer else "warn"






    ai_color = GREEN if ai_status == "ok" else AMBER






    st.components.v1.html(f"""






    <div style="display:flex; align-items:center; margin-bottom:16px;">






        <span class="status-dot status-{ai_status}"></span>






        <span style="font-size:11px; color:{ai_color};">AI Layer: {'Ready' if ai_status == 'ok' else 'Unavailable'}</span>






    </div>






    """, height=100, scrolling=False)













    # ─── CHAT SIDEBAR ───






    st.components.v1.html('<div class="section-label">💬 AI Chat</div>')






    






    if not BACKEND_AVAILABLE:






        st.info("Backend not available for chat.")






    elif not (_AI_AVAILABLE and _ai_reporter):






        st.info("AI chat not available.")






    else:






        for msg in st.session_state.chat_history:






            if msg['role'] == 'user':






                st.components.v1.html(f'<div class="chat-bubble-user">{msg["content"]}</div>')






            else:






                st.components.v1.html(f'<div class="chat-bubble-ai">{msg["content"]}</div>')






        






        chat_query = st.text_input("Ask about market data...", key="chat_input", placeholder="e.g., Why is RSI high?")






        if st.button("Send", key="chat_send", use_container_width=True):






            if chat_query.strip():






                st.session_state.chat_history.append({"role": "user", "content": chat_query})






                try:






                    indicators = None






                    pred_for_chat = locals().get('prediction') or globals().get('prediction')






                    if pred_for_chat and 'indicators' in pred_for_chat:






                        indicators = {'coin': coin, 'timestamp': pred_for_chat.get('timestamp'), 'indicators': pred_for_chat['indicators']}






                    






                    answer = _ai_chat.ask(






                        query=chat_query,






                        coin_id=coin,






                        context_hours=24,






                        db=CryptoDatabase(settings.DB_PATH) if BACKEND_AVAILABLE else None,






                    )






                    answer_text = answer.get('response', str(answer)) if isinstance(answer, dict) else str(answer)






                    st.session_state.chat_history.append({"role": "ai", "content": answer_text})






                except Exception as e:






                    st.session_state.chat_history.append({"role": "ai", "content": f"Error: {str(e)}"})






                st.rerun()






        






        if st.button("Clear Chat", key="clear_chat", use_container_width=True):






            st.session_state.chat_history = []






            st.rerun()













    st.components.v1.html(f"""






    <div style="font-size:10px; color:#3A5060; margin-top:24px; border-top:0.5px solid rgba(0,255,136,0.07); padding-top:12px;">






        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}






    </div>






    """, height=100, scrolling=False)













    if st.button("🔄 Refresh Data", use_container_width=True):






        st.cache_data.clear()






        st.rerun()













# =============================================================================






# LOAD DATA






# =============================================================================






price_df = load_price_data(coin, data_window) if BACKEND_AVAILABLE else None






indicators_df = load_indicators(coin, data_window) if BACKEND_AVAILABLE else None






prediction = get_prediction(coin) if BACKEND_AVAILABLE else None






backtest = get_backtest_results(coin, confidence_threshold) if BACKEND_AVAILABLE else None






models_info = get_model_info() if BACKEND_AVAILABLE else None






drift_data = get_drift_data(coin) if BACKEND_AVAILABLE else None













current_price = prediction.get('current_price', 0) if prediction else (






    price_df['price_usd'].iloc[-1] if price_df is not None and not price_df.empty and 'price_usd' in price_df.columns else 0






)













if prediction and BACKEND_AVAILABLE:






    signal_entry = {






        'time': datetime.now().strftime('%H:%M:%S'),






        'price': current_price,






        'direction': prediction.get('prediction', 'NEUTRAL').upper(),






        'confidence': prediction.get('confidence', 0)






    }






    if not st.session_state.signal_history or st.session_state.signal_history[-1]['time'] != signal_entry['time']:






        st.session_state.signal_history.append(signal_entry)






        if len(st.session_state.signal_history) > 50:






            st.session_state.signal_history = st.session_state.signal_history[-50:]













# =============================================================================






# TOP COMPONENTS (above tabs)






# =============================================================================






st.components.v1.html(waves_header_html(coin, current_price, prediction), height=170, scrolling=False)






st.components.v1.html(gooey_nav_html(coin), height=88, scrolling=False)






# ─── DRIFT ALERT BANNER ───






if drift_data and drift_data.get('results'):






    results = drift_data['results']






    if results.get('drift_detected'):






        drift_pct = results.get('drift_percentage', 0)






        if drift_pct > 50:






            alert_class = "drift-alert-critical"






            alert_color = RED






            alert_icon = "🔴"






            alert_text = f"CRITICAL DRIFT: {len(results.get('drifted_features', []))} features drifted ({drift_pct:.0f}%). Retrain immediately."






        elif drift_pct > 20:






            alert_class = "drift-alert-warning"






            alert_color = AMBER






            alert_icon = "🟡"






            alert_text = f"WARNING: {len(results.get('drifted_features', []))} features showing drift ({drift_pct:.0f}%). Schedule retraining."






        else:






            alert_class = "drift-alert-warning"






            alert_color = AMBER






            alert_icon = "🟡"






            alert_text = f"CAUTION: Minor drift detected on {len(results.get('drifted_features', []))} features. Monitor closely."






        






        st.components.v1.html(f"""






        <div class="{alert_class}">






            <div style="display:flex; align-items:center; gap:10px;">






                <span style="font-size:18px;">{alert_icon}</span>






                <div>






                    <div style="font-size:11px; color:{alert_color}; font-weight:700; letter-spacing:.1em; text-transform:uppercase;">Drift Alert</div>






                    <div style="font-size:13px; color:#C8DDE8; margin-top:2px;">{alert_text}</div>






                </div>






            </div>






        </div>






        """, height=100, scrolling=False)






    else:






        st.components.v1.html(f"""






        <div class="drift-alert-ok">






            <div style="display:flex; align-items:center; gap:10px;">






                <span style="font-size:18px;">🟢</span>






                <div>






                    <div style="font-size:11px; color:{GREEN}; font-weight:700; letter-spacing:.1em; text-transform:uppercase;">Data Distribution Stable</div>






                    <div style="font-size:13px; color:#C8DDE8; margin-top:2px;">No significant drift detected. Model inputs remain within expected ranges.</div>






                </div>






            </div>






        </div>






        """, height=100, scrolling=False)













# =============================================================================






# TABS






# =============================================================================






tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([






    "⬡ Live Signal", 






    "📈 Price Chart", 






    "📊 Indicators", 






    "📉 Backtest", 






    "🤖 Models",






    "🧠 AI Briefing",






    "🔍 Drift Monitor"






])













# -----------------------------------------------------------------------------






# TAB 1 — LIVE SIGNAL






# -----------------------------------------------------------------------------






with tab1:






    if not BACKEND_AVAILABLE:






        st.warning("Backend modules not available. Please ensure src/ modules are properly configured.")






    elif prediction is None:






        st.info("No prediction available. Please check that models are trained and data is available.")






    else:






        st.components.v1.html(letter_glitch_signal_html(prediction), height=300, scrolling=False)













        if indicators_df is not None and not indicators_df.empty:






            latest = indicators_df.iloc[-1]






            rsi_val = latest.get('rsi', 0)






            bb_pct = latest.get('bb_percent', 0) if 'bb_percent' in latest else latest.get('bb_pct', 0)






            macd_val = latest.get('macd_histogram', 0) if 'macd_histogram' in latest else latest.get('macd', 0)






            vol_val = latest.get('volatility', 0)













            metrics = [






                ("RSI", f"{rsi_val:.1f}", GREEN if rsi_val < 30 else RED if rsi_val > 70 else AMBER),






                ("Confidence", f"{prediction.get('confidence', 0):.1%}", GREEN if prediction.get('confidence', 0) > 0.6 else AMBER),






                ("Probability", f"{prediction.get('probability', 0):.1%}", BLUE),






                ("BB %B", f"{bb_pct:.2f}", GREEN if bb_pct < 0.2 else RED if bb_pct > 0.8 else AMBER),






                ("Volatility", f"{vol_val:.2f}%", AMBER),






                ("MACD Hist", f"{macd_val:.4f}", GREEN if macd_val > 0 else RED)






            ]






            st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)













        # ─── AI EXPLANATION CARD ───






        _ensure_ai()






        if _ai_explainer and prediction:






            with st.spinner("Generating AI explanation..."):






                try:






                    indicators = prediction.get('indicators', {})






                    explanation = _ai_explainer.explain(






                        coin_id=coin,






                        prediction=prediction,






                        indicators={'coin': coin, 'timestamp': prediction.get('timestamp'), 'indicators': indicators},






                    )






                    exp_dict = explanation.dict() if hasattr(explanation, 'dict') else explanation






                    summary = exp_dict.get('summary', str(exp_dict)) if isinstance(exp_dict, dict) else str(exp_dict)






                    






                    st.components.v1.html(f"""






                    <div class="ai-card">






                        <div class="ai-card-title">🧠 AI Explanation</div>






                        <div class="ai-card-body">{summary}</div>






                    </div>






                    """, height=150, scrolling=False)






                except Exception as e:






                    st.warning(f"AI explanation failed: {e}")













        col1, col2 = st.columns([1, 1])













        with col1:






            st.components.v1.html('<div class="section-label">Technical Indicators</div>')













            if indicators_df is not None and not indicators_df.empty:






                latest = indicators_df.iloc[-1]













                rsi = latest.get('rsi', 50)






                rsi_color = GREEN if rsi < 30 else RED if rsi > 70 else AMBER






                rsi_badge = '<span class="badge-up">OVERSOLD</span>' if rsi < 30 else '<span class="badge-down">OVERBOUGHT</span>' if rsi > 70 else '<span class="badge-neutral">NEUTRAL</span>'






                st.components.v1.html(f"""






                <div class="indicator-row">






                    <div style="display:flex; align-items:center; gap:12px;">






                        <span style="font-size:16px;">📊</span>






                        <div>






                            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">RSI (14)</div>






                            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{rsi_color};">{rsi:.1f}</div>






                        </div>






                    </div>






                    {rsi_badge}






                </div>






                """, height=80, scrolling=False)













                macd = latest.get('macd_histogram', 0) if 'macd_histogram' in latest else latest.get('macd', 0)






                macd_color = GREEN if macd > 0 else RED






                macd_badge = '<span class="badge-up">BULLISH</span>' if macd > 0 else '<span class="badge-down">BEARISH</span>'






                st.components.v1.html(f"""






                <div class="indicator-row">






                    <div style="display:flex; align-items:center; gap:12px;">






                        <span style="font-size:16px;">📈</span>






                        <div>






                            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">MACD Histogram</div>






                            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{macd_color};">{macd:.4f}</div>






                        </div>






                    </div>






                    {macd_badge}






                </div>






                """, height=80, scrolling=False)













                bb = latest.get('bb_percent', 0.5) if 'bb_percent' in latest else latest.get('bb_pct', 0.5)






                bb_color = GREEN if bb < 0.2 else RED if bb > 0.8 else AMBER






                bb_badge = '<span class="badge-up">LOW BAND</span>' if bb < 0.2 else '<span class="badge-down">HIGH BAND</span>' if bb > 0.8 else '<span class="badge-neutral">MID BAND</span>'






                st.components.v1.html(f"""






                <div class="indicator-row">






                    <div style="display:flex; align-items:center; gap:12px;">






                        <span style="font-size:16px;">〰️</span>






                        <div>






                            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Bollinger %B</div>






                            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{bb_color};">{bb:.2f}</div>






                        </div>






                    </div>






                    {bb_badge}






                </div>






                """, height=80, scrolling=False)













                vol = latest.get('volatility', 0)






                vol_color = RED if vol > 5 else AMBER if vol > 2 else GREEN






                vol_badge = '<span class="badge-down">HIGH</span>' if vol > 5 else '<span class="badge-neutral">ELEVATED</span>' if vol > 2 else '<span class="badge-up">LOW</span>'






                st.components.v1.html(f"""






                <div class="indicator-row">






                    <div style="display:flex; align-items:center; gap:12px;">






                        <span style="font-size:16px;">⚡</span>






                        <div>






                            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Volatility</div>






                            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{vol_color};">{vol:.2f}%</div>






                        </div>






                    </div>






                    {vol_badge}






                </div>






                """, height=80, scrolling=False)













        with col2:






            st.components.v1.html('<div class="section-label">Signal History</div>')













            history = st.session_state.signal_history[-10:][::-1]






            for sig in history:






                arrow = '↑' if sig['direction'] == 'UP' else '↓' if sig['direction'] == 'DOWN' else '◈'






                color = GREEN if sig['direction'] == 'UP' else RED if sig['direction'] == 'DOWN' else AMBER






                st.components.v1.html(f"""






                <div class="signal-history-row">






                    <span style="color:#7A9BAD; font-size:11px;">{sig['time']}</span>






                    <span style="color:#C8DDE8; font-family:'Syne',sans-serif; font-weight:800;">${sig['price']:,.2f}</span>






                    <span style="color:{color}; font-weight:700;">{arrow} {sig['direction']}</span>






                    <span style="color:#3A5060; font-size:11px;">{sig['confidence']:.1%}</span>






                </div>






                """, height=50, scrolling=False)













            st.components.v1.html('<div class="section-label" style="margin-top:16px;">Session Stats</div>')













            total = len(st.session_state.signal_history)






            bull_count = sum(1 for s in st.session_state.signal_history if s['direction'] == 'UP')






            bear_count = sum(1 for s in st.session_state.signal_history if s['direction'] == 'DOWN')






            avg_conf = np.mean([s['confidence'] for s in st.session_state.signal_history]) if st.session_state.signal_history else 0













            c1, c2, c3, c4 = st.columns(4)






            with c1:






                st.metric("Total Signals", total)






            with c2:






                st.metric("Avg Confidence", f"{avg_conf:.1%}")






            with c3:






                st.metric("Bull Count", bull_count, delta_color="off")






            with c4:






                st.metric("Bear Count", bear_count, delta_color="off")













# -----------------------------------------------------------------------------






# TAB 2 — PRICE CHART






# -----------------------------------------------------------------------------






with tab2:






    if price_df is not None and not price_df.empty:






        current_price = prediction.get('current_price', 0) if prediction else (






    price_df['price_usd'].iloc[-1] if price_df is not None and not price_df.empty and 'price_usd' in price_df.columns else (






        price_df['close'].iloc[-1] if price_df is not None and not price_df.empty and 'close' in price_df.columns else 0






    )






)






       






        






        change_1h = 0






        if len(price_df) > 1:






            prev = price_df['price_usd'].iloc[-2] if 'price_usd' in price_df.columns else price_df['close'].iloc[-2]






            change_1h = ((current_price / prev) - 1) * 100






        high = price_df['price_usd'].max() if 'price_usd' in price_df.columns else price_df['close'].max()






        low = price_df['price_usd'].min() if 'price_usd' in price_df.columns else price_df['close'].min()






        records = len(price_df)













        metrics = [






            ("Current", f"${current_price:,.2f}", GREEN if change_1h >= 0 else RED),






            ("1h Change", f"{change_1h:+.2f}%", GREEN if change_1h >= 0 else RED),






            ("High", f"${high:,.2f}", GREEN),






            ("Low", f"${low:,.2f}", RED),






            ("Records", str(records), BLUE)






        ]






        st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)













        fig = go.Figure()













        if indicators_df is not None and not indicators_df.empty:






            fig.add_trace(go.Scatter(






                x=indicators_df.index,






                y=indicators_df['bb_upper'],






                mode='lines',






                line=dict(color=AMBER, width=1, dash='dash'),






                name='BB Upper',






                showlegend=True






            ))






            fig.add_trace(go.Scatter(






                x=indicators_df.index,






                y=indicators_df['bb_lower'],






                mode='lines',






                line=dict(color=AMBER, width=1, dash='dash'),






                name='BB Lower',






                fill='tonexty',






                fillcolor='rgba(255,183,0,0.08)',






                showlegend=True






            ))













        price_col = 'price_usd' if 'price_usd' in price_df.columns else 'close'






        fig.add_trace(go.Scatter(






            x=price_df.index,






            y=price_df[price_col],






            mode='lines',






            line=dict(color=GREEN if change_1h >= 0 else RED, width=1.5),






            name='Price',






            fill='tozeroy',






            fillcolor=f"rgba({0 if change_1h >= 0 else 255}, {255 if change_1h >= 0 else 68}, {136 if change_1h >= 0 else 102}, 0.1)"






        ))













        fig.update_layout(






            paper_bgcolor=BG_CARD,






            plot_bgcolor=BG_CARD,






            font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






            xaxis=dict(gridcolor='rgba(0,255,136,0.05)', showgrid=True),






            yaxis=dict(gridcolor='rgba(0,255,136,0.05)', showgrid=True),






            hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






            margin=dict(l=10, r=10, t=36, b=10),






            title=dict(text=f"{coin.upper()}/USD Price", font=dict(size=14, color=TEXT_PRIMARY)),






            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10))






        )













        st.plotly_chart(fig, use_container_width=True)













        if 'volume_24h' in price_df.columns:






            vol_fig = go.Figure()






            vol_fig.add_trace(go.Bar(






                x=price_df.index,






                y=price_df['volume_24h'],






                marker_color='rgba(77,195,255,0.35)',






                name='Volume'






            ))






            vol_fig.update_layout(






                paper_bgcolor=BG_CARD,






                plot_bgcolor=BG_CARD,






                font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






                xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                yaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






                margin=dict(l=10, r=10, t=36, b=10),






                title=dict(text="24h Volume", font=dict(size=14, color=TEXT_PRIMARY)),






                height=160






            )






            st.plotly_chart(vol_fig, use_container_width=True)






    else:






        st.warning("No price data available. Please check database connection.")













# -----------------------------------------------------------------------------






# TAB 3 — INDICATORS






# -----------------------------------------------------------------------------






with tab3:






    if indicators_df is not None and not indicators_df.empty:






        latest = indicators_df.iloc[-1]






        rsi_val = latest.get('rsi', 0)






        macd_hist = latest.get('macd_histogram', 0) if 'macd_histogram' in latest else latest.get('macd_hist', 0)






        bb_pct = latest.get('bb_percent', 0) if 'bb_percent' in latest else latest.get('bb_pct', 0)






        vol_val = latest.get('volatility', 0)






        delta_1h = 0






        if price_df is not None and len(price_df) > 1:






            price_col = 'price_usd' if 'price_usd' in price_df.columns else 'close'






            delta_1h = ((price_df[price_col].iloc[-1] / price_df[price_col].iloc[-2]) - 1) * 100













        metrics = [






            ("RSI Latest", f"{rsi_val:.1f}", GREEN if rsi_val < 30 else RED if rsi_val > 70 else AMBER),






            ("MACD Hist", f"{macd_hist:.4f}", GREEN if macd_hist > 0 else RED),






            ("BB %B", f"{bb_pct:.2f}", GREEN if bb_pct < 0.2 else RED if bb_pct > 0.8 else AMBER),






            ("Volatility", f"{vol_val:.2f}%", AMBER),






            ("1h Delta", f"{delta_1h:+.2f}%", GREEN if delta_1h >= 0 else RED)






        ]






        st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)













        col1, col2 = st.columns(2)













        with col1:






            rsi_fig = go.Figure()






            rsi_fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="rgba(255,68,102,0.1)")






            rsi_fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="rgba(0,255,136,0.1)")













            rsi_fig.add_trace(go.Scatter(






                x=indicators_df.index,






                y=indicators_df['rsi'],






                mode='lines',






                line=dict(color=BLUE, width=1.5),






                name='RSI'






            ))













            extreme_mask = (indicators_df['rsi'] > 70) | (indicators_df['rsi'] < 30)






            if extreme_mask.any():






                rsi_fig.add_trace(go.Scatter(






                    x=indicators_df.index[extreme_mask],






                    y=indicators_df['rsi'][extreme_mask],






                    mode='markers',






                    marker=dict(color=RED, size=6),






                    name='Extreme'






                ))













            rsi_fig.add_hline(y=30, line_dash="dash", line_color=GREEN, line_width=1, opacity=0.5)






            rsi_fig.add_hline(y=50, line_dash="dash", line_color=TEXT_MUTED, line_width=1, opacity=0.3)






            rsi_fig.add_hline(y=70, line_dash="dash", line_color=RED, line_width=1, opacity=0.5)













            rsi_fig.update_layout(






                paper_bgcolor=BG_CARD,






                plot_bgcolor=BG_CARD,






                font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






                xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                yaxis=dict(gridcolor='rgba(0,255,136,0.05)', range=[0, 100]),






                hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






                margin=dict(l=10, r=10, t=36, b=10),






                title=dict(text="RSI (14)", font=dict(size=14, color=TEXT_PRIMARY)),






                showlegend=False






            )






            st.plotly_chart(rsi_fig, use_container_width=True)













        with col2:






            macd_fig = go.Figure()













            macd_hist_col = 'macd_histogram' if 'macd_histogram' in indicators_df.columns else 'macd_hist'






            colors = [GREEN if h >= 0 else RED for h in indicators_df[macd_hist_col]]






            macd_fig.add_trace(go.Bar(






                x=indicators_df.index,






                y=indicators_df[macd_hist_col],






                marker_color=colors,






                name='Histogram',






                opacity=0.7






            ))













            macd_line_col = 'macd_line' if 'macd_line' in indicators_df.columns else 'macd'






            macd_fig.add_trace(go.Scatter(






                x=indicators_df.index,






                y=indicators_df[macd_line_col],






                mode='lines',






                line=dict(color=BLUE, width=1.5),






                name='MACD'






            ))













            signal_col = 'macd_signal' if 'macd_signal' in indicators_df.columns else 'macd_signal'






            if signal_col in indicators_df.columns:






                macd_fig.add_trace(go.Scatter(






                    x=indicators_df.index,






                    y=indicators_df[signal_col],






                    mode='lines',






                    line=dict(color=AMBER, width=1, dash='dash'),






                    name='Signal'






                ))













            macd_fig.update_layout(






                paper_bgcolor=BG_CARD,






                plot_bgcolor=BG_CARD,






                font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






                xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                yaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






                margin=dict(l=10, r=10, t=36, b=10),






                title=dict(text="MACD", font=dict(size=14, color=TEXT_PRIMARY)),






                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10))






            )






            st.plotly_chart(macd_fig, use_container_width=True)













        vol_fig = go.Figure()






        vol_fig.add_trace(go.Scatter(






            x=indicators_df.index,






            y=indicators_df['volatility'],






            mode='lines',






            line=dict(color=AMBER, width=1.5),






            fill='tozeroy',






            fillcolor='rgba(255,183,0,0.1)',






            name='Volatility'






        ))













        vol_fig.update_layout(






            paper_bgcolor=BG_CARD,






            plot_bgcolor=BG_CARD,






            font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






            xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






            yaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






            hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






            margin=dict(l=10, r=10, t=36, b=10),






            title=dict(text="Volatility", font=dict(size=14, color=TEXT_PRIMARY)),






            showlegend=False






        )






        st.plotly_chart(vol_fig, use_container_width=True)













        with st.expander("📋 Raw Indicator Data"):






            display_df = indicators_df.tail(10).copy()






            display_df = display_df.round(4)






            st.dataframe(display_df, use_container_width=True)






    else:






        st.warning("No indicator data available.")













# -----------------------------------------------------------------------------






# TAB 4 — BACKTEST (with classification metrics)






# -----------------------------------------------------------------------------






with tab4:






    if backtest is not None:






        trading = backtest.get('trading', {})






        classification = backtest.get('classification', {})






        trades = trading.get('trades', [])






        win_rate = trading.get('win_rate', 0)






        total_return = trading.get('total_return', 0)






        final_capital = trading.get('final_capital', 0)






        sharpe = trading.get('sharpe_ratio', 0)






        max_dd = trading.get('max_drawdown', 0)






        equity = trading.get('equity_curve', [])













        metrics = [






            ("Trades", str(trading.get('total_trades', 0)), BLUE),






            ("Win Rate", f"{win_rate:.1%}", GREEN if win_rate > 0.5 else RED),






            ("Return", f"{total_return:+.2f}%", GREEN if total_return > 0 else RED),






            ("Final Cap", f"${final_capital:,.2f}", GREEN if total_return > 0 else RED),






            ("Sharpe", f"{sharpe:.2f}", GREEN if sharpe > 1 else AMBER),






            ("Max DD", f"{max_dd:.2f}%", RED if max_dd > 20 else AMBER)






        ]






        st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)













        # ─── Classification Metrics Row ───






        if classification and 'error' not in classification:






            cls_metrics = [






                ("Accuracy", f"{classification.get('accuracy', 0):.1%}", GREEN if classification.get('accuracy', 0) > 0.55 else AMBER),






                ("Precision", f"{classification.get('precision', 0):.1%}", BLUE),






                ("Recall", f"{classification.get('recall', 0):.1%}", BLUE),






                ("F1 Score", f"{classification.get('f1_score', 0):.1%}", BLUE),






                ("Predictions", str(classification.get('total_predictions', 0)), TEXT_SECONDARY),






                ("Up/Down", f"{classification.get('actual_up_count', 0)}/{classification.get('actual_down_count', 0)}", TEXT_SECONDARY)






            ]






            st.components.v1.html(shuffle_metrics_html(cls_metrics), height=72, scrolling=False)













            # Confusion matrix display






            cm = classification.get('confusion_matrix')






            if cm:






                st.components.v1.html('<div class="section-label">Confusion Matrix</div>')






                cm_df = pd.DataFrame(






                    cm,






                    index=['Actual Down', 'Actual Up'],






                    columns=['Pred Down', 'Pred Up']






                )






                st.dataframe(cm_df, use_container_width=True)













        col1, col2 = st.columns([2, 1])













        with col1:






            if equity:






                eq_fig = go.Figure()






                eq_fig.add_trace(go.Scatter(






                    x=list(range(len(equity))),






                    y=equity,






                    mode='lines',






                    line=dict(color=GREEN if total_return > 0 else RED, width=1.5),






                    fill='tozeroy',






                    fillcolor=f"rgba({0 if total_return > 0 else 255}, {255 if total_return > 0 else 68}, {136 if total_return > 0 else 102}, 0.1)",






                    name='Equity'






                ))













                eq_fig.update_layout(






                    paper_bgcolor=BG_CARD,






                    plot_bgcolor=BG_CARD,






                    font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






                    xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                    yaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                    hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






                    margin=dict(l=10, r=10, t=36, b=10),






                    title=dict(text="Equity Curve", font=dict(size=14, color=TEXT_PRIMARY)),






                    showlegend=False






                )






                st.plotly_chart(eq_fig, use_container_width=True)













        with col2:






            st.components.v1.html('<div class="section-label">Recent Trades</div>')













            recent_trades = trades[-8:][::-1] if trades else []






            for trade in recent_trades:






                pnl = trade.get('pnl', 0) if 'pnl' in trade else trade.get('profit_pct', 0)






                is_pos = pnl > 0






                row_class = 'trade-row-pos' if is_pos else 'trade-row-neg'






                color = GREEN if is_pos else RED






                arrow = '↑' if is_pos else '↓'













                entry = trade.get('entry_price', 0)






                exit_p = trade.get('exit_price', 0)






                time_str = trade.get('time', trade.get('timestamp', 'N/A'))













                st.components.v1.html(f"""






                <div class="{row_class}">






                    <div style="display:flex; justify-content:space-between; align-items:center;">






                        <span style="font-size:11px; color:#7A9BAD;">{time_str}</span>






                        <span style="color:{color}; font-weight:700;">{arrow} {pnl:+.2f}%</span>






                    </div>






                    <div style="display:flex; justify-content:space-between; margin-top:4px;">






                        <span style="font-size:11px; color:#3A5060;">Entry: ${entry:,.2f}</span>






                        <span style="font-size:11px; color:#3A5060;">Exit: ${exit_p:,.2f}</span>






                    </div>






                </div>






                """, height=100, scrolling=False)













        # Strategy Diagnosis






        st.components.v1.html('<div class="section-label">Strategy Diagnosis</div>')













        acc = classification.get('accuracy', 0) if classification else 0






        diag1_color = GREEN if acc > 0.55 else AMBER if acc > 0.45 else RED






        diag1_text = "Model accuracy is strong — good predictive edge." if acc > 0.55 else "Accuracy is acceptable but monitor for degradation." if acc > 0.45 else "Accuracy is poor — consider retraining model."













        diag2_color = GREEN if sharpe > 1 else AMBER if sharpe > 0.5 else RED






        diag2_text = "Sharpe ratio indicates good risk-adjusted returns." if sharpe > 1 else "Sharpe ratio is moderate — review position sizing." if sharpe > 0.5 else "Sharpe ratio is low — high risk relative to return."













        diag3_color = RED if max_dd > 25 else AMBER if max_dd > 15 else GREEN






        diag3_text = "Maximum drawdown is within acceptable limits." if max_dd < 15 else "Drawdown is elevated — consider tighter stops." if max_dd < 25 else "Severe drawdown detected — risk management critical."













        st.components.v1.html(f"""






        <div style="display:flex; flex-direction:column; gap:8px;">






            <div style="display:flex; align-items:center; gap:8px; padding:10px 14px; background:#111C24; border-radius:8px; border:0.5px solid rgba(0,255,136,0.08);">






                <span class="status-dot status-{'ok' if acc > 0.55 else 'warn' if acc > 0.45 else 'err'}"></span>






                <span style="font-size:12px; color:{diag1_color};">{diag1_text}</span>






            </div>






            <div style="display:flex; align-items:center; gap:8px; padding:10px 14px; background:#111C24; border-radius:8px; border:0.5px solid rgba(0,255,136,0.08);">






                <span class="status-dot status-{'ok' if sharpe > 1 else 'warn' if sharpe > 0.5 else 'err'}"></span>






                <span style="font-size:12px; color:{diag2_color};">{diag2_text}</span>






            </div>






            <div style="display:flex; align-items:center; gap:8px; padding:10px 14px; background:#111C24; border-radius:8px; border:0.5px solid rgba(0,255,136,0.08);">






                <span class="status-dot status-{'err' if max_dd > 25 else 'warn' if max_dd > 15 else 'ok'}"></span>






                <span style="font-size:12px; color:{diag3_color};">{diag3_text}</span>






            </div>






        </div>






        """, height=100, scrolling=False)






    else:






        st.info("No backtest data available. Run backtester to see results.")













# -----------------------------------------------------------------------------






# TAB 5 — MODELS






# -----------------------------------------------------------------------------






with tab5:






    if models_info:






        best_model = models_info[0] if models_info else None






        val_acc = best_model.get('val_accuracy', 0) if best_model else 0






        seq_len = best_model.get('sequence_length', 60) if best_model else 60






        n_features = best_model.get('n_features', 10) if best_model else 10






        arch = best_model.get('architecture', 'LSTM') if best_model else 'LSTM'






        status = best_model.get('status', 'Unknown') if best_model else 'Unknown'













        metrics = [






            ("Val Acc", f"{val_acc:.1%}", GREEN if val_acc > 0.6 else AMBER),






            ("Seq Len", str(seq_len), BLUE),






            ("Features", str(n_features), BLUE),






            ("Architecture", arch, BLUE),






            ("Status", status, GREEN if status == 'Active' else AMBER)






        ]






        st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)













        col1, col2 = st.columns([1, 1])













        with col1:






            st.components.v1.html('<div class="section-label">Model Details</div>')













            if best_model:






                st.components.v1.html(f"""






                <div style="background:#111C24; border:0.5px solid rgba(0,255,136,0.08); border-radius:10px; padding:16px;">






                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">






                        <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Model Name</span>






                        <span style="font-size:12px; color:#C8DDE8;">{best_model.get('name', 'N/A')}</span>






                    </div>






                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">






                        <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Trained On</span>






                        <span style="font-size:12px; color:#C8DDE8;">{best_model.get('trained_on', 'N/A')}</span>






                    </div>






                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">






                        <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Epochs</span>






                        <span style="font-size:12px; color:#C8DDE8;">{best_model.get('epochs', 'N/A')}</span>






                    </div>






                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">






                        <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Batch Size</span>






                        <span style="font-size:12px; color:#C8DDE8;">{best_model.get('batch_size', 'N/A')}</span>






                    </div>






                    <div style="display:flex; justify-content:space-between;">






                        <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Last Updated</span>






                        <span style="font-size:12px; color:#C8DDE8;">{best_model.get('last_updated', 'N/A')}</span>






                    </div>






                </div>






                """, height=100, scrolling=False)













                acc_color = GREEN if val_acc > 0.7 else AMBER if val_acc > 0.55 else RED






                rating = "EXCELLENT" if val_acc > 0.75 else "GOOD" if val_acc > 0.65 else "FAIR" if val_acc > 0.55 else "POOR"






                rating_color = GREEN if val_acc > 0.75 else GREEN if val_acc > 0.65 else AMBER if val_acc > 0.55 else RED













                st.components.v1.html(f"""






                <div style="background:#0B1319; border:0.5px solid rgba(0,255,136,0.12); border-radius:12px; padding:24px; margin-top:16px; text-align:center;">






                    <div style="font-size:10px; color:#3A5060; text-transform:uppercase; letter-spacing:.12em; margin-bottom:8px;">Validation Accuracy</div>






                    <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:48px; color:{acc_color}; text-shadow: 0 0 40px {acc_color}44;">






                        {val_acc:.1%}






                    </div>






                    <div style="margin-top:12px;">






                        <span style="background: {rating_color}22; color:{rating_color}; border:0.5px solid {rating_color}55; border-radius:6px; padding:4px 12px; font-size:11px; font-weight:700;">






                            {rating}






                        </span>






                    </div>






                </div>






                """, height=100, scrolling=False)













        with col2:






            st.components.v1.html('<div class="section-label">Training History</div>')













            history = best_model.get('history', {}) if best_model else {}






            if history:






                epochs = list(range(1, len(history.get('train_acc', [])) + 1))






                train_acc = history.get('train_acc', [])






                val_acc_hist = history.get('val_acc', [])






                val_loss = history.get('val_loss', [])













                hist_fig = make_subplots(specs=[[{"secondary_y": True}]])













                hist_fig.add_trace(go.Scatter(






                    x=epochs, y=train_acc,






                    mode='lines',






                    line=dict(color=BLUE, width=1.5),






                    name='Train Acc'






                ), secondary_y=False)













                hist_fig.add_trace(go.Scatter(






                    x=epochs, y=val_acc_hist,






                    mode='lines',






                    line=dict(color=GREEN, width=1.5, dash='dash'),






                    name='Val Acc'






                ), secondary_y=False)













                hist_fig.add_trace(go.Scatter(






                    x=epochs, y=val_loss,






                    mode='lines',






                    line=dict(color=RED, width=1.5, dash='dash'),






                    name='Val Loss'






                ), secondary_y=True)













                hist_fig.update_layout(






                    paper_bgcolor=BG_CARD,






                    plot_bgcolor=BG_CARD,






                    font=dict(family='JetBrains Mono', color=TEXT_SECONDARY, size=11),






                    xaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                    yaxis=dict(gridcolor='rgba(0,255,136,0.05)'),






                    yaxis2=dict(gridcolor='rgba(0,255,136,0.05)'),






                    hoverlabel=dict(bgcolor=BG_CARD, bordercolor='rgba(0,255,136,0.3)'),






                    margin=dict(l=10, r=10, t=36, b=10),






                    title=dict(text="Training History", font=dict(size=14, color=TEXT_PRIMARY)),






                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10))






                )






                st.plotly_chart(hist_fig, use_container_width=True)













            st.components.v1.html('<div class="section-label">Training Diagnosis</div>')













            if history and len(history.get('val_acc', [])) > 5:






                recent_val = history['val_acc'][-5:]






                trend = "improving" if recent_val[-1] > recent_val[0] else "degrading"













                if trend == "improving":






                    diag_color = GREEN






                    diag_text = "Model is still learning — validation accuracy trending upward."






                else:






                    diag_color = AMBER






                    diag_text = "Model may be overfitting — consider early stopping or regularization."













                st.components.v1.html(f"""






                <div style="display:flex; align-items:center; gap:8px; padding:10px 14px; background:#111C24; border-radius:8px; border:0.5px solid rgba(0,255,136,0.08);">






                    <span class="status-dot status-{'ok' if trend == 'improving' else 'warn'}"></span>






                    <span style="font-size:12px; color:{diag_color};">{diag_text}</span>






                </div>






                """, height=100, scrolling=False)













        st.components.v1.html('<div class="section-label">Model Comparison</div>')













        if models_info:






            model_df = pd.DataFrame(models_info)






            if 'name' in model_df.columns:






                model_df['Active'] = model_df['name'].apply(lambda x: '★' if x == best_model.get('name') else '')






                cols = ['Active'] + [c for c in model_df.columns if c != 'Active']






                model_df = model_df[cols]






            st.dataframe(model_df, use_container_width=True)






    else:






        st.info("No models found. Train a model to see details here.")













# -----------------------------------------------------------------------------






# TAB 6 — AI BRIEFING (NEW)






# -----------------------------------------------------------------------------






with tab6:






    if not BACKEND_AVAILABLE:






        st.warning("Backend modules not available.")






    elif not (_AI_AVAILABLE and _ai_reporter):






       _ensure_ai()






    if not (_AI_AVAILABLE and _ai_reporter):






        st.info("AI Briefing not available. Check Groq API key in .env file.")






    else:






        # Cache key for briefing






        briefing_key = f"{coin}_{datetime.now().strftime('%Y-%m-%d_%H')}"






        






        if briefing_key not in st.session_state.briefing_cache:






            with st.spinner("Generating AI market briefing..."):






                try:






                    pred = prediction if prediction else get_prediction(coin)






                    indicators = None






                    if pred and 'indicators' in pred:






                        indicators = {'coin': coin, 'timestamp': pred.get('timestamp'), 'indicators': pred['indicators']}






                    






                    drift_summary = None






                    if drift_data and drift_data.get('summary'):






                        drift_summary = drift_data['summary']






                    






                    briefing = _ai_reporter.generate(






                        coin_id=coin,






                        prediction=pred,






                        indicators=indicators,






                        drift_summary=drift_summary,






                    )






                    






                    briefing_dict = briefing.dict() if hasattr(briefing, 'dict') else briefing






                    st.session_state.briefing_cache[briefing_key] = briefing_dict






                except Exception as e:






                    st.error(f"Failed to generate briefing: {e}")






                    st.session_state.briefing_cache[briefing_key] = None






        






        cached = st.session_state.briefing_cache.get(briefing_key)






        if cached:






            title = cached.get('title', 'Market Briefing')






            summary = cached.get('summary', 'No summary available.')






            outlook = cached.get('outlook', 'No outlook available.')






            risk_level = cached.get('risk_level', 'unknown')






            risk_color = GREEN if risk_level == 'low' else AMBER if risk_level == 'medium' else RED if risk_level == 'high' else TEXT_SECONDARY






            






            st.components.v1.html(f"""






            <div style="background:#0B1319; border:0.5px solid rgba(77,195,255,0.15); border-radius:12px; padding:20px; margin-bottom:16px;">






                <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:24px; color:#4DC3FF; margin-bottom:8px;">{title}</div>






                <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.12em;">{coin.upper()} / USD · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>






            </div>






            """, height=100, scrolling=False)






            






            st.components.v1.html(f"""






            <div class="ai-card">






                <div class="ai-card-title">📋 Executive Summary</div>






                <div class="ai-card-body">{summary}</div>






            </div>






            """, height=150, scrolling=False)






            






            st.components.v1.html(f"""






            <div class="ai-card">






                <div class="ai-card-title">🔮 Outlook</div>






                <div class="ai-card-body">{outlook}</div>






            </div>






            """, height=150, scrolling=False)






            






            st.components.v1.html(f"""






            <div style="display:flex; align-items:center; gap:12px; padding:12px 16px; background:#111C24; border-radius:10px; border:0.5px solid rgba(0,255,136,0.08);">






                <span style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">Risk Level</span>






                <span style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:{risk_color}; text-transform:uppercase;">{risk_level}</span>






            </div>






            """, height=100, scrolling=False)






            






            if st.button("🔄 Regenerate Briefing", use_container_width=True):






                if briefing_key in st.session_state.briefing_cache:






                    del st.session_state.briefing_cache[briefing_key]






                st.rerun()






        else:






            st.info("Briefing could not be generated. Ensure Ollama is running and data is available.")













# -----------------------------------------------------------------------------






# TAB 7 — DRIFT MONITOR (NEW)






# -----------------------------------------------------------------------------






with tab7:






    if not BACKEND_AVAILABLE:






        st.warning("Backend modules not available.")






    elif drift_data is None:






        st.info("No drift data available. Ensure sufficient historical data exists.")






    else:






        _ensure_ai()






        






        # Extract results and summary from drift_data






        results = drift_data.get('results', {})






        summary = drift_data.get('summary', {})        






        # Summary metrics






        drift_detected = results.get('drift_detected', False)






        drift_pct = results.get('drift_percentage', 0)






        n_drifted = len(results.get('drifted_features', []))






        n_total = len(results.get('features', {}))






        






        status_text = "DRIFT DETECTED" if drift_detected else "STABLE"






        status_color = RED if drift_detected and drift_pct > 50 else AMBER if drift_detected else GREEN






        






        metrics = [






            ("Status", status_text, status_color),






            ("Drifted Features", f"{n_drifted}/{n_total}", RED if n_drifted > 0 else GREEN),






            ("Drift %", f"{drift_pct:.1f}%", RED if drift_pct > 20 else GREEN),






            ("Recent Checks", str(summary.get('total_checks', 0)), TEXT_SECONDARY),






            ("Recent Drifts", str(summary.get('drift_count_recent', 0)), RED if summary.get('drift_count_recent', 0) > 0 else GREEN),






            ("Drift Rate", f"{summary.get('drift_rate', 0):.1%}", RED if summary.get('drift_rate', 0) > 0.2 else GREEN)






        ]






        st.components.v1.html(shuffle_metrics_html(metrics), height=72, scrolling=False)






        






        # AI Drift Narrative






        if _AI_AVAILABLE and _ai_drift_narrator and drift_detected:






            with st.spinner("Generating drift narrative..."):






                try:






                    narrative = _ai_drift_narrator.narrate(results)






                    narrative_dict = narrative.dict() if hasattr(narrative, 'dict') else narrative






                    narrative_text = narrative_dict.get('narrative', str(narrative_dict)) if isinstance(narrative_dict, dict) else str(narrative_dict)






                    






                    st.components.v1.html(f"""






                    <div class="ai-card">






                        <div class="ai-card-title">🧠 AI Drift Analysis</div>






                        <div class="ai-card-body">{narrative_text}</div>






                    </div>






                    """, height=150, scrolling=False)






                except Exception as e:






                    st.warning(f"Drift narrative failed: {e}")






        






        # Per-feature drift details






        st.components.v1.html('<div class="section-label">Per-Feature Drift Details</div>')






        






        features = results.get('features', {})






        if features:






            for feature_name, metrics in features.items():






                is_drifted = metrics.get('drift_detected', False)






                psi = metrics.get('psi', 0)






                ks_p = metrics.get('ks_pvalue', 1)






                mean_shift = metrics.get('mean_shift', 0)






                






                feat_color = RED if is_drifted else GREEN






                feat_badge = '<span class="badge-down">DRIFTED</span>' if is_drifted else '<span class="badge-up">STABLE</span>'






                






                st.components.v1.html(f"""






                <div class="indicator-row">






                    <div style="display:flex; align-items:center; gap:12px;">






                        <span style="font-size:16px;">{'⚠️' if is_drifted else '✅'}</span>






                        <div>






                            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em;">{feature_name}</div>






                            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:16px; color:{feat_color};">PSI: {psi:.4f} · KS p: {ks_p:.4f}</div>






                        </div>






                    </div>






                    {feat_badge}






                </div>






                """, height=80, scrolling=False)






        else:






            st.info("No feature-level drift data available.")






        






        # Recommendation






        recommendation = results.get('recommendation', 'No recommendation available.')






        rec_color = RED if 'CRITICAL' in recommendation else AMBER if 'WARNING' in recommendation else GREEN if 'OK' in recommendation else TEXT_SECONDARY






        






        st.components.v1.html(f"""






        <div style="margin-top:16px; padding:12px 16px; background:#111C24; border-radius:10px; border:0.5px solid rgba(0,255,136,0.08);">






            <div style="font-size:11px; color:#3A5060; text-transform:uppercase; letter-spacing:.1em; margin-bottom:4px;">System Recommendation</div>






            <div style="font-size:13px; color:{rec_color};">{recommendation}</div>






        </div>






        """, height=100, scrolling=False)













# =============================================================================






# AUTO REFRESH






# =============================================================================






if auto_refresh:






    time.sleep(refresh_interval)






    st.cache_data.clear()






    st.rerun()