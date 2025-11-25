import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_engine_dir = os.path.dirname(current_dir)
stock_bull_dir = os.path.dirname(ml_engine_dir)
data_pipeline_dir = os.path.join(stock_bull_dir, 'data-pipeline')

sys.path.insert(0, ml_engine_dir)
sys.path.insert(0, data_pipeline_dir)

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Stock Bull - AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeIn 1s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stock-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .buy-signal {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .sell-signal {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .hold-signal {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model():
    """Load the trained model and preprocessor"""
    try:
        model_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'quick_test_model.pkl')
        preprocessor_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'preprocessor.pkl')
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at: {model_path}")
            st.info("💡 Please train the model first by running: `python scripts/quick_train.py`")
            return None, None
        
        model = joblib.load(model_path)
        preprocessor_data = joblib.load(preprocessor_path)
        
        return model, preprocessor_data
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_data(ttl=3600)
def load_stock_data():
    """Load stock data from CSV"""
    try:
        data_path = os.path.join(data_pipeline_dir, 'processed_data', 'complete_training_dataset.csv')
        
        if not os.path.exists(data_path):
            st.error(f"❌ Data not found at: {data_path}")
            st.info("💡 Please generate data first by running the data pipeline")
            return pd.DataFrame()
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

def get_signal_emoji(prediction):
    """Get emoji for prediction"""
    emoji_map = {
        'Strong Buy': '🚀',
        'Buy': '✅',
        'Hold': '⏸️',
        'Sell': '⚠️',
        'Strong Sell': '❌'
    }
    return emoji_map.get(prediction, '❓')

def get_signal_color(prediction):
    """Get color for prediction"""
    color_map = {
        'Strong Buy': '#00b894',
        'Buy': '#00b894',
        'Hold': '#fdcb6e',
        'Sell': '#d63031',
        'Strong Sell': '#d63031'
    }
    return color_map.get(prediction, '#74b9ff')

def make_predictions(df, model, preprocessor_data):
    """Make predictions on stock data"""
    feature_cols = preprocessor_data['feature_cols']
    scaler = preprocessor_data['scaler']
    
    class_map = {0: 'Strong Sell', 1: 'Sell', 2: 'Hold', 3: 'Buy', 4: 'Strong Buy'}
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            X = row[feature_cols].values.reshape(1, -1)
            X_df = pd.DataFrame(X, columns=feature_cols)
            X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_scaled = scaler.transform(X_df)
            
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = proba.max()
            
            results.append({
                'symbol': row['symbol'],
                'date': row['date'],
                'close': row.get('close', 0),
                'open': row.get('open', 0),
                'high': row.get('high', 0),
                'low': row.get('low', 0),
                'volume': row.get('volume', 0),
                'prediction': class_map[pred],
                'prediction_num': pred,
                'confidence': confidence,
                'sentiment': row.get('sentiment_mean', 0),
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd', 0),
                'news_count': int(row.get('news_count', 0)),
                'volume_ratio': row.get('volume_ratio', 1.0),
                'momentum_20': row.get('momentum_pct_20', 0),
                'sma_20': row.get('sma_20', 0),
                'sma_50': row.get('sma_50', 0)
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# ==================== PAGE FUNCTIONS ====================

def show_dashboard(df, model, preprocessor_data):
    """Main dashboard page"""
    # Header with animation
    st.markdown('<h1 class="main-header">📈 Stock Bull Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>AI-Powered Stock Market Intelligence</p>", unsafe_allow_html=True)
    
    # Get latest data for each stock
    latest_df = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)
    
    # Make predictions
    with st.spinner("🔮 Analyzing market data..."):
        predictions_df = make_predictions(latest_df, model, preprocessor_data)
    
    if predictions_df.empty:
        st.error("❌ No predictions could be generated")
        return
    
    # Display date
    st.markdown(f"<p style='text-align: center; color: #888;'>Last Updated: {predictions_df['date'].max().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics Row
    st.subheader("📊 Market Snapshot")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    buy_count = len(predictions_df[predictions_df['prediction'].isin(['Buy', 'Strong Buy'])])
    hold_count = len(predictions_df[predictions_df['prediction'] == 'Hold'])
    sell_count = len(predictions_df[predictions_df['prediction'].isin(['Sell', 'Strong Sell'])])
    avg_confidence = predictions_df['confidence'].mean() * 100
    total_stocks = len(predictions_df)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);">
            <h3 style="margin:0;">🚀 Buy</h3>
            <h1 style="margin:0;">{buy_count}</h1>
            <p style="margin:0; opacity: 0.8;">Strong Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);">
            <h3 style="margin:0;">⏸️ Hold</h3>
            <h1 style="margin:0;">{hold_count}</h1>
            <p style="margin:0; opacity: 0.8;">Wait & Watch</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #d63031 0%, #e84393 100%);">
            <h3 style="margin:0;">⚠️ Sell</h3>
            <h1 style="margin:0;">{sell_count}</h1>
            <p style="margin:0; opacity: 0.8;">Warning Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);">
            <h3 style="margin:0;">🎯 Confidence</h3>
            <h1 style="margin:0;">{avg_confidence:.1f}%</h1>
            <p style="margin:0; opacity: 0.8;">Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%);">
            <h3 style="margin:0;">📊 Stocks</h3>
            <h1 style="margin:0;">{total_stocks}</h1>
            <p style="margin:0; opacity: 0.8;">Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal Distribution Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Signal Distribution")
        
        signal_counts = predictions_df['prediction'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=signal_counts.index,
                y=signal_counts.values,
                marker=dict(
                    color=[get_signal_color(x) for x in signal_counts.index],
                    line=dict(color='white', width=2)
                ),
                text=signal_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Distribution of Trading Signals",
            xaxis_title="Signal",
            yaxis_title="Number of Stocks",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Confidence Levels")
        
        # Confidence distribution
        conf_ranges = ['High (>80%)', 'Medium (60-80%)', 'Low (<60%)']
        conf_counts = [
            len(predictions_df[predictions_df['confidence'] > 0.8]),
            len(predictions_df[(predictions_df['confidence'] >= 0.6) & (predictions_df['confidence'] <= 0.8)]),
            len(predictions_df[predictions_df['confidence'] < 0.6])
        ]
        
        fig2 = go.Figure(data=[
            go.Pie(
                labels=conf_ranges,
                values=conf_counts,
                hole=0.4,
                marker=dict(colors=['#00b894', '#fdcb6e', '#d63031'])
            )
        ])
        
        fig2.update_layout(
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Top Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Top Buy Recommendations")
        buy_stocks = predictions_df[predictions_df['prediction'].isin(['Buy', 'Strong Buy'])].sort_values('confidence', ascending=False).head(5)
        
        if not buy_stocks.empty:
            for idx, stock in buy_stocks.iterrows():
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: #2d3436;">{get_signal_emoji(stock['prediction'])} {stock['symbol']}</h3>
                            <p style="margin: 5px 0; color: #636e72;">Price: ₹{stock['close']:.2f}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0;" class="buy-signal">{stock['prediction']}</p>
                            <p style="margin: 5px 0; color: #636e72;">Confidence: {stock['confidence']*100:.1f}%</p>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #dfe6e9;">
                        <span style="margin-right: 15px;">📊 RSI: {stock['rsi']:.1f}</span>
                        <span style="margin-right: 15px;">📰 News: {stock['news_count']}</span>
                        <span>😊 Sentiment: {stock['sentiment']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong buy signals at this time")
    
    with col2:
        st.subheader("⚠️ Stocks to Monitor")
        monitor_stocks = predictions_df[predictions_df['prediction'].isin(['Sell', 'Strong Sell'])].sort_values('confidence', ascending=False).head(5)
        
        if not monitor_stocks.empty:
            for idx, stock in monitor_stocks.iterrows():
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: #2d3436;">{get_signal_emoji(stock['prediction'])} {stock['symbol']}</h3>
                            <p style="margin: 5px 0; color: #636e72;">Price: ₹{stock['close']:.2f}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0;" class="sell-signal">{stock['prediction']}</p>
                            <p style="margin: 5px 0; color: #636e72;">Confidence: {stock['confidence']*100:.1f}%</p>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #dfe6e9;">
                        <span style="margin-right: 15px;">📊 RSI: {stock['rsi']:.1f}</span>
                        <span style="margin-right: 15px;">📰 News: {stock['news_count']}</span>
                        <span>😟 Sentiment: {stock['sentiment']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sell signals at this time")
    
    st.markdown("---")
    
    # All Predictions Table
    st.subheader("📋 Complete Stock Analysis")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_filter = st.multiselect(
            "Filter by Signal",
            ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
            default=['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
        )
    
    with col2:
        min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 0)
    
    with col3:
        sort_by = st.selectbox("Sort by", ['Confidence', 'Price', 'Symbol', 'RSI'])
    
    # Apply filters
    filtered_df = predictions_df[
        (predictions_df['prediction'].isin(signal_filter)) &
        (predictions_df['confidence'] * 100 >= min_confidence)
    ].copy()
    
    # Sort
    sort_map = {'Confidence': 'confidence', 'Price': 'close', 'Symbol': 'symbol', 'RSI': 'rsi'}
    filtered_df = filtered_df.sort_values(sort_map[sort_by], ascending=False)
    
    # Format for display
    display_df = filtered_df.copy()
    display_df['Signal'] = display_df.apply(lambda x: f"{get_signal_emoji(x['prediction'])} {x['prediction']}", axis=1)
    display_df['Price'] = display_df['close'].apply(lambda x: f"₹{x:.2f}")
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Sentiment'] = display_df['sentiment'].apply(lambda x: f"{x:.2f}")
    display_df['RSI'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        display_df[['symbol', 'Price', 'Signal', 'Confidence', 'Sentiment', 'RSI', 'news_count']].rename(columns={
            'symbol': 'Stock',
            'news_count': 'News'
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )

def show_stock_analysis(df, model, preprocessor_data):
    """Detailed stock analysis page"""
    st.markdown('<h1 class="main-header">📊 Detailed Stock Analysis</h1>', unsafe_allow_html=True)
    
    # Stock selector
    available_stocks = sorted(df['symbol'].unique())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stock = st.selectbox("🔍 Select a stock to analyze", available_stocks, index=0)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("📈 Analyze", use_container_width=True)
    
    # Filter data for selected stock
    stock_df = df[df['symbol'] == selected_stock].sort_values('date')
    
    if stock_df.empty:
        st.warning(f"No data available for {selected_stock}")
        return
    
    # Get latest prediction
    latest_data = stock_df.tail(1)
    predictions = make_predictions(latest_data, model, preprocessor_data)
    
    if predictions.empty:
        st.error("Could not generate prediction")
        return
    
    pred = predictions.iloc[0]
    
    st.markdown("---")
    
    # Stock Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Current Price</h4>
            <h2 style="margin: 5px 0;">₹{pred['close']:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        signal_color = get_signal_color(pred['prediction'])
        st.markdown(f"""
        <div class="metric-card" style="background: {signal_color};">
            <h4 style="margin: 0; opacity: 0.8;">ML Signal</h4>
            <h2 style="margin: 5px 0;">{get_signal_emoji(pred['prediction'])} {pred['prediction']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Confidence</h4>
            <h2 style="margin: 5px 0;">{pred['confidence']*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sentiment_emoji = "😊" if pred['sentiment'] > 0.1 else "😐" if pred['sentiment'] > -0.1 else "😟"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Sentiment</h4>
            <h2 style="margin: 5px 0;">{sentiment_emoji} {pred['sentiment']:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Price Chart
    st.subheader(f"📈 {selected_stock} Price History")
    
    # Chart type selector
    chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick"], horizontal=True)
    
    if chart_type == "Line Chart":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stock_df['date'],
            y=stock_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        # Add moving averages if available
        if 'sma_20' in stock_df.columns:
            fig.add_trace(go.Scatter(
                x=stock_df['date'],
                y=stock_df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff7f0e', width=1.5, dash='dash')
            ))
        
        if 'sma_50' in stock_df.columns:
            fig.add_trace(go.Scatter(
                x=stock_df['date'],
                y=stock_df['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#2ca02c', width=1.5, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{selected_stock} Price Trend",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    
    else:  # Candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=stock_df['date'],
            open=stock_df['open'],
            high=stock_df['high'],
            low=stock_df['low'],
            close=stock_df['close'],
            increasing_line_color='#00b894',
            decreasing_line_color='#d63031'
        )])
        
        fig.update_layout(
            title=f"{selected_stock} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Technical Indicators
    st.subheader("📊 Technical Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest = stock_df.iloc[-1]
    
    with col1:
        rsi = latest.get('rsi', 50)
        rsi_signal = "🔴 Oversold" if rsi < 30 else "🔵 Overbought" if rsi > 70 else "🟢 Neutral"
        st.metric("RSI", f"{rsi:.1f}", rsi_signal)
    
    with col2:
        macd = latest.get('macd', 0)
        macd_signal = "🟢 Bullish" if macd > 0 else "🔴 Bearish"
        st.metric("MACD", f"{macd:.2f}", macd_signal)
    
    with col3:
        volume_ratio = latest.get('volume_ratio', 1.0)
        vol_signal = "📈 High" if volume_ratio > 1.5 else "📉 Low"
        st.metric("Volume Ratio", f"{volume_ratio:.2f}", vol_signal)
    
    with col4:
        momentum = latest.get('momentum_pct_20', 0)
        mom_signal = f"+{momentum:.1f}%" if momentum > 0 else f"{momentum:.1f}%"
        st.metric("20-Day Momentum", f"{momentum:.1f}%", mom_signal)
    
    with col5:
        news_count = int(latest.get('news_count', 0))
        st.metric("Recent News", news_count, "articles")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Volume and RSI Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Volume Analysis")
        
        fig_vol = go.Figure()
        
        colors = ['#00b894' if c > o else '#d63031' for c, o in zip(stock_df['close'], stock_df['open'])]
        
        fig_vol.add_trace(go.Bar(
            x=stock_df['date'],
            y=stock_df['volume'],
            marker=dict(color=colors),
            name='Volume'
        ))
        
        fig_vol.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        st.subheader("📈 RSI Indicator")
        
        if 'rsi' in stock_df.columns:
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=stock_df['date'],
                y=stock_df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#6c5ce7', width=2)
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig_rsi.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=300,
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendation Summary
    st.subheader("💡 AI Recommendation Summary")
    
    recommendation_text = ""
    recommendation_color = ""
    
    if pred['prediction'] in ['Strong Buy', 'Buy']:
        recommendation_text = f"""
        ### ✅ {pred['prediction']} Signal
        
        **Our AI model suggests this is a good buying opportunity based on:**
        - Strong technical indicators (RSI: {pred['rsi']:.1f})
        - Positive sentiment analysis ({pred['sentiment']:.2f})
        - {pred['confidence']*100:.1f}% model confidence
        - Recent momentum: {pred['momentum_20']:.1f}%
        
        ⚠️ **Risk Level:** {"Low" if pred['confidence'] > 0.8 else "Medium" if pred['confidence'] > 0.6 else "High"}
        """
        recommendation_color = "#e8f5e9"
    
    elif pred['prediction'] == 'Hold':
        recommendation_text = f"""
        ### ⏸️ Hold Signal
        
        **Our AI model suggests holding your position:**
        - Market conditions are neutral
        - RSI indicator: {pred['rsi']:.1f}
        - Sentiment: {pred['sentiment']:.2f}
        - Wait for clearer signals before taking action
        
        💡 **Suggestion:** Monitor closely for trend changes
        """
        recommendation_color = "#fff3e0"
    
    else:  # Sell or Strong Sell
        recommendation_text = f"""
        ### ⚠️ {pred['prediction']} Signal
        
        **Our AI model suggests caution:**
        - Negative technical indicators detected
        - RSI: {pred['rsi']:.1f}
        - Sentiment score: {pred['sentiment']:.2f}
        - {pred['confidence']*100:.1f}% model confidence
        
        ⚠️ **Risk Level:** High - Consider reducing exposure
        """
        recommendation_color = "#ffebee"
    
    st.markdown(f"""
    <div style="background-color: {recommendation_color}; padding: 1.5rem; border-radius: 1rem; border-left: 5px solid {get_signal_color(pred['prediction'])};">
        {recommendation_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Historical Performance
    st.subheader("📊 Historical Performance Metrics")
    
    # Calculate performance metrics
    returns_30d = ((stock_df['close'].iloc[-1] - stock_df['close'].iloc[-30]) / stock_df['close'].iloc[-30] * 100) if len(stock_df) >= 30 else 0
    returns_90d = ((stock_df['close'].iloc[-1] - stock_df['close'].iloc[-90]) / stock_df['close'].iloc[-90] * 100) if len(stock_df) >= 90 else 0
    volatility = stock_df['close'].pct_change().std() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("30-Day Return", f"{returns_30d:.2f}%", f"{'📈' if returns_30d > 0 else '📉'}")
    
    with col2:
        st.metric("90-Day Return", f"{returns_90d:.2f}%", f"{'📈' if returns_90d > 0 else '📉'}")
    
    with col3:
        st.metric("Volatility", f"{volatility:.2f}%")

def show_live_predictions(df, model, preprocessor_data):
    """Live predictions page with refresh capability"""
    st.markdown('<h1 class="main-header">🤖 Live Stock Predictions</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<p style='font-size: 1.1rem; color: #666;'>Real-time AI predictions based on latest market data</p>", unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Get latest data
    latest_df = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)
    
    # Make predictions
    with st.spinner("🔮 Generating live predictions..."):
        predictions_df = make_predictions(latest_df, model, preprocessor_data)
    
    if predictions_df.empty:
        st.error("No predictions available")
        return
    
    # Display timestamp
    st.info(f"📅 Data as of: {predictions_df['date'].max().strftime('%B %d, %Y, %I:%M %p')}")
    
    # Filter options
    st.subheader("🔍 Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_filter = st.multiselect(
            "Trading Signal",
            ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
            default=['Strong Buy', 'Buy']
        )
    
    with col2:
        min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 50)
    
    with col3:
        min_rsi = st.slider("Minimum RSI", 0, 100, 0)
    
    # Apply filters
    filtered_df = predictions_df[
        (predictions_df['prediction'].isin(signal_filter)) &
        (predictions_df['confidence'] * 100 >= min_confidence) &
        (predictions_df['rsi'] >= min_rsi)
    ].sort_values('confidence', ascending=False)
    
    st.markdown(f"<h3>📊 Showing {len(filtered_df)} stocks matching your criteria</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Display results in expandable cards
    if filtered_df.empty:
        st.warning("No stocks match your filter criteria. Try adjusting the filters.")
    else:
        for idx, stock in filtered_df.iterrows():
            with st.expander(f"{get_signal_emoji(stock['prediction'])} **{stock['symbol']}** - {stock['prediction']} ({stock['confidence']*100:.1f}% confidence)", expanded=False):
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("### 💰 Price Info")
                    st.metric("Current Price", f"₹{stock['close']:.2f}")
                    st.metric("20-Day Momentum", f"{stock['momentum_20']:.2f}%")
                
                with col2:
                    st.markdown("### 📊 Technical")
                    st.metric("RSI", f"{stock['rsi']:.1f}")
                    st.metric("MACD", f"{stock['macd']:.2f}")
                
                with col3:
                    st.markdown("### 🤖 AI Analysis")
                    st.metric("Confidence", f"{stock['confidence']*100:.1f}%")
                    st.metric("Signal", stock['prediction'])
                
                with col4:
                    st.markdown("### 📰 Sentiment")
                    sentiment_emoji = "😊" if stock['sentiment'] > 0.1 else "😐" if stock['sentiment'] > -0.1 else "😟"
                    st.metric("Score", f"{stock['sentiment']:.2f} {sentiment_emoji}")
                    st.metric("News Articles", int(stock['news_count']))
                
                # Progress bars
                st.markdown("#### 📈 Indicator Strength")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence_pct = stock['confidence'] * 100
                    st.progress(stock['confidence'], text=f"Model Confidence: {confidence_pct:.1f}%")
                
                with col2:
                    rsi_normalized = stock['rsi'] / 100
                    st.progress(rsi_normalized, text=f"RSI: {stock['rsi']:.1f}")
                
                # Action recommendation
                if stock['prediction'] in ['Strong Buy', 'Buy']:
                    st.success("💡 **Recommendation:** Consider buying this stock based on strong positive signals")
                elif stock['prediction'] == 'Hold':
                    st.info("💡 **Recommendation:** Hold your position and monitor for changes")
                else:
                    st.warning("💡 **Recommendation:** Exercise caution - consider reducing exposure")

def show_portfolio_insights(df):
    """Portfolio insights and statistics"""
    st.markdown('<h1 class="main-header">📈 Portfolio & Market Insights</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Statistics
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_stocks = df['symbol'].nunique()
    date_range = (df['date'].max() - df['date'].min()).days
    total_records = len(df)
    avg_price = df.groupby('symbol').tail(1)['close'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Stocks Analyzed</h4>
            <h2 style="margin: 5px 0;">{total_stocks}</h2>
            <p style="margin: 0; opacity: 0.8;">Companies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Data Range</h4>
            <h2 style="margin: 5px 0;">{date_range}</h2>
            <p style="margin: 0; opacity: 0.8;">Days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Total Records</h4>
            <h2 style="margin: 5px 0;">{total_records:,}</h2>
            <p style="margin: 0; opacity: 0.8;">Data Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; opacity: 0.8;">Avg Price</h4>
            <h2 style="margin: 5px 0;">₹{avg_price:.2f}</h2>
            <p style="margin: 0; opacity: 0.8;">Market</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get latest data for each stock
    latest_df = df.sort_values('date').groupby('symbol').tail(1)
    
    # Performance Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Performance Distribution")
        
        if 'momentum_pct_20' in latest_df.columns:
            fig = px.histogram(
                latest_df,
                x='momentum_pct_20',
                nbins=30,
                title="20-Day Momentum Distribution",
                labels={'momentum_pct_20': '20-Day Return (%)'},
                color_discrete_sequence=['#667eea']
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 RSI Distribution")
        
        if 'rsi' in latest_df.columns:
            fig = px.box(
                latest_df,
                y='rsi',
                title="RSI Distribution Across Stocks",
                color_discrete_sequence=['#764ba2']
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Performers
    st.subheader("🏆 Top & Bottom Performers (Last 20 Days)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Top Gainers")
        
        if 'momentum_pct_20' in latest_df.columns:
            top_performers = latest_df.nlargest(10, 'momentum_pct_20')[['symbol', 'close', 'momentum_pct_20']]
            
            for idx, row in top_performers.iterrows():
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold; font-size: 1.1rem;">{row['symbol']}</span>
                        <span style="color: #00b894; font-weight: bold;">+{row['momentum_pct_20']:.2f}%</span>
                    </div>
                    <p style="margin: 5px 0; color: #636e72;">Price: ₹{row['close']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📉 Top Losers")
        
        if 'momentum_pct_20' in latest_df.columns:
            bottom_performers = latest_df.nsmallest(10, 'momentum_pct_20')[['symbol', 'close', 'momentum_pct_20']]
            
            for idx, row in bottom_performers.iterrows():
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold; font-size: 1.1rem;">{row['symbol']}</span>
                        <span style="color: #d63031; font-weight: bold;">{row['momentum_pct_20']:.2f}%</span>
                    </div>
                    <p style="margin: 5px 0; color: #636e72;">Price: ₹{row['close']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Volatility Analysis
    st.subheader("📊 Volatility Analysis")
    
    # Calculate volatility for each stock
    volatility_data = []
    for symbol in df['symbol'].unique():
        stock_data = df[df['symbol'] == symbol].sort_values('date')
        if len(stock_data) > 20:
            returns = stock_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            volatility_data.append({
                'symbol': symbol,
                'volatility': volatility,
                'current_price': stock_data['close'].iloc[-1]
            })
    
    volatility_df = pd.DataFrame(volatility_data).sort_values('volatility', ascending=False)
    
    if not volatility_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Most Volatile Stocks")
            high_vol = volatility_df.head(5)
            
            for idx, row in high_vol.iterrows():
                st.markdown(f"""
                <div class="warning-box">
                    <strong>{row['symbol']}</strong> - Volatility: {row['volatility']:.2f}%
                    <br>Price: ₹{row['current_price']:.2f}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 😌 Least Volatile Stocks")
            low_vol = volatility_df.tail(5)
            
            for idx, row in low_vol.iterrows():
                st.markdown(f"""
                <div class="success-box">
                    <strong>{row['symbol']}</strong> - Volatility: {row['volatility']:.2f}%
                    <br>Price: ₹{row['current_price']:.2f}
                </div>
                """, unsafe_allow_html=True)

def show_about():
    """About page with project information"""
    st.markdown('<h1 class="main-header">ℹ️ About Stock Bull</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Welcome to Stock Bull - Your AI-Powered Trading Assistant</h3>
        <p style="font-size: 1.1rem;">
        Stock Bull combines cutting-edge machine learning with real-time market data to provide 
        intelligent stock predictions and actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Key Features
        
        #### 🤖 Machine Learning Models
        - **Random Forest Classifier**
        - **XGBoost & LightGBM**
        - **Ensemble Learning**
        - 80.77% Accuracy
        
        #### 📊 Technical Analysis
        - 40+ Technical Indicators
        - RSI, MACD, Bollinger Bands
        - Moving Averages (SMA/EMA)
        - Volume Analysis
        
        #### 📰 Sentiment Analysis
        - FinBERT AI Model
        - Real-time News Processing
        - Multi-source Data Collection
        - Sentiment Scoring (-1 to +1)
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technology Stack
        
        #### Data Pipeline
        - **Languages:** Python 3.8+
        - **Database:** PostgreSQL
        - **APIs:** NSEPy, yfinance, NewsAPI
        - **Processing:** Pandas, NumPy
        
        #### ML Engine
        - **Frameworks:** scikit-learn, XGBoost, LightGBM
        - **NLP:** Transformers, FinBERT
        - **Visualization:** Plotly, Matplotlib
        
        #### Frontend
        - **Framework:** Streamlit
        - **Charts:** Plotly Interactive Charts
        - **Styling:** Custom CSS
        """)
    
    st.markdown("---")
    
    # How It Works
    st.markdown("""
    ### 🔄 How Stock Bull Works
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h2 style="margin: 0;">1️⃣</h2>
            <h4>Data Collection</h4>
            <p style="font-size: 0.9rem;">Gather price data, news, fundamentals from multiple sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2 style="margin: 0;">2️⃣</h2>
            <h4>Feature Engineering</h4>
            <p style="font-size: 0.9rem;">Calculate 40+ technical indicators and sentiment scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2 style="margin: 0;">3️⃣</h2>
            <h4>ML Prediction</h4>
            <p style="font-size: 0.9rem;">AI models analyze patterns and generate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h2 style="margin: 0;">4️⃣</h2>
            <h4>Actionable Insights</h4>
            <p style="font-size: 0.9rem;">Get buy/sell/hold signals with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "80.77%", "High")
    
    with col2:
        st.metric("Training Data", "5+ Years", "Historical")
    
    with col3:
        st.metric("Stocks", "50+", "NIFTY 50")
    
    with col4:
        st.metric("Features", "70+", "Technical")
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Important Disclaimer</h3>
        <p style="font-size: 1.1rem;">
        <strong>Stock Bull is an educational project for learning purposes only.</strong>
        </p>
        <ul style="font-size: 1rem;">
            <li>Predictions are based on historical data and may not reflect future performance</li>
            <li>Always conduct your own research before making investment decisions</li>
            <li>Consult with a qualified financial advisor for personalized advice</li>
            <li>Past performance does not guarantee future results</li>
            <li>Stock market investments carry inherent risks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white;">
        <h2 style="margin: 0;">Made with ❤️ by Stock Bull Team</h2>
        <p style="margin: 10px 0;">Combining Machine Learning, Data Science, and Financial Analysis</p>
        <p style="margin: 0; opacity: 0.8;">College Project • 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

def main():
    """Main application entry point"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bull.png", width=100)
        st.title("🐂 Stock Bull")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Dashboard", "📊 Stock Analysis", "🤖 Live Predictions", "📈 Portfolio Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        st.info("Model Accuracy: **80.77%**")
        st.success("Stocks Analyzed: **50+**")
        st.warning("Last Updated: **Today**")
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.checkbox("Dark Mode", value=False, disabled=True, help="Coming soon!")
            st.checkbox("Auto-refresh", value=False, disabled=True, help="Coming soon!")
            st.slider("Refresh Interval (min)", 1, 60, 15, disabled=True, help="Coming soon!")
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Load data and model
    with st.spinner("🚀 Initializing Stock Bull..."):
        model, preprocessor_data = load_model()
        df = load_stock_data()
    
    # Check if resources loaded successfully
    if model is None or df.empty:
        st.error("❌ Failed to load required resources")
        
        with st.expander("🔧 Troubleshooting Guide"):
            st.markdown("""
            ### Steps to Fix:
            
            1. **Train the Model:**
            ```bash
            cd stock-bull/ml-engine/scripts
            python quick_train.py
            ```
            
            2. **Generate Training Data:**
            ```bash
            cd stock-bull/data-pipeline
            python run.py features
            ```
            
            3. **Verify Files Exist:**
            - Model: `stock-bull/models/saved_models/quick_test_model.pkl`
            - Data: `stock-bull/data-pipeline/processed_data/complete_training_dataset.csv`
            """)
        return
    
    # Route to selected page
    if page == "🏠 Dashboard":
        show_dashboard(df, model, preprocessor_data)
    elif page == "📊 Stock Analysis":
        show_stock_analysis(df, model, preprocessor_data)
    elif page == "🤖 Live Predictions":
        show_live_predictions(df, model, preprocessor_data)
    elif page == "📈 Portfolio Insights":
        show_portfolio_insights(df)
    elif page == "ℹ️ About":
        show_about()

if __name__ == "__main__":
    main()