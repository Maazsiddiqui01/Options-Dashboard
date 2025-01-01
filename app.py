import streamlit as st
import requests
import datetime
import pandas as pd
import plotly.express as px
import yfinance as yf
import requests
import openpyxl
import praw
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai  # Gemini Integration
from textblob import TextBlob
import streamlit.components.v1 as components
import time


# --- API Keys ---
model = genai.GenerativeModel('gemini-1.5-flash')
GENAI_API_KEY = "AIzaSyAXukbX28-gUndSdFOOGSU9d5SQ5FzxIFc"
NEWSAPI_KEY = "f8d7120cc5fb44a0ae8e9a2ba56df18a"
reddit = praw.Reddit(
    client_id='IqXpaotDyX364rHDlOTbww',
    client_secret='DcD7NG9Rb0II3aYWRV3KikhsgAfCSg',
    user_agent='Southern_Screen409'
)

# --- Initialize APIs ---
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
genai.configure(api_key=GENAI_API_KEY)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Analysis Dashboard", layout="wide")



# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_stock = st.sidebar.text_input("Enter Stock Ticker", value='TSLA')
strike_price = st.sidebar.number_input("Select Strike Price", value=450)
date_range = st.slider("Select Date Range (days)", 1, 30, 7)
timeframe = st.sidebar.selectbox("Select Timeframe", [ "day","minute", "hour", "week", "month"])
show_rsi = st.sidebar.checkbox("Show RSI Indicator", value=True)
show_sma = st.sidebar.checkbox("Show Moving Average (SMA)", value=True)
show_ema = st.sidebar.checkbox("Show Exponential MA (EMA)", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_options = st.sidebar.checkbox("Show Option Chain Data", value=True)
show_news = st.sidebar.checkbox("Show Latest News", value=True)
show_ai_query = st.sidebar.checkbox("Enable AI Analysis (Gemini)", value=True)

ticker = selected_stock


def tradingview_heatmap_html():
    return """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
      {
      "exchanges": ["NASDAQ"],
      "dataSource": "SPX500",
      "grouping": "sector",
      "blockSize": "market_cap_basic",
      "blockColor": "change",
      "locale": "en",
      "symbolUrl": "",
      "colorTheme": "dark",
      "hasTopBar": false,
      "isDataSetEnabled": false,
      "isZoomEnabled": true,
      "hasSymbolTooltip": true,
      "isMonoSize": false,
      "width": "100%",
      "height": "400"
      }
      </script>
    </div>
    """

# Function to Embed Financials Widget
def tradingview_financials_widget(symbol):
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js" async>
      {{
      "isTransparent": false,
      "largeChartUrl": "",
      "displayMode": "regular",
      "width": "100%",
      "height": 600,
      "colorTheme": "dark",
      "symbol": "{symbol}",
      "locale": "en"
      }}
      </script>
    </div>
    """


def tradingview_screener():
    html_code = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      {
      "width": "100%",
      "height": 500,
      "defaultColumn": "overview",
      "defaultScreen": "most_capitalized",
      "market": "america",
      "showToolbar": true,
      "colorTheme": "dark",
      "locale": "en"
      }
      </script>
    </div>
    """
    components.html(html_code, height=500)


def mini_chart_widget(ticker=f"PYTH:{selected_stock}"):
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
      {{
      "symbol": "{ticker}",
      "width": "100%",
      "height": "400",
      "locale": "en",
      "dateRange": "12M",
      "colorTheme": "dark",
      "isTransparent": false,
      "autosize": true,
      "largeChartUrl": ""
      }}
      </script>
    </div>
    """
    components.html(html_code, height=400)





def tradingview_live_chart(ticker):
    return f"""
    <div class="tradingview-widget-container" style="height:600px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
      "autosize": true,
      "symbol": "NASDAQ:{ticker}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "allow_symbol_change": true,
      "calendar": false,
      "studies": [
        "STD;Momentum",
        "STD;RSI"
      ]
      }}
      </script>
    </div>
    """

def tradingview_technical_analysis(ticker):
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
      "interval": "5m",
      "width": "100%",
      "isTransparent": false,
      "height": "400",
      "symbol": "NASDAQ:{ticker}",
      "showIntervalTabs": true,
      "locale": "en",
      "colorTheme": "dark"
      }}
      </script>
    </div>
    """


# TradingView Widget for Real-time Buy/Sell
def get_tradingview_widget(ticker):
    return f"""
    <iframe src="https://www.tradingview.com/embed-widget/technical-analysis/?symbol={ticker}" 
        width="100%" height="400" frameborder="0"></iframe>
 
 """
# Fetch TradingView Buy/Sell Call
def fetch_tradingview_call(ticker):
    try:
        url = f"https://api.tradingview.com/symbols/{ticker}/"
        response = requests.get(url)
        data = response.json()
        return data['recommendation']  # Example: 'BUY', 'SELL', 'HOLD'
    except:
        return "NO DATA"


# Streamlit Layout
st.title(f"{selected_stock} - Stock Dashboard")



# --- TradingView Widget Function ---
def tradingview_widget(ticker, view_type):
    if view_type == "chart":
        return f"""
        <iframe src="https://www.tradingview.com/widgetembed/?symbol=NASDAQ:{ticker}&interval=1D" 
        width="100%" height="500" frameborder="0"></iframe>
        """
    elif view_type == "technical":
        return f"""
        <iframe src="https://www.tradingview.com/widgetembed/?symbol=NASDAQ:{ticker}&hide_top_toolbar=1&interval=1D" 
        width="100%" height="400" frameborder="0"></iframe>
        """
    else:
        return ""

    
#POLYGON CALCULATIONS
POLYGON_API_KEY = "e2z0h4I7rwdV5mtkT_OG1Rc_l15VMSky"

# --- Retry and Backoff Logic ---
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")    
    
def fetch_polygon_technical_data(ticker, timespan, retries=5, backoff_factor=2):


    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"

    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url)

            # --- Handle Rate Limit (429 Error) ---
            if response.status_code == 429:
                wait_time = backoff_factor ** attempt
                st.warning(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
                continue

            response.raise_for_status()
            data = response.json()

            # --- Process Data ---
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')

                # Calculate indicators
                df['SMA_20'] = df['c'].rolling(window=20).mean()
                df['EMA_20'] = df['c'].ewm(span=20, adjust=False).mean()
                df['RSI'] = calculate_rsi(df['c'])

                return df[['timestamp', 'c', 'SMA_20', 'EMA_20', 'RSI']].dropna()
            else:
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            break

    # --- After Maximum Retries ---
    st.error(f"Failed to fetch data after {retries} attempts.")
    return None

def fetch_polygon_stock_data(ticker, multiplier=1, timespan='day', retries=5, backoff_factor=2):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
    
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url)

            # --- Handle Rate Limit (429 Error) ---
            if response.status_code == 429:
                wait_time = backoff_factor ** attempt
                st.warning(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
                continue

            response.raise_for_status()
            data = response.json()

            # --- Process Data ---
            if 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df = df.sort_values(by='timestamp', ascending=True)

                # Rename columns to match table conventions
                df = df.rename(columns={
                    'o': 'Open', 
                    'h': 'High', 
                    'l': 'Low', 
                    'c': 'Close', 
                    'v': 'Volume'
                })

                return df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
            else:
                st.warning(f"No data found for {ticker}")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            break

    # --- After Maximum Retries ---
    st.error(f"Failed to fetch stock data after {retries} attempts.")
    return None




# --- Calculate RSI ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- Generate Buy/Sell Signal ---
def generate_signal(df):
    if df['RSI'].iloc[-1] > 70:
        return "Sell"
    elif df['RSI'].iloc[-1] < 30:
        return "Buy"
    else:
        return "Neutral"


# --- Fetch Data for Selected Timeframes ---
timeframes = ["minute", "hour", "day"]
time_labels = ["1 Min", "1 Hour", "1 Day"]
time_data = {}

for tf in timeframes:
    time_data[tf] = fetch_polygon_technical_data(f"{ticker}", tf)


# --- Display Data in Columns ---
columns = st.columns(3)


# --- Fetch Technical Data from Polygon ---
def fetch_polygon_technical_data(ticker, timeframe):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            
            # Sort data and rename columns
            df = df.sort_values(by='timestamp', ascending=True)
            df.rename(columns={
                "v": "Volume",
                "vw": "Volume Weighted Price",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "t": "Timestamp",
                "n": "Transactions"
            }, inplace=True)
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Failed to fetch data from Polygon: {e}")
        return None

    
# --- Calculate Technical Indicators (SMA, EMA, RSI, Bollinger Bands) ---
def calculate_indicators(df):
    if df is not None:
        # Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        if show_bollinger:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()

        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df
    else:
        return None
    
    
    
# TOP LAYOUT    
for idx, (tf, label) in enumerate(zip(timeframes, time_labels)):
    with columns[idx]:
        st.subheader(label)

        if time_data[tf] is not None and not time_data[tf].empty:
            signal = generate_signal(time_data[tf])

            # Display Buy/Sell Signal
            if signal == "Buy":
                st.markdown(
                    f"<div style='text-align:center; color: white; background-color: green; padding: 10px; border-radius: 10px;'><b>Buy</b></div>",
                    unsafe_allow_html=True
                )
            elif signal == "Sell":
                st.markdown(
                    f"<div style='text-align:center; color: white; background-color: red; padding: 10px; border-radius: 10px;'><b>Sell</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; color: black; background-color: yellow; padding: 10px; border-radius: 10px;'><b>Neutral</b></div>",
                    unsafe_allow_html=True
                )

            # Safe Access to Latest Data
            if len(time_data[tf]) > 0:
                latest_data = time_data[tf].iloc[-1]
                st.write(f"**Close Price:** {latest_data['c']:.2f}")
                st.write(f"**SMA 20:** {latest_data['SMA_20']:.2f}")
                st.write(f"**EMA 20:** {latest_data['EMA_20']:.2f}")
                st.write(f"**RSI:** {latest_data['RSI']:.2f}")
            else:
                st.warning(f"No data available for {label}")

        else:
            st.markdown(
                f"<div style='text-align:center; color: black; background-color: grey; padding: 10px; border-radius: 10px;'><b>No Data</b></div>",
                unsafe_allow_html=True
            )
            st.warning(f"No data available for {label}")
    
st.title(f"📊 Technical Buy/Sell for {selected_stock}")
# Function to Generate Widget HTML for Different Intervals
def tradingview_widget_html(symbol, interval):
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
      "interval": "{interval}",
      "width": "100%",
      "isTransparent": false,
      "height": "380",
      "symbol": "{symbol}",
      "showIntervalTabs": true,
      "displayMode": "single",
      "locale": "en",
      "colorTheme": "dark"
      }}
      </script>
    </div>
    """

# Define Ticker and Intervals
intervals = ["1m", "1h", "1D", "1W"]
labels = ["1 Min", "1 Hour", "1 Day", "1 Week"]

# Create 5 Columns
columns = st.columns(4)

# Embed Widgets in Each Column
for col, interval, label in zip(columns, intervals, labels):
    with col:
        st.subheader(label)
        components.html(tradingview_widget_html(ticker, interval), height=450)      
    
    
# --- Streamlit Layout FROM TRADING VIEW ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(f"Market Price & Trend for {selected_stock}")
    mini_chart_widget(ticker=f"PYTH:{ticker}")

with col2:
    st.subheader(f" Fundamentals for {selected_stock}")
    components.html(tradingview_financials_widget(ticker), height=400)

with col3:
    st.subheader("Stock Heatmap Widget")
    components.html(tradingview_heatmap_html(), height=400)
    
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{selected_stock} Live Chart")
    st.markdown(tradingview_widget(selected_stock, "chart"), unsafe_allow_html=True)

with col2:
    st.subheader("📋 Screener Widget")
    tradingview_screener()
    

     




# --- Visualization Function (With Secondary Axis) ---
def visualize_stock_data(df):
    if df is not None:
        # Create Candlestick Chart
        fig = go.Figure()

        # Candlestick trace on secondary y-axis
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick",
            yaxis='y2'
        ))

        # Overlay Moving Averages
        if show_sma:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], mode='lines', name='SMA 20', yaxis='y2'))

        if show_ema:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], mode='lines', name='EMA 20', yaxis='y2'))

        # Bollinger Bands
        if show_bollinger:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Upper'], line=dict(color='rgba(255,0,0,0.3)'), name='Bollinger Upper', yaxis='y2'))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Lower'], line=dict(color='rgba(0,255,0,0.3)'), name='Bollinger Lower', yaxis='y2'))

        # Volume on primary y-axis
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['Volume'], name='Volume', marker_color='blue', opacity=0.4))

        # Layout Customization
        fig.update_layout(
            title=f'{selected_stock} Price and Volume (Last 30 Days)',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Volume', side='left'),
            yaxis2=dict(title='Price (USD)', overlaying='y', side='right'),
            xaxis_rangeslider_visible=False,
            height=600
        )

        # Plot RSI if selected
        if show_rsi:
            fig_rsi, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(x=df['timestamp'], y=df['RSI'], ax=ax, color='orange')
            ax.axhline(70, linestyle='--', color='red', label="Overbought (70)")
            ax.axhline(30, linestyle='--', color='green', label="Oversold (30)")
            ax.set_title(f"{selected_stock} RSI Indicator")
            ax.set_xlabel("Date")
            ax.set_ylabel("RSI Value")
            ax.legend()

            st.pyplot(fig_rsi)

        # Display Candlestick and Volume Chart
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No technical data available for the selected ticker and timeframe.")


# --- Main Execution ---
st.title(f"Technical Analysis for {selected_stock}")
technical_data = fetch_polygon_technical_data(selected_stock, timeframe)

if technical_data is not None:
    technical_data = calculate_indicators(technical_data)
    visualize_stock_data(technical_data)
else:
    st.warning("No data found for the selected timeframe.")



# ======== Option Chain & Rolling PUT Strategy ========
ticker = yf.Ticker(selected_stock)
expiration_dates = ticker.options
rolling_puts = pd.DataFrame()

for exp_date in expiration_dates:
    option_chain = ticker.option_chain(exp_date).puts
    filtered_puts = option_chain[option_chain['strike'] == strike_price]

    if not filtered_puts.empty:
        filtered_puts = filtered_puts.copy()
        filtered_puts['expirationDate'] = exp_date
        filtered_puts['tradeDate'] = (datetime.strptime(exp_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        filtered_puts['Trade Action'] = 'SELL'
        rolling_puts = pd.concat([rolling_puts, filtered_puts])

        # BUY Transaction
        filtered_puts['tradeDate'] = exp_date
        filtered_puts['Trade Action'] = 'BUY'
        rolling_puts = pd.concat([rolling_puts, filtered_puts])

rolling_puts = rolling_puts[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility',
                             'volume', 'openInterest', 'expirationDate', 'tradeDate', 'Trade Action']]
rolling_puts['Contracts'] = 1
rolling_puts['Premium Collected'] = rolling_puts.apply(lambda x: x['bid'] * 100 if x['Trade Action'] == 'SELL' else 0, axis=1)
rolling_puts['Buyback Cost'] = rolling_puts.apply(lambda x: x['ask'] * 100 if x['Trade Action'] == 'BUY' else 0, axis=1)
rolling_puts['P/L (Per Contract)'] = rolling_puts['Premium Collected'] - rolling_puts['Buyback Cost']
rolling_puts['Cumulative P/L'] = rolling_puts['P/L (Per Contract)'].cumsum()




col1, col2 = st.columns(2)
with col1:
    st.subheader(f"{selected_stock} - Recent Stock Data (Polygon)")
    stock_data = fetch_polygon_stock_data(selected_stock)

    if stock_data is not None:
        st.dataframe(stock_data)
    else:
        st.warning("No recent stock data available.")
    
with col2:
    st.subheader("Rolling PUT Strategy")
    st.write(rolling_puts)


# --- Fetch Latest News ---
def fetch_latest_news(ticker):
    articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt')
    return articles['articles']

if show_news:
    st.subheader("📰 Latest News")
    news = fetch_latest_news(selected_stock)
    for article in news[:5]:
        st.markdown(f"[{article['title']}]({article['url']}) - {article['source']['name']}")


# ======== Reddit Sentiment ========
st.subheader("Reddit Sentiment")
posts = reddit.subreddit('stocks').search(selected_stock, limit=5)
for post in posts:
    sentiment_score = TextBlob(post.title).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    st.write(f"[{post.title}]({post.url}) - Sentiment: {sentiment}")        

# ======== AI Query for Real-Time Insights ========
user_query = st.text_input("Ask about real-time stock trends, performance, or news")
if user_query:
    def ai_query(prompt):
        response = model.generate_content(prompt)
        return response.text

    query_result = ai_query(user_query)
    st.subheader("AI Query Results")
    st.write(query_result)
            
    
# --- Footer ---
st.markdown(
    "<div style='text-align:center;margin-top:20px;'>Data powered by <b>TradingView</b> and <b>Polygon.io</b>. Created by <b>Maaz Siddiqui</b></div>",
    unsafe_allow_html=True
)
