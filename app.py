import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import openpyxl
import praw
import google.generativeai as genai
from newsapi import NewsApiClient
from textblob import TextBlob
import streamlit as st


genai.configure(api_key="AIzaSyAXukbX28-gUndSdFOOGSU9d5SQ5FzxIFc")
model = genai.GenerativeModel('gemini-pro')

reddit = praw.Reddit(
    client_id='IqXpaotDyX364rHDlOTbww',
    client_secret='DcD7NG9Rb0II3aYWRV3KikhsgAfCSg',
    user_agent='Southern_Screen409'
)

newsapi = NewsApiClient(api_key='f8d7120cc5fb44a0ae8e9a2ba56df18a')


# ======== Sidebar - Stock Selection ========
st.sidebar.header("Stock Selection & Analysis")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="TSLA")
strike_price = st.sidebar.number_input("Strike Price", value=450)

# ======== Real-Time Price Fetching ========
def fetch_realtime_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period='1d')['Close'].iloc[-1]
    return price

current_price = fetch_realtime_price(selected_stock)
st.sidebar.metric("Current Price", f"${current_price:.2f}")


# ======== Real-Time Stock Predictions ========
def predict_stock_performance(ticker):
    prompt = f"""
    Predict the performance of {ticker} over the next 7 days.
    Use technical analysis and recent financial news to provide BUY/SELL recommendations.
    Highlight growth opportunities, risks, and trends influencing price movement.
    """
    response = model.generate_content(prompt)
    return response.text

st.title(f"{selected_stock} - Advanced Stock Analysis")
st.subheader("AI Stock Predictions (Next 7 Days)")
prediction_report = predict_stock_performance(selected_stock)
st.write(prediction_report)


# ======== AI Financial Report ========
def generate_financial_report(ticker, price):
    prompt = f"""
    Provide a financial overview for {ticker}. The current stock price is ${price:.2f}.
    Analyze market trends, risks, and potential catalysts. Recommend if investors should BUY, SELL, or HOLD.
    """
    response = model.generate_content(prompt)
    return response.text

financial_report = generate_financial_report(selected_stock, current_price)
st.subheader("AI-Driven Financial Report")
st.write(financial_report)


# ======== AI Query for Real-Time Insights ========
user_query = st.text_input("Ask about real-time stock trends, performance, or news")
if user_query:
    def ai_query(prompt):
        response = model.generate_content(prompt)
        return response.text

    query_result = ai_query(user_query)
    st.subheader("AI Query Results")
    st.write(query_result)


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


# ======== Export to Excel ========
output_path = f"{selected_stock}_{strike_price}_PUT_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    rolling_puts.to_excel(writer, sheet_name=f'{selected_stock}_PUTS', index=False)
print(f"Simulation saved to {output_path}")


# ======== TradingView Chart ========
st.subheader("TradingView Advanced Chart")
st.markdown(f"""
<iframe
  src="https://www.tradingview.com/embed-widget/advanced-chart/?symbol={selected_stock}"
  width="100%" height="500"
  frameborder="0"
  allowtransparency="true"
></iframe>
""", unsafe_allow_html=True)


# ======== Rolling PUT Table ========
st.subheader("Rolling PUT Strategy")
st.write(rolling_puts)


# ======== Real-Time News ========
st.subheader("Latest News & Sentiment")
news_articles = newsapi.get_everything(q=selected_stock, language='en', sort_by='publishedAt', page_size=5)
for article in news_articles['articles']:
    sentiment_score = TextBlob(article['title']).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    st.write(f"[{article['title']}]({article['url']}) - Sentiment: {sentiment}")


# ======== Reddit Sentiment ========
st.subheader("Reddit Sentiment")
posts = reddit.subreddit('stocks').search(selected_stock, limit=5)
for post in posts:
    sentiment_score = TextBlob(post.title).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    st.write(f"[{post.title}]({post.url}) - Sentiment: {sentiment}")
