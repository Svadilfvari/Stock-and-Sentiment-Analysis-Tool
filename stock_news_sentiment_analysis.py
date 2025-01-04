import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests

# Function to fetch historical stock price data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)  # Reset index to convert 'Date' index to a column
    return stock_data

# Function to fetch news articles
def get_news_data(api_key, query, start_date, end_date):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': start_date,
        'to': end_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['status'] == 'error':
        raise Exception(f"NewsAPI Error: {data['message']}")
    return pd.DataFrame(data['articles'])

# Analyze sentiment using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Example Usage
api_key = '672991f967e74f01a4386d3254d911c5'
ticker = 'AAPL'
start_date = '2024-10-03'
end_date = '2024-11-01'

# Fetch data
stock_data = get_stock_data(ticker, start_date, end_date)
news_data = get_news_data(api_key, 'Apple', start_date, end_date)

# Rename columns to avoid conflict
stock_data.rename(columns={'Date': 'StockDate'}, inplace=True)
news_data.rename(columns={'publishedAt': 'NewsDate'}, inplace=True)
stock_data['StockDate'] = pd.to_datetime(stock_data['StockDate']).dt.date
news_data['NewsDate'] = pd.to_datetime(news_data['NewsDate']).dt.date

# Analyze sentiment
news_data['sentiment'] = news_data['content'].apply(analyze_sentiment)

# Merge datasets
merged_data = pd.merge(stock_data, news_data, left_on='StockDate', right_on='NewsDate', how='inner')

# Normalize data
scaler = MinMaxScaler()

# Normalize sentiment
sentiment_normalized = scaler.fit_transform(merged_data[['sentiment']])
merged_data['sentiment_normalized'] = sentiment_normalized

# Remove duplicates by keeping the minimum sentiment value
merged_data = merged_data.loc[merged_data.groupby('StockDate')['sentiment'].idxmin()]

# Normalize stock prices (using closing prices here)
stock_prices_normalized = scaler.fit_transform(merged_data[['Close']])
merged_data['Close_normalized'] = stock_prices_normalized

# Calculate daily stock returns
merged_data['Stock_Return'] = merged_data['Close_normalized'].pct_change()

# Create lagged sentiment values for Granger causality test

# max_lag = 1  # Define the maximum number of lags to include
# for lag in range(1, max_lag + 1):
#     merged_data[f'sentiment_lag_{lag}'] = merged_data['sentiment_normalized'].shift(lag)

# # Drop rows with any NaN or infinite values in the necessary columns
# lagged_columns = ['Stock_Return'] + [f'sentiment_lag_{lag}' for lag in range(1, max_lag + 1)]
# merged_data = merged_data[lagged_columns].replace([float('inf'), -float('inf')], float('nan')).dropna()

# # Check if there are any NaNs left in the data
# if merged_data[lagged_columns].isnull().any().any():
#     raise ValueError("NaNs detected after dropping — inspect data preparation steps.")

# # Run Granger causality test on Stock_Return with lagged sentiment values
# print("Running Granger causality test...")
# granger_results = grangercausalitytests(
#     merged_data[lagged_columns], 
#     max_lag, 
#     verbose=True
# )






# Plotting
plt.figure(figsize=(14, 7))

# Plot normalized sentiment trend
plt.plot(merged_data['NewsDate'], merged_data['sentiment_normalized'], label='Normalized Sentiment', color='blue')

# Plot normalized stock prices
plt.plot(merged_data['NewsDate'], merged_data['Close_normalized'], label='Normalized Stock Close Price', color='green')

# Adding titles and labels
plt.title(f'Normalized Sentiment and Stock Prices for {ticker}')
plt.xlabel('Date')
plt.ylabel('Normalized Values')
plt.legend()

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
