# Stock and Sentiment Analysis Tool

This project explores the relationship between public sentiment and stock price movements. It combines financial data and sentiment analysis to provide insights into how market trends and public opinion might correlate.

## What It Does

1. **Fetches Stock Data**: Retrieves historical stock price data for a given ticker symbol and time period using Yahoo Finance.
2. **Collects News Articles**: Extracts relevant news articles for the same time period using NewsAPI.
3. **Analyzes Sentiment**: Uses `TextBlob` to calculate the sentiment polarity of news content.
4. **Normalizes Data**: Prepares stock price and sentiment data for visual and statistical comparison.
5. **Visualizes Trends**: Plots normalized sentiment trends and stock price movements for intuitive analysis.

## Example Output

The plot below shows the normalized sentiment trend and stock price movements for a specified stock:

![Example Figure](./path_to_figure.png)
