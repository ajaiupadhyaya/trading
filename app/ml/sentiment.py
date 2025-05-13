import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class MarketSentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_news_sentiment(self, symbol):
        # Get news from Yahoo Finance
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return None
            
        # Analyze sentiment for each news item
        sentiments = []
        for item in news:
            # Get title and summary
            title = item.get('title', '')
            summary = item.get('summary', '')
            
            # Combine title and summary
            text = f"{title} {summary}"
            
            # Get sentiment scores
            sentiment = self.sia.polarity_scores(text)
            sentiments.append({
                'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                'title': title,
                'compound': sentiment['compound'],
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu']
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(sentiments)
        
        # Calculate aggregate sentiment
        if not df.empty:
            recent_sentiment = df['compound'].mean()
            sentiment_trend = df['compound'].diff().mean()
            
            return {
                'current_sentiment': recent_sentiment,
                'sentiment_trend': sentiment_trend,
                'news_count': len(df),
                'sentiment_details': df.to_dict('records')
            }
        return None
        
    def get_social_sentiment(self, symbol):
        # This is a placeholder for social media sentiment analysis
        # In a production environment, you would integrate with Twitter, Reddit, etc.
        return {
            'twitter_sentiment': 0,
            'reddit_sentiment': 0,
            'overall_social_sentiment': 0
        }
        
    def get_market_sentiment(self, symbol):
        # Combine news and social sentiment
        news_sentiment = self.analyze_news_sentiment(symbol)
        social_sentiment = self.get_social_sentiment(symbol)
        
        if news_sentiment is None:
            return None
            
        # Calculate overall sentiment score
        overall_sentiment = (
            news_sentiment['current_sentiment'] * 0.7 +  # Weight news more heavily
            social_sentiment['overall_social_sentiment'] * 0.3
        )
        
        return {
            'overall_sentiment': overall_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'sentiment_signal': 'bullish' if overall_sentiment > 0.2 else 'bearish' if overall_sentiment < -0.2 else 'neutral'
        }
        
    def should_adjust_position(self, symbol, current_position):
        sentiment = self.get_market_sentiment(symbol)
        if sentiment is None:
            return False, None
            
        # Adjust position based on sentiment
        if current_position == 'call' and sentiment['sentiment_signal'] == 'bearish':
            return True, 'close'
        elif current_position == 'put' and sentiment['sentiment_signal'] == 'bullish':
            return True, 'close'
        elif current_position is None:
            if sentiment['sentiment_signal'] == 'bullish':
                return True, 'call'
            elif sentiment['sentiment_signal'] == 'bearish':
                return True, 'put'
                
        return False, None 