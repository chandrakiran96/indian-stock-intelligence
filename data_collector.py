import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np

# ---------------------------------------------------------
# 1. MACRO DATA COLLECTION (Automated via FRED)
# ---------------------------------------------------------
def fetch_indian_macro_data(start_date, end_date):
    """
    Fetches 10 years of Indian macroeconomic data from FRED.
    These are monthly/quarterly metrics that we will forward-fill.
    """
    print("Fetching Macro Data from FRED...")
    
    # FRED Series IDs for India
    indicators = {
        'INDCPIALLMINMEI': 'CPI_Inflation',      # Consumer Price Index
        'INTDSRINM193N': 'Interest_Rate',        # Central Bank Discount Rate
        'RBINBIS': 'Real_Exchange_Rate'          # Broad Effective Exchange Rate
    }
    
    macro_df = web.DataReader(list(indicators.keys()), 'fred', start_date, end_date)
    macro_df.rename(columns=indicators, inplace=True)
    
    # Macro data is usually monthly. We need a daily grid to align with stock markets.
    # Create a daily calendar and forward-fill the last known macro values.
    daily_index = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' = Business days
    macro_daily = macro_df.reindex(daily_index)
    macro_daily.fillna(method='ffill', inplace=True)
    
    # Drop NaNs that appear before the first release of a metric
    macro_daily.dropna(inplace=True)
    print(f"Macro data shape after daily alignment: {macro_daily.shape}")
    return macro_daily

# ---------------------------------------------------------
# 2. NEWS EMBEDDING STUB
# ---------------------------------------------------------
def load_and_tag_news_embeddings(stock_ticker, start_date, end_date):
    """
    Stub for the news pipeline.
    In reality, you will:
    1. Load the Kaggle CSV + Scraped Data.
    2. Filter rows where text contains the stock_ticker (e.g., 'RELIANCE').
    3. Pass text through FinBERT to get a 768-D vector.
    4. Group by Date and mean-pool the embeddings.
    """
    print(f"Mocking FinBERT embeddings for {stock_ticker}...")
    
    # Generating dummy daily FinBERT embeddings (768 dimensions)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    dummy_embeddings = np.random.randn(len(dates), 768) 
    
    news_df = pd.DataFrame(dummy_embeddings, index=dates)
    return news_df

# ---------------------------------------------------------
# 3. WINDOW ALIGNMENT (Preparing the Tensors)
# ---------------------------------------------------------
def create_multimodal_windows(price_1min_df, news_daily_df, macro_daily_df, lookback_days=10, horizon_days=5):
    """
    Aligns 1-min prices, daily news, and daily macro into the sliding windows.
    """
    print("Aligning data into multimodal sliding windows...")
    X_price, X_news, X_macro, Y_labels = [], [], [], []
    
    # Identify unique trading days in the price data
    trading_days = np.sort(price_1min_df['Date'].dt.date.unique())
    
    for i in range(lookback_days, len(trading_days) - horizon_days):
        anchor_date = trading_days[i]
        lookback_start = trading_days[i - lookback_days]
        future_date = trading_days[i + horizon_days]
        
        # 1. Slice Price Data (Extract the exact 1-min bars for the 10-day window)
        price_window = price_1min_df[
            (price_1min_df['Date'].dt.date > lookback_start) & 
            (price_1min_df['Date'].dt.date <= anchor_date)
        ]
        # In practice, ensure price_window is exactly padded/truncated to (3750, Features)
        
        # 2. Slice News & Macro Data (Daily values for the 10-day window)
        news_window = news_daily_df.loc[str(lookback_start):str(anchor_date)]
        macro_window = macro_daily_df.loc[str(lookback_start):str(anchor_date)]
        
        # 3. Create Label (Calculate 5-day forward return)
        current_close = price_window.iloc[-1]['Close']
        future_close = price_1min_df[price_1min_df['Date'].dt.date == future_date].iloc[-1]['Close']
        
        forward_return = (future_close - current_close) / current_close
        
        # Label mapping: 1 (UP), 0 (FLAT), -1 (DOWN)
        if forward_return > 0.02: label = 1
        elif forward_return < -0.02: label = -1
        else: label = 0
            
        X_price.append(price_window.values)
        X_news.append(news_window.values)
        X_macro.append(macro_window.values)
        Y_labels.append(label)
        
    return X_price, X_news, X_macro, Y_labels

if __name__ == "__main__":
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2024, 1, 1)
    
    # Step 1: Get Macro
    macro_data = fetch_indian_macro_data(start, end)
    
    # Step 2: Get News (Mocked for now)
    news_data = load_and_tag_news_embeddings('RELIANCE', start, end)
    
    print("\nPhase 1 Architecture Setup Complete. Ready to inject real 1-min data.")