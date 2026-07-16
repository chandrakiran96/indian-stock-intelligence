This is an exceptionally well-thought-out blueprint for handling high-frequency Indian equity data. You have correctly identified the core failure modes of standard architectures on the NSE: **attention dispersion** in Transformers due to micro-noise, and **modality collapse** where a model ignores price features entirely to overfit on text/news data.

Using a **Dilated Temporal Convolutional Network (TCN)** as a self-supervised **Masked Autoencoder (MAE)** is an industry-grade approach. It forces the network to learn the mathematical representation of intraday market microstructure (like volume-weighted momentum and mean reversion) before introducing cross-modal inputs.

Before we write the PyTorch code, let’s resolve the mathematical foundation of your TCN: its **Receptive Field**. To ensure your model can "see" the entire 375-minute Indian trading day from the final layer, your kernel size ($k$), dilation base ($b$), and number of layers ($L$) must satisfy:

$$R = 1 + \sum_{l=0}^{L-1} (k - 1) \cdot b^l$$

---

### **Interactive TCN Configuration & Tensor Flow Simulator**

Use this tool to experiment with different TCN hyperparameters. It will automatically compute your network's total receptive field, verify if it fully covers the **375-minute NSE session**, and track how your input tensor transforms across the pipeline.

---

### **Core Microstructure Doubts to Clarify First**

Before moving into data loading and model building, we need to address three subtle pitfalls unique to the Indian market structure:

#### **1. The 9:15 AM Overnight Gap Problem**

* **The Issue:** The log return from 3:29 PM (previous day) to 9:15 AM (current day) contains massive overnight structural variance (driven by global markets, SGX Nifty/GIFT Nifty, and earnings releases).
* **The Pitfall:** If you treat this return exactly like an intraday 1-minute return, its extreme volatility will dominate the MSE loss during your Masked Autoencoder training, causing the network to ignore subtle intraday patterns.
* **The Fix:** We should create an auxiliary input channel or standardize overnight returns separately from the intraday returns.

#### **2. Zero-Volume Minutes & Illiquidity**

* **The Issue:** Even for high-liquidity NIFTY 50 stocks, there are minutes (especially mid-day around 1:00 PM) where the price doesn't change, resulting in a log return of exactly `0.0`.
* **The Pitfall:** If 40% of your day consists of zero returns, a masked autoencoder can cheat by simply predicting `0.0` everywhere to achieve low MSE loss.
* **The Fix:** We must implement a custom loss function that applies a higher weight to non-zero or high-volatility minutes during the masking reconstruction step.

#### **3. Handling Missing Sequence Elements**

* **The Issue:** Occasionally, data providers omit minutes where no trades occurred, resulting in days with 372 or 374 rows instead of 375.
* **The Pitfall:** PyTorch convolutions require rigid tensor dimensions. A single missing minute breaks your entire batch sequence.
* **The Fix:** We must build a strict forward-fill or alignment mechanism in our preprocessing script to guarantee every single day tensor is exactly padded/aligned to 375 elements.

---

### **How would you like to proceed?**

To guide our next step properly, what specific doubts do you have about this architecture, and what is the current status of your historical 1-minute data?



-----

It is completely natural to find this confusing. Transitioning from tabular data (like a spreadsheet or a standard pandas DataFrame) to the **3D and 4D Tensors** required by deep learning models is one of the steepest conceptual hurdles in quantitative finance.

Let’s demystify exactly how we structure, stack, and slice multi-stock data across multiple years without writing a single line of code.

---

### **The Three-Step Data Hierarchy**

Instead of thinking about a massive 5-year timeline, we break the data down into independent units called **Daily Profiles**. For self-supervised pre-training (Masked Autoencoder), the model doesn't need to know if a sequence belongs to TCS on a Tuesday or Reliance on a Friday. It just needs to look at 375 sequential points and learn market structure.

#### **Level 1: The Atomic Unit (1 Stock, 1 Day)**

For any single stock (e.g., RELIANCE) on a single trading day, your data is a straightforward 2D matrix:

* **Rows:** 375 (every minute from 9:15 AM to 3:30 PM).
* **Columns:** 1 (the Log Return for that minute).
* **Shape:** `(375, 1)`

#### **Level 2: The Global Dataset (All Stocks, All Years Pool)**

An average Indian trading year has roughly **250 trading days** (excluding weekends and market holidays).

* 5 Years $\times$ 250 Days = **1,250 trading days per stock**.
* If you train on **10 stocks**, you have: 10 stocks $\times$ 1,250 days = **12,500 unique daily profiles**.

We stack these 12,500 daily profiles on top of each other like a deck of cards. This creates a massive 3D matrix (a 3D Tensor):

* **Global Tensor Shape:** `(12500, 375, 1)`
* Meaning: `(Total Daily Profiles, Minutes per Day, Features)`

#### **Level 3: The PyTorch Swap & Batching**

PyTorch 1D Convolutional networks expect input features to come *before* time steps. So, during preprocessing, we flip the dimensions slightly:

* **Transposed Shape:** `(12500, 1, 375)` — `(Total Daily Profiles, Channels/Features, Minutes)`

When training begins, you don't feed all 12,500 days at once. You feed them in small groups called **Batches** (e.g., a Batch Size of 64). PyTorch randomly grabs 64 days out of your 12,500 "deck of cards" and hands it to the model.

* **Batch Tensor Shape:** `(64, 1, 375)`

---

### **Interactive Tensor Shape & Batch Explorer**

To visualize how changing your universe size (number of stocks or years) alters your data dimensions and how PyTorch slices them into training batches, adjust the parameters below.

---

### **Key Takeaway Before Moving Forward**

By organizing data this way, your TCN gets exposed to thousands of different market regimes. It sees high-volatility days, flat mid-day sideways trends, and massive gap-up mornings across multiple assets simultaneously. This prevents the model from overfitting to one specific company's quirks and forces it to become a **universal reader of NSE intraday price action**.

Now that the data layout and shape concepts are crystal clear, **which of the three microstructure problems should we tackle first?**

1. **The 9:15 AM Overnight Gap Problem** (Separate scaling vs. joint scaling)
2. **The Zero-Volume / Illiquidity "Cheating" Problem** (Custom weighted MSE loss)
3. **The Missing Sequence Elements Problem** (Rigid forward-fill alignment pipelines)

---

You are spot on with your thinking about how to use this model.

To clarify your first point: **This architecture is not strictly for day trading.** The Dilated TCN acts like a high-definition microscope. Its only job is to watch 375 minutes of rapid-fire price action and compress it into a single, dense mathematical summary (the Price Embedding Vector).

If you are an intraday trader, you can feed that vector directly into a classifier to predict the 3:25 PM closing price. However, if you are a long-term investor, you take that same vector and feed it into a slower model (like an LSTM) alongside daily news and macroeconomic data to predict the stock's price 3 months from now. The TCN just ensures your long-term model understands the "true" daily action, rather than just looking at a flat daily closing price.

Now, let's break down the three massive data problems unique to high-frequency trading and exactly how we will fix them in our data pipeline.

---

## 1. The 9:15 AM Overnight Gap Problem

**The Issue:** The Indian market is heavily influenced by what happens in the US and Asian markets while India is asleep. This means the price difference between yesterday's 3:30 PM close and today's 9:15 AM open is often violent—a massive "gap" up or down.

**The Pitfall:** If you treat that 17-hour overnight jump as just another "1-minute log return" in your sequence, the mathematical value will be so massive that it skews your entire dataset. Your autoencoder will panic, focus entirely on predicting the morning gap, and ignore the subtle intraday momentum you actually want it to learn.

**The Fix (Separation & Scaling):**
In our preprocessing script, we will calculate the overnight return (Previous Close to Today's Open) separately.

1. We will standard-scale the 374 intraday minutes together.
2. We will standard-scale the 1 overnight minute separately.
3. We pass the overnight gap as a distinct, parallel feature to the model, rather than mixing it blindly into the 1-minute sequence.

---

## 2. The Zero-Volume / Illiquidity "Cheating" Problem

**The Issue:** Even in highly liquid stocks like RELIANCE or HDFCBANK, the mid-day session (around 1:00 PM to 2:00 PM) often flatlines. You will have multiple consecutive minutes where the price does not move a single rupee, resulting in a log return of exactly `0.0`.

**The Pitfall:** A Masked Autoencoder learns by trying to guess missing data to minimize Mean Squared Error (MSE). If 30% of your trading day is mathematically `0.0`, the neural network will realize it can "cheat." It will just predict `0.0` for everything, achieving a very low error rate without actually learning any market dynamics.

**The Fix (Weighted Loss Function):**
We cannot use standard PyTorch MSE loss. We will write a custom loss function. During training, if the model incorrectly guesses a minute where a large price swing happened, we will multiply the penalty by 3x. If it incorrectly guesses a flat `0.0` minute, we apply a normal 1x penalty. This forces the network to care deeply about volatility.

---

## 3. The Missing Sequence Elements Problem

**The Issue:** The real world of NSE data is messy. Sometimes your broker's API drops a minute. Sometimes a stock hits a circuit breaker and trading halts for 15 minutes. This leaves you with a day that has 368 rows instead of the mandatory 375.

**The Pitfall:** PyTorch Convolutional layers are rigid. If you hand the TCN a batch where Monday has 375 minutes but Tuesday only has 374, the tensor matrix misaligns, and your training script will instantly crash with a `RuntimeError: size mismatch`.

**The Fix (The Rigid Forward-Fill Pipeline):**
Before the data ever touches PyTorch, our pandas script will generate a "Master Template" of exactly 375 minute timestamps (9:15 to 3:30) for every single calendar trading day. We will `left-join` your raw stock data onto this template. Any missing minutes will instantly become `NaN`. We will then use a strict "Forward-Fill" (`ffill`) method, carrying the last known closing price forward so the log return naturally becomes `0.0` without breaking the tensor shape.

---

-----


To answer your first question directly: **Yes, 5 years is absolutely perfect.** In high-frequency data, 5 years is massive.

* 5 years $\times$ 250 trading days = 1,250 days.
* 1,250 days $\times$ 375 minutes = **468,750 rows per stock**.
* If you do this for 10 NIFTY 50 stocks, you are looking at nearly **4.7 million rows** of data.

This is the sweet spot. It is enough data for the Masked Autoencoder to learn almost every possible market condition (Covid crash, budget days, flat summers) without being so large that your RAM crashes when you try to load it.

### The Reality Check: Pulling 1-Minute Indian Data

Here is the candid truth about algorithmic trading in India: **You cannot pull 5 years of 1-minute historical NSE data for free using standard open Python APIs.**

Libraries like `yfinance` only store 1-minute data for the **last 30 days** due to server costs. NSE's own website restricts historical intraday scraping. If you want 5 years of high-frequency data, you have three options:

1. **The Free Archive Route (Highly Recommended for Now):** Go to Kaggle.com and search for "NIFTY 50 1-minute historical data." Many researchers upload 5-to-10-year dumps of NSE intraday data as massive CSV or Parquet files for free.
2. **The Broker Route:** If you have an active account with Indian brokers like **Fyers** or **Zerodha (Kite Connect)**, they allow you to pull historical 1-minute data via their Python APIs. Fyers API is currently free and goes back roughly 2–3 years, while Zerodha charges ₹2,000/month for historical data access.
3. **The Paid Vendor Route:** Buying standardized, clean 5-year data from TrueData or GlobalDataFeeds (costs a few thousand rupees but guarantees zero missing minutes).

---

### Phase 1: The "Pipeline Builder" Python Script

Because you shouldn't pay for data or waste hours downloading gigabytes of CSVs until your Python code actually works, **we are going to build the pipeline using the last 30 days of free data first.** Once our TCN runs flawlessly on 30 days of data, you can simply swap out the data source for your 5-year Kaggle CSVs.

Here is the exact Python code to pull 1-minute NSE data for free right now using `yfinance` to get us started.

```python
import yfinance as yf
import pandas as pd
import datetime

def fetch_free_nse_intraday(ticker_symbol, days=7):
    """
    Fetches the maximum allowed free 1-minute data (up to 7-30 days depending on yfinance limits).
    Note: For Indian stocks on yfinance, you MUST append '.NS' to the symbol.
    """
    # e.g., 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'
    ticker = f"{ticker_symbol}.NS"
    
    print(f"Fetching 1-minute data for {ticker}...")
    
    # yfinance allows 1m data for a maximum of 7 days per API call, 
    # but we can get up to 30 days if we loop. For testing, 7 days is plenty.
    data = yf.download(
        tickers=ticker, 
        period=f"{days}d", 
        interval="1m",
        progress=False
    )
    
    if data.empty:
        print(f"Warning: No data found for {ticker}")
        return pd.DataFrame()
        
    # Clean up the dataframe
    data.reset_index(inplace=True)
    
    # yfinance sometimes names the index 'Datetime', let's standardize it
    if 'Datetime' in data.columns:
        data.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        
    # Convert to localized timezone (Indian Standard Time)
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.tz_convert('Asia/Kolkata')
    
    # Keep only the essential columns
    # Note: Depending on yf version, columns might be MultiIndex. We flatten them if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    data = data[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"Successfully downloaded {len(data)} minutes of data.")
    return data

# --- TEST THE SCRIPT ---
if __name__ == "__main__":
    # Let's test with Reliance Industries
    reliance_data = fetch_free_nse_intraday("RELIANCE", days=7)
    
    # Show the first 5 rows
    print(reliance_data.head())

```

### What to do next:

1. Copy the code above into your IDE (Jupyter, VS Code, or Google Colab) and run it.
2. Ensure it successfully prints out a clean table with `Timestamp`, `Open`, `High`, `Low`, `Close`, and `Volume`.

Once you confirm this script works on your machine, we will move to the exciting part: writing the pandas script that fixes the **Overnight Gap** and mathematically enforces the rigid **375-minute limit** for our PyTorch tensors.

Are you able to run this script, or do you want me to write the Kaggle CSV loading script instead assuming you already have a file?

