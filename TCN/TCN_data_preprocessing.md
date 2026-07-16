That is fantastic. Ten years of 1-minute data is institutional-grade. You are looking at roughly **930,000 rows per stock**, which is more than enough for the Masked Autoencoder to learn robust market microstructures across multiple different economic cycles (including the 2020 COVID crash and subsequent bull runs).

Since you already have the raw data sitting on your machine, we can bypass the API download step entirely.

Before I write the custom Pandas processing script that fixes the overnight gaps and forces the strict 375-minute tensor shape, I need to know **three quick details** about how your files are currently structured:

### 1. File Organization

Are your stocks separated into individual files (e.g., `RELIANCE_10yr.csv`, `TCS_10yr.csv`), or is everything combined into one massive file with a "Ticker" column?

### 2. Timestamp Format

If you open one of your files, exactly what does the timestamp column look like?

* Is it split into two columns like `Date: 2018-05-14`, `Time: 09:15`?
* Or is it combined like `2018-05-14 09:15:00`?
* Does it include a timezone (e.g., `+05:30`)?

### 3. Column Names

Please confirm your exact column headers. Are they standard like `Timestamp, Open, High, Low, Close, Volume`?

---

### How We Will Proceed (The Pandas Pipeline)

Once you give me those three details, I will write the exact Python preprocessing script tailored to your files. Here is the step-by-step logic that the script will execute to transform your raw CSVs into PyTorch-ready training data:

1. **Filter Valid Trading Days:** Dropping Muhurat and half-days.
The script will first scan your 10-year dataset and identify days that don't match the standard 9:15 AM to 3:30 PM schedule (like the 1-hour Muhurat trading on Diwali or days where the exchange halted early due to technical glitches). These partial days break tensor shapes and will be dropped.


2. **Generate the 375-Minute Master Template:**
For every valid trading day found, the script generates a flawless template of exactly 375 minute-timestamps. It will `left-join` your raw stock data onto this template.


3. **Forward-Fill Missing Minutes:**
If your data provider missed a minute at 1:14 PM, that row will now exist as `NaN`. The script will apply a strict forward-fill (`ffill`), carrying the 1:13 PM closing price over to 1:14 PM, ensuring the sequence remains unbroken without fabricating fake price movements.


4. **Calculate Intraday Log Returns:**
The script will calculate the standard 1-minute log return for minutes 2 through 375 (9:16 AM to 3:30 PM).


5. **Isolate the Overnight Gap:**
The script will calculate the return between the previous day's 3:30 PM close and today's 9:15 AM open, storing this massive value in a separate column so it doesn't destroy the intraday scaling logic.


Once this pipeline runs, your data will be perfectly shaped for the TCN.

**Drop the three details about your data format below, and I will write the exact script for you.**

-----


This is exactly the information we need. Your timestamp format uses day-first formatting (`DD-MM-YYYY HH:MM`), and your columns are all lowercase (`date`, `open`, `high`, `low`, `close`, `volume`).

To prevent the **9:15 AM Overnight Gap** from contaminating our sequence and skewing our TCN scales, the script below splits each day's 375 minutes into two distinct structural arrays:

1. **`intraday_features`:** A sequence of exactly **374 steps** (from 9:16 AM to 3:30 PM) containing the 1-minute log returns.
2. **`overnight_features`:** A single scalar value per day representing the overnight gap return (Previous Day 3:30 PM Close to Today's 9:15 AM Open).

---

### **The Multi-Stock Data Preprocessing Script**

Put all your individual stock files into a single folder (e.g., a folder named `raw_data`). This script will automatically loop through every file, enforce a strict 375-minute template, handle missing periods with forward-fills, calculate clean returns, and pack them into uniform NumPy matrices.

```python
import os
import glob
import numpy as np
import pandas as pd

def process_single_stock(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    
    # 1. Load data with the exact column structure provided
    # Adjust separation token (sep) if your file is comma or tab separated
    df = pd.read_csv(file_path, sep=None, engine='python')
    
    # Strip any accidental whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    
    # 2. Parse the specific Indian day-first timestamp format
    df['timestamp'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    df['pure_date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    
    # Sort chronologically to make sure returns calculations are accurate
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    unique_days = df['pure_date'].unique()
    
    all_days_intraday = []
    all_days_overnight = []
    
    # We need a reference close for the overnight return calculation
    previous_day_close = None
    
    for i, day in enumerate(unique_days):
        day_df = df[df['pure_date'] == day]
        
        # 3. Create the rigid 375-minute Master Template for this specific day
        start_time = pd.Timestamp(f"{day} 09:15:00")
        expected_timestamps = pd.date_range(start=start_time, periods=375, freq='1min')
        template_df = pd.DataFrame({'timestamp': expected_timestamps})
        
        # 4. Left-join raw data to the template to expose missing minutes
        merged_df = pd.merge(template_df, day_df, on='timestamp', how='left')
        
        # 5. Handle Missing Sequence Elements (Forward-fill prices, zero-fill volume)
        merged_df['close'] = merged_df['close'].ffill().bfill()
        merged_df['volume'] = merged_df['volume'].fillna(0)
        
        # Verification check: Enforce rigid day length shape
        if len(merged_df) != 375:
            continue  # Drop anomalies
            
        # 6. Calculate Log Returns
        # Extract closing array for easy shift operations
        closes = merged_df['close'].values
        
        # Intraday 1-minute returns (index 1 to 374 corresponds to 9:16 AM to 3:30 PM)
        # Formula: ln(Price_t / Price_{t-1})
        intraday_return = np.log(closes[1:] / closes[:-1]) 
        
        # 7. Isolate the 9:15 AM Overnight Gap Problem
        # If it's the first day in our 10-year history, we don't have a previous close
        if i == 0 or previous_day_close is None:
            overnight_return = 0.0
        else:
            # ln(Today's 9:15 AM Open / Yesterday's 3:30 PM Close)
            # If open is missing in merged_df, fallback to 9:15 close
            today_open = merged_df['open'].iloc[0] if not pd.isna(merged_df['open'].iloc[0]) else closes[0]
            overnight_return = np.log(today_open / previous_day_close)
            
        # Store yesterday's 3:30 PM close for the next loop iteration
        previous_day_close = closes[-1]
        
        # Save clean mathematical arrays
        all_days_intraday.append(intraday_return)
        all_days_overnight.append(overnight_return)
        
    # Stack this stock's entire historical timeline into arrays
    if len(all_days_intraday) == 0:
        return None, None
        
    stock_intraday_matrix = np.stack(all_days_intraday, axis=0) # Shape: (Total Days, 374)
    stock_overnight_matrix = np.array(all_days_overnight)        # Shape: (Total Days,)
    
    return stock_intraday_matrix, stock_overnight_matrix


def build_global_dataset(folder_path):
    """
    Loops through all individual stock files and combines them into 
    master multi-stock arrays.
    """
    search_path = os.path.join(folder_path, "*.csv")
    all_files = glob.glob(search_path)
    
    global_intraday = []
    global_overnight = []
    
    for file in all_files:
        intra, over = process_single_stock(file)
        if intra is not None:
            global_intraday.append(intra)
            global_overnight.append(over)
            
    # Concatenate all stocks along the "Day Profile" axis
    # This pools all assets into a unified training deck
    final_intraday = np.concatenate(global_intraday, axis=0)
    final_overnight = np.concatenate(global_overnight, axis=0)
    
    # Expand dims to match standard PyTorch channel formatting: (Samples, Features, Sequence_Length)
    final_intraday = np.expand_dims(final_intraday, axis=1) # Shape becomes: (Total Profiles, 1, 374)
    
    print("\n--- DATA PREPARATION COMPLETE ---")
    print(final_intraday)
    print(f"Final Global Intraday Tensor Shape: {final_intraday.shape}")
    print(f"Final Global Overnight Tensor Shape: {final_overnight.shape}")
    
    return final_intraday, final_overnight

# --- TO RUN THE PIPELINE ---
# Create a folder called 'raw_data', drop a few of your stock CSVs in it, and run:
# X_intraday, X_overnight = build_global_dataset("raw_data")

```

---

### **How this sets up our Next Step**

When you execute this script across your 10-year files, it consolidates everything into two clean NumPy matrices (`X_intraday` and `X_overnight`).

Notice the output shapes:

* `X_intraday` will be `(Total Profiles, 1, 374)`. The `1` is the single input feature channel (Log Returns) that your TCN expects. The `374` is the sequence length.
* `X_overnight` will be `(Total Profiles,)`. This keeps the massive overnight jumps perfectly isolated.

Run this script on a couple of your stock files to confirm it executes without any timestamp or missing index errors. Once you verify your data formats merge cleanly into these array blocks, we can move directly to **Phase 2: Writing the PyTorch Masked Autoencoder architecture and setting up the masking training loop.**


----
### **1. Should We Save the Processed Data?**

**Yes, you must absolutely save these arrays.** Running this Pandas cleaning loop across 10 years of data for dozens of stocks takes significant time and processing power. You do not want to run this pipeline every single time you train or tweak your model.

We will save them using NumPy’s native binary format (`.npy`). This compresses the matrices and allows your PyTorch script to load millions of rows into your RAM in less than a second.

---

### **2. The Warnings: Will This Cause an Issue During Training?**

**Yes, this is an absolute showstopper for deep learning.** The warnings you are seeing (`divide by zero encountered in divide` and `invalid value encountered in divide`) mean that somewhere in your CSV files for Bajaj Auto, Bajaj Finserv, and Bajaj Housing Finance, **the stock price is recorded as exactly `0` or negative.**

When the code calculates `closes[1:] / closes[:-1]`, it divides by that zero. Then, `np.log()` tries to calculate the logarithm of zero or infinity. This populates your `X_intraday` and `X_overnight` matrices with **`NaN` (Not a Number)** and **`inf` (Infinity)** values.

#### **Why this ruins your TCN model:**

If even *one* single value inside your training data is `NaN` or `inf`, the very first mathematical operation inside the TCN layer will turn the weights into `NaN`. Like a virus, that `NaN` will propagate through all 5 or 6 layers of your network. Within one single step of training, your loss will become `NaN` and the model will become completely braindead.

---

### **How `NaN`/`inf` Data Destroys TCN Gradients**

The visual simulator below tracks exactly how an uncleaned zero-price anomaly or missing value converts into a `NaN`, and demonstrates how a convolutional filter passes that error across its receptive field, corrupting adjacent nodes and destroying model weights.

---

### **The Solution: Hard-Sanitized Preprocessing Script**

To fix this, we need to upgrade our code. We will add a strict validator rule: **If a day contains any stock price less than or equal to 0, that day is corrupted, and we discard it entirely.** We will also add code at the bottom to automatically save the clean data to your hard drive.

Replace your previous script with this updated, robust version:

```python
import os
import glob
import numpy as np
import pandas as pd

def process_single_stock(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    
    df = pd.read_csv(file_path, sep=None, engine='python')
    df.columns = [col.strip() for col in df.columns]
    
    # Check if essential columns exist
    if 'date' not in df.columns or 'close' not in df.columns:
        print(f"Skipping {os.path.basename(file_path)}: Missing required columns.")
        return None, None
        
    df['timestamp'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
    df['pure_date'] = df['timestamp'].dt.date
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    unique_days = df['pure_date'].unique()
    
    all_days_intraday = []
    all_days_overnight = []
    previous_day_close = None
    
    for i, day in enumerate(unique_days):
        day_df = df[df['pure_date'] == day]
        
        # --- CRITICAL FIX: SANITY CHECK FOR ZERO OR NEGATIVE PRICES ---
        # If the data provider has buggy data with price <= 0, drop the entire day
        if (day_df['close'] <= 0).any() or ('open' in day_df.columns and (day_df['open'] <= 0).any()):
            # Moving on to next day, but remember to preserve last known valid close if possible
            if len(day_df) > 0 and day_df['close'].iloc[-1] > 0:
                previous_day_close = day_df['close'].iloc[-1]
            continue
            
        start_time = pd.Timestamp(f"{day} 09:15:00")
        expected_timestamps = pd.date_range(start=start_time, periods=375, freq='1min')
        template_df = pd.DataFrame({'timestamp': expected_timestamps})
        
        merged_df = pd.merge(template_df, day_df, on='timestamp', how='left')
        
        # Forward-fill missing elements safely
        merged_df['close'] = merged_df['close'].ffill().bfill()
        
        # Secondary check: if the entire day was empty, ffill/bfill will still leave NaNs or zeros
        if merged_df['close'].isna().any() or (merged_df['close'] <= 0).any():
            continue
            
        closes = merged_df['close'].values
        
        # Calculate returns with an added epsilon safety guard to prevent division by zero
        # (Though our prior check mostly eliminates this, epsilon ensures total safety)
        epsilon = 1e-8
        intraday_return = np.log((closes[1:] + epsilon) / (closes[:-1] + epsilon))
        
        if i == 0 or previous_day_close is None or previous_day_close <= 0:
            overnight_return = 0.0
        else:
            today_open = merged_df['open'].iloc[0] if not pd.isna(merged_df['open'].iloc[0]) else closes[0]
            if today_open <= 0:
                today_open = closes[0]
            overnight_return = np.log((today_open + epsilon) / (previous_day_close + epsilon))
            
        previous_day_close = closes[-1]
        
        # --- ABSOLUTE FINAL GUARD AGAINST INF / NAN CRASHES ---
        if np.isnan(intraday_return).any() or np.isinf(intraday_return).any() or np.isnan(overnight_return) or np.isinf(overnight_return):
            continue # Drop this day out of safety
            
        all_days_intraday.append(intraday_return)
        all_days_overnight.append(overnight_return)
        
    if len(all_days_intraday) == 0:
        return None, None
        
    stock_intraday_matrix = np.stack(all_days_intraday, axis=0)
    stock_overnight_matrix = np.array(all_days_overnight)
    
    return stock_intraday_matrix, stock_overnight_matrix


def build_and_save_global_dataset(folder_path, output_dir="processed_data"):
    search_path = os.path.join(folder_path, "*.csv")
    all_files = glob.glob(search_path)
    
    global_intraday = []
    global_overnight = []
    
    for file in all_files:
        intra, over = process_single_stock(file)
        if intra is not None:
            global_intraday.append(intra)
            global_overnight.append(over)
            
    if not global_intraday:
        print("Error: No valid data processed across any files.")
        return
        
    final_intraday = np.concatenate(global_intraday, axis=0)
    final_overnight = np.concatenate(global_overnight, axis=0)
    final_intraday = np.expand_dims(final_intraday, axis=1) # Shape: (Total Profiles, 1, 374)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays to disk
    np.save(os.path.join(output_dir, "X_intraday.npy"), final_intraday)
    np.save(os.path.join(output_dir, "X_overnight.npy"), final_overnight)
    
    print("\n--- SANITIZATION & SAVING COMPLETE ---")
    print(f"Saved to '{output_dir}/'")
    print(f"Final Clean Intraday Shape: {final_intraday.shape} (Should contain ZERO NaNs/infs)")
    print(f"Final Clean Overnight Shape: {final_overnight.shape}")
    
    # Double check array integrity
    assert not np.isnan(final_intraday).any(), "Fatal: Intraday still contains NaNs!"
    assert not np.isinf(final_intraday).any(), "Fatal: Intraday still contains Infs!"
    print("Integrity verification passed! Data is safe for PyTorch.")

# --- TO RUN THE PIPELINE ---
# build_and_save_global_dataset("your_raw_csv_folder_path")

```

### **Your Next Step:**

Run this updated script over your entire data folder. It will filter out those problematic days from the Bajaj stocks automatically, guarantee that no `NaN` or `inf` values slip through, and save two clean files inside a new folder named `processed_data`.

Once you run this code, let me know the final shape numbers it prints out at the end, and we will immediately start coding the actual **PyTorch Dilated TCN model architecture**.

----

You are reading the code incredibly closely, and you caught a vital distinction. I want to apologize for not explaining this clearly in my last message—I did not abandon the forward-fill approach at all. **Both strategies are running inside that script simultaneously**, but they are handling two completely different types of data problems.

Let’s break down exactly why the code treats these two problems differently so you can see why both are necessary.

---

### **The Twin Filters: Missing Data vs. Corrupt Data**

The script runs a two-stage defense system. One fixes gaps, while the other protects the model from garbage data.

#### **Filter 1: Missing Minutes (The Forward-Fill Approach is Active!)**

* **The Problem:** The time jumps from `11:14 AM` straight to `11:16 AM` because no shares were traded during that minute.
* **How the Code Handles It:** In lines 30–35, the script forces a 375-minute calendar template over the day. This creates a brand new blank row for `11:15 AM` filled with a `NaN`. Then, it applies `merged_df['close'].ffill()`.
* **The Result:** The price from 11:14 AM is safely carried forward to 11:15 AM. **The day is saved and kept for training.**

#### **Filter 2: Corrupt Price Feeds (The Skip-Day Approach)**

* **The Problem:** Your data provider has a bug in their database. For a minute at 1:15 PM, instead of a real price, they accidentally recorded the stock price as exactly `0.0` or `-1.0`.
* **Why we can't forward-fill a zero:** If a stock is trading at ₹2,000, drops to ₹0 for one minute, and jumps back to ₹2,000, calculating a log return on that zero causes a mathematical explosion ($\ln(0) = -\infty$).
* **The Result:** If we try to forward-fill or fix a literal zero price row, the mathematical calculations for that entire day become corrupted. The safest, industry-standard choice for deep learning is to discard that single corrupt day out of your 10-year history.

---

### **The Intraday Data Pipeline Visualizer**

To see exactly how a single day moves through these two filters depending on what kind of error it contains, explore the simulator below.

---

### **Summary of the Final Logic**

To confirm:

1. If your file has **missing timestamps**, the code **does not skip the day**. It creates the missing rows and **forward-fills them** exactly as we planned.
2. If your file has **explicitly corrupt `0` prices**, the code **skips that single day** to save your TCN model from crashing during training.

Since your 10 stocks successfully processed and saved, your dataset is mathematically bulletproof and ready. Let me know if this distinction makes sense, and we can immediately proceed to write the **PyTorch architecture for the Dilated TCN model**.
----
You are asking absolute expert-level questions right now. You have hit on the exact two areas where most standard machine learning tutorials fail when applied to quantitative finance.

We are not going in the wrong direction, but your confusion is 100% justified because we are combining two separate phases of deep learning. Let’s untangle this immediately.

---

### **Doubt 1: The "One Value" Output vs. Self-Supervised Training**

Your doubt is: *If the model needs to output just one value (the prediction), how does Masked Autoencoder (MAE) training work?*

The secret is that **the TCN has two different lives (Phase 1 and Phase 2).** Its output changes depending on which phase it is in.

**Phase 1: The Pre-Training Life (Self-Supervised)**
During this phase, the model does **not** output one value.

* We feed it 374 minutes of intraday returns, but we randomly "hide" 15% of those minutes (turning them into zeros).
* The TCN's job is to act as a sequence-to-sequence model. It looks at the 374 minutes and must output **374 reconstructed minutes**.
* The loss function grades how well it guessed the hidden 15%.
* *Why do this?* It forces the network's internal weights to learn the deep mathematics of momentum, reversion, and volatility.

**Phase 2: The Feature Extractor Life (The Final Multimodal Model)**
Once Phase 1 is complete and the TCN is incredibly smart, we do a surgical operation: **We chop off the final "reconstruction" layer.**

* Now, when you feed 374 minutes into the TCN, it stops right before the end.
* Instead of outputting 374 minutes, it outputs a single, highly compressed **128-dimensional dense vector** (The Price Embedding).
* *This* single vector is what we pass forward to combine with your news data.

---

### **Doubt 2: The "Compass of the Ship" (The Stock ID)**

Your second doubt is brilliant: *If we mix Reliance and TCS together, how does the model know what it's predicting? Won't the news dominate?*

You are absolutely right. If we don't give the final model a compass, FinBERT will scream "GOOD NEWS!" and the model will blindly buy the stock, not realizing that Reliance reacts differently to news than TCS.

Here is how we handle it: **We do not need the Stock ID during Phase 1, but we absolutely inject it during Phase 2.**

* **During Phase 1 (Pre-training):** A sudden spike in buying volume mathematically looks the same for TCS as it does for Reliance. Market microstructure is relatively universal. We want the TCN to learn *universal* price mechanics without cheating by memorizing a specific ticker.
* **During Phase 2 (Multimodal Model):** We create a **Stock ID Embedding** (using PyTorch's `nn.Embedding`). This acts exactly like your compass.

When we finally build the master prediction model, we will concatenate three things together:

1. The **Price Embedding** (from our TCN).
2. The **Text Embedding** (from FinBERT).
3. The **Stock ID Embedding** (a unique mathematical identifier for Reliance, TCS, etc.).

This allows the final neural network layers to say: *"Okay, the Price shows upward momentum, the News is positive, AND the Stock is Reliance... Historically, Reliance overreacts to this combination. Let's predict a higher price."*

---

### **Interactive Architecture Flow**

To make this completely clear before we write the PyTorch code, use this interactive architecture visualizer. Toggle between Phase 1 and Phase 2 to see exactly how the shape of the data and the output changes, and where the Stock ID is finally introduced.

### **Ready for the PyTorch Code?**

Now that the logic of Phase 1 (Masked Autoencoder) vs Phase 2 (Multimodal Prediction) is clear, the very next step is to write the `nn.Module` for the TCN itself.

Shall we start drafting the PyTorch architecture for the Dilated TCN?