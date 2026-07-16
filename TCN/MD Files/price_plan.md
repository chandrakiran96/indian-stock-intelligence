To build a robust institutional-grade price embedding from **60-minute log returns**, we need to evaluate exactly what structural properties your network must capture.

By choosing 60-minute log returns, you have already made an excellent choice: log returns are stationary (unlike raw prices), and a 60-minute interval significantly reduces high-frequency market microstructure noise (like bid-ask bounce) while preserving intraday momentum.

However, a standard trading month has roughly 140 to 160 hourly bars. This means your input sequence length is relatively short, making it highly susceptible to overfitting if your embedding network is too large, or underfitting if it cannot handle the multi-scale nature of volatility.

---

### **Three High-Alpha Architectures for Your Price Embedding Box**

Here are the three most effective, production-tested architectures to fill your missing block, ranked by their specific engineering benefits.

#### **1. The Multi-Scale Wavelet-GRU (Best for Regime Switching)**

Instead of passing raw 60-min returns, you apply a **Discrete Wavelet Transform (DWT)** to break the returns into high-frequency details (intraday shocks) and low-frequency approximations (weekly trends).

```
                  ┌──> [Detail Coeffs (Noise/Shocks)] ──> GRU Alpha ──┐
[60-min Returns] ─┼                                                    ├──> [Concat/Dense] ──> Price Embedding
                  └──> [Approx Coeffs (Trends)] ────────> GRU Beta ───┘

```

* **Why it works:** Markets behave differently across time horizons. Volatility traders care about high frequencies; macro traders care about low frequencies. This architecture explicitly splits them so separate recurrent units can process them before creating the final vector.
* **Implementation Tip:** Use a Gated Recurrent Unit (GRU) instead of an LSTM here; it has fewer parameters, trains faster, and performs identically on stationary log-return sequences of this length.

#### **2. Transformer Encoder with Time2Vec (Best for Fusion Alignment)**

Since your final model is a Transformer, using a mini-Transformer Encoder for the price sequence ensures that the cross-attention layers in your final stage receive structurally compatible representations.

```
[60-min Returns] ──> [ Linear Projection + Time2Vec ] ──> [ Transformer Encoder ] ──> [ Mean Pooling ] ──> Price Embedding

```

* **Why it works:** Standard positional encodings only capture sequence indices ($1, 2, 3$). **Time2Vec** learns continuous time representations (sine/cosine functions of actual market time, like time-of-day or day-of-week effects). This allows the network to realize that the first hour of the market open behaves differently than the lunch hour.
* **Implementation Tip:** Apply *Layer Normalization* immediately after the pooling step to stabilize the output values before concatenation.

#### **3. Dilated TCN with Causal Attention (Best for Long-Term Memory)**

If you decide to avoid explicit frequency filters like Wavelets, a Temporal Convolutional Network (TCN) with **dilated causal convolutions** is your best alternative.

```
[60-min Returns] ──> [ Dilated Causal Conv1D ] ──> [ Spatial Attention ] ──> Price Embedding

```

* **Why it works:** Dilated convolutions allow the network to have a massive receptive field (looking back weeks) with very few layers, completely avoiding the vanishing gradient issues of RNNs. Adding a light attention mechanism over the convolutional features allows the model to selectively remember massive historical return shocks (e.g., an earnings announcement 10 days ago).

---

### **Critical Architectural Warning: The Concatenation Trap**

When merging these three distinct data sources, you face a major issue with **Dimensionality and Information Density Imbalance**:

* **FinBERT:** Outputs a highly dense, 768-dimensional vector representing rich semantic spaces.
* **Macro Data:** Likely outputs a tiny, 16-to-64-dimensional vector (CPI, Unemployment, Interest Rates shift slowly).
* **Price Data:** Outputs a sequential representation.

If you blindly concatenate a 768-dim News vector with a 32-dim Macro vector and a 64-dim Price vector, the **FinBERT embeddings will completely dominate the gradient updates**, and your model will turn into a news-sentiment predictor that ignores price trends and macro shifts.

#### **How to Fix It (The Bottle-Neck Projection Strategy)**

Before your final concatenation, pass each individual embedding through its own linear projection layer (with LayerNorm) to balance their dimensions and bring them into a shared latent space scale (e.g., projecting everything to exactly 128 or 256 dimensions).

```
Price Embedding (128-dim)  ──> [ LayerNorm ] ──┐
News Embedding (768-dim)   ──> [ Dense 128 ] ──┼─> [ Concatenate (384-dim) ] ──> Final Dense Layer
Macro Embedding (32-dim)   ──> [ Dense 128 ] ──┘

```

---

The interactive sandbox below allows you to experiment with how these different processing choices transform your raw log returns into embedding vectors, demonstrating how features are extracted, smoothed, or encoded before being merged with your multimodal system.

### **Recommended Immediate Next Step**

Start by extracting **60-minute log returns** alongside **realized volatility** (e.g., the rolling standard deviation of 5-minute returns aggregated to the hour mark). Passing *both* the directional return and the volatility measure into a **Dual-Branch GRU** gives your model an explicit grasp of market fear alongside market direction, creating an incredibly rich price embedding to pair with your FinBERT sentiment.


------

Switching from 60-minute data to 1-minute data completely changes the mathematical nature of your input. At a 1-minute resolution, you are dealing with **market microstructure noise** (bid-ask bounces, execution lags, and order-book imbalances). A single US trading day contains 390 minutes, meaning your sequence length just multiplied by 390.

Here is how your data shift impacts the three architectures, followed by a deep dive into exactly how you train this embedding model without explicit ground-truth labels.

---

### **Part 1: Do the Three Architectures Still Work on 1-Minute Data?**

Yes, but their effectiveness shifts drastically because of the long sequence length ($390+$ steps per day):

1. **Multi-Scale Wavelet + GRU:** Standard Recurrent networks (GRUs/LSTMs) struggle with sequences longer than 100 steps due to gradient stability issues. If you use this, you *must* use the Wavelet transform to heavily downsample the high frequencies, or restrict your lookback window to only the last 60 to 90 minutes.
2. **Transformer Encoder + Time2Vec:** **Highly Recommended.** Transformers handle long sequences easily because every minute can directly attend to every other minute regardless of distance. Furthermore, at a 1-minute scale, intra-day time effects (e.g., high volatility at market open at 9:30 AM vs. lunch-hour lull at 12:00 PM) are highly predictive. Time2Vec captures these cyclic patterns perfectly.
3. **Dilated TCN:** **Excellent Fit.** TCNs excel at high-frequency data because their dilated layers allow them to have a lookback window of 400+ minutes with only 4 or 5 convolutional layers, capturing long-term dependencies without sequence-length bottlenecks.

> **Crucial Rule for 1-Min Data:** You **must** use **1-minute log returns**, not raw prices. Raw 1-minute prices are highly non-stationary and will cause your embeddings to drift completely out of distribution when a stock enters a new price regime.

---

### **Part 2: How to Train the Embedding Model Without Labels**

You have hit on one of the most critical challenges in quantitative deep learning. If you don't have downstream labels for the embedding model specifically, how does it learn what a "good" price representation is?

You have three primary institutional strategies to train your price encoder.

#### **Strategy A: End-to-End Joint Training (The Simplest Approach)**

You do not train the price embedding model in isolation. Instead, you hook the Price Encoder up to your final multimodal Transformer alongside FinBERT and the Macro MLP.

```
[1-Min Returns] ──> [ Price Encoder ] ──┐
[News Text]     ──> [ FinBERT ]         ┼─> [ Multimodal Transformer ] ──> [ MSE Loss ] ──> (Next Price)
[Macro Data]    ──> [ Macro MLP ]       │

```

* **How it trains:** Your ground label is simply the **Next Price Return** at the very end of the network. During backpropagation, the gradients flow backwards from the final prediction loss, through the multimodal Transformer, and straight into your Price Encoder.
* **Pros:** The price embedding learns to extract *exactly* what the final Transformer needs to make its prediction.
* **Cons:** Training a massive multimodal model from scratch all at once can be highly unstable and computationally expensive.

#### **Strategy B: Masked Autoencoding / Self-Supervised Pre-training (Recommended)**

This is exactly how BERT was trained on text, but applied to time-series. You train your Price Encoder to understand the "language" of the stock market before mixing it with news or macro data.

```
1. Take a 390-min sequence of log returns.
2. Randomly "mask" (blank out) 15% of the minutes.
3. Pass it through your Price Encoder -> Decoder.
4. Target Label: Reconstruct the missing 15% minutes.

```

* **How it trains:** The model acts as its own supervisor (Self-Supervised Learning). The loss function is the Mean Squared Error (MSE) between the predicted missing returns and the actual missing returns.
* **Pros:** It forces the encoder to learn the deep, underlying statistical properties, correlations, and momentum structures of the market entirely from raw data. Once trained, you freeze or fine-tune this encoder inside your multimodal pipeline.

#### **Strategy C: Contrastive Learning (SimCLR for Time Series)**

You train the encoder to recognize that two slightly different versions of the same trading day should look similar, while two completely different trading days should look different.

```
Original Day Window ──┬─> [ Add Random Jitter ] ──> Encoder ──┐  Goal: Minimize 
                      └─> [ Random Crop ]       ──> Encoder ──┴─> Vector Distance

```

* **How it trains:** You take a 1-minute price sequence, create two "augmented" variations of it (e.g., adding slight noise or cropping intervals), and pass them through the encoder. The loss function forces the resulting embedding vectors to be nearly identical, while pushing the vectors of *different* trading days far apart in vector space.

---

The interactive visualizer below demonstrates **Strategy B (Self-Supervised Masked Autoencoding)**. You can see how an encoder learns to build an underlying embedding by trying to reconstruct missing segments of noisy 1-minute stock data.

### **Step-by-Step Implementation Framework**

If you want to build this system cleanly, follow this deployment order:

1. **Step 1 (Data Prep):** Convert your 1-minute raw prices into 1-minute log returns using $R_t = \ln(P_t / P_{t-1})$.
2. **Step 2 (Pre-training):** Implement a small Transformer Encoder. Train it using **Strategy B (Masked Autoencoding)** on thousands of days of historical 1-minute log returns. Your objective is simply to reconstruct the masked minutes.
3. **Step 3 (Extraction):** Once the reconstruction loss stops dropping, detach the decoder. The remaining Transformer Encoder is now your dedicated **Price Embedding Model**.
4. **Step 4 (Multimodal Integration):** Feed your 1-minute sequences into this trained encoder to get your fixed-size price embedding vector, and pass that vector right into your final multimodal forecasting pipeline.
-----

For the Indian stock market (NSE/BSE) specifically, trading runs from 9:15 AM to 3:30 PM. This means a single trading day consists of exactly **375 minutes**. Because you are dealing with high-frequency, highly noisy 1-minute data, we need an architecture that filters noise natively and a training method that doesn't collapse under the volatility.

Here is the exact blueprint I recommend we lock in.

### 1. The Architecture: Dilated TCN with Spatial Attention

While Transformers with Time2Vec are brilliant for 60-minute data, at the 1-minute level, they often suffer from "attention dispersion"—they waste compute trying to find relationships between random micro-fluctuations.

Instead, lock in the **Dilated TCN (Temporal Convolutional Network)**.

* **Why for Indian Stocks?** The NSE is heavily algorithmic and highly volatile. A standard 1D Convolution acts as a mathematical low-pass filter, natively smoothing out bid-ask bounce noise at the lowest layers.
* **The Receptive Field Advantage:** By using *dilated* convolutions (where the filter skips steps, e.g., looking at $t$, $t-2$, $t-4$, then $t$, $t-4$, $t-8$), a TCN can look back over the entire 375-minute Indian trading day using only 5 or 6 layers. This provides massive historical context with a very small memory footprint.
* **The Spatial Attention Add-On:** After the TCN processes the 375-minute sequence into a dense feature map, apply a single layer of Spatial Attention at the very end. This allows the model to heavily weight specific structural events (like the 9:15 AM open volatility or the 3:00 PM closing rush) before outputting the final Price Embedding Vector.

### 2. The Training Strategy: Masked Autoencoding (Self-Supervised)

Do not try to train this TCN directly on your final "Predict Next Price" target right away. Joint training of multimodal high-frequency models usually results in the model completely ignoring the noisy price data and overfitting to the FinBERT news data.

Lock in **Strategy B: Masked Autoencoding**.

1. **Input:** Pass in a full day of 1-minute log returns (375 steps).
2. **Mask:** Randomly zero-out 15% of those minutes.
3. **Forward Pass:** Run it through your TCN.
4. **Target:** The TCN must output the missing 15% of the log returns. Use Mean Squared Error (MSE) loss.
5. **Why this works:** To successfully guess a missing 1-minute return at 1:15 PM, the TCN *must* learn the underlying momentum, volatility clustering, and microstructure of the stock. It is forced to become a master feature extractor.
6. **Integration:** Once the MSE loss plateaus, you stop training. You chop off the final output layer of the TCN, freeze the weights, and use the resulting dense vector as your **Price Embedding** to concatenate with FinBERT and Macro data.

---

### Designing Your TCN for the NSE

The most critical parameter of your TCN is its **Receptive Field**—how many minutes of history the final layer can "see." If your receptive field is too small, the model has amnesia. If it's too large, you waste compute. To cover a full Indian trading day, you need a receptive field of at least 375.

Use the sandbox below to calculate the optimal layer depth and dilation base for your network.

By the way, to unlock the full functionality of all Apps, enable [Gemini Apps Activity](https://myactivity.google.com/product/gemini).

----

To build and train a **Dilated TCN with Spatial Attention** using a **Masked Autoencoder** approach for 1-minute Indian stock data, your environment and data pipeline must be configured to handle high-frequency time series.

Below are the exact prerequisites you need to prepare. Once you have these ready, we will write the model code and training loops.

---

### **1. Data Prerequisites (The NSE/BSE Dataset)**

Because Indian markets run from **9:15 AM to 3:30 PM**, you need continuous, unadjusted 1-minute intraday bars.

* **Data Columns Needed:** At minimum, you need `Timestamp`, `Open`, `High`, `Low`, `Close`, and `Volume`.
* **Day Length Consistency:** Filter your data to ensure every full trading day contains exactly **375 rows** (minutes). Delete partial trading days (like Muhurat trading days or days with massive market halts) to keep your tensor shapes uniform.
* **Asset Volume:** Gather data for at least 10–20 liquid stocks (e.g., NIFTY 50 components like RELIANCE, TCS, HDFCBANK) covering 3 to 5 years. Self-supervised pre-training requires thousands of daily profiles to prevent overfitting.

### **2. Software & Library Environment**

Ensure you have a Python environment set up with modern deep learning frameworks. GPU acceleration (CUDA or Apple Silicon MPC) is strictly mandatory for 1-minute sequence training.

```bash
pip install torch torchvision torchaudio  # PyTorch (Core DL Framework)
pip install pandas numpy scikit-learn    # Data Manipulation & Scaling
pip install matplotlib seaborn            # Visualization

```

### **3. Mathematical Preprocessing Pipeline**

You cannot feed raw prices into the TCN. You must build a script to transform raw prices into stationary log returns:

$$\text{Log Return } (R_t) = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$

For the first minute of the day (9:15 AM), calculate the log return relative to the **previous day's close** to capture overnight gap risk, which is a major alpha driver in Indian equities.

### **4. Target Data Tensors (Expected Shapes)**

Before passing data to PyTorch, your data loader must group individual minutes into daily blocks:

* **Raw Batch Input Shape:** `(Batch_Size, 375, 1)` representing `(Daily Profiles, Minutes in NSE Session, Log Return Feature)`.
* **TCN Expected Input Shape:** `(Batch_Size, 1, 375)` because PyTorch 1D Convolutions expect the channel dimension (features) to come *before* the sequence length dimension.

---

The interactive sandbox below allows you to inspect exactly how your raw 1-minute NSE stock data changes shape, gets masked, and flows through the dimensions expected by the TCN layers. Use it to verify your tensor alignments before we write the PyTorch code.

### **Next Step Alignment**

Do you already have historical 1-minute OHLCV data for your target Indian stocks downloaded in CSV or Parquet format, or do you need assistance structuring the data-loading script first?