Your thought process is spot on. In enterprise data engineering, what you are describing is called **Server-Side Filtering** (or "Pushdown Computation").

Instead of downloading 10 years of the entire internet to your hard drive (which would require hundreds of terabytes and crash your system), you ask the server to run the search *on their side*, and you only download the exact sentences or articles that match your 2,400 companies.

Once downloaded, you stream them through the NLP engine we just built and route them directly into **2,400 individual JSONL files** (e.g., `INFY_news.jsonl`, `TCS_news.jsonl`). This creates a highly efficient, localized "Data Lake" on your machine.

Here is the exact strategy on how to execute this for each of your data sources.

---

### 1. The GDELT Project (The "BigQuery" Hack)

GDELT processes millions of articles daily. If you try to download their raw CSVs for 10 years, you will run out of storage immediately.

**The Solution:** GDELT hosts its entire 10-year database for free on **Google BigQuery**.

* You do not download the files. Instead, you write a SQL query directly in BigQuery that says: `SELECT Date, URL, DocumentText FROM gdelt-bq.gdeltv2.gkg WHERE V2Organizations CONTAINS "Infosys" OR V2Organizations CONTAINS "Tata Consultancy"`.
* Google's supercomputers scan 10 years of global news in about 30 seconds.
* You only download the resulting filtered CSV (which will be a few megabytes instead of terabytes). Then, you pass it through our NLP script to map it perfectly to your stocks.

### 2. Finnhub API (The Direct Ticker Loop)

Finnhub is inherently designed for your 2,400-file architecture because you query them *by ticker symbol*.

**The Solution:** You write a Python script that loops through your 2,400 tickers one by one.

* **The Call:** Ask Finnhub for `symbol=INFY`, from `2014-01-01` to `2024-01-01`.
* **The Catch:** Finnhub's free tier only goes back 1 year. To get the deep 10-to-20-year historical company news, you will need one of their premium tiers.
* Because Finnhub data is strictly financial, you can route the returned JSON directly into `INFY.jsonl` without even needing the heavy NLP gatekeeper.

### 3. NSE Historical Reports (Corporate Announcements)

This is the cleanest data you will find. The National Stock Exchange (NSE) categorizes every board meeting, quarterly result, and auditor change by the exact BSE/NSE symbol.

**The Solution:** You do not need NLP for this at all. You can use an open-source Python library like `jugaad-data` or `nsepy` to scrape the "Corporate Disclosures" directly from the NSE archives. Since they are explicitly tagged with the exact ticker, you just download them and append them straight to the respective company's file.

### 4. NewsData.io (The Recent Backfill)

As we discovered earlier, NewsData.io caps out at about 2 years of history.

**The Solution:** Use this strictly as a "live" or "recent" feed. Once you have built your 10-year historical database using GDELT and Finnhub, you can run a daily cron job that pings NewsData.io for yesterday's news, runs it through our NLP gatekeeper, and appends the fresh data to your 2,400 files.

---

### How to Build the 2,400-File Storage System

Writing to 2,400 different files simultaneously can cause an "OS Too Many Open Files" error. To handle this efficiently, your architecture should look like this:

1. **The Ingestion Queue:** Download a filtered chunk of data from BigQuery or Finnhub.
2. **The NLP Router:** Feed the chunk into the `spaCy` NLP script we just built. The script identifies that an article belongs to `INFY` and `TCS`.
3. **The Appender:** The script opens `data/stocks/INFY.jsonl` in `append` mode (`"a"`), writes the single JSON line, and closes it. It then does the same for `TCS.jsonl`.

By structuring it this way, you never hold more than one article in your RAM at a time, and your disk is perfectly organized into 2,400 individual timelines.


-----


This is where the rubber meets the road. Transitioning from processing a single daily file to organizing a **10-year historical timeline for 2,400 individual stocks** requires architectural shift.

If you download everything into one giant file, your analytics will be impossibly slow. If you open 2,400 files at the same time in Python, your operating system will crash with a `Too many open files` error.

To map every article to its relative stock—including its ticker, date, title, body, and inherited **industry/sector macro metrics**—you need a **Two-Stage Ingestion Pipeline**.

---

## The Storage & Mapping Architecture

Instead of having your NLP script directly write to 2,400 files, we use a **Map-and-Flush** system:

```
[ Raw Filtered News Chunk (BigQuery / GDELT) ]
                       │
                       ▼
[ SpaCy NLP Router / Multi-Word Matcher ]
                       │
                       ▼
[ In-Memory Aggregator (Python Dictionary) ]
  e.g., {"INFY": [art1, art2], "TCS": [art1]}
                       │
                       ▼  (Flush data block-by-block)
[ 2,400 Individual JSONL Files in data/stocks/ ]

```

### Why this structure is essential for you:

1. **Zero Open-File Friction:** Python opens an individual stock's file (e.g., `data/stocks/INFY.jsonl`), appends the new array of articles, and immediately closes it.
2. **Dynamic Industry/Sector Enrichment:** Every time an article maps to a stock, the script automatically injects the `Sector` and `Industry` from your master CSV into that specific record. If you query that stock's file later, its sector context is already baked in.
3. **Memory Safety:** It processes your massive GDELT BigQuery dumps in chunks of 50,000 articles, completely protecting your RAM.

---

## Step 1: Prepare the Company Metadata Map

We need a fast look-up dictionary so that the moment a symbol is matched, the script instantly knows its clean company name, sector, and industry.

```python
import pandas as pd
import json

COMPANY_CSV = "data/processed/companies_master.csv"
df = pd.read_csv(COMPANY_CSV)

# Build a master mapping dictionary
# Key: Symbol -> Value: Clean Metadata
company_metadata = {}

for _, row in df.iterrows():
    sym = str(row["Symbol"]).strip().upper()
    company_metadata[sym] = {
        "symbol": sym,
        "company_name": str(row["Company Name"]).strip(),
        "sector": str(row["Sector"]).strip().upper(),
        "industry": str(row["Industry"]).strip().upper()
    }

```

---

## Step 2: The Production-Grade Data Separator

Here is the complete script to take your filtered incoming news data (from your GDELT BigQuery exports or API pulls), pass it through your working **Hybrid NLP Engine**, and distribute it cleanly into 2,400 stock files.

```python
import os
import json
import re
import spacy
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# =====================================================
# 1. INITIALIZATION & CONFIG
# =====================================================
print("Initializing Enterprise Routing Engine...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

COMPANY_CSV = "data/processed/companies_master.csv"
INPUT_NEWS_DUMP = "data/news/raw/gdelt_bigquery_export.jsonl" # Your massive downloaded historical file
OUTPUT_DIR = "data/stocks"

# Create output folder for the 2,400 stocks
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load master mappings
df = pd.read_csv(COMPANY_CSV)
REMOVE_SUFFIX = r"\b(LIMITED|LTD|PRIVATE|PVT|INC|CORP|CORPORATION|LLC|LLP|PLC|CO|COMPANY)\b"

multi_word_dict = {}
single_word_dict = {}
symbols_dict = {}

company_metadata = {}
MEGA_CAPS = {
    "RELIANCE": "RELIANCE", "WIPRO": "WIPRO", "INFOSYS": "INFY",
    "TCS": "TCS", "HDFC": "HDFCBANK", "ICICI": "ICICIBANK",
    "ITC": "ITC", "LARSEN": "LT", "MARUTI": "MARUTI"
}

print("Building entity dictionaries & metadata maps...")
for _, row in df.iterrows():
    sym = str(row["Symbol"]).strip().upper()
    raw_name = str(row["Company Name"]).strip().upper()
    
    # Store corporate metadata for injection later
    company_metadata[sym] = {
        "company_name": str(row["Company Name"]).strip(),
        "sector": str(row["Sector"]).strip().upper(),
        "industry": str(row["Industry"]).strip().upper()
    }

    # Base name extraction
    base_name = re.sub(r"[^\w\s&]", " ", raw_name)
    base_name = re.sub(REMOVE_SUFFIX, "", base_name).strip()
    base_name = re.sub(r"\s+", " ", base_name).strip()

    if len(base_name) < 3: 
        continue

    if len(base_name.split()) > 1:
        multi_word_dict[base_name] = sym
    else:
        single_word_dict[base_name] = sym

    if len(sym) >= 3 and sym not in {"IDEA", "PAGE", "TIME", "STAR", "FOCUS", "CAMPUS"}:
        symbols_dict[sym] = sym

# Pre-compile multi-word regex
sorted_multi = sorted(list(multi_word_dict.keys()), key=len, reverse=True)
regex_multi = re.compile(r"\b(" + "|".join(map(re.escape, sorted_multi)) + r")\b", re.IGNORECASE)

# =====================================================
# 2. THE ROUTING & SEPARATION ENGINE
# =====================================================
print("Processing and separating historical news...")

# In-memory buffer to store matches before flushing to disk
# Format: {"INFY": [article1, article2], "TCS": [article1]}
buffer_data = defaultdict(list)
BUFFER_LIMIT = 20000  # Flush to files every 20k articles processed to manage memory safely

def flush_buffer_to_disk(buffer_dict):
    """Safely appends grouped articles to their respective stock files."""
    for symbol, articles in buffer_dict.items():
        file_path = os.path.join(OUTPUT_DIR, f"{symbol}_news.jsonl")
        with open(file_path, "a", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
    buffer_dict.clear()

with open(INPUT_NEWS_DUMP, "r", encoding="utf-8") as fin:
    for count, line in enumerate(tqdm(fin)):
        raw_article = json.loads(line)
        
        # Standardize your incoming API / BigQuery keys here
        title = raw_article.get("title", "")
        body = raw_article.get("text", raw_article.get("body", ""))
        date = raw_article.get("date", raw_article.get("publish_date", ""))
        url = raw_article.get("url", "")
        
        full_text = f"{title}\n{body}"
        matched_symbols_in_article = set()

        # --- Layer 1: Multi-Word Matcher ---
        for match in regex_multi.findall(full_text):
            sym = multi_word_dict[match.upper()]
            matched_symbols_in_article.add(sym)

        # --- Layer 2: SpaCy NLP Gatekeeper ---
        doc = nlp(full_text)
        org_entities = [ent.text.upper() for ent in doc.ents if ent.label_ == "ORG"]

        for org in org_entities:
            clean_org = re.sub(REMOVE_SUFFIX, "", org).strip()
            clean_org = re.sub(r"[^\w\s&]", " ", clean_org)
            clean_org = re.sub(r"\s+", " ", clean_org).strip()

            if clean_org in single_word_dict:
                matched_symbols_in_article.add(single_word_dict[clean_org])
                continue

            for word in clean_org.split():
                if word in MEGA_CAPS:
                    matched_symbols_in_article.add(MEGA_CAPS[word])
                elif word in symbols_dict:
                    matched_symbols_in_article.add(symbols_dict[word])
                elif word in single_word_dict:
                    matched_symbols_in_article.add(single_word_dict[word])

        # --- Layer 3: Route Enriched Structures to Buffer ---
        for symbol in matched_symbols_in_article:
            meta = company_metadata[symbol]
            
            # Formulate the final payload tailored specifically for this stock file
            payload = {
                "date": date,
                "ticker": symbol,
                "company_name": meta["company_name"],
                "sector": meta["sector"],
                "industry": meta["industry"],
                "title": title,
                "body": body,
                "url": url
            }
            
            buffer_data[symbol].append(payload)

        # Periodic flush to disk to maintain absolute stability
        if count > 0 and count % BUFFER_LIMIT == 0:
            flush_buffer_to_disk(buffer_data)

    # Final sweep flush for remaining records
    if buffer_data:
        flush_buffer_to_disk(buffer_data)

print("Pipeline Execution Complete. 2400 Stock Data Lake successfully generated.")

```

---

## How this structures your Output Files

Once you execute this script against your historical data dumps, your `data/stocks/` directory will look like this:

* `data/stocks/INFY_news.jsonl`
* `data/stocks/TCS_news.jsonl`
* `data/stocks/RELIANCE_news.jsonl`

If you open `data/stocks/INFY_news.jsonl`, every single line is an independent, perfectly structured JSON object structured exactly like this:

```json
{
  "date": "2014-06-15",
  "ticker": "INFY",
  "company_name": "INFOSYS LIMITED",
  "sector": "INFORMATION TECHNOLOGY",
  "industry": "IT SOFTWARE PRODUCTS",
  "title": "Infosys hits fresh high on new deal announcements",
  "body": "Shares of the IT major rose 3% today after winning a multi-year digital transformation contract in Europe...",
  "url": "https://financialexpress.com/market-news/infosys-hits-fresh-high..."
}

```

## How to Handle Standalone Industry News

If you have broad macroeconomic articles that mention an entire industry (like *“Indian Banking Sector NPA crisis”*) but **do not** name a specific company, you do not want to force-feed that article into all 100+ banking stock files. That would cause massive data duplication.

Instead, let the engine create **Industry Master Files** alongside the stock files.


----


Fair point. Shifting approaches and debugging false positives across multiple scripts created unnecessary confusion and wasted your time. Let's reset and lay down one clear, definitive pipeline.

Here is the complete blueprint of how your system works end-to-end, broken down by **what happens, which script does what, and the exact order you must run them in.**

---

## The End-to-End Method (How It Works)

Your goal is to take **10 years of raw historical news** and route it into **2,400 individual stock timeline files** (e.g., `INFY_news.jsonl`, `TCS_news.jsonl`), with every article tagged with its exact date, title, body, ticker, sector, and industry.

To do this cleanly without crashing your RAM or blowing up your hard drive, the system uses a **3-Step Assembly Line**:

```
[ Step 1: Preprocess Master CSV ] ──► Builds clean lookup dictionaries & Sector maps
                                             │
[ Step 2: Fetch Filtered News ]    ──► Pulls raw matching articles from GDELT/BigQuery
                                             │
[ Step 3: Run Matcher & Separator] ──► Runs Hybrid NLP + Multi-word Matcher and 
                                       flushes enriched data into 2,400 stock files

```

---

## The Script Breakdown & Execution Order

You only need **two Python scripts** and **one SQL query step** in total. Here is what each script does and why they exist in this sequence:

### Step 1: `01_preprocess_master.py` (Run First)

* **Why run this first?** Your raw `companies_master.csv` contains corporate clutter like `"LIMITED"`, `"PVT"`, `"CORP"`, and punctuation. Searching these raw strings directly will either miss matches or cause false positives.
* **What it does:**
1. Cleans company names (e.g., `"BANK OF MAHARASHTRA LIMITED"` becomes `"BANK OF MAHARASHTRA"`).
2. Categorizes names into **Multi-Word Names** (matched via fast regex) and **Single-Word/Risk Names** (handled via NLP).
3. Pre-builds a complete lookup table of every ticker's **Company Name, Sector, and Industry**.


* **Output:** Generates `data/processed/companies_master_cleaned.csv` and a JSON metadata file.

---

### Step 2: GDELT / BigQuery News Download (Data Ingestion)

* **Why do this here?** You cannot download 10 years of all global news (hundreds of terabytes) to your local machine.
* **What it does:** Runs a targeted SQL search on GDELT's free BigQuery database using your cleaned company list. Google's servers scan 10 years of news in seconds and return **only** articles mentioning your companies.
* **Output:** Generates a raw, manageable news dump on your machine (`data/news/raw/gdelt_10yr_dump.jsonl`).

---

### Step 3: `02_match_and_separate.py` (The Core Engine - Run Last)

* **Why run this last?** This takes the raw news dump from Step 2, runs it through your verified matching logic, attaches company metadata, and routes articles into 2,400 individual stock files.
* **What it does:**
1. Reads the raw news line-by-line (uses almost zero RAM).
2. Passes text through **Layer 1** (Multi-word exact regex match) and **Layer 2** (spaCy NLP gatekeeper for single-word / ticker matches).
3. Injects company metadata into the article payload:
```json
{
  "date": "2014-06-15",
  "ticker": "INFY",
  "company_name": "INFOSYS LIMITED",
  "sector": "INFORMATION TECHNOLOGY",
  "industry": "IT SOFTWARE PRODUCTS",
  "title": "...",
  "body": "...",
  "url": "..."
}

```


4. Stores results in a memory buffer and appends them block-by-block to `data/stocks/INFY_news.jsonl`, `TCS_news.jsonl`, etc.


* **Output:** 2,400 clean, localized stock timeline files inside `data/stocks/`.

---

## Pipeline Summary Table

| Execution Order | Component | Purpose | Input | Output |
| --- | --- | --- | --- | --- |
| **1. Run Script 1** | `01_preprocess_master.py` | Clean names & build sector/industry lookup maps | `companies_master.csv` | Cleaned master CSV & metadata map |
| **2. Run SQL Query** | GDELT BigQuery Search | Server-side filter to download only relevant historical news | Clean company list | `gdelt_10yr_dump.jsonl` |
| **3. Run Script 2** | `02_match_and_separate.py` | Hybrid NLP matching + routing into individual stock files | Raw news dump | 2,400 files in `data/stocks/` |

---