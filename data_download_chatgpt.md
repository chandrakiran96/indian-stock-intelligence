Yes. In fact, I think you're approaching this the right way.

If you're building a **serious Indian stock prediction model**, the news component is **far more than sentiment**. Most people simply feed headlines into FinBERT and expect magic. That almost never works.

A good system understands:

> **What happened → Who was affected → How important was it → How long does it influence the market → Which companies/sectors are connected.**

This is much closer to how hedge funds build news pipelines.

---

# Step 1: What news actually affects Indian stocks?

Think of the market as 8 different layers.

```
                Global Events
                     ↓
          Indian Macro Economy
                     ↓
         Government & RBI Policies
                     ↓
          Industry / Sector News
                     ↓
        Company Specific Events
                     ↓
      Financial Reports & Guidance
                     ↓
      Social Sentiment & Analyst Views
                     ↓
           Stock Price Movement
```

Each layer has different impact.

---

# Layer 1 — Company Specific News (Highest Impact)

These affect one stock immediately.

Examples

* Quarterly Results
* Earnings Beat/Miss
* Profit Warning
* CEO resignation
* CFO resignation
* New CEO
* Acquisition
* Merger
* Demerger
* Dividend Announcement
* Bonus Issue
* Stock Split
* Buyback
* Rights Issue
* Bankruptcy
* Insolvency
* Credit Rating Downgrade
* New Product Launch
* Plant Shutdown
* Fire
* Cyber Attack
* Tax Notice
* SEBI Investigation
* CBI/ED Investigation
* Court Judgement
* Patent Approval
* Patent Rejection
* Export Ban
* Large Order Win
* Order Cancellation
* Share Pledge
* Promoter Selling
* Promoter Buying
* Insider Trading
* Foreign Investment
* Institutional Buying
* Block Deals

Example

```
TCS wins $2 Billion contract

↓

IT Sector Positive

↓

TCS up 4%

↓

Infosys +1%

↓

Wipro +0.8%
```

---

# Layer 2 — Sector News

These affect entire industries.

Example

Oil

```
Crude Oil ↑

↓

Indian Oil
BPCL
HPCL
Airlines
Paint Companies
Tyre Companies
```

Banking

```
RBI increases Repo Rate

↓

Banks
NBFC
Housing Finance
Real Estate
```

IT

```
US recession fears

↓

Infosys
TCS
LTIM
HCL
Tech Mahindra
```

Auto

```
Lithium prices rise

↓

EV companies

↓

Battery companies
```

Real Estate

```
Interest Rate ↓

↓

Home Loan ↑

↓

Real Estate ↑
```

---

# Layer 3 — Macro Economic News

These affect almost every stock.

Examples

GDP

Inflation

CPI

WPI

Repo Rate

Reverse Repo

CRR

Fiscal Deficit

Budget

Trade Deficit

Exports

Imports

PMI

Manufacturing PMI

Services PMI

Industrial Production

Unemployment

Retail Sales

Tax Changes

Monsoon

Agriculture Production

Fuel Prices

Exchange Rate

Forex Reserve

---

# Layer 4 — Government News

Extremely important in India.

Examples

Union Budget

GST

Corporate Tax

PLI Scheme

FDI Policy

Export Policy

Import Ban

Import Duty

Railway Budget

Defense Spending

Infrastructure Spending

Semiconductor Policy

Renewable Energy

Electric Vehicle Subsidy

PLI incentives

Mining License

Telecom Auctions

---

# Layer 5 — RBI News

Huge market mover.

Examples

Repo Rate

MPC Meeting

Inflation Target

Liquidity

OMO

Dollar Reserve

Forex Intervention

Interest Rate Guidance

Bank Regulations

---

# Layer 6 — Global News

India is highly connected.

Examples

US Fed Rate

ECB Rate

BOJ Rate

China GDP

Oil Prices

Gold Prices

War

Sanctions

OPEC Meeting

US CPI

US Jobs Data

NASDAQ Crash

S&P500

Dow Jones

Taiwan Semiconductor

Middle East Conflict

Russia Ukraine

COVID

Supply Chain

Semiconductor Shortage

---

# Layer 7 — Political News

Examples

Election

Government Stability

Cabinet Change

State Election

Policy Change

Trade Agreement

India-US

India-China

India-UAE

India-Europe

---

# Layer 8 — Market Sentiment

Examples

FII Buying

DII Buying

VIX

Fear Index

Analyst Upgrade

Analyst Downgrade

Broker Recommendation

Social Media

Retail Sentiment

---

# News Sources You Should Collect

This is where many projects fail.

Do **NOT** rely on one website.

You should combine many sources.

| Source                                                                                        | Importance | Bias     |
| --------------------------------------------------------------------------------------------- | ---------- | -------- |
| National Stock Exchange Corporate Announcements                                               | Very High  | Very Low |
| BSE Corporate Filings                                                                         | Very High  | Very Low |
| Securities and Exchange Board of India Orders                                                 | Very High  | Very Low |
| Reserve Bank of India Press Releases                                                          | Very High  | Very Low |
| Ministry of Finance                                                                           | Very High  | Very Low |
| [Moneycontrol](https://www.moneycontrol.com/?utm_source=chatgpt.com)                          | High       | Medium   |
| [LiveMint](https://www.livemint.com?utm_source=chatgpt.com)                                   | High       | Low      |
| [Business Standard](https://www.business-standard.com?utm_source=chatgpt.com)                 | High       | Low      |
| [Economic Times Markets](https://economictimes.indiatimes.com/markets?utm_source=chatgpt.com) | High       | Medium   |
| [Reuters India](https://www.reuters.com/world/india/?utm_source=chatgpt.com)                  | Very High  | Very Low |
| [Bloomberg India](https://www.bloomberg.com/asia?utm_source=chatgpt.com)                      | Very High  | Very Low |
| [CNBC TV18](https://www.cnbctv18.com?utm_source=chatgpt.com)                                  | High       | Medium   |

---

# Which sources are NOT biased?

There is **no completely unbiased news source**.

Instead, separate sources into:

### Level A (Objective)

* NSE Filings
* BSE Filings
* SEBI
* RBI
* Ministry releases
* Company filings

These contain factual information.

---

### Level B

Reuters

Bloomberg

AP

These have editorial writing but very low opinion.

---

### Level C

Moneycontrol

Economic Times

Mint

Business Standard

CNBC TV18

These are excellent but contain analysis and opinion alongside facts.

---

# How to Build the Dataset

Instead of only storing:

```
Headline

Date

Sentiment
```

Store much richer information:

| Field               | Example           |
| ------------------- | ----------------- |
| NewsID              | 12345             |
| Datetime            | 2021-04-23 10:15  |
| Source              | Reuters           |
| Company             | TCS               |
| NSE Symbol          | TCS               |
| Sector              | IT                |
| Headline            | TCS wins contract |
| Full Text           | ...               |
| FinBERT Score       | Positive          |
| FinBERT Confidence  | 0.94              |
| Event Type          | Contract Win      |
| Event Importance    | High              |
| Market Cap          | Large             |
| Related Companies   | Infosys, Wipro    |
| Related Sector      | IT                |
| NIFTY Return (t+1)  | +0.6%             |
| Stock Return (t+1)  | +3.1%             |
| Stock Return (t+5)  | +5.8%             |
| Stock Return (t+30) | +7.4%             |

This creates a much more useful supervised learning dataset.

---

# Where can you get 2014–2025 news?

There is **no single free dataset** with complete 10-year Indian financial news because of copyright restrictions.

## Best Free Sources

### 1. [GDELT Project](https://www.gdeltproject.org/data.html?utm_source=chatgpt.com) ⭐⭐⭐⭐⭐

Probably your best option.

Advantages

* Starts well before 2014
* Millions of news articles
* Includes India
* Includes timestamps
* Includes source URLs
* Can be queried by company names
* Completely free
* Downloadable
* BigQuery support

Ideal for research-scale pipelines. ([GDELT Project][1])

---

### 2. Kaggle

Examples include historical Moneycontrol headline datasets and other community datasets, but coverage is often incomplete or limited to headlines. ([Kaggle][2])

---

### 3. [NewsData.io](https://newsdata.io?utm_source=chatgpt.com)

Offers APIs and has documented financial datasets, though extensive historical access is generally part of paid offerings. ([NewsData][3])

---

### 4. [Finnhub API](https://finnhub.io/docs/api/market-news?utm_source=chatgpt.com)

Good API for financial news, but deep historical coverage depends on subscription level. ([Finnhub][4])

---

### 5. [NSE Historical Reports](https://www.nseindia.com/reports-indices-historical-index-data?utm_source=chatgpt.com)

Download:

* Index history
* Corporate announcements
* Historical market reports
* Price/volume archives

These are invaluable for aligning news with price movements. ([NSE India][5])

---

### 6. DesiQuant

A community project offering historical Indian market datasets including announcements and news from multiple publishers, useful as a supplementary source. ([Reddit][6])

---

# My recommendation for a research-grade pipeline

If I were building a publishable Indian stock prediction system, I would create the following architecture:

```
2014–2025

          GDELT
             │
      Reuters
             │
      Moneycontrol
             │
      LiveMint
             │
     Business Standard
             │
       NSE Filings
             │
       BSE Filings
             │
      RBI Releases
             │
      SEBI Orders
             │
      Government PR
             │
      Deduplicate
             │
 Named Entity Recognition
             │
 Map to NSE Symbol
             │
 Event Classification
             │
      FinBERT Embeddings
             │
 Time Alignment (T, T+1, T+5, T+30)
             │
 Merge with OHLCV
             │
 Store in PostgreSQL / Parquet
             │
        Train Model
```

This avoids relying on any single publisher, reduces editorial bias by emphasizing primary regulatory filings, and gives your model structured event context in addition to sentiment.

## I also recommend adding these non-news signals

For the strongest prediction model, include:

* Daily OHLCV data
* Delivery percentage
* FII/DII flows
* Corporate actions
* Quarterly financial statements
* Insider trades
* Shareholding pattern
* Options open interest
* India VIX
* RBI macroeconomic releases
* USD/INR exchange rate
* Brent crude prices
* Gold prices
* US market indices
* Global macroeconomic calendar

These complementary features often improve predictive performance more than sentiment alone.

**I can also help you build a complete end-to-end data engineering pipeline for this project**, including:

1. automatic downloading of **2014–2025 news** from multiple sources,
2. mapping every article to affected NSE-listed companies,
3. generating **FinBERT embeddings**,
4. creating a clean training dataset with **news + technical indicators + macroeconomic data**, and
5. organizing everything into a reproducible data lake (Parquet/PostgreSQL) ready for transformer or multimodal stock prediction models.

[1]: https://www.gdeltproject.org/data.html?utm_source=chatgpt.com "Data: Querying, Analyzing and Downloading"
[2]: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/moneycontrol-news?utm_source=chatgpt.com "MONEYCONTROL NEWS"
[3]: https://newsdata.io/blog/indian-stock-market-news-dataset/?utm_source=chatgpt.com "Indian Stock Market News Dataset: A Mega Compilation"
[4]: https://finnhub.io/docs/api/market-news?utm_source=chatgpt.com "Real-time Market News API"
[5]: https://www.nseindia.com/reports-indices-historical-index-data?utm_source=chatgpt.com "Historical Index Data"
[6]: https://www.reddit.com/r/IndiaInvestments/comments/197348s/desiquant_free_quality_datasets_for_indian_stocks/?utm_source=chatgpt.com "DesiQuant: Free Quality Datasets for Indian Stocks Markets ..."


------

I love this approach.

I think we should treat this like a **research project**, not just a coding project. If we get the data engineering wrong, even the best FinBERT or LLM won't rescue the model.

I'd actually like to change one thing from what I initially suggested.

> **We should not start by downloading everything.**
>
> We should first **design the entire data lake**, because once you have ~50 million news articles, changing the schema becomes painful.

GDELT alone contains billions of records across its historical archive, and a single year of some datasets (like the Global Knowledge Graph) can be terabytes in size. We need to be selective rather than downloading everything indiscriminately. ([GDELT Project][1])

---

# This is the roadmap I recommend

Instead of this:

```
Download News
      ↓
FinBERT
      ↓
Train
```

We'll build something much closer to what a quantitative research team would use:

```
Stage 0
Research & Design
        ↓
Stage 1
Raw News Collection
        ↓
Stage 2
Cleaning
        ↓
Stage 3
Deduplication
        ↓
Stage 4
Company Mapping
        ↓
Stage 5
Event Classification
        ↓
Stage 6
FinBERT Embeddings
        ↓
Stage 7
Feature Engineering
        ↓
Stage 8
Merge with Stock Data
        ↓
Stage 9
Training Dataset
```

I don't want to skip **any** stage.

---

# Our Goal

By the end, every news article should look like this.

| Field             | Example                |
| ----------------- | ---------------------- |
| news_id           | UUID                   |
| datetime          | 2021-07-12 10:43       |
| source            | Reuters                |
| title             | TCS wins cloud deal    |
| article           | Full text              |
| url               | original URL           |
| publisher         | Reuters                |
| language          | English                |
| country           | India                  |
| companies         | ["TCS"]                |
| sector            | IT                     |
| event             | Contract Win           |
| importance        | High                   |
| finbert_embedding | 768 or 1024 dimensions |
| sentiment         | Positive               |
| confidence        | 0.98                   |
| duplicate_cluster | 18234                  |
| market_session    | Pre-market             |
| nse_symbols       | TCS                    |

Notice that **FinBERT is almost the last step**, not the first.

---

# Stage 0 — Decide Our Data Sources

This is the most important decision of the entire project.

I'd divide sources into three tiers.

## Tier A (Primary, factual)

These are the "ground truth" sources.

* Corporate announcements (NSE)
* Corporate announcements (BSE)
* SEBI orders
* RBI press releases
* Ministry of Finance
* Company annual reports
* Exchange circulars

These contain almost no editorial bias.

---

## Tier B (Professional financial journalism)

These explain events.

* Reuters
* Bloomberg
* Moneycontrol
* Mint
* Business Standard
* Economic Times
* CNBC-TV18

These provide context around the primary events.

---

## Tier C (Broad news aggregators)

These help with coverage.

* GDELT
* Google News archives
* Kaggle historical datasets
* News APIs (where licensing permits)

These are useful because they let us find articles that mention companies, sectors, or macro events at scale. GDELT, for example, monitors global news in many languages and exposes historical archives and metadata for large-scale analysis. ([GDELT Project][1])

---

# Stage 1 — What Should We Download?

This is where most people make a mistake.

They scrape:

```
Headline
Date
```

That is **not enough**.

Instead, for every article we want:

```
Title

Subtitle

Body

Author

Published Time

Updated Time

Publisher

URL

Categories

Tags

Images (optional)

Language

Country

Source Type

Raw HTML (optional)

Article Text
```

Because five years later you may discover you need something you threw away.

Storage is cheap.

Research time is expensive.

---

# Stage 2 — Folder Structure

I don't like mixing everything.

I'd organize it like this:

```
StockPrediction/

    raw/

        Reuters/

            2014/

            2015/

            ...

        Moneycontrol/

        GDELT/

        NSE/

        RBI/

    cleaned/

    processed/

    embeddings/

    parquet/

    metadata/

    logs/

    company_mapping/

    models/
```

Everything is reproducible.

---

# Stage 3 — Raw Storage Format

I strongly recommend **JSONL (JSON Lines)**.

Each line contains one article.

Example:

```json
{
  "id":"...",
  "source":"Reuters",
  "datetime":"2020-08-01T09:35:11",
  "title":"TCS signs cloud deal",
  "body":"....",
  "url":"...",
  "language":"en"
}
```

Never store raw news in CSV.

CSV breaks on quotes, commas, multiline articles, and Unicode.

---

# Stage 4 — Database

We'll eventually use PostgreSQL (or DuckDB for local analytics), but **raw data stays on disk**.

The workflow becomes:

```
Internet

↓

Raw JSONL

↓

Clean JSONL

↓

Parquet

↓

Database

↓

Training
```

Parquet is dramatically faster for analytics than CSV.

---

# Stage 5 — Company Master Table

This is one of the most important datasets we'll build.

Example:

| Symbol   | Company                   | Sector    |
| -------- | ------------------------- | --------- |
| TCS      | Tata Consultancy Services | IT        |
| INFY     | Infosys                   | IT        |
| HDFCBANK | HDFC Bank                 | Banking   |
| RELIANCE | Reliance Industries       | Oil & Gas |
| SBIN     | State Bank of India       | Banking   |

But that's not enough.

We'll also store aliases.

Example:

```
TCS

"TCS"

"Tata Consultancy Services"

"Tata Consultancy"

"TCS Ltd"

"Tata Consultancy Services Ltd"
```

Why?

Because articles rarely use the exact NSE ticker.

---

# Stage 6 — Industry Dictionary

We'll also create a sector graph.

Example:

```
Oil

↓

Reliance

ONGC

BPCL

HPCL

IOC

Chennai Petroleum
```

Now suppose Reuters writes

```
Oil refiners under pressure...
```

No company is named.

But we still know the affected stocks.

---

# Stage 7 — Event Dictionary

We'll classify articles into event types.

Example:

```
Earnings

Merger

Acquisition

Dividend

CEO Change

Buyback

Stock Split

Fraud

Cyber Attack

Factory Fire

Government Policy

Tax Change

Repo Rate

Oil Prices

Inflation

Election

War

Natural Disaster
```

Eventually this will be around **100–150 standardized event classes**.

---

# Stage 8 — FinBERT

Only after all the above do we generate embeddings.

Instead of feeding only:

```
"TCS wins contract."
```

We'll enrich the context.

For example:

```
Title

+

Body

+

Company

+

Sector

+

Event Type
```

That generally produces more informative embeddings because the model sees the relevant context rather than an isolated headline.

---

# The first thing we'll build

I think this is the highest priority.

**A unified news downloader.**

Not a Reuters downloader.

Not a Moneycontrol downloader.

One framework.

```
news_pipeline/

    downloaders/

        gdelt.py

        reuters.py

        moneycontrol.py

        nse.py

        sebi.py

        rbi.py

        mint.py

    cleaners/

    parsers/

    deduplicator/

    mapper/

    finbert/

    exporter/
```

Then every downloader returns the **same schema**.

---

## Before we write a single line of code

There is one architectural decision that will save us months of work.

I propose that we **do not scrape publisher websites first** (Reuters, Moneycontrol, Mint, etc.).

Instead, we should:

1. Build the master company database (every NSE-listed company with aliases, sectors, ISINs, and metadata).
2. Download **GDELT** as our backbone because it gives us broad historical coverage and source URLs. ([GDELT Project][1])
3. Use GDELT metadata to identify relevant financial articles.
4. Supplement gaps with primary sources like NSE, BSE, RBI, and SEBI.
5. Only then add publisher-specific historical collections where legally and technically appropriate.

This gives us a scalable, legally cleaner foundation and avoids maintaining dozens of fragile web scrapers.

I also have a proposal that will make this project much stronger than a typical FinBERT pipeline: we'll build a **knowledge graph** connecting companies, sectors, macro events, government policies, commodities, currencies, and supply chains. Every news article will become a node connected to the entities it affects. That graph can generate richer features than sentiment alone and often captures indirect market impacts that simple embeddings miss.

I genuinely think this is the difference between a good academic project and something that could approach institutional-grade research.

[1]: https://www.gdeltproject.org/?utm_source=chatgpt.com "The GDELT Project"

---

Perfect. This is exactly how I like to work on research projects.

From this point onwards, let's think like we're building something that could be published in a paper or used by a quant fund. That means every decision should be reproducible, scalable, and explainable.

---

# Project Vision

Our goal is **not** to predict tomorrow's stock price.

Our goal is to build a **Financial Intelligence Engine**.

The prediction model will be just one consumer of that engine.

Think of it like this:

```
                    Financial Intelligence Engine
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
    FinBERT             Knowledge Graph        Time Series
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                    Prediction Models
```

This architecture lets us reuse the same data for different ML models later.

---

# Before Stage 1: Define the Research Question

This sounds academic, but it changes how we collect data.

For example:

### Question A

> Can today's news predict tomorrow's stock return?

Needs:

* Daily news
* Daily OHLCV

---

### Question B

> Can macro + company news predict the next 5 trading days?

Needs:

* Macro events
* Company events
* Global news
* 5-day returns

---

### Question C

> Can FinBERT embeddings improve transformer-based stock prediction?

Needs:

* FinBERT embeddings
* Historical prices
* Technical indicators
* Time alignment

---

I think we should aim for something broader:

> **Can a multimodal financial intelligence system combining market data, company events, macroeconomic events, and contextual financial news improve prediction of Indian stock returns across multiple forecast horizons?**

That gives us flexibility to experiment with different targets (next day, next week, volatility, direction, etc.).

---

# Phase 1 (The Foundation)

We are **not collecting news yet**.

We're building the dictionary that every future stage depends on.

```
Phase 1

Master Company Database

↓

Sector Database

↓

Industry Database

↓

Macro Dictionary

↓

Event Dictionary

↓

Alias Dictionary

↓

Knowledge Graph Skeleton
```

Without these, we can't reliably answer questions like:

* Which company is mentioned?
* Which sector is affected?
* Is this a macro event?
* Is this company-specific?

---

# Task 1 — Master Company Database

This is where we'll start.

Not downloading news.

Downloading **every NSE-listed company**.

For each company, we want:

| Field               | Description                   |
| ------------------- | ----------------------------- |
| NSE Symbol          | TCS                           |
| BSE Code            | 532540                        |
| Company Name        | Tata Consultancy Services Ltd |
| ISIN                | INE467B01029                  |
| Sector              | IT                            |
| Industry            | IT Services                   |
| Sub Industry        | Software Consulting           |
| Listing Date        | Exchange listing date         |
| Market Cap Category | Large / Mid / Small           |
| NIFTY Membership    | Yes / No                      |
| Registered Office   | Optional                      |
| Website             | Official website              |
| Status              | Active / Delisted             |

This becomes our **ground truth**.

---

# Task 2 — Company Alias Dictionary

This is where most NLP pipelines fail.

Example:

```
Tata Consultancy Services Ltd

↓

TCS

↓

Tata Consultancy

↓

TCS Ltd

↓

Tata Consultancy Services
```

All should resolve to the same canonical company.

We'll also capture:

```
Reliance Industries

↓

Reliance

↓

RIL

↓

Reliance Industries Ltd
```

This is crucial for entity mapping.

---

# Task 3 — Sector Taxonomy

Instead of a flat list, we'll use a hierarchy.

```
Financial Services

    Banking

        Private Banks

        PSU Banks

        Small Finance Banks

    NBFC

    Insurance

    Asset Management
```

Another example:

```
Information Technology

    IT Services

    Software Products

    Cloud

    Cybersecurity
```

This lets us infer broader impacts.

---

# Task 4 — Event Taxonomy

This is one of the most important parts of the project.

Instead of using sentiment alone, we'll classify **what actually happened**.

Here's an initial version:

### Company Events

* Earnings
* Dividend
* Bonus Issue
* Stock Split
* Buyback
* Rights Issue
* Promoter Buying
* Promoter Selling
* Large Order
* Order Cancellation
* Acquisition
* Merger
* Demerger
* Joint Venture
* CEO Change
* CFO Change
* Credit Rating Upgrade
* Credit Rating Downgrade
* Litigation
* Tax Notice
* Regulatory Action
* Factory Shutdown
* Plant Expansion
* Product Launch
* Patent Approval
* Patent Rejection
* Cyber Attack
* Data Breach
* Insolvency

### Macro Events

* Inflation
* GDP
* CPI
* WPI
* IIP
* PMI
* Repo Rate
* Reverse Repo
* CRR
* SLR
* Fiscal Deficit
* Budget
* Monsoon
* Currency
* Oil Price
* Gold Price
* US Fed
* ECB
* OPEC
* Trade Agreement
* War
* Pandemic

Eventually, we'll expand this to 100–150 standardized event types.

---

# Task 5 — Knowledge Graph Skeleton

This is the part I'm most excited about because it differentiates our project.

Imagine the following relationships:

```
Oil Price ↑

↓

Oil Refining

↓

Reliance
BPCL
HPCL
IOC
```

Or:

```
US Fed

↓

Dollar

↓

USDINR

↓

Indian IT Companies
```

Or:

```
Monsoon

↓

Agriculture

↓

Fertilizer

↓

Coromandel
Chambal Fertilisers
```

This graph lets us capture indirect effects that simple keyword matching misses.

---

# Project Folder Structure

I'd recommend this from day one:

```
indian-stock-intelligence/

│

├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── external/
│   └── embeddings/
│
├── metadata/
│   ├── companies/
│   ├── sectors/
│   ├── events/
│   ├── aliases/
│   └── macro/
│
├── knowledge_graph/
│
├── pipelines/
│   ├── download/
│   ├── clean/
│   ├── map/
│   ├── enrich/
│   ├── embed/
│   └── export/
│
├── models/
│
├── notebooks/
│
├── tests/
│
└── docs/
```

---

# Our First Milestone

I want us to complete this before touching FinBERT:

* ✅ Master company database
* ✅ Alias database
* ✅ Sector hierarchy
* ✅ Industry hierarchy
* ✅ Event taxonomy (v1)
* ✅ Macro event taxonomy
* ✅ Folder structure

Once these are in place, every subsequent stage becomes cleaner and more reliable.

---

## One improvement I'd like to make

Rather than thinking of this as "collecting news," I'd like us to think of it as building a **Financial Intelligence Data Lake**.

Everything—news, company filings, macro releases, prices, technical indicators, insider trades, FII/DII flows, commodities, currencies—will be different "layers" in the same lake.

That means if, a year from now, you decide to replace FinBERT with a newer finance language model or build a graph neural network, you won't have to rebuild your data pipeline.

### The next step

I suggest our **first coding task** is to build the **Master Company Database**. We'll automatically collect every NSE-listed company, BSE mapping, ISIN, sector, industry, aliases, and related metadata into a canonical dataset. This database will become the backbone for every news article we process later.

I also recommend we maintain a `docs/` directory where we document every design decision. Treat it like a research lab notebook. Months from now, when we're tuning models or writing a paper, those records will be incredibly valuable.

-----

😂 Fair enough. That's actually a good correction.

You're right.

For the rest of this project, I'll switch to **Senior ML Engineer / Quant Engineer mode**.

My responses will be:

* ✅ What we're building
* ✅ Why (2-3 lines max)
* ✅ Code
* ✅ Output
* ✅ Next Step

No more essay unless you ask for it.

---

# Step 1 — Master Company Database

## Goal

Create one canonical dataset of all NSE companies.

Output:

```
companies_master.parquet
companies_master.csv
```

Fields:

```
symbol
company_name
isin
industry
sector
series
listing_date
market_cap
bse_code
website
status
aliases
```

---

# Project Structure

```
indian-stock-intelligence/

    data/
        raw/
        processed/

    scripts/

        build_master_company_db.py

    config.py

    requirements.txt
```

---

# Install

```bash
pip install pandas requests beautifulsoup4 lxml pyarrow tqdm yfinance nselib
```

---

# build_master_company_db.py

```python
import pandas as pd
from nselib import capital_market
from tqdm import tqdm

print("Downloading NSE Equity List...")

equities = capital_market.equity_list()

df = pd.DataFrame(equities)

print(df.head())

df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Rename important columns
rename_map = {
    "symbol": "symbol",
    "name_of_company": "company_name",
    "series": "series",
    "date_of_listing": "listing_date",
    "isin_number": "isin",
    "industry": "industry"
}

df = df.rename(columns=rename_map)

# Keep useful columns
cols = [
    "symbol",
    "company_name",
    "series",
    "listing_date",
    "isin",
    "industry"
]

df = df[cols]

# Status
df["status"] = "ACTIVE"

# Placeholder columns
df["sector"] = None
df["market_cap"] = None
df["bse_code"] = None
df["website"] = None

# Aliases
def aliases(row):

    company = row["company_name"]

    aliases = set()

    aliases.add(company)

    aliases.add(company.replace("Limited","").strip())

    aliases.add(company.replace("Ltd.","").strip())

    aliases.add(company.replace("Ltd","").strip())

    aliases.add(company.replace("LIMITED","").strip())

    aliases.add(row["symbol"])

    return list(aliases)

df["aliases"] = df.apply(aliases, axis=1)

print(df.head())

df.to_csv("data/processed/companies_master.csv", index=False)

df.to_parquet("data/processed/companies_master.parquet", index=False)

print(f"\nSaved {len(df)} companies.")
```

---

# Expected Output

```
symbol company_name industry

TCS
INFY
SBIN
RELIANCE
HDFCBANK
...
```

Around

```
~1900–2200 companies
```

depending on current listings.

---

# Problem

This is **NOT enough**.

Because NSE does **not** give us

* sector
* market cap
* BSE code
* website
* aliases
* former names

---

# What we'll do next (important)

We'll build a **Company Enrichment Pipeline**.

It will automatically enrich every company using multiple sources.

```
NSE
      +
BSE
      +
Tickertape
      +
Screener
      +
Yahoo Finance
      +
Official Website

↓

Master Company Database
```

Final database:

```
symbol

company_name

isin

sector

industry

sub_industry

market_cap

market_cap_category

bse_code

face_value

listing_date

website

cin

registered_office

promoters

aliases

former_names

nifty_member

sensex_member
```

This becomes our **single source of truth**.

---

## Your Task

Run this first.

Don't improve anything yet.

Just confirm:

```
companies_master.csv
```

gets created successfully.

**After that, we'll build `company_enrichment.py`, which is where the real engineering starts.** That script will make this dataset institutional-grade instead of just an NSE download.

------

You're right. I overexplained and drifted from execution. From now on I'll keep it to **task → code → next**.

You're also correct about the flow.

## Current Pipeline

```
Master Company DB ✅

↓

Enrich Company DB (sector, aliases, etc.)

↓

Collect News (2014-2025)

↓

Clean

↓

Deduplicate

↓

Map News → Companies

↓

Map News → Industries

↓

Event Classification

↓

FinBERT Embeddings

↓

Merge with Stock Data
```

---

## The CSV you have now

Keep it.

**Don't worry about these yet:**

* market_cap
* website
* bse_code

They are **not needed** for news mapping.

The only important fields right now are:

```
symbol
company_name
industry
aliases
```

---

## Next Task

**Build a proper Alias Dictionary.**

Your current alias generation is weak.

Example:

```
Tata Consultancy Services Limited

↓

TCS

↓

Tata Consultancy Services

↓

Tata Consultancy

↓

TCS Ltd
```

Current code doesn't generate these properly.

Create a new file:

```
scripts/build_alias_dictionary.py
```

```python
import pandas as pd
import re
import ast

df = pd.read_csv("data/processed/companies_master.csv")

REMOVE_WORDS = [
    "LIMITED",
    "LTD",
    "LTD.",
    "LIMITED.",
    "PRIVATE",
    "PVT",
    "PVT.",
    "INDIA",
    "(INDIA)"
]


def clean(name):
    name = str(name).upper()

    name = re.sub(r"[^\w\s]", " ", name)

    words = name.split()

    words = [w for w in words if w not in REMOVE_WORDS]

    return " ".join(words).strip()


def generate_aliases(row):

    aliases = set()

    company = clean(row["company_name"])

    aliases.add(company)

    aliases.add(row["symbol"].upper())

    words = company.split()

    # Tata Consultancy Services
    aliases.add(" ".join(words))

    # Tata Consultancy
    if len(words) >= 2:
        aliases.add(" ".join(words[:2]))

    # First word
    aliases.add(words[0])

    # Without last word
    if len(words) >= 3:
        aliases.add(" ".join(words[:-1]))

    return sorted(list(aliases))


df["aliases"] = df.apply(generate_aliases, axis=1)

df.to_csv(
    "data/processed/companies_master.csv",
    index=False
)

print(df[["symbol", "company_name", "aliases"]].head())
```

Run it.

---

## After this

**We are NOT touching FinBERT.**

Next we'll start collecting news.

The first source will be **GDELT** because it gives us timestamps, source URLs, and historical coverage.

We'll build:

```
download_gdelt.py
```

which downloads news day-by-day and stores **JSONL** exactly as we discussed.

That is the next coding step after aliases.
