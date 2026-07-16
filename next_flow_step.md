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


