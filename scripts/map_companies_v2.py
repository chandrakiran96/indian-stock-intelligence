import ast
import json
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

COMPANY_CSV = "data/processed/companies_master.csv"
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"

# =====================================================
# LOAD COMPANY DATABASE
# =====================================================

print("Loading company database...")

companies = pd.read_csv(COMPANY_CSV, encoding="latin1")

patterns = []

for _, row in companies.iterrows():

    symbol = str(row["Symbol"]).strip().upper()
    company = str(row["Company Name"]).strip()
    sector = str(row["Sector"]).strip()
    industry = str(row["Industry"]).strip()

    raw_aliases = str(row["Expanded Aliases"])
    
    # Search for anything between brackets [...]
    match = re.search(r'\[.*?\]', raw_aliases)

    raw_aliases = str(row["Expanded Aliases"])
    
    # Search for anything between brackets [...]
    match = re.search(r'\[.*?\]', raw_aliases)
    
    aliases = []
    if match:
        try:
            aliases = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            print(f"Warning: Failed to evaluate aliases for {symbol}: {match.group(0)}")
            continue
    else:
        # If no list is found, skip or leave aliases empty
        continue

    for alias in aliases:
        alias = str(alias).strip().upper()

        if len(alias) == 0:
            continue

        # -------------------------------------------------
        # Remove dangerous short aliases
        # -------------------------------------------------

        if len(alias) < 4 and alias != symbol:
            continue

        # -------------------------------------------------
        # Whole word regex
        # -------------------------------------------------

        pattern = re.compile(
            r"(?<![A-Z0-9])"
            + re.escape(alias)
            + r"(?![A-Z0-9])"
        )

        patterns.append(
            {
                "pattern": pattern,
                "alias": alias,
                "symbol": symbol,
                "company": company,
                "sector": sector,
                "industry": industry,
            }
        )

print(f"Loaded {len(patterns)} alias patterns")

# =====================================================
# MATCH ARTICLES
# =====================================================

print("Mapping articles...")

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):

        article = json.loads(line)

        text = (
            article.get("title", "")
            + "\n"
            + article.get("text", "")
        ).upper()

        matched = {}

        for item in patterns:

            matches = item["pattern"].findall(text)

            if not matches:
                continue

            symbol = item["symbol"]

            if symbol not in matched:

                matched[symbol] = {

                    "company": item["company"],
                    "sector": item["sector"],
                    "industry": item["industry"],
                    "aliases": set(),
                    "mention_count": 0,
                    "score": 0,
                }

            matched[symbol]["aliases"].add(item["alias"])

            matched[symbol]["mention_count"] += len(matches)

            # -----------------------------
            # Score
            # -----------------------------

            alias = item["alias"]

            if alias == symbol:

                matched[symbol]["score"] += 3

            elif len(alias.split()) >= 2:

                matched[symbol]["score"] += 5

            else:

                matched[symbol]["score"] += 4

        article["matched_symbols"] = sorted(matched.keys())

        article["matched_companies"] = [
            matched[s]["company"]
            for s in article["matched_symbols"]
        ]

        article["matched_sectors"] = sorted(
            list(
                {
                    matched[s]["sector"]
                    for s in matched
                }
            )
        )

        article["matched_industries"] = sorted(
            list(
                {
                    matched[s]["industry"]
                    for s in matched
                }
            )
        )

        article["matched_aliases"] = {

            s: sorted(list(matched[s]["aliases"]))
            for s in matched

        }

        article["mention_count"] = {

            s: matched[s]["mention_count"]
            for s in matched

        }

        article["match_score"] = {

            s: matched[s]["score"]
            for s in matched

        }

        fout.write(
            json.dumps(article, ensure_ascii=False)
            + "\n"
        )

print("Done.")