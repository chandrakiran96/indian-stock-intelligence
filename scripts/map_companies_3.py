import json
import re
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================
COMPANY_CSV = "data/processed/companies_master_enriched_2.csv"
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"

GENERIC_TICKERS = {
    "GLOBAL", "SHANTI", "PRAKASH", "FUTURE", "NATIONAL", "PUNJAB", 
    "BANK", "INDIA", "SOUTH", "WEST", "EAST", "NORTH", "STAR", 
    "SUN", "TIME", "MODERN", "PREMIER", "SMART", "EXCEL", "CHOICE", 
    "MAX", "TRENT", "PAGE", "DELTA", "ACC"
}

# Financial context keywords required to validate High-Risk aliases or isolated Symbols
FINANCIAL_CONTEXT = {
    "STOCK", "STOCKS", "SHARE", "SHARES", "MARKET", "MARKETS", "NSE", "BSE", 
    "NIFTY", "SENSEX", "PROFIT", "DIVIDEND", "QUARTER", "EARNINGS", "BOARD MEETING", 
    "IPO", "INVESTOR", "INVESTORS", "CMP", "BULLISH", "BEARISH", "VOLUME", "DELIST"
}

# Indian Market Sector Keyword Taxonomy (for standalone industry news tracking)
SECTOR_TAXONOMY = {
    "FINANCIAL SERVICES": ["BANKING", "NIFTY BANK", "NPA", "RBI", "LOAN GROWTH", "NBFC", "PSU BANK"],
    "INFORMATION TECHNOLOGY": ["NIFTY IT", "SOFTWARE SERVICES", "IT OUTSOURCING", "DIGITAL TRANSFORMATION", "TCS", "INFRA"],
    "AUTOMOBILE": ["AUTO SALES", "PASSENGER VEHICLES", "TWO-WHEELERS", "EV ADOPTION", "COMMERCIAL VEHICLES"],
    "PHARMACEUTICALS": ["PHARMA", "USFDA APPROVAL", "DRUG FORMS", "HEALTHCARE SECTOR", "CLINICAL TRIALS"],
    "FAST MOVING CONSUMER GOODS": ["FMCG", "RURAL DEMAND", "CONSUMER GOODS", "INFLATION IMPACT"]
}

# =====================================================
# LOAD & ORGANIZE COMPANY DATABASE
# =====================================================
print("Loading company database and building high-speed profiles...")
companies_df = pd.read_csv(COMPANY_CSV)

company_profiles = {}
all_high_confidence = {}
all_high_risk = {}
all_symbols = {}

for _, row in companies_df.iterrows():
    symbol = str(row["Symbol"]).strip().upper()
    
    # Safe JSON parsing of lists generated in the previous step
    try:
        high_conf_aliases = json.loads(row["High_Confidence_Aliases"])
        high_risk_aliases = json.loads(row["High_Risk_Aliases"])
    except:
        continue

    company_profiles[symbol] = {
        "company_name": str(row["Company Name"]).strip(),
        "sector": str(row["Sector"]).strip().upper(),
        "industry": str(row["Industry"]).strip().upper()
    }

    # Map aliases back to the parent stock symbol
    for alias in high_conf_aliases:
        all_high_confidence[alias.upper()] = symbol
        
    for alias in high_risk_aliases:
        all_high_risk[alias.upper()] = symbol

    all_symbols[symbol] = symbol

# High-speed compilation: Group terms into master regex patterns to avoid slow individual looping
def compile_bulk_regex(items_list):
    if not items_list:
        return re.compile(r"肢") # Unmatchable fallback
    # Sort by length descending so longer phrases match before sub-words
    sorted_items = sorted(list(items_list), key=len, reverse=True)
    pattern_str = r"\b(" + "|".join(map(re.escape, sorted_items)) + r")\b"
    return re.compile(pattern_str)

print("Compiling global pattern matchers...")
regex_high_conf = compile_bulk_regex(all_high_confidence.keys())
regex_high_risk = compile_bulk_regex(all_high_risk.keys())
regex_symbols = compile_bulk_regex(all_symbols.keys())

# =====================================================
# PROCESSING PIPELINE
# =====================================================
print("Processing and screening articles...")

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):
        article = json.loads(line)
        
        # Combine text fields and normalize to uppercase
        title_text = article.get("title", "")
        body_text = article.get("text", "")
        full_text = f"{title_text}\n{body_text}".upper()

        # Tokenize words quickly to check for financial context presence
        tokens = set(re.findall(r"\b[A-Z]+\b", full_text))
        has_financial_context = not tokens.isdisjoint(FINANCIAL_CONTEXT)

        matched_companies_data = {}
        
        # -----------------------------------------------------------------
        # LAYER 1: High Confidence Aliases (Direct Matches)
        # -----------------------------------------------------------------
        hc_matches = regex_high_conf.findall(full_text)
        for match in hc_matches:
            sym = all_high_confidence[match]
            if sym not in matched_companies_data:
                matched_companies_data[sym] = {"source": "High_Confidence", "aliases": set(), "count": 0}
            matched_companies_data[sym]["aliases"].add(match)
            matched_companies_data[sym]["count"] += 1

        # -----------------------------------------------------------------
        # LAYER 2: Strict Stock Symbols (Requires Context to avoid generic nouns)
        # -----------------------------------------------------------------
        sym_matches = regex_symbols.findall(full_text)
        for match in sym_matches:
            # Trap 1: Is it a financial keyword itself? (e.g. "PROFIT")
            if match in FINANCIAL_CONTEXT:
                if not has_financial_context:
                    continue
            
            # Trap 2: Is it short, OR is it a known dictionary/generic name?
            is_short = len(match) <= 4
            is_generic = match in GENERIC_TICKERS
            
            if (is_short or is_generic) and not has_financial_context:
                continue
            
            if match not in matched_companies_data:
                matched_companies_data[match] = {"source": "Symbol", "aliases": set(), "count": 0}
            matched_companies_data[match]["aliases"].add(match)
            matched_companies_data[match]["count"] += 1

        # -----------------------------------------------------------------
        # LAYER 3: High Risk Aliases (Strict Context Enforcement)
        # -----------------------------------------------------------------
        if has_financial_context:
            hr_matches = regex_high_risk.findall(full_text)
            for match in hr_matches:
                sym = all_high_risk[match]
                if sym not in matched_companies_data:
                    matched_companies_data[sym] = {"source": "High_Risk_Enforced", "aliases": set(), "count": 0}
                matched_companies_data[sym]["aliases"].add(match)
                matched_companies_data[sym]["count"] += 1

        # -----------------------------------------------------------------
        # LAYER 4: SECTOR & INDUSTRY MATCHING ENGINE
        # -----------------------------------------------------------------
        final_sectors = set()
        final_industries = set()
        article_tags = []

        if matched_companies_data:
            # Case A: Explicit company mentions automatically inherit their sectors
            for sym, info in matched_companies_data.items():
                profile = company_profiles[sym]
                final_sectors.add(profile["sector"])
                final_industries.add(profile["industry"])
            article_tags.append("COMPANY_SPECIFIC")
        else:
            # Case B: Standalone Sector News (No specific company found)
            for sector_name, keywords in SECTOR_TAXONOMY.items():
                # If sector keywords appear in a financially contextual article
                if has_financial_context and any(kw in full_text for kw in keywords):
                    final_sectors.add(sector_name)
                    article_tags.append("SECTOR_MACRO")

        # -----------------------------------------------------------------
        # FORMAT TRANSFORMATIONS & WRITE OUTPUT
        # -----------------------------------------------------------------
        article["matched_symbols"] = sorted(list(matched_companies_data.keys()))
        article["matched_companies"] = [company_profiles[s]["company_name"] for s in article["matched_symbols"]]
        article["matched_sectors"] = sorted(list(final_sectors))
        article["matched_industries"] = sorted(list(final_industries))
        article["processing_tags"] = article_tags
        
        # Compile metadata profiles for tracking validation metrics
        article["match_metadata"] = {
            sym: {
                "detected_aliases": sorted(list(info["aliases"])),
                "total_mentions": info["count"],
                "confidence_tier": info["source"]
            } for sym, info in matched_companies_data.items()
        }

        fout.write(json.dumps(article, ensure_ascii=False) + "\n")

print("Historical mapping script execution complete.")
