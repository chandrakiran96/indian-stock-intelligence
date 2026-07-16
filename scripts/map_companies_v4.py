import json
import re
import pandas as pd
from tqdm import tqdm

COMPANY_CSV = "data/processed/companies_new_master_enriched.csv"
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"

FINANCIAL_CONTEXT = {
    "STOCK", "STOCKS", "SHARE", "SHARES", "MARKET", "MARKETS", "NSE", "BSE", 
    "NIFTY", "SENSEX", "PROFIT", "DIVIDEND", "QUARTER", "EARNINGS", "BOARD", 
    "IPO", "INVESTOR", "INVESTORS", "BULLISH", "BEARISH", "VOLUME", "REVENUE",
    "SEBI", "CEO", "CFO", "STAKE", "EQUITY", "FUND", "BROKER"
}

print("Loading algorithmic profiles...")
df = pd.read_csv(COMPANY_CSV)
all_high_confidence, all_high_risk, all_symbols, company_profiles = {}, {}, {}, {}

for _, row in df.iterrows():
    sym = str(row["Symbol"]).strip().upper()
    try:
        hc_aliases = json.loads(row["High_Confidence_Aliases"])
        hr_aliases = json.loads(row["High_Risk_Aliases"])
    except: continue

    company_profiles[sym] = {"company_name": str(row["Company Name"]).strip(), "sector": str(row["Sector"]).strip()}
    
    for a in hc_aliases: all_high_confidence[a] = sym
    for a in hr_aliases: all_high_risk[a] = sym
    all_symbols[sym] = sym

# Regex compilers
def compile_regex(items):
    if not items: return re.compile(r"^\b$")
    sorted_items = sorted(list(items), key=len, reverse=True)
    # Ignore case for full names and aliases
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted_items)) + r")\b", re.IGNORECASE)

def compile_strict_case_regex(items):
    if not items: return re.compile(r"^\b$")
    sorted_items = sorted(list(items), key=len, reverse=True)
    # Strict case for Symbols to prevent 'campus' matching 'CAMPUS'
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted_items)) + r")\b")

regex_high_conf = compile_regex(all_high_confidence.keys())
regex_high_risk = compile_regex(all_high_risk.keys())
regex_symbols = compile_strict_case_regex(all_symbols.keys())

regex_financial_context = compile_regex(FINANCIAL_CONTEXT)

def has_local_context(text, match_span, window_size=400):
    """Checks if financial keywords exist within X characters of the found alias."""
    start = max(0, match_span[0] - window_size)
    end = min(len(text), match_span[1] + window_size)
    local_text = text[start:end]
    
    # FIX: Use the compiled regex object here
    return bool(regex_financial_context.search(local_text))

print("Processing articles with Proximity & Case constraints...")
with open(INPUT_JSONL, encoding="utf-8") as fin, open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for line in tqdm(fin):
        article = json.loads(line)
        # We DO NOT use .upper() anymore. We preserve the original text.
        full_text = f"{article.get('title', '')}\n{article.get('text', '')}"
        matched_companies = {}

        # LAYER 1: High Confidence (Matches anywhere, case-insensitive)
        for match in regex_high_conf.finditer(full_text):
            found_str = match.group().upper() # uppercase just for dictionary lookup
            sym = all_high_confidence[found_str]
            matched_companies[sym] = matched_companies.get(sym, set())
            matched_companies[sym].add(found_str)

        # LAYER 2: Symbols (Strict Casing + Local Context if short)
        for match in regex_symbols.finditer(full_text):
            found_str = match.group() # Keeps original casing
            
            # If symbol is short, or it is entirely lowercase (e.g., 'campus' at start of sentence), require context
            if len(found_str) <= 4 or found_str.islower():
                if not has_local_context(full_text, match.span()):
                    continue
                    
            sym = all_symbols[found_str.upper()]
            matched_companies[sym] = matched_companies.get(sym, set())
            matched_companies[sym].add(found_str)

        # LAYER 3: High Risk (Strict Local Context Required)
        for match in regex_high_risk.finditer(full_text):
            found_str = match.group().upper()
            if has_local_context(full_text, match.span()):
                sym = all_high_risk[found_str]
                matched_companies[sym] = matched_companies.get(sym, set())
                matched_companies[sym].add(found_str)

        # Write Output
        article["matched_symbols"] = list(matched_companies.keys())
        article["match_metadata"] = {sym: list(aliases) for sym, aliases in matched_companies.items()}
        fout.write(json.dumps(article, ensure_ascii=False) + "\n")