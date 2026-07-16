import spacy
import json
import re
import pandas as pd
from tqdm import tqdm

print("Initializing Enterprise NLP Engine...")
# We use the lightning-fast core, disabling unused components to process years of data rapidly
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# =====================================================
# 1. LOAD RAW DATA (No Alias Generator needed)
# =====================================================
COMPANY_CSV = "data/processed/companies_master.csv"  # Use your RAW master file
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"

df = pd.read_csv(COMPANY_CSV)

REMOVE_SUFFIX = r"\b(LIMITED|LTD|PRIVATE|PVT|INC|CORP|CORPORATION|LLC|LLP|PLC|CO|COMPANY)\b"


multi_word_dict = {}  # E.g., "WESTERN INDIA PLYWOODS" -> "WIPL"
single_word_dict = {} # E.g., "INFOSYS" -> "INFY", "ARVIND" -> "ARVIND"
symbols_dict = {}     # E.g., "TCS" -> "TCS"

# Mega-Caps that journalists globally abbreviate to one word
MEGA_CAPS = {
    "RELIANCE": "RELIANCE", "WIPRO": "WIPRO", "INFOSYS": "INFY",
    "TCS": "TCS", "HDFC": "HDFCBANK", "ICICI": "ICICIBANK",
    "ITC": "ITC", "LARSEN": "LT", "MARUTI": "MARUTI"
}

print("Building strict Entity Dictionaries...")
for _, row in df.iterrows():
    sym = str(row["Symbol"]).strip().upper()
    raw_name = str(row["Company Name"]).strip().upper()

    # Create a pure Base Name (e.g., "BANK OF MAHARASHTRA LIMITED" -> "BANK OF MAHARASHTRA")
    base_name = re.sub(r"[^\w\s&]", " ", raw_name)
    base_name = re.sub(REMOVE_SUFFIX, "", base_name).strip()
    base_name = re.sub(r"\s+", " ", base_name).strip()

    if len(base_name) < 3: 
        continue

    # Sort into precise matching buckets (NEVER split the words)
    if len(base_name.split()) > 1:
        multi_word_dict[base_name] = sym
    else:
        single_word_dict[base_name] = sym

    # Save Symbol (Ignoring standard English words that double as tickers)
    if len(sym) >= 3 and sym not in {"IDEA", "PAGE", "TIME", "STAR", "FOCUS", "CAMPUS"}:
        symbols_dict[sym] = sym

# Pre-compile the multi-word exact match regex for blazing speed
sorted_multi = sorted(list(multi_word_dict.keys()), key=len, reverse=True)
regex_multi = re.compile(r"\b(" + "|".join(map(re.escape, sorted_multi)) + r")\b", re.IGNORECASE)

# =====================================================
# 2. THE SINGLE-PASS ARCHITECTURE
# =====================================================
print("Processing News Corpus...")

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):
        article = json.loads(line)
        full_text = f"{article.get('title', '')}\n{article.get('text', '')}"
        
        matches = set()
        metadata_tracker = {}

        # -------------------------------------------------------------
        # LAYER 1: MULTI-WORD EXACT MATCHER (Bypasses AI)
        # -------------------------------------------------------------
        # Instantly catches "Bank of Maharashtra", "Western India Plywoods", "Exide Industries"
        for match in regex_multi.findall(full_text):
            clean_match = match.upper()
            sym = multi_word_dict[clean_match]
            matches.add(sym)
            metadata_tracker[sym] = "Multi-Word Base Name"

        # -------------------------------------------------------------
        # LAYER 2: NLP AI GATEKEEPER (For short/risky names)
        # -------------------------------------------------------------
        doc = nlp(full_text)
        org_entities = [ent.text.upper() for ent in doc.ents if ent.label_ == "ORG"]

        for org in org_entities:
            clean_org = re.sub(REMOVE_SUFFIX, "", org).strip()
            clean_org = re.sub(r"[^\w\s&]", " ", clean_org)
            clean_org = re.sub(r"\s+", " ", clean_org).strip()

            # A. Check if the entire extracted ORG is a perfect match
            if clean_org in single_word_dict:
                sym = single_word_dict[clean_org]
                matches.add(sym)
                metadata_tracker[sym] = f"NLP Exact: {clean_org}"
                continue

            # B. If not a perfect match, unpack the phrase and look for Mega-Caps or Symbols
            for word in clean_org.split():
                if word in MEGA_CAPS:
                    sym = MEGA_CAPS[word]
                    matches.add(sym)
                    metadata_tracker[sym] = f"NLP Mega-Cap: {word}"
                
                elif word in symbols_dict:
                    sym = symbols_dict[word]
                    matches.add(sym)
                    metadata_tracker[sym] = f"NLP Symbol: {word}"
                    
                elif word in single_word_dict:
                    # Catch dictionary words ONLY if the AI explicitly grouped them into an ORG
                    sym = single_word_dict[word]
                    matches.add(sym)
                    metadata_tracker[sym] = f"NLP Verified Word: {word}"
                    
        # Export Format
        article["matched_symbols"] = list(matches)
        article["match_logic"] = metadata_tracker
        fout.write(json.dumps(article, ensure_ascii=False) + "\n")

print("Deep Entity Resolution Pipeline Complete.")