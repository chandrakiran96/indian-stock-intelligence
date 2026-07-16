import spacy
import json
import re
import pandas as pd
from tqdm import tqdm

# =====================================================
# 1. INITIALIZE NLP (Disable unused pipes for max speed)
# =====================================================
print("Loading NLP Engine...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# =====================================================
# 2. DUAL-STREAM PROFILE LOADING
# =====================================================
COMPANY_CSV = "data/processed/companies_new_master_enriched.csv"
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"

df = pd.read_csv(COMPANY_CSV)

# Stream A: Safe, unique names (can match anywhere in text)
safe_aliases = {}
# Stream B: Risky, generic names (MUST be tagged as ORG by AI)
risk_aliases = {}

for _, row in df.iterrows():
    sym = str(row["Symbol"]).strip().upper()
    try:
        hc_aliases = json.loads(row["High_Confidence_Aliases"])
        hr_aliases = json.loads(row["High_Risk_Aliases"])
    except:
        continue
        
    for alias in hc_aliases: safe_aliases[alias.upper()] = sym
    for alias in hr_aliases: risk_aliases[alias.upper()] = sym

def compile_regex(aliases_dict):
    if not aliases_dict: return re.compile(r"^\b$")
    sorted_items = sorted(list(aliases_dict.keys()), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted_items)) + r")\b", re.IGNORECASE)

regex_safe = compile_regex(safe_aliases)
regex_risk = compile_regex(risk_aliases)

# =====================================================
# 3. THE HYBRID PARSING PIPELINE
# =====================================================
print("Processing with Hybrid NLP/Regex Engine...")

with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    
    for line in tqdm(fin):
        article = json.loads(line)
        full_text = f"{article.get('title', '')}\n{article.get('text', '')}"
        
        matched_companies_data = {}

        # -------------------------------------------------------------
        # LAYER 1: UNSTOPPABLE REGEX (For safe, multi-word, unique names)
        # -------------------------------------------------------------
        # Matches Wockhardt, Exide Industries, Infosys instantly, regardless of casing.
        for match in regex_safe.findall(full_text):
            clean_match = match.upper()
            sym = safe_aliases[clean_match]
            matched_companies_data.setdefault(sym, set()).add(clean_match)

        # -------------------------------------------------------------
        # LAYER 2: AI GATEKEEPER (For risky names like "Campus", "Arvind")
        # -------------------------------------------------------------
        # 1. AI reads the text
        doc = nlp(full_text)
        
        # 2. Extract ONLY confirmed Organizations
        org_chunks = [ent.text.upper() for ent in doc.ents if ent.label_ == "ORG"]
        safe_org_text = " || ".join(org_chunks)
        
        # 3. Search for risky names ONLY inside the AI-approved text
        for match in regex_risk.findall(safe_org_text):
            clean_match = match.upper()
            sym = risk_aliases[clean_match]
            matched_companies_data.setdefault(sym, set()).add(clean_match)

        # -------------------------------------------------------------
        # FORMAT & EXPORT
        # -------------------------------------------------------------
        article["matched_symbols"] = list(matched_companies_data.keys())
        article["match_metadata"] = {sym: list(aliases) for sym, aliases in matched_companies_data.items()}
        
        fout.write(json.dumps(article, ensure_ascii=False) + "\n")

print("10-Year Pipeline is now production-ready.")