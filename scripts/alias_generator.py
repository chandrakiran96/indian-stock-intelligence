import pandas as pd
import re
import json

INPUT = "data/processed/companies_master.csv"
OUTPUT = "data/processed/companies_new_master_enriched.csv"

# 1. Words to strip completely from the START of a name
PREFIX_REGEX = r"^(THE|A|AN|DR|MR|MRS|SHRI|SHREE)\s+"

# 2. Suffixes to strip from the END of a name
REMOVE_SUFFIX = {
    "LIMITED", "LTD", "PRIVATE", "PVT", "INC", "CORP", 
    "CORPORATION", "LLC", "LLP", "PLC", "CO", "COMPANY", "LTD."
}

# 3. Known generic words that corrupt multi-word chunks
GENERIC_WORDS = {
    "BANK", "INDIA", "STATE", "NEW", "SOUTH", "NORTH", "GLOBAL", 
    "SHARE", "INVESTMENT", "PRIME", "NATIONAL", "INTERNATIONAL", 
    "FINANCE", "CAPITAL", "TECH", "TECHNOLOGY", "INDUSTRIES", "GROUP"
}

# 4. Mega-Cap exceptions (Companies universally known by their first word)
MEGA_CAP_SHORTNAMES = {
    "RELIANCE", "TATA", "MAHINDRA", "BAJAJ", "MARUTI", "WIPRO", 
    "INFOSYS", "HDFC", "ICICI", "BIOCON", "GODREJ", "ADANI", 
    "VEDANTA", "ITC", "LARSEN"
}

def normalize(s):
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_base_name(name):
    s = normalize(name)
    s = re.sub(PREFIX_REGEX, "", s) # Strip "THE ", "SHRI ", etc.
    words = s.split()
    while words and words[-1] in REMOVE_SUFFIX:
        words.pop()
    return " ".join(words)

# --- Main Loop ---
df = pd.read_csv(INPUT)
high_confidence_list, high_risk_list = [], []

for _, row in df.iterrows():
    company = row["Company Name"]
    base = clean_base_name(company)
    words = base.split()
    
    high_conf, high_risk = set(), set()
    
    if not words:
        high_confidence_list.append("[]")
        high_risk_list.append("[]")
        continue

    # 1. Full Base Name Processing
    if len(words) > 1:
        # e.g., "NANDAN DENIM", "BANK OF MAHARASHTRA"
        if words[0] in GENERIC_WORDS:
            high_risk.add(base)
        else:
            high_conf.add(base)
    elif len(words) == 1:
        # True Single-word companies (e.g. "INFOSYS")
        w = words[0]
        if w in GENERIC_WORDS or len(w) <= 3:
            high_risk.add(w)
        else:
            high_conf.add(w)

    # 2. Extract 2-Word and 3-Word chunks
    if len(words) >= 2:
        two_word = " ".join(words[:2])
        # Force high risk if it's generic like "BANK OF"
        if words[0] in GENERIC_WORDS or words[1] in {"OF", "AND", "&", "IN"}:
            high_risk.add(two_word)
        else:
            high_conf.add(two_word)
            
    if len(words) >= 3:
        high_conf.add(" ".join(words[:3]))

    # 3. THE 1-WORD RULE (The Ultimate False-Positive Killer)
    # If the company is multi-word, DO NOT extract the first word as an alias...
    if len(words) > 1:
        first_word = words[0]
        # ...UNLESS it is a recognized Mega-Cap bluechip.
        if first_word in MEGA_CAP_SHORTNAMES:
            high_risk.add(first_word) # Still high risk to demand context

    # Final Cleanup
    high_conf = {a for a in high_conf if len(a) > 2}
    high_risk = {a for a in high_risk if len(a) > 2}
    high_conf = high_conf - high_risk # Risk always wins conflicts
    
    high_confidence_list.append(json.dumps(list(high_conf)))
    high_risk_list.append(json.dumps(list(high_risk)))

df["High_Confidence_Aliases"] = high_confidence_list
df["High_Risk_Aliases"] = high_risk_list
df.to_csv(OUTPUT, index=False)
print("Master sheet regenerated with the Two-Word Minimum Rule!")