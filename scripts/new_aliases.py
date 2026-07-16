import pandas as pd
import re
import json

INPUT = "data/processed/companies_master.csv"
OUTPUT = "data/processed/companies_master_enriched_2.csv"

# Comprehensive list of corporate noise
REMOVE_SUFFIX = {
    "LIMITED", "LTD", "PRIVATE", "PVT", "INC", "CORP", 
    "CORPORATION", "LLC", "LLP", "PLC", "CO", "COMPANY"
}

# Prefixes that corrupt alias generation
STRIP_PREFIXES = ["THE ", "DR ", "DR. ", "SHREE ", "SHRI ", "MR ", "MRS "]

# Words that make any multi-word phrase instantly High-Risk
GENERIC_WORDS = {
    "GLOBAL", "SHANTI", "NATIONAL", "INTERNATIONAL", "OVERSEAS", 
    "ENTERPRISES", "INDUSTRIES", "TECHNOLOGIES", "SOLUTIONS", 
    "VENTURES", "CAPITAL", "HOLDINGS", "GROUP", "FINANCE", 
    "SERVICES", "FUTURE", "FIRST", "ADVANCED", "DEVELOPERS", 
    "INFRASTRUCTURE", "COMMERCIAL", "AGRO", "STEEL", 
    "BANK", "INDIA", "NEW", "SOUTH", "NORTH", "EAST", "WEST", "STATE"
}
NUCLEAR_BANNED_ALIASES = {
    "BANK", "INDIA", "NEW", "STATE", "ALL", "CAN", "RAJ", "FUTURE", "NATIONAL",
    "INTERNATIONAL", "GLOBAL", "CEMENT", "STEEL", "POWER", "ENERGY", "FINANCE",
    "CAPITAL", "TECH", "TECHNOLOGY", "INDUSTRY", "INDUSTRIES", "ENTERPRISES",
    "OVERSEAS", "MOTORS", "CORPORATION", "CORP", "GROUP", "HOLDINGS", "VENTURES",
    "SOLUTIONS", "SERVICES", "SOUTH", "NORTH", "EAST", "WEST", "WESTERN",
    "SOUTHERN", "NORTHERN", "EASTERN", "CENTRAL", "THE", "A", "AN", "DR", "MR", 
    "MRS", "LTD", "LIMITED", "INC"
}

def normalize(s):
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_base_name(name):
    s = normalize(name)
    # Strip dangerous prefixes first
    for prefix in STRIP_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            
    words = s.split()
    while words and words[-1] in REMOVE_SUFFIX:
        words.pop()
    return " ".join(words)

# --- Main Loop ---
df = pd.read_csv(INPUT)
high_confidence_list = []
high_risk_list = []

for _, row in df.iterrows():
    company = row["Company Name"]
    high_conf, high_risk = set(), set()
    
    official = normalize(company)
    base = clean_base_name(company)
    
    high_conf.add(official)
    if len(base) > 3:
        high_conf.add(base)
    
    words = base.split()
    
    # 2. Process Multi-word chunks
    if len(words) >= 2:
        two_word = " ".join(words[:2])
        # If it starts with a generic word OR contains "OF" / "AND" / "&"
        if words[0] in GENERIC_WORDS or words[1] in {"OF", "AND", "&"} or two_word in GENERIC_WORDS:
            high_risk.add(two_word)
        else:
            high_conf.add(two_word)
            
    if len(words) >= 3:
        three_word = " ".join(words[:3])
        if words[1] in {"OF", "AND", "&"} and words[0] in GENERIC_WORDS:
             high_risk.add(three_word)
        else:
             high_conf.add(three_word)
        
    # 3. Process Single Words
    if len(words) == 1:
        w = words[0]
        if w in GENERIC_WORDS or len(w) <= 3:
            high_risk.add(w)
        else:
            high_conf.add(w)
    elif len(words) > 1:
        first_word = words[0]
        if first_word in GENERIC_WORDS or len(first_word) <= 3:
            high_risk.add(first_word)

    # Clean duplicates (High Risk overrides High Conf if there's a conflict for safety)
    high_conf = high_conf - high_risk
    
    # Final cleanup
    high_conf = [a for a in high_conf if len(a) > 2 and a not in NUCLEAR_BANNED_ALIASES]
    high_risk = [a for a in high_risk if len(a) > 2 and a not in NUCLEAR_BANNED_ALIASES]
    
    high_confidence_list.append(json.dumps(list(high_conf)))
    high_risk_list.append(json.dumps(list(high_risk)))

df["High_Confidence_Aliases"] = high_confidence_list
df["High_Risk_Aliases"] = high_risk_list
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

df.to_csv(OUTPUT, index=False)
print("Master sheet successfully regenerated with strict Risk Classifications!")