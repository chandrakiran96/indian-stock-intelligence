import pandas as pd
import re
import json

INPUT = "data/processed/companies_master.csv"
OUTPUT = "data/processed/companies_master_enriched.csv"

# Load your master file
df = pd.read_csv(INPUT)

# Comprehensive list of corporate noise
REMOVE_SUFFIX = {
    "LIMITED", "LTD", "PRIVATE", "PVT", "INC", "CORP", 
    "CORPORATION", "LLC", "LLP", "PLC", "CO", "COMPANY", "INDIA"
}

# Words that appear in hundreds of company names and cause chaos if matched alone
GENERIC_WORDS = {
    "GLOBAL", "SHANTI", "NATIONAL", "INTERNATIONAL", "OVERSEAS", 
    "ENTERPRISES", "INDUSTRIES", "TECHNOLOGIES", "SOLUTIONS", 
    "VENTURES", "CAPITAL", "HOLDINGS", "GROUP", "FINANCE", 
    "SERVICES", "FUTURE", "FIRST", "ADVANCED", "INDUSTRIES", 
    "DEVELOPERS", "INFRASTRUCTURE", "COMMERCIAL", "AGRO", "STEEL"
}

def normalize(s):
    s = str(s).upper().strip()
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_base_name(name):
    words = normalize(name).split()
    while words and words[-1] in REMOVE_SUFFIX:
        words.pop()
    return " ".join(words)

high_confidence_list = []
high_risk_list = []

for _, row in df.iterrows():
    company = row["Company Name"]
    
    high_conf = set()
    high_risk = set()
    
    official = normalize(company)
    base = clean_base_name(company)
    
    # 1. Full names are always High Confidence
    high_conf.add(official)
    high_conf.add(base)
    
    words = base.split()
    
    # 2. Process Multi-word chunks
    if len(words) >= 2:
        two_word = " ".join(words[:2])
        if words[0] in GENERIC_WORDS or two_word in GENERIC_WORDS:
            high_risk.add(two_word)
        else:
            high_conf.add(two_word)
            
    if len(words) >= 3:
        high_conf.add(" ".join(words[:3]))
        
    # 3. Process Single Words safely
    if len(words) == 1:
        w = words[0]
        if w in GENERIC_WORDS or len(w) <= 3:
            high_risk.add(w)
        else:
            high_conf.add(w)
    elif len(words) > 1:
        first_word = words[0]
        if first_word in GENERIC_WORDS:
            high_risk.add(first_word)

    # Clean duplicates between groups (High Confidence wins)
    high_risk = high_risk - high_conf
    
    # Final cleanup (lengths, empty items)
    high_conf = [a for a in high_conf if len(a) > 2]
    high_risk = [a for a in high_risk if len(a) > 2]
    
    high_confidence_list.append(json.dumps(list(high_conf)))
    high_risk_list.append(json.dumps(list(high_risk)))

# Add the new targeted columns to your master sheet
df["High_Confidence_Aliases"] = high_confidence_list
df["High_Risk_Aliases"] = high_risk_list

# Keep Symbol, BSE Code, Sector, Industry explicitly clean
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

df.to_csv(OUTPUT, index=False)
print("Master sheet successfully generated with Risk Classifications!")