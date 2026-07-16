import ast
import json
import ahocorasick
import pandas as pd
from tqdm import tqdm

# -------------------------
COMPANY_CSV = "data/processed/companies_master.csv"
INPUT_JSONL = "data/news/raw/20140101_articles.jsonl"
OUTPUT_JSONL = "data/news/processed/20140101_mapped.jsonl"
# -------------------------

print("Loading company database...")

companies = pd.read_csv(COMPANY_CSV, encoding="latin1")

A = ahocorasick.Automaton()

for _, row in companies.iterrows():

    # Convert to string to handle any NaN/null values safely
    raw_aliases = str(row["Expanded Aliases"])
    
    # Find the positions of the first '[' and the last ']'
    start_idx = raw_aliases.find('[')
    end_idx = raw_aliases.rfind(']')
    
    # Check if we actually found a list
    if start_idx != -1 and end_idx != -1:
        clean_aliases = raw_aliases[start_idx:end_idx+1]
        
        try:
            aliases = ast.literal_eval(clean_aliases)
        except Exception as e:
            print(f"Warning: Skipping invalid alias format for {row.get('Symbol', 'Unknown')}: {clean_aliases}")
            continue
    else:
        # Skip this row if no brackets were found
        continue

    # Proceed with the rest of your logic
    for alias in aliases:
        alias = alias.upper().strip()

        if len(alias) < 3:
            continue

        A.add_word(
            alias,
            (
                row["Symbol"],
                row["Company Name"],
                row["Sector"],
                row["Industry"],
                alias,
            ),
        )

A.make_automaton()

print("Aliases loaded:", len(A))

print("Mapping articles...")

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in tqdm(fin):

        article = json.loads(line)

        text = (
            article.get("title", "") +
            "\n" +
            article.get("text", "")
        ).upper()

        symbols = set()
        companies_found = set()
        sectors = set()
        industries = set()
        aliases_found = set()

        for _, value in A.iter(text):

            symbol, company, sector, industry, alias = value

            symbols.add(symbol)
            companies_found.add(company)
            sectors.add(sector)
            industries.add(industry)
            aliases_found.add(alias)

        article["matched_symbols"] = sorted(symbols)
        article["matched_companies"] = sorted(companies_found)
        article["matched_sectors"] = sorted(sectors)
        article["matched_industries"] = sorted(industries)
        article["matched_aliases"] = sorted(aliases_found)

        fout.write(json.dumps(article, ensure_ascii=False) + "\n")

print("Done.")