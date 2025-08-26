import pandas as pd
import re
import numpy as np

# Expanded regex patterns for key fields
HEIGHT_PATTERNS = [
    r"height[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"ht[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"body height[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"(\d{2,3}(?:\.\d+)?)\s*cm[s]?",
    r"height.*is\s*(\d{2,3}(?:\.\d+)?)",
    r"height in centimeters[^\d]*(\d{2,3}(?:\.\d+)?)"
]
WEIGHT_PATTERNS = [
    r"weight[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"wt[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"body weight[^\d]*(\d{2,3}(?:\.\d+)?)",
    r"(\d{2,3}(?:\.\d+)?)\s*kg[s]?",
    r"weight.*is\s*(\d{2,3}(?:\.\d+)?)",
    r"weight in kilograms[^\d]*(\d{2,3}(?:\.\d+)?)"
]
AGE_PATTERNS = [
    r"age[^\d]*(\d{1,3})",
    r"(\d{1,3})\s*years?",
    r"(\d{1,3})\s*yrs?",
    r"respondent age[^\d]*(\d{1,3})"
]
SEX_PATTERNS = [
    r"sex[^a-zA-Z]*(male|female)",
    r"gender[^a-zA-Z]*(male|female)",
    r"\b(male|female)\b",
    r"\b(m|f)\b"
]
HOUSEHOLD_PATTERNS = [
    r"household[^\d]*(\d+)",
    r"hhid[^\d]*(\d+)",
    r"household number[^\d]*(\d+)"
]
CLUSTER_PATTERNS = [
    r"cluster[^\d]*(\d+)",
    r"clusterno[^\d]*(\d+)",
    r"cluster number[^\d]*(\d+)"
]
LINE_PATTERNS = [
    r"line number[^\d]*(\d+)",
    r"line no[^\d]*(\d+)"
]
REGION_PATTERNS = [
    r"state[^,\n]+",
    r"region[^,\n]+",
    r"district[^,\n]+",
    r"area[^,\n]+"
]

ADL_PATTERNS = [
    r"difficulty walking", r"difficulty dressing", r"adl limitation", r"mobility", r"self-care"
]
ORG_DYS_PATTERNS = [
    r"breathlessness", r"vision", r"fatigue", r"blood pressure", r"anaemia", r"organ dysfunction",
    r"chronic disease", r"hypertension", r"diabetes", r"heart", r"kidney"
]

# Block delimiters (add more as needed)
BLOCK_DELIMITERS = [
    r"NAME OF RESPONDENT", r"NEW RESPONDENT", r"HOUSEHOLD SCHEDULE", r"^$"
]

def search_patterns(patterns, text, group=1, flags=re.IGNORECASE):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            try:
                return m.group(group)
            except IndexError:
                continue
    return None

def split_blocks(lines):
    blocks = []
    block = []
    for line in lines:
        if any(re.search(delim, line, re.IGNORECASE) for delim in BLOCK_DELIMITERS) or line.strip() == '':
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line.strip())
    if block:
        blocks.append(block)
    return blocks

def is_invalid_value(val):
    try:
        v = float(val)
        return v in [0, 999, 9999, 99999]
    except Exception:
        return True

def parse_blocks(input_csv, output_csv, debug_blocks=10):
    with open(input_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    blocks = split_blocks(lines)
    records = []
    debug_out = []
    for i, block in enumerate(blocks):
        text = ' '.join(block)
        rec = dict()
        rec['height_cm'] = search_patterns(HEIGHT_PATTERNS, text)
        rec['weight_kg'] = search_patterns(WEIGHT_PATTERNS, text)
        rec['age'] = search_patterns(AGE_PATTERNS, text)
        sex = search_patterns(SEX_PATTERNS, text)
        if sex:
            rec['sex'] = 'F' if 'female' in sex.lower() or sex.lower() == 'f' else 'M'
        else:
            rec['sex'] = None
        rec['household_id'] = search_patterns(HOUSEHOLD_PATTERNS, text)
        rec['cluster_id'] = search_patterns(CLUSTER_PATTERNS, text)
        rec['line_number'] = search_patterns(LINE_PATTERNS, text)
        region = search_patterns(REGION_PATTERNS, text)
        rec['region'] = region.strip().replace('state','').replace('region','').replace('district','').replace('area','').strip() if region else None
        rec['adl_limitation_flag'] = int(any(re.search(p, text, re.IGNORECASE) for p in ADL_PATTERNS))
        rec['organ_dysfunction_flag'] = int(any(re.search(p, text, re.IGNORECASE) for p in ORG_DYS_PATTERNS))
        # Only keep if both height and weight are present and valid
        if rec['height_cm'] and rec['weight_kg'] and not is_invalid_value(rec['height_cm']) and not is_invalid_value(rec['weight_kg']):
            try:
                rec['height_cm'] = float(rec['height_cm'])
                rec['weight_kg'] = float(rec['weight_kg'])
            except Exception:
                continue
            # Convert numeric columns
            for col in ['age', 'adl_limitation_flag', 'organ_dysfunction_flag']:
                if col in rec and rec[col] is not None:
                    try:
                        rec[col] = float(rec[col])
                    except Exception:
                        rec[col] = np.nan
            records.append(rec)
            if len(debug_out) < debug_blocks:
                debug_out.append({'block': block, 'fields': rec})
    df = pd.DataFrame(records)
    print(f'Extracted {len(df)} valid records.')
    print('First few valid samples:')
    print(df.head(debug_blocks))
    df.to_csv(output_csv, index=False)
    print(f'Saved structured data to {output_csv}')

def extract_household_table(input_csv, output_csv, debug_rows=5):
    with open(input_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r'LINE\s*NO', line, re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        print('Could not find LINE NO. header in household.csv')
        return
    # Get header columns
    header_line = lines[header_idx].strip().replace('"', '')
    columns = [c.strip().lower().replace(' ', '_') for c in header_line.split(',')]
    # Find data rows (start with two digits and a comma)
    data_rows = []
    for line in lines[header_idx+1:]:
        if re.match(r'^[0-9]{2},', line):
            data_rows.append(line.strip())
        elif data_rows and not line.strip():
            break  # Stop at first empty line after data rows
    # Parse data rows
    records = []
    for row in data_rows:
        values = [v.strip() for v in row.split(',')]
        rec = dict(zip(columns, values))
        records.append(rec)
    df = pd.DataFrame(records)
    # Add placeholder columns for merging
    df['household_id'] = 1
    df['cluster_id'] = 1
    # Rename line_no._of to line_number if present
    for col in df.columns:
        if col.startswith('line_no'):
            df = df.rename(columns={col: 'line_number'})
    print(f'Extracted {len(df)} household records.')
    print('First few rows:')
    print(df.head(debug_rows))
    df.to_csv(output_csv, index=False)
    print(f'Saved structured data to {output_csv}')

if __name__ == "__main__":
    parse_blocks('biomarker.csv', 'biomarker_structured.csv')
    parse_blocks('womans.csv', 'womans_structured.csv')
    parse_blocks('household.csv', 'household_structured.csv')
    extract_household_table('household.csv', 'household_structured.csv') 