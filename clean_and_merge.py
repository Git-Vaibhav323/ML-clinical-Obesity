import pandas as pd

def clean_columns(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        'ha1': 'height_cm',
        'ha2': 'weight_kg',
        'hhid': 'household_id',
        'clusterno': 'cluster_id',
        'lineno': 'line_number',
    })
    return df

def main():
    print('Loading structured CSVs...')
    bio = pd.read_csv('biomarker_structured.csv')
    wom = pd.read_csv('womans_structured.csv')
    hh = pd.read_csv('household_structured.csv')

    print('Cleaning columns...')
    bio = clean_columns(bio)
    wom = clean_columns(wom)
    hh = clean_columns(hh)

    # Merge on household_id, cluster_id, line_number (inner join to keep only matched records)
    print('Merging datasets...')
    merged = pd.merge(bio, wom, on=['household_id', 'cluster_id', 'line_number'], how='outer', suffixes=('_bio', '_wom'))
    merged = pd.merge(merged, hh, on=['household_id', 'cluster_id', 'line_number'], how='outer', suffixes=('', '_hh'))

    print('Dropping duplicates and irrelevant columns...')
    merged = merged.drop_duplicates()
    # Drop columns that are all NaN or irrelevant (keep only columns with at least one non-NaN value)
    merged = merged.dropna(axis=1, how='all')

    print('Final merged shape:', merged.shape)
    merged.to_csv('merged_cleaned.csv', index=False)
    print('Saved merged_cleaned.csv')

if __name__ == '__main__':
    main() 