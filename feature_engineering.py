import pandas as pd
import numpy as np

def main():
    print('Loading merged_cleaned.csv...')
    df = pd.read_csv('merged_cleaned.csv')

    # Unified height and weight: prefer bio, fallback to wom
    def get_first_valid(row, cols):
        for col in cols:
            if col in row and pd.notnull(row[col]):
                try:
                    val = float(row[col])
                    if not np.isnan(val):
                        return val
                except Exception:
                    continue
        return np.nan

    print('Creating unified height, weight, ADL, and organ dysfunction columns...')
    df['height_cm'] = df.apply(lambda row: get_first_valid(row, ['height_cm_bio', 'height_cm_wom']), axis=1)
    df['weight_kg'] = df.apply(lambda row: get_first_valid(row, ['weight_kg_bio', 'weight_kg_wom']), axis=1)
    df['adl_limitation_flag'] = df.apply(lambda row: get_first_valid(row, ['adl_limitation_flag_bio', 'adl_limitation_flag_wom']), axis=1)
    df['organ_dysfunction_flag'] = df.apply(lambda row: get_first_valid(row, ['organ_dysfunction_flag_bio', 'organ_dysfunction_flag_wom']), axis=1)

    print('Calculating BMI...')
    df['BMI'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2

    # Fill missing flags with 0
    df['adl_limitation_flag'] = df['adl_limitation_flag'].fillna(0).astype(int)
    df['organ_dysfunction_flag'] = df['organ_dysfunction_flag'].fillna(0).astype(int)

    print('Creating clinical_obesity_label...')
    def label_row(row):
        if pd.isna(row['BMI']):
            return np.nan
        if row['BMI'] >= 30 and (row['adl_limitation_flag'] == 1 or row['organ_dysfunction_flag'] == 1):
            return 2  # Clinical Obesity
        elif 25 <= row['BMI'] < 30 and row['adl_limitation_flag'] == 0 and row['organ_dysfunction_flag'] == 0:
            return 1  # Preclinical Obesity
        elif row['BMI'] < 25:
            return 0  # Normal
        else:
            return np.nan
    df['clinical_obesity_label'] = df.apply(label_row, axis=1)

    print('Label value counts:')
    print(df['clinical_obesity_label'].value_counts(dropna=False))

    df.to_csv('final_dataset.csv', index=False)
    print('Saved final_dataset.csv')

if __name__ == '__main__':
    main() 