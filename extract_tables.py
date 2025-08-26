import pdfplumber
import pandas as pd
import os

# Filenames
pdf_files = [
    ("NFHS-5Biomarker.pdf", "biomarker.csv"),
    ("NFHS-5Womans.pdf", "womans.csv"),
    ("NFHS-5Household.pdf", "household.csv")
]

def extract_tables_from_pdf(pdf_path, output_csv):
    print(f"Extracting tables from {pdf_path}...")
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table)
                    # Remove empty rows
                    df = df.dropna(how='all')
                    if not df.empty:
                        all_tables.append(df)
        if all_tables:
            # Concatenate all tables vertically
            full_df = pd.concat(all_tables, ignore_index=True)
            # Save to CSV
            full_df.to_csv(output_csv, index=False)
            print(f"Saved extracted tables to {output_csv}")
        else:
            print(f"No tables found in {pdf_path}.")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    for pdf, csv in pdf_files:
        if os.path.exists(pdf):
            extract_tables_from_pdf(pdf, csv)
        else:
            print(f"File not found: {pdf}") 
            