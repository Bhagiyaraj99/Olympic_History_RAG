import pandas as pd
import os
import re
 
DATA_DIR = "data/"
 
def clean_text(text):
    """Lowercase, strip extra whitespace, remove special characters."""
    if not isinstance(text, str):
        return str(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s,.\'-]', '', text)   # keep letters, numbers, basic punctuation
    text = re.sub(r'\s+', ' ', text)            # collapse multiple spaces
    return text
 
def preprocess_csv(filepath):
    filename = os.path.basename(filepath)
    print(f"\n── {filename} ──")
 
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  Rows before: {len(df)}")
 
    # Drop fully duplicate rows
    df = df.drop_duplicates()
 
    # Fill NaN safely without inplace on slices
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("unknown")
 
    # Clean all string columns
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(clean_text)
 
    print(f"  Rows after:  {len(df)}")
 
    # Save cleaned file back
    out_path = filepath.replace(".csv", "_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")
 
def preprocess_txt(filepath):
    filename = os.path.basename(filepath)
    print(f"\n── {filename} ──")
 
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
 
    print(f"  Characters before: {len(text)}")
 
    # Normalize whitespace and remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ASCII
    text = re.sub(r'\n{3,}', '\n\n', text)         # collapse excessive newlines
    text = re.sub(r' {2,}', ' ', text)             # collapse extra spaces
    text = text.strip()
 
    print(f"  Characters after:  {len(text)}")
 
    out_path = filepath.replace(".txt", "_clean.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {out_path}")
 
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found.")
        exit(1)
 
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
 
        # Skip already-cleaned files
        if "_clean" in fname:
            continue
 
        if fname.endswith(".csv"):
            preprocess_csv(fpath)
        elif fname.endswith(".txt"):
            preprocess_txt(fpath)
 
    print("\n Preprocessing complete!")
    print("Update your app.py to load '_clean' files from the data/ folder.")