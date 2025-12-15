import pandas as pd

path = "data/CICIDS2017.csv"

df = pd.read_csv(path)

print("Original columns found:")
print(df.columns.tolist())

# Strip leading/trailing spaces from ALL column names
df.columns = [c.strip() for c in df.columns]

# Ensure label column is exactly 'Label'
if "Label" not in df.columns:
    raise Exception("ERROR: No proper Label column found even after strip().")

print("\nCleaned columns:")
print(df.columns.tolist())

df.to_csv(path, index=False)
print("\nFixed and saved cleaned dataset:", path)