import pandas as pd

df = pd.read_csv(
    "data/network_traffic.csv",
    nrows=10000,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

if "Label" in df.columns:
    print("\nLabels:")
    print(df["Label"].value_counts())