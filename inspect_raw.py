import pandas as pd

FILES = {
    "IMDb_movies": "data_raw/IMDb_movies.csv",
    "IMDb_ratings": "data_raw/IMDb_ratings.csv",
    "BOM": "data_raw/BOM.csv",
    "Reddit": "data_raw/Reddit.csv",
}

def inspect(path, name, max_rows=None):
    print("=" * 80)
    print(f"{name} ({path})")
    df = pd.read_csv(path, nrows=max_rows)  

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Column names:")
    print(list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nCardinality (nunique) per column:")
    print(df.nunique(dropna=True))
    print("=" * 80)
    print()


if __name__ == "__main__":
    for name, path in FILES.items():
        inspect(path, name)
