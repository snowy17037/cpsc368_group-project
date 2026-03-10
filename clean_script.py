import os
import re
import pandas as pd

DATA_RAW_DIR = "data_raw"
DATA_CLEAN_DIR = "data_clean"

os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

# Helpers
def clean_title(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)          # collapse spaces
    s = s.str.replace(r"[^\w\s]", "", regex=True)       # remove punctuation
    return s

def drop_mostly_null(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Keep only columns where the fraction of missing values <= threshold.
    This is allowed as 'filtering columns' in the project rules.
    """
    keep_cols = df.columns[df.isna().mean() <= threshold]
    return df[keep_cols]

# 1) Clean IMDb_movies -> imdb_movies_clean.csv
def clean_imdb_movies():
    path = os.path.join(DATA_RAW_DIR, "IMDb_movies.csv")
    df = pd.read_csv(path)

    df = df[
        [
            "imdb_title_id",
            "title",
            "year",
            "genre",
            "duration",
            "avg_vote",
            "votes",
            "language",
        ]
    ]

    df = drop_mostly_null(df, threshold=0.5)

    # Filtering by: English, 2010–2018, selected genres
    df = df[df["language"].astype(str).str.contains("English", na=False)]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2010, 2018, inclusive="both")]

    genre_mask = df["genre"].astype(str).str.contains(
        r"\b(horror|action|comedy)\b", case=False, regex=True
    )
    df = df[genre_mask]

    # Dropping rows with missing key fields
    for col in ["imdb_title_id", "title", "year", "duration", "avg_vote", "votes"]:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Standardizing title + year
    if "title" in df.columns:
        df["title"] = clean_title(df["title"])
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    
    df = df.sort_values(["imdb_title_id"]).drop_duplicates(subset=["imdb_title_id"], keep="first")

    out_path = os.path.join(DATA_CLEAN_DIR, "imdb_movies_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {df.shape}")

# 2) Clean Reddit -> reddit_mentions_clean.csv

IMDB_ID_PATTERN = re.compile(r"tt\d{7,8}")

def extract_imdb_ids(text: str):
    if not isinstance(text, str):
        return []
    return IMDB_ID_PATTERN.findall(text)

def clean_reddit():
    path = os.path.join(DATA_RAW_DIR, "Reddit.csv")
    df = pd.read_csv(path)

    df = df[["turn_id", "upvotes", "is_seeker", "processed"]]

    df = drop_mostly_null(df, threshold=0.5)

    # Extracting IMDb IDs and creating one row per (turn_id, imdb_title_id)
    rows = []
    for _, row in df.iterrows():
        ids = extract_imdb_ids(row.get("processed", ""))
        for imdb_id in ids:
            rows.append(
                {
                    "turn_id": row.get("turn_id"),
                    "imdb_title_id": imdb_id.lower().strip(),
                    "upvotes": row.get("upvotes"),
                    "is_seeker": row.get("is_seeker"),
                }
            )

    mentions = pd.DataFrame(rows)

    mentions = (
        mentions
        .sort_values(["turn_id", "imdb_title_id"])
        .drop_duplicates(subset=["turn_id", "imdb_title_id"], keep="first")
    )


    out_path = os.path.join(DATA_CLEAN_DIR, "reddit_mentions_clean.csv")
    mentions.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {mentions.shape}")


# 3) Clean BOM -> bom_gross_clean.csv

def clean_bom():
    path = os.path.join(DATA_RAW_DIR, "BOM.csv")
    df = pd.read_csv(path)

    # core columns
    df = df[["movie", "year", "domestic_gross", "foreign_gross"]]

    #columns that are mostly null
    df = drop_mostly_null(df, threshold=0.5)

    # Rename and standardize title
    df = df.rename(columns={"movie": "title"})
    df["title"] = clean_title(df["title"])

    # numeric year and filter to BOM range
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2010, 2018, inclusive="both")]
    df["year"] = df["year"].astype(int)

    df["domestic_gross"] = pd.to_numeric(df["domestic_gross"], errors="coerce")
    df["foreign_gross"] = pd.to_numeric(df["foreign_gross"], errors="coerce")

    df = df.dropna(subset=["title", "year"])
    df = df.dropna(subset=["domestic_gross", "foreign_gross"], how="all")

    df = df[(df["domestic_gross"] >= 0) | (df["foreign_gross"] >= 0)]

    df = (
        df
        .sort_values(["title", "year"])
        .drop_duplicates(subset=["title", "year"], keep="first")
    )


    out_path = os.path.join(DATA_CLEAN_DIR, "bom_gross_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {df.shape}")

if __name__ == "__main__":
    clean_imdb_movies()
    clean_reddit()
    clean_bom()
