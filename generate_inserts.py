import pandas as pd
import math

DATA_CLEAN_DIR = "data_clean"
OUTPUT_SQL = "project_load.sql"

def sql_escape(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    # treat everything else as string
    s = str(value).replace("'", "''")
    return f"'{s}'"

def write_schema(f):
    f.write("""
-- DROP and CREATE statements
DROP TABLE REDDIT_MENTIONS PURGE;
DROP TABLE BOM_GROSS PURGE;
DROP TABLE IMDB_MOVIES PURGE;

CREATE TABLE IMDB_MOVIES (
  imdb_title_id    VARCHAR2(15) PRIMARY KEY,
  title            VARCHAR2(400),
  year             NUMBER(4),
  genre            VARCHAR2(200),
  duration         NUMBER,
  avg_vote         NUMBER(3,1),
  votes            NUMBER,
  language         VARCHAR2(200)
);

CREATE TABLE BOM_GROSS (
  title           VARCHAR2(400),
  year            NUMBER(4),
  domestic_gross  NUMBER,
  foreign_gross   NUMBER
);

CREATE TABLE REDDIT_MENTIONS (
  turn_id         VARCHAR2(50),
  imdb_title_id   VARCHAR2(15),
  upvotes         NUMBER,
  is_seeker       CHAR(1),
  CONSTRAINT fk_reddit_imdb
    FOREIGN KEY (imdb_title_id) REFERENCES IMDB_MOVIES(imdb_title_id)
);
""")
    f.write("\n-- INSERT statements\n")

def generate_imdb_inserts(f):
    df = pd.read_csv(f"{DATA_CLEAN_DIR}/imdb_movies_clean.csv")
    for _, row in df.iterrows():
        values = [
            sql_escape(row["imdb_title_id"]),
            sql_escape(row["title"]),
            sql_escape(row["year"]),
            sql_escape(row["genre"]),
            sql_escape(row["duration"]),
            sql_escape(row["avg_vote"]),
            sql_escape(row["votes"]),
            sql_escape(row["language"]),
        ]
        f.write(
            "INSERT INTO IMDB_MOVIES "
            "(imdb_title_id, title, year, genre, duration, avg_vote, votes, language) "
            f"VALUES ({', '.join(values)});\n"
        )

def generate_bom_inserts(f):
    df = pd.read_csv(f"{DATA_CLEAN_DIR}/bom_gross_clean.csv")
    for _, row in df.iterrows():
        values = [
            sql_escape(row["title"]),
            sql_escape(row["year"]),
            sql_escape(row["domestic_gross"]),
            sql_escape(row["foreign_gross"]),
        ]
        f.write(
            "INSERT INTO BOM_GROSS "
            "(title, year, domestic_gross, foreign_gross) "
            f"VALUES ({', '.join(values)});\n"
        )

def generate_reddit_inserts(f):
    df = pd.read_csv(f"{DATA_CLEAN_DIR}/reddit_mentions_clean.csv")
    # If is_seeker is boolean, convert to 'Y'/'N' or '1'/'0'
    if df["is_seeker"].dtype == bool:
        df["is_seeker"] = df["is_seeker"].map({True: "Y", False: "N"})

    for _, row in df.iterrows():
        values = [
            sql_escape(row["turn_id"]),
            sql_escape(row["imdb_title_id"]),
            sql_escape(row["upvotes"]),
            sql_escape(row["is_seeker"]),
        ]
        f.write(
            "INSERT INTO REDDIT_MENTIONS "
            "(turn_id, imdb_title_id, upvotes, is_seeker) "
            f"VALUES ({', '.join(values)});\n"
        )

if __name__ == "__main__":
    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:
        write_schema(f)
        generate_imdb_inserts(f)
        generate_bom_inserts(f)
        generate_reddit_inserts(f)
        f.write("\nCOMMIT;\n")
    print(f"Wrote all statements to {OUTPUT_SQL}")
