from pathlib import Path

import oracledb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.formula.api as smf


OUTPUT_DIR = Path("phase3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ORACLE_USERNAME = "ora_ansh1304"
ORACLE_PASSWORD = "a27811678"

FIG_DPI = 300


def get_connection():
    """
    Connect to Oracle through an SSH tunnel running on localhost:1522.

    Start the tunnel in a separate terminal and keep it open:
    ssh -L 1522:dbhost.students.cs.ubc.ca:1522 ansh1304@remote.students.cs.ubc.ca
    """
    dsn = oracledb.makedsn("localhost", 1522, service_name="stu")
    return oracledb.connect(
        user=ORACLE_USERNAME,
        password=ORACLE_PASSWORD,
        dsn=dsn
    )


def run_query(connection, sql: str) -> pd.DataFrame:
    """
    Execute a SQL query using a direct Oracle connection and return a pandas DataFrame.
    """
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0].lower() for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved dataframe to {path}")


def save_text(text: str, filename: str) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text output to {path}")


def clean_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def map_primary_genre(genre_value: str) -> str:
    if not isinstance(genre_value, str):
        return "Other"

    genre_lower = genre_value.lower()

    if "horror" in genre_lower:
        return "Horror"
    if "action" in genre_lower:
        return "Action"
    if "comedy" in genre_lower:
        return "Comedy"

    return "Other"


def add_heatmap_labels(ax, corr_df: pd.DataFrame) -> None:
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            value = corr_df.iloc[i, j]
            label = "NaN" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black")


def format_axis_labels(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)


# RQ1

RQ1_SQL = """
SELECT
    im.imdb_title_id,
    REPLACE(im.title, '|', ' ') AS title,
    im.year,
    REPLACE(im.genre, '|', ' ') AS genre,
    NVL(r.num_recommendations, 0) AS num_recommendations,
    b.domestic_gross,
    b.foreign_gross,
    (b.domestic_gross + b.foreign_gross) AS total_revenue,
    (b.foreign_gross / NULLIF(b.domestic_gross + b.foreign_gross, 0)) AS foreign_share
FROM imdb_movies im
JOIN bom_gross b
    ON LOWER(TRIM(im.title)) = LOWER(TRIM(b.title))
   AND im.year = b.year
LEFT JOIN (
    SELECT
        imdb_title_id,
        COUNT(*) AS num_recommendations
    FROM reddit_mentions
    WHERE imdb_title_id IS NOT NULL
    GROUP BY imdb_title_id
) r
    ON im.imdb_title_id = r.imdb_title_id
WHERE
    im.genre LIKE '%Horror%' OR
    im.genre LIKE '%Action%' OR
    im.genre LIKE '%Comedy%'
ORDER BY num_recommendations DESC
"""


def analyze_rq1(connection) -> pd.DataFrame:
    """
    RQ1:
    How does Reddit Recommendation Frequency Relate to the Proportion of a Movie's
    Box Office Revenue Earned Internationally vs Domestically Among Horror, Action,
    and Comedy Movies?
    """
    df = run_query(connection, RQ1_SQL)

    df["genre"] = df["genre"].astype(str).str.strip()

    df = clean_numeric_columns(
        df,
        [
            "year",
            "num_recommendations",
            "domestic_gross",
            "foreign_gross",
            "total_revenue",
            "foreign_share",
        ],
    )

    df["genre_filtered"] = df["genre"].apply(map_primary_genre)
    df = df.dropna(subset=["num_recommendations", "foreign_share"])
    df = df[df["genre_filtered"].isin(["Horror", "Action", "Comedy"])]

    save_dataframe(df, "rq1_query_results.csv")

    plt.figure(figsize=(9, 6))

    genre_order = ["Horror", "Action", "Comedy"]
    plotted_any = False

    for genre in genre_order:
        subset = df[df["genre_filtered"] == genre]
        if not subset.empty:
            plt.scatter(
                subset["num_recommendations"],
                subset["foreign_share"],
                alpha=0.7,
                label=genre,
            )
            plotted_any = True

    ax = plt.gca()
    format_axis_labels(
    ax,
    "How Reddit Recommendation Frequency Relates to\nForeign and Fomestic Revenue Share",
    "Number of Reddit Recommendations",
    "Share of Total Revenue Earned Internationally\n(International share = 1 − Domestic Share)"
    )

    if plotted_any:
        plt.legend(title="Movie Genre", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rq1_scatter.png", dpi=FIG_DPI)
    plt.close()
    print(f"Saved plot to {OUTPUT_DIR / 'rq1_scatter.png'}")

    stats_lines = []

    if len(df) >= 2:
        overall_corr, overall_p = pearsonr(df["num_recommendations"], df["foreign_share"])
        stats_lines.append(
            f"Overall Pearson correlation: {overall_corr:.6f}, p-value: {overall_p:.6f}"
        )
    else:
        stats_lines.append("Overall Pearson correlation: not enough rows")

    for genre in genre_order:
        subset = df[df["genre_filtered"] == genre]
        if len(subset) >= 2:
            corr, pval = pearsonr(subset["num_recommendations"], subset["foreign_share"])
            stats_lines.append(
                f"{genre} Pearson correlation: {corr:.6f}, p-value: {pval:.6f}"
            )
        else:
            stats_lines.append(f"{genre} Pearson correlation: not enough rows")

    rq1_model_df = df.dropna(
        subset=["num_recommendations", "foreign_share", "genre_filtered"]
    ).copy()

    if len(rq1_model_df) >= 3:
        rq1_model = smf.ols(
            "foreign_share ~ num_recommendations * C(genre_filtered)",
            data=rq1_model_df,
        ).fit()
        stats_lines.append("\nRQ1 regression summary:\n")
        stats_lines.append(str(rq1_model.summary()))
    else:
        stats_lines.append("\nRQ1 regression summary:\nNot enough rows to fit the model.")

    save_text("\n".join(stats_lines), "rq1_stats_and_regression.txt")
    print("\n".join(stats_lines))

    return df


# RQ2

RQ2_SQL = """
SELECT
    i.avg_vote,
    (b.domestic_gross + b.foreign_gross) AS total_gross,
    COUNT(*) AS reddit_discussion_count
FROM imdb_movies i
JOIN reddit_mentions r
    ON i.imdb_title_id = r.imdb_title_id
JOIN bom_gross b
    ON LOWER(TRIM(i.title)) = LOWER(TRIM(b.title))
   AND i.year = b.year
GROUP BY
    i.title,
    i.imdb_title_id,
    b.domestic_gross,
    b.foreign_gross,
    i.avg_vote
"""


def analyze_rq2(connection) -> pd.DataFrame:
    """
    RQ2:
    How does the average IMDb vote on a movie relate to the number of Reddit
    recommendations versus the box office success?
    """
    df = run_query(connection, RQ2_SQL)

    df = clean_numeric_columns(df, ["avg_vote", "total_gross", "reddit_discussion_count"])
    df = df.dropna(subset=["avg_vote", "total_gross", "reddit_discussion_count"])

    save_dataframe(df, "rq2_query_results.csv")

    corr_df = df[["avg_vote", "reddit_discussion_count", "total_gross"]].corr()
    corr_df.to_csv(OUTPUT_DIR / "rq2_correlation_matrix.csv")
    print(f"Saved correlation matrix to {OUTPUT_DIR / 'rq2_correlation_matrix.csv'}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    im = ax.imshow(corr_df.values, aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation Coefficient", fontsize=11)

    pretty_labels = [
        "Average IMDb Rating",
        "Reddit Discussion Count",
        "Total Box Office Gross"
    ]

    ax.set_xticks(range(len(pretty_labels)))
    ax.set_yticks(range(len(pretty_labels)))
    ax.set_xticklabels(pretty_labels, rotation=25, ha="right")
    ax.set_yticklabels(pretty_labels)

    add_heatmap_labels(ax, corr_df)

    format_axis_labels(
        ax,
        "Correlation Between IMDb Rating,\nReddit Discussion, and Box Office Gross",
        "",
        ""
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rq2_correlation_heatmap.png", dpi=FIG_DPI)
    plt.close()
    print(f"Saved plot to {OUTPUT_DIR / 'rq2_correlation_heatmap.png'}")

    stats_lines = []
    stats_lines.append("Correlation matrix:\n")
    stats_lines.append(corr_df.to_string())
    stats_lines.append("\n")

    if len(df) >= 3:
        model_1 = smf.ols("total_gross ~ avg_vote", data=df).fit()
        stats_lines.append("Regression 1: total_gross ~ avg_vote\n")
        stats_lines.append(str(model_1.summary()))
        stats_lines.append("\n")

        model_2 = smf.ols("total_gross ~ reddit_discussion_count", data=df).fit()
        stats_lines.append("Regression 2: total_gross ~ reddit_discussion_count\n")
        stats_lines.append(str(model_2.summary()))
        stats_lines.append("\n")

        model_3 = smf.ols(
            "total_gross ~ avg_vote + reddit_discussion_count",
            data=df,
        ).fit()
        stats_lines.append("Regression 3: total_gross ~ avg_vote + reddit_discussion_count\n")
        stats_lines.append(str(model_3.summary()))
        stats_lines.append("\n")

        stats_lines.append(
            "R-squared comparison:\n"
            f"- total_gross ~ avg_vote: {model_1.rsquared:.6f}\n"
            f"- total_gross ~ reddit_discussion_count: {model_2.rsquared:.6f}\n"
            f"- total_gross ~ avg_vote + reddit_discussion_count: {model_3.rsquared:.6f}\n"
        )
    else:
        stats_lines.append("Not enough rows to fit regression models.")

    save_text("\n".join(stats_lines), "rq2_stats_and_regressions.txt")
    print("\n".join(stats_lines))

    return df


# RQ3

RQ3_SQL = """
SELECT
    duration_bins,
    genre_filtered,
    AVG(upvotes) AS avg_upvotes
FROM (
    SELECT
        r.upvotes,
        CASE
            WHEN i.duration >= 60 AND i.duration < 80 THEN '60-79'
            WHEN i.duration >= 80 AND i.duration < 100 THEN '80-99'
            WHEN i.duration >= 100 AND i.duration < 120 THEN '100-119'
            WHEN i.duration >= 120 AND i.duration < 140 THEN '120-139'
            WHEN i.duration >= 140 AND i.duration < 160 THEN '140-159'
            WHEN i.duration >= 160 AND i.duration < 180 THEN '160-179'
            WHEN i.duration >= 180 AND i.duration <= 200 THEN '180-200'
            ELSE 'N/A'
        END AS duration_bins,
        CASE
            WHEN i.genre LIKE '%Horror%' THEN 'Horror'
            WHEN i.genre LIKE '%Action%' THEN 'Action'
            WHEN i.genre LIKE '%Comedy%' THEN 'Comedy'
        END AS genre_filtered
    FROM imdb_movies i
    JOIN reddit_mentions r
        ON i.imdb_title_id = r.imdb_title_id
    WHERE
        i.genre LIKE '%Horror%' OR
        i.genre LIKE '%Action%' OR
        i.genre LIKE '%Comedy%'
)
GROUP BY duration_bins, genre_filtered
HAVING AVG(upvotes) IS NOT NULL
"""


def analyze_rq3(connection) -> pd.DataFrame:
    """
    RQ3:
    How does the average number of upvotes for Reddit recommendations and discussion
    posts vary by movie duration, and how does this relationship change based on genre?
    """
    df = run_query(connection, RQ3_SQL)

    df["avg_upvotes"] = pd.to_numeric(df["avg_upvotes"].astype(str).str.strip(), errors="coerce")
    df["duration_bins"] = df["duration_bins"].astype(str).str.strip()
    df["genre_filtered"] = df["genre_filtered"].astype(str).str.strip()

    df = df.dropna(subset=["duration_bins", "genre_filtered", "avg_upvotes"])

    duration_order = ["60-79", "80-99", "100-119", "120-139", "140-159", "160-179", "180-200"]
    df["duration_bins"] = pd.Categorical(df["duration_bins"], categories=duration_order, ordered=True)
    df = df.sort_values(["duration_bins", "genre_filtered"])

    save_dataframe(df, "rq3_query_results.csv")

    pivot_df = df.pivot(index="duration_bins", columns="genre_filtered", values="avg_upvotes")
    pivot_df = pivot_df.reindex(duration_order)
    pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce")

    if pivot_df.dropna(how="all").empty:
        print("RQ3 plot skipped: no numeric data to plot.")
    else:
        ax = pivot_df.plot(kind="bar", figsize=(10, 6), rot=0)
        format_axis_labels(
            ax,
            "Average Reddit Upvotes Across Movie Lengths and Genres",
            "Movie Duration Range (Minutes)",
            "Average Number of Reddit Upvotes"
        )
        ax.legend(title="Movie Genre", fontsize=10, title_fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rq3_grouped_bar_chart.png", dpi=FIG_DPI)
        plt.close()
        print(f"Saved plot to {OUTPUT_DIR / 'rq3_grouped_bar_chart.png'}")

    return df


def main():
    print("Connecting to Oracle through SSH tunnel...")
    connection = get_connection()
    print("Connected.\n")

    try:
        print("Running RQ1 analysis...")
        rq1_df = analyze_rq1(connection)
        print(f"RQ1 rows: {len(rq1_df)}\n")

        print("Running RQ2 analysis...")
        rq2_df = analyze_rq2(connection)
        print(f"RQ2 rows: {len(rq2_df)}\n")

        print("Running RQ3 analysis...")
        rq3_df = analyze_rq3(connection)
        print(f"RQ3 rows: {len(rq3_df)}\n")

        print("All outputs saved in:", OUTPUT_DIR.resolve())
    finally:
        connection.close()
        print("Oracle connection closed.")


if __name__ == "__main__":
    main()