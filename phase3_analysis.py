from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

# Configuration

OUTPUT_DIR = Path("phase3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RQ1_CSV = Path("rq1.csv")
RQ2_CSV = Path("rq2.csv")
RQ3_CSV = Path("rq3.csv")

FIG_DPI = 300

# Helpers

def load_query_csv(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    """
    Load a query result CSV where:
    - first row contains column names
    - second row is a separator row (dashes/underscores)
    - actual data starts on row 3
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path, skiprows=[1])

    if df.shape[1] != len(expected_columns):
        raise ValueError(
            f"{path.name} has {df.shape[1]} columns, but {len(expected_columns)} were expected."
        )

    df.columns = expected_columns
    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved dataframe to {output_path}")


def save_text(text: str, filename: str) -> None:
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved text output to {output_path}")


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

def analyze_rq1() -> pd.DataFrame:
    df = load_query_csv(
        RQ1_CSV,
        [
            "imdb_title_id",
            "title",
            "year",
            "genre",
            "num_recommendations",
            "domestic_gross",
            "foreign_gross",
            "total_revenue",
            "foreign_share",
        ],
    )

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

    save_dataframe(df, "rq1_query_results_cleaned.csv")

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
        "How Reddit Recommendation Frequency Relates to Foreign Revenue Share",
        "Number of Reddit Recommendations",
        "Share of Total Revenue Earned Internationally",
    )
    ax.set_ylim(0, 1)

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
    return df

# RQ2

def analyze_rq2() -> pd.DataFrame:
    df = load_query_csv(
        RQ2_CSV,
        [
            "avg_vote",
            "total_gross",
            "reddit_discussion_count",
        ],
    )

    df = clean_numeric_columns(df, ["avg_vote", "total_gross", "reddit_discussion_count"])
    df = df.dropna(subset=["avg_vote", "total_gross", "reddit_discussion_count"])

    save_dataframe(df, "rq2_query_results_cleaned.csv")

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
        "Total Box Office Gross",
    ]

    ax.set_xticks(range(len(pretty_labels)))
    ax.set_yticks(range(len(pretty_labels)))
    ax.set_xticklabels(pretty_labels, rotation=25, ha="right")
    ax.set_yticklabels(pretty_labels)

    add_heatmap_labels(ax, corr_df)

    format_axis_labels(
        ax,
        "Correlation Between IMDb Rating, Reddit Discussion, and Total Box Office Gross",
        "Variables Compared",
        "Variables Compared",
    )

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
    return df

# RQ3

def analyze_rq3() -> pd.DataFrame:
    df = load_query_csv(
        RQ3_CSV,
        [
            "duration_bins",
            "genre_filtered",
            "avg_upvotes",
        ],
    )

    df["duration_bins"] = df["duration_bins"].astype(str).str.strip()
    df["genre_filtered"] = df["genre_filtered"].astype(str).str.strip()
    df["avg_upvotes"] = pd.to_numeric(df["avg_upvotes"], errors="coerce")

    df = df.dropna(subset=["duration_bins", "genre_filtered", "avg_upvotes"])

    duration_order = ["60-79", "80-99", "100-119", "120-139", "140-159", "160-179", "180-200"]
    df["duration_bins"] = pd.Categorical(df["duration_bins"], categories=duration_order, ordered=True)
    df = df.sort_values(["duration_bins", "genre_filtered"])

    save_dataframe(df, "rq3_query_results_cleaned.csv")

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
            "Average Number of Reddit Upvotes",
        )
        ax.legend(title="Movie Genre", fontsize=10, title_fontsize=11)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rq3_grouped_bar_chart.png", dpi=FIG_DPI)
        plt.close()
        print(f"Saved plot to {OUTPUT_DIR / 'rq3_grouped_bar_chart.png'}")

    return df


# Main

def main():
    print("Running RQ1 analysis from rq1.csv...")
    rq1_df = analyze_rq1()
    print(f"RQ1 rows: {len(rq1_df)}\n")

    print("Running RQ2 analysis from rq2.csv...")
    rq2_df = analyze_rq2()
    print(f"RQ2 rows: {len(rq2_df)}\n")

    print("Running RQ3 analysis from rq3.csv...")
    rq3_df = analyze_rq3()
    print(f"RQ3 rows: {len(rq3_df)}\n")

    print("All outputs saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()