"""Beginner-friendly Netflix clustering pipeline.

Run from the project root with:
    py python/pipeline.py

The script intentionally keeps the analysis steps visible and prints the
requested inspection commands/results for a college lab record.
"""

from collections import Counter
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "netflix_titles.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "server" / "data"
PLOTS_DIR = PROJECT_ROOT / "python" / "plots"


def split_values(value):
    """Turn a comma-separated cell into clean values."""
    if pd.isna(value) or str(value).strip().lower() in {"", "unknown"}:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def normalise_multi_value(value):
    """Use readable title case for country and genre display values."""
    values = split_values(value)
    return ", ".join(part.title() for part in values) if values else "Unknown"


def multivalue_counts(series):
    counts = Counter()
    for value in series:
        counts.update(split_values(value))
    return counts


def load_and_prepare(csv_path=DEFAULT_INPUT):
    """Load and clean the source CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Put Kaggle's netflix_titles.csv there."
        )

    df = pd.read_csv(csv_path)
    required_columns = {
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    print("\n--- First five rows: df.head() ---")
    print(df.head())
    print("\n--- Last five rows: df.tail() ---")
    print(df.tail())
    print("\n--- Data information: df.info() ---")
    df.info()
    print("\n--- Numeric summary: df.describe() ---")
    print(df.describe())
    print(f"\nOriginal shape: {df.shape}")
    print("\nNull values before cleaning: df.isnull().sum()")
    print(df.isnull().sum())

    # Empty strings are blanks too, so turn them into NaN before dropna().
    for column in ["date_added", "rating", "duration"]:
        df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    # A few source rows incorrectly store values such as "74 min" in rating.
    # Move that value to duration and use Unknown for the missing rating.
    rating_text = df["rating"].astype("string").str.strip()
    malformed_rating = rating_text.str.match(r"^\d+\s*min$", na=False)
    duration_is_blank = df["duration"].isna()
    df.loc[malformed_rating & duration_is_blank, "duration"] = rating_text[
        malformed_rating & duration_is_blank
    ]
    df.loc[malformed_rating, "rating"] = "Unknown"

    before_drop = len(df)
    df = df.dropna(subset=["date_added", "rating", "duration"]).copy()
    print(f"\nRows removed by dropna(date_added, rating, duration): {before_drop - len(df)}")

    # Fill the requested descriptive columns before text normalisation.
    for column in ["director", "cast", "country"]:
        df[column] = df[column].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    df["description"] = df["description"].fillna("Unknown")

    # Lowercase/strip first, then replace common values with display labels.
    df["type"] = (
        df["type"].astype("string").str.lower().str.strip().replace(
            {"movie": "Movie", "tv show": "TV Show", "tvshow": "TV Show"}
        )
    )
    rating_replacements = {
        "tv-ma": "TV-MA",
        "tv-14": "TV-14",
        "tv-pg": "TV-PG",
        "tv-g": "TV-G",
        "tv-y": "TV-Y",
        "tv-y7": "TV-Y7",
        "pg-13": "PG-13",
        "not rated": "NR",
        "nr": "NR",
        "unknown": "Unknown",
    }
    df["rating"] = (
        df["rating"].astype("string").str.lower().str.strip().replace(rating_replacements)
    )
    df["rating"] = df["rating"].fillna("Unknown")
    df["country"] = df["country"].astype("string").str.lower().str.strip().replace(
        {"unknown": "Unknown"}
    )
    df["country"] = df["country"].map(normalise_multi_value)
    df["listed_in"] = df["listed_in"].map(normalise_multi_value)

    # Convert dates and extract the year added.
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["date_added"]).copy()
    df["year_added"] = df["date_added"].dt.year.astype(int)

    # Extract the number from values such as "112 min" and "2 Seasons".
    df["duration_num"] = pd.to_numeric(
        df["duration"].astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    )
    df = df.dropna(subset=["duration_num"]).copy()
    df["duration_num"] = df["duration_num"].astype(int)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype(int)
    df["genre_count"] = df["listed_in"].map(lambda value: len(split_values(value))).astype(int)

    # Remove exact duplicate rows and report the count as requested.
    duplicate_count = int(df.duplicated().sum())
    print(f"\ndf.duplicated().sum(): {duplicate_count}")
    df = df.drop_duplicates().copy()

    # Categories save memory and make the categorical nature of these columns clear.
    df["type"] = df["type"].astype("category")
    df["rating"] = df["rating"].astype("category")

    print("\nNull values after cleaning: df.isnull().sum()")
    print(df.isnull().sum())
    print(f"Clean shape: {df.shape}")
    return df


def add_features(df):
    """Create top-10 genre flags and the scaled feature matrix."""
    genre_counts = multivalue_counts(df["listed_in"])
    top_genres = [genre for genre, _ in genre_counts.most_common(10)]
    genre_flag_columns = []
    row_genres = df["listed_in"].map(set)

    for genre in top_genres:
        slug = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
        column = f"genre_{slug}"
        genre_flag_columns.append(column)
        df[column] = row_genres.map(lambda values, item=genre: int(item in values))

    base_columns = ["release_year", "duration_num", "genre_count"]
    feature_columns = base_columns + genre_flag_columns

    # Outlier checks are reported only; no rows are removed for being outliers.
    q1 = df["duration_num"].quantile(0.25)
    q3 = df["duration_num"].quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = int(
        ((df["duration_num"] < q1 - 1.5 * iqr) | (df["duration_num"] > q3 + 1.5 * iqr)).sum()
    )
    z_scores = np.abs(stats.zscore(df["duration_num"], nan_policy="omit"))
    z_outliers = int((z_scores > 3).sum())
    print(f"\nDuration IQR: {iqr:.2f}; IQR outliers reported (not removed): {iqr_outliers}")
    print(f"Duration Z-score outliers reported (|z| > 3, not removed): {z_outliers}")

    standard_scaler = StandardScaler()
    standard_values = standard_scaler.fit_transform(df[feature_columns])
    standard_preview = pd.DataFrame(standard_values, columns=feature_columns, index=df.index)
    print("\nStandardScaler output: .head()")
    print(standard_preview.head())

    minmax_scaler = MinMaxScaler()
    minmax_values = minmax_scaler.fit_transform(df[feature_columns])
    minmax_preview = pd.DataFrame(minmax_values, columns=feature_columns, index=df.index)
    print("\nMinMaxScaler output: .head()")
    print(minmax_preview.head())

    df["features"] = standard_values.round(6).tolist()
    return df, {
        "top_genres": top_genres,
        "genre_counts": genre_counts,
        "base_columns": base_columns,
        "feature_columns": feature_columns,
        "standard_values": standard_values,
    }


def save_figure(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_eda_plots(df):
    """Create and save the requested exploratory charts."""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="type", ax=ax)
    ax.set_title("Content Type")
    save_figure(fig, "countplot_type.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    rating_order = df["rating"].value_counts().head(15).index
    sns.countplot(data=df, x="rating", order=rating_order, ax=ax)
    ax.set_title("Content Rating")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "countplot_rating.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="release_year", bins=15, kde=False, ax=ax)
    ax.set_title("Release Year")
    save_figure(fig, "histogram_release_year.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    movie_durations = df.loc[df["type"].astype(str) == "Movie", "duration_num"]
    sns.histplot(movie_durations, bins=15, kde=False, ax=ax)
    ax.set_title("Movie Duration (minutes)")
    ax.set_xlabel("duration_num")
    save_figure(fig, "histogram_movie_duration.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    year_counts = df["year_added"].value_counts().sort_index()
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax)
    ax.set_title("Titles Added per Year")
    ax.set_xlabel("Year added")
    ax.set_ylabel("Titles")
    save_figure(fig, "titles_added_per_year.png")

    country_counts = multivalue_counts(df["country"])
    fig, ax = plt.subplots(figsize=(8, 5))
    country_series = pd.Series(dict(country_counts.most_common(10))).sort_values()
    country_series.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Countries")
    ax.set_xlabel("Titles")
    save_figure(fig, "top_10_countries.png")

    genre_counts = multivalue_counts(df["listed_in"])
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_series = pd.Series(dict(genre_counts.most_common(10))).sort_values()
    genre_series.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Genres")
    ax.set_xlabel("Titles")
    save_figure(fig, "top_10_genres.png")

    correlation_columns = ["release_year", "duration_num", "genre_count", "year_added"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[correlation_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_figure(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="type", y="duration_num", ax=ax)
    ax.set_title("Duration by Type")
    save_figure(fig, "boxplot_duration_by_type.png")


def make_cluster_profiles(df):
    """Return a readable profile row for every cluster."""
    rows = []
    for cluster_number in sorted(df["cluster"].unique()):
        group = df[df["cluster"] == cluster_number]
        type_split = {str(key): int(value) for key, value in group["type"].value_counts().items()}
        top_genres = [genre for genre, _ in multivalue_counts(group["listed_in"]).most_common(3)]
        rows.append(
            {
                "cluster": int(cluster_number),
                "size": int(len(group)),
                "avg_release_year": round(float(group["release_year"].mean()), 2),
                "avg_duration_num": round(float(group["duration_num"].mean()), 2),
                "avg_genre_count": round(float(group["genre_count"].mean()), 2),
                "top_genres": top_genres,
                "type_split": type_split,
                "sample_titles": group["title"].head(3).tolist(),
            }
        )
    return rows


def run_clustering(df, feature_info):
    """Fit KMeans, choose the default K=5, and save cluster charts."""
    features = feature_info["standard_values"]
    possible_k = range(2, min(10, len(df) - 1) + 1)
    inertia_values = []
    silhouette_values = []

    for k in possible_k:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        inertia_values.append({"k": int(k), "inertia": round(float(model.inertia_), 4)})
        silhouette_values.append(
            {"k": int(k), "score": round(float(silhouette_score(features, labels)), 4)}
        )

    chosen_k = 5 if len(df) > 5 else max(2, len(df) - 1)
    model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(features).astype(int)
    chosen_silhouette = float(silhouette_score(features, df["cluster"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([item["k"] for item in inertia_values], [item["inertia"] for item in inertia_values], marker="o")
    ax.set_title("KMeans Elbow Plot")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    save_figure(fig, "elbow_plot.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([item["k"] for item in silhouette_values], [item["score"] for item in silhouette_values], marker="o")
    ax.axvline(chosen_k, color="red", linestyle="--", label=f"Chosen K={chosen_k}")
    ax.set_title("Silhouette Scores")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    save_figure(fig, "silhouette_scores.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    cluster_counts = df["cluster"].value_counts().sort_index()
    cluster_counts.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Titles")
    save_figure(fig, "cluster_size_bar.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="release_year", y="duration_num", hue="cluster", palette="tab10", s=70, ax=ax)
    ax.set_title("Release Year vs Duration by Cluster")
    save_figure(fig, "cluster_scatter.png")

    profile_rows = make_cluster_profiles(df)
    profile_display = pd.DataFrame(profile_rows)
    print("\n--- Cluster profile table ---")
    print(profile_display.to_string(index=False))
    print("\n--- Short cluster interpretations ---")
    for row in profile_rows:
        dominant_type = max(row["type_split"], key=row["type_split"].get)
        print(
            f"Cluster {row['cluster']}: {row['size']} titles, mostly {dominant_type}; "
            f"average release year {row['avg_release_year']}, average duration {row['avg_duration_num']}; "
            f"top genres: {', '.join(row['top_genres']) or 'Unknown'}."
        )

    return {
        "chosen_k": int(chosen_k),
        "chosen_silhouette": chosen_silhouette,
        "inertia_values": inertia_values,
        "silhouette_values": silhouette_values,
        "cluster_profiles": profile_rows,
    }


def build_summary(df, cluster_info):
    country_counts = multivalue_counts(df["country"])
    genre_counts = multivalue_counts(df["listed_in"])
    return {
        "counts_by_type": {str(key): int(value) for key, value in df["type"].value_counts().items()},
        "titles_added_per_year": {
            str(int(key)): int(value) for key, value in df["year_added"].value_counts().sort_index().items()
        },
        "top_10_genres": [
            {"genre": genre, "count": int(count)} for genre, count in genre_counts.most_common(10)
        ],
        "top_10_countries": [
            {"country": country, "count": int(count)} for country, count in country_counts.most_common(10)
        ],
        "cluster_summary": cluster_info["cluster_profiles"],
        "chosen_k": cluster_info["chosen_k"],
        "silhouette_score": round(cluster_info["chosen_silhouette"], 4),
        "elbow_inertia_values": cluster_info["inertia_values"],
        "silhouette_values": cluster_info["silhouette_values"],
    }


def write_outputs(df, summary, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "show_id",
        "title",
        "type",
        "release_year",
        "rating",
        "listed_in",
        "description",
        "country",
        "duration_num",
        "genre_count",
        "year_added",
        "cluster",
        "features",
    ]
    records = []
    for _, row in df[output_columns].iterrows():
        records.append(
            {
                "show_id": str(row["show_id"]),
                "title": str(row["title"]),
                "type": str(row["type"]),
                "release_year": int(row["release_year"]),
                "rating": str(row["rating"]),
                "listed_in": str(row["listed_in"]),
                "description": str(row["description"]),
                "country": str(row["country"]),
                "duration_num": int(row["duration_num"]),
                "genre_count": int(row["genre_count"]),
                "year_added": int(row["year_added"]),
                "cluster": int(row["cluster"]),
                "features": [round(float(value), 6) for value in row["features"]],
            }
        )

    clustered_path = output_dir / "netflix_clustered.json"
    summary_path = output_dir / "summary.json"
    clustered_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {clustered_path}")
    print(f"Wrote {summary_path}")


def run_pipeline(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    df = load_and_prepare(input_path)
    df, feature_info = add_features(df)
    save_eda_plots(df)
    cluster_info = run_clustering(df, feature_info)
    summary = build_summary(df, cluster_info)
    write_outputs(df, summary, output_dir)
    print(f"\nChosen K: {summary['chosen_k']}")
    print(f"Silhouette score: {summary['silhouette_score']}")
    print(f"PNG charts saved in: {PLOTS_DIR}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Netflix preprocessing, EDA, and KMeans clustering.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to netflix_titles.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Folder for server JSON output")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
