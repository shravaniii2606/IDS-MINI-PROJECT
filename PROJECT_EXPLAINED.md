# Netflix Content Clustering and Recommender: Complete Explanation

This document describes the actual current files and generated outputs. The full Kaggle CSV is in data/netflix_titles.csv. The current run produced 8,705 cleaned records, selected K=5, and printed a silhouette score of 0.4646.

## 1. OVERVIEW

This project reads Netflix movie and TV-show metadata from a CSV file. Python cleans the data, creates numerical features, draws exploratory charts, and applies K-Means clustering. The clusters group titles using release year, duration, genre count, and ten binary genre flags. Python writes the results as JSON; there is no database. Express loads the JSON and provides APIs. React/Vite displays charts, cluster profiles, search suggestions, and recommendations.

### Final folder structure

~~~text
data/                               Input data folder.
  netflix_titles.csv                 Kaggle Netflix dataset.
python/                             Python analysis folder.
  pipeline.py                        Preprocessing, EDA, K-Means, and JSON export.
  netflix_analysis.ipynb             Notebook calling the same pipeline functions.
  plots/                             Generated PNG charts.
    .gitkeep                         Keeps the plots folder in version control.
    boxplot_duration_by_type.png     Duration boxplot by content type.
    cluster_scatter.png              Release year versus duration by cluster.
    cluster_size_bar.png             Number of records in each cluster.
    correlation_heatmap.png          Correlation heatmap.
    countplot_rating.png             Count of titles by rating.
    countplot_type.png               Count of movies and TV shows.
    elbow_plot.png                   K-Means inertia for K=2 through K=10.
    histogram_movie_duration.png     Histogram of movie duration values.
    histogram_release_year.png       Histogram of release years.
    silhouette_scores.png            Silhouette scores for K=2 through K=10.
    titles_added_per_year.png        Titles added by year line chart.
    top_10_countries.png             Top countries chart.
    top_10_genres.png                Top genres chart.
server/                             Express server and generated JSON folder.
  index.js                           Express server and API routes.
  data/                              JSON output folder loaded by Express.
  data/netflix_clustered.json        Cleaned titles, clusters, and feature arrays.
  data/summary.json                  Dashboard and clustering summary.
client/                             React/Vite front-end folder.
  index.html                         Browser HTML entry page.
  vite.config.js                     Vite settings and API proxy.
  src/                               React source folder.
  src/main.jsx                       React entry point.
  src/App.jsx                        Tabs, components, charts, and API calls.
  src/styles.css                     Plain responsive CSS.
package.json                         Node dependencies and scripts.
package-lock.json                    Exact Node dependency versions.
requirements.txt                     Pinned Python dependency versions.
README.md                            Windows run steps and API tests.
PROJECT_EXPLAINED.md                 This explanation.
.gitignore                           Ignored dependencies, caches, and build output.
~~~

The local .venv and node_modules folders may exist after installation but are ignored. client/dist is temporary Vite build output.

## 2. DATASET

The input is data/netflix_titles.csv from Kaggle's Netflix Movies and TV Shows dataset.

Actual size:

- Before cleaning: 8,807 rows and 12 columns.
- After main cleaning: 8,705 rows and 15 columns.
- After ten genre flags: 8,705 rows and 25 internal columns.
- After cluster and features: 8,705 rows and 27 internal columns.

The original columns are:

| Column | Meaning |
|---|---|
| show_id | Unique ID, such as s1. |
| type | Movie or TV Show. |
| title | Title name. |
| director | Director value. |
| cast | Cast values. |
| country | Country or countries. |
| date_added | Date added to Netflix. |
| release_year | Original release year. |
| rating | Content rating. |
| duration | Minutes for a movie or seasons for a TV show. |
| listed_in | Comma-separated genres/categories. |
| description | Short description. |

K-Means uses 13 numerical features:

1. release_year
2. duration_num
3. genre_count
4. genre_international_movies
5. genre_dramas
6. genre_comedies
7. genre_international_tv_shows
8. genre_documentaries
9. genre_action_adventure
10. genre_independent_movies
11. genre_tv_dramas
12. genre_children_family_movies
13. genre_romantic_movies

The last ten are multi-hot flags. Title, description, director, cast, country, and rating are not direct K-Means features.

## 3. PYTHON PART: EXACT ORDER

The main implementation is python/pipeline.py. The notebook imports its functions, so both use the same logic.

### 3.1 Imports and paths

The script imports pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn. Important imports are:

~~~python
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
~~~

Paths are calculated as:

~~~python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "netflix_titles.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "server" / "data"
PLOTS_DIR = PROJECT_ROOT / "python" / "plots"
~~~

### 3.2 Loading and inspecting

The CSV is loaded with:

~~~python
df = pd.read_csv(csv_path)
~~~

The exact inspection commands are:

~~~python
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.isnull().sum()
~~~

Actual original shape: (8807, 12). The first title was Dick Johnson Is Dead with show_id s1. release_year was the only numeric column before feature creation; its mean was about 2014.18, minimum 1925, and maximum 2021.

### 3.3 Missing values

Before cleaning:

| Column | Missing |
|---|---:|
| show_id | 0 |
| type | 0 |
| title | 0 |
| director | 2,634 |
| cast | 825 |
| country | 831 |
| date_added | 10 |
| release_year | 0 |
| rating | 4 |
| duration | 3 |
| listed_in | 0 |
| description | 0 |

Blank strings in date_added, rating, and duration are converted to NaN:

~~~python
df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)
~~~

Three malformed rows contain 74 min, 84 min, or 66 min in rating while duration is blank. The code detects them with:

~~~python
rating_text = df["rating"].astype("string").str.strip()
malformed_rating = rating_text.str.match(r"^\d+\s*min$", na=False)
duration_is_blank = df["duration"].isna()
~~~

It moves the value to duration and sets rating to Unknown:

~~~python
df.loc[malformed_rating & duration_is_blank, "duration"] = rating_text[
    malformed_rating & duration_is_blank
]
df.loc[malformed_rating, "rating"] = "Unknown"
~~~

After repair, it drops rows still missing date_added, rating, or duration:

~~~python
df = df.dropna(
    subset=["date_added", "rating", "duration"]
).copy()
~~~

Actual output:

~~~text
Rows removed by dropna(date_added, rating, duration): 14
~~~

The descriptive columns are filled with Unknown:

~~~python
for column in ["director", "cast", "country"]:
    df[column] = df[column].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")
df["description"] = df["description"].fillna("Unknown")
~~~

After later cleaning, the null count is zero for every remaining column.

## 3.4 Duplicates

The exact commands are:

~~~python
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
~~~

The actual result was df.duplicated().sum(): 0, so no duplicates were removed.

## 3.5 Text cleaning

Type is lowercased, stripped, and normalized:

~~~python
df["type"] = (
    df["type"].astype("string").str.lower().str.strip().replace(
        {"movie": "Movie", "tv show": "TV Show", "tvshow": "TV Show"}
    )
)
~~~

Ratings are lowercased and stripped, then selected values such as tv-ma, tv-14, pg-13, nr, and unknown are replaced with display labels.

Country and listed_in are sent through normalise_multi_value. The helper splits comma-separated values, strips them, and applies title case:

~~~python
df["country"] = df["country"].astype("string").str.lower().str.strip()
df["country"] = df["country"].map(normalise_multi_value)
df["listed_in"] = df["listed_in"].map(normalise_multi_value)
~~~

Actual limitation: the rating replacement dictionary does not cover every possible label. Lower-case r, pg, and g remain in the final chart data.

## 3.6 Dates, durations, and new columns

Date conversion and year extraction are:

~~~python
df["date_added"] = pd.to_datetime(
    df["date_added"], errors="coerce"
)
df = df.dropna(subset=["date_added"]).copy()
df["year_added"] = df["date_added"].dt.year.astype(int)
~~~

The final cleaning shape was (8705, 15). The 15 columns are the original 12 plus year_added, duration_num, and genre_count.

The initial required-field drop removed 14 rows. Date conversion then caused 88 more rows to become NaT and be removed. Some current Kaggle date values contain leading spaces or formats that this Pandas version does not parse consistently.

Duration number is extracted from values such as 90 min and 2 Seasons:

~~~python
df["duration_num"] = pd.to_numeric(
    df["duration"].astype("string").str.extract(r"(\d+)")[0],
    errors="coerce"
)
df["duration_num"] = df["duration_num"].astype(int)
~~~

For movies this generally means minutes; for TV shows it generally means seasons.

Other numeric fields:

~~~python
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
).astype(int)

df["genre_count"] = df["listed_in"].map(
    lambda value: len(split_values(value))
).astype(int)
~~~

## 3.7 Genre flags

The code takes the ten most frequent genres:

~~~python
genre_counts = multivalue_counts(df["listed_in"])
top_genres = [
    genre for genre, _ in genre_counts.most_common(10)
]
~~~

Actual top-ten counts:

| Rank | Genre | Count |
|---:|---|---:|
| 1 | International Movies | 2,752 |
| 2 | Dramas | 2,426 |
| 3 | Comedies | 1,674 |
| 4 | International Tv Shows | 1,327 |
| 5 | Documentaries | 869 |
| 6 | Action & Adventure | 859 |
| 7 | Independent Movies | 756 |
| 8 | Tv Dramas | 739 |
| 9 | Children & Family Movies | 641 |
| 10 | Romantic Movies | 616 |

Each genre receives a safe column name and binary flag:

~~~python
slug = re.sub(
    r"[^a-z0-9]+", "_", genre.lower()
).strip("_")
column = f"genre_{slug}"
df[column] = row_genres.map(
    lambda values, item=genre: int(item in values)
)
~~~

For example, a title in Dramas receives genre_dramas=1.

## 3.8 Category conversion

The code converts type and rating to category:

~~~python
df["type"] = df["type"].astype("category")
df["rating"] = df["rating"].astype("category")
~~~

They are used for grouping and charts, not directly as K-Means numerical features.

## 3.9 Outlier detection

IQR detection:

~~~python
q1 = df["duration_num"].quantile(0.25)
q3 = df["duration_num"].quantile(0.75)
iqr = q3 - q1
iqr_outliers = int(
    (
        (df["duration_num"] < q1 - 1.5 * iqr) |
        (df["duration_num"] > q3 + 1.5 * iqr)
    ).sum()
)
~~~

Z-score detection:

~~~python
z_scores = np.abs(
    stats.zscore(df["duration_num"], nan_policy="omit")
)
z_outliers = int((z_scores > 3).sum())
~~~

Actual results:

- IQR = 103.00.
- IQR outliers = 2.
- Z-score outliers with absolute value above 3 = 9.

No outliers are removed.

## 3.10 Scaling

MinMaxScaler is demonstrated:

~~~python
minmax_scaler = MinMaxScaler()
minmax_values = minmax_scaler.fit_transform(
    df[feature_columns]
)
~~~

StandardScaler is also demonstrated:

~~~python
standard_scaler = StandardScaler()
standard_values = standard_scaler.fit_transform(
    df[feature_columns]
)
~~~

K-Means uses StandardScaler values:

~~~python
df["features"] = standard_values.round(6).tolist()
~~~

MinMax values are printed for comparison but do not feed the final model. StandardScaler is used because K-Means uses distances; scaling prevents one feature from dominating only because of its numerical units.

## 3.11 EDA charts

The function save_eda_plots creates nine EDA charts.

1. countplot_type.png uses sns.countplot(data=df, x="type"). It shows 6,129 movies and 2,576 TV shows, so movies are the majority.

2. countplot_rating.png uses a countplot of the 15 most common ratings. TV-MA is largest with 3,183 and TV-14 has 2,133. Lower-case r, pg, and g reveal incomplete normalization.

3. histogram_release_year.png uses sns.histplot on release_year with 15 bins. Years range from 1925 to 2021, with most content concentrated after 2015.

4. histogram_movie_duration.png uses only Movie rows and duration_num. Movie mean duration is about 99.57 minutes and median is 98 minutes.

5. titles_added_per_year.png is actually a line chart:

~~~python
year_counts = df["year_added"].value_counts().sort_index()
sns.lineplot(
    x=year_counts.index,
    y=year_counts.values,
    marker="o",
    ax=ax
)
~~~

Counts rise sharply after 2016: 2016=417, 2017=1,162, 2018=1,624, 2019=1,999, 2020=1,878, and 2021=1,498. The highest count is 2019.

6. top_10_countries.png is a horizontal bar chart. Actual top counts are United States 3,643, India 1,045, United Kingdom 787, Canada 432, France 389, Japan 313, Spain 228, South Korea 226, Germany 225, and Mexico 169. Multi-country titles can count more than once.

7. top_10_genres.png is a horizontal bar chart. International Movies, Dramas, and Comedies are the largest categories. A title can have several genres.

8. correlation_heatmap.png uses release_year, duration_num, genre_count, and year_added. Actual important correlations are release_year/duration_num=-0.255, release_year/year_added=0.111, duration_num/genre_count=0.086, and genre_count/year_added=0.048. These are mostly weak relationships.

9. boxplot_duration_by_type.png compares duration_num by type. Movies average about 99.57; TV shows average about 1.69 because their values are seasons. This also reveals the unit mismatch.

## 3.12 K-Means

The feature matrix is:

~~~python
features = feature_info["standard_values"]
~~~

It has 8,705 rows and 13 columns.

For each K from 2 through 10:

~~~python
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
labels = model.fit_predict(features)
~~~

Actual inertia values:

| K | Inertia |
|---:|---:|
| 2 | 18,506.5639 |
| 3 | 13,333.4457 |
| 4 | 8,496.4675 |
| 5 | 6,756.7140 |
| 6 | 5,526.8334 |
| 7 | 4,591.2460 |
| 8 | 4,016.4673 |
| 9 | 3,501.5784 |
| 10 | 3,184.8740 |

Actual silhouette values:

| K | Score |
|---:|---:|
| 2 | 0.3439 |
| 3 | 0.3930 |
| 4 | 0.4621 |
| 5 | 0.4646 |
| 6 | 0.4972 |
| 7 | 0.5065 |
| 8 | 0.5322 |
| 9 | 0.5424 |
| 10 | 0.5408 |

The code calculates both measures but chooses K using:

~~~python
chosen_k = 5 if len(df) > 5 else max(2, len(df) - 1)
~~~

Therefore final K is 5 because the dataset has more than five rows. K=5 is the requested/default lab choice, not the highest-silhouette choice. K=9 has the highest tested silhouette, 0.5424.

The final model adds labels:

~~~python
model = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)
df["cluster"] = model.fit_predict(
    features
).astype(int)
~~~

The Python side then saves elbow, silhouette, cluster-size, and cluster-scatter charts.

## 3.13 Cluster profile

| Cluster | Size | Avg year | Avg duration | Avg genres | Type split | Top genres | Sample titles |
|---:|---:|---:|---:|---:|---|---|---|
| 0 | 485 | 1985.26 | 110.88 | 2.33 | Movie 464, TV Show 21 | Dramas, International Movies, Comedies | Sankofa; Avvai Shanmughi; Jeans |
| 1 | 2,101 | 2014.96 | 102.42 | 1.96 | Movie 2,101 | International Movies, Dramas, Comedies | The Starling; Je Suis Karl; Confessions of an Invisible Girl |
| 2 | 2,143 | 2017.17 | 2.90 | 2.61 | TV Show 2,053, Movie 90 | International Tv Shows, Tv Dramas, Tv Comedies | Blood & Water; Ganglands; Jailbirds New Orleans |
| 3 | 2,145 | 2015.19 | 110.24 | 3.00 | Movie 2,145 | International Movies, Dramas, Comedies | Ankahi Kahaniya; The Father Who Moves Mountains; The Stronghold |
| 4 | 1,831 | 2016.34 | 56.24 | 1.00 | Movie 1,329, TV Show 502 | Documentaries, Stand-Up Comedy, Kids' Tv | Dick Johnson Is Dead; My Little Pony: A New Generation; Intrusion |

Interpretation:

- Cluster 0: mostly old movies, average year 1985, movie-like duration.
- Cluster 1: mostly modern movies, average year about 2015, duration about 102 minutes.
- Cluster 2: mostly TV shows because season counts are small.
- Cluster 3: movies with about three genres and movie-like duration.
- Cluster 4: mixed shorter-duration content with documentaries, stand-up comedy, and children’s content.

Cluster numbers are labels, not quality ranks.

## 3.14 JSON outputs

netflix_clustered.json is a list with 8,705 records. Each record contains show_id, title, type, release_year, rating, listed_in, description, country, duration_num, genre_count, year_added, cluster, and features.

features is the 13-number StandardScaler vector used by the server. director, cast, and date_added are not written to this JSON.

summary.json contains counts_by_type, titles_added_per_year, top_10_genres, top_10_countries, cluster_summary, chosen_k, silhouette_score, elbow_inertia_values, and silhouette_values.

## 4. EXPRESS SERVER

The server is server/index.js. It uses Express, cors, fs, and path.

### Startup

The data folder is:

~~~javascript
const DATA_DIR = path.join(__dirname, 'data');
~~~

At startup it reads both files synchronously:

~~~javascript
titles = readJson('netflix_clustered.json');
summary = readJson('summary.json');
~~~

It loads them into memory and prints the number of titles. There is no database and no automatic reload while the server is running.

### GET /api/summary

Request:

~~~text
GET http://localhost:5000/api/summary
~~~

There are no parameters. It returns the full summary.json object. A shortened response is:

~~~json
{
  "counts_by_type": {
    "Movie": 6129,
    "TV Show": 2576
  },
  "chosen_k": 5,
  "silhouette_score": 0.4646
}
~~~

The actual response also contains yearly counts, top genres, top countries, cluster profiles, inertia values, and silhouette values.

### GET /api/search?q=

Request:

~~~text
GET http://localhost:5000/api/search?q=the
~~~

The q value is trimmed and lowercased. The route checks whether a title contains the query, case-insensitively, and returns at most ten records.

An actual response starts with:

~~~json
[
  {
    "show_id": "s9",
    "title": "The Great British Baking Show",
    "type": "TV Show",
    "release_year": 2021,
    "cluster": 2
  },
  {
    "show_id": "s10",
    "title": "The Starling",
    "type": "Movie",
    "release_year": 2021,
    "cluster": 1
  }
]
~~~

If q is empty, it returns [].

### GET /api/recommend?id=<show_id>&n=5

Example:

~~~text
GET http://localhost:5000/api/recommend?id=s1&n=3
~~~

id identifies the selected title. n defaults to 5 and is clamped between 3 and 10.

The current selected title s1 is Dick Johnson Is Dead in cluster 4. The response has selected and recommendations:

~~~json
{
  "selected": {
    "show_id": "s1",
    "title": "Dick Johnson Is Dead",
    "type": "Movie",
    "cluster": 4
  },
  "recommendations": [
    {
      "show_id": "s230",
      "title": "The Old Ways",
      "type": "Movie",
      "release_year": 2020,
      "listed_in": "Horror Movies",
      "description": "...",
      "country": "United States",
      "cluster": 4,
      "distance": 0
    }
  ]
}
~~~

If the title is not found, the route returns HTTP 404 with an error object.

### Recommendation distance

The server:

1. Finds the selected title by show_id.
2. Removes that title from candidates.
3. Keeps candidates with the same cluster.
4. Computes Euclidean distance on the saved features array.
5. Sorts from smallest distance to largest.
6. Rounds to four decimal places.
7. Returns the first n titles.

The distance calculation is:

~~~javascript
const difference = Number(first[index]) - Number(second[index]);
total += difference * difference;
return Math.sqrt(total);
~~~

Small example:

~~~text
Selected vector:       [1, 2, 0]
Candidate vector:      [2, 4, 1]
Differences:           [1, 2, 1]
Squared differences:   [1, 4, 1]
Sum:                   6
Distance:              sqrt(6) = 2.4495
~~~

The real application uses 13 values. Smaller distance means greater similarity according to the metadata features.

## 5. REACT FRONT END

The client uses React, Vite, Axios, and Recharts. It has no router, Redux, TypeScript, authentication, or database.

### Files and components

- client/index.html: HTML root page.
- client/src/main.jsx: React entry point and StrictMode rendering.
- client/src/App.jsx: App, Dashboard, Clusters, Recommend, ChartCard, and SimpleBarChart components.
- client/src/styles.css: plain responsive styling.
- client/vite.config.js: Vite port 5173 and /api proxy to port 5000.

### Dashboard tab

App first requests /api/summary. Dashboard uses the response to show:

1. Pie chart: Movie versus TV Show.
2. Bar chart: titles added per year.
3. Horizontal bar chart: top genres.
4. Horizontal bar chart: top countries.

The app shows a loading message during the request and an error message if it fails.

### Clusters tab

Clusters shows:

- Chosen K=5.
- Silhouette score 0.4646.
- Cluster-size bar chart.
- Elbow-inertia line chart.
- Profile table with cluster, size, average year, average duration, top genres, and sample titles.

The Python side creates silhouette_scores.png and stores all values, but the current React tab does not draw a separate silhouette line chart.

### Recommend tab

Recommend provides:

- Search input.
- Up to ten search suggestions.
- Number input from 3 to 10, default 5.
- Recommend button.
- Selected title and cluster.
- Cards with title, type, year, genres, description, country, and distance.

After a 250-millisecond debounce, typing calls the search route. Selecting a title stores it in React state. Clicking Recommend calls the recommendation route.

### Complete click flow

1. User clicks Recommend.
2. React displays the search interface.
3. User types a title fragment.
4. React calls GET /api/search?q=....
5. Express searches in-memory JSON.
6. React displays suggestions.
7. User selects a title and chooses n.
8. React calls GET /api/recommend?id=...&n=....
9. Express filters the selected cluster and calculates distances.
10. Express returns JSON.
11. React renders recommendation cards.

## 6. HOW EVERYTHING CONNECTS

~~~text
data/netflix_titles.csv
          |
          v
Python pipeline.py
  clean data
  create features
  save EDA charts
  StandardScaler + K-Means
          |
          +----------------------+
          v                      v
server/data/                python/plots/
  summary.json              PNG charts
  netflix_clustered.json
          |
          v
Express server on port 5000
  /api/summary
  /api/search
  /api/recommend
          |
          v
React/Vite client on port 5173
          |
          v
User sees charts, clusters, and recommendations
~~~

## 7. HOW TO RUN

In PowerShell:

~~~powershell
cd C:\Users\Admin\IDS-MINI-PROJECT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

If activation is blocked:

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
~~~

Install and run Python:

~~~powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python python\pipeline.py
~~~

This refreshes the PNG files and both JSON files.

Install JavaScript dependencies:

~~~powershell
npm install
~~~

PowerShell window 1:

~~~powershell
npm start
~~~

PowerShell window 2:

~~~powershell
npm run dev
~~~

Open:

~~~text
http://localhost:5173
~~~

API checks:

~~~powershell
Invoke-RestMethod http://localhost:5000/api/summary
Invoke-RestMethod "http://localhost:5000/api/search?q=the"
Invoke-RestMethod "http://localhost:5000/api/recommend?id=s1&n=5"
~~~

The current full dataset uses IDs such as s1. The older demo example s001 is not the current ID.

## 8. REPORT MAPPING

| Report section | What to include | Actual source |
|---|---|---|
| Problem Statement | Netflix has many titles, so grouping and simple similarity search are useful. | README.md and project behavior. |
| Aim | Clean metadata, find groups using K-Means, and display recommendations. | pipeline.py and App.jsx. |
| Dataset | Kaggle source, 8,807 rows, 12 columns, and important fields. | data/netflix_titles.csv and df.info(). |
| Theory | Missing values, scaling, unsupervised learning, K-Means, centroids, distance, inertia, silhouette. | This explanation. |
| Data Preprocessing | Null handling, malformed ratings, duplicates, text cleaning, dates, durations, genre count, categories, and flags. | pipeline.py and terminal output. |
| Data Analysis and Visualization | Explain nine EDA charts and observations. | python/plots/*.png and save_eda_plots(). |
| Data Science Model (K-Means) | Explain 13 standardized features, K values, K=5, profiles, and limitations. | run_clustering() and summary.json. |
| Front End | Explain three tabs, Axios, Recharts, search, and recommendation flow. | App.jsx, styles.css, vite.config.js. |
| Complete Code | Reference Python, Node, and React source. | pipeline.py, server/index.js, client/src/App.jsx. |
| Output Screenshots | Include terminal, chart, dashboard, cluster, and recommendation images. | List below. |
| Observations | Movie majority, recent additions, top genres/countries, TV-season cluster, and K limitation. | summary.json and PNGs. |
| Conclusion | K-Means produced five broad metadata groups and the app provides same-cluster recommendations. | Cluster table and browser UI. |
| References | Cite Kaggle and official library documentation. | requirements.txt and package.json. |

### Screenshots to take

1. Pipeline terminal showing head, tail, shape, null values, and final K/silhouette.
2. countplot_type.png.
3. countplot_rating.png.
4. histogram_release_year.png.
5. histogram_movie_duration.png.
6. titles_added_per_year.png.
7. top_10_countries.png.
8. top_10_genres.png.
9. correlation_heatmap.png.
10. boxplot_duration_by_type.png.
11. elbow_plot.png.
12. silhouette_scores.png.
13. cluster_size_bar.png.
14. cluster_scatter.png.
15. Browser Dashboard tab.
16. Browser Clusters tab and profile table.
17. Recommend tab with search suggestions.
18. Recommend tab with recommendation cards and distances.

## 9. VIVA Q&A

### What is the real-world application?

A streaming service can use metadata to organize a large catalog into broad groups.

Those groups can support content exploration and simple similarity recommendations.

This lab version uses only metadata; a production service would also use user behavior.

### Why is this dataset suitable?

It has thousands of real movie and TV-show records.

It includes type, year, duration, country, rating, and genres.

Those fields can be converted into numerical features for clustering.

### Which preprocessing techniques were used and why?

Missing descriptive values are filled with Unknown.

Required missing date/rating/duration values are dropped, malformed minute values are repaired, and duplicates are checked.

Text is lowercased, stripped, and normalized; dates, durations, genre count, and genre flags become usable features.

Categories and scaling are used before clustering.

### Why K-Means and why scaling?

K-Means is simple and suitable when there is no target label.

It assigns records to the nearest centroid.

K-Means uses distances, so scaling prevents a large-unit feature from dominating.

This project uses StandardScaler for the actual model.

### How does the front end interact with data science?

Python produces JSON rather than a live Python API.

Express loads the JSON and exposes HTTP endpoints.

React uses Axios to request the data, then uses Recharts and HTML to display it.

### What are limitations and future improvements?

K=5 is hardcoded even though K=9 has the highest measured silhouette.

Movie minutes and TV seasons are mixed in duration_num.

The model does not use user history, ratings, watch behavior, or descriptions.

Future work could use automatic K selection, separate movie/TV features, a database, user behavior, and a stronger recommender.

### Additional likely viva questions

#### What is a centroid?

A centroid is the center point of a cluster.

Each centroid feature is an average of the records assigned to that cluster.

K-Means assigns records to nearby centroids and repeatedly updates them.

#### What is Euclidean distance?

It is the straight-line distance between two points.

For vectors, it is the square root of the sum of squared coordinate differences.

The server uses it to rank same-cluster titles.

#### What is the elbow method?

It plots K against inertia.

Inertia decreases as K increases, so we look for a bend where extra clusters give smaller improvements.

This project evaluates K from 2 through 10.

#### What is a silhouette score?

It compares how close a record is to its own cluster with how close it is to other clusters.

A higher value generally means better separation and compactness.

K=5 is 0.4646; K=9 is 0.5424 in this run.

#### Why is this unsupervised learning?

There is no correct cluster label in the input CSV.

K-Means discovers groups from feature patterns.

Therefore this is clustering, not classification or regression.

#### What is a multi-hot feature?

It represents categories with several 0/1 columns.

A title can have genre_dramas=1 and genre_comedies=1 at the same time.

This supports multiple genres for one title.

#### Why was no database used?

The lab requested no database.

The two JSON files are small enough for Express to load into memory.

A production application would normally use a database or data service.

#### What is an API?

An API is a defined way for programs to communicate.

React requests JSON from Express through HTTP GET routes.

The routes are /api/summary, /api/search, and /api/recommend.

#### What does random_state=42 do?

K-Means can start with different random centroid positions.

random_state=42 makes the process repeatable.

This helps produce comparable results.

#### Why is n_init=10 used?

K-Means can give different results from different starting centroids.

n_init=10 tries multiple initializations and keeps a good result.

This lowers the chance of a poor initialization.

#### Why restrict recommendations to the same cluster?

The cluster is a broad similarity filter.

Euclidean distance then ranks titles within that group.

This is simple and fast, although a close title in another cluster is excluded.

#### What is inertia?

Inertia is the total squared distance from records to their assigned centroids.

Lower inertia means points are closer to cluster centers.

It normally decreases as K increases, so it is interpreted with the elbow plot.

## 10. ISSUES, SHORTCUTS, AND DIFFERENCES

### K is not automatically optimized

The code calculates elbow and silhouette values but uses a hardcoded default K=5 for a dataset larger than five rows. In the actual run, K=9 has the best silhouette. The report should state that K=5 is the requested/default lab choice, not the automatic optimum.

### Date parsing removes extra records

The code does not strip date_added immediately before pd.to_datetime(). With the current Pandas version, 88 records became NaT and were dropped. This contributes to the final 8,705 rows.

### Movie minutes and TV seasons are mixed

The same duration_num column contains minutes for movies and seasons for TV shows. This is a simple feature but not a perfect unit comparison.

### Rating normalization is incomplete

The replacement dictionary does not cover every rating. Lower-case r, pg, and g remain. Rating is not used for clustering, so this mainly affects EDA readability.

### The notebook is compact

The notebook has markdown headings and small code cells, but most detailed preprocessing is inside load_and_prepare(). It does not show every individual pandas operation in a separate cell.

### The browser does not show the silhouette line chart

Python creates silhouette_scores.png and summary.json stores all silhouette values. The Clusters tab shows the selected score as a metric but does not draw that separate chart.

### Some README examples are stale

The current full dataset uses show_id s1, while older README text contains the demo ID s001. Use a real ID from the current CSV.

### The server startup error text is stale

server/index.js says to run py python/pipeline.py if JSON files are missing. This setup uses python with the project virtual environment. README.md contains the correct commands.

### No saved K-Means model

Only output labels and feature vectors are saved. The model is recalculated every time pipeline.py runs.

### No automated unit tests

The project was manually tested with PowerShell API requests and a Vite production build. There are no test files.

### Bundle-size warning

The Vite build passes, but Recharts makes the JavaScript bundle larger than Vite’s default warning threshold. This is a warning, not a failure.

### Encoding artifact

Some separators in the current App.jsx display as Â· instead of a normal middle dot in text output. This is a minor encoding/display issue and does not affect the data or API logic.

### Features not used

The project does not use classification, regression, TF-IDF, cosine similarity, T-tests, a database, authentication, Redux, TypeScript, or a router. The only machine-learning technique is K-Means.




