# Netflix Content Clustering and Recommender

Mini project for Intelligent Data Systems 1, Experiment 10. The project has no database: Python writes two JSON files, and the Express server loads those files into memory.

The repository includes a small demo CSV so that the project can be run immediately. For the full lab result, replace `data/netflix_titles.csv` with Kaggle's **Netflix Movies and TV Shows** CSV, keeping the same filename and columns.

Small JSON fixtures are included in `server\data` for an immediate UI/API smoke test. The Python pipeline overwrites them with the computed results for whichever CSV is present.

## Windows run steps

Open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python python\pipeline.py
```

Important: run the pipeline with the project virtual environment activated. If you run the global `python` and see `Application Control policy has blocked this file` from scikit-learn, use these commands instead:

```powershell
.\.venv\Scripts\Activate.ps1
python python\pipeline.py
```

The pinned versions in `requirements.txt` are selected for Python 3.11 on Windows.

If PowerShell blocks activation, run this once in the same PowerShell window and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The pipeline prints the requested inspections and writes:

- `python\plots\` - EDA, elbow, silhouette, and cluster charts as PNG files
- `server\data\netflix_clustered.json` - titles, cluster labels, and scaled feature arrays
- `server\data\summary.json` - dashboard and cluster summary data

Install the JavaScript dependencies:

```powershell
npm install
```

Start the Express API in PowerShell window 1:

```powershell
npm start
```

Start the React/Vite client in PowerShell window 2:

```powershell
npm run dev
```

Open `http://localhost:5173` in a browser.

## API smoke tests

With the server running, use these PowerShell commands:

```powershell
Invoke-RestMethod http://localhost:5000/api/summary
Invoke-RestMethod "http://localhost:5000/api/search?q=the"
Invoke-RestMethod "http://localhost:5000/api/recommend?id=s001&n=5"
```

The last request uses the included demo CSV. With the Kaggle CSV, first search for a real `show_id` and use that value in the recommendation URL.

## Notebook

Open `python\netflix_analysis.ipynb` after installing the requirements. It follows the same pipeline functions in small, commented cells with markdown headings.

## Project structure

```text
data/netflix_titles.csv
python/netflix_analysis.ipynb
python/pipeline.py
python/plots/
server/index.js
server/data/
client/index.html
client/src/App.jsx
client/src/main.jsx
client/src/styles.css
client/vite.config.js
README.md
requirements.txt
package.json
```
