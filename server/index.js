const cors = require('cors');
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;
const DATA_DIR = path.join(__dirname, 'data');
const CLIENT_DIST_DIR = path.join(__dirname, '..', 'client', 'dist');

function readJson(filename) {
  const filePath = path.join(DATA_DIR, filename);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing ${filePath}. Run: py python/pipeline.py`);
  }
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

let titles = [];
let summary = {};

function loadData() {
  titles = readJson('netflix_clustered.json');
  summary = readJson('summary.json');
  console.log(`Loaded ${titles.length} titles and summary.json`);
}

function roundDistance(value) {
  return Number(value.toFixed(4));
}

function euclideanDistance(first, second) {
  const length = Math.min(first.length, second.length);
  let total = 0;
  for (let index = 0; index < length; index += 1) {
    const difference = Number(first[index]) - Number(second[index]);
    total += difference * difference;
  }
  return Math.sqrt(total);
}

app.use(cors());
app.use(express.json());
app.use(express.static(CLIENT_DIST_DIR));

app.get('/api/summary', (req, res) => {
  res.json(summary);
});

app.get('/api/search', (req, res) => {
  const query = String(req.query.q || '').trim().toLowerCase();
  if (!query) {
    return res.json([]);
  }

  const matches = titles
    .filter((item) => item.title.toLowerCase().includes(query))
    .slice(0, 10)
    .map((item) => ({
      show_id: item.show_id,
      title: item.title,
      type: item.type,
      release_year: item.release_year,
      cluster: item.cluster,
    }));
  return res.json(matches);
});

app.get('/api/recommend', (req, res) => {
  const showId = String(req.query.id || '').trim();
  const requestedNumber = Number.parseInt(req.query.n || '5', 10);
  const number = Number.isFinite(requestedNumber) ? Math.min(10, Math.max(3, requestedNumber)) : 5;
  const selected = titles.find((item) => item.show_id === showId);

  if (!selected) {
    return res.status(404).json({ error: 'Title not found. Search and select a title first.' });
  }

  const recommendations = titles
    .filter((item) => item.show_id !== selected.show_id && item.cluster === selected.cluster)
    .map((item) => ({
      show_id: item.show_id,
      title: item.title,
      type: item.type,
      release_year: item.release_year,
      listed_in: item.listed_in,
      description: item.description,
      country: item.country,
      cluster: item.cluster,
      distance: roundDistance(euclideanDistance(selected.features, item.features)),
    }))
    .sort((first, second) => first.distance - second.distance)
    .slice(0, number);

  return res.json({
    selected: {
      show_id: selected.show_id,
      title: selected.title,
      type: selected.type,
      cluster: selected.cluster,
    },
    recommendations,
  });
});

app.get('*', (req, res, next) => {
  if (req.path === '/api' || req.path.startsWith('/api/')) {
    return next();
  }
  return res.sendFile(path.join(CLIENT_DIST_DIR, 'index.html'));
});

try {
  loadData();
  app.listen(PORT, () => {
    console.log(`Express server running at http://localhost:${PORT}`);
    console.log('Routes: GET /api/summary, /api/search?q=, /api/recommend?id=&n=5');
  });
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
