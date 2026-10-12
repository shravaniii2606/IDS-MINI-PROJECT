import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const tabs = ['Dashboard', 'Clusters', 'Recommend'];
const pieColors = ['#e50914', '#2f80ed'];

function chartEntries(value, labelName) {
  return Object.entries(value || {}).map(([name, count]) => ({
    name,
    [labelName]: Number(count),
  }));
}

function Dashboard({ summary }) {
  const typeData = chartEntries(summary.counts_by_type, 'count');
  const yearData = chartEntries(summary.titles_added_per_year, 'count');
  const genreData = (summary.top_10_genres || []).map((item) => ({ name: item.genre, count: item.count }));
  const countryData = (summary.top_10_countries || []).map((item) => ({ name: item.country, count: item.count }));

  return (
    <section className="dashboard-grid">
      <ChartCard title="Movies vs TV Shows">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={typeData} dataKey="count" nameKey="name" outerRadius={90} label>
              {typeData.map((entry, index) => <Cell key={entry.name} fill={pieColors[index % pieColors.length]} />)}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Titles Added per Year">
        <SimpleBarChart data={yearData} color="#f2994a" />
      </ChartCard>
      <ChartCard title="Top 10 Genres">
        <SimpleBarChart data={genreData} color="#27ae60" horizontal />
      </ChartCard>
      <ChartCard title="Top 10 Countries">
        <SimpleBarChart data={countryData} color="#9b51e0" horizontal />
      </ChartCard>
    </section>
  );
}

function ChartCard({ title, children }) {
  return <div className="chart-card"><h2>{title}</h2>{children}</div>;
}

function SimpleBarChart({ data, color, horizontal = false }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ left: horizontal ? 30 : 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        {horizontal ? <><XAxis type="number" /><YAxis dataKey="name" type="category" width={110} /></> : <><XAxis dataKey="name" angle={-35} textAnchor="end" height={65} /><YAxis /></>}
        <Tooltip />
        <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function Clusters({ summary }) {
  const clusterData = (summary.cluster_summary || []).map((item) => ({
    name: `C${item.cluster}`,
    size: item.size,
  }));
  const elbowData = summary.elbow_inertia_values || [];

  return (
    <section>
      <div className="metric-row">
        <div className="metric"><span>Chosen K</span><strong>{summary.chosen_k}</strong></div>
        <div className="metric"><span>Silhouette score</span><strong>{summary.silhouette_score}</strong></div>
      </div>
      <div className="two-column">
        <ChartCard title="Cluster Sizes">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={clusterData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="size" fill="#e50914" /></BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Elbow Inertia">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={elbowData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="k" /><YAxis /><Tooltip /><Line type="monotone" dataKey="inertia" stroke="#2f80ed" strokeWidth={3} /></LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <div className="table-card">
        <h2>Cluster Profile</h2>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Cluster</th><th>Size</th><th>Avg year</th><th>Avg duration</th><th>Top genres</th><th>Sample titles</th></tr></thead>
            <tbody>
              {(summary.cluster_summary || []).map((item) => (
                <tr key={item.cluster}>
                  <td>Cluster {item.cluster}</td><td>{item.size}</td><td>{item.avg_release_year}</td><td>{item.avg_duration_num}</td>
                  <td>{(item.top_genres || []).join(', ')}</td><td>{(item.sample_titles || []).join(' · ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function Recommend() {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selected, setSelected] = useState(null);
  const [number, setNumber] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      if (!query.trim() || (selected && query === selected.title)) {
        setSuggestions([]);
        return;
      }
      try {
        const response = await axios.get(`/api/search?q=${encodeURIComponent(query)}`);
        setSuggestions(response.data);
      } catch (requestError) {
        setError(requestError.response?.data?.error || 'Could not search titles.');
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query, selected]);

  async function getRecommendations() {
    if (!selected) {
      setError('Select a title from the search suggestions first.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`/api/recommend?id=${encodeURIComponent(selected.show_id)}&n=${number}`);
      setResults(response.data.recommendations);
    } catch (requestError) {
      setError(requestError.response?.data?.error || 'Could not load recommendations.');
    } finally {
      setLoading(false);
    }
  }

  function chooseTitle(item) {
    setSelected(item);
    setQuery(item.title);
    setSuggestions([]);
    setResults([]);
    setError('');
  }

  return (
    <section className="recommend-panel">
      <div className="search-area">
        <label htmlFor="title-search">Search for a title</label>
        <input id="title-search" value={query} onChange={(event) => { setQuery(event.target.value); setSelected(null); }} placeholder="Try The, Code, Ocean..." />
        {suggestions.length > 0 && <div className="suggestions">{suggestions.map((item) => <button type="button" key={item.show_id} onClick={() => chooseTitle(item)}>{item.title} <span>{item.type} · {item.release_year}</span></button>)}</div>}
      </div>
      <div className="recommend-controls">
        <label htmlFor="recommend-count">Number of recommendations</label>
        <input id="recommend-count" type="number" min="3" max="10" value={number} onChange={(event) => setNumber(Math.min(10, Math.max(3, Number(event.target.value) || 3)))} />
        <button className="primary-button" type="button" onClick={getRecommendations} disabled={loading}>{loading ? 'Loading...' : 'Recommend'}</button>
      </div>
      {selected && <p className="selected-note">Selected: <strong>{selected.title}</strong> · cluster {selected.cluster}</p>}
      {error && <p className="error-message">{error}</p>}
      {results.length > 0 && <div className="results"><h2>Recommended titles</h2><div className="result-grid">{results.map((item) => <article className="result-card" key={item.show_id}><div className="result-heading"><h3>{item.title}</h3><span className="distance">distance {item.distance}</span></div><p className="muted">{item.type} · {item.release_year} · {item.listed_in}</p><p>{item.description}</p><small>{item.country}</small></article>)}</div></div>}
    </section>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    axios.get('/api/summary')
      .then((response) => setSummary(response.data))
      .catch((requestError) => setError(requestError.response?.data?.error || 'Could not load the summary. Run the Python pipeline first.'))
      .finally(() => setLoading(false));
  }, []);

  const content = useMemo(() => {
    if (!summary) return null;
    if (activeTab === 'Clusters') return <Clusters summary={summary} />;
    if (activeTab === 'Recommend') return <Recommend />;
    return <Dashboard summary={summary} />;
  }, [activeTab, summary]);

  return (
    <main className="app-shell">
      <header className="hero"><div><p className="eyebrow">Intelligent Data Systems 1 · Experiment 10</p><h1>Netflix Content Clustering</h1><p className="subtitle">A small KMeans-based explorer for movies and TV shows.</p></div></header>
      <nav className="tabs" aria-label="Main sections">{tabs.map((tab) => <button key={tab} type="button" className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav>
      {loading && <div className="state-message">Loading summary...</div>}
      {!loading && error && <div className="state-message error-message">{error}</div>}
      {!loading && !error && content}
      <footer>Features: release year, duration, genre count, and top-10 genre flags · StandardScaler · KMeans</footer>
    </main>
  );
}
