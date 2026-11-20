import { useCallback, useEffect, useRef, useState } from "react";

import {
  getHealth,
  getHistory,
  getSamples,
  getStats,
  predictFile,
  predictSample,
  reportUrl
} from "./api.js";

const fmtDate = (iso) =>
  new Date(iso).toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short"
  });

const API_BASE = "";

// =====================================
// WELCOME SCREEN
// =====================================

function WelcomeScreen({ onStart }) {
  return (
    <div className="welcome-screen">

      <div className="welcome-glow glow-one"></div>
      <div className="welcome-glow glow-two"></div>

      <div className="welcome-content">

        <div className="welcome-icon">🧠</div>

        <p className="welcome-tag">
          AI POWERED HEALTHCARE TECHNOLOGY
        </p>

        <h1>
          NeuroScan <span>AI</span>
        </h1>

        <h2>Brain Tumor Detection System</h2>

        <p className="welcome-description">
          Intelligent MRI Analysis with Deep Learning,
          CNN Classification and Grad-CAM Visualization.
        </p>

        <button
          className="start-button"
          onClick={onStart}
        >
          START APPLICATION <span>→</span>
        </button>

        <div className="welcome-footer">
          <span>AI RESEARCH PROJECT</span>
          <span>•</span>
          <span>4-CLASS CNN MODEL</span>
        </div>

      </div>
    </div>
  );
}

// =====================================
// STATUS PILL
// =====================================

function StatusPill({ health }) {
  const online = health?.status === "online";
  const demo = health?.model_mode === "demo";

  return (
    <div className={`pill ${online ? (demo ? "warn" : "ok") : "bad"}`}>
      <span className="dot" />

      {!online
        ? "Backend offline"
        : demo
        ? "Online - demo model"
        : "Online - model loaded"}
    </div>
  );
}

// =====================================
// DASHBOARD
// =====================================

function Dashboard({ stats, history, health, go }) {
  return (
    <section>
      <h1>Dashboard</h1>

      <div className="grid4">

        <div className="card stat">
          <b>{stats.total}</b>
          <span>Total scans</span>
        </div>

        <div className="card stat">
          <b className="red">{stats.tumor}</b>
          <span>Tumor detected</span>
        </div>

        <div className="card stat">
          <b>{stats.normal}</b>
          <span>No tumor</span>
        </div>

        <div className="card stat">
          <b className="small">
            {health?.model_mode === "trained"
              ? "Trained CNN"
              : health
              ? "Demo mode"
              : "Offline"}
          </b>

          <span>System status</span>
        </div>

      </div>

      <div className="row">
        <h2>Recent predictions</h2>

        <button
          className="btn primary"
          onClick={() => go("analyze")}
        >
          Upload MRI
        </button>
      </div>

      <History rows={history.slice(0, 5)} compact />

      {health?.model_mode === "demo" && (
        <p className="note">
          The backend has no trained model yet.
        </p>
      )}
    </section>
  );
}

// =====================================
// CONFIDENCE BAR
// =====================================

function Confidence({ value, tumor }) {
  const [w, setW] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setW(value * 100), 80);
    return () => clearTimeout(t);
  }, [value]);

  return (
    <div
      className="bar"
      role="progressbar"
      aria-valuenow={Math.round(value * 100)}
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <div
        className={`fill ${tumor ? "red" : ""}`}
        style={{ width: `${w}%` }}
      />
    </div>
  );
}

// =====================================
// PROBABILITY BREAKDOWN
// =====================================

function ProbabilityBreakdown({ probabilities }) {
  if (!probabilities) return null;

  const labels = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    pituitary: "Pituitary",
    notumor: "No Tumor"
  };

  return (
    <div style={{ marginTop: 24 }}>
      <h3>Class-wise Probability</h3>

      {Object.entries(labels).map(([key, label]) => {
        const value = Number(probabilities[key] || 0);

        return (
          <div key={key} style={{ marginBottom: 15 }}>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 6
              }}
            >
              <span>{label}</span>
              <strong>{(value * 100).toFixed(2)}%</strong>
            </div>

            <div
              style={{
                height: 9,
                background: "#262626",
                borderRadius: 20,
                overflow: "hidden"
              }}
            >
              <div
                style={{
                  width: `${value * 100}%`,
                  height: "100%",
                  background: "#00FF88",
                  borderRadius: 20
                }}
              />
            </div>

          </div>
        );
      })}
    </div>
  );
}

// =====================================
// GRAD-CAM VIEWER
// =====================================

function GradCAMViewer({ result }) {
  const imageStyle = {
    width: "100%",
    height: 260,
    objectFit: "contain",
    background: "#111111",
    borderRadius: 12,
    display: "block"
  };

  return (
    <div className="card" style={{ marginTop: 24 }}>

      <h2>3. Grad-CAM Visualization</h2>

      <p className="muted">
        Comparison of the original MRI scan and AI-generated
        Grad-CAM visualization.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 20,
          marginTop: 20
        }}
      >

        <div>
          <h3>Original MRI</h3>

          <img
            src={`${API_BASE}${result.image_url}`}
            alt="Original MRI scan"
            style={imageStyle}
          />
        </div>

        <div>
          <h3>Grad-CAM Heatmap</h3>

          {result.gradcam_url ? (
            <img
              src={`${API_BASE}${result.gradcam_url}`}
              alt="Grad-CAM visualization"
              style={imageStyle}
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          ) : (
            <div
              style={{
                ...imageStyle,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white"
              }}
            >
              Grad-CAM unavailable
            </div>
          )}
        </div>

      </div>

      {result.gradcam_error && (
        <p className="note">
          Grad-CAM error: {result.gradcam_error}
        </p>
      )}

    </div>
  );
}

// =====================================
// ANALYZE MRI
// =====================================

function Analyze({ refresh }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [sampleName, setSampleName] = useState(null);
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);

  const input = useRef();

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch(() => {});
  }, []);

  const pick = (f) => {
    if (!f) return;

    if (!f.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, BMP or WEBP image.");
      return;
    }

    setError("");
    setResult(null);
    setSampleName(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const run = async (fn) => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const [r] = await Promise.all([
        fn(),
        new Promise((resolve) => setTimeout(resolve, 1000))
      ]);

      setResult(r);
      refresh();

    } catch (e) {
      setError(e.message);
    }

    setLoading(false);
  };

  const trySample = (s) => {
    setFile(null);
    setSampleName(s.name);
    setPreview(s.url);

    run(() => predictSample(s.name));
  };

  const tumor = result?.prediction === "Tumor Detected";

  const canAnalyze = (file || sampleName) && !loading;

  return (
    <section>
      <h1>Analyze MRI</h1>

      <div className="split">

        {/* UPLOAD SECTION */}

        <div className="card">
          <h2>1. Upload MRI</h2>

          <div
            className={`drop ${drag ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              pick(e.dataTransfer.files[0]);
            }}
          >
            {preview ? (
              <div className={`preview ${loading ? "scanning" : ""}`}>
                <img src={preview} alt="MRI preview" />

                {loading && <div className="beam" />}
              </div>
            ) : (
              <p>Drag and drop an MRI image here</p>
            )}
          </div>

          <input
            ref={input}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => pick(e.target.files[0])}
          />

          <div className="actions">

            <button
              className="btn"
              onClick={() => input.current.click()}
            >
              Browse image
            </button>

            <button
              className="btn primary"
              disabled={!canAnalyze}
              onClick={() =>
                run(() =>
                  file
                    ? predictFile(file)
                    : predictSample(sampleName)
                )
              }
            >
              {loading ? "Analyzing..." : "Analyze MRI"}
            </button>

          </div>

          {error && (
            <p className="err" role="alert">
              {error}
            </p>
          )}

          <h2 className="mt">Try Sample MRI</h2>

          {samples.length ? (
            <div className="samples">
              {samples.map((s, i) => (
                <button
                  key={s.name}
                  className="sample"
                  onClick={() => trySample(s)}
                  disabled={loading}
                >
                  <img src={s.url} alt={`Sample ${i + 1}`} />
                  <span>Sample {i + 1}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="muted">
              No samples available.
            </p>
          )}

        </div>

        {/* RESULT SECTION */}

        <div className="card result">
          <h2>2. AI Analysis Result</h2>

          {!result && !loading && (
            <p className="muted">
              Upload or select an MRI sample to begin.
            </p>
          )}

          {loading && (
            <p className="pulse">
              Running CNN analysis...
            </p>
          )}

          {result && (
            <>
              <div className={`verdict ${tumor ? "red" : ""}`}>
                {result.prediction}
              </div>

              <p className="score">
                Model confidence:
                <b> {(result.confidence * 100).toFixed(2)}%</b>
              </p>

              <Confidence
                value={result.confidence}
                tumor={tumor}
              />

              <div className="bar-labels">
                <span>0%</span>
                <span>Confidence</span>
                <span>100%</span>
              </div>

              <div style={{ marginTop: 22 }}>
                <h3>Tumor Probability</h3>

                <h2>
                  {(result.tumor_probability * 100).toFixed(2)}%
                </h2>
              </div>

              <ProbabilityBreakdown
                probabilities={result.probabilities}
              />

              <p className="muted">
                {result.scan_id} - Analysis completed
              </p>

              <a
                className="btn primary"
                href={reportUrl(result.id)}
              >
                Download PDF Report
              </a>

              <p className="note">
                AI research prototype. Not a medical diagnosis.
                Consult a qualified medical professional.
              </p>
            </>
          )}

        </div>
      </div>

      {result && (
        <GradCAMViewer result={result} />
      )}

    </section>
  );
}

// =====================================
// SCAN HISTORY
// =====================================

function History({ rows, compact }) {
  return (
    <div className="card tablewrap">

      {!compact && (
        <h1 className="inner">Scan History</h1>
      )}

      {rows.length === 0 ? (
        <p className="muted">
          No scans yet.
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Scan ID</th>
              <th>Prediction</th>
              <th>Score</th>
              <th>Date</th>
              <th />
            </tr>
          </thead>

          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>

                <td>{r.scan_id}</td>

                <td
                  className={
                    r.prediction === "Tumor Detected"
                      ? "red"
                      : "green"
                  }
                >
                  {r.prediction}
                </td>

                <td>
                  {(r.confidence * 100).toFixed(0)}%
                </td>

                <td>{fmtDate(r.created_at)}</td>

                <td>
                  <a href={reportUrl(r.id)}>Report</a>
                </td>

              </tr>
            ))}
          </tbody>
        </table>
      )}

    </div>
  );
}

// =====================================
// PERFORMANCE COMPARISON
// =====================================

function MetricComparisonChart({ metrics }) {
  const classes = [
    { key: "glioma", label: "Glioma" },
    { key: "meningioma", label: "Meningioma" },
    { key: "notumor", label: "No Tumor" },
    { key: "pituitary", label: "Pituitary" }
  ];

  const metricTypes = [
    { key: "precision", label: "Precision" },
    { key: "recall", label: "Recall" },
    { key: "f1-score", label: "F1 Score" }
  ];

  const report = metrics?.classification_report;

  if (!report) return null;

  return (
    <div className="card" style={{ marginBottom: 22 }}>

      <h2>Class-wise Performance Comparison</h2>

      <p className="muted">
        Comparison of Precision, Recall and F1-score
        across all four classification classes.
      </p>

      {classes.map((item) => {
        const data = report[item.key];

        if (!data) return null;

        return (
          <div key={item.key} style={{ marginTop: 24 }}>

            <h3>{item.label}</h3>

            {metricTypes.map((metric) => {
              const value = Number(data[metric.key] || 0);

              return (
                <div key={metric.key} style={{ marginBottom: 14 }}>

                  <div className="row">
                    <span>{metric.label}</span>

                    <strong>
                      {(value * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div
                    style={{
                      height: 11,
                      background: "#262626",
                      borderRadius: 20,
                      overflow: "hidden",
                      marginTop: 7
                    }}
                  >
                    <div
                      style={{
                        width: `${value * 100}%`,
                        height: "100%",
                        background: "#00FF88",
                        borderRadius: 20,
                        transition: "width 0.5s ease"
                      }}
                    />
                  </div>

                </div>
              );
            })}

          </div>
        );
      })}

    </div>
  );
}

// =====================================
// MODEL ANALYTICS
// =====================================

function ModelAnalytics({ health }) {
  const model = health?.model;
  const metrics = model?.metrics;

  if (!metrics) {
    return (
      <section>
        <h1>Model Analytics</h1>

        <div className="card">
          <p className="muted">
            Model evaluation metrics are not available.
            Please check backend connection.
          </p>
        </div>
      </section>
    );
  }

  const percentage = (value) =>
    `${(Number(value) * 100).toFixed(2)}%`;

  const classes = [
    { key: "glioma", label: "Glioma" },
    { key: "meningioma", label: "Meningioma" },
    { key: "notumor", label: "No Tumor" },
    { key: "pituitary", label: "Pituitary" }
  ];

  const matrix = metrics.confusion_matrix || [];

  const maxValue = Math.max(
    ...matrix.flat().map(Number),
    1
  );

  const summaryCards = [
    {
      label: "Accuracy",
      value: percentage(metrics.accuracy)
    },
    {
      label: "Precision",
      value: percentage(metrics.precision)
    },
    {
      label: "Recall",
      value: percentage(metrics.recall)
    },
    {
      label: "F1 Score",
      value: percentage(metrics.f1_score)
    }
  ];

  return (
    <section>
      <h1>Model Analytics</h1>

      <p className="muted" style={{ marginBottom: 24 }}>
        Performance evaluation of your trained CNN model.
      </p>

      {/* MODEL INFORMATION */}

      <div className="card" style={{ marginBottom: 22 }}>

        <div className="row">
          <h2>Model Information</h2>

          <span className="green">
            ● {health.model_mode === "trained"
              ? "Trained Model"
              : "Demo Model"}
          </span>
        </div>

        <div className="grid4">

          <div>
            <span className="muted">Architecture</span>
            <h3>{model.name || "Custom CNN"}</h3>
          </div>

          <div>
            <span className="muted">Input Size</span>
            <h3>{model.input || "224x224 RGB"}</h3>
          </div>

          <div>
            <span className="muted">Output</span>
            <h3>4-Class Softmax</h3>
          </div>

          <div>
            <span className="muted">Classification</span>
            <h3>Multi-Class</h3>
          </div>

        </div>
      </div>

      {/* OVERALL PERFORMANCE */}

      <h2>Overall Performance</h2>

      <div className="grid4">
        {summaryCards.map((item) => (
          <div className="card stat" key={item.label}>
            <b>{item.value}</b>
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {/* CLASS-WISE PERFORMANCE */}

      <div className="card" style={{ marginBottom: 22 }}>

        <h2>Class-wise Performance</h2>

        <p className="muted">
          Precision, Recall and F1-score for all four classes.
        </p>

        <div style={{ overflowX: "auto" }}>
          <table>

            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Support</th>
              </tr>
            </thead>

            <tbody>
              {classes.map((item) => {
                const data =
                  metrics.classification_report?.[item.key];

                if (!data) return null;

                return (
                  <tr key={item.key}>
                    <td>{item.label}</td>
                    <td>{percentage(data.precision)}</td>
                    <td>{percentage(data.recall)}</td>
                    <td>{percentage(data["f1-score"])}</td>
                    <td>{data.support}</td>
                  </tr>
                );
              })}
            </tbody>

          </table>
        </div>
      </div>

      {/* PERFORMANCE COMPARISON */}

      <MetricComparisonChart metrics={metrics} />

      {/* CONFUSION MATRIX */}

      <div className="card" style={{ marginBottom: 22 }}>

        <h2>Confusion Matrix</h2>

        <p className="muted">
          Actual classes vs predicted classes.
          Higher diagonal values indicate correct predictions.
        </p>

        <div style={{ overflowX: "auto", marginTop: 20 }}>

          <table
            style={{
              textAlign: "center",
              minWidth: 480
            }}
          >

            <thead>
              <tr>
                <th>Actual ↓ / Predicted →</th>

                {classes.map((item) => (
                  <th key={item.key}>{item.label}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>

                  <th>{classes[i]?.label}</th>

                  {row.map((value, j) => {
                    const intensity = Number(value) / maxValue;
                    const diagonal = i === j;

                    return (
                      <td
                        key={j}
                        style={{
                          background: diagonal
                            ? `rgba(0,255,136,${0.12 + intensity * 0.65})`
                            : `rgba(255,77,94,${intensity * 0.35})`,
                          color: diagonal ? "#00FF88" : "#ffffff",
                          fontWeight: 700,
                          borderRadius: 5,
                          textAlign: "center"
                        }}
                      >
                        {value}
                      </td>
                    );
                  })}

                </tr>
              ))}
            </tbody>

          </table>
        </div>

        <div className="note">
          Green diagonal cells represent correctly classified
          samples. Off-diagonal cells represent misclassifications.
        </div>

      </div>

      {/* PERFORMANCE INSIGHTS */}

      <div className="card">

        <h2>Performance Insights</h2>

        {classes.map((item) => {
          const data =
            metrics.classification_report?.[item.key];

          if (!data) return null;

          return (
            <div key={item.key} style={{ marginBottom: 16 }}>

              <div className="row">
                <strong>{item.label}</strong>

                <span className="muted">
                  Recall: {percentage(data.recall)}
                </span>
              </div>

              <div className="bar">
                <div
                  className="fill"
                  style={{
                    width: percentage(data.recall)
                  }}
                />
              </div>

            </div>
          );
        })}

        <p className="note">
          These metrics represent evaluation results on the
          model's test dataset. They do not establish clinical
          diagnostic accuracy.
        </p>

      </div>

    </section>
  );
}

// =====================================
// MAIN APPLICATION
// =====================================

export default function App() {

  const [started, setStarted] = useState(false);

  const [tab, setTab] = useState("dashboard");

  const [stats, setStats] = useState({
    total: 0,
    tumor: 0,
    normal: 0
  });

  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);

  const refresh = useCallback(() => {
    getStats().then(setStats).catch(() => {});
    getHistory().then(setHistory).catch(() => {});
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(refresh, [refresh]);

  // WELCOME SCREEN

  if (!started) {
    return (
      <WelcomeScreen
        onStart={() => setStarted(true)}
      />
    );
  }

  // MAIN APPLICATION

  return (
    <div className="app">

      <aside>

        <div className="brand">
          NeuroScan <em>AI</em>
        </div>

        {[
          ["dashboard", "Dashboard"],
          ["analyze", "Analyze MRI"],
          ["history", "Scan History"],
          ["analytics", "Model Analytics"]
        ].map(([k, l]) => (
          <button
            key={k}
            className={`nav ${tab === k ? "active" : ""}`}
            onClick={() => setTab(k)}
          >
            {l}
          </button>
        ))}

        <StatusPill health={health} />

      </aside>

      <main>

        {tab === "dashboard" && (
          <Dashboard
            stats={stats}
            history={history}
            health={health}
            go={setTab}
          />
        )}

        {tab === "analyze" && (
          <Analyze refresh={refresh} />
        )}

        {tab === "history" && (
          <section>
            <History rows={history} />
          </section>
        )}

        {tab === "analytics" && (
          <ModelAnalytics health={health} />
        )}

      </main>

    </div>
  );
}