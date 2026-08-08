import { useEffect, useMemo, useState } from "react";
import { cancelSearch, createSearch, getProgress, getResult } from "./api/client";
import { BeamTreeView } from "./BeamTreeView";
import type {
  BeamParameters,
  CandidateMolecule,
  SearchProgress,
  SearchResult,
  TargetProperties,
} from "./types";

const DEFAULT_TARGETS: TargetProperties = {
  density: 1.8,
  det_velocity: 8500,
  det_pressure: 30,
  hf_solid: 100,
};

const DEFAULT_BEAM: BeamParameters = {
  depth: 10,
  width: 5,
  top_k: 3,
  mape_threshold: 1,
  random_seed: 42,
  enable_literature: true,
  enable_patent_search: true,
  novelty_as_gate: false,
  early_stop_on_mape: false,
};

const PROPERTY_CHANNELS = [
  {
    key: "density" as const,
    label: "ρ",
    name: "Density",
    unit: "g·cm⁻³",
    title: "Crystal bulk density",
    shortUnit: "g/cm³",
  },
  {
    key: "det_velocity" as const,
    label: "D",
    name: "Det. velocity",
    unit: "m·s⁻¹",
    title: "Chapman–Jouguet detonation velocity",
    shortUnit: "m/s",
  },
  {
    key: "det_pressure" as const,
    label: "P",
    name: "Det. pressure",
    unit: "GPa",
    title: "CJ detonation pressure",
    shortUnit: "GPa",
  },
  {
    key: "hf_solid" as const,
    label: "ΔHf",
    name: "Heat of formation",
    unit: "kJ·mol⁻¹",
    title: "Solid-phase standard enthalpy of formation",
    shortUnit: "kJ/mol",
  },
] as const;

const PROPERTY_LABELS: Record<keyof TargetProperties, string> = {
  density: "Density",
  det_velocity: "Detonation velocity",
  det_pressure: "Detonation pressure",
  hf_solid: "Heat of formation (solid)",
};

const PROPERTY_UNITS: Record<keyof TargetProperties, string> = {
  density: "g·cm⁻³",
  det_velocity: "m·s⁻¹",
  det_pressure: "GPa",
  hf_solid: "kJ·mol⁻¹",
};

function formatTargetValue(key: keyof TargetProperties, value: number): string {
  if (key === "density") return value.toFixed(2);
  if (key === "det_pressure") return value.toFixed(1);
  return String(Math.round(value));
}

function AppHeader() {
  return (
    <header className="app-header">
      <div className="wordmark">
        <span className="wordmark-symbol" aria-hidden>⌬</span>
        <span>IGNITE</span>
      </div>
      <div className="header-context">
        <span>Energetic molecular design</span>
        <span className="header-rule" />
        <span>Workspace 01</span>
      </div>
      <span className="preview-label">Research preview</span>
    </header>
  );
}

function TargetProfile({ targets }: { targets: TargetProperties }) {
  return (
    <section className="objective" aria-label="Target property profile">
      <div className="objective-heading">
        <p className="eyebrow">Objective vector</p>
        <p>Four target properties define the search direction.</p>
      </div>
      <div className="objective-vector">
        {PROPERTY_CHANNELS.map((ch, index) => (
          <div className="objective-value" key={ch.key} title={ch.title}>
            <div className="objective-index">{String(index + 1).padStart(2, "0")}</div>
            <div>
              <div className="objective-symbol">{ch.label}</div>
              <div className="objective-number">
                {formatTargetValue(ch.key, targets[ch.key])}
                <span>{ch.shortUnit}</span>
              </div>
              <div className="objective-name">{ch.name}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ScoringReference() {
  return (
    <details className="method-note">
      <summary>Scoring methodology</summary>
      <div className="method-content">
        <p>
          Each valid structure receives a total score (lower is better). The beam keeps the top K survivors
          each iteration; the lowest total is returned as the best candidate.
        </p>
        <h3>Property MAPE — weight 0.65</h3>
        <p>Mean absolute percentage error across all four targets (equal weights 0.25 each).</p>
        <pre>MAPE% = average_i( |pred_i − target_i| / |target_i| × 100 )</pre>
        <h3>Synthetic accessibility — weight 0.25</h3>
        <p>RDKit SAScore mapped to [0, 1]. Lower synthesis difficulty means less penalty.</p>
        <h3>Minimal edit — weight 0.10</h3>
        <p>
          Prefers simpler modifications: <code>edit = min(1, atom_count_delta / 10)</code>
        </p>
        <h3>Combined total</h3>
        <pre>
          {`total = 0.65 × mape_norm + 0.25 × sascore + 0.10 × edit`}
        </pre>
      </div>
    </details>
  );
}

function NumberField({
  id,
  label,
  unit,
  hint,
  value,
  step,
  onChange,
}: {
  id: string;
  label: string;
  unit: string;
  hint?: string;
  value: number;
  step?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="control">
      <label htmlFor={id}>{label}</label>
      {hint ? <span className="control-hint">{hint}</span> : null}
      <div className="control-input">
        <input
          id={id}
          type="number"
          step={step}
          value={value}
          onChange={(e) => onChange(+e.target.value)}
        />
        <span className="input-unit">{unit}</span>
      </div>
    </div>
  );
}

function ToggleField({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  description: string;
}) {
  return (
    <label className="option">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="option-switch" aria-hidden />
      <span className="option-copy">
        <span className="option-label">{label}</span>
        <span className="option-description">{description}</span>
      </span>
    </label>
  );
}

function StructureImage({ smiles, size = 200 }: { smiles: string; size?: number }) {
  const src = `/api/v1/molecule/depict?smiles=${encodeURIComponent(smiles)}&size=${size}`;
  return (
    <div className="molecule-figure">
      <img src={src} alt="" width={size} height={size} loading="lazy" />
    </div>
  );
}

function plannerBadge(planner?: string, generation?: number): { label: string; cls: string } | null {
  if (generation === 0) return { label: "Seed", cls: "tag tag--seed" };
  if (planner === "learned") return { label: "Learned", cls: "tag tag--learned" };
  if (planner === "rule_based") return { label: "Rule-based", cls: "tag" };
  return null;
}

function MoleculeCard({
  title,
  mol,
  rank,
  featured,
}: {
  title: string;
  mol: CandidateMolecule;
  rank?: number;
  featured?: boolean;
}) {
  const props = mol.properties || {};
  const badge = plannerBadge(mol.modification?.planner, mol.generation);

  return (
    <article className={`candidate${featured ? " candidate--featured" : ""}`}>
      <header className="candidate-heading">
        <div>
          <span className="candidate-rank">{rank != null ? `Candidate ${String(rank).padStart(2, "0")}` : "Search result"}</span>
          <h3>{title}</h3>
        </div>
        <div className="candidate-summary">
          <span><small>MAPE</small>{mol.mape?.toFixed(2)}%</span>
          {mol.score && <span><small>Score</small>{mol.score.total.toFixed(3)}</span>}
          {badge && <span className={badge.cls}>{badge.label}</span>}
        </div>
      </header>
      <div className="candidate-content">
        <StructureImage smiles={mol.canonical_smiles} />
        <div className="candidate-data">
          <code className="smiles">{mol.canonical_smiles}</code>
          {mol.inchikey ? <p className="meta">InChIKey · {mol.inchikey}</p> : null}
          <div className="tag-row">
            {mol.modification && <span className="tag">{mol.modification.reaction_family}</span>}
            <span className="tag">Generation {mol.generation}</span>
            {Object.values(props)[0] && <span className="tag">{Object.values(props)[0].source}</span>}
          </div>
          <div className="property-table-wrap">
            <table className="property-table">
              <thead>
                <tr>
                  <th>Property</th>
                  <th>Estimate</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(props).map(([k, v]) => (
                  <tr key={k}>
                    <td>{PROPERTY_LABELS[k as keyof TargetProperties] ?? k.replace(/_/g, " ")}</td>
                    <td className="numeric">{v.value.toFixed(3)}</td>
                    <td>{v.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {mol.score && (
            <p className="score-detail">
              MAPE {mol.score.mape.toFixed(2)}% · SA {mol.score.sascore_penalty.toFixed(3)} · edit{" "}
              {mol.score.minimal_edit_penalty.toFixed(3)} · <strong>total {mol.score.total.toFixed(3)}</strong>
            </p>
          )}
        </div>
      </div>
      {mol.lineage?.length ? (
        <footer className="lineage">
          <p className="eyebrow">Modification lineage</p>
          <div className="lineage-strip">
            {mol.lineage.map((s) => (
              <StructureImage key={s} smiles={s} size={96} />
            ))}
            <StructureImage smiles={mol.canonical_smiles} size={96} />
          </div>
        </footer>
      ) : null}
    </article>
  );
}

function statusClass(status: string): string {
  if (status === "running" || status === "queued") return "run-status run-status--active";
  if (status === "completed") return "run-status run-status--done";
  if (status === "failed" || status === "cancelled") return "run-status run-status--error";
  return "run-status";
}

function RunMonitor({
  progress,
  result,
}: {
  progress: SearchProgress | null;
  result: SearchResult | null;
}) {
  if (!progress) {
    return (
      <div className="search-empty">
        <div className="empty-orbit" aria-hidden>
          <span /><span /><span />
        </div>
        <p className="eyebrow">Search trajectory</p>
        <h2>Ready to explore molecular space.</h2>
        <p>Set the experiment brief, then start the search. Accepted transformations and beam decisions will appear here.</p>
      </div>
    );
  }

  const pct =
    progress.max_iterations > 0
      ? Math.min(100, (progress.iteration / progress.max_iterations) * 100)
      : 0;

  return (
    <div className="run-monitor">
      <div className="run-heading">
        <span className={statusClass(progress.status)}>{progress.status}</span>
        <span className="run-id">Run {progress.job_id.slice(0, 8)}</span>
      </div>

      <div className="iteration-progress">
        <div className="iteration-label">
          <span>Beam iterations</span>
          <span className="numeric">
            {progress.iteration} / {progress.max_iterations}
          </span>
        </div>
        <div
          className="iteration-track"
          role="progressbar"
          aria-valuenow={progress.iteration}
          aria-valuemin={0}
          aria-valuemax={progress.max_iterations}
        >
          <div className="iteration-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <dl className="run-metrics">
        <div>
          <dt>Beam size</dt>
          <dd>{progress.beam_size}</dd>
        </div>
        <div>
          <dt>Generated</dt>
          <dd>{progress.candidates_generated}</dd>
        </div>
        <div>
          <dt>Rejected</dt>
          <dd>{progress.rejected_count}</dd>
        </div>
        <div className="metric-accent">
          <dt>Best MAPE</dt>
          <dd>{progress.best_mape?.toFixed(2) ?? "—"}%</dd>
        </div>
        <div>
          <dt>Best score</dt>
          <dd>{progress.best_score?.toFixed(4) ?? "—"}</dd>
        </div>
      </dl>

      {progress.message ? <p className="run-log">{progress.message}</p> : null}

      {result?.best && (
        <div className="run-result">
          <MoleculeCard title="Best candidate" mol={result.best} featured />
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [targets, setTargets] = useState(DEFAULT_TARGETS);
  const [beam, setBeam] = useState(DEFAULT_BEAM);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<SearchProgress | null>(null);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const setT = (k: keyof TargetProperties, v: number) => setTargets((t) => ({ ...t, [k]: v }));
  const setB = <K extends keyof BeamParameters>(k: K, v: BeamParameters[K]) => setBeam((b) => ({ ...b, [k]: v }));

  useEffect(() => {
    if (!jobId || !running) return;
    const id = setInterval(async () => {
      try {
        const p = await getProgress(jobId);
        setProgress(p);
        if (["completed", "failed", "cancelled"].includes(p.status)) {
          const r = await getResult(jobId);
          setResult(r);
          setRunning(false);
        }
      } catch (e: unknown) {
        setError(String((e as Error).message || e));
        setRunning(false);
      }
    }, 800);
    return () => clearInterval(id);
  }, [jobId, running]);

  const onStart = async () => {
    setError(null);
    setResult(null);
    setProgress(null);
    setRunning(true);
    try {
      const p = await createSearch({ targets, beam });
      setJobId(p.job_id);
      setProgress(p);
    } catch (e: unknown) {
      setError(String((e as Error).message || e));
      setRunning(false);
    }
  };

  const onCancel = async () => {
    if (!jobId) return;
    await cancelSearch(jobId);
  };

  const exportHref = useMemo(() => (jobId ? `/api/v1/search/${jobId}/export.csv` : "#"), [jobId]);

  return (
    <>
      <AppHeader />
      <main className="workstation">
        <aside className="brief">
          <header className="brief-heading">
            <p className="eyebrow">Experiment brief</p>
            <h1>Define the search.</h1>
            <p>Specify the desired energetic profile and the bounds of molecular exploration.</p>
          </header>

          <section className="brief-section">
            <div className="brief-section-heading">
              <span>01</span>
              <h2>Target properties</h2>
            </div>
            <div className="controls-grid">
                {(Object.keys(PROPERTY_LABELS) as (keyof TargetProperties)[]).map((key) => (
                  <NumberField
                    key={key}
                    id={key}
                    label={PROPERTY_LABELS[key]}
                    unit={PROPERTY_UNITS[key]}
                    value={targets[key]}
                    step={key === "density" ? "0.01" : key === "det_pressure" ? "0.1" : undefined}
                    onChange={(v) => setT(key, v)}
                  />
                ))}
            </div>
          </section>

          <section className="brief-section">
            <div className="brief-section-heading">
              <span>02</span>
              <h2>Search bounds</h2>
            </div>
            <div className="controls-grid">
                <NumberField
                  id="depth"
                  label="Depth"
                  unit="iterations"
                  hint="Number of expansion rounds"
                  value={beam.depth}
                  onChange={(v) => setB("depth", v)}
                />
                <NumberField
                  id="width"
                  label="Width"
                  unit="per node"
                  hint="Children generated per parent"
                  value={beam.width}
                  onChange={(v) => setB("width", v)}
                />
                <NumberField
                  id="top_k"
                  label="Top K"
                  unit="survivors"
                  hint="Candidates kept each round"
                  value={beam.top_k}
                  onChange={(v) => setB("top_k", v)}
                />
                <NumberField
                  id="mape_threshold"
                  label="MAPE threshold"
                  unit="%"
                  hint="Explore until best MAPE is below"
                  value={beam.mape_threshold}
                  step="0.1"
                  onChange={(v) => setB("mape_threshold", v)}
                />
            </div>
            <div className="options">
                <ToggleField
                  checked={beam.enable_literature}
                  onChange={(v) => setB("enable_literature", v)}
                  label="Literature properties"
                  description="Prefer curated literature values over model predictions"
                />
                <ToggleField
                  checked={beam.novelty_as_gate}
                  onChange={(v) => setB("novelty_as_gate", v)}
                  label="Novelty gate"
                  description="Reject structures with known literature or patent matches"
                />
                <ToggleField
                  checked={beam.early_stop_on_mape}
                  onChange={(v) => setB("early_stop_on_mape", v)}
                  label="Stop at MAPE threshold"
                  description="End search early once the best candidate meets the threshold"
                />
            </div>
          </section>

          <ScoringReference />

          <div className="brief-actions">
              <button type="button" className="action action--primary" onClick={onStart} disabled={running}>
                {running ? "Searching…" : "Start search"}
              </button>
              <button type="button" className="action action--secondary" onClick={onCancel} disabled={!running}>
                Cancel
              </button>
              {jobId && result?.status === "completed" && (
                <a className="action action--link" href={exportHref}>
                  Export CSV
                </a>
              )}
          </div>
          {error && <div className="error-message">{error}</div>}
        </aside>

        <div className="analysis">
          <header className="analysis-heading">
            <div>
              <p className="eyebrow">Target-directed search</p>
              <h1>Molecular search canvas</h1>
            </div>
            <p>Seed selection → learned modification → property estimation → beam ranking</p>
          </header>

          <TargetProfile targets={targets} />
          <section className="trajectory">
            <RunMonitor progress={progress} result={result} />
          </section>

        {result?.beam_tree?.nodes?.length ? (
          <section className="analysis-section">
            <header className="analysis-section-heading">
              <p className="eyebrow">Search topology</p>
              <h2>Reaction tree</h2>
            </header>
            <p className="prose">
              Each column is one modification generation. Depth controls iteration count, not guaranteed
              columns — later rounds may add no new structures once edits are exhausted.
              {result.beam_tree.max_generation != null && result.beam_tree.iterations_completed != null ? (
                <>
                  {" "}
                  This run: {result.beam_tree.iterations_completed}/{beam.depth} iterations, generation{" "}
                  {result.beam_tree.max_generation} from seed.
                </>
              ) : null}
            </p>
            <BeamTreeView tree={result.beam_tree} />
          </section>
        ) : null}

        {result?.top_k?.length ? (
          <section className="analysis-section">
            <header className="analysis-section-heading">
              <p className="eyebrow">Final survivors</p>
              <h2>Ranked candidates</h2>
            </header>
            <div className="candidates">
              {result.top_k.map((m, i) => (
                <MoleculeCard key={m.id} title={`Candidate ${i + 1}`} mol={m} rank={i + 1} />
              ))}
            </div>
          </section>
        ) : null}
        </div>
      </main>
    </>
  );
}
