import { useEffect, useMemo, useState } from "react";
import { cancelSearch, createSearch, getProgress, getResult } from "./api/client";
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
};

function StructureImage({ smiles, size = 220 }: { smiles: string; size?: number }) {
  const src = `/api/v1/molecule/depict?smiles=${encodeURIComponent(smiles)}&size=${size}`;
  return (
    <div className="structure">
      <img src={src} alt={`2D structure for ${smiles}`} width={size} height={size} loading="lazy" />
    </div>
  );
}

function ScoringGuide() {
  return (
    <section className="card scoring-guide">
      <h2>How candidate scoring works</h2>
      <p>
        After each beam iteration, every valid molecule receives a <strong>total score</strong> (lower is better).
        Candidates are ranked by this total; the top <em>K</em> survive to the next iteration, and the lowest-scoring
        molecule is returned as the best result.
      </p>

      <h3>1. Property MAPE (weight 0.65)</h3>
      <p>
        Mean Absolute Percentage Error across density, detonation velocity, detonation pressure, and Hf solid
        (equal weights 0.25 each):
      </p>
      <pre>MAPE% = average_i( |pred_i − target_i| / |target_i| × 100 )</pre>
      <p>
        For ranking this is normalized to <code>mape_normalized = min(MAPE% / 100, 1)</code> so 0% error → 0 and
        ≥100% error → 1.
      </p>

      <h3>2. Synthetic accessibility / SAScore penalty (weight 0.25)</h3>
      <p>
        RDKit SAScore (≈1 easy … 10 hard) is mapped to [0, 1]. Lower values mean easier synthesis and contribute less
        penalty.
      </p>

      <h3>3. Minimal-edit penalty (weight 0.10)</h3>
      <p>
        Prefers simpler modifications: <code>edit = min(1, atom_count_delta / 10)</code>. Seeds (no edit) score 0 here.
      </p>

      <h3>Combined total (lower is better)</h3>
      <pre>
{`total =
  0.65 × mape_normalized
+ 0.25 × sascore_penalty
+ 0.10 × minimal_edit_penalty`}
      </pre>
      <p>
        During search (while best MAPE is above the threshold), an explore mode temporarily raises the MAPE weight
        and lightly penalizes unedited dataset seeds so generated children can enter the beam. Final displayed
        rankings use the weights above. Optional novelty gate (user input) can hard-reject known molecules
        without changing the ranking formula.
      </p>
    </section>
  );
}

function MoleculeCard({ title, mol }: { title: string; mol: CandidateMolecule }) {
  const props = mol.properties || {};
  return (
    <div className="mol">
      <h3>{title}</h3>
      <div className="mol-layout">
        <StructureImage smiles={mol.canonical_smiles} />
        <div className="mol-meta">
          <div><code>{mol.canonical_smiles}</code></div>
          {mol.inchikey ? <div className="progress">InChIKey: {mol.inchikey}</div> : null}
          <div className="badges">
            <span>MAPE {mol.mape?.toFixed(2)}%</span>
            {mol.score && <span>Score {mol.score.total.toFixed(3)}</span>}
            {mol.modification && <span>{mol.modification.reaction_family}</span>}
            <span>gen {mol.generation}</span>
            {Object.values(props)[0] && <span>source {Object.values(props)[0].source}</span>}
          </div>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Property</th><th>Value</th><th>Source</th></tr>
        </thead>
        <tbody>
          {Object.entries(props).map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v.value.toFixed(3)}</td>
              <td>{v.source}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {mol.score && (
        <div className="score-breakdown progress">
          <div>Score breakdown (lower better)</div>
          <div>
            MAPE {mol.score.mape.toFixed(2)}% (norm {mol.score.mape_normalized.toFixed(3)}) ·
            SA {mol.score.sascore_penalty.toFixed(3)} ·
            edit {mol.score.minimal_edit_penalty.toFixed(3)} ·
            <strong> total {mol.score.total.toFixed(3)}</strong>
          </div>
        </div>
      )}
      {mol.lineage?.length ? (
        <div className="lineage">
          <p className="progress">Modification lineage</p>
          <div className="lineage-row">
            {mol.lineage.map((s) => (
              <StructureImage key={s} smiles={s} size={120} />
            ))}
            <StructureImage smiles={mol.canonical_smiles} size={120} />
          </div>
          <p className="progress">{[...mol.lineage, mol.canonical_smiles].join(" → ")}</p>
        </div>
      ) : null}
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
      } catch (e: any) {
        setError(String(e.message || e));
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
    } catch (e: any) {
      setError(String(e.message || e));
      setRunning(false);
    }
  };

  const onCancel = async () => {
    if (!jobId) return;
    await cancelSearch(jobId);
  };

  const exportHref = useMemo(() => (jobId ? `/api/v1/search/${jobId}/export.csv` : "#"), [jobId]);

  return (
    <div className="app">
      <header>
        <h1>IGNITE</h1>
        <p>Intelligent Generation and Navigation through Iterative Transformations of Energetics</p>
      </header>
      <div className="disclaimer">
        Computational design candidates only — not experimental validation or a legal novelty / patentability opinion.
      </div>

      <ScoringGuide />

      <div className="grid">
        <section className="card">
          <h2>Target properties</h2>
          <div className="row">
            <div>
              <label>Density (g/cm³)</label>
              <input type="number" step="0.01" value={targets.density} onChange={(e) => setT("density", +e.target.value)} />
            </div>
            <div>
              <label>Detonation velocity (m/s)</label>
              <input type="number" value={targets.det_velocity} onChange={(e) => setT("det_velocity", +e.target.value)} />
            </div>
            <div>
              <label>Detonation pressure (GPa)</label>
              <input type="number" step="0.1" value={targets.det_pressure} onChange={(e) => setT("det_pressure", +e.target.value)} />
            </div>
            <div>
              <label>Hf solid (kJ/mol)</label>
              <input type="number" value={targets.hf_solid} onChange={(e) => setT("hf_solid", +e.target.value)} />
            </div>
          </div>
          <h2 style={{ marginTop: "1.2rem" }}>Beam parameters</h2>
          <div className="row">
            <div>
              <label>Depth</label>
              <input type="number" value={beam.depth} onChange={(e) => setB("depth", +e.target.value)} />
            </div>
            <div>
              <label>Width</label>
              <input type="number" value={beam.width} onChange={(e) => setB("width", +e.target.value)} />
            </div>
            <div>
              <label>Top K</label>
              <input type="number" value={beam.top_k} onChange={(e) => setB("top_k", +e.target.value)} />
            </div>
            <div>
              <label>MAPE threshold (%)</label>
              <input type="number" step="0.1" value={beam.mape_threshold} onChange={(e) => setB("mape_threshold", +e.target.value)} />
            </div>
          </div>
          <div className="checks">
            <label><input type="checkbox" checked={beam.enable_literature} onChange={(e) => setB("enable_literature", e.target.checked)} /> Literature properties</label>
            <label><input type="checkbox" checked={beam.novelty_as_gate} onChange={(e) => setB("novelty_as_gate", e.target.checked)} /> Novelty gate</label>
          </div>
          <button onClick={onStart} disabled={running}>Start search</button>
          <button className="secondary" onClick={onCancel} disabled={!running}>Cancel</button>
          {jobId && result?.status === "completed" && (
            <a href={exportHref} style={{ marginLeft: "0.5rem", color: "#7dd3fc" }}>Export CSV</a>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        <section className="card">
          <h2>Live progress</h2>
          {progress ? (
            <div className="progress">
              <div>Job: {progress.job_id}</div>
              <div>Status: {progress.status}</div>
              <div>Iteration: {progress.iteration}/{progress.max_iterations}</div>
              <div>Beam size: {progress.beam_size}</div>
              <div>Generated: {progress.candidates_generated} · Rejected: {progress.rejected_count}</div>
              <div>Best MAPE: {progress.best_mape?.toFixed(2) ?? "—"}%</div>
              <div>Best score: {progress.best_score?.toFixed(4) ?? "—"}</div>
              <div>{progress.message}</div>
            </div>
          ) : (
            <p className="progress">No active run.</p>
          )}
          {result?.best && <MoleculeCard title="Best molecule" mol={result.best} />}
          {result?.disclaimer && <p className="progress">{result.disclaimer}</p>}
        </section>
      </div>

      {result?.top_k?.length ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h2>Ranked candidates</h2>
          {result.top_k.map((m, i) => (
            <MoleculeCard key={m.id} title={`#${i + 1}`} mol={m} />
          ))}
        </section>
      ) : null}
    </div>
  );
}
