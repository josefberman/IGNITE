import { useEffect, useMemo, useState } from "react";
import { cancelSearch, createSearch, getProgress, getResult } from "./api/client";
import type { BeamParameters, CandidateMolecule, SearchProgress, SearchResult, TargetProperties } from "./types";

const DEFAULT_TARGETS: TargetProperties = {
  density: 1.8,
  det_velocity: 8500,
  det_pressure: 30,
  hf_solid: 100,
};

const DEFAULT_BEAM: BeamParameters = {
  depth: 3,
  width: 4,
  top_k: 3,
  mape_threshold: 8,
  random_seed: 42,
  enable_literature: true,
  enable_patent_search: true,
  novelty_as_gate: false,
};

function MoleculeCard({ title, mol }: { title: string; mol: CandidateMolecule }) {
  const props = mol.properties || {};
  return (
    <div className="mol">
      <h3>{title}</h3>
      <div><code>{mol.canonical_smiles}</code></div>
      <div className="badges">
        <span>MAPE {mol.mape?.toFixed(2)}%</span>
        {mol.score && <span>Score {mol.score.total.toFixed(3)}</span>}
        {mol.novelty && <span>Novelty {mol.novelty.status}</span>}
        {mol.safety && <span>Safety risk {mol.safety.score.toFixed(2)}</span>}
        {mol.modification && <span>{mol.modification.reaction_family}</span>}
      </div>
      <table>
        <thead>
          <tr><th>Property</th><th>Value</th><th>Interval</th><th>Source</th><th>Conf.</th></tr>
        </thead>
        <tbody>
          {Object.entries(props).map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td>{v.value.toFixed(3)}</td>
              <td>{v.lower != null && v.upper != null ? `[${v.lower.toFixed(2)}, ${v.upper.toFixed(2)}]` : "—"}</td>
              <td>{v.source}</td>
              <td>{(v.confidence * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      {mol.score && (
        <p className="progress">
          penalties: SA {mol.score.sascore_penalty.toFixed(2)} · unc {mol.score.uncertainty_penalty.toFixed(2)} ·
          safety {mol.score.safety_penalty.toFixed(2)} · edit {mol.score.minimal_edit_penalty.toFixed(2)}
        </p>
      )}
      {mol.novelty?.notes?.length ? <p className="progress">{mol.novelty.notes.join(" · ")}</p> : null}
      {mol.lineage?.length ? <p className="progress">Lineage: {mol.lineage.join(" → ")} → {mol.canonical_smiles}</p> : null}
      {mol.safety?.disclaimer ? <p className="progress">{mol.safety.disclaimer}</p> : null}
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
        <p>Learned energetic-material generation with beam search, uncertainty, safety, and novelty evidence.</p>
      </header>
      <div className="disclaimer">
        Computational candidates only. Not a safety certification, legal novelty opinion, or patentability advice.
        Unavailable novelty providers are reported as incomplete — never as “novel.”
      </div>
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
              <label>Depth X</label>
              <input type="number" value={beam.depth} onChange={(e) => setB("depth", +e.target.value)} />
            </div>
            <div>
              <label>Width N</label>
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
            <label><input type="checkbox" checked={beam.enable_literature} onChange={(e) => setB("enable_literature", e.target.checked)} /> Literature</label>
            <label><input type="checkbox" checked={beam.enable_patent_search} onChange={(e) => setB("enable_patent_search", e.target.checked)} /> Patents</label>
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
              <div>Uncertainty: {progress.best_uncertainty?.toFixed(3) ?? "—"}</div>
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
