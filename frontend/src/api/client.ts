import type { BeamParameters, SearchProgress, SearchResult, TargetProperties } from "../types";

export type SearchRequest = { targets: TargetProperties; beam: BeamParameters };

const BASE = "";

export async function createSearch(req: SearchRequest): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getProgress(jobId: string): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}/progress`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getResult(jobId: string): Promise<SearchResult> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function cancelSearch(jobId: string): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}/cancel`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
