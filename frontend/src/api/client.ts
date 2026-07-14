import type { BeamParameters, SearchProgress, SearchResult, TargetProperties } from "../types";

export type SearchRequest = { targets: TargetProperties; beam: BeamParameters };

const BASE = "";

async function raiseForBadResponse(r: Response): Promise<never> {
  const text = await r.text();
  let detail = text;
  try {
    const parsed = JSON.parse(text);
    if (parsed?.detail) {
      detail = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    }
  } catch {
    // keep raw text
  }
  if (r.status === 400 && /bad request/i.test(text) && !text.includes("detail")) {
    throw new Error(
      `${r.status} ${r.statusText || "Bad Request"} — is the IGNITE API running on port 8080? ` +
        `(port 8000 is often the HPLIP printer UI). Start with: uvicorn backend.app.main:app --port 8080`
    );
  }
  throw new Error(`${r.status}: ${detail || r.statusText}`);
}

export async function createSearch(req: SearchRequest): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) await raiseForBadResponse(r);
  return r.json();
}

export async function getProgress(jobId: string): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}/progress`);
  if (!r.ok) await raiseForBadResponse(r);
  return r.json();
}

export async function getResult(jobId: string): Promise<SearchResult> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}`);
  if (!r.ok) await raiseForBadResponse(r);
  return r.json();
}

export async function cancelSearch(jobId: string): Promise<SearchProgress> {
  const r = await fetch(`${BASE}/api/v1/search/${jobId}/cancel`, { method: "POST" });
  if (!r.ok) await raiseForBadResponse(r);
  return r.json();
}
