export type TargetProperties = {
  density: number;
  det_velocity: number;
  det_pressure: number;
  hf_solid: number;
};

export type BeamParameters = {
  depth: number;
  width: number;
  top_k: number;
  mape_threshold: number;
  random_seed: number;
  enable_literature: boolean;
  enable_patent_search: boolean;
  novelty_as_gate: boolean;
};

export type SearchProgress = {
  job_id: string;
  status: string;
  iteration: number;
  max_iterations: number;
  beam_size: number;
  candidates_generated: number;
  rejected_count: number;
  best_mape?: number;
  best_score?: number;
  message: string;
};

export type CandidateMolecule = {
  id: string;
  smiles: string;
  canonical_smiles: string;
  inchikey?: string;
  generation: number;
  mape: number;
  properties: Record<
    string,
    { value: number; source: string; confidence?: number; lower?: number; upper?: number; citation?: string }
  >;
  score?: {
    mape: number;
    total: number;
    mape_normalized: number;
    sascore_penalty: number;
    minimal_edit_penalty: number;
    weights?: Record<string, number>;
  };
  modification?: { reaction_family: string; edit_distance: number; planner: string };
  lineage: string[];
};

export type SearchResult = {
  job_id: string;
  status: string;
  best?: CandidateMolecule;
  top_k: CandidateMolecule[];
  progress: SearchProgress;
  disclaimer: string;
};
