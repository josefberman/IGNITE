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
  early_stop_on_mape: boolean;
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

export type BeamTreeNode = {
  id: string;
  parent_id?: string | null;
  generation: number;
  canonical_smiles: string;
  mape: number;
  score_total?: number | null;
  reaction_family?: string | null;
  planner?: string | null;
  in_beam: boolean;
};

export type BeamSearchTree = {
  nodes: BeamTreeNode[];
  beam_layers: string[][];
  best_id?: string | null;
  max_generation?: number;
  iterations_completed?: number;
};

export type SearchResult = {
  job_id: string;
  status: string;
  best?: CandidateMolecule;
  top_k: CandidateMolecule[];
  progress: SearchProgress;
  beam_tree?: BeamSearchTree | null;
  disclaimer: string;
};
