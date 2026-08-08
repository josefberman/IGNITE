import { useMemo, useState } from "react";
import type { BeamSearchTree, BeamTreeNode } from "./types";

const NODE_W = 148;
const NODE_H = 62;
const COL_GAP = 120;
const GAP_X = NODE_W + COL_GAP;
const GAP_Y = 20;
const PAD = 48;

type LayoutNode = BeamTreeNode & { x: number; y: number };

function plannerLabel(planner?: string | null, generation?: number): string {
  if (generation === 0) return "Seed";
  if (planner === "learned") return "Learned";
  if (planner === "rule_based") return "Rule";
  return "—";
}

function buildLayout(nodes: BeamTreeNode[]): LayoutNode[] {
  const byGen = new Map<number, BeamTreeNode[]>();
  for (const n of nodes) {
    const g = byGen.get(n.generation) ?? [];
    g.push(n);
    byGen.set(n.generation, g);
  }

  const parentOrder = new Map<string, number>();
  nodes.forEach((n, i) => parentOrder.set(n.id, i));

  const layout: LayoutNode[] = [];
  const gens = [...byGen.keys()].sort((a, b) => a - b);
  const colHeights = gens.map((g) => {
    const count = (byGen.get(g) ?? []).length;
    return count * NODE_H + Math.max(0, count - 1) * GAP_Y;
  });
  const maxColH = Math.max(...colHeights, NODE_H);

  gens.forEach((g, gi) => {
    const row = (byGen.get(g) ?? []).sort((a, b) => {
      const pa = a.parent_id ? parentOrder.get(a.parent_id) ?? 0 : 0;
      const pb = b.parent_id ? parentOrder.get(b.parent_id) ?? 0 : 0;
      return pa - pb || a.mape - b.mape;
    });
    const colH = colHeights[gi];
    const offset = (maxColH - colH) / 2;
    row.forEach((n, i) => {
      layout.push({
        ...n,
        x: PAD + g * GAP_X + NODE_W / 2,
        y: PAD + offset + i * (NODE_H + GAP_Y) + NODE_H / 2,
      });
    });
  });
  return layout;
}

function pathToRoot(id: string, byId: Map<string, LayoutNode>): Set<string> {
  const path = new Set<string>();
  let cur: string | undefined = id;
  while (cur) {
    path.add(cur);
    cur = byId.get(cur)?.parent_id ?? undefined;
  }
  return path;
}

function StructureThumb({ smiles, size = 48 }: { smiles: string; size?: number }) {
  const src = `/api/v1/molecule/depict?smiles=${encodeURIComponent(smiles)}&size=${size}`;
  return (
    <img
      src={src}
      alt=""
      width={size}
      height={size}
      style={{ borderRadius: 4, background: "#fff" }}
      loading="lazy"
    />
  );
}

function PlannerBadge({ planner, generation }: { planner?: string | null; generation: number }) {
  const label = plannerLabel(planner, generation);
  const cls =
    generation === 0
      ? "beam-planner beam-planner-seed"
      : planner === "learned"
        ? "beam-planner beam-planner-learned"
        : planner === "rule_based"
          ? "beam-planner beam-planner-rule"
          : "beam-planner";
  return (
    <g transform={`translate(${NODE_W - 52}, 6)`}>
      <rect width={46} height={16} rx={4} className={cls} />
      <text x={23} y={11} textAnchor="middle" className="beam-planner-text">
        {label}
      </text>
    </g>
  );
}

export function BeamTreeView({ tree }: { tree: BeamSearchTree }) {
  const [selectedId, setSelectedId] = useState<string | null>(tree.best_id ?? null);

  const layout = useMemo(() => buildLayout(tree.nodes), [tree.nodes]);
  const byId = useMemo(() => new Map(layout.map((n) => [n.id, n])), [layout]);
  const bestPath = useMemo(
    () => (tree.best_id ? pathToRoot(tree.best_id, byId) : new Set<string>()),
    [tree.best_id, byId],
  );
  const selected = selectedId ? byId.get(selectedId) : undefined;

  const maxGen = layout.reduce((m, n) => Math.max(m, n.generation), 0);
  const maxY = layout.reduce((m, n) => Math.max(m, n.y + NODE_H / 2), 0);
  const svgW = PAD * 2 + maxGen * GAP_X + NODE_W;
  const svgH = Math.max(320, maxY + PAD + 20);

  const edges = layout
    .filter((n) => n.parent_id && byId.has(n.parent_id))
    .map((n) => {
      const p = byId.get(n.parent_id!)!;
      return { id: `${p.id}-${n.id}`, p, n };
    });

  return (
    <div className="beam-tree">
      <div className="beam-tree-legend">
        <span className="beam-legend-item beam-legend-best">Best path</span>
        <span className="beam-legend-item beam-legend-beam">In beam</span>
        <span className="beam-legend-item beam-legend-pruned">Expanded / pruned</span>
        <span className="beam-legend-item beam-planner-learned">Learned planner</span>
        <span className="beam-legend-item beam-planner-rule">Rule-based</span>
      </div>
      <div className="beam-tree-scroll beam-tree-scroll-h">
        <svg width={svgW} height={svgH} className="beam-tree-svg" role="img" aria-label="Beam search tree">
          {Array.from({ length: maxGen + 1 }, (_, g) => (
            <text
              key={`label-${g}`}
              x={PAD + g * GAP_X + NODE_W / 2}
              y={18}
              textAnchor="middle"
              className="beam-layer-label"
            >
              {g === 0 ? "Seed" : `Gen ${g}`}
            </text>
          ))}

          {edges.map(({ id, p, n }) => {
            const onPath = bestPath.has(p.id) && bestPath.has(n.id);
            const x1 = p.x + NODE_W / 2;
            const y1 = p.y;
            const x2 = n.x - NODE_W / 2;
            const y2 = n.y;
            const midX = (x1 + x2) / 2;
            const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
            return (
              <path
                key={id}
                d={d}
                className={onPath ? "beam-edge beam-edge-best" : "beam-edge"}
                fill="none"
              />
            );
          })}

          {layout.map((n) => {
            const isBest = n.id === tree.best_id;
            const onPath = bestPath.has(n.id);
            const cls = isBest
              ? "beam-node beam-node-best"
              : n.in_beam
                ? "beam-node beam-node-beam"
                : onPath
                  ? "beam-node beam-node-path"
                  : "beam-node beam-node-pruned";
            return (
              <g
                key={n.id}
                transform={`translate(${n.x - NODE_W / 2}, ${n.y - NODE_H / 2})`}
                className={cls}
                onClick={() => setSelectedId(n.id)}
                style={{ cursor: "pointer" }}
              >
                <rect width={NODE_W} height={NODE_H} rx={8} className="beam-node-bg" />
                <PlannerBadge planner={n.planner} generation={n.generation} />
                <text x={8} y={22} className="beam-node-title">
                  {n.reaction_family?.slice(0, 18) ?? (n.generation === 0 ? "seed" : "—")}
                </text>
                <text x={8} y={38} className="beam-node-meta">
                  MAPE {n.mape.toFixed(1)}%
                </text>
                <text x={8} y={52} className="beam-node-meta">
                  {n.score_total != null ? `score ${n.score_total.toFixed(3)}` : ""}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {selected && (
        <div className="beam-tree-detail">
          <div className="beam-tree-detail-layout">
            <StructureThumb smiles={selected.canonical_smiles} size={100} />
            <div>
              <div className="beam-tree-detail-title">
                {selected.reaction_family ?? "seed"} · gen {selected.generation}
              </div>
              <code className="beam-tree-smiles">{selected.canonical_smiles}</code>
              <div className="badges">
                <span>{plannerLabel(selected.planner, selected.generation)}</span>
                <span>MAPE {selected.mape.toFixed(2)}%</span>
                {selected.score_total != null && <span>Score {selected.score_total.toFixed(3)}</span>}
                {selected.in_beam && <span>in beam</span>}
                {selected.id === tree.best_id && <span>best</span>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
