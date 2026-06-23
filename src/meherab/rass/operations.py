"""RASS (Representation-Aware Search Space) operations (paper Sec. 3.3).

Three operation families over a homophilic-graph node/edge:

* **Adapter-Inject (AI)** -- rank-r PCA-residual blend at a node (Eq. 4).
  Dominates empirically: appears in all ten discovered pathways.
* **Semantic-Compress (SC)** -- keep the top-(kappa*d) highest-variance
  feature dimensions at a node. Only generated for multi-member cluster
  nodes.
* **Cross-Fuse (CF)** -- L2-normalize and average the two endpoints of a
  hyperedge. Appears in only one paper pathway (PatternNet), consistent
  with the near-zero cross-node synergy predicted by Proposition 2.

A ``RASSCandidate`` is a 3-tuple of operations drawn without replacement;
``RASSFactory`` enumerates the full operation set for a given homophilic
graph and implements the genetic operators (mutation, crossover) used by
the evolutionary search in ``meherab.search.evolutionary``.

Extracted and refactored (explicit parameters instead of a module-level
``cfg`` global) from the original pipeline, Cell 10.
"""
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TType(Enum):
    SC = "SC"  # Semantic Compress
    CF = "CF"  # Cross-Scale Fuse
    AI = "AI"  # Adapter Inject


@dataclass(frozen=True)
class RASSOp:
    op_type: TType
    node_id: Optional[int] = None
    edge_id: Optional[int] = None
    param: float = 1.0

    def __repr__(self):
        if self.op_type == TType.SC:
            return f"SC(n{self.node_id},k{int(self.param * 100)}pct)"
        if self.op_type == TType.CF:
            return f"CF(e{self.edge_id})"
        return f"AI(n{self.node_id},r{int(self.param)})"


@dataclass
class RASSCandidate:
    ops: List[RASSOp]
    mds_score: float = -999.0

    def __hash__(self):
        return hash(tuple(self.ops))


class RASSFactory:
    """Enumerates all RASS ops from a SemanticHypergraph and implements the
    genetic operators (mutation, crossover) used by the evolutionary search.
    """

    def __init__(
        self,
        hg,
        sc_keep_ratios=(0.5, 0.75, 1.0),
        adapter_ranks=(4, 8, 16),
        n_ops: int = 3,
        evo_mutation: float = 0.3,
    ):
        self.hg = hg
        self.n_ops = n_ops
        self.evo_mutation = evo_mutation

        ops: List[RASSOp] = []
        # SC: only for multi-member nodes
        for nid in hg.nodes:
            if len(hg.node_members[nid]) > 1:
                for kr in sc_keep_ratios:
                    ops.append(RASSOp(TType.SC, node_id=nid, param=kr))
        # CF: one per hyperedge
        for eid in range(len(hg.hyperedges)):
            ops.append(RASSOp(TType.CF, edge_id=eid))
        # AI: one per node x rank
        for nid in hg.nodes:
            for rank in adapter_ranks:
                ops.append(RASSOp(TType.AI, node_id=nid, param=float(rank)))

        # Dedup + guarantee at least one op
        self.all_ops = list(dict.fromkeys(ops))
        if not self.all_ops:
            self.all_ops = [RASSOp(TType.SC, node_id=hg.nodes[0], param=1.0)]

    def random_candidate(self) -> RASSCandidate:
        # Sample WITHOUT replacement -- prevents degenerate "same op x 3".
        n = min(self.n_ops, len(self.all_ops))
        return RASSCandidate(ops=random.sample(self.all_ops, n))

    def mutate(self, c: RASSCandidate) -> RASSCandidate:
        used, new_ops = set(c.ops), []
        for op in c.ops:
            if random.random() < self.evo_mutation:
                pool = [o for o in self.all_ops if o not in used]
                repl = random.choice(pool) if pool else op
                new_ops.append(repl)
                used.add(repl)
            else:
                new_ops.append(op)
        return RASSCandidate(ops=new_ops)

    def crossover(self, p1: RASSCandidate, p2: RASSCandidate) -> RASSCandidate:
        n = len(p1.ops)
        k = random.randint(1, max(1, n - 1))
        seen, out = set(), []
        for op in p1.ops[:k] + p2.ops[k:]:
            if op not in seen:
                out.append(op)
                seen.add(op)
        while len(out) < n:
            pool = [o for o in self.all_ops if o not in seen]
            if not pool:
                break
            pick = random.choice(pool)
            out.append(pick)
            seen.add(pick)
        return RASSCandidate(ops=out[:n])
