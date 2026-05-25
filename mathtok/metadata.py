"""
Layer 6: Structural Attention Metadata Generator

For every token in the serialized stream, generate a rich metadata
record capturing its full tree context.  This metadata is the primary
research contribution of MathTok — it enables structure-aware attention
in downstream transformer models without architectural changes.

Metadata fields per token
─────────────────────────
  position         : flat index in sequence
  token            : token string
  token_id         : vocabulary ID (filled if vocab is provided)
  node_id          : AST node ID
  parent_id        : parent node ID (-1 = root)
  children_ids     : list of direct child node IDs
  depth            : tree depth (root = 0)
  child_index      : index among siblings
  subtree_size     : total nodes in subtree
  is_leaf          : terminal node flag
  num_children     : number of direct children
  token_category   : 'operator' | 'function' | 'variable' | 'constant'
                     | 'structural' | 'boundary' | 'text'
  tree_position_key: dot-notation path from root, e.g. "0.1.2"
  sibling_count    : total number of siblings (including self)

Attention mask helpers
──────────────────────
  to_attention_mask_hints() returns binary NxN matrices for:
    parent_mask   — attend to parent
    children_mask — attend to children
    sibling_mask  — attend to siblings
    subtree_mask  — attend within own subtree
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

from .serializer import SerializedToken


# ── Token classification ───────────────────────────────────────────────────

_BOUNDARY_TOKENS = {
    "[MATH_START]", "[MATH_END]",
    "[TEXT_START]", "[TEXT_END]",
    "[BOS]", "[EOS]", "[PAD]", "[UNK]",
    "[SCOPE_OPEN]", "[SCOPE_CLOSE]",
}


def _classify(token: str) -> str:
    if token in _BOUNDARY_TOKENS:
        return "boundary"
    if token.startswith("OP_") or token == "FRAC":
        return "operator"
    if token.startswith("FUNC_"):
        return "function"
    if token.startswith("VAR_"):
        return "variable"
    if (token.startswith("CONST_") or token.startswith("NUM_")
            or token.startswith("FLOAT_")):
        return "constant"
    if token.startswith("SUBTREE_REF_") or token == "SUBTREE_TRUNCATED":
        return "structural"
    return "text"


# ── Metadata dataclass ────────────────────────────────────────────────────

@dataclass
class TokenMetadata:
    """
    Rich structural metadata for one token position.

    This record provides all information needed to implement
    structure-aware attention, tree positional encoding, or
    graph-neural-network processing of math token sequences.
    """
    # ── Identity ─────────────────────────────────────────────────────────
    position:         int
    token:            str
    token_id:         int           # -1 if vocab not provided

    # ── Tree structure ────────────────────────────────────────────────────
    node_id:          int
    parent_id:        int
    parent_token:     str
    children_ids:     list[int]
    depth:            int
    child_index:      int

    # ── Subtree info ──────────────────────────────────────────────────────
    subtree_size:     int
    is_leaf:          bool
    num_children:     int

    # ── Semantic category ─────────────────────────────────────────────────
    token_category:   str           # operator | function | variable | constant | boundary | text

    # ── Positional hints ──────────────────────────────────────────────────
    tree_position_key: str          # e.g. "0.1.2" = root→child[1]→child[2]
    sibling_count:    int

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"TokenMetadata(pos={self.position}, token={self.token!r}, "
            f"depth={self.depth}, cat={self.token_category!r})"
        )


# ── Generator ────────────────────────────────────────────────────────────

class MetadataGenerator:
    """
    Generate structural metadata for a serialized token stream.

    Usage
    -----
    >>> gen = MetadataGenerator()
    >>> meta = gen.generate(serialized_tokens, vocab={"OP_ADD": 8, ...})
    >>> for m in meta:
    ...     print(m.tree_position_key, m.token_category)
    """

    def generate(
        self,
        tokens: list[SerializedToken],
        vocab: Optional[dict[str, int]] = None,
    ) -> list[TokenMetadata]:
        """
        Generate TokenMetadata for every token in the stream.

        Parameters
        ----------
        tokens : list[SerializedToken]
            Output of StructuralSerializer.serialize().
        vocab : dict[str, int] | None
            Optional vocabulary mapping token → ID.

        Returns
        -------
        list[TokenMetadata]
        """
        vocab = vocab or {}

        # Build structural lookup tables
        node_to_pos:        dict[int, int]       = {}
        node_to_token:      dict[int, str]       = {}
        parent_to_children: dict[int, list[int]] = defaultdict(list)

        for pos, st in enumerate(tokens):
            if st.node_id >= 0:
                node_to_pos[st.node_id] = pos
                node_to_token[st.node_id] = st.token
            if st.parent_id >= 0:
                parent_to_children[st.parent_id].append(st.node_id)

        position_keys = self._build_position_keys(tokens)

        result: list[TokenMetadata] = []
        for pos, st in enumerate(tokens):
            children_ids = parent_to_children.get(st.node_id, [])
            siblings     = parent_to_children.get(st.parent_id, []) if st.parent_id >= 0 else []

            meta = TokenMetadata(
                position          = pos,
                token             = st.token,
                token_id          = vocab.get(st.token, -1),
                node_id           = st.node_id,
                parent_id         = st.parent_id,
                parent_token      = node_to_token.get(st.parent_id, ""),
                children_ids      = list(children_ids),
                depth             = max(st.depth, 0),
                child_index       = st.child_index,
                subtree_size      = st.subtree_size,
                is_leaf           = st.is_leaf,
                num_children      = st.num_children,
                token_category    = _classify(st.token),
                tree_position_key = position_keys.get(st.node_id, "root"),
                sibling_count     = len(siblings),
            )
            result.append(meta)

        return result

    def to_attention_mask_hints(
        self,
        metadata: list[TokenMetadata],
    ) -> dict[str, list[list[int]]]:
        """
        Generate NxN binary attention mask hints from metadata.

        Returns
        -------
        dict with keys:
          'parent_mask'   : token i can attend to its parent
          'children_mask' : token i can attend to all its children
          'sibling_mask'  : token i can attend to its siblings
          'subtree_mask'  : token i can attend to all nodes in its subtree

        Each mask value is a list-of-lists of 0/1 integers (N x N).
        """
        n = len(metadata)
        node_to_pos: dict[int, int] = {m.node_id: m.position for m in metadata if m.node_id >= 0}

        parent_mask   = [[0] * n for _ in range(n)]
        children_mask = [[0] * n for _ in range(n)]
        sibling_mask  = [[0] * n for _ in range(n)]
        subtree_mask  = [[0] * n for _ in range(n)]

        # Build subtree membership: node_id → set of all descendant node_ids
        subtree_members = self._build_subtree_members(metadata, node_to_pos)

        for m in metadata:
            i = m.position

            # Parent
            if m.parent_id >= 0 and m.parent_id in node_to_pos:
                parent_mask[i][node_to_pos[m.parent_id]] = 1

            # Children
            for child_id in m.children_ids:
                if child_id in node_to_pos:
                    children_mask[i][node_to_pos[child_id]] = 1

            # Siblings (same parent, different node)
            if m.parent_id >= 0:
                for m2 in metadata:
                    if m2.parent_id == m.parent_id and m2.position != i:
                        sibling_mask[i][m2.position] = 1

            # Subtree
            for desc_pos in subtree_members.get(m.node_id, set()):
                subtree_mask[i][desc_pos] = 1

        return {
            "parent_mask":   parent_mask,
            "children_mask": children_mask,
            "sibling_mask":  sibling_mask,
            "subtree_mask":  subtree_mask,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_position_keys(self, tokens: list[SerializedToken]) -> dict[int, str]:
        """
        Build a dot-separated path string for every node.
        The root gets key "0"; each child appends ".{child_index}".
        """
        keys: dict[int, str] = {}

        # Find root node(s) — parent_id == -1 and not a boundary
        for st in tokens:
            if st.parent_id == -1 and st.node_id >= 0:
                keys[st.node_id] = "0"

        # Iterative BFS propagation
        changed = True
        while changed:
            changed = False
            for st in tokens:
                if st.node_id not in keys and st.parent_id in keys:
                    keys[st.node_id] = f"{keys[st.parent_id]}.{st.child_index}"
                    changed = True

        return keys

    def _build_subtree_members(
        self,
        metadata: list[TokenMetadata],
        node_to_pos: dict[int, int],
    ) -> dict[int, set[int]]:
        """
        For each node, compute the set of *positions* of all its descendants.
        Used for building the subtree attention mask.
        """
        # Build parent→children mapping
        children_of: dict[int, list[int]] = defaultdict(list)
        for m in metadata:
            if m.parent_id >= 0:
                children_of[m.parent_id].append(m.node_id)

        subtree: dict[int, set[int]] = {}

        def collect(node_id: int) -> set[int]:
            if node_id in subtree:
                return subtree[node_id]
            members: set[int] = set()
            if node_id in node_to_pos:
                members.add(node_to_pos[node_id])
            for child_id in children_of.get(node_id, []):
                members |= collect(child_id)
            subtree[node_id] = members
            return members

        for m in metadata:
            if m.node_id >= 0:
                collect(m.node_id)

        return subtree
