"""
Layer 5: Structural Serialization

Flattens the ASTNode tree into a 1-D token sequence suitable for
transformer consumption via DFS preorder traversal.

Three output formats
────────────────────
  flat     [OP_ADD, VAR_X, CONST_1]              ← primary output
  sexp     (OP_ADD VAR_X CONST_1)                ← Lisp-style, human readable
  indented  OP_ADD                               ← indented tree
              VAR_X
              CONST_1

Each emitted token is wrapped in a SerializedToken dataclass that
carries position, depth, parent, child-index, and subtree-size metadata.
This metadata is used by the MetadataGenerator (Layer 6).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

from .ast_generator import ASTNode


# ── Boundary tokens ───────────────────────────────────────────────────────

MATH_START = "[MATH_START]"
MATH_END   = "[MATH_END]"
TEXT_START = "[TEXT_START]"
TEXT_END   = "[TEXT_END]"
SCOPE_OPEN = "[SCOPE_OPEN]"
SCOPE_CLOSE = "[SCOPE_CLOSE]"


# ── Token dataclass ───────────────────────────────────────────────────────

@dataclass
class SerializedToken:
    """
    One token in the flattened structural stream.

    Attributes
    ----------
    token        : MathTok vocabulary string.
    position     : Index in the flat sequence (0-based).
    depth        : Tree depth at emission time (root = 0).
    node_id      : Unique AST node identifier.
    parent_id    : Parent's node_id (-1 for root / boundary tokens).
    child_index  : This node's index among its siblings (0-based).
    num_children : Number of direct children of this node.
    is_leaf      : True iff no children.
    subtree_size : Total nodes in the subtree rooted here.
    is_boundary  : True for [MATH_START], [MATH_END], etc.
    """
    token:        str
    position:     int
    depth:        int
    node_id:      int
    parent_id:    int
    child_index:  int
    num_children: int
    is_leaf:      bool
    subtree_size: int
    is_boundary:  bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"SerializedToken(pos={self.position}, token={self.token!r}, "
            f"depth={self.depth}, children={self.num_children})"
        )


# ── Serializer ────────────────────────────────────────────────────────────

class StructuralSerializer:
    """
    Serialize an ASTNode tree into a flat SerializedToken stream.

    The serialization order is DFS preorder (root first, then children
    left-to-right). This ordering is:
      - recoverable given depth metadata
      - compatible with causal language model training
      - established practice for tree-to-sequence in NLP research

    Parameters
    ----------
    include_boundaries : bool
        Wrap the token stream with [MATH_START] / [MATH_END] sentinels.
    """

    def __init__(
        self,
        include_boundaries: bool = True,
        emit_scope_tokens: bool = True,
        dedup_subtrees: bool = False,
    ) -> None:
        self.include_boundaries = include_boundaries
        self.emit_scope_tokens = emit_scope_tokens
        self.dedup_subtrees = dedup_subtrees
        self._hash_cache: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def serialize(self, root: ASTNode) -> list[SerializedToken]:
        """
        Serialize the AST to a flat SerializedToken stream.

        Parameters
        ----------
        root : ASTNode
            Root node output by ASTGenerator.

        Returns
        -------
        list[SerializedToken]
        """
        tokens: list[SerializedToken] = []
        self._hash_cache.clear()

        if self.include_boundaries:
            tokens.append(_boundary_token(MATH_START, 0))

        self._dfs(root, tokens)

        if self.include_boundaries:
            tokens.append(_boundary_token(MATH_END, len(tokens)))

        # Fix positions after boundary prepend
        for i, t in enumerate(tokens):
            object.__setattr__(t, "position", i) if hasattr(t, "__dataclass_fields__") else None
            t.position = i

        return tokens

    def to_token_list(self, root: ASTNode) -> list[str]:
        """Return just the token strings (for vocabulary mapping)."""
        return [st.token for st in self.serialize(root)]

    def to_sexp(self, root: ASTNode) -> str:
        """Serialize to a Lisp-style S-expression string."""
        return self._sexp(root)

    def to_indented(self, root: ASTNode, indent: int = 2) -> str:
        """Serialize to an indented tree string."""
        lines: list[str] = []
        self._indent(root, lines, 0, indent)
        return "\n".join(lines)

    def reconstruct_depth_sequence(self, tokens: list[SerializedToken]) -> list[int]:
        """Return the depth of each token position (useful for pos-encoding)."""
        return [max(t.depth, 0) for t in tokens]

    def subtree_hash(self, node: ASTNode) -> str:
        """Compute a stable MD5 structural hash of the subtree rooted at node."""
        hasher = hashlib.md5()
        hasher.update(node.token.encode('utf-8'))
        for child in node.children:
            hasher.update(self.subtree_hash(child).encode('utf-8'))
        return hasher.hexdigest()

    # ── DFS preorder traversal ────────────────────────────────────────────

    def _dfs(
        self,
        node: ASTNode,
        tokens: list[SerializedToken],
        child_index: int = 0,
    ) -> None:
        """Emit current node then recurse into children."""
        if self.dedup_subtrees and not node.is_leaf:
            node_hash = self.subtree_hash(node)
            if node_hash in self._hash_cache:
                tokens.append(SerializedToken(
                    token=f"SUBTREE_REF_{node_hash[:8]}",
                    position=len(tokens),
                    depth=node.depth,
                    node_id=node.node_id,
                    parent_id=node.parent_id,
                    child_index=child_index,
                    num_children=0,
                    is_leaf=True,
                    subtree_size=1,
                ))
                return
            self._hash_cache[node_hash] = node.node_id

        pos = len(tokens)
        tokens.append(SerializedToken(
            token=node.token,
            position=pos,
            depth=node.depth,
            node_id=node.node_id,
            parent_id=node.parent_id,
            child_index=child_index,
            num_children=len(node.children),
            is_leaf=node.is_leaf,
            subtree_size=node.subtree_size,
        ))

        is_function = node.token.startswith("FUNC_")
        if is_function and self.emit_scope_tokens and not node.is_leaf:
            tokens.append(_boundary_token(SCOPE_OPEN, len(tokens), depth=node.depth + 1, parent_id=node.node_id))

        for i, child in enumerate(node.children):
            self._dfs(child, tokens, child_index=i)

        if is_function and self.emit_scope_tokens and not node.is_leaf:
            tokens.append(_boundary_token(SCOPE_CLOSE, len(tokens), depth=node.depth + 1, parent_id=node.node_id))

    # ── S-expression ──────────────────────────────────────────────────────

    def _sexp(self, node: ASTNode) -> str:
        if node.is_leaf:
            return node.token
        child_parts = " ".join(self._sexp(c) for c in node.children)
        return f"({node.token} {child_parts})"

    # ── Indented tree ─────────────────────────────────────────────────────

    def _indent(self, node: ASTNode, lines: list[str], level: int, indent: int) -> None:
        lines.append(" " * (level * indent) + node.token)
        for child in node.children:
            self._indent(child, lines, level + 1, indent)


# ── Helpers ───────────────────────────────────────────────────────────────

def _boundary_token(tok: str, pos: int, depth: int = -1, parent_id: int = -1) -> SerializedToken:
    return SerializedToken(
        token=tok, position=pos, depth=depth, node_id=-1,
        parent_id=parent_id, child_index=0, num_children=0,
        is_leaf=True, subtree_size=0, is_boundary=True,
    )
