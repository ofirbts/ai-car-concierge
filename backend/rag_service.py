from __future__ import annotations

import math
import re
from pathlib import Path

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.gemini_service import embed_query, embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"

_default_service: PolicyRAGService | None = None


class PolicyChunk(BaseModel):
    source: str
    heading: str
    content: str
    score: float = 0.0


class PolicySearchResult(BaseModel):
    query: str
    chunks: list[PolicyChunk] = Field(default_factory=list)
    retrieval_mode: str = "keyword"


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_markdown(text: str, source: str) -> list[dict[str, str]]:
    lines = text.strip().splitlines()
    title = source
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    sections: list[dict[str, str]] = []
    current_heading = title
    current_lines: list[str] = []

    for line in lines[1:]:
        if line.startswith("## "):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(
                    {
                        "source": source,
                        "heading": current_heading,
                        "content": body,
                    }
                )
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append(
            {
                "source": source,
                "heading": current_heading,
                "content": body,
            }
        )

    if not sections and text.strip():
        sections.append(
            {
                "source": source,
                "heading": title,
                "content": text.strip(),
            }
        )
    return sections


def load_policy_chunks(policies_dir: Path | None = None) -> list[dict[str, str]]:
    base = policies_dir or POLICIES_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"Policies directory not found: {base}")

    chunks: list[dict[str, str]] = []
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(_split_markdown(text, path.name))
    return chunks


def _keyword_score(query: str, chunk: dict[str, str]) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    body_tokens = _tokenize(chunk["content"])
    heading_tokens = _tokenize(chunk["heading"])
    overlap = len(query_tokens & body_tokens)
    heading_overlap = len(query_tokens & heading_tokens)
    return overlap + heading_overlap * 1.5


def _source_prior_boost(query: str, chunk: dict[str, str]) -> float:
    q = query.lower()
    source = chunk.get("source", "").lower()
    if ("refund" in q or "deposit" in q) and source == "faqs.md":
        return 0.35
    if ("shipping" in q or "delivery" in q) and source == "shipping.md":
        return 0.2
    if ("2022" in q or "de-listing" in q or "delisting" in q) and source == "policy.md":
        return 0.25
    return 0.0


class PolicyRAGService:
    def __init__(
        self,
        policies_dir: Path | None = None,
        use_embeddings: bool | None = None,
    ):
        self._chunks = load_policy_chunks(policies_dir)
        self._embeddings: list[list[float]] | None = None
        self._embeddings_ready = False
        settings = get_settings()
        self._use_embeddings = (
            use_embeddings if use_embeddings is not None else settings.has_google_api()
        )

    @property
    def retrieval_mode(self) -> str:
        return "gemini_embeddings" if self._use_embeddings else "keyword"

    def _ensure_embeddings(self) -> None:
        if self._embeddings_ready or not self._use_embeddings:
            return
        if not self._chunks:
            self._embeddings = []
            self._embeddings_ready = True
            return
        texts = [f"{c['heading']}\n{c['content']}" for c in self._chunks]
        self._embeddings = embed_texts(texts)
        self._embeddings_ready = True

    def search(self, query: str, top_k: int = 3) -> PolicySearchResult:
        mode = self.retrieval_mode
        if not self._chunks or top_k <= 0:
            return PolicySearchResult(query=query, chunks=[], retrieval_mode=mode)

        query_stripped = query.strip()
        if not query_stripped:
            return PolicySearchResult(query=query, chunks=[], retrieval_mode=mode)

        ranked: list[tuple[float, dict[str, str]]] = []
        if self._use_embeddings:
            self._ensure_embeddings()
            if self._embeddings and len(self._embeddings) == len(self._chunks):
                query_vec = embed_query(query_stripped)
                if query_vec:
                    for chunk, vec in zip(self._chunks, self._embeddings):
                        semantic = _cosine_similarity(query_vec, vec)
                        score = (
                            semantic
                            + _keyword_score(query_stripped, chunk) * 0.08
                            + _source_prior_boost(query_stripped, chunk)
                        )
                        ranked.append((score, chunk))
            if not ranked:
                for chunk in self._chunks:
                    ranked.append((_keyword_score(query_stripped, chunk), chunk))
        else:
            for chunk in self._chunks:
                ranked.append((_keyword_score(query_stripped, chunk), chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        top = ranked[:top_k]
        chunks = [
            PolicyChunk(
                source=item["source"],
                heading=item["heading"],
                content=item["content"],
                score=score,
            )
            for score, item in top
        ]
        return PolicySearchResult(query=query, chunks=chunks, retrieval_mode=mode)

    @staticmethod
    def format_context(result: PolicySearchResult) -> str:
        if not result.chunks:
            return ""
        blocks = [
            f"[{chunk.source} — {chunk.heading}]\n{chunk.content}"
            for chunk in result.chunks
        ]
        return "\n\n".join(blocks)


def reset_policy_rag_service() -> None:
    global _default_service
    _default_service = None
    from backend.gemini_service import reset_gemini_client

    reset_gemini_client()


def get_policy_rag_service() -> PolicyRAGService:
    global _default_service
    if _default_service is None:
        _default_service = PolicyRAGService()
    return _default_service


def search_policies(query: str, top_k: int = 3) -> PolicySearchResult:
    return get_policy_rag_service().search(query, top_k=top_k)
