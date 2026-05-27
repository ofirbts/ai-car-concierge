from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from backend.idempotency_utils import DedupStore, stable_idempotency_key

_DEDUP = DedupStore()

_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".sql",
    ".md",
    ".txt",
}
_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".repo_audit",
}
_EXCLUDED_FILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Cargo.lock",
}
_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]


class PackagerRequest(BaseModel):
    root_path: str
    include_globs: list[str] = Field(default_factory=list)
    max_files: int = Field(default=300, ge=1, le=2000)
    max_chars_per_file: int = Field(default=12000, ge=100, le=100000)


class PackedFile(BaseModel):
    path: str
    content: str
    estimated_tokens: int


class SecurityFinding(BaseModel):
    path: str
    pattern: str


class PackagerResponse(BaseModel):
    idempotency_key: str
    root_path: str
    files: list[PackedFile]
    skipped_files: int
    total_estimated_tokens: int
    security_findings: list[SecurityFinding]


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_EXTENSIONS


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _collect_files(root: Path, include_globs: list[str]) -> list[Path]:
    if include_globs:
        collected: list[Path] = []
        seen: set[str] = set()
        for pattern in include_globs:
            for candidate in root.glob(pattern):
                if not candidate.is_file():
                    continue
                if any(part in _EXCLUDED_DIRS for part in candidate.parts):
                    continue
                if candidate.name in _EXCLUDED_FILES:
                    continue
                if not _is_text_file(candidate):
                    continue
                key = str(candidate.resolve())
                if key in seen:
                    continue
                seen.add(key)
                collected.append(candidate)
        return sorted(collected)

    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in candidate.parts):
            continue
        if candidate.name in _EXCLUDED_FILES:
            continue
        if not _is_text_file(candidate):
            continue
        files.append(candidate)
    return sorted(files)


def _security_scan(relative_path: str, content: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(SecurityFinding(path=relative_path, pattern=pattern.pattern))
    return findings


def package_codebase(request: PackagerRequest) -> PackagerResponse:
    key = stable_idempotency_key("codebase_packager", request.model_dump(mode="json"))
    cached = _DEDUP.get(key)
    if cached is not None:
        return cached

    root = Path(request.root_path).resolve()
    files = _collect_files(root, request.include_globs)

    packed_files: list[PackedFile] = []
    security_findings: list[SecurityFinding] = []
    skipped = 0
    total_tokens = 0

    for path in files:
        if len(packed_files) >= request.max_files:
            skipped += 1
            continue
        rel = str(path.relative_to(root))
        content = path.read_text(encoding="utf-8", errors="ignore")
        security_findings.extend(_security_scan(rel, content))
        compact = content[: request.max_chars_per_file]
        tokens = _estimate_tokens(compact)
        total_tokens += tokens
        packed_files.append(PackedFile(path=rel, content=compact, estimated_tokens=tokens))

    response = PackagerResponse(
        idempotency_key=key,
        root_path=str(root),
        files=packed_files,
        skipped_files=skipped,
        total_estimated_tokens=total_tokens,
        security_findings=security_findings,
    )
    _DEDUP.put(key, response, ttl_seconds=300)
    return response

