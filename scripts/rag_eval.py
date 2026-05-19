import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag_service import PolicyRAGService

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rag_golden.json"


def main() -> int:
    service = PolicyRAGService(use_embeddings=False)
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    hits = 0
    for case in cases:
        result = service.search(case["query"], top_k=3)
        ok = bool(result.chunks)
        if ok and "must_source" in case:
            ok = any(case["must_source"] in c.source for c in result.chunks)
        if ok and "must_heading_substr" in case:
            ok = any(
                case["must_heading_substr"].lower() in (c.heading or "").lower()
                for c in result.chunks
            )
        hits += int(ok)
        status = "OK" if ok else "MISS"
        print(f"{status}  {case['query']}")
    total = len(cases)
    print(f"\nrecall@3: {hits}/{total} ({100 * hits / total:.0f}%)")
    return 0 if hits == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
