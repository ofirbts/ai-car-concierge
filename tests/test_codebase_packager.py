from pathlib import Path

from backend.codebase_packager import PackagerRequest, package_codebase


def test_codebase_packager_filters_and_scans(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "secret.py").write_text("API_KEY='sk-AAAAAAAAAAAAAAAAAAAA'\n", encoding="utf-8")
    (root / "package-lock.json").write_text('{"name":"x"}', encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02")

    response = package_codebase(PackagerRequest(root_path=str(root)))

    paths = {item.path for item in response.files}
    assert "app.py" in paths
    assert "secret.py" in paths
    assert "package-lock.json" not in paths
    assert all(not path.endswith(".bin") for path in paths)
    assert any("sk-" in finding.pattern for finding in response.security_findings)


def test_codebase_packager_idempotent_cache(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n", encoding="utf-8")

    request = PackagerRequest(root_path=str(root))
    first = package_codebase(request)
    second = package_codebase(request)

    assert first.idempotency_key == second.idempotency_key
    assert first.total_estimated_tokens == second.total_estimated_tokens
