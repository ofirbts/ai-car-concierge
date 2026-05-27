from types import SimpleNamespace

import backend.gemini_service as gemini_service


class _FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(text='{"intent":"general_chat"}')


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_generate_structured_uses_dedup_cache(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(gemini_service, "get_gemini_client", lambda: fake_client)
    gemini_service._dedup._entries.clear()

    schema = __import__("backend.intent", fromlist=["ExtractedIntent"]).ExtractedIntent
    first = gemini_service.generate_structured("system", "user", schema)
    second = gemini_service.generate_structured("system", "user", schema)

    assert first is not None
    assert second is not None
    assert fake_client.models.calls == 1
