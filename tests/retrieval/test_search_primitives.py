import pytest

from retrieval.search_primitives import Caps, EgressDenied, make_primitives


class FakeWebClient:
    def search(self, query, k):
        return [{"url": "https://ok.test/a", "snippet": query}][:k]

    def fetch(self, url):
        return "x" * 1000


def test_web_search_maps_to_client():
    p = make_primitives(FakeWebClient(), allow=["ok.test"], caps=Caps(max_calls=5, max_bytes=10_000))
    assert p["web_search"]("cves", k=1)[0]["url"].startswith("https://ok.test")


def test_extract_blocks_disallowed_host():
    p = make_primitives(FakeWebClient(), allow=["ok.test"], caps=Caps(max_calls=5, max_bytes=10_000))
    with pytest.raises(EgressDenied):
        p["web_extract"]("https://evil.test/x")


def test_rate_cap_enforced():
    p = make_primitives(FakeWebClient(), allow=["ok.test"], caps=Caps(max_calls=1, max_bytes=10_000))
    p["web_extract"]("https://ok.test/a")
    with pytest.raises(EgressDenied, match="rate"):
        p["web_extract"]("https://ok.test/b")


def test_byte_cap_enforced():
    # FakeWebClient.fetch returns 1000 bytes; a 500-byte cap is exceeded on the first fetch.
    p = make_primitives(FakeWebClient(), allow=["ok.test"], caps=Caps(max_calls=5, max_bytes=500))
    with pytest.raises(EgressDenied, match="byte"):
        p["web_extract"]("https://ok.test/a")
