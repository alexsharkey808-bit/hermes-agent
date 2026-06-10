from retrieval.corpus_search import grep_corpus, needs_vector_fallback


def test_grep_finds_literal_match(tmp_path):
    (tmp_path / "a.txt").write_text("alpha needle beta\n")
    (tmp_path / "b.txt").write_text("nothing here\n")
    hits = grep_corpus("needle", root=str(tmp_path))
    assert len(hits) == 1 and hits[0].path.endswith("a.txt") and hits[0].line_number == 1


def test_grep_is_fixed_string_not_regex(tmp_path):
    (tmp_path / "c.txt").write_text("value = a.b.c\n")
    assert len(grep_corpus("a.b.c", root=str(tmp_path))) == 1


def test_grep_respects_max_results(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("needle\n")
    assert len(grep_corpus("needle", root=str(tmp_path), max_results=3)) == 3


def test_grep_hits_present_no_fallback():
    hits = [{"score": 0.9}, {"score": 0.8}]
    assert needs_vector_fallback(hits, min_hits=1, min_score=0.5) is False


def test_zero_hits_triggers_fallback():
    assert needs_vector_fallback([], min_hits=1, min_score=0.5) is True


def test_low_confidence_hits_trigger_fallback():
    hits = [{"score": 0.2}, {"score": 0.1}]  # all below min_score
    assert needs_vector_fallback(hits, min_hits=1, min_score=0.5) is True
