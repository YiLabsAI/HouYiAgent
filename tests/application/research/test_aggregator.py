from __future__ import annotations

from houyi.application.research.aggregator import SourceAggregator
from houyi.application.research.types import SearchResult, SourceReference


def _src(url: str, title: str = "T", snippet: str = "S", score: float = 0.5) -> SourceReference:
    return SourceReference(url=url, title=title, snippet=snippet, reliability_score=score)


def _result(qid: str, sources: list[SourceReference]) -> SearchResult:
    return SearchResult(question_id=qid, sources=sources)


class TestUrlDedup:
    async def test_same_url_merged(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com", score=0.3)]),
            _result("q2", [_src("https://a.com", score=0.8)]),
        ]
        out = await agg.aggregate(results)
        assert len(out.sources) == 1
        assert out.deduplicated_count == 1
        assert out.sources[0].reliability_score == 0.8

    async def test_different_urls_kept(self):
        agg = SourceAggregator()
        results = [
            _result(
                "q1",
                [
                    _src("https://a.com", "Title A", "Snip A"),
                    _src("https://b.com", "Title B", "Snip B"),
                ],
            )
        ]
        out = await agg.aggregate(results)
        assert len(out.sources) == 2
        assert out.deduplicated_count == 0


class TestContentDedup:
    async def test_same_title_snippet_merged(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com", "Title", "Snippet")]),
            _result("q2", [_src("https://b.com", "Title", "Snippet")]),
        ]
        out = await agg.aggregate(results)
        assert len(out.sources) == 1

    async def test_different_content_kept(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com", "Title A", "Snippet A")]),
            _result("q2", [_src("https://b.com", "Title B", "Snippet B")]),
        ]
        out = await agg.aggregate(results)
        assert len(out.sources) == 2


class TestCoverage:
    async def test_coverage_by_question(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com"), _src("https://b.com")]),
            _result("q2", [_src("https://c.com")]),
        ]
        out = await agg.aggregate(results)
        assert "q1" in out.coverage_by_question
        assert "q2" in out.coverage_by_question

    async def test_grouped_by_question(self):
        agg = SourceAggregator()
        results = [_result("q1", [_src("https://a.com")])]
        out = await agg.aggregate(results)
        assert len(out.grouped_by_question["q1"]) == 1

    async def test_cross_question_dedup(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com", score=0.3)]),
            _result("q2", [_src("https://a.com", score=0.8)]),
        ]
        out = await agg.aggregate(results)

        assert len(out.sources) == 1
        assert len(out.grouped_by_question["q1"]) == 1
        assert len(out.grouped_by_question["q2"]) == 1
        assert out.grouped_by_question["q1"][0] == out.grouped_by_question["q2"][0]


class TestRanking:
    async def test_sorted_by_reliability(self):
        agg = SourceAggregator()
        results = [
            _result(
                "q1",
                [
                    _src("https://low.com", "Low", "low", 0.2),
                    _src("https://high.com", "High", "high", 0.9),
                ],
            ),
        ]
        out = await agg.aggregate(results)
        assert out.sources[0].reliability_score >= out.sources[-1].reliability_score


class TestBoundary:
    async def test_single_source(self):
        agg = SourceAggregator()
        results = [_result("q1", [_src("https://only.com")])]
        out = await agg.aggregate(results)
        assert len(out.sources) == 1
        assert out.deduplicated_count == 0

    async def test_many_sources_preserved(self):
        agg = SourceAggregator()
        sources = [_src(f"https://{i}.com", f"T{i}", f"S{i}") for i in range(50)]
        results = [_result("q1", sources)]
        out = await agg.aggregate(results)
        assert len(out.sources) == 50


class TestInvalidData:
    async def test_source_with_no_url(self):
        agg = SourceAggregator()
        src = SourceReference(url=None, title="No URL", snippet="snip", reliability_score=0.5)
        results = [SearchResult(question_id="q1", sources=[src])]
        out = await agg.aggregate(results)
        assert len(out.sources) == 1


class TestNoiseFiltering:
    async def test_irrelevant_source_removed(self):
        agg = SourceAggregator()
        results = [
            _result(
                "q1",
                [
                    _src("https://a.com", "Apache RocketMQ", "Open-source messaging platform"),
                    _src("https://b.com", "Martial Arts Novel", "Wuxia fiction story"),
                    _src("https://c.com", "RocketMQ Architecture", "Distributed messaging"),
                    _src("https://d.com", "RocketMQ Docs", "Apache documentation"),
                ],
            ),
        ]
        out = await agg.aggregate(results, user_query="Apache RocketMQ architecture")
        assert len(out.sources) == 3
        assert all("Martial" not in s.title for s in out.sources)

    async def test_relevant_source_kept(self):
        agg = SourceAggregator()
        results = [
            _result(
                "q1",
                [
                    _src("https://a.com", "RocketMQ Guide", "Architecture overview"),
                    _src("https://b.com", "Messaging Systems", "RocketMQ comparison"),
                ],
            ),
        ]
        out = await agg.aggregate(results, user_query="RocketMQ architecture")
        assert len(out.sources) == 2

    async def test_short_content_filtered(self):
        agg = SourceAggregator()
        results = [
            _result(
                "q1",
                [
                    _src("https://a.com", "Good", "Sufficient content here for evidence"),
                    _src("https://b.com", "X", ""),
                    _src("https://c.com", "Good Details", "More good content here"),
                    _src("https://d.com", "Good Analysis", "Analytical good content"),
                ],
            ),
        ]
        out = await agg.aggregate(results, user_query="Good content")
        assert len(out.sources) == 3

    async def test_no_filter_without_query(self):
        agg = SourceAggregator()
        results = [
            _result("q1", [_src("https://a.com", "Anything", "Random content")]),
        ]
        out = await agg.aggregate(results, user_query="")
        assert len(out.sources) == 1


class TestEmpty:
    async def test_empty_results(self):
        agg = SourceAggregator()
        out = await agg.aggregate([])
        assert len(out.sources) == 0
        assert out.deduplicated_count == 0

    async def test_empty_coverage_map(self):
        agg = SourceAggregator()
        out = await agg.aggregate([])
        assert out.coverage_by_question == {}
