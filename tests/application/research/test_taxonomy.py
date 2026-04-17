from __future__ import annotations

from houyi.application.research.taxonomy import (
    ANALYTIC_TOPIC_CJK_HINTS,
    ANALYTIC_TOPIC_EN_HINTS,
    COUNTER_EVIDENCE_MARKERS,
    ENTITY_QUERY_HINTS,
    IDENTITY_SOURCE_MARKERS,
)


class TestTaxonomy:
    def test_entity_hints_exist(self):
        assert len(ENTITY_QUERY_HINTS) > 0

    def test_analytic_hints_exist(self):
        assert len(ANALYTIC_TOPIC_CJK_HINTS) > 0
        assert len(ANALYTIC_TOPIC_EN_HINTS) > 0

    def test_identity_markers_exist(self):
        assert len(IDENTITY_SOURCE_MARKERS) > 0

    def test_has_en_and_cjk(self):
        en = [m for m in COUNTER_EVIDENCE_MARKERS if m.isascii()]
        cjk = [m for m in COUNTER_EVIDENCE_MARKERS if not m.isascii()]
        assert len(en) >= 6, "need broad English counter-evidence markers"
        assert len(cjk) >= 5, "need CJK counter-evidence markers for bilingual queries"

    def test_counter_evidence_cjk_match(self):
        # zhengyi (controversy) should match CJK text
        text = "\u8fd9\u4e2a\u9879\u76ee\u5b58\u5728\u4e89\u8bae"
        assert any(m in text for m in COUNTER_EVIDENCE_MARKERS)
