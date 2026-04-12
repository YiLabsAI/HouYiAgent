"""Tests for houyi.arena.bench2_compat.extract.clean_fact."""

from __future__ import annotations

import pytest

from houyi.arena.bench2_compat.extract import _MIN_FACT_CHARS, clean_fact, remove_urls


def _process(fact: str) -> str:
    return clean_fact(remove_urls(fact))


# ---------------------------------------------------------------------------
# Leading URL-tail fragments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_prefix",
    [
        # html) prefix
        (
            "html)。尽管整体收入水平提升，但城乡二元结构依然是影响阶层收入分布的核心变量。",
            "尽管整体收入水平提升",
        ),
        # html) prefix short sentence
        (
            "html)。这意味着底层阶层的财务脆弱性依然极高",
            "这意味着底层阶层的财务脆弱性依然极高",
        ),
        # pdf) prefix
        (
            "pdf)与家庭金融调查数据可以看出，家庭杠杆率随着房产占比的上升而居高不下。",
            "与家庭金融调查数据可以看出",
        ),
        # URL path + ) prefix
        (
            "com/news/china-middle-class-growth-policy-and-consumption/)，但收入增长往往被资产价格和生活成本的上涨所抵消。",
            "但收入增长往往被资产价格和生活成本的上涨所抵消",
        ),
        # org path + ) prefix
        (
            'org/yingjie-guo/3181/article)。这种政治上的"安全性"虽然促进了该概念的普及',
            "这种政治上的",
        ),
    ],
)
def test_strips_leading_url_tail(raw: str, expected_prefix: str) -> None:
    result = _process(raw)
    assert result.startswith(expected_prefix), repr(result)
    assert len(result) >= _MIN_FACT_CHARS


# ---------------------------------------------------------------------------
# Trailing truncated URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "2025年全国居民人均可支配收入为43377元 [title](https://www",
        '2025年全国居民人均可支配收入为43377元，而处于正中间位置的"中间收入组"人均收入仅为35536元 [中华人民共和国2025年国民经济和社会发展统计公报](https://www',
    ],
)
def test_strips_trailing_truncated_url(raw: str) -> None:
    result = _process(raw)
    assert "https://" not in result
    assert len(result) >= _MIN_FACT_CHARS


# ---------------------------------------------------------------------------
# Leading ref token
# ---------------------------------------------------------------------------


def test_strips_leading_ref_token() -> None:
    raw = "[ref_5cfbcf88]。对于中产阶级而言，重大疾病是击穿家庭财务防线的主要风险之一"
    result = _process(raw)
    assert result.startswith("对于中产阶级而言")


def test_strips_inline_ref_token() -> None:
    raw = "8万户[ref_edc897e8]。这一极小比例的高净值群体占据了社会财富的顶端。"
    result = _process(raw)
    assert "[ref_" not in result
    assert "这一极小比例的高净值群体" in result


# ---------------------------------------------------------------------------
# Pure URL path – should be discarded (returns empty → filtered out)
# ---------------------------------------------------------------------------


def test_discards_pure_url_path() -> None:
    raw = "org/core/journals/japanese-journal-of-political-science/article/chinas-middle-class-unified-or-fragmented/F1A646014C8B1E"
    result = _process(raw)
    assert len(result) < _MIN_FACT_CHARS


# ---------------------------------------------------------------------------
# Clean facts should not be altered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "中国中产阶级家庭资产配置的最显著特征是实物资产占据绝对主导地位，房产成为家庭财富的核心载体。",
        "2025年城乡居民人均可支配收入比值为2.21，比上年缩小0.04。",
        "住房问题是中产阶级财富焦虑的核心，高杠杆的财务结构极大地限制了家庭抵御风险的能力。",
    ],
)
def test_clean_fact_unchanged_for_clean_input(raw: str) -> None:
    result = _process(raw)
    assert result == raw
