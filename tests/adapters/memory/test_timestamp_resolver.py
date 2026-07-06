"""Tests for deterministic relative-time resolution of event timestamps.

The resolver anchors on the per-turn observation_date and rewrites only
recognized relative expressions; absolute and unrecognized values pass
through unchanged.
"""

from __future__ import annotations

from houyi.adapters.memory.timestamp_resolver import (
    extract_observation_date,
    resolve_relative_timestamp,
)


class TestExtractObservationDate:
    def test_from_json_blob(self):
        blob = '{"observation_date": "2022-01-21", "system_date": "2024-01-15", "text": "x", "speaker_name": "Sam"}'
        assert extract_observation_date(blob) == "2022-01-21"

    def test_plain_text_returns_none(self):
        assert extract_observation_date("I watched it 3 years ago") is None

    def test_missing_field_returns_none(self):
        assert extract_observation_date('{"text": "x"}') is None

    def test_empty(self):
        assert extract_observation_date("") is None
        assert extract_observation_date(None) is None


class TestYearsAgo:
    def test_around_n_years(self):
        assert resolve_relative_timestamp("around 3 years ago", "2022-01-21") == "2019"

    def test_n_years(self):
        assert resolve_relative_timestamp("3 years ago", "2022-01-21") == "2019"

    def test_approximately_n_years(self):
        assert resolve_relative_timestamp("approximately 5 years ago", "2022-01-21") == "2017"

    def test_a_few_years(self):
        # Vague quantifier preserved verbatim; do not invent a fake-precise
        # absolute year. Anchor stays with the session via the phrase itself.
        assert resolve_relative_timestamp("a few years ago", "2023-05-03") == "a few years ago"

    def test_few_years_no_article(self):
        assert resolve_relative_timestamp("few years ago", "2023-05-03") == "few years ago"

    def test_single_year(self):
        assert resolve_relative_timestamp("1 year ago", "2023-05-03") == "2022"


class TestMonthsAgo:
    def test_n_months(self):
        assert resolve_relative_timestamp("2 months ago", "2023-05-04") == "2023-03"

    def test_n_months_year_rollover(self):
        assert resolve_relative_timestamp("2 months ago", "2023-01-15") == "2022-11"

    def test_last_month(self):
        assert resolve_relative_timestamp("last month", "2023-05-04") == "2023-04"

    def test_last_month_year_rollover(self):
        assert resolve_relative_timestamp("last month", "2023-01-15") == "2022-12"


class TestDaysAndWeeks:
    def test_n_days(self):
        assert resolve_relative_timestamp("5 days ago", "2023-05-08") == "2023-05-03"

    def test_yesterday(self):
        assert resolve_relative_timestamp("yesterday", "2023-05-04") == "2023-05-03"

    def test_today(self):
        assert resolve_relative_timestamp("today", "2023-05-04") == "2023-05-04"

    def test_n_weeks(self):
        assert resolve_relative_timestamp("2 weeks ago", "2023-05-18") == "2023-05-04"

    def test_last_week(self):
        assert resolve_relative_timestamp("last week", "2023-05-04") == "2023-04-27"

    def test_last_weekend(self):
        # 2023-05-04 is Thursday; most recent Saturday is 2023-04-29.
        assert resolve_relative_timestamp("last weekend", "2023-05-04") == "2023-04-29"

    def test_last_weekend_sunday(self):
        # 2023-03-26 is Sunday; "last weekend" must skip the current
        # weekend (Mar 25-26) and return the previous Saturday Mar 18.
        assert resolve_relative_timestamp("last weekend", "2023-03-26") == "2023-03-18"

    def test_last_weekend_saturday(self):
        # 2023-03-25 is Saturday; "last weekend" must skip the current
        # weekend starting today and return the previous Saturday Mar 18.
        assert resolve_relative_timestamp("last weekend", "2023-03-25") == "2023-03-18"

    def test_last_weekend_monday(self):
        # 2023-03-27 is Monday; the just-passed weekend is Mar 25-26,
        # so most recent Saturday is Mar 25 (no skip needed).
        assert resolve_relative_timestamp("last weekend", "2023-03-27") == "2023-03-25"


class TestFutureAndLast:
    def test_tomorrow(self):
        assert resolve_relative_timestamp("tomorrow", "2023-05-04") == "2023-05-05"

    def test_next_week(self):
        assert resolve_relative_timestamp("next week", "2023-05-04") == "2023-05-11"

    def test_next_month(self):
        assert resolve_relative_timestamp("next month", "2023-05-04") == "2023-06"

    def test_next_month_year_rollover(self):
        assert resolve_relative_timestamp("next month", "2023-12-15") == "2024-01"

    def test_next_year(self):
        assert resolve_relative_timestamp("next year", "2023-05-04") == "2024"

    def test_last_year(self):
        assert resolve_relative_timestamp("last year", "2023-05-04") == "2022"


class TestPassthrough:
    def test_absolute_year(self):
        assert resolve_relative_timestamp("2019", "2022-01-21") == "2019"

    def test_absolute_iso(self):
        assert resolve_relative_timestamp("2022-01-21", "2023-05-04") == "2022-01-21"

    def test_absolute_month_name(self):
        assert resolve_relative_timestamp("March 26, 2023", "2023-05-04") == "March 26, 2023"

    def test_absolute_dmy(self):
        assert resolve_relative_timestamp("15 March 2020", "2023-05-04") == "15 March 2020"

    def test_unrecognized(self):
        assert resolve_relative_timestamp("unspecified", "2023-05-04") == "unspecified"

    def test_unknown_verbatim(self):
        assert resolve_relative_timestamp("a few years", "2023-05-04") == "a few years"

    def test_empty_observation(self):
        assert resolve_relative_timestamp("3 years ago", "") == "3 years ago"

    def test_none_observation(self):
        assert resolve_relative_timestamp("3 years ago", None) == "3 years ago"

    def test_unparseable_observation(self):
        assert resolve_relative_timestamp("3 years ago", "not-a-date") == "3 years ago"

    def test_empty_timestamp(self):
        assert resolve_relative_timestamp("", "2023-05-04") == ""
