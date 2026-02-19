"""Tests for SkillService consent management."""

from __future__ import annotations

import asyncio

import pytest


class TestConsentManagement:
    def test_create_consent_request(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="web_search",
            tool_name="search",
            reason="Needs network access",
            permissions=["network"],
        )
        assert req_id.startswith("consent_")
        assert req_id in skill_service._pending_consents

    def test_respond_to_consent(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="web_search",
            tool_name="search",
            reason="test",
            permissions=["network"],
        )
        found = skill_service.respond_to_consent(req_id, granted=True, remember=True)
        assert found is True
        req = skill_service._pending_consents[req_id]
        assert req.granted is True
        assert req.remember is True

    def test_respond_to_nonexistent(self, skill_service):
        found = skill_service.respond_to_consent("nonexistent", granted=True)
        assert found is False

    @pytest.mark.asyncio
    async def test_wait_for_consent_granted(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="test",
            tool_name="tool",
            reason="test",
            permissions=[],
        )

        async def respond_later():
            await asyncio.sleep(0.05)
            skill_service.respond_to_consent(req_id, granted=True, remember=False)

        asyncio.create_task(respond_later())
        granted, remember = await skill_service.wait_for_consent(req_id, timeout=2.0)
        assert granted is True
        assert remember is False

    @pytest.mark.asyncio
    async def test_wait_for_consent_timeout(self, skill_service):
        req_id = skill_service.create_consent_request(
            skill_name="test",
            tool_name="tool",
            reason="test",
            permissions=[],
        )
        granted, remember = await skill_service.wait_for_consent(req_id, timeout=0.1)
        assert granted is False
        assert remember is False
        assert req_id not in skill_service._pending_consents

    @pytest.mark.asyncio
    async def test_wait_for_nonexistent(self, skill_service):
        granted, remember = await skill_service.wait_for_consent("nonexistent")
        assert granted is False
