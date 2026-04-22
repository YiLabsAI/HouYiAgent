"""Tests for SimpleSkill consent system.

Reference: SimpleSkill Specification 0.1.0 Section 5.3
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from houyi.domain.skill.consent import (
    CLIConsentHandler,
    ConsentManager,
    ConsentRequest,
    ConsentResponse,
    ConsentResult,
    ConsentType,
    FileConsentStore,
    InMemoryConsentStore,
    PolicyBasedConsentHandler,
)
from houyi.domain.skill.policy import (
    InvocationPolicy,
    Permissions,
    SideEffect,
)


class TestConsentRequest:
    """Test ConsentRequest dataclass."""

    def test_describe_permission_grant(self):
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="file_writer",
            permissions=Permissions(),
        )
        description = request.describe()
        assert "file_writer" in description
        assert "permissions" in description.lower()

    def test_describe_operation_confirm(self):
        request = ConsentRequest(
            consent_type=ConsentType.OPERATION_CONFIRM,
            skill_name="file_writer",
            operation="delete /tmp/important.txt",
        )
        description = request.describe()
        assert "file_writer" in description
        assert "delete" in description

    def test_describe_invoke_confirm(self):
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="network_tool",
            policy=InvocationPolicy(side_effect=SideEffect.NETWORK),
        )
        description = request.describe()
        assert "network_tool" in description
        assert "network" in description.lower()


class TestConsentResponse:
    """Test ConsentResponse dataclass."""

    def test_is_granted(self):
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test",
        )

        response = ConsentResponse(result=ConsentResult.GRANTED, request=request)
        assert response.is_granted()

        response = ConsentResponse(result=ConsentResult.REMEMBERED, request=request)
        assert response.is_granted()

        response = ConsentResponse(result=ConsentResult.DENIED, request=request)
        assert not response.is_granted()

        response = ConsentResponse(result=ConsentResult.TIMEOUT, request=request)
        assert not response.is_granted()

    def test_to_dict(self):
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
            operation="test_operation",
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        data = response.to_dict()
        assert data["result"] == "granted"
        assert data["skill_name"] == "test_skill"
        assert data["consent_type"] == "invoke_confirm"


class TestInMemoryConsentStore:
    """Test InMemoryConsentStore."""

    def test_save_and_load(self):
        store = InMemoryConsentStore()
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )

        store.save(response)

        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is not None
        assert loaded.result == ConsentResult.REMEMBERED  # Loaded consents are "remembered"

    def test_not_saved_if_denied(self):
        store = InMemoryConsentStore()
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.DENIED,
            request=request,
        )

        store.save(response)

        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is None

    def test_not_saved_without_remember(self):
        store = InMemoryConsentStore()
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=False,  # Not remembering
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )

        store.save(response)

        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is None

    def test_revoke_specific(self):
        store = InMemoryConsentStore()
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        store.save(response)

        store.revoke("test_skill", ConsentType.PERMISSION_GRANT)

        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is None

    def test_revoke_all(self):
        store = InMemoryConsentStore()

        # Save multiple consent types
        for consent_type in [ConsentType.PERMISSION_GRANT, ConsentType.INVOKE_CONFIRM]:
            request = ConsentRequest(
                consent_type=consent_type,
                skill_name="test_skill",
                remember=True,
            )
            response = ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )
            store.save(response)

        store.revoke("test_skill")  # Revoke all

        assert store.load("test_skill", ConsentType.PERMISSION_GRANT) is None
        assert store.load("test_skill", ConsentType.INVOKE_CONFIRM) is None


class TestPolicyBasedConsentHandler:
    """Test PolicyBasedConsentHandler."""

    @pytest.mark.asyncio
    async def test_auto_grant(self):
        handler = PolicyBasedConsentHandler(
            auto_grant_skills={"trusted_skill"},
        )
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="trusted_skill",
        )

        response = await handler.request_consent(request)
        assert response.is_granted()
        assert "policy" in (response.reason or "").lower()

    @pytest.mark.asyncio
    async def test_auto_deny(self):
        handler = PolicyBasedConsentHandler(
            auto_deny_skills={"dangerous_skill"},
        )
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="dangerous_skill",
        )

        response = await handler.request_consent(request)
        assert not response.is_granted()
        assert "denied" in (response.reason or "").lower()

    @pytest.mark.asyncio
    async def test_default_deny(self):
        handler = PolicyBasedConsentHandler(
            default_grant=False,
        )
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="unknown_skill",
        )

        response = await handler.request_consent(request)
        assert not response.is_granted()

    @pytest.mark.asyncio
    async def test_default_grant(self):
        handler = PolicyBasedConsentHandler(
            default_grant=True,
        )
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="unknown_skill",
        )

        response = await handler.request_consent(request)
        assert response.is_granted()


class TestConsentManager:
    """Test ConsentManager."""

    @pytest.mark.asyncio
    async def test_non_interactive_mode(self):
        manager = ConsentManager(interactive=False)
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
        )

        response = await manager.request_consent(request)
        assert response.result == ConsentResult.NOT_INTERACTIVE

    @pytest.mark.asyncio
    async def test_check_remembered_consent(self):
        store = InMemoryConsentStore()

        # Pre-populate store
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        initial_response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        store.save(initial_response)

        manager = ConsentManager(store=store, interactive=False)

        # Should return remembered consent
        new_request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
        )
        response = await manager.request_consent(new_request)
        assert response.result == ConsentResult.REMEMBERED

    @pytest.mark.asyncio
    async def test_with_policy_handler(self):
        handler = PolicyBasedConsentHandler(
            auto_grant_skills={"trusted_skill"},
        )
        manager = ConsentManager(handler=handler)

        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="trusted_skill",
        )

        response = await manager.request_consent(request)
        assert response.is_granted()

    def test_check_permission(self):
        store = InMemoryConsentStore()

        # Pre-populate store
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        store.save(response)

        manager = ConsentManager(store=store)

        assert manager.check_permission("test_skill")
        assert not manager.check_permission("other_skill")

    @pytest.mark.asyncio
    async def test_audit_log(self):
        handler = PolicyBasedConsentHandler(default_grant=True)
        manager = ConsentManager(handler=handler)

        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
        )

        await manager.request_consent(request)

        audit_log = manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0].request.skill_name == "test_skill"

    def test_revoke_consent(self):
        store = InMemoryConsentStore()

        # Pre-populate store
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        store.save(response)

        manager = ConsentManager(store=store)

        assert manager.check_permission("test_skill")
        manager.revoke_consent("test_skill")
        assert not manager.check_permission("test_skill")

    @pytest.mark.asyncio
    async def test_no_handler_configured(self):
        """Test consent request when no handler is configured."""
        manager = ConsentManager(handler=None, interactive=True)
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
        )

        response = await manager.request_consent(request)
        assert response.result == ConsentResult.DENIED
        assert "no consent handler" in (response.reason or "").lower()

    @pytest.mark.asyncio
    async def test_stores_remembered_consent(self):
        """Test that manager stores consent when remember=True."""
        handler = PolicyBasedConsentHandler(default_grant=True)
        store = InMemoryConsentStore()
        manager = ConsentManager(handler=handler, store=store)

        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )

        await manager.request_consent(request)

        # Verify it was stored
        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is not None
        assert loaded.result == ConsentResult.REMEMBERED

    def test_export_audit_log(self):
        """Test exporting audit log to file."""
        manager = ConsentManager()

        # Add some entries to audit log
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
        )
        manager._audit(response)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "nested" / "audit.json"
            manager.export_audit_log(export_path)

            assert export_path.exists()
            with open(export_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["skill_name"] == "test_skill"


class TestInMemoryConsentStoreExpiration:
    """Test consent expiration handling."""

    def test_expired_consent_is_removed(self):
        """Test that expired consent is removed on load."""
        store = InMemoryConsentStore()

        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
            expires_at=datetime.now(UTC) - timedelta(hours=1),  # Already expired
        )

        # Force save by directly setting in store
        key = (request.skill_name, request.consent_type.value)
        store._consents[key] = response

        # Load should return None for expired
        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is None

        # Verify it was removed
        assert key not in store._consents

    def test_valid_consent_not_expired(self):
        """Test that non-expired consent is returned."""
        store = InMemoryConsentStore()

        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            remember=True,
        )
        response = ConsentResponse(
            result=ConsentResult.GRANTED,
            request=request,
            expires_at=datetime.now(UTC) + timedelta(hours=1),  # Future expiration
        )

        # Force save
        key = (request.skill_name, request.consent_type.value)
        store._consents[key] = response

        loaded = store.load("test_skill", ConsentType.PERMISSION_GRANT)
        assert loaded is not None
        assert loaded.result == ConsentResult.REMEMBERED


class TestFileConsentStore:
    """Test FileConsentStore for persistence."""

    def test_save_and_load(self):
        """Test saving and loading consent from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"
            store = FileConsentStore(store_path)

            request = ConsentRequest(
                consent_type=ConsentType.PERMISSION_GRANT,
                skill_name="file_test_skill",
                remember=True,
            )
            response = ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )

            store.save(response)

            # Verify file was created
            assert store_path.exists()

            # Load should return the consent
            loaded = store.load("file_test_skill", ConsentType.PERMISSION_GRANT)
            assert loaded is not None
            assert loaded.result == ConsentResult.REMEMBERED

    def test_persistence_across_instances(self):
        """Test that consent persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"

            # First instance saves
            store1 = FileConsentStore(store_path)
            request = ConsentRequest(
                consent_type=ConsentType.PERMISSION_GRANT,
                skill_name="persistent_skill",
                remember=True,
            )
            response = ConsentResponse(
                result=ConsentResult.GRANTED,
                request=request,
            )
            store1.save(response)

            # Second instance loads
            store2 = FileConsentStore(store_path)
            loaded = store2.load("persistent_skill", ConsentType.PERMISSION_GRANT)
            assert loaded is not None
            assert loaded.result == ConsentResult.REMEMBERED

    def test_expired_removed_from_file(self):
        """Test that expired consent is removed from file on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"

            # Create expired consent directly in file
            expired_data = {
                "persistent_skill:permission_grant": {
                    "result": "granted",
                    "skill_name": "persistent_skill",
                    "consent_type": "permission_grant",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "expires_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                    "remember": True,
                }
            }
            store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(store_path, "w") as f:
                json.dump(expired_data, f)

            store = FileConsentStore(store_path)
            loaded = store.load("persistent_skill", ConsentType.PERMISSION_GRANT)

            assert loaded is None

    def test_revoke_specific_type(self):
        """Test revoking specific consent type from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"
            store = FileConsentStore(store_path)

            # Save two different consent types
            for ct in [ConsentType.PERMISSION_GRANT, ConsentType.INVOKE_CONFIRM]:
                request = ConsentRequest(
                    consent_type=ct,
                    skill_name="multi_consent_skill",
                    remember=True,
                )
                response = ConsentResponse(
                    result=ConsentResult.GRANTED,
                    request=request,
                )
                store.save(response)

            # Revoke only one type
            store.revoke("multi_consent_skill", ConsentType.PERMISSION_GRANT)

            assert store.load("multi_consent_skill", ConsentType.PERMISSION_GRANT) is None
            assert store.load("multi_consent_skill", ConsentType.INVOKE_CONFIRM) is not None

    def test_revoke_all_types(self):
        """Test revoking all consent types from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"
            store = FileConsentStore(store_path)

            # Save two different consent types
            for ct in [ConsentType.PERMISSION_GRANT, ConsentType.INVOKE_CONFIRM]:
                request = ConsentRequest(
                    consent_type=ct,
                    skill_name="all_revoke_skill",
                    remember=True,
                )
                response = ConsentResponse(
                    result=ConsentResult.GRANTED,
                    request=request,
                )
                store.save(response)

            # Revoke all
            store.revoke("all_revoke_skill")

            assert store.load("all_revoke_skill", ConsentType.PERMISSION_GRANT) is None
            assert store.load("all_revoke_skill", ConsentType.INVOKE_CONFIRM) is None

    def test_load_invalid_json_file(self):
        """Test handling of invalid JSON in consent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "consent.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("{ invalid json }")

            # Should not crash, just have empty consents
            store = FileConsentStore(store_path)
            assert store.load("any_skill", ConsentType.PERMISSION_GRANT) is None

    def test_default_path(self):
        """Test that default path is used when none provided."""
        with patch.object(Path, "home", return_value=Path("/tmp/test_home")):
            with patch.object(Path, "exists", return_value=False):
                store = FileConsentStore()
                assert ".houyi" in str(store._path)
                assert "consent.json" in str(store._path)


class TestCLIConsentHandler:
    """Test CLIConsentHandler with mocked input."""

    @pytest.mark.asyncio
    async def test_grant_consent(self):
        """Test granting consent via CLI."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", return_value="y"):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.GRANTED

    @pytest.mark.asyncio
    async def test_grant_yes(self):
        """Test granting with 'yes' input."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", return_value="yes"):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.GRANTED

    @pytest.mark.asyncio
    async def test_remember_consent(self):
        """Test remembering consent via CLI."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", return_value="r"):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.GRANTED
        assert request.remember is True

    @pytest.mark.asyncio
    async def test_remember_full_word(self):
        """Test remembering with 'remember' input."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", return_value="remember"):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.GRANTED
        assert request.remember is True

    @pytest.mark.asyncio
    async def test_deny_consent(self):
        """Test denying consent via CLI."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", return_value="n"):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.DENIED

    @pytest.mark.asyncio
    async def test_eof_error(self):
        """Test handling EOFError during input."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", side_effect=EOFError):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.DENIED
        assert "cancelled" in (response.reason or "").lower()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt(self):
        """Test handling KeyboardInterrupt during input."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            response = await handler.request_consent(request)

        assert response.result == ConsentResult.DENIED

    def test_check_remembered_returns_none(self):
        """Test that check_remembered always returns None for CLI handler."""
        handler = CLIConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="cli_skill",
        )

        result = handler.check_remembered(request)
        assert result is None


class TestPolicyBasedConsentHandlerExtended:
    """Extended tests for PolicyBasedConsentHandler."""

    def test_check_remembered_returns_none(self):
        """Test that check_remembered always returns None for policy handler."""
        handler = PolicyBasedConsentHandler()
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
        )

        result = handler.check_remembered(request)
        assert result is None


class TestConsentRequestDescribe:
    """Test ConsentRequest.describe() method variations."""

    def test_describe_with_empty_perms(self):
        """Test describe for permission grant with no specific permissions."""
        request = ConsentRequest(
            consent_type=ConsentType.PERMISSION_GRANT,
            skill_name="test_skill",
            permissions=None,
        )
        description = request.describe()
        assert "test_skill" in description
        assert "No specific permissions" in description

    def test_describe_invoke_no_policy(self):
        """Test describe for invoke confirm without policy."""
        request = ConsentRequest(
            consent_type=ConsentType.INVOKE_CONFIRM,
            skill_name="test_skill",
            policy=None,
        )
        description = request.describe()
        assert "test_skill" in description
        assert "unknown" in description.lower()

    def test_describe_default_case(self):
        """Test describe for custom/unknown consent type scenario."""
        # Create a request and manually check the fallback branch
        request = ConsentRequest(
            consent_type=ConsentType.OPERATION_CONFIRM,
            skill_name="test_skill",
            operation=None,
        )
        description = request.describe()
        # Should still produce a valid description
        assert "test_skill" in description
