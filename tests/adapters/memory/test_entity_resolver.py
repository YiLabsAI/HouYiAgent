from __future__ import annotations

from houyi.adapters.memory.entity_resolver import (
    AliasMappingResolver,
    ConditionalResolver,
    DefaultEntityResolver,
    FirstMatchChain,
    NamespacePrefixedResolver,
    RoleBasedEntityResolver,
    SpeakerNameResolver,
    TransformPipeline,
    TurnContext,
    get_default_resolver,
)


def _turn(text: str = "", speaker: str | None = None, **metadata) -> TurnContext:
    return TurnContext(text=text, speaker_id=speaker, metadata=metadata or None)


class TestDefaultResolver:
    def test_returns_user(self) -> None:
        assert DefaultEntityResolver().resolve(_turn(speaker="alice")) == "user"

    def test_module_default(self) -> None:
        assert get_default_resolver().resolve(_turn()) == "user"


class TestSpeakerNameResolver:
    def test_returns_speaker_id(self) -> None:
        assert SpeakerNameResolver().resolve(_turn(speaker="Caroline")) == "Caroline"

    def test_strips_whitespace(self) -> None:
        assert SpeakerNameResolver().resolve(_turn(speaker="  Jon  ")) == "Jon"

    def test_empty_falls_to_user(self) -> None:
        assert SpeakerNameResolver().resolve(_turn(speaker=None)) == "user"
        assert SpeakerNameResolver().resolve(_turn(speaker="   ")) == "user"


class TestRoleBasedResolver:
    def test_canonical_name_passthrough(self) -> None:
        r = RoleBasedEntityResolver(primary="Alice", secondary="Bob")
        assert r.resolve(_turn(speaker="Alice")) == "Alice"
        assert r.resolve(_turn(speaker="Bob")) == "Bob"

    def test_role_indicator_mapping(self) -> None:
        r = RoleBasedEntityResolver(primary="Alice", secondary="Bob")
        assert r.resolve(_turn(speaker="speaker_a")) == "Alice"
        assert r.resolve(_turn(speaker="speaker_b")) == "Bob"
        assert r.resolve(_turn(speaker="primary")) == "Alice"
        assert r.resolve(_turn(speaker="secondary")) == "Bob"

    def test_fuzzy_match(self) -> None:
        r = RoleBasedEntityResolver(primary="Alice")
        assert r.resolve(_turn(speaker="alice_in_wonderland")) == "Alice"

    def test_unknown_falls_to_primary(self) -> None:
        r = RoleBasedEntityResolver(primary="Alice")
        assert r.resolve(_turn(speaker="someone")) == "Alice"
        assert r.resolve(_turn(speaker=None)) == "Alice"

    def test_optional_secondary(self) -> None:
        r = RoleBasedEntityResolver(primary="Alice")
        assert r.resolve(_turn(speaker="speaker_b")) == "Alice"


class TestNamespacePrefixedResolver:
    def test_default_template(self) -> None:
        r = NamespacePrefixedResolver("ws_1")
        assert r.resolve(_turn()) == "ws_1::user"

    def test_custom_template(self) -> None:
        r = NamespacePrefixedResolver("ws_1", template="{ns}/{entity}")
        assert r.resolve(_turn()) == "ws_1/user"

    def test_with_inner(self) -> None:
        r = NamespacePrefixedResolver("ws_1").with_inner(SpeakerNameResolver())
        assert r.resolve(_turn(speaker="Alice")) == "ws_1::Alice"


class TestFirstMatchChain:
    def test_first_non_default_wins(self) -> None:
        chain = FirstMatchChain([DefaultEntityResolver(), SpeakerNameResolver()])
        assert chain.resolve(_turn(speaker="Alice")) == "Alice"

    def test_default_when_all_skip(self) -> None:
        chain = FirstMatchChain([DefaultEntityResolver()], default="anon")
        assert chain.resolve(_turn()) == "anon"

    def test_skip_none_returns_first(self) -> None:
        chain = FirstMatchChain([DefaultEntityResolver()], skip_default=None)
        assert chain.resolve(_turn()) == "user"


class TestTransformPipeline:
    def test_layered_transforms(self) -> None:
        pipeline = TransformPipeline(
            [
                SpeakerNameResolver(),
                NamespacePrefixedResolver("ws_1").with_inner(SpeakerNameResolver()),
            ]
        )
        assert pipeline.resolve(_turn(speaker="Alice")) == "ws_1::Alice"

    def test_empty_pipeline_returns_user(self) -> None:
        assert TransformPipeline([]).resolve(_turn()) == "user"


class TestConditionalResolver:
    def test_routes_on_true(self) -> None:
        r = ConditionalResolver(
            condition=lambda t: bool(t.metadata and t.metadata.get("personal")),
            if_true=SpeakerNameResolver(),
            if_false=DefaultEntityResolver(),
        )
        assert r.resolve(_turn(speaker="Alice", personal=True)) == "Alice"

    def test_routes_on_false(self) -> None:
        r = ConditionalResolver(
            condition=lambda t: False,
            if_true=SpeakerNameResolver(),
            if_false=DefaultEntityResolver(),
        )
        assert r.resolve(_turn(speaker="Alice")) == "user"


class TestAliasMappingResolver:
    def test_alias_applied(self) -> None:
        r = AliasMappingResolver({"Bob": "Robert"})
        assert r.resolve(_turn(speaker="Bob")) == "Robert"

    def test_passthrough_no_match(self) -> None:
        r = AliasMappingResolver({"Bob": "Robert"})
        assert r.resolve(_turn(speaker="Alice")) == "Alice"

    def test_no_passthrough_user(self) -> None:
        r = AliasMappingResolver({"Bob": "Robert"}, passthrough=False)
        assert r.resolve(_turn(speaker="Alice")) == "user"

    def test_with_inner_resolver(self) -> None:
        r = AliasMappingResolver(
            {"Alice": "A"},
            inner=SpeakerNameResolver(),
        )
        assert r.resolve(_turn(speaker="Alice")) == "A"

    def test_no_inner_uses_speaker(self) -> None:
        r = AliasMappingResolver({})
        assert r.resolve(_turn(speaker=None)) == "user"
