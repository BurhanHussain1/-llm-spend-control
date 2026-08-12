"""The registry loads, validates, and prices correctly.

Cost arithmetic is the foundation of every dollar figure this project reports,
so it gets a hand-checked example rather than only property-style assertions.
"""

import pytest

from app.registry import Model, ModelRegistry, RegistryError


def write_registry(tmp_path, body: str):
    path = tmp_path / "models.yaml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = """
baseline_model: strong
models:
  - name: cheap
    provider: mock
    tier: 1
    input_cost_per_mtok: 1.0
    output_cost_per_mtok: 5.0
    typical_latency_ms: 500
    max_context_tokens: 200000
  - name: strong
    provider: mock
    tier: 3
    input_cost_per_mtok: 5.0
    output_cost_per_mtok: 25.0
    typical_latency_ms: 5000
    max_context_tokens: 1000000
"""


# --- cost arithmetic ---------------------------------------------------------


def test_cost_matches_a_hand_calculated_example():
    model = Model(
        name="claude-haiku-4-5",
        provider="anthropic",
        tier=1,
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=5.00,
        typical_latency_ms=900,
        max_context_tokens=200_000,
        supports_vision=True,
        supports_tools=True,
    )

    # 1000/1e6 * 1.00 + 500/1e6 * 5.00 = 0.001 + 0.0025
    assert model.cost(input_tokens=1000, output_tokens=500) == 0.0035


def test_zero_tokens_cost_nothing():
    registry = ModelRegistry.load()
    assert registry.get("claude-opus-5").cost(0, 0) == 0.0


def test_small_requests_do_not_round_away_to_zero():
    registry = ModelRegistry.load()
    cost = registry.get("gpt-4o-mini").cost(input_tokens=10, output_tokens=1)
    assert cost > 0


def test_negative_token_counts_are_rejected():
    registry = ModelRegistry.load()
    with pytest.raises(ValueError):
        registry.get("claude-opus-5").cost(-1, 0)


def test_local_models_are_free():
    registry = ModelRegistry.load()
    assert registry.get("llama3.1:8b").is_free
    assert not registry.get("claude-opus-5").is_free


# --- the shipped registry ----------------------------------------------------


def test_shipped_registry_loads_and_covers_every_tier():
    registry = ModelRegistry.load()

    for tier in (1, 2, 3):
        assert registry.for_tier(tier), f"tier {tier} has no models"

    assert registry.baseline().name == "claude-opus-5"
    assert registry.pricing_as_of != "unknown"


def test_shipped_registry_has_a_full_mock_tier_ladder():
    """The simulation runs entirely on mock models, so all three tiers must exist."""
    registry = ModelRegistry.load()
    mock_tiers = {model.tier for model in registry.for_provider("mock")}
    assert mock_tiers == {1, 2, 3}


def test_for_tier_returns_cheapest_first():
    registry = ModelRegistry.load()
    tier_one = registry.for_tier(1)
    costs = [model.input_cost_per_mtok for model in tier_one]
    assert costs == sorted(costs)


def test_strongest_picks_the_top_tier_of_a_candidate_set():
    registry = ModelRegistry.load()
    mock_models = registry.for_provider("mock")
    assert registry.strongest(among=mock_models).name == "mock-strong"


# --- validation --------------------------------------------------------------


def test_unknown_model_names_the_known_ones(tmp_path):
    registry = ModelRegistry.load(write_registry(tmp_path, MINIMAL))
    with pytest.raises(RegistryError, match="cheap"):
        registry.get("does-not-exist")


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        ModelRegistry.load(tmp_path / "nope.yaml")


def test_duplicate_model_names_are_rejected(tmp_path):
    body = MINIMAL.replace("name: strong", "name: cheap", 1).replace(
        "baseline_model: strong", "baseline_model: cheap"
    )
    with pytest.raises(RegistryError, match="duplicate"):
        ModelRegistry.load(write_registry(tmp_path, body))


def test_invalid_tier_is_rejected(tmp_path):
    body = MINIMAL.replace("tier: 1", "tier: 7")
    with pytest.raises(RegistryError, match="tier"):
        ModelRegistry.load(write_registry(tmp_path, body))


def test_negative_cost_is_rejected(tmp_path):
    body = MINIMAL.replace("input_cost_per_mtok: 1.0", "input_cost_per_mtok: -1.0")
    with pytest.raises(RegistryError, match="negative"):
        ModelRegistry.load(write_registry(tmp_path, body))


def test_missing_field_names_the_model_and_the_field(tmp_path):
    body = MINIMAL.replace("    typical_latency_ms: 500\n", "")
    with pytest.raises(RegistryError, match="cheap.*typical_latency_ms"):
        ModelRegistry.load(write_registry(tmp_path, body))


def test_baseline_must_be_a_defined_model(tmp_path):
    body = MINIMAL.replace("baseline_model: strong", "baseline_model: ghost")
    with pytest.raises(RegistryError, match="baseline_model"):
        ModelRegistry.load(write_registry(tmp_path, body))
