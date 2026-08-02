from coregulation_poc.codebook import load_state_codebook


def test_codebook_contains_four_states() -> None:
    codebook = load_state_codebook()
    assert set(codebook["states"]) == {
        "normal",
        "fluctuation",
        "dysregulation",
        "high_risk",
    }
    assert codebook["states"]["high_risk"]["history_required"] is True
    assert codebook["evidence_policy"]["require_both_modalities"] is False
    assert codebook["intervention_policy"]["no_intervention_states"] == [
        "normal",
        "fluctuation",
    ]
