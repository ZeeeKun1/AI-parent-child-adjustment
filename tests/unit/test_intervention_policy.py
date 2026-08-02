from coregulation_poc.control.intervention_policy import load_intervention_policy
from coregulation_poc.models import CoregulationState, InterventionAction


def test_policy_maps_all_four_research_states() -> None:
    policy = load_intervention_policy()

    assert policy.state_actions[CoregulationState.NORMAL].action is (
        InterventionAction.NO_INTERVENTION
    )
    assert policy.state_actions[CoregulationState.FLUCTUATION].action is (
        InterventionAction.OBSERVE
    )
    assert policy.state_actions[CoregulationState.DYSREGULATION].action is (
        InterventionAction.INTERVENE
    )
    assert policy.state_actions[CoregulationState.HIGH_RISK].action is (
        InterventionAction.PROGRESSIVE_SUPPORT
    )


def test_policy_preserves_research_boundaries() -> None:
    policy = load_intervention_policy()

    assert policy.principles.use_hard_time_or_count_thresholds is False
    assert policy.principles.use_single_signal_as_trigger is False
    assert policy.principles.require_natural_turn_boundary_for_intervention is True
    assert policy.principles.require_post_intervention_response_before_repeat is True
    assert policy.state_actions[CoregulationState.HIGH_RISK].history_required is True
