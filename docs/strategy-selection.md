# Module Three: Target-aware Strategy Selection

Module three answers four questions only after module two has authorized an intervention:

1. Who should receive the intervention: parent, child, or both?
2. What observable interaction process should be repaired?
3. Which research-coded strategy card is applicable and not contraindicated?
4. Which approved wording should be used?

It does not reclassify the co-regulation state or override module-two timing decisions.

## Research alignment

The versioned library in `config/strategy_cards.yaml` is derived from the final three-stage thematic analysis and formative-study strategy and consequence coding.

| Research strategy family | Module-three representation |
|---|---|
| Parent emotion and tone regulation | Parent-targeted tone and pace card |
| Child emotion recognition and support | Child-targeted pressure-support card |
| Parent-child mutual understanding and relationship repair | Dyad-targeted intention translation and neutral brake cards |
| Need inquiry and expression guidance | Child-targeted needs inquiry card |
| Caregiver burden empathy | Parent-targeted caregiver empathy card |
| Task pacing | Parent or dyad pacing-reset cards |
| Learning strategy support | Parent scaffolding and dyad task-reset cards |
| Autonomy support and role boundaries | Parent autonomy-space and dyad role-restart cards |

Positive reinforcement is not a standalone module-three trigger because the approved module-two policy does not intervene in normal or fluctuation states. It may later be used only as a constrained wording component inside an already authorized intervention.

## Selection contract

Module three receives the original `StateAssessment`, the matching `InterventionDecision`, and the most recent `InterventionPlan` when one exists. Selection is held unless module two sets both `intervention_permitted` and `strategy_selection_required`.

The selector first matches the current `interaction_performance` to the ordered routing rules. A parent- or child-specific card requires evidence explicitly attributed to that actor. `actor=unknown` or `actor=both` cannot be used to single out one person; a neutral dyadic card is selected when one is available.

Target actor and delivery modality are separate responsibilities. Module three records who the strategy addresses; module four applies the research team's fixed prominent-text plus spoken-voice output contract. This prevents individual strategy cards from silently changing the intervention medium.

## Strategy card fields

Each card records:

- applicable states;
- target actor;
- repair target;
- use and avoid conditions;
- one primary action;
- an approved fallback template;
- expected observable recovery indicators;
- research codes and sources;
- a response-gated next strategy when appropriate.

The approved template is checked for length, sentence count, prohibited claims and explicit actor. The current implementation uses approved templates. A later constrained language-model rephrasing layer may change wording only; it may not alter the strategy, target actor, repair target or action.

## Recovery and progressive support

Expected recovery is an observation guide rather than a guaranteed outcome. The recorded categories map directly to the WOZ consequence coding: recovered, partial recovery, not recovered, deteriorated, or indeterminate.

Module two must observe a post-intervention response before another plan can be created. If the state deteriorates, the selector prioritizes the approved neutral dyadic brake. If the response is observed without recovery, the next strategy declared by the previous card is used instead of immediately repeating the same message.

## Offline verification

`strategy-test` accepts the same trajectory JSON format as `trajectory-test`, runs modules two and three together without an API call, and saves:

- `strategy_library.json`;
- `decisions.json`;
- `strategy_selections.json`;
- `intervention_plans.json`;
- `state_trajectory.json`;
- hashes, events, and a summary result.

The included `examples/strategy_replay.json` provides a minimal end-to-end replay.
