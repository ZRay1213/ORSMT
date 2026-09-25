from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from .constructive import ContractStatus, CounterpartStatus, ReformulationVerdict
from .evidence import EvidenceRecord, Relation, Verdict, aggregate


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class RelationProfile:
    total: int
    applicable: int
    decided: int
    certified: int
    refuted: int
    undecided: int
    not_applicable: int
    availability_rate_all: float
    not_applicable_rate_all: float
    certified_rate_all: float
    refuted_rate_all: float
    undecided_rate_all: float
    certified_rate_applicable: float
    refuted_rate_applicable: float
    undecided_rate_applicable: float
    coverage_rate_applicable: float
    refuted_rate_decided: float


def relation_profile(records: Iterable[EvidenceRecord]) -> RelationProfile:
    rows = list(records)
    total = len(rows)
    applicable_rows = [record for record in rows if record.reason != "not_applicable"]
    applicable = len(applicable_rows)
    not_applicable = total - applicable
    certified = sum(record.verdict == Verdict.CERTIFIED for record in applicable_rows)
    refuted = sum(record.verdict == Verdict.REFUTED for record in applicable_rows)
    undecided = sum(record.verdict == Verdict.UNDECIDED for record in applicable_rows)
    decided = certified + refuted
    return RelationProfile(
        total=total,
        applicable=applicable,
        decided=decided,
        certified=certified,
        refuted=refuted,
        undecided=undecided,
        not_applicable=not_applicable,
        availability_rate_all=_ratio(applicable, total),
        not_applicable_rate_all=_ratio(not_applicable, total),
        certified_rate_all=_ratio(certified, total),
        refuted_rate_all=_ratio(refuted, total),
        undecided_rate_all=_ratio(undecided, total),
        certified_rate_applicable=_ratio(certified, applicable),
        refuted_rate_applicable=_ratio(refuted, applicable),
        undecided_rate_applicable=_ratio(undecided, applicable),
        coverage_rate_applicable=_ratio(decided, applicable),
        refuted_rate_decided=_ratio(refuted, decided),
    )


@dataclass(frozen=True)
class SemanticConstraintMetrics:
    reference_assigned: int
    reference_total: int
    reference_not_applicable: int
    candidate_assigned: int
    candidate_total: int
    candidate_not_applicable: int
    certified_reference: int
    refuted_reference: int
    undecided_reference: int
    certified_candidate: int
    semantic_recall: float
    semantic_precision: float
    semantic_f1: float


def semantic_constraint_metrics(
    reference_queries: Sequence[EvidenceRecord], candidate_queries: Sequence[EvidenceRecord]
) -> SemanticConstraintMetrics:
    """Compute exact semantic constraint recall and precision.

    Reference queries ask whether the candidate entails each source obligation.
    Candidate queries ask whether the source/reference entails each candidate row.
    Every supplied query remains in its denominator, including undecided records.
    """

    applicable_reference = [
        row for row in reference_queries if row.reason != "not_applicable"
    ]
    applicable_candidate = [
        row for row in candidate_queries if row.reason != "not_applicable"
    ]
    ref_total = len(applicable_reference)
    cand_total = len(applicable_candidate)
    certified_ref = sum(
        row.verdict == Verdict.CERTIFIED for row in applicable_reference
    )
    refuted_ref = sum(row.verdict == Verdict.REFUTED for row in applicable_reference)
    undecided_ref = ref_total - certified_ref - refuted_ref
    certified_cand = sum(
        row.verdict == Verdict.CERTIFIED for row in applicable_candidate
    )
    recall = _ratio(certified_ref, ref_total)
    precision = _ratio(certified_cand, cand_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SemanticConstraintMetrics(
        reference_assigned=len(reference_queries),
        reference_total=ref_total,
        reference_not_applicable=len(reference_queries) - ref_total,
        candidate_assigned=len(candidate_queries),
        candidate_total=cand_total,
        candidate_not_applicable=len(candidate_queries) - cand_total,
        certified_reference=certified_ref,
        refuted_reference=refuted_ref,
        undecided_reference=undecided_ref,
        certified_candidate=certified_cand,
        semantic_recall=recall,
        semantic_precision=precision,
        semantic_f1=f1,
    )


@dataclass(frozen=True)
class PipelineProfile:
    by_relation: Mapping[Relation, RelationProfile]
    end_to_end: Verdict
    certified_relations: int
    refuted_relations: int
    undecided_relations: int


def pipeline_profile(records: Iterable[EvidenceRecord]) -> PipelineProfile:
    rows = list(records)
    by_relation: dict[Relation, RelationProfile] = {}
    relation_verdicts: list[Verdict] = []
    for relation in Relation:
        selected = [row for row in rows if row.relation == relation]
        if not selected:
            continue
        by_relation[relation] = relation_profile(selected)
        relation_verdicts.append(aggregate(selected))
    if any(verdict == Verdict.REFUTED for verdict in relation_verdicts):
        end_to_end = Verdict.REFUTED
    elif relation_verdicts and all(verdict == Verdict.CERTIFIED for verdict in relation_verdicts):
        end_to_end = Verdict.CERTIFIED
    else:
        end_to_end = Verdict.UNDECIDED
    return PipelineProfile(
        by_relation=by_relation,
        end_to_end=end_to_end,
        certified_relations=sum(v == Verdict.CERTIFIED for v in relation_verdicts),
        refuted_relations=sum(v == Verdict.REFUTED for v in relation_verdicts),
        undecided_relations=sum(v == Verdict.UNDECIDED for v in relation_verdicts),
    )


@dataclass(frozen=True)
class ValuePath:
    source_optimum: Fraction | None
    written_optimum: Fraction | None
    runtime_optimum: Fraction | None
    reported_value: Fraction | None
    source_to_written_gap: float | None
    written_to_runtime_gap: float | None
    runtime_to_reported_gap: float | None
    first_gap: str | None
    first_gap_signed: float | None


def _relative_gap(left: Fraction | None, right: Fraction | None) -> float | None:
    if left is None or right is None:
        return None
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left - right)) / scale


def _signed_relative_gap(left: Fraction | None, right: Fraction | None) -> float | None:
    if left is None or right is None:
        return None
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return float(right - left) / scale


def value_path(
    source_optimum: Fraction | None,
    written_optimum: Fraction | None,
    runtime_optimum: Fraction | None,
    reported_value: Fraction | None,
    tolerance: float = 0.0,
) -> ValuePath:
    transitions = (
        ("source_to_written", source_optimum, written_optimum),
        ("written_to_runtime", written_optimum, runtime_optimum),
        ("runtime_to_reported", runtime_optimum, reported_value),
    )
    first_gap = None
    first_gap_signed = None
    for label, left, right in transitions:
        signed = _signed_relative_gap(left, right)
        if signed is not None and abs(signed) > tolerance:
            first_gap = label
            first_gap_signed = signed
            break
    return ValuePath(
        source_optimum,
        written_optimum,
        runtime_optimum,
        reported_value,
        _relative_gap(source_optimum, written_optimum),
        _relative_gap(written_optimum, runtime_optimum),
        _relative_gap(runtime_optimum, reported_value),
        first_gap,
        first_gap_signed,
    )


def masked_semantic_error(value_agrees: bool, records: Iterable[EvidenceRecord]) -> bool:
    return value_agrees and any(record.verdict == Verdict.REFUTED for record in records)


@dataclass(frozen=True)
class FailureAttribution:
    label: str
    failed_relations: tuple[Relation, ...]
    masked_by_value_agreement: bool


def failure_attribution(
    records: Iterable[EvidenceRecord], value_agrees: bool = False
) -> FailureAttribution:
    failed = tuple(
        relation
        for relation in Relation
        if any(
            row.relation == relation and row.verdict == Verdict.REFUTED
            for row in records
        )
    )
    if not failed:
        label = "no_refutation"
    elif len(failed) > 1:
        label = "multilayer"
    else:
        label = failed[0].value
    return FailureAttribution(label, failed, bool(value_agrees and failed))


@dataclass(frozen=True)
class DetectionOverlap:
    universe: int
    ours: int
    baseline: int
    shared: int
    ours_only: int
    baseline_only: int
    union: int


def detection_overlap(
    universe_ids: Iterable[str], ours_ids: Iterable[str], baseline_ids: Iterable[str]
) -> DetectionOverlap:
    universe = set(universe_ids)
    ours = set(ours_ids)
    baseline = set(baseline_ids)
    if not ours <= universe or not baseline <= universe:
        raise ValueError("detected ids must be subsets of the evaluation universe")
    return DetectionOverlap(
        universe=len(universe),
        ours=len(ours),
        baseline=len(baseline),
        shared=len(ours & baseline),
        ours_only=len(ours - baseline),
        baseline_only=len(baseline - ours),
        union=len(ours | baseline),
    )


@dataclass(frozen=True)
class SeverityProfile:
    refutations: int
    with_margin: int
    mean_margin: float | None
    median_margin: float | None
    max_margin: float | None
    with_normalized_margin: int
    mean_normalized_margin: float | None
    median_normalized_margin: float | None


def severity_profile(records: Iterable[EvidenceRecord]) -> SeverityProfile:
    refuted = [row for row in records if row.verdict == Verdict.REFUTED]
    margins = [float(row.violation_margin) for row in refuted if row.violation_margin is not None]
    normalized = [
        row.normalized_violation
        for row in refuted
        if row.normalized_violation is not None
    ]
    return SeverityProfile(
        refutations=len(refuted),
        with_margin=len(margins),
        mean_margin=mean(margins) if margins else None,
        median_margin=median(margins) if margins else None,
        max_margin=max(margins) if margins else None,
        with_normalized_margin=len(normalized),
        mean_normalized_margin=mean(normalized) if normalized else None,
        median_normalized_margin=median(normalized) if normalized else None,
    )


@dataclass(frozen=True)
class ConstraintErrorProfile:
    total_constraints: int
    applicable_constraints: int
    not_applicable_constraints: int
    erroneous_constraints: int
    certified_constraints: int
    undecided_constraints: int
    error_rate_all: float
    error_rate_applicable: float
    certified_rate_all: float
    undecided_rate_all: float
    coverage_rate_applicable: float


def constraint_error_profile(records: Iterable[EvidenceRecord]) -> ConstraintErrorProfile:
    profile = relation_profile(records)
    return ConstraintErrorProfile(
        total_constraints=profile.total,
        applicable_constraints=profile.applicable,
        not_applicable_constraints=profile.not_applicable,
        erroneous_constraints=profile.refuted,
        certified_constraints=profile.certified,
        undecided_constraints=profile.undecided,
        error_rate_all=profile.refuted_rate_all,
        error_rate_applicable=profile.refuted_rate_applicable,
        certified_rate_all=profile.certified_rate_all,
        undecided_rate_all=profile.undecided_rate_all,
        coverage_rate_applicable=profile.coverage_rate_applicable,
    )


@dataclass(frozen=True)
class CorrespondenceProposalProfile:
    """Coverage and exact adjudication for nonidentity correspondences.

    Every row is a pair for which direct identity alignment is unavailable.
    Proposal generation and schema validation are measured separately from the
    downstream SMT verdict, so a mapper cannot receive credit for deciding its
    own proposal.
    """

    routed_pairs: int
    returned_pairs: int
    missing_pairs: int
    admitted_pairs: int
    ill_typed_pairs: int
    certified_pairs: int
    refuted_pairs: int
    undecided_pairs: int
    decided_pairs: int
    return_rate: float
    proposal_coverage: float
    well_typed_rate_returned: float
    adjudication_decision_rate: float


def correspondence_proposal_profile(
    proposed: Sequence[bool],
    well_typed: Sequence[bool],
    verdicts: Sequence[Verdict | None],
) -> CorrespondenceProposalProfile:
    """Summarize a proposal route over its full required-pair denominator.

    ``proposed`` records whether the mapper returned a correspondence;
    ``well_typed`` records deterministic schema admission. Missing or
    ill-typed proposals have no downstream SMT query and therefore use a null
    verdict rather than ``UNDECIDED``. Admitted proposals must receive one
    exact C/R/U adjudication.
    """

    if len({len(proposed), len(well_typed), len(verdicts)}) != 1:
        raise ValueError("correspondence metric inputs must have equal length")
    for index, (has_proposal, typed, verdict) in enumerate(
        zip(proposed, well_typed, verdicts)
    ):
        if typed and not has_proposal:
            raise ValueError(f"well-typed correspondence {index} was not proposed")
        if not typed and verdict is not None:
            raise ValueError(
                f"unavailable correspondence {index} cannot receive an SMT verdict"
            )
        if typed and verdict is None:
            raise ValueError(
                f"admitted correspondence {index} is missing an SMT adjudication"
            )

    total = len(verdicts)
    returned_pairs = sum(proposed)
    admitted_pairs = sum(well_typed)
    certified = sum(verdict == Verdict.CERTIFIED for verdict in verdicts)
    refuted = sum(verdict == Verdict.REFUTED for verdict in verdicts)
    undecided = sum(verdict == Verdict.UNDECIDED for verdict in verdicts)
    decided = certified + refuted
    return CorrespondenceProposalProfile(
        routed_pairs=total,
        returned_pairs=returned_pairs,
        missing_pairs=total - returned_pairs,
        admitted_pairs=admitted_pairs,
        ill_typed_pairs=returned_pairs - admitted_pairs,
        certified_pairs=certified,
        refuted_pairs=refuted,
        undecided_pairs=undecided,
        decided_pairs=decided,
        return_rate=_ratio(returned_pairs, total),
        proposal_coverage=_ratio(admitted_pairs, total),
        well_typed_rate_returned=_ratio(admitted_pairs, returned_pairs),
        adjudication_decision_rate=_ratio(decided, admitted_pairs),
    )


@dataclass(frozen=True)
class CorrespondenceRouteProfile:
    """End-to-end contribution of a mapper-assisted correspondence route."""

    total_pairs: int
    direct_certified: int
    direct_refuted: int
    direct_undecided: int
    route_eligible: int
    route_returned: int
    route_admitted: int
    recovered_certifications: int
    recovered_refutations: int
    route_undecided: int
    effective_certified: int
    effective_refuted: int
    effective_undecided: int
    direct_decision_rate_all: float
    effective_decision_rate_all: float
    decision_coverage_gain: float


@dataclass(frozen=True)
class ConstructiveContractProfile:
    """Proposal, contract, and model evidence for constructive mappings."""

    routed_pairs: int
    returned_pairs: int
    admitted_pairs: int
    rejected_pairs: int
    contract_valid: int
    contract_invalid: int
    contract_unknown: int
    model_certified: int
    model_refuted: int
    model_undecided: int
    counterpart_exists: int
    counterpart_absent: int
    counterpart_unknown: int
    contract_invalid_without_counterpart_check: int
    return_rate: float
    admission_rate_all: float
    admission_rate_returned: float
    contract_decision_rate_admitted: float
    model_decision_rate_all: float


def constructive_contract_profile(
    returned: Sequence[bool],
    admitted: Sequence[bool],
    contract_statuses: Sequence[ContractStatus | None],
    model_verdicts: Sequence[ReformulationVerdict | None],
    counterpart_statuses: Sequence[CounterpartStatus | None],
) -> ConstructiveContractProfile:
    """Keep mapper availability distinct from SMT and lifted model evidence."""

    if len(
        {
            len(returned),
            len(admitted),
            len(contract_statuses),
            len(model_verdicts),
            len(counterpart_statuses),
        }
    ) != 1:
        raise ValueError("constructive contract metric inputs must have equal length")
    for index, (has_payload, typed, contract, model, counterpart) in enumerate(
        zip(returned, admitted, contract_statuses, model_verdicts, counterpart_statuses)
    ):
        if typed and not has_payload:
            raise ValueError(f"admitted constructive contract {index} was not returned")
        if typed and (contract is None or model is None):
            raise ValueError(f"admitted constructive contract {index} lacks adjudication")
        if not typed and (contract is not None or model is not None or counterpart is not None):
            raise ValueError(f"unadmitted constructive contract {index} has SMT evidence")
        if counterpart is not None and contract != ContractStatus.INVALID:
            raise ValueError(f"counterpart absence {index} lacks an invalid contract witness")
        if model == ReformulationVerdict.R and counterpart != CounterpartStatus.ABSENT:
            raise ValueError(f"model refutation {index} lacks lifted counterpart evidence")

    total = len(returned)
    returned_count = sum(returned)
    admitted_count = sum(admitted)
    contract_valid = sum(status == ContractStatus.VALID for status in contract_statuses)
    contract_invalid = sum(status == ContractStatus.INVALID for status in contract_statuses)
    contract_unknown = sum(status == ContractStatus.UNKNOWN for status in contract_statuses)
    contract_decided = contract_valid + contract_invalid
    model_certified = sum(verdict == ReformulationVerdict.C for verdict in model_verdicts)
    model_refuted = sum(verdict == ReformulationVerdict.R for verdict in model_verdicts)
    # A missing or rejected proposal is still an undecided routed pair.  Keep
    # the model-level C/R/U partition on the complete route denominator rather
    # than silently dropping pre-SMT failures.
    model_undecided = total - model_certified - model_refuted
    return ConstructiveContractProfile(
        routed_pairs=total,
        returned_pairs=returned_count,
        admitted_pairs=admitted_count,
        rejected_pairs=returned_count - admitted_count,
        contract_valid=contract_valid,
        contract_invalid=contract_invalid,
        contract_unknown=contract_unknown,
        model_certified=model_certified,
        model_refuted=model_refuted,
        model_undecided=model_undecided,
        counterpart_exists=sum(
            status == CounterpartStatus.EXISTS for status in counterpart_statuses
        ),
        counterpart_absent=sum(
            status == CounterpartStatus.ABSENT for status in counterpart_statuses
        ),
        counterpart_unknown=sum(
            status == CounterpartStatus.UNKNOWN for status in counterpart_statuses
        ),
        contract_invalid_without_counterpart_check=sum(
            contract == ContractStatus.INVALID and counterpart is None
            for contract, counterpart in zip(contract_statuses, counterpart_statuses)
        ),
        return_rate=_ratio(returned_count, total),
        admission_rate_all=_ratio(admitted_count, total),
        admission_rate_returned=_ratio(admitted_count, returned_count),
        contract_decision_rate_admitted=_ratio(contract_decided, admitted_count),
        model_decision_rate_all=_ratio(model_certified + model_refuted, total),
    )


def correspondence_route_profile(
    direct_verdicts: Sequence[Verdict],
    route_eligible: Sequence[bool],
    proposed: Sequence[bool],
    well_typed: Sequence[bool],
    proposal_verdicts: Sequence[Verdict | None],
) -> CorrespondenceRouteProfile:
    """Measure what an LLM mapper adds after the deterministic verifier.

    All inputs use the complete pair denominator. Only direct alignment
    abstentions may be route eligible. Direct C/R outcomes cannot be replaced,
    and missing or ill-typed proposals contribute no SMT verdict.
    """

    lengths = {
        len(direct_verdicts),
        len(route_eligible),
        len(proposed),
        len(well_typed),
        len(proposal_verdicts),
    }
    if len(lengths) != 1:
        raise ValueError("correspondence route inputs must have equal length")

    effective: list[Verdict] = []
    for index, (direct, eligible, returned, typed, proposal) in enumerate(
        zip(
            direct_verdicts,
            route_eligible,
            proposed,
            well_typed,
            proposal_verdicts,
        )
    ):
        if eligible and direct != Verdict.UNDECIDED:
            raise ValueError(
                f"correspondence route {index} cannot replace a direct decision"
            )
        if not eligible and (returned or typed or proposal is not None):
            raise ValueError(
                f"ineligible correspondence route {index} contains mapper output"
            )
        if typed and not returned:
            raise ValueError(f"well-typed route {index} was not returned")
        if not typed and proposal is not None:
            raise ValueError(
                f"unadmitted correspondence route {index} has an SMT verdict"
            )
        if typed and proposal is None:
            raise ValueError(
                f"admitted correspondence route {index} lacks an SMT verdict"
            )
        effective.append(proposal if eligible and typed else direct)

    total = len(direct_verdicts)
    direct_decided = sum(
        verdict != Verdict.UNDECIDED for verdict in direct_verdicts
    )
    effective_decided = sum(verdict != Verdict.UNDECIDED for verdict in effective)
    return CorrespondenceRouteProfile(
        total_pairs=total,
        direct_certified=sum(
            verdict == Verdict.CERTIFIED for verdict in direct_verdicts
        ),
        direct_refuted=sum(verdict == Verdict.REFUTED for verdict in direct_verdicts),
        direct_undecided=sum(
            verdict == Verdict.UNDECIDED for verdict in direct_verdicts
        ),
        route_eligible=sum(route_eligible),
        route_returned=sum(proposed),
        route_admitted=sum(well_typed),
        recovered_certifications=sum(
            eligible and typed and proposal == Verdict.CERTIFIED
            for eligible, typed, proposal in zip(
                route_eligible, well_typed, proposal_verdicts
            )
        ),
        recovered_refutations=sum(
            eligible and typed and proposal == Verdict.REFUTED
            for eligible, typed, proposal in zip(
                route_eligible, well_typed, proposal_verdicts
            )
        ),
        route_undecided=sum(
            eligible
            and (
                not typed
                or proposal is None
                or proposal == Verdict.UNDECIDED
            )
            for eligible, typed, proposal in zip(
                route_eligible, well_typed, proposal_verdicts
            )
        ),
        effective_certified=sum(verdict == Verdict.CERTIFIED for verdict in effective),
        effective_refuted=sum(verdict == Verdict.REFUTED for verdict in effective),
        effective_undecided=sum(verdict == Verdict.UNDECIDED for verdict in effective),
        direct_decision_rate_all=_ratio(direct_decided, total),
        effective_decision_rate_all=_ratio(effective_decided, total),
        decision_coverage_gain=_ratio(effective_decided - direct_decided, total),
    )


@dataclass(frozen=True)
class UniversalClaimProfile:
    """Evidence profile for claims quantified over all problem instances.

    A concrete ``CERTIFIED`` verdict is deliberately not lifted to a universal
    certificate.  A concrete ``REFUTED`` verdict is lifted only when its
    witness survived native replay.  Parametric certificates come from an
    external proof-producing route, such as a proof assistant.
    """

    total_claims: int
    parametrically_certified_claims: int
    refuted_by_replayed_instance_claims: int
    unresolved_claims: int
    fixed_instance_certified_not_lifted: int
    fixed_instance_undecided: int
    unreplayed_refutations: int
    certification_rate_all: float
    refutation_rate_all: float
    decision_coverage_all: float


@dataclass(frozen=True)
class UniversalClaimEvaluation:
    """Label-aware evaluation without treating abstention as a decision."""

    total_claims: int
    valid_claims: int
    invalid_claims: int
    correct_certifications: int
    false_certifications: int
    correct_refutations: int
    false_refutations: int
    unresolved_valid_claims: int
    unresolved_invalid_claims: int
    decided_claims: int
    unsafe_decisions: int
    valid_certification_recall: float
    invalid_refutation_recall: float
    accuracy_all: float
    balanced_accuracy: float
    certification_precision: float
    refutation_precision: float
    decision_coverage_all: float
    correct_evidence_coverage_all: float
    unsafe_decision_rate_all: float
    unsafe_decision_rate_decided: float


@dataclass(frozen=True)
class UniversalDecisionAudit:
    """Audit reported universal decisions against their supporting evidence.

    Classification and evidence are intentionally separate.  A reported
    certification or refutation is supported only when the corresponding
    parametric proof or replayed counterexample exists.  Unsupported decisions
    are not counted as contradictions unless available evidence proves the
    opposite verdict.
    """

    total_claims: int
    reported_certifications: int
    reported_refutations: int
    reported_undecided: int
    supported_certifications: int
    supported_refutations: int
    unsupported_certifications: int
    unsupported_refutations: int
    contradicted_certifications: int
    contradicted_refutations: int
    evidence_available_but_abstained: int
    reported_decisions: int
    supported_decisions: int
    unsupported_decisions: int
    contradicted_decisions: int
    supported_rate_reported: float
    unsupported_rate_reported: float
    contradiction_rate_reported: float


def _lift_universal_claims(
    concrete_verdicts: Sequence[Verdict],
    replayed_refutations: Sequence[bool],
    parametric_certificates: Sequence[bool],
) -> list[Verdict]:
    lengths = {
        len(concrete_verdicts),
        len(replayed_refutations),
        len(parametric_certificates),
    }
    if len(lengths) != 1:
        raise ValueError("universal claim inputs must have equal length")
    lifted: list[Verdict] = []
    for index, (verdict, replayed, certified) in enumerate(
        zip(concrete_verdicts, replayed_refutations, parametric_certificates)
    ):
        if replayed and verdict != Verdict.REFUTED:
            raise ValueError(
                f"replayed universal counterexample {index} lacks a concrete refutation"
            )
        if certified and verdict == Verdict.REFUTED:
            raise ValueError(
                f"universal claim {index} has contradictory proof and refutation evidence"
            )
        if certified:
            lifted.append(Verdict.CERTIFIED)
        elif verdict == Verdict.REFUTED and replayed:
            lifted.append(Verdict.REFUTED)
        else:
            lifted.append(Verdict.UNDECIDED)
    return lifted


def universal_claim_profile(
    concrete_verdicts: Sequence[Verdict],
    replayed_refutations: Sequence[bool],
    parametric_certificates: Sequence[bool] | None = None,
) -> UniversalClaimProfile:
    """Lift sound instance evidence to universal claim outcomes.

    ``parametric_certificates`` may be omitted for an instance verifier.  In
    that case, concrete certifications remain unresolved at universal scope.
    """

    certificates = (
        list(parametric_certificates)
        if parametric_certificates is not None
        else [False] * len(concrete_verdicts)
    )
    lifted = _lift_universal_claims(
        concrete_verdicts, replayed_refutations, certificates
    )
    certified = sum(verdict == Verdict.CERTIFIED for verdict in lifted)
    refuted = sum(verdict == Verdict.REFUTED for verdict in lifted)
    unresolved = len(lifted) - certified - refuted
    return UniversalClaimProfile(
        total_claims=len(lifted),
        parametrically_certified_claims=certified,
        refuted_by_replayed_instance_claims=refuted,
        unresolved_claims=unresolved,
        fixed_instance_certified_not_lifted=sum(
            verdict == Verdict.CERTIFIED and not certificate
            for verdict, certificate in zip(concrete_verdicts, certificates)
        ),
        fixed_instance_undecided=sum(
            verdict == Verdict.UNDECIDED and not certificate
            for verdict, certificate in zip(concrete_verdicts, certificates)
        ),
        unreplayed_refutations=sum(
            verdict == Verdict.REFUTED and not replayed
            for verdict, replayed in zip(concrete_verdicts, replayed_refutations)
        ),
        certification_rate_all=_ratio(certified, len(lifted)),
        refutation_rate_all=_ratio(refuted, len(lifted)),
        decision_coverage_all=_ratio(certified + refuted, len(lifted)),
    )


def universal_claim_evaluation(
    ground_truth_valid: Sequence[bool],
    concrete_verdicts: Sequence[Verdict],
    replayed_refutations: Sequence[bool],
    parametric_certificates: Sequence[bool] | None = None,
) -> UniversalClaimEvaluation:
    """Evaluate proof and counterexample evidence on labeled universal claims."""

    certificates = (
        list(parametric_certificates)
        if parametric_certificates is not None
        else [False] * len(concrete_verdicts)
    )
    lifted = _lift_universal_claims(
        concrete_verdicts, replayed_refutations, certificates
    )
    if len(ground_truth_valid) != len(lifted):
        raise ValueError("ground-truth labels must align with universal claims")

    valid = sum(ground_truth_valid)
    invalid = len(ground_truth_valid) - valid
    correct_certifications = sum(
        truth and verdict == Verdict.CERTIFIED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    false_certifications = sum(
        not truth and verdict == Verdict.CERTIFIED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    correct_refutations = sum(
        not truth and verdict == Verdict.REFUTED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    false_refutations = sum(
        truth and verdict == Verdict.REFUTED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    unresolved_valid = sum(
        truth and verdict == Verdict.UNDECIDED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    unresolved_invalid = sum(
        not truth and verdict == Verdict.UNDECIDED
        for truth, verdict in zip(ground_truth_valid, lifted)
    )
    certified = correct_certifications + false_certifications
    refuted = correct_refutations + false_refutations
    decided = certified + refuted
    unsafe = false_certifications + false_refutations
    correct = correct_certifications + correct_refutations
    total = len(lifted)
    return UniversalClaimEvaluation(
        total_claims=total,
        valid_claims=valid,
        invalid_claims=invalid,
        correct_certifications=correct_certifications,
        false_certifications=false_certifications,
        correct_refutations=correct_refutations,
        false_refutations=false_refutations,
        unresolved_valid_claims=unresolved_valid,
        unresolved_invalid_claims=unresolved_invalid,
        decided_claims=decided,
        unsafe_decisions=unsafe,
        valid_certification_recall=_ratio(correct_certifications, valid),
        invalid_refutation_recall=_ratio(correct_refutations, invalid),
        accuracy_all=_ratio(correct, total),
        balanced_accuracy=(
            _ratio(correct_certifications, valid)
            + _ratio(correct_refutations, invalid)
        )
        / 2,
        certification_precision=_ratio(correct_certifications, certified),
        refutation_precision=_ratio(correct_refutations, refuted),
        decision_coverage_all=_ratio(decided, total),
        correct_evidence_coverage_all=_ratio(correct, total),
        unsafe_decision_rate_all=_ratio(unsafe, total),
        unsafe_decision_rate_decided=_ratio(unsafe, decided),
    )


def universal_decision_audit(
    reported_verdicts: Sequence[Verdict],
    concrete_verdicts: Sequence[Verdict],
    replayed_refutations: Sequence[bool],
    parametric_certificates: Sequence[bool] | None = None,
) -> UniversalDecisionAudit:
    """Separate a method's universal labels from machine-checkable evidence."""

    certificates = (
        list(parametric_certificates)
        if parametric_certificates is not None
        else [False] * len(concrete_verdicts)
    )
    evidence_verdicts = _lift_universal_claims(
        concrete_verdicts, replayed_refutations, certificates
    )
    if len(reported_verdicts) != len(evidence_verdicts):
        raise ValueError("reported verdicts must align with universal claims")

    supported_certifications = sum(
        reported == evidence == Verdict.CERTIFIED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    supported_refutations = sum(
        reported == evidence == Verdict.REFUTED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    unsupported_certifications = sum(
        reported == Verdict.CERTIFIED and evidence == Verdict.UNDECIDED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    unsupported_refutations = sum(
        reported == Verdict.REFUTED and evidence == Verdict.UNDECIDED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    contradicted_certifications = sum(
        reported == Verdict.CERTIFIED and evidence == Verdict.REFUTED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    contradicted_refutations = sum(
        reported == Verdict.REFUTED and evidence == Verdict.CERTIFIED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    evidence_available_but_abstained = sum(
        reported == Verdict.UNDECIDED and evidence != Verdict.UNDECIDED
        for reported, evidence in zip(reported_verdicts, evidence_verdicts)
    )
    reported_certifications = sum(
        verdict == Verdict.CERTIFIED for verdict in reported_verdicts
    )
    reported_refutations = sum(
        verdict == Verdict.REFUTED for verdict in reported_verdicts
    )
    reported_decisions = reported_certifications + reported_refutations
    supported = supported_certifications + supported_refutations
    unsupported = unsupported_certifications + unsupported_refutations
    contradicted = contradicted_certifications + contradicted_refutations
    return UniversalDecisionAudit(
        total_claims=len(reported_verdicts),
        reported_certifications=reported_certifications,
        reported_refutations=reported_refutations,
        reported_undecided=len(reported_verdicts) - reported_decisions,
        supported_certifications=supported_certifications,
        supported_refutations=supported_refutations,
        unsupported_certifications=unsupported_certifications,
        unsupported_refutations=unsupported_refutations,
        contradicted_certifications=contradicted_certifications,
        contradicted_refutations=contradicted_refutations,
        evidence_available_but_abstained=evidence_available_but_abstained,
        reported_decisions=reported_decisions,
        supported_decisions=supported,
        unsupported_decisions=unsupported,
        contradicted_decisions=contradicted,
        supported_rate_reported=_ratio(supported, reported_decisions),
        unsupported_rate_reported=_ratio(unsupported, reported_decisions),
        contradiction_rate_reported=_ratio(contradicted, reported_decisions),
    )


@dataclass(frozen=True)
class CostProfile:
    queries: int
    timed_queries: int
    total_ms: float
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None


def cost_profile(records: Iterable[EvidenceRecord]) -> CostProfile:
    rows = list(records)
    durations = sorted(row.duration_ms for row in rows if row.duration_ms is not None)
    p95 = None
    if durations:
        p95 = durations[min(len(durations) - 1, max(0, ceil(0.95 * len(durations)) - 1))]
    return CostProfile(
        queries=len(rows),
        timed_queries=len(durations),
        total_ms=sum(durations),
        mean_ms=mean(durations) if durations else None,
        median_ms=median(durations) if durations else None,
        p95_ms=p95,
    )


@dataclass(frozen=True)
class LocalizationMetrics:
    cases: int
    labeled_cases: int
    localized_cases: int
    exact_hits: int
    top_k_hits: int
    precision: float
    recall: float
    exact_set_accuracy: float
    hit_at_k: float

    @property
    def top_k_recall(self) -> float:
        """Backward-compatible alias for case-level hit@k."""

        return self.hit_at_k


def localization_metrics(
    predicted: Sequence[Sequence[str]], truth: Sequence[Sequence[str]], k: int = 3
) -> LocalizationMetrics:
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have equal length")
    true_positive = 0
    predicted_total = 0
    truth_total = 0
    exact_hits = 0
    top_k_hits = 0
    labeled_cases = 0
    localized_cases = 0
    for guesses, labels in zip(predicted, truth):
        guess_set = set(guesses)
        label_set = set(labels)
        overlap = guess_set & label_set
        if not label_set:
            continue
        labeled_cases += 1
        localized_cases += bool(guess_set)
        true_positive += len(overlap)
        predicted_total += len(guess_set)
        truth_total += len(label_set)
        exact_hits += bool(label_set and guess_set == label_set)
        top_k_hits += bool(label_set & set(guesses[:k]))
    return LocalizationMetrics(
        cases=len(truth),
        labeled_cases=labeled_cases,
        localized_cases=localized_cases,
        exact_hits=exact_hits,
        top_k_hits=top_k_hits,
        precision=_ratio(true_positive, predicted_total),
        recall=_ratio(true_positive, truth_total),
        exact_set_accuracy=_ratio(exact_hits, labeled_cases),
        hit_at_k=_ratio(top_k_hits, labeled_cases),
    )


@dataclass(frozen=True)
class RepairMetrics:
    total: int
    certified_before: int
    refuted_before: int
    undecided_before: int
    semantic_certified_to_certified: int
    semantic_refuted_to_certified: int
    semantic_certified_to_refuted: int
    semantic_refuted_to_refuted: int
    semantic_refuted_to_undecided: int
    semantic_certified_to_undecided: int
    semantic_undecided_to_certified: int
    semantic_undecided_to_refuted: int
    semantic_undecided_to_undecided: int
    net_semantic_gain: int
    relation_discharge_rate: float
    certificate_retention_rate: float
    accepted_rate: float

    @property
    def wrong_before(self) -> int:
        return self.refuted_before

    @property
    def right_before(self) -> int:
        return self.certified_before

    @property
    def wrong_to_right(self) -> int:
        return self.semantic_refuted_to_certified

    @property
    def right_to_wrong(self) -> int:
        return self.semantic_certified_to_refuted

    @property
    def net_gain(self) -> int:
        return self.net_semantic_gain


@dataclass(frozen=True)
class ValueTransitionMetrics:
    total: int
    observed_pairs: int
    wrong_before: int
    right_before: int
    value_wrong_to_right: int
    value_right_to_wrong: int
    net_value_gain: int


def value_transition_metrics(
    before: Sequence[bool | None], after: Sequence[bool | None]
) -> ValueTransitionMetrics:
    if len(before) != len(after):
        raise ValueError("value transition inputs must have equal length")
    observed = [(left, right) for left, right in zip(before, after) if left is not None and right is not None]
    wrong_to_right = sum(left is False and right is True for left, right in observed)
    right_to_wrong = sum(left is True and right is False for left, right in observed)
    return ValueTransitionMetrics(
        total=len(before),
        observed_pairs=len(observed),
        wrong_before=sum(left is False for left, _ in observed),
        right_before=sum(left is True for left, _ in observed),
        value_wrong_to_right=wrong_to_right,
        value_right_to_wrong=right_to_wrong,
        net_value_gain=wrong_to_right - right_to_wrong,
    )


def repair_metrics(
    before: Sequence[Verdict],
    after: Sequence[Verdict],
    discharged: Sequence[bool],
    retained: Sequence[bool],
    accepted: Sequence[bool],
) -> RepairMetrics:
    lengths = {len(before), len(after), len(discharged), len(retained), len(accepted)}
    if len(lengths) != 1:
        raise ValueError("repair metric inputs must have equal length")
    refuted_before = sum(v == Verdict.REFUTED for v in before)
    certified_before = sum(v == Verdict.CERTIFIED for v in before)
    undecided_before = sum(v == Verdict.UNDECIDED for v in before)
    refuted_to_certified = sum(a == Verdict.REFUTED and b == Verdict.CERTIFIED for a, b in zip(before, after))
    certified_to_refuted = sum(a == Verdict.CERTIFIED and b == Verdict.REFUTED for a, b in zip(before, after))
    return RepairMetrics(
        total=len(before),
        certified_before=certified_before,
        refuted_before=refuted_before,
        undecided_before=undecided_before,
        semantic_certified_to_certified=sum(
            a == Verdict.CERTIFIED and b == Verdict.CERTIFIED
            for a, b in zip(before, after)
        ),
        semantic_refuted_to_certified=refuted_to_certified,
        semantic_certified_to_refuted=certified_to_refuted,
        semantic_refuted_to_refuted=sum(
            a == Verdict.REFUTED and b == Verdict.REFUTED
            for a, b in zip(before, after)
        ),
        semantic_refuted_to_undecided=sum(
            a == Verdict.REFUTED and b == Verdict.UNDECIDED
            for a, b in zip(before, after)
        ),
        semantic_certified_to_undecided=sum(
            a == Verdict.CERTIFIED and b == Verdict.UNDECIDED
            for a, b in zip(before, after)
        ),
        semantic_undecided_to_certified=sum(
            a == Verdict.UNDECIDED and b == Verdict.CERTIFIED
            for a, b in zip(before, after)
        ),
        semantic_undecided_to_refuted=sum(
            a == Verdict.UNDECIDED and b == Verdict.REFUTED
            for a, b in zip(before, after)
        ),
        semantic_undecided_to_undecided=sum(
            a == Verdict.UNDECIDED and b == Verdict.UNDECIDED
            for a, b in zip(before, after)
        ),
        net_semantic_gain=refuted_to_certified - certified_to_refuted,
        relation_discharge_rate=_ratio(sum(discharged), len(discharged)),
        certificate_retention_rate=_ratio(sum(retained), len(retained)),
        accepted_rate=_ratio(sum(accepted), len(accepted)),
    )


@dataclass(frozen=True)
class ArtifactVerdictProfile:
    """All-denominator artifact outcomes; abstentions never disappear."""

    total: int
    certified: int
    refuted: int
    undecided: int
    certified_rate_all: float
    refuted_rate_all: float
    undecided_rate_all: float


@dataclass(frozen=True)
class DiagnosticProfile:
    """Counterexample diagnostics over refuted claims and artifacts."""

    refuted_claims: int
    refuted_artifacts: int
    underconstraint_claims: int
    overconstraint_claims: int
    preference_claims: int
    reporting_claims: int
    unclassified_claims: int
    replayed_refutations: int
    replay_rate: float
    localized_refutations: int
    localization_rate: float
    multi_claim_artifacts: int
    multi_claim_rate_all: float
    multi_layer_artifacts: int
    multi_layer_rate_all: float
    value_checked_artifacts: int
    value_agreement_artifacts: int
    value_disagreement_artifacts: int
    value_agreement_rate_checked: float
    value_checked_refuted_artifacts: int
    value_masked_artifacts: int
    value_mask_rate_among_refuted: float
    refutation_rate_given_value_agreement: float


@dataclass(frozen=True)
class AuditScorecard:
    """A compact verification/diagnosis scorecard for a batch of artifacts.

    ``record_groups`` in :func:`audit_scorecard` must contain one complete
    evidence ledger per assigned artifact.  This keeps parse failures,
    unsupported relations, and solver timeouts in the artifact denominator.
    Query-level and artifact-level profiles are intentionally separate.
    """

    artifacts: ArtifactVerdictProfile
    artifacts_by_relation: Mapping[Relation, ArtifactVerdictProfile]
    queries: RelationProfile
    queries_by_relation: Mapping[Relation, RelationProfile]
    semantic_constraints: SemanticConstraintMetrics | None
    diagnostics: DiagnosticProfile
    severity: SeverityProfile
    cost: CostProfile

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": asdict(self.artifacts),
            "artifacts_by_relation": {
                relation.value: asdict(profile)
                for relation, profile in self.artifacts_by_relation.items()
            },
            "queries": asdict(self.queries),
            "queries_by_relation": {
                relation.value: asdict(profile)
                for relation, profile in self.queries_by_relation.items()
            },
            "semantic_constraints": (
                asdict(self.semantic_constraints)
                if self.semantic_constraints is not None
                else None
            ),
            "diagnostics": asdict(self.diagnostics),
            "severity": asdict(self.severity),
            "cost": asdict(self.cost),
        }


def _artifact_verdict_profile(verdicts: Sequence[Verdict]) -> ArtifactVerdictProfile:
    total = len(verdicts)
    certified = sum(verdict == Verdict.CERTIFIED for verdict in verdicts)
    refuted = sum(verdict == Verdict.REFUTED for verdict in verdicts)
    undecided = total - certified - refuted
    return ArtifactVerdictProfile(
        total=total,
        certified=certified,
        refuted=refuted,
        undecided=undecided,
        certified_rate_all=_ratio(certified, total),
        refuted_rate_all=_ratio(refuted, total),
        undecided_rate_all=_ratio(undecided, total),
    )


def audit_scorecard(
    record_groups: Sequence[Sequence[EvidenceRecord]],
    value_agreements: Sequence[bool | None] | None = None,
) -> AuditScorecard:
    """Aggregate exact evidence without conditioning away undecided artifacts.

    ``value_agreements`` is optional endpoint information aligned one-to-one
    with ``record_groups``.  It is used only to measure how often an equal
    terminal value masks a refuted semantic relation; it never affects a
    verifier verdict.
    """

    groups = [tuple(group) for group in record_groups]
    if value_agreements is None:
        agreements: list[bool | None] = [None] * len(groups)
    else:
        agreements = list(value_agreements)
        if len(agreements) != len(groups):
            raise ValueError("value_agreements must align with record_groups")

    all_records = [record for group in groups for record in group]
    artifact_verdicts = [aggregate(group) for group in groups]
    present_relations = tuple(
        relation
        for relation in Relation
        if any(record.relation == relation for record in all_records)
    )
    artifacts_by_relation: dict[Relation, ArtifactVerdictProfile] = {}
    queries_by_relation: dict[Relation, RelationProfile] = {}
    for relation in present_relations:
        relation_groups = [
            tuple(record for record in group if record.relation == relation)
            for group in groups
        ]
        relation_verdicts = [
            aggregate(group) if group else Verdict.UNDECIDED for group in relation_groups
        ]
        artifacts_by_relation[relation] = _artifact_verdict_profile(relation_verdicts)
        queries_by_relation[relation] = relation_profile(
            record for record in all_records if record.relation == relation
        )

    recall_rows = [
        row for row in all_records if row.metadata.get("metric_role") == "recall"
    ]
    precision_rows = [
        row for row in all_records if row.metadata.get("metric_role") == "precision"
    ]
    semantic_constraints = (
        semantic_constraint_metrics(recall_rows, precision_rows)
        if recall_rows or precision_rows
        else None
    )

    refuted_rows = [row for row in all_records if row.verdict == Verdict.REFUTED]
    fault_counts = {
        "underconstraint": 0,
        "overconstraint": 0,
        "preference": 0,
        "reporting": 0,
        "unclassified": 0,
    }
    for row in refuted_rows:
        fault = row.metadata.get("candidate_fault_if_refuted")
        if fault not in fault_counts:
            if row.relation == Relation.REPORTING:
                fault = "reporting"
            elif row.claim_id.startswith("preference:"):
                fault = "preference"
            else:
                fault = "unclassified"
        fault_counts[str(fault)] += 1

    refuted_artifacts = 0
    multi_claim_artifacts = 0
    multi_layer_artifacts = 0
    value_checked_refuted_artifacts = 0
    value_masked_artifacts = 0
    value_checked_artifacts = sum(agreement is not None for agreement in agreements)
    value_agreement_artifacts = sum(agreement is True for agreement in agreements)
    value_disagreement_artifacts = sum(agreement is False for agreement in agreements)
    for group, agreement in zip(groups, agreements):
        refuted = [row for row in group if row.verdict == Verdict.REFUTED]
        if not refuted:
            continue
        refuted_artifacts += 1
        multi_claim_artifacts += len(refuted) > 1
        multi_layer_artifacts += len({row.relation for row in refuted}) > 1
        value_checked_refuted_artifacts += agreement is not None
        value_masked_artifacts += agreement is True

    diagnostic = DiagnosticProfile(
        refuted_claims=len(refuted_rows),
        refuted_artifacts=refuted_artifacts,
        underconstraint_claims=fault_counts["underconstraint"],
        overconstraint_claims=fault_counts["overconstraint"],
        preference_claims=fault_counts["preference"],
        reporting_claims=fault_counts["reporting"],
        unclassified_claims=fault_counts["unclassified"],
        replayed_refutations=sum(
            bool(row.replay and row.replay.passed) for row in refuted_rows
        ),
        replay_rate=_ratio(
            sum(bool(row.replay and row.replay.passed) for row in refuted_rows),
            len(refuted_rows),
        ),
        localized_refutations=sum(bool(row.implicated) for row in refuted_rows),
        localization_rate=_ratio(
            sum(bool(row.implicated) for row in refuted_rows), len(refuted_rows)
        ),
        multi_claim_artifacts=multi_claim_artifacts,
        multi_claim_rate_all=_ratio(multi_claim_artifacts, len(groups)),
        multi_layer_artifacts=multi_layer_artifacts,
        multi_layer_rate_all=_ratio(multi_layer_artifacts, len(groups)),
        value_checked_artifacts=value_checked_artifacts,
        value_agreement_artifacts=value_agreement_artifacts,
        value_disagreement_artifacts=value_disagreement_artifacts,
        value_agreement_rate_checked=_ratio(
            value_agreement_artifacts, value_checked_artifacts
        ),
        value_checked_refuted_artifacts=value_checked_refuted_artifacts,
        value_masked_artifacts=value_masked_artifacts,
        value_mask_rate_among_refuted=_ratio(
            value_masked_artifacts, value_checked_refuted_artifacts
        ),
        refutation_rate_given_value_agreement=_ratio(
            value_masked_artifacts, value_agreement_artifacts
        ),
    )
    return AuditScorecard(
        artifacts=_artifact_verdict_profile(artifact_verdicts),
        artifacts_by_relation=artifacts_by_relation,
        queries=relation_profile(all_records),
        queries_by_relation=queries_by_relation,
        semantic_constraints=semantic_constraints,
        diagnostics=diagnostic,
        severity=severity_profile(all_records),
        cost=cost_profile(all_records),
    )
