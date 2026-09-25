"""Choose which artifact to repair, then reverify the proposed artifact pair.

Inputs are already formal MILP objects. Free-text source interpretation and
LLM proposals are not verification certificates. A complete reference may be
supplied by the host only after independent review of its provenance.
"""
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .constructive import model_fingerprint
from .evidence import EvidenceRecord, Verdict, aggregate
from .ir import ModelIR, ResultRecord
from .verify import Verifier


@dataclass(frozen=True)
class RepairDiagnosis:
    target: str
    basis: str
    implementation: tuple[EvidenceRecord, ...]
    reporting: tuple[EvidenceRecord, ...]
    reference_to_written: tuple[EvidenceRecord, ...]
    reference_to_program: tuple[EvidenceRecord, ...]
    written_fingerprint: str
    program_fingerprint: str
    reference_fingerprint: str | None
    reference_origin: str | None

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        for key in ('implementation', 'reporting', 'reference_to_written', 'reference_to_program'):
            rows = getattr(self, key)
            out[key] = {'status': aggregate(list(rows)).value,
                        'records': [r.to_dict() for r in rows]}
        return out


def compare_models(verifier: Verifier, left: ModelIR, right: ModelIR):
    return (*verifier.region_equivalence(left, right),
            verifier.preference_equivalence(left, right),
            verifier.objective_value_equivalence(left, right))


def diagnose_repair(
    written: ModelIR, program: ModelIR, result: ResultRecord | None = None, *,
    reference: ModelIR | None = None, reference_origin: str | None = None,
    written_is_specification: bool = False, verifier: Verifier | None = None,
) -> RepairDiagnosis:
    """Never infer fault ownership from inclusion direction or value agreement.

``reference`` is a complete, independently established formal model, not a
single entailed source clause. ``written_is_specification`` is an explicit host
policy for implementation-only repair. Neither is inferred from an LLM claim.
    """
    if reference is not None and not reference_origin:
        raise ValueError('A complete reference needs an explicit external origin')
    if reference is None and reference_origin is not None:
        raise ValueError('Reference origin supplied without a reference model')
    v = verifier or Verifier()
    w, p = replace(written, name='written'), replace(program, name='program')
    implementation = compare_models(v, w, p)
    reporting = v.verify_result(p, result) if result is not None else ()
    rw, rp = (), ()
    if reference is not None:
        r = replace(reference, name='reference')
        rw, rp = compare_models(v, r, w), compare_models(v, r, p)
    pair_state = aggregate(list(implementation))
    ws, ps = aggregate(list(rw)), aggregate(list(rp))
    C, R = Verdict.CERTIFIED, Verdict.REFUTED
    if reference is not None:
        if ws == R and ps == C:
            target, basis = 'written_model', 'reference_agrees_with_program_only'
        elif ws == C and ps == R:
            target, basis = 'program', 'reference_agrees_with_written_only'
        elif ws == R and ps == R:
            target, basis = 'both', 'both_artifacts_disagree_with_reference'
        elif ws != C or ps != C:
            target, basis = 'review', 'reference_comparison_unresolved'
        elif pair_state != C:
            target, basis = 'review', 'pair_comparison_unresolved'
        elif aggregate(list(reporting)) == R:
            target, basis = 'result', 'models_agree_but_report_is_refuted'
        else:
            target, basis = 'none', 'no_refuted_checked_relation'
    elif pair_state == R:
        if written_is_specification:
            target, basis = 'program', 'explicit_written_specification'
        else:
            target, basis = 'review', 'pair_disagreement_does_not_identify_the_faulty_side'
    elif pair_state != C:
        target, basis = 'review', 'pair_comparison_unresolved'
    elif aggregate(list(reporting)) == R:
        target, basis = 'result', 'models_agree_but_report_is_refuted'
    else:
        target, basis = 'none', 'no_refuted_checked_relation'
    return RepairDiagnosis(target, basis, tuple(implementation), tuple(reporting),
                           tuple(rw), tuple(rp), model_fingerprint(written),
                           model_fingerprint(program),
                           model_fingerprint(reference) if reference is not None else None,
                           reference_origin)


def repair_prompt_payload(
    diagnosis: RepairDiagnosis, *, case_id: str, question: str,
    formulation: Mapping[str, Any], code: str,
) -> dict[str, Any]:
    """Small, label-free host payload. The caller can send it to an LLM.

No result correctness label or benchmark optimum is accepted by this API.
For review, request a proposed fault location, not an automatic repair verdict.
Result-only corrections should be obtained by resolving the unchanged model.
    """
    action = {'review': 'propose_fault_location_for_review',
              'result': 'resolve_unchanged_program_with_native_solver',
              'none': 'no_repair'}.get(diagnosis.target, 'propose_artifact_revision')
    return {
        'id': case_id, 'action': action, 'target': diagnosis.target,
        'question': question, 'formulation': formulation, 'code': code,
        'evidence': diagnosis.to_dict(),
        'instruction': (
            'Use the question and the concrete witness to explain the discrepant relation. '
            'Revise only the selected artifact(s); return the complete resulting formulation '
            'and code. Keep the unselected artifact unchanged. '
            'Do not report a verification verdict: the revised models will be checked by SMT. '
            'If the target is review, propose a target and quote its source evidence; '
            'do not assume that the written model or code is the correct specification.'
        ),
        'response_fields': ['id', 'proposed_target', 'source_quotes', 'reason', 'formulation', 'code'],
    }


def evaluate_revision(
    before: RepairDiagnosis, written: ModelIR, program: ModelIR,
    result: ResultRecord | None = None, *, reference: ModelIR | None = None,
    verifier: Verifier | None = None,
) -> dict[str, Any]:
    """``program`` must be recaptured by the host from the proposed code.

Reference hashes bind the assessment to the same reference used in diagnosis.
No benchmark answer is consulted. Consistency, formal-reference agreement and
result verification are returned separately.
    """
    if (model_fingerprint(reference) if reference is not None else None) != before.reference_fingerprint:
        raise ValueError('Revision must be evaluated against the same reference')
    after = diagnose_repair(written, program, result, reference=reference,
                            reference_origin=before.reference_origin,
                            written_is_specification=before.basis == 'explicit_written_specification',
                            verifier=verifier)
    changed = set()
    if after.written_fingerprint != before.written_fingerprint:
        changed.add('written_model')
    if after.program_fingerprint != before.program_fingerprint:
        changed.add('program')
    allowed = {'written_model': {'written_model'}, 'program': {'program'},
               'both': {'written_model', 'program'}, 'result': set()}.get(before.target)
    C = Verdict.CERTIFIED
    aligned = aggregate(list(after.implementation)) == C
    result_verified = result is not None and aggregate(list(after.reporting)) == C
    reference_agrees = (aggregate(list(after.reference_to_written)) == C
                        and aggregate(list(after.reference_to_program)) == C) if reference is not None else None
    eligible = allowed is not None and changed <= allowed
    progress = (before.target == 'result' or bool(changed))
    accepted = eligible and progress and aligned and result_verified and reference_agrees is not False
    return {'accepted': accepted, 'repair_target': before.target,
            'changed_artifacts': sorted(changed), 'target_respected': eligible,
            'model_code_consistent': aligned, 'result_verified': result_verified,
            'formal_reference_agreement': reference_agrees,
            'natural_language_verified': False,
            'claim_scope': ('agreement_with_provided_formal_reference' if reference is not None
                            else 'model_code_and_result_consistency'),
            'after': after.to_dict()}
