from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping
from typing import Iterable

from .evidence import Relation, Verdict
from .plugin import AuditReport


ClaimKey = tuple[Relation, str]


@dataclass(frozen=True)
class RepairEvaluation:
    accepted: bool
    target_discharged: bool
    certificates_retained: bool
    no_new_refutation: bool
    edit_localized: bool
    retained_certificates: int
    prior_certificates: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RepairRequest:
    target: ClaimKey
    witness: Mapping[str, Fraction]
    implicated: tuple[str, ...]
    source_span: str | None
    certificates_to_preserve: tuple[ClaimKey, ...]
    instruction: str


def build_repair_request(report: AuditReport, target: ClaimKey) -> RepairRequest:
    rows = {(row.relation, row.claim_id): row for row in report.records}
    evidence = rows.get(target)
    if evidence is None:
        raise KeyError(target)
    if evidence.verdict != Verdict.REFUTED:
        raise ValueError("repair targets must be refuted claims")
    certificates = tuple(
        sorted(
            (
                (row.relation, row.claim_id)
                for row in report.records
                if row.verdict == Verdict.CERTIFIED
            ),
            key=lambda item: (item[0].value, item[1]),
        )
    )
    instruction = (
        f"Repair the {target[0].value} relation {target[1]}. "
        "Change only the implicated semantic objects, make the counterexample query "
        "unsatisfiable, and preserve every listed certificate."
    )
    return RepairRequest(
        target=target,
        witness=evidence.witness,
        implicated=evidence.implicated,
        source_span=evidence.source_span,
        certificates_to_preserve=certificates,
        instruction=instruction,
    )


def evaluate_repair(
    before: AuditReport,
    after: AuditReport,
    target: ClaimKey,
    changed_relations: Iterable[Relation],
    allowed_relations: Iterable[Relation],
) -> RepairEvaluation:
    """Apply a relation-local, certificate-preserving repair contract."""

    before_map = {(row.relation, row.claim_id): row for row in before.records}
    after_map = {(row.relation, row.claim_id): row for row in after.records}
    target_before = before_map.get(target)
    target_after = after_map.get(target)
    target_discharged = bool(
        target_before
        and target_after
        and target_before.verdict == Verdict.REFUTED
        and target_after.verdict == Verdict.CERTIFIED
    )

    prior_certificates = {
        key for key, row in before_map.items() if row.verdict == Verdict.CERTIFIED
    }
    retained = {
        key
        for key in prior_certificates
        if key in after_map and after_map[key].verdict == Verdict.CERTIFIED
    }
    certificates_retained = retained == prior_certificates
    new_refutations = {
        key
        for key, row in after_map.items()
        if row.verdict == Verdict.REFUTED
        and (key not in before_map or before_map[key].verdict != Verdict.REFUTED)
    }
    no_new_refutation = not new_refutations
    changed = set(changed_relations)
    allowed = set(allowed_relations)
    edit_localized = changed <= allowed

    reasons: list[str] = []
    if not target_discharged:
        reasons.append("target_not_discharged")
    if not certificates_retained:
        reasons.append("certificate_regression")
    if not no_new_refutation:
        reasons.append("new_refutation")
    if not edit_localized:
        reasons.append("edit_outside_contract")
    return RepairEvaluation(
        accepted=not reasons,
        target_discharged=target_discharged,
        certificates_retained=certificates_retained,
        no_new_refutation=no_new_refutation,
        edit_localized=edit_localized,
        retained_certificates=len(retained),
        prior_certificates=len(prior_certificates),
        reasons=tuple(reasons),
    )
