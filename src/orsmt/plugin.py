from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .alignment import align_variable_families
from .adapters.protocols import ModelAdapter, ResultAdapter, RuntimeAdapter, SourceAdapter
from .constructive import (
    ConstructiveAudit,
    ConstructiveVerifier,
    ContractStatus,
    DSLValidationError,
    ReformulationVerdict,
    SemanticView,
    model_fingerprint,
    parse_constructive_contract,
)
from .family_mapping import FAMILY_SCHEMA_VERSION, parse_family_constructive_contract
from .correspondence import parse_candidate_projection
from .evidence import EvidenceRecord, Relation, Verdict
from .ir import LinearExpr, ModelIR, ResultRecord, SourceObligation
from .metrics import AuditScorecard, PipelineProfile, audit_scorecard, pipeline_profile
from .projection import with_observable_projection
from .verify import Verifier


@dataclass(frozen=True)
class AuditReport:
    records: tuple[EvidenceRecord, ...]
    profile: PipelineProfile
    metadata: Mapping[str, Any]

    def by_relation(self, relation: Relation) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.relation == relation)

    def scorecard(self, value_agreement: bool | None = None) -> AuditScorecard:
        """Return the same all-denominator scorecard used by batch evaluation."""

        return audit_scorecard((self.records,), (value_agreement,))

    def to_dict(self) -> dict[str, Any]:
        profile = asdict(self.profile)
        profile["end_to_end"] = self.profile.end_to_end.value
        profile["by_relation"] = {
            relation.value: asdict(value) for relation, value in self.profile.by_relation.items()
        }
        return {
            "records": [record.to_dict() for record in self.records],
            "profile": profile,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReformulationAuditBundle:
    """Preserve direct and mapper-assisted reformulation evidence separately."""

    direct: AuditReport
    proposal: AuditReport | None
    effective: AuditReport
    route_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct": self.direct.to_dict(),
            "proposal": self.proposal.to_dict() if self.proposal is not None else None,
            "effective": self.effective.to_dict(),
            "route_eligible": self.route_eligible,
        }


@dataclass(frozen=True)
class ConstructiveRouteResult:
    """Keep map-contract evidence separate from the model verdict."""

    audit: ConstructiveAudit
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"audit": self.audit.to_dict(), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class ConstructiveReformulationBundle:
    """Direct relation evidence plus an optional constructive fallback."""

    direct: AuditReport
    constructive: ConstructiveRouteResult | None
    effective_verdict: Verdict
    route_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct": self.direct.to_dict(),
            "constructive": (
                self.constructive.to_dict()
                if self.constructive is not None
                else None
            ),
            "effective_verdict": self.effective_verdict.value,
            "route_eligible": self.route_eligible,
        }


class ORSMTPlugin:
    """Plug-in entry point for audit and counterexample-guided repair systems.

    The core never asks an LLM to decide correctness. An upstream LLM may
    generate artifacts and a downstream LLM may consume the evidence records.
    """

    def __init__(self, verifier: Verifier | None = None) -> None:
        self.verifier = verifier or Verifier()

    def audit(
        self,
        *,
        written_model: ModelIR | None,
        runtime_model: ModelIR | None,
        source_obligations: Iterable[SourceObligation] = (),
        source_model: ModelIR | None = None,
        result: ResultRecord | None = None,
        constraint_correspondence: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        records: list[EvidenceRecord] = []

        obligations = tuple(source_obligations)
        if written_model is None:
            records.append(
                EvidenceRecord(
                    Relation.FORMALIZATION,
                    "formalization:availability",
                    Verdict.UNDECIDED,
                    "source specification agrees with written model",
                    reason="written_model_unavailable",
                )
            )
        elif source_model is None and not obligations:
            records.append(
                EvidenceRecord(
                    Relation.FORMALIZATION,
                    "formalization:availability",
                    Verdict.UNDECIDED,
                    "source specification agrees with written model",
                    reason="source_target_unavailable",
                )
            )
        else:
            records.extend(self.verifier.audit_obligations(written_model, obligations))

        if source_model is not None and written_model is not None:
            if constraint_correspondence:
                recall, precision = self.verifier.constraint_correspondence(
                    source_model, written_model, relation=Relation.FORMALIZATION
                )
                records.extend(recall)
                records.extend(precision)
            records.extend(
                self.verifier.region_equivalence(
                    source_model, written_model, relation=Relation.FORMALIZATION
                )
            )
            records.append(
                self.verifier.preference_equivalence(
                    source_model, written_model, relation=Relation.FORMALIZATION
                )
            )

        if written_model is None or runtime_model is None:
            missing = "written_model_unavailable" if written_model is None else "runtime_model_unavailable"
            records.append(
                EvidenceRecord(
                    Relation.IMPLEMENTATION,
                    "implementation:availability",
                    Verdict.UNDECIDED,
                    "written model agrees with captured runtime model",
                    reason=missing,
                )
            )
        else:
            records.extend(
                self.verifier.region_equivalence(
                    written_model, runtime_model, relation=Relation.IMPLEMENTATION
                )
            )
            records.append(
                self.verifier.preference_equivalence(
                    written_model, runtime_model, relation=Relation.IMPLEMENTATION
                )
            )

        if runtime_model is None:
            records.append(
                EvidenceRecord(
                    Relation.REPORTING,
                    "reporting:availability",
                    Verdict.UNDECIDED,
                    "returned result agrees with captured runtime model",
                    reason="runtime_model_unavailable",
                )
            )
        elif result is None:
            records.append(
                EvidenceRecord(
                    Relation.REPORTING,
                    "reporting:availability",
                    Verdict.UNDECIDED,
                    "returned result agrees with captured runtime model",
                    reason="result_unavailable",
                )
            )
        else:
            records.extend(self.verifier.verify_result(runtime_model, result))

        return AuditReport(tuple(records), pipeline_profile(records), metadata or {})

    def audit_artifact(
        self,
        *,
        source_specification: Any = None,
        written_artifact: Any = None,
        executable_artifact: Any = None,
        returned_artifact: Any = None,
        source_reference: Any = None,
        source_adapter: SourceAdapter | None = None,
        model_adapter: ModelAdapter,
        runtime_adapter: RuntimeAdapter,
        result_adapter: ResultAdapter,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Adapt one host artifact and preserve every adapter failure as evidence."""

        adapter_errors: dict[str, str] = {}

        def convert(name: str, operation, value):
            if value is None:
                return None
            try:
                return operation(value)
            except Exception as error:  # host adapters are an explicit trust boundary
                adapter_errors[name] = f"{type(error).__name__}:{str(error)[:240]}"
                return None

        obligations = ()
        if source_specification is not None and source_adapter is not None:
            obligations = convert(
                "source_obligations", source_adapter.obligations, source_specification
            ) or ()
        source_model = convert("source_model", model_adapter.to_ir, source_reference)
        written_model = convert("written_model", model_adapter.to_ir, written_artifact)
        runtime_model = convert("runtime_model", runtime_adapter.capture, executable_artifact)
        result = convert("result", result_adapter.to_result, returned_artifact)
        audit_metadata = dict(metadata or {})
        audit_metadata["adapter_errors"] = adapter_errors
        return self.audit(
            source_obligations=obligations,
            source_model=source_model,
            written_model=written_model,
            runtime_model=runtime_model,
            result=result,
            metadata=audit_metadata,
        )

    def audit_reformulation(
        self,
        reference: ModelIR,
        reformulation: ModelIR,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        records = list(
            self.verifier.region_equivalence(
                reference, reformulation, relation=Relation.REFORMULATION
            )
        )
        records.append(
            self.verifier.preference_equivalence(
                reference, reformulation, relation=Relation.REFORMULATION
            )
        )
        return AuditReport(tuple(records), pipeline_profile(records), metadata or {})

    def audit_reformulation_pair(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Align an instantiated pair without labels, then issue relational queries."""

        alignment = align_variable_families(
            reference,
            candidate,
            semantic_mode="objective_anchored",
        )
        aligned_metadata = dict(metadata or {})
        aligned_metadata.update(
            {
                "alignment_available": alignment.available,
                "family_map": dict(alignment.family_map),
                "auxiliary_families": alignment.auxiliary_families,
                "semantic_families": alignment.semantic_families,
                "semantic_mode": alignment.semantic_mode,
                "alignment_reason": alignment.reason,
            }
        )
        if not alignment.available:
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "alignment:availability",
                Verdict.UNDECIDED,
                "reference and candidate admit a unique observable alignment",
                reason=alignment.reason,
            )
            return AuditReport((record,), pipeline_profile((record,)), aligned_metadata)
        assert alignment.reference is not None and alignment.candidate is not None
        return self.audit_reformulation(
            alignment.reference,
            alignment.candidate,
            metadata=aligned_metadata,
        )

    def audit_reformulation_projection_pair(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        reference_projection: Mapping[str, LinearExpr],
        candidate_projection: Mapping[str, LinearExpr],
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Adjudicate a proposed affine semantic correspondence.

        The proposal may come from a deterministic mapper or an LLM. Invalid,
        incomplete, or differently keyed proposals never become verdicts.
        """

        projection_metadata = dict(metadata or {})
        projection_metadata.setdefault("projection_mode", "affine_definitional")
        if not reference_projection or set(reference_projection) != set(
            candidate_projection
        ):
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "projection:availability",
                Verdict.UNDECIDED,
                "reference and candidate expose the same semantic decisions",
                reason="projection_schema_mismatch",
            )
            return AuditReport(
                (record,), pipeline_profile((record,)), projection_metadata
            )
        try:
            projected_reference = with_observable_projection(
                reference,
                reference_projection,
                name=f"{reference.name}[semantic]",
            )
            projected_candidate = with_observable_projection(
                candidate,
                candidate_projection,
                name=f"{candidate.name}[semantic]",
            )
        except (TypeError, ValueError) as error:
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "projection:availability",
                Verdict.UNDECIDED,
                "reference and candidate expose the same semantic decisions",
                reason=f"invalid_projection:{type(error).__name__}",
                metadata={"detail": str(error)[:240]},
            )
            return AuditReport(
                (record,), pipeline_profile((record,)), projection_metadata
            )
        projection_metadata["projection_names"] = tuple(
            sorted(reference_projection)
        )
        projection_metadata["alignment_available"] = True
        projection_metadata["alignment_reason"] = "shared_projection_schema"
        return self.audit_reformulation(
            projected_reference,
            projected_candidate,
            metadata=projection_metadata,
        )

    def audit_candidate_projection_pair(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        candidate_projection: Mapping[str, LinearExpr],
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Adjudicate an untrusted candidate-to-reference affine proposal.

        The semantic vocabulary is fixed to every observable reference
        variable.  A mapper may define only the candidate-side expression for
        each such variable; it cannot delete a difficult reference decision or
        collapse both models to a constant projection.  This is the preferred
        entry point for LLM-proposed correspondences.
        """

        proposal_metadata = dict(metadata or {})
        proposal_metadata.update(
            {
                "projection_mode": "candidate_to_reference_identity",
                "reference_observable_count": len(reference.observable_names),
                "proposal_key_count": len(candidate_projection),
            }
        )
        reference_names = tuple(reference.observable_names)
        if not reference_names or set(candidate_projection) != set(reference_names):
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "projection:availability",
                Verdict.UNDECIDED,
                "every observable reference decision has one candidate expression",
                reason="projection_schema_mismatch",
            )
            return AuditReport(
                (record,), pipeline_profile((record,)), proposal_metadata
            )
        constant_only = tuple(
            sorted(
                name
                for name, expression in candidate_projection.items()
                if not expression.support
            )
        )
        if constant_only:
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "projection:availability",
                Verdict.UNDECIDED,
                "candidate expressions depend on captured candidate decisions",
                reason="constant_only_candidate_projection",
                metadata={"semantic_names": constant_only},
            )
            return AuditReport(
                (record,), pipeline_profile((record,)), proposal_metadata
            )
        reference_projection = {
            name: LinearExpr.from_mapping({name: 1}) for name in reference_names
        }
        return self.audit_reformulation_projection_pair(
            reference,
            candidate,
            reference_projection=reference_projection,
            candidate_projection=candidate_projection,
            metadata=proposal_metadata,
        )

    def audit_candidate_projection_payload(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        payload: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditReport:
        """Admit an untrusted JSON-like mapper response, then adjudicate it.

        The parser performs only deterministic schema, exact-number, and
        variable-vocabulary checks.  A missing or malformed proposal receives
        no SMT verdict and is represented as an unavailable ``UNDECIDED``
        route.  A well-typed proposal is passed unchanged to the exact relation
        checker.
        """

        proposal = parse_candidate_projection(reference, candidate, payload)
        proposal_metadata = dict(metadata or {})
        proposal_metadata.update(
            {
                "projection_mode": "candidate_to_reference_payload",
                "proposal_returned": proposal.returned,
                "proposal_well_typed": proposal.well_typed,
                "proposal_reason": proposal.reason,
            }
        )
        if not proposal.well_typed:
            record = EvidenceRecord(
                Relation.REFORMULATION,
                "projection:availability",
                Verdict.UNDECIDED,
                "mapper returns one well-typed affine candidate expression per reference decision",
                reason=f"projection_proposal:{proposal.reason}",
            )
            return AuditReport(
                (record,), pipeline_profile((record,)), proposal_metadata
            )
        return self.audit_candidate_projection_pair(
            reference,
            candidate,
            candidate_projection=proposal.projection,
            metadata=proposal_metadata,
        )

    def audit_reformulation_with_candidate_projection_payload(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        payload: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReformulationAuditBundle:
        """Run deterministic alignment first and route only alignment abstentions.

        Direct C/R evidence is final and never overwritten by a mapper.  A
        proposal is requested only when structural alignment is unavailable;
        solver timeouts and other post-alignment unknowns are not sent to the
        mapper because correspondence is not their cause.  The returned bundle
        retains both evidence paths for exact contribution accounting.
        """

        direct = self.audit_reformulation_pair(
            reference, candidate, metadata=metadata
        )
        route_eligible = (
            direct.profile.end_to_end == Verdict.UNDECIDED
            and direct.metadata.get("alignment_available") is False
        )
        if not route_eligible:
            return ReformulationAuditBundle(
                direct=direct,
                proposal=None,
                effective=direct,
                route_eligible=False,
            )

        proposal_metadata = dict(metadata or {})
        proposal_metadata["route_trigger"] = "structural_alignment_unavailable"
        proposal = self.audit_candidate_projection_payload(
            reference,
            candidate,
            payload=payload,
            metadata=proposal_metadata,
        )
        return ReformulationAuditBundle(
            direct=direct,
            proposal=proposal,
            effective=proposal,
            route_eligible=True,
        )

    def audit_constructive_reformulation_payload(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        payload: Mapping[str, Any] | None,
        semantic_view: SemanticView | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConstructiveRouteResult:
        """Admit and verify a total bidirectional constructive map.

        The optional mapper supplies only the map contract.  The semantic view
        is fixed by a trusted caller or by objective-anchored deterministic
        alignment, and the exact engine alone assigns both contract status and
        model C/R/U.  A bad map becomes a model refutation only after an
        independent counterpart query and replay establish a genuine region
        difference.
        """

        route_metadata = dict(metadata or {})
        route_metadata["proposal_returned"] = payload is not None
        left = reference
        right = candidate
        view = semantic_view
        if view is None:
            alignment = align_variable_families(
                reference,
                candidate,
                semantic_mode="objective_anchored",
            )
            route_metadata.update(
                {
                    "alignment_available": alignment.available,
                    "alignment_reason": alignment.reason,
                    "family_map": dict(alignment.family_map),
                    "semantic_families": alignment.semantic_families,
                    "semantic_mode": alignment.semantic_mode,
                    "semantic_view_source": "deterministic_objective_anchored",
                }
            )
            if not alignment.available:
                return ConstructiveRouteResult(
                    ConstructiveAudit(
                        ContractStatus.UNKNOWN,
                        ReformulationVerdict.U,
                        reason=f"semantic_view_unavailable:{alignment.reason}",
                    ),
                    route_metadata,
                )
            assert alignment.reference is not None and alignment.candidate is not None
            left = alignment.reference
            right = alignment.candidate
            view = SemanticView.by_name(left, right)
        else:
            route_metadata["semantic_view_source"] = "caller_trusted"

        route_metadata.update(
            {
                "reference_fingerprint": model_fingerprint(left),
                "candidate_fingerprint": model_fingerprint(right),
                "semantic_pairs": tuple(view.pairs),
                "contract_schema": (
                    FAMILY_SCHEMA_VERSION
                    if isinstance(payload, Mapping)
                    and payload.get("schema_version") == FAMILY_SCHEMA_VERSION
                    else "exact_total_bidirectional_affine_ite"
                ),
            }
        )
        if payload is None:
            return ConstructiveRouteResult(
                ConstructiveAudit(
                    ContractStatus.UNKNOWN,
                    ReformulationVerdict.U,
                    reason="contract_not_returned",
                ),
                route_metadata,
            )
        try:
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == FAMILY_SCHEMA_VERSION
            ):
                contract = parse_family_constructive_contract(payload, left, right)
            else:
                contract = parse_constructive_contract(payload, left, right)
        except DSLValidationError as error:
            route_metadata["contract_admitted"] = False
            return ConstructiveRouteResult(
                ConstructiveAudit(
                    ContractStatus.INVALID,
                    ReformulationVerdict.U,
                    reason=f"contract_not_admitted:{str(error)[:240]}",
                ),
                route_metadata,
            )
        route_metadata["contract_admitted"] = True
        engine = ConstructiveVerifier(
            timeout_ms=self.verifier.timeout_ms,
            native_replay=self.verifier.native_replay,
        )
        return ConstructiveRouteResult(
            engine.verify(left, right, contract, view),
            route_metadata,
        )

    def audit_reformulation_with_constructive_payload(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        *,
        payload: Mapping[str, Any] | None,
        semantic_view: SemanticView | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConstructiveReformulationBundle:
        """Use constructive transport only after the direct exact route abstains."""

        direct = self.audit_reformulation_pair(
            reference,
            candidate,
            metadata=metadata,
        )
        route_eligible = direct.profile.end_to_end == Verdict.UNDECIDED
        if not route_eligible:
            return ConstructiveReformulationBundle(
                direct,
                None,
                direct.profile.end_to_end,
                False,
            )
        constructive = self.audit_constructive_reformulation_payload(
            reference,
            candidate,
            payload=payload,
            semantic_view=semantic_view,
            metadata=metadata,
        )
        effective = {
            ReformulationVerdict.C: Verdict.CERTIFIED,
            ReformulationVerdict.R: Verdict.REFUTED,
            ReformulationVerdict.U: Verdict.UNDECIDED,
        }[constructive.audit.reformulation_verdict]
        return ConstructiveReformulationBundle(
            direct,
            constructive,
            effective,
            True,
        )
