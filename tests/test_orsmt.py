from __future__ import annotations

import unittest
import json
from fractions import Fraction
from tempfile import TemporaryDirectory
from pathlib import Path

from orsmt import (
    AuditReport,
    EvidenceRecord,
    LinearConstraint,
    LinearExpr,
    ModelIR,
    Objective,
    ORSMTPlugin,
    ReplayRecord,
    align_variable_families,
    audit_scorecard,
    Relation,
    ResultRecord,
    SourceObligation,
    Variable,
    Verdict,
    Verifier,
    build_repair_request,
    cost_profile,
    constraint_error_profile,
    detection_overlap,
    evaluate_repair,
    failure_attribution,
    localization_metrics,
    masked_semantic_error,
    pipeline_profile,
    relation_profile,
    repair_metrics,
    semantic_constraint_metrics,
    severity_profile,
    value_path,
    value_transition_metrics,
    validate_exact_ids,
    with_observable_projection,
    write_reports_atomic,
)
from orsmt.adapters import (
    CallableRuntimeAdapter,
    DictModelAdapter,
    DictResultAdapter,
    DictSourceAdapter,
)
from orsmt.batch import audit_rows


def brownie_model(name: str, share: dict[str, int]) -> ModelIR:
    return ModelIR(
        (
            Variable("brownies", "integer", lb=0),
            Variable("other", "integer", lb=0),
        ),
        (
            LinearConstraint(
                "all_items", LinearExpr.from_mapping({"brownies": 1, "other": 1}), "==", 8
            ),
            LinearConstraint("share", LinearExpr.from_mapping(share), ">=", 0),
        ),
        Objective("max", LinearExpr.from_mapping({"brownies": 2, "other": 4})),
        name,
    )


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = Verifier(timeout_ms=3_000)
        self.source = brownie_model("source", {"brownies": 3, "other": -2})
        self.written = brownie_model("written", {"brownies": 5, "other": -2})
        self.obligation = SourceObligation(
            "share_40",
            LinearConstraint(
                "share", LinearExpr.from_mapping({"brownies": 3, "other": -2}), ">=", 0
            ),
            "at least 40% of all items are brownies",
        )

    def test_entailment_returns_replayed_counterexample(self) -> None:
        record = self.verifier.entails(self.written, self.obligation)
        self.assertEqual(record.verdict, Verdict.REFUTED)
        self.assertTrue(record.replay and record.replay.passed)
        self.assertEqual(record.witness, {"brownies": Fraction(3), "other": Fraction(5)})
        self.assertEqual(record.violation_margin, Fraction(1))
        self.assertGreater(record.normalized_violation or 0, 0)

    def test_native_replay_failure_never_becomes_refutation(self) -> None:
        class RejectingReplay:
            def replay(self, model, point):
                return False, "native solver rejected point"

        record = Verifier(
            timeout_ms=3_000, native_replay=RejectingReplay()
        ).entails(self.written, self.obligation)
        self.assertEqual(record.verdict, Verdict.UNDECIDED)
        self.assertEqual(record.reason, "native_replay_failed")
        self.assertEqual(record.replay.backend, "z3-exact+native")

    def test_bidirectional_region_queries_are_asymmetric(self) -> None:
        source_minus_written, written_minus_source = self.verifier.region_equivalence(
            self.source, self.written, Relation.FORMALIZATION
        )
        self.assertEqual(source_minus_written.verdict, Verdict.CERTIFIED)
        self.assertEqual(written_minus_source.verdict, Verdict.REFUTED)
        self.assertEqual(source_minus_written.metadata["region_direction"], "left_minus_right")
        self.assertEqual(
            written_minus_source.metadata["candidate_fault_if_refuted"],
            "underconstraint",
        )
        self.assertTrue(written_minus_source.replay and written_minus_source.replay.passed)

    def test_incremental_region_discharges_copied_scaled_rows_without_smt(self) -> None:
        variables = (Variable("x", "integer", lb=0, ub=5),)
        left = ModelIR(
            variables,
            (LinearConstraint("left_name", LinearExpr.from_mapping({"x": 1}), "<=", 4),),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "left",
        )
        right = ModelIR(
            variables,
            (LinearConstraint("right_name", LinearExpr.from_mapping({"x": 2}), "<=", 8),),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "right",
        )

        left_minus_right, right_minus_left = self.verifier.region_equivalence(left, right)

        self.assertEqual(left_minus_right.verdict, Verdict.CERTIFIED)
        self.assertEqual(right_minus_left.verdict, Verdict.CERTIFIED)
        self.assertEqual(left_minus_right.metadata["rows_checked"], 0)
        self.assertEqual(right_minus_left.metadata["rows_checked"], 0)
        self.assertEqual(left_minus_right.metadata["common_rows_discharged"], 3)

    def test_positive_objective_scaling_preserves_preference(self) -> None:
        scaled = ModelIR(
            self.source.variables,
            self.source.constraints,
            Objective("max", LinearExpr.from_mapping({"brownies": 4, "other": 8}, 7)),
            "scaled",
        )
        record = self.verifier.preference_equivalence(self.source, scaled)
        self.assertEqual(record.verdict, Verdict.CERTIFIED)
        self.assertEqual(record.metadata["query_strategy"], "positive_affine_score")
        self.assertEqual(record.metadata["score_scale"], Fraction(2))

    def test_objective_sense_is_part_of_direct_preference_check(self) -> None:
        reversed_order = ModelIR(
            self.source.variables,
            self.source.constraints,
            Objective("min", LinearExpr.from_mapping({"brownies": 2, "other": 4})),
            "reversed",
        )
        record = self.verifier.preference_equivalence(self.source, reversed_order)
        self.assertEqual(record.verdict, Verdict.REFUTED)

    def test_different_objective_is_refuted(self) -> None:
        changed = ModelIR(
            self.source.variables,
            self.source.constraints,
            Objective("max", LinearExpr.from_mapping({"brownies": 4, "other": 1})),
            "changed",
        )
        record = self.verifier.preference_equivalence(self.source, changed)
        self.assertEqual(record.verdict, Verdict.REFUTED)
        self.assertTrue(record.replay and record.replay.passed)

    def test_auxiliary_objective_is_verified_through_complete_witnesses(self) -> None:
        reference = ModelIR(
            (Variable("x", "integer", lb=0, ub=3),),
            (),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "reference",
        )
        candidate = ModelIR(
            (
                Variable("x", "integer", lb=0, ub=3),
                Variable("z", "integer", observable=False),
            ),
            (LinearConstraint("define_z", LinearExpr.from_mapping({"z": 1, "x": -2}), "==", 0),),
            Objective("max", LinearExpr.from_mapping({"z": 1})),
            "candidate",
        )
        record = self.verifier.preference_equivalence(reference, candidate)
        self.assertEqual(record.verdict, Verdict.CERTIFIED)

    def test_reporting_checks_are_independent(self) -> None:
        result = ResultRecord({"brownies": 3, "other": 5}, 26, "optimal")
        feasibility, binding, no_better = self.verifier.verify_result(self.written, result)
        self.assertEqual(feasibility.verdict, Verdict.CERTIFIED)
        self.assertEqual(binding.verdict, Verdict.CERTIFIED)
        self.assertEqual(no_better.verdict, Verdict.CERTIFIED)

        wrong_value = ResultRecord({"brownies": 3, "other": 5}, 25, "optimal")
        feasibility, binding, no_better = self.verifier.verify_result(self.written, wrong_value)
        self.assertEqual(feasibility.verdict, Verdict.CERTIFIED)
        self.assertEqual(binding.verdict, Verdict.REFUTED)
        self.assertEqual(no_better.verdict, Verdict.REFUTED)

    def test_plugin_exposes_masked_semantic_error(self) -> None:
        result = ResultRecord({"brownies": 3, "other": 5}, 26, "optimal")
        report = ORSMTPlugin(self.verifier).audit(
            source_obligations=(self.obligation,),
            source_model=self.source,
            written_model=self.written,
            runtime_model=self.written,
            result=result,
        )
        self.assertEqual(report.profile.end_to_end, Verdict.REFUTED)
        self.assertTrue(masked_semantic_error(True, report.records))
        reporting = report.by_relation(Relation.REPORTING)
        self.assertTrue(all(row.verdict == Verdict.CERTIFIED for row in reporting))
        metric_roles = {
            row.metadata.get("metric_role") for row in report.by_relation(Relation.FORMALIZATION)
        }
        self.assertIn("recall", metric_roles)
        self.assertIn("precision", metric_roles)
        target_record = next(
            row
            for row in report.records
            if row.claim_id == "share_40" and row.verdict == Verdict.REFUTED
        )
        request = build_repair_request(
            report, (target_record.relation, target_record.claim_id)
        )
        self.assertEqual(request.witness["brownies"], 3)
        self.assertIn("share", request.implicated)

    def test_reformulation_uses_region_and_preference(self) -> None:
        scaled = ModelIR(
            self.source.variables,
            self.source.constraints,
            Objective("max", LinearExpr.from_mapping({"brownies": 6, "other": 12}, 1)),
            "scaled",
        )
        report = ORSMTPlugin(self.verifier).audit_reformulation(self.source, scaled)
        self.assertEqual(report.profile.end_to_end, Verdict.CERTIFIED)
        self.assertTrue(all(row.relation == Relation.REFORMULATION for row in report.records))

    def test_affine_projection_adjudicates_variable_splitting(self) -> None:
        reference = ModelIR(
            (Variable("x", "integer", lb=0, ub=5),),
            (),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "reference",
        )
        candidate = ModelIR(
            (
                Variable("left", "integer", lb=0),
                Variable("right", "integer", lb=0),
            ),
            (
                LinearConstraint(
                    "capacity",
                    LinearExpr.from_mapping({"left": 1, "right": 1}),
                    "<=",
                    5,
                ),
            ),
            Objective("max", LinearExpr.from_mapping({"left": 1, "right": 1})),
            "split",
        )
        report = ORSMTPlugin(self.verifier).audit_reformulation_projection_pair(
            reference,
            candidate,
            reference_projection={
                "decision": LinearExpr.from_mapping({"x": 1})
            },
            candidate_projection={
                "decision": LinearExpr.from_mapping({"left": 1, "right": 1})
            },
        )
        self.assertEqual(report.profile.end_to_end, Verdict.CERTIFIED)
        self.assertEqual(
            report.metadata["projection_mode"], "affine_definitional"
        )

    def test_invalid_projection_is_undecided(self) -> None:
        projected = with_observable_projection(
            self.source,
            {
                "total": LinearExpr.from_mapping(
                    {"brownies": 1, "other": 1}
                )
            },
        )
        self.assertEqual(
            projected.observable_names, ("__orsmt_observable__total",)
        )
        report = ORSMTPlugin(self.verifier).audit_reformulation_projection_pair(
            self.source,
            self.written,
            reference_projection={
                "decision": LinearExpr.from_mapping({"missing": 1})
            },
            candidate_projection={
                "decision": LinearExpr.from_mapping({"brownies": 1})
            },
        )
        self.assertEqual(report.profile.end_to_end, Verdict.UNDECIDED)
        self.assertTrue(
            report.records[0].reason.startswith("invalid_projection")
        )

    def test_unique_family_alignment_handles_renaming_and_auxiliaries(self) -> None:
        reference = ModelIR(
            (Variable("x[0]", "integer"), Variable("x[1]", "integer")),
            (LinearConstraint("cap", LinearExpr.from_mapping({"x[0]": 1, "x[1]": 1}), "<=", 3),),
            Objective("max", LinearExpr.from_mapping({"x[0]": 1, "x[1]": 2})),
            "reference",
        )
        candidate = ModelIR(
            (
                Variable("renamed[0]", "integer"),
                Variable("renamed[1]", "integer"),
                Variable("slack", "continuous", lb=0),
            ),
            (
                LinearConstraint(
                    "cap_with_slack",
                    LinearExpr.from_mapping({"renamed[0]": 1, "renamed[1]": 1, "slack": 1}),
                    "==",
                    3,
                ),
            ),
            Objective("max", LinearExpr.from_mapping({"renamed[0]": 1, "renamed[1]": 2})),
            "candidate",
        )
        alignment = align_variable_families(reference, candidate)
        self.assertTrue(alignment.available)
        self.assertEqual(alignment.family_map, {"renamed": "x"})
        self.assertEqual(alignment.auxiliary_families, ("slack",))
        self.assertEqual(set(alignment.reference.observable_names), {"x[0]", "x[1]"})
        self.assertEqual(set(alignment.candidate.observable_names), {"x[0]", "x[1]"})
        report = ORSMTPlugin(self.verifier).audit_reformulation(
            alignment.reference, alignment.candidate
        )
        self.assertEqual(report.profile.end_to_end, Verdict.CERTIFIED)

    def test_ambiguous_family_alignment_abstains(self) -> None:
        reference = ModelIR(
            (Variable("x[0]", "integer"), Variable("x[1]", "integer")),
            (),
            None,
            "reference",
        )
        candidate = ModelIR(
            (
                Variable("a[0]", "integer"),
                Variable("a[1]", "integer"),
                Variable("b[0]", "integer"),
                Variable("b[1]", "integer"),
            ),
            (),
            None,
            "candidate",
        )
        alignment = align_variable_families(reference, candidate)
        self.assertFalse(alignment.available)
        self.assertIn("ambiguous_or_missing", alignment.reason or "")
        report = ORSMTPlugin(self.verifier).audit_reformulation_pair(reference, candidate)
        self.assertEqual(report.profile.end_to_end, Verdict.UNDECIDED)
        self.assertFalse(report.metadata["alignment_available"])


class MetricAndAdapterTests(unittest.TestCase):
    @staticmethod
    def record(verdict: Verdict, name: str = "c") -> EvidenceRecord:
        return EvidenceRecord(Relation.FORMALIZATION, name, verdict, "q")

    def test_all_denominator_metrics(self) -> None:
        rows = [
            self.record(Verdict.CERTIFIED, "a"),
            self.record(Verdict.REFUTED, "b"),
            self.record(Verdict.UNDECIDED, "c"),
        ]
        profile = relation_profile(rows)
        self.assertEqual(profile.total, 3)
        self.assertAlmostEqual(profile.certified_rate_all, 1 / 3)
        metrics = semantic_constraint_metrics(rows, rows[:2])
        self.assertAlmostEqual(metrics.semantic_recall, 1 / 3)
        self.assertAlmostEqual(metrics.semantic_precision, 1 / 2)

        with_na = rows + [
            EvidenceRecord(
                Relation.FORMALIZATION,
                "d",
                Verdict.UNDECIDED,
                "q",
                reason="not_applicable",
            )
        ]
        applicable = relation_profile(with_na)
        self.assertEqual(applicable.total, 4)
        self.assertEqual(applicable.applicable, 3)
        self.assertEqual(applicable.not_applicable, 1)
        self.assertEqual(applicable.undecided, 1)
        self.assertAlmostEqual(applicable.availability_rate_all, 3 / 4)
        self.assertAlmostEqual(applicable.not_applicable_rate_all, 1 / 4)
        self.assertAlmostEqual(applicable.refuted_rate_applicable, 1 / 3)
        semantic_with_na = semantic_constraint_metrics(with_na, with_na)
        self.assertEqual(semantic_with_na.reference_assigned, 4)
        self.assertEqual(semantic_with_na.reference_total, 3)
        self.assertEqual(semantic_with_na.reference_not_applicable, 1)
        self.assertAlmostEqual(semantic_with_na.semantic_recall, 1 / 3)

    def test_value_localization_and_repair_metrics(self) -> None:
        path = value_path(Fraction(24), Fraction(26), Fraction(26), Fraction(26))
        self.assertGreater(path.source_to_written_gap or 0, 0)
        self.assertEqual(path.written_to_runtime_gap, 0)
        self.assertEqual(path.first_gap, "source_to_written")
        self.assertGreater(path.first_gap_signed or 0, 0)
        localization = localization_metrics([["share", "obj"]], [["share"]], k=1)
        self.assertEqual(localization.top_k_recall, 1)
        repair = repair_metrics(
            [Verdict.REFUTED, Verdict.CERTIFIED],
            [Verdict.CERTIFIED, Verdict.CERTIFIED],
            [True, False],
            [True, True],
            [True, False],
        )
        self.assertEqual(repair.wrong_to_right, 1)
        self.assertEqual(repair.right_to_wrong, 0)
        self.assertEqual(repair.semantic_refuted_to_certified, 1)
        self.assertEqual(repair.semantic_certified_to_certified, 1)
        transition_total = sum(
            (
                repair.semantic_certified_to_certified,
                repair.semantic_certified_to_refuted,
                repair.semantic_certified_to_undecided,
                repair.semantic_refuted_to_certified,
                repair.semantic_refuted_to_refuted,
                repair.semantic_refuted_to_undecided,
                repair.semantic_undecided_to_certified,
                repair.semantic_undecided_to_refuted,
                repair.semantic_undecided_to_undecided,
            )
        )
        self.assertEqual(transition_total, repair.total)
        values = value_transition_metrics([False, True, None], [True, False, True])
        self.assertEqual(values.observed_pairs, 2)
        self.assertEqual(values.value_wrong_to_right, 1)
        self.assertEqual(values.value_right_to_wrong, 1)

    def test_detection_severity_cost_and_attribution(self) -> None:
        rows = [
            EvidenceRecord(
                Relation.FORMALIZATION,
                "a",
                Verdict.REFUTED,
                "q",
                violation_margin=Fraction(3, 2),
                duration_ms=2.0,
            ),
            EvidenceRecord(
                Relation.IMPLEMENTATION,
                "b",
                Verdict.CERTIFIED,
                "q",
                duration_ms=4.0,
            ),
        ]
        overlap = detection_overlap(["a", "b", "c"], ["a", "b"], ["b", "c"])
        self.assertEqual(overlap.ours_only, 1)
        self.assertEqual(overlap.baseline_only, 1)
        severity = severity_profile(rows)
        self.assertEqual(severity.max_margin, 1.5)
        constraints = constraint_error_profile(rows)
        self.assertEqual(constraints.error_rate_all, 0.5)
        cost = cost_profile(rows)
        self.assertEqual(cost.total_ms, 6.0)
        attribution = failure_attribution(rows, value_agrees=True)
        self.assertEqual(attribution.label, "formalization")
        self.assertTrue(attribution.masked_by_value_agreement)

    def test_audit_scorecard_keeps_artifacts_and_diagnoses_faults(self) -> None:
        groups = [
            (
                EvidenceRecord(
                    Relation.FORMALIZATION,
                    "missing-capacity",
                    Verdict.REFUTED,
                    "q",
                    implicated=("capacity",),
                    replay=ReplayRecord(True, "native"),
                    metadata={
                        "metric_role": "recall",
                        "candidate_fault_if_refuted": "underconstraint",
                    },
                    duration_ms=2.0,
                ),
                EvidenceRecord(
                    Relation.IMPLEMENTATION,
                    "region:written\\runtime",
                    Verdict.CERTIFIED,
                    "q",
                    replay=ReplayRecord(True, "native"),
                    duration_ms=3.0,
                ),
            ),
            (
                EvidenceRecord(
                    Relation.FORMALIZATION,
                    "precision:extra-row",
                    Verdict.REFUTED,
                    "q",
                    replay=ReplayRecord(True, "native"),
                    metadata={
                        "metric_role": "precision",
                        "candidate_fault_if_refuted": "overconstraint",
                    },
                    duration_ms=4.0,
                ),
                EvidenceRecord(
                    Relation.REPORTING,
                    "result:value_binding",
                    Verdict.REFUTED,
                    "q",
                    implicated=("objective",),
                    replay=ReplayRecord(True, "native"),
                    duration_ms=5.0,
                ),
            ),
            (
                EvidenceRecord(
                    Relation.FORMALIZATION,
                    "availability",
                    Verdict.UNDECIDED,
                    "q",
                    reason="source_target_unavailable",
                ),
            ),
        ]
        scorecard = audit_scorecard(groups, [True, False, None])
        self.assertEqual(scorecard.artifacts.total, 3)
        self.assertEqual(scorecard.artifacts.refuted, 2)
        self.assertEqual(scorecard.artifacts.undecided, 1)
        self.assertEqual(scorecard.diagnostics.underconstraint_claims, 1)
        self.assertEqual(scorecard.diagnostics.overconstraint_claims, 1)
        self.assertEqual(scorecard.diagnostics.reporting_claims, 1)
        self.assertEqual(scorecard.diagnostics.multi_claim_artifacts, 1)
        self.assertEqual(scorecard.diagnostics.multi_layer_artifacts, 1)
        self.assertEqual(scorecard.diagnostics.value_checked_artifacts, 2)
        self.assertEqual(scorecard.diagnostics.value_agreement_artifacts, 1)
        self.assertEqual(scorecard.diagnostics.value_disagreement_artifacts, 1)
        self.assertEqual(scorecard.diagnostics.value_agreement_rate_checked, 0.5)
        self.assertEqual(scorecard.diagnostics.value_checked_refuted_artifacts, 2)
        self.assertEqual(scorecard.diagnostics.value_masked_artifacts, 1)
        self.assertEqual(scorecard.diagnostics.value_mask_rate_among_refuted, 0.5)
        self.assertEqual(
            scorecard.diagnostics.refutation_rate_given_value_agreement, 1.0
        )
        self.assertEqual(scorecard.diagnostics.replay_rate, 1.0)
        self.assertEqual(scorecard.semantic_constraints.semantic_recall, 0.0)
        self.assertEqual(scorecard.semantic_constraints.semantic_precision, 0.0)
        self.assertEqual(scorecard.cost.total_ms, 14.0)
        json.dumps(scorecard.to_dict())

    def test_certificate_preserving_repair_contract(self) -> None:
        target = (Relation.FORMALIZATION, "share")
        before_rows = (
            EvidenceRecord(Relation.FORMALIZATION, "share", Verdict.REFUTED, "q"),
            EvidenceRecord(Relation.IMPLEMENTATION, "region", Verdict.CERTIFIED, "q"),
        )
        after_rows = (
            EvidenceRecord(Relation.FORMALIZATION, "share", Verdict.CERTIFIED, "q"),
            EvidenceRecord(Relation.IMPLEMENTATION, "region", Verdict.CERTIFIED, "q"),
        )
        before = AuditReport(before_rows, pipeline_profile(before_rows), {})
        after = AuditReport(after_rows, pipeline_profile(after_rows), {})
        evaluation = evaluate_repair(
            before,
            after,
            target,
            changed_relations=(Relation.FORMALIZATION,),
            allowed_relations=(Relation.FORMALIZATION,),
        )
        self.assertTrue(evaluation.accepted)
        rejected = evaluate_repair(
            before,
            after,
            target,
            changed_relations=(Relation.FORMALIZATION, Relation.IMPLEMENTATION),
            allowed_relations=(Relation.FORMALIZATION,),
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("edit_outside_contract", rejected.reasons)

    def test_atomic_evidence_ledger_and_exact_ids(self) -> None:
        row = EvidenceRecord(Relation.FORMALIZATION, "a", Verdict.CERTIFIED, "q")
        report = AuditReport((row,), pipeline_profile((row,)), {"source": "test"})
        with TemporaryDirectory() as directory:
            output = Path(directory) / "ledger.jsonl"
            write_reports_atomic(output, [("case-1", report)])
            self.assertIn('"id": "case-1"', output.read_text())
            validate_exact_ids(["case-1"], ["case-1"])
            with self.assertRaises(ValueError):
                validate_exact_ids(["case-1"], ["case-2"])

    def test_dict_adapter(self) -> None:
        ir = DictModelAdapter().to_ir(
            {
                "name": "tiny",
                "variables": [{"name": "x", "type": "int", "lb": 0}],
                "constraints": [
                    {"name": "cap", "lhs": [{"var": "x", "coef": 1}], "op": "<=", "rhs": 3}
                ],
                "objective": {"sense": "max", "terms": {"x": 2}},
            }
        )
        self.assertEqual(ir.variables[0].kind, "integer")
        self.assertEqual(ir.constraints[0].lhs.evaluate({"x": 2}), 2)

    def test_dict_adapter_moves_structured_rhs_exactly(self) -> None:
        ir = DictModelAdapter().to_ir(
            {
                "variables": [
                    {"name": "x", "type": "int"},
                    {"name": "y", "type": "int"},
                ],
                "constraints": [
                    {
                        "name": "link",
                        "lhs": [{"var": "x", "coef": 1}],
                        "op": ">=",
                        "rhs": {"$divide": [{"$multiply": [2, "y"]}, 4]},
                    }
                ],
                "objective": {"sense": "max", "lhs": [{"var": "x", "coef": 1}]},
            }
        )
        row = ir.constraints[0]
        self.assertEqual(row.rhs, 0)
        self.assertEqual(row.lhs.evaluate({"x": 3, "y": 4}), 1)

    def test_host_capture_adapter_drives_complete_artifact_audit(self) -> None:
        raw_model = {
            "variables": [
                {"name": "brownies", "type": "int", "lb": 0},
                {"name": "other", "type": "int", "lb": 0},
            ],
            "constraints": [
                {
                    "name": "all_items",
                    "lhs": [{"var": "brownies", "coef": 1}, {"var": "other", "coef": 1}],
                    "op": "==",
                    "rhs": 8,
                },
                {
                    "name": "share",
                    "lhs": [{"var": "brownies", "coef": 3}, {"var": "other", "coef": -2}],
                    "op": ">=",
                    "rhs": 0,
                },
            ],
            "objective": {
                "sense": "max",
                "lhs": [{"var": "brownies", "coef": 2}, {"var": "other", "coef": 4}],
            },
        }
        obligation = {
            "obligations": [
                {
                    "claim_id": "share_40",
                    "source_span": "at least 40% of all items are brownies",
                    "constraint": raw_model["constraints"][1],
                }
            ]
        }
        model_adapter = DictModelAdapter()
        report = ORSMTPlugin(Verifier(timeout_ms=3_000)).audit_artifact(
            source_specification=obligation,
            source_reference=raw_model,
            written_artifact=raw_model,
            executable_artifact="solver program",
            returned_artifact={
                "assignment": {"brownies": 4, "other": 4},
                "objective_value": 24,
                "status": "optimal",
            },
            source_adapter=DictSourceAdapter(),
            model_adapter=model_adapter,
            runtime_adapter=CallableRuntimeAdapter(
                lambda _code: {"status": "CAPTURED", "ir": raw_model}, model_adapter
            ),
            result_adapter=DictResultAdapter(),
            metadata={"id": "trace"},
        )
        self.assertEqual(report.profile.end_to_end, Verdict.CERTIFIED)
        self.assertEqual(report.metadata["adapter_errors"], {})

    def test_capture_failure_is_explicitly_undecided(self) -> None:
        raw_model = {
            "variables": [{"name": "x", "type": "int", "lb": 0, "ub": 1}],
            "constraints": [],
            "objective": {"sense": "max", "lhs": [{"var": "x", "coef": 1}]},
        }
        model_adapter = DictModelAdapter()
        report = ORSMTPlugin(Verifier(timeout_ms=3_000)).audit_artifact(
            written_artifact=raw_model,
            executable_artifact="bad program",
            model_adapter=model_adapter,
            runtime_adapter=CallableRuntimeAdapter(
                lambda _code: {"status": "TIMEOUT", "error": "host timeout"}, model_adapter
            ),
            result_adapter=DictResultAdapter(),
        )
        self.assertIn("runtime_model", report.metadata["adapter_errors"])
        self.assertTrue(
            all(
                row.verdict == Verdict.UNDECIDED
                for row in report.records
                if row.relation in (Relation.IMPLEMENTATION, Relation.REPORTING)
            )
        )

    def test_partial_natural_rows_remain_in_all_denominators(self) -> None:
        formulation = {
            "variables": [{"name": "x", "type": "int", "lb": 0, "ub": 3}],
            "constraints": [
                {"name": "cap", "lhs": [{"var": "x", "coef": 1}], "op": "<=", "rhs": 3}
            ],
            "objective": {"sense": "max", "lhs": [{"var": "x", "coef": 2}]},
        }
        reports = audit_rows(
            [
                {
                    "id": "complexor:1",
                    "source": "complexor",
                    "model_key": "14b",
                    "formulation": formulation,
                    "parse_note": "ok",
                },
                {
                    "id": "text2opt_template:long",
                    "source": None,
                    "model_key": "14b",
                    "formulation": None,
                    "code": None,
                    "parse_note": "prompt_over_budget",
                },
            ],
            timeout_ms=3_000,
        )
        self.assertEqual([row_id for row_id, _ in reports], ["complexor:1", "text2opt_template:long"])
        for _, report in reports:
            self.assertEqual(set(report.profile.by_relation), {
                Relation.FORMALIZATION,
                Relation.IMPLEMENTATION,
                Relation.REPORTING,
            })
            self.assertEqual(report.profile.end_to_end, Verdict.UNDECIDED)
        self.assertEqual(reports[1][1].metadata["source"], "text2opt_template")
        self.assertEqual(
            reports[1][1].by_relation(Relation.FORMALIZATION)[0].reason,
            "written_model_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
