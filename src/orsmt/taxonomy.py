"""Separate proved semantic effects, observed syntax, and redundancy properties."""
from collections import Counter
from dataclasses import replace

from .evidence import Relation, Verdict
from .ir import SourceObligation
from .repair_workflow import diagnose_repair


EFFECTS = {
    'underconstraint': ('constraints', 'feasible_region_expansion'),
    'overconstraint': ('constraints', 'feasible_region_restriction'),
    'preference': ('objective', 'objective_preference_mismatch'),
    'objective_value': ('objective', 'objective_value_mismatch'),
}
RESULTS = {'result:feasible': 'infeasible_solution', 'result:value_binding': 'reported_value_mismatch',
           'result:no_better': 'suboptimal_solution'}


def classify_records(records):
    findings = []
    for record in records:
        if record.verdict != Verdict.REFUTED:
            continue
        if record.claim_id in RESULTS:
            level, category = 'result', RESULTS[record.claim_id]
        else:
            level, category = EFFECTS.get(record.metadata.get('candidate_fault_if_refuted'),
                                          ('relation', 'relation_mismatch'))
        findings.append({'category': category, 'level': level, 'evidence_kind': 'smt_refutation',
                         'relation': record.relation.value, 'evidence': record.to_dict()})
    return findings


def repair_feedback(diagnosis):
    return {'schema': 'orsmt-feedback-v1', 'repair_target': diagnosis.target,
            'target_basis': diagnosis.basis,
            'findings': classify_records((*diagnosis.implementation, *diagnosis.reporting)),
            'source_alignment_certified': False,
            'instruction': 'Use separating assignments to inspect the original requirements. A region discrepancy does not by itself identify which artifact is wrong. Propose only the selected changes; verification will be rerun.'}


def structural_differences(written, program):
    """Candidate explanations only: a different representation need not be wrong."""
    out = []
    wv, pv = written.variable_map, program.variable_map
    for name in sorted(wv.keys() | pv.keys()):
        if name not in wv or name not in pv:
            out.append({'category': 'variable_schema_difference', 'variable': name})
        else:
            for attr, category in [('kind', 'variable_type_difference'), ('lb', 'lower_bound_difference'), ('ub', 'upper_bound_difference')]:
                a, b = getattr(wv[name], attr), getattr(pv[name], attr)
                if a != b:
                    out.append({'category': category, 'variable': name,
                                'written': None if a is None else str(a), 'program': None if b is None else str(b)})
    wc = Counter(r.name for r in written.constraints)
    pc = Counter(r.name for r in program.constraints)
    wr = {r.name: r for r in written.constraints if wc[r.name] == 1}
    pr = {r.name: r for r in program.constraints if pc[r.name] == 1}
    for name in sorted(wr.keys() & pr.keys()):
        a, b = wr[name], pr[name]
        for different, category in [(a.lhs.terms != b.lhs.terms, 'constraint_coefficient_difference'),
                                    (a.sense != b.sense, 'constraint_relation_difference'),
                                    (a.rhs-a.lhs.constant != b.rhs-b.lhs.constant, 'constraint_rhs_difference')]:
            if different:
                out.append({'category': category, 'constraint': name, 'correspondence_basis': 'same_unique_row_name'})
    if written.objective is not None and program.objective is not None:
        a, b = written.objective, program.objective
        for different, category in [(a.sense != b.sense, 'objective_sense_difference'),
                                    (a.expression.terms != b.expression.terms, 'objective_coefficient_difference'),
                                    (a.expression.constant != b.expression.constant, 'objective_offset_difference')]:
            if different:
                out.append({'category': category})
    return [dict(r, evidence_kind='structural_observation', semantic_error_confirmed=False) for r in out]


def row_redundancy(model, verifier):
    rows = []
    for i, constraint in enumerate(model.constraints):
        rest = replace(model, constraints=model.constraints[:i]+model.constraints[i+1:])
        record = verifier.entails(rest, SourceObligation(f'redundancy:{i}:{constraint.name}', constraint), Relation.IMPLEMENTATION)
        redundant = True if record.verdict == Verdict.CERTIFIED else False if record.verdict == Verdict.REFUTED else None
        record = replace(record, metadata={'property': 'individual_constraint_redundancy', 'row_index': i})
        rows.append({'row_index': i, 'constraint': constraint.name, 'redundant': redundant,
                     'normalization_issue': redundant, 'semantic_error_confirmed': False,
                     'category': 'redundant_constraint', 'evidence': record.to_dict()})
    return rows


def summarize_rows(rows, key):
    n = len(rows)
    true = sum(r[key] is True for r in rows)
    false = sum(r[key] is False for r in rows)
    unknown = n-true-false
    return {'total': n, 'true': true, 'false': false, 'undecided': unknown,
            'true_percentage_all_rows': 100*true/n if n else None,
            'false_percentage_all_rows': 100*false/n if n else None,
            'decision_coverage_percentage': 100*(true+false)/n if n else None}


def analyze_case(written, program, result, *, verifier, diagnosis=None):
    diagnosis = diagnosis or diagnose_repair(written, program, result, verifier=verifier)
    w, p = replace(written, name='written'), replace(program, name='program')
    domain_findings = []
    for record in diagnosis.implementation:
        direction = record.metadata.get('region_direction')
        if record.verdict != Verdict.REFUTED or direction not in {'left_minus_right', 'right_minus_left'}:
            continue
        target = p if direction == 'left_minus_right' else w
        for variable in target.variables:
            if variable.name not in record.witness:
                continue
            value = record.witness[variable.name]
            violations = []
            if variable.kind in {'integer', 'binary'} and value.denominator != 1:
                violations.append('integrality')
            if variable.kind == 'binary' and value not in {0, 1}:
                violations.append('binary_domain')
            if variable.lb is not None and value < variable.lb:
                violations.append('lower_bound')
            if variable.ub is not None and value > variable.ub:
                violations.append('upper_bound')
            if violations:
                domain_findings.append({'category': 'variable_domain_mismatch', 'level': 'variables',
                    'evidence_kind': 'smt_refutation', 'variable': variable.name,
                    'violated_artifact': target.name, 'violations': violations,
                    'evidence': record.to_dict()})
    local = {'written_constraints_preserved_by_program': [], 'program_constraints_entailed_by_written': []}
    # Row expressions cannot be transported through unmatched variable schemas.
    row_compatible = (set(w.variable_map) == set(p.variable_map) and not w.auxiliary_names and not p.auxiliary_names)
    for key, left, right in [('written_constraints_preserved_by_program', w, p),
                             ('program_constraints_entailed_by_written', p, w)]:
        for i, row in enumerate(left.constraints):
            if row_compatible:
                record = verifier.entails(right, SourceObligation(f'{key}:{i}:{row.name}', row), Relation.IMPLEMENTATION)
                preserved = True if record.verdict == Verdict.CERTIFIED else False if record.verdict == Verdict.REFUTED else None
                data = record.to_dict()
            else:
                preserved, data = None, {'verdict': 'UNDECIDED', 'reason': 'row_correspondence_requires_variable_mapping'}
            local[key].append({'row_index': i, 'constraint': row.name, 'preserved': preserved, 'evidence': data})
    redundant = {'written': row_redundancy(w, verifier), 'program': row_redundancy(p, verifier)}
    feedback = repair_feedback(diagnosis)
    feedback['findings'].extend(domain_findings)
    for artifact, rows in redundant.items():
        feedback['findings'].extend({
            'category': 'redundant_constraint', 'level': 'constraints',
            'artifact': artifact, 'row_index': row['row_index'],
            'constraint': row['constraint'], 'evidence_kind': 'smt_entailment',
            'normalization_issue': True, 'semantic_error_confirmed': False,
            'evidence': row['evidence']} for row in rows if row['redundant'] is True)
    return {**feedback, 'structural_observations': structural_differences(w, p),
            'constraint_localization': local, 'redundancy': redundant,
            'metrics': {**{key: summarize_rows(rows, 'preserved') for key, rows in local.items()},
                        **{key+'_constraint_redundancy': summarize_rows(rows, 'redundant') for key, rows in redundant.items()}},
            'redundancy_note': 'Each row is tested separately. Individually redundant rows cannot necessarily be removed together.',
            'constraint_count_note': 'Row fractions describe this encoding; equivalent row splitting can change their denominator.'}
