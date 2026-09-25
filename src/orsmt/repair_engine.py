"""Executable repair loop: formal-reference repair or source-grounded proposals.

The host supplies native execution and, optionally, an LLM callback. Benchmark
answers are not inputs. Every attempted revision is recaptured and rechecked.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Any

from .evidence import aggregate, Verdict
from .ir import ModelIR, ResultRecord
from .milp_codegen import emit_gurobi, model_to_dict
from .repair_workflow import diagnose_repair, evaluate_revision, repair_prompt_payload
from .verify import Verifier
from .translator import ModelTranslator
from .taxonomy import repair_feedback


@dataclass(frozen=True)
class ProgramExecution:
    model: ModelIR
    result: ResultRecord | None


def repair_case(
    *, case_id: str, question: str, written: ModelIR, code: str,
    execute: Callable[[str], ProgramExecution],
    proposer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    reference: ModelIR | None = None, reference_origin: str | None = None,
    written_is_specification: bool = False, verifier: Verifier | None = None,
    max_rounds: int = 3, max_llm_calls: int = 1,
    translator: ModelTranslator | None = None,
) -> dict[str, Any]:
    if max_rounds < 1 or max_llm_calls < 0:
        raise ValueError('invalid repair budget')
    v = verifier or Verifier()
    translator = translator or ModelTranslator()
    started = perf_counter()
    history, llm_calls, native_calls = [], 0, 0
    current_written, current_code = written, code
    initial = None

    def result_dict(execution):
        if execution is None or execution.result is None:
            return None
        return {'assignment': {k: str(x) for k, x in execution.result.point.items()},
                'objective_value': None if execution.result.value is None else str(execution.result.value)}

    def run(source, template=None):
        nonlocal native_calls
        native_calls += 1
        result = execute(source)
        # Generated code preserves the known variable schema, including auxiliaries.
        if template is not None and set(result.model.variable_map) == set(template.variable_map):
            result = replace(result, model=replace(result.model, variables=tuple(
                replace(x, observable=template.variable_map[x.name].observable) for x in result.model.variables)))
        return result

    def finish(status, accepted=False, assessment=None, execution=None):
        return {'id': case_id, 'status': status, 'accepted': accepted,
                'llm_calls': llm_calls, 'native_calls': native_calls,
                'history': history, 'formulation': model_to_dict(current_written),
                'code': current_code, 'assessment': assessment,
                'result': result_dict(execution), 'initial_result': result_dict(initial),
                'elapsed_seconds': perf_counter() - started,
                'natural_language_verified': False,
                'reference_origin': reference_origin,
                'initial_written_is_specification': written_is_specification}

    try:
        current = run(current_code)
        initial = current
    except Exception as exc:
        history.append({'stage': 'initial_execution', 'error': f'{type(exc).__name__}: {exc}'})
        return finish('initial_execution_unresolved')

    for iteration in range(max_rounds):
        diagnosis = diagnose_repair(current_written, current.model, current.result,
                                   reference=reference, reference_origin=reference_origin,
                                   written_is_specification=written_is_specification, verifier=v)
        step = {'round': iteration+1, 'diagnosis': diagnosis.to_dict()}
        target = diagnosis.target
        if target == 'none' and current.result is None and aggregate(list(diagnosis.implementation)) == Verdict.CERTIFIED:
            target = 'result'
            diagnosis = replace(diagnosis, target=target, basis='missing_returned_result')
            step['diagnosis'] = diagnosis.to_dict()
        step['feedback'] = repair_feedback(diagnosis)
        if target == 'none':
            complete = (aggregate(list(diagnosis.implementation)) == Verdict.CERTIFIED
                        and current.result is not None
                        and aggregate(list(diagnosis.reporting)) == Verdict.CERTIFIED)
            history.append(step)
            return finish('already_verified' if complete else 'verification_incomplete',
                          execution=current)
        proposal = None
        if target == 'review':
            if proposer is None:
                history.append(step)
                return finish('needs_source_review', execution=current)
            if llm_calls >= max_llm_calls:
                history.append(step)
                return finish('llm_budget_exhausted', execution=current)
            payload = repair_prompt_payload(diagnosis, case_id=case_id, question=question,
                                            formulation=model_to_dict(current_written), code=current_code)
            payload['prior_attempts'] = history
            payload['feedback'] = repair_feedback(diagnosis)
            payload['response_fields'] = ['id', 'proposed_target', 'source_quotes', 'reason', 'formulation', 'formulation_format']
            payload['instruction'] = (
                'Decide which artifact disagrees with the original requirements, using the source text '
                'and the separating assignments. Quote exact supporting source spans. '
                'Return id, proposed_target (written_model, program, or both), source_quotes, reason, '
                'and a complete linear formulation when the written model changes. '
                'Use a JSON model object or LP/MPS text with an explicit formulation_format field. '
                'Do not generate Python: any program revision is compiled from the selected formulation. '
                'Your proposal is not a verification verdict; the actual artifacts will be rechecked.')
            llm_calls += 1
            try:
                proposal = proposer(payload)
                if proposal.get('id') != case_id or proposal.get('proposed_target') not in {'written_model', 'program', 'both'}:
                    raise ValueError('invalid proposal ID or target')
                quotes = proposal.get('source_quotes')
                if not isinstance(quotes, list) or not quotes or not all(
                    isinstance(s, str) and s.strip() and s in question for s in quotes):
                    raise ValueError('proposal must cite exact nonempty source spans')
                if not isinstance(proposal.get('reason'), str) or not proposal['reason'].strip():
                    raise ValueError('proposal must explain the discrepancy')
                target = proposal['proposed_target']
                # A source review resolves the route, not the truth of the natural language.
                diagnosis = replace(diagnosis, target=target, basis='source_grounded_llm_proposal')
                step.update(proposal=proposal, proposal_basis='source_grounded_llm_proposal')
            except Exception as exc:
                step['proposal_error'] = f'{type(exc).__name__}: {exc}'
                history.append(step)
                continue
        new_written, new_code = current_written, current_code
        try:
            if target in {'written_model', 'both'}:
                if proposal is not None:
                    translated = translator.translate(proposal['formulation'], format=proposal.get('formulation_format', 'auto'), allow_llm=False)
                    new_written = translated.model
                    step['translation'] = translated.metadata()
                elif reference is not None:
                    new_written = reference
                else:
                    raise ValueError('written-model repair needs a formal reference or a source proposal')
            if target in {'program', 'both'}:
                new_code = emit_gurobi(new_written)
                step['program_revision'] = 'deterministic_instance_codegen'
            if target == 'result':
                # Re-solve the same mathematical program and replace only the returned result.
                solved = run(emit_gurobi(current.model), current.model)
                checked = diagnose_repair(current.model, solved.model, verifier=v,
                                          written_is_specification=True)
                if aggregate(list(checked.implementation)) != Verdict.CERTIFIED:
                    raise ValueError('result resolver changed the optimization model')
                revised = ProgramExecution(current.model, solved.result)
                step['program_revision'] = 'unchanged_program_result_resolved'
            else:
                revised = run(new_code, new_written if new_code != current_code else current.model)
            assessment = evaluate_revision(diagnosis, new_written, revised.model, revised.result,
                                           reference=reference, verifier=v)
            step['assessment'] = assessment
            history.append(step)
            current_written, current_code, current = new_written, new_code, revised
            if assessment['accepted']:
                return finish('verified_revision', True, assessment, current)
        except Exception as exc:
            step['revision_error'] = f'{type(exc).__name__}: {exc}'
            history.append(step)
            if proposal is None:
                return finish('mechanical_revision_unresolved', execution=current)
    return finish('round_budget_exhausted', execution=current)
