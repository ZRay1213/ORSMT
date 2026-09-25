"""Format-independent host entry for verification, diagnostic feedback, and repair."""
from .adapters.dict_ir import DictResultAdapter
from .ir import ResultRecord
from .repair_engine import repair_case
from .repair_workflow import diagnose_repair
from .taxonomy import analyze_case
from .translator import ModelTranslator
from .verify import Verifier
from .normalization import normalize_constraints


class ORSMTWorkflow:
    def __init__(self, *, translator=None, verifier=None, source_checker=None):
        self.translator = translator or ModelTranslator()
        self.verifier = verifier or Verifier()
        self.source_checker = source_checker

    def prepare_source(self, question, *, data=''):
        if self.source_checker is None:
            raise ValueError('configure a SourceChecker with explicit LLM clients')
        return self.source_checker.prepare(question, data=data)

    def check_source(self, product, artifact, *, format='auto', variable_map=None, explain=False):
        if self.source_checker is None:
            raise ValueError('configure a SourceChecker with explicit LLM clients')
        out = self.source_checker.check(product, artifact, format=format, variable_map=variable_map)
        if explain:
            out['explanation'] = self.source_checker.explain(product, out)
        return out

    def check(self, *, case_id, written, program, result=None,
              written_format='auto', program_format='auto', reference=None, reference_origin=None,
              source_product=None, source_variable_maps=None):
        # These two artifacts are translated independently. Never regenerate the
        # written model from code and then compare the code with its own translation.
        w = self.translator.translate(written, format=written_format)
        p = self.translator.translate(program, format=program_format)
        r = result if isinstance(result, ResultRecord) or result is None else DictResultAdapter().to_result(result)
        diagnosis = diagnose_repair(w.model, p.model, r, verifier=self.verifier,
                                    reference=reference, reference_origin=reference_origin)
        out = {'id': case_id, 'translations': {'written': w.metadata(), 'program': p.metadata()},
                'original_artifacts': {'written': w.original, 'program': p.original},
                'diagnosis': diagnosis.to_dict(),
                'feedback': analyze_case(w.model, p.model, r, verifier=self.verifier, diagnosis=diagnosis)}
        if source_product is not None:
            maps = source_variable_maps or {}
            out['source_checks'] = {name: self.check_source(source_product, model, variable_map=maps.get(name))
                                    for name, model in [('written', w.model), ('program', p.model)]}
            out['feedback']['source_alignment'] = out['source_checks']
        return out

    def repair_from_source(self, *, source_product, **options):
        if self.source_checker is None:
            raise ValueError('configure a SourceChecker')
        self.source_checker.validate_product(source_product)
        if source_product['status'] != 'REVIEW_APPROVED' or not source_product['complete']:
            raise ValueError('source-driven mechanical repair requires a complete reviewed formal interpretation')
        if any(k in options for k in ('reference', 'reference_origin', 'written_is_specification')):
            raise ValueError('source repair fixes its reference to the reviewed product')
        question = options.pop('question', source_product['source'])
        if question != source_product['source']:
            raise ValueError('repair question must match the frozen source')
        reference = self.translator.translate(source_product['model'], format='dict', allow_llm=False).model
        out = self.repair(question=question, reference=reference,
                          reference_origin='llm_reviewed_source:' + source_product['product_sha256'], **options)
        out['source_product_sha256'] = source_product['product_sha256']
        out['source_recheck'] = self.check_source(source_product, out['formulation'], format='dict')
        out['acceptance_scope'] = 'reviewed_formal_interpretation_only'
        if out['source_recheck']['verdict'] != 'CERTIFIED':
            out['accepted'] = False
        return out

    def repair(self, *, case_id, question, written, code, execute, written_format='auto', **options):
        w = self.translator.translate(written, format=written_format)
        out = repair_case(case_id=case_id, question=question, written=w.model, code=code,
                          execute=execute, verifier=self.verifier, translator=self.translator, **options)
        out['input_translation'] = w.metadata()
        out['original_written_artifact'] = w.original
        return out

    def normalize(self, *, artifact, format='auto', **options):
        translated = self.translator.translate(artifact, format=format)
        out = normalize_constraints(translated.model, self.verifier, **options)
        out['translation'] = translated.metadata()
        return out
