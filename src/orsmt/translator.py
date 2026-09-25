"""Translate independent artifacts to MILP IR; retain how each translation arose."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from .adapters.dict_ir import DictModelAdapter
from .adapters.gurobi_model import GurobiModelAdapter
from .constructive import model_fingerprint
from .ir import ModelIR


@dataclass(frozen=True)
class Translation:
    model: ModelIR
    format: str
    mechanism: str
    original: str | None
    details: dict

    def metadata(self):
        return {'format': self.format, 'mechanism': self.mechanism,
                'translation_llm_calls': int(self.mechanism == 'llm_proposed'),
                'original_sha256': sha256(self.original.encode()).hexdigest() if self.original is not None else None,
                'model_sha256': model_fingerprint(self.model), 'details': self.details,
                'source_alignment_certified': False}


class ModelTranslator:
    """IR, JSON, LP/MPS, Gurobi, Pyomo, PuLP, and supported OR-Tools models.

    Free-form LaTeX/text uses a configured LLM client or host translator. This produces
    a candidate interpretation, never a certificate about the original text.
    Arbitrary Python must go through the host's runtime capture, not this parser.
    """
    BUILTINS = frozenset({'auto', 'ir', 'dict', 'json', 'lp', 'mps', 'gurobi', 'latex', 'text', 'python',
                          'pyomo', 'pulp', 'ortools', 'ortools_linear', 'ortools_cp_sat'})

    def __init__(self, *, llm_translate: Callable[[dict], dict] | None = None):
        self.llm_translate = llm_translate
        self.adapters = {}

    @classmethod
    def from_client(cls, client):
        from .language_frontend import MathematicalTranscriber
        return cls(llm_translate=MathematicalTranscriber(client))

    def register(self, name: str, adapter: Callable[[Any], ModelIR]):
        if name in self.BUILTINS or name in self.adapters or not callable(adapter):
            raise ValueError('adapter name must be new and adapter must be callable')
        self.adapters[name] = adapter

    @staticmethod
    def _validate(model):
        if not isinstance(model, ModelIR) or not model.variables or model.objective is None:
            raise ValueError('translation requires declared variables and an explicit linear objective')
        return model

    @staticmethod
    def _native(model):
        model.update()
        if int(getattr(model, 'NumObj', 1)) != 1 or int(getattr(model, 'NumScenarios', 0)):
            raise ValueError('only single-objective, single-instance models are admitted')
        if int(getattr(model, 'NumPWLObjVars', 0)):
            raise ValueError('piecewise-linear objectives require an explicit linear reformulation before translation')
        return GurobiModelAdapter().to_ir(model)

    def translate(self, artifact, *, format='auto', allow_llm=True) -> Translation:
        original = None
        details = {}
        if isinstance(artifact, Path):
            if format == 'auto':
                format = artifact.suffix.lower().lstrip('.')
                format = {'tex': 'latex', 'txt': 'text'}.get(format, format)
            original = artifact.read_text(encoding='utf-8')
            artifact = original
        if isinstance(artifact, str):
            original = artifact if original is None else original
            text = artifact.strip()
            fence = re.fullmatch(r'```([^\n`]*)\n(.*?)\n```', text, re.S)
            if fence:
                hint, text = fence.group(1).strip().lower(), fence.group(2)
                if format == 'auto' and hint in {'json', 'lp', 'mps', 'latex', 'text', 'python'}:
                    format = hint
            artifact = text
            if format == 'auto':
                if text.startswith('{'):
                    format = 'json'
                elif re.match(r'(?i)^(minimize|maximize|minimum|maximum)\b', text):
                    format = 'lp'
                elif re.match(r'^NAME\b', text):
                    format = 'mps'
                else:
                    raise ValueError('declare the artifact format; free text is not silently interpreted')
        elif isinstance(artifact, ModelIR) and format == 'auto':
            format = 'ir'
        elif isinstance(artifact, Mapping) and format == 'auto':
            format = 'dict'
        elif format in {'auto', 'ortools'}:
            module = type(artifact).__module__
            for prefix, detected in [('pyomo.', 'pyomo'), ('pulp.', 'pulp'),
                                     ('ortools.linear_solver.', 'ortools_linear'),
                                     ('ortools.sat.python.', 'ortools_cp_sat'), ('gurobipy', 'gurobi')]:
                if module.startswith(prefix):
                    format = detected
                    break

        mechanism = 'mechanical'
        if format == 'ir':
            model = artifact
        elif format in {'dict', 'json'}:
            from .language_frontend import strict_json
            data = strict_json(artifact) if isinstance(artifact, str) else artifact
            if not isinstance(data, Mapping) or not {'variables', 'constraints', 'objective'} <= data.keys():
                raise ValueError('model fields variables, constraints and objective must be explicit')
            for key in ('n_quadratic', 'quadratic_constraints', 'general_constraints', 'sos_constraints'):
                if data.get(key):
                    raise ValueError('unsupported model construct: ' + key)
            original = original or json.dumps(data, ensure_ascii=False, default=str)
            model = DictModelAdapter().to_ir(data)
            details['numeric_basis'] = 'exact_rational_input'
        elif format == 'gurobi':
            model = self._native(artifact)
            details['numeric_basis'] = 'native_solver_decimal_roundtrip'
        elif format in {'pyomo', 'pulp', 'ortools_linear', 'ortools_cp_sat'}:
            from .adapters.native_models import pyomo_model, pulp_model, ortools_linear_model
            from .adapters.cp_sat import cp_sat_model
            model = {'pyomo': pyomo_model, 'pulp': pulp_model, 'ortools_linear': ortools_linear_model,
                     'ortools_cp_sat': cp_sat_model}[format](artifact)
            details.update(model.provenance)
        elif format in {'lp', 'mps'}:
            if not isinstance(artifact, str):
                raise TypeError('LP/MPS input must be text or a Path')
            import gurobipy as gp
            with tempfile.TemporaryDirectory(prefix='orsmt-model-') as folder:
                path = Path(folder) / ('model.' + format)
                path.write_text(artifact, encoding='utf-8')
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', 0)
                    env.start()
                    with gp.read(str(path), env=env) as native:
                        if format == 'lp' and any(re.match(r'^[0-9<>=()\[\],]', v.VarName) for v in native.getVars()):
                            raise ValueError('invalid LP variable name; a bare objective constant may have been parsed as a variable')
                        model = self._native(native)
            details['numeric_basis'] = 'native_solver_decimal_roundtrip'
            details['parser'] = 'gurobi.read'
        elif format in {'latex', 'text'}:
            if not allow_llm or not isinstance(artifact, str) or self.llm_translate is None:
                raise ValueError('free-form mathematical text requires an explicit translation callback')
            response = self.llm_translate({'artifact': original, 'format': format,
                'instruction': 'Structure this written optimization model without correcting it. Preserve its variable domains, constraints, coefficients, objective and mistakes. Return formulation, assumptions and translator_id. Do not add requirements from another source.'})
            if not isinstance(response, Mapping) or not isinstance(response.get('assumptions'), list) or not response.get('translator_id'):
                raise ValueError('translation callback must retain assumptions and translator identity')
            translated = self.translate(response['formulation'], format='dict')
            model = translated.model
            mechanism = 'llm_proposed'
            details = {'assumptions': response['assumptions'], 'translator_id': response['translator_id'],
                       'usage': response.get('usage'), 'source_spans': response.get('source_spans'),
                       'call_record': response.get('call_record'), 'translation_equivalence_certified': False}
        elif format in self.adapters:
            model = self.adapters[format](artifact)
            mechanism = 'host_adapter'
        elif format == 'python':
            raise ValueError('execute Python through isolated runtime capture; do not translate its intent with an LLM')
        else:
            raise ValueError('unsupported artifact format: ' + str(format))
        return Translation(self._validate(model), format, mechanism, original, details)
