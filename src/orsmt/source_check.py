"""Candidate-blind source interpretation followed by conditional SMT checks."""
from dataclasses import replace
from .evidence import Relation, Verdict, aggregate
from .ir import LinearExpr, SourceObligation
from .language_frontend import digest, validate_formulation
from .milp_codegen import model_to_dict
from .repair_workflow import compare_models
from .translator import ModelTranslator
from .verify import Verifier
from .taxonomy import classify_records, structural_differences


def validate_proposal(source, data, proposal):
    required={'formulation','requirements','ambiguities','missing_information','complete'}
    if not required<=proposal.keys() or type(proposal['complete']) is not bool:
        raise ValueError('incomplete grounding schema')
    for key in ('requirements','ambiguities','missing_information'):
        if not isinstance(proposal[key],list):raise ValueError('invalid grounding '+key)
    validate_formulation(proposal['formulation'])
    model=ModelTranslator().translate(proposal['formulation'],format='dict',allow_llm=False).model
    if len({c.name for c in model.constraints})!=len(model.constraints):
        raise ValueError('source constraint names must be unique')
    targets={'variable':set(model.variable_map),'constraint':{c.name for c in model.constraints},'objective':{'objective'}}
    covered={k:set() for k in targets};seen=set()
    for req in proposal['requirements']:
        if not isinstance(req,dict) or not isinstance(req.get('id'),str) or not req['id'] or req['id'] in seen:
            raise ValueError('source requirement ID missing/duplicate')
        seen.add(req['id']);kind=req.get('kind');names=req.get('targets')
        if kind not in targets or not isinstance(names,list) or not names or not set(names)<=targets[kind]:
            raise ValueError('requirement targets missing or outside formulation')
        covered[kind].update(names)
        if not isinstance(req.get('source_spans'),list) or not req['source_spans']:
            raise ValueError('requirement requires source provenance')
        for span in req['source_spans']:
            document={'source':source,'data':data}.get(span.get('document'))
            a,b=span.get('start'),span.get('end')
            if document is None or type(a) is not int or type(b) is not int or not 0<=a<b<=len(document) or document[a:b]!=span.get('quote'):
                raise ValueError('source span does not match original text')
    if covered!=targets:raise ValueError('source provenance must cover every model component')
    return model


class SourceChecker:
    def __init__(self, grounder, *, translator=None, verifier=None):
        self.grounder=grounder;self.translator=translator or ModelTranslator();self.verifier=verifier or Verifier()

    def prepare(self, source, *, data=''):
        """Must run once, before any candidate is supplied; product is hash-bound."""
        if not isinstance(source,str) or not source.strip() or not isinstance(data,str):
            raise ValueError('source and rendered instance data must be strings')
        product={'schema':'orsmt-source-product-v1','source':source,'data':data,
                 'source_sha256':digest({'source':source,'data':data}),'natural_language_certified':False}
        clients = {id(c): c for c in [getattr(self.grounder, 'proposer', None), getattr(self.grounder, 'reviewer', None)] if c is not None}
        starts = {key: len(c.calls) for key, c in clients.items()}
        try:
            proposal=self.grounder.propose(source,data)
            model=validate_proposal(source,data,proposal)
            h=digest(proposal);review=self.grounder.review(source,data,proposal,h)
            product.update(proposal=proposal,proposal_sha256=h,review=review,model=model_to_dict(model))
            fields={'approved','complete','issues','rejected_requirement_ids','uncovered_source_spans','reviewed_sha256'}
            if not isinstance(review,dict) or not fields<=review.keys() or review['reviewed_sha256']!=h:
                raise ValueError('review is not bound to the exact grounding proposal')
            if any(type(review[k]) is not bool for k in ('approved','complete')) or any(
                not isinstance(review[k],list) for k in ('issues','rejected_requirement_ids','uncovered_source_spans')):
                raise ValueError('invalid grounding review schema')
            approved=review['approved'] and not review['issues'] and not review['rejected_requirement_ids']
            approved=approved and not proposal['ambiguities'] and not proposal['missing_information']
            product['status']='REVIEW_APPROVED' if approved else 'REVIEW_NOT_APPROVED'
            product['complete']=bool(approved and proposal['complete'] and review['complete'] and not review['uncovered_source_spans'])
        except Exception as exc:
            product.update(status='UNDECIDED',complete=False,error_type=type(exc).__name__,reason=str(exc))
        product['llm_calls'] = [dict(record) for key, c in clients.items() for record in c.calls[starts[key]:]]
        product['product_sha256']=digest(product)
        return product

    @staticmethod
    def validate_product(product):
        if digest({k:v for k,v in product.items() if k!='product_sha256'})!=product.get('product_sha256'):
            raise ValueError('source product was modified after review')
        if product.get('source_sha256')!=digest({'source':product['source'],'data':product['data']}):
            raise ValueError('source text hash changed')
        if product.get('status')=='REVIEW_APPROVED':
            review, proposal = product['review'], product['proposal']
            if review.get('approved') is not True or review.get('issues') or review.get('rejected_requirement_ids') or proposal.get('ambiguities') or proposal.get('missing_information'):
                raise ValueError('source review approval is inconsistent')
            complete = proposal['complete'] and review['complete'] and not review['uncovered_source_spans']
            if product['complete'] != complete:raise ValueError('source completeness is inconsistent')
            if digest(product['proposal'])!=product['proposal_sha256'] or product['review']['reviewed_sha256']!=product['proposal_sha256']:
                raise ValueError('source proposal/review binding changed')
            m=validate_proposal(product['source'],product['data'],product['proposal'])
            if model_to_dict(m)!=product['model']:raise ValueError('reviewed source model differs from proposal')

    def check(self, product, candidate, *, format='auto', variable_map=None):
        self.validate_product(product)
        result={'schema':'orsmt-source-check-v1','product_sha256':product['product_sha256'],
                'source_review_status':product['status'],'natural_language_certified':False,
                'verdict':'UNDECIDED','records':[],'proof_scope':'reviewed_formal_interpretation_only'}
        if product['status']!='REVIEW_APPROVED':
            return dict(result,reason='source_interpretation_not_approved')
        try:
            translated=self.translator.translate(candidate,format=format)
            model=translated.model
            reference=self.translator.translate(product['model'],format='dict',allow_llm=False).model
            result['translation']=translated.metadata()
            if variable_map is not None:
                # Explicit correspondence maps each SOURCE name to a CANDIDATE name.
                if set(variable_map)!=set(reference.variable_map) or len(set(variable_map.values()))!=len(variable_map) or not set(variable_map.values())<=set(model.variable_map):
                    raise ValueError('source-to-candidate map must be complete and injective')
                reverse={v:k for k,v in variable_map.items()}
                def expr(e):return LinearExpr(tuple((reverse.get(n,n),c) for n,c in e.terms),e.constant)
                model=replace(model,variables=tuple(replace(v,name=reverse.get(v.name,v.name)) for v in model.variables),
                    constraints=tuple(replace(c,lhs=expr(c.lhs)) for c in model.constraints),
                    objective=replace(model.objective,expression=expr(model.objective.expression)))
                result['correspondence_basis']='explicit_host_mapping'
            else:
                result['correspondence_basis']='identical_names_assumed_same_business_meaning'
            if not set(reference.variable_map)<=set(model.variable_map):
                raise ValueError('source variables require correspondence')
            if product['complete']:
                records=[replace(r,relation=Relation.FORMALIZATION) for r in compare_models(self.verifier,reference,model)]
                result['coverage']='full_reviewed_formal_model'
            else:
                records=[]
                for row in reference.constraints:
                    req=next(q for q in product['proposal']['requirements'] if q['kind']=='constraint' and row.name in q['targets'])
                    source_span=' | '.join(s['quote'] for s in req['source_spans'])
                    records.append(self.verifier.entails(model,SourceObligation(req['id']+':'+row.name,row,source_span)))
                result['coverage']='reviewed_constraint_obligations_only'
                result['unchecked']=['variable_domains','objective','overconstraint','uncovered_requirements']
            result['records']=[r.to_dict() for r in records]
            result['findings']=classify_records(records)
            result['structural_observations']=structural_differences(reference,model)
            if product['complete']:
                result['source_constraint_checks']=[]
                for row in reference.constraints:
                    req=next(q for q in product['proposal']['requirements'] if q['kind']=='constraint' and row.name in q['targets'])
                    evidence=self.verifier.entails(model,SourceObligation(req['id']+':'+row.name,row,
                        ' | '.join(s['quote'] for s in req['source_spans'])))
                    result['source_constraint_checks'].append({'requirement_id':req['id'],'source_spans':req['source_spans'],
                                                              'evidence':evidence.to_dict()})
            result['checked_relations_verdict']=aggregate(records).value
            result['verdict']=aggregate(records).value if product['complete'] or any(r.verdict==Verdict.REFUTED for r in records) else 'UNDECIDED'
            result['complete']=product['complete']
        except Exception as exc:
            result.update(reason=str(exc),error_type=type(exc).__name__)
        return result

    def explain(self, product, report):
        self.validate_product(product)
        if report.get('product_sha256')!=product['product_sha256']:
            raise ValueError('report belongs to another source product')
        explanation=self.grounder.explain(product['source'],report)
        quotes=explanation.get('source_quotes')
        if not isinstance(quotes,list) or any(not isinstance(q,str) or not q or q not in product['source'] for q in quotes):
            raise ValueError('explanation source quotes are not grounded')
        return {'evidence_kind':'llm_explanation','formal_verdict':report['verdict'],
                'report_sha256':digest(report),'proposal':explanation}
